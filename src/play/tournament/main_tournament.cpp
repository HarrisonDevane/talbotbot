// =============================================================================
// main_tournament.cpp
//
// Entry point for talbot_tournament -- one process, one pairing.
//
// LAUNCH CONTRACT:
//   talbot_tournament [--tournament]                # ignored flag, kept for compat
//                     --config_file  <yaml>
//                     --model_a      <onnx path>
//                     --model_b      <onnx path>    # may be the SAME file as model_a
//                     --run_dir      <dir>          # top-level; we make a subdir per pairing
//                   [ --name_a       <label> ]      # default: model_a file stem
//                   [ --name_b       <label> ]      # default: model_b file stem
//                   [ --override_a   section.key=value ]   # repeatable
//                   [ --override_b   section.key=value ]   # repeatable
//
// The original 4-flag call (used by the model-vs-model orchestrator) is
// unchanged: names default to the file stems, no overrides.
//
// PER-SIDE CONFIG
//   Each side gets its own copy of the YAML's `mcts:` and `selection:` blocks,
//   with that side's overrides applied, then run through the shared
//   load_configs(). Only `mcts.*` and `selection.*` keys may be overridden, and
//   only keys that already exist in the YAML (typos fail loudly). Examples:
//     --override_a mcts.gumbel.sigma_cap=50
//     --override_a mcts.search_depth=3200            # per-side node budget
//     --override_b mcts.search=halving               # SH root vs deficit root
//     --override_b mcts.search=puct --override_b mcts.puct.cpuct=2.0
//   Everything else (inference, pool sizing, cores, openings) is shared.
//
// SEARCH
//   Fixed node budget only (mcts.search_depth, per side). mcts.search
//   picks the root strategy per side:
//     deficit -> run_simulations_fixed_inference (the UCI search)
//     halving -> run_simulations_fixed           (the training SH search)
//
// WHAT IT DOES:
//   1. Loads YAML config, builds per-side search configs.
//   2. Builds TWO InferenceBatchers, one per side, sharing the GPU.
//   3. Spawns N game-worker threads; each holds TWO MCTSEngines (one per batcher)
//      so it can play either side with either colour.
//   4. Generates game specs: games_per_match / 2 openings, each played TWICE
//      with colours swapped so opening effects cancel.
//   5. Distributes specs across workers stride-style; each plays its subset
//      sequentially via SelfPlaySession + GameWorker::run_one_game.
//
// OUTPUTS (in <run_dir>/<name_a>_vs_<name_b>/):
//   games.pgn   -- every game concatenated as a PGN block, White/Black tags are
//                  the side NAMES (labels). THIS is the file the downstream
//                  rating tool (Ordo/BayesElo) consumes.
//   summary.txt -- plain wins/draws/losses counts + config echo. NO Elo.
//   per-worker log files -- for diagnostics.
// =============================================================================

#include <pthread.h>
#include <sched.h>
#include <unistd.h>
#include <sys/wait.h>
#include <fcntl.h>

#include <iostream>
#include <fstream>
#include <vector>
#include <thread>
#include <atomic>
#include <string>
#include <sstream>
#include <filesystem>
#include <chrono>
#include <memory>
#include <mutex>
#include <cmath>
#include <stdexcept>
#include <unordered_set>

#include "concurrentqueue.h"
#include <torch/torch.h>
#include <yaml-cpp/yaml.h>

#include "chess.hpp"
#include "mcts_engine.hpp"
#include "action_selector.hpp"
#include "inference_batcher.hpp"
#include "logger.hpp"
#include "game_worker.hpp"
#include "self_play_session.hpp"
#include "opening_book.hpp"
#include "tbprobe.h"

namespace fs = std::filesystem;

// =============================================================================
// TRT COMPILE (out-of-process, mirrors main_data_generator.cpp)
//
// Engine building lives in talbot_trt_compile. This exe finds it next to
// itself (same build dir) and shells out synchronously. The engine is written
// to disk at a deterministic path derived from the ONNX path; if it already
// exists we skip the build. When both sides use the same ONNX (parameter
// sweeps), side A builds it and side B hits the cache.
// =============================================================================
static std::string resolve_trt_compile_exe() {
    char buf[4096];
    ssize_t n = readlink("/proc/self/exe", buf, sizeof(buf) - 1);
    if (n <= 0) return "talbot_trt_compile";
    buf[n] = '\0';
    return (fs::path(buf).parent_path() / "talbot_trt_compile").string();
}

// Run talbot_trt_compile synchronously. Returns exit code, -1 on launch failure.
static int run_trt_compile(const std::string& compile_exe,
                           const std::string& onnx_path,
                           const std::string& engine_path,
                           int max_batch,
                           int input_planes) {
    pid_t pid = fork();
    if (pid < 0) return -1;

    if (pid == 0) {
        int null_fd = open("/dev/null", O_WRONLY);
        if (null_fd >= 0) {
            dup2(null_fd, STDOUT_FILENO);
            close(null_fd);
        }

        std::string batch_str  = std::to_string(max_batch);
        std::string planes_str = std::to_string(input_planes);

        execl(compile_exe.c_str(), compile_exe.c_str(),
              onnx_path.c_str(), engine_path.c_str(),
              "--input-planes", planes_str.c_str(),
              "--max-batch",    batch_str.c_str(),
              "--force", (char*)nullptr);
        _exit(127);
    }

    int status = 0;
    if (waitpid(pid, &status, 0) < 0) return -1;
    if (WIFEXITED(status)) return WEXITSTATUS(status);
    return -1;
}

// Ensure an engine file exists for the given ONNX. Returns the engine path,
// or empty string on failure. Engine path = onnx path with .engine extension
// (lives alongside the ONNX). Existing non-empty engine file is treated as
// cached and returned without rebuild.
static std::string ensure_engine(const std::string& onnx_path,
                                 int max_batch,
                                 int input_planes,
                                 Logger& logger) {
    fs::path engine_path = fs::path(onnx_path);
    engine_path.replace_extension(".engine");

    if (fs::exists(engine_path)) {
        std::error_code ec;
        const auto sz = fs::file_size(engine_path, ec);
        if (!ec && sz > 0) {
            logger.log("INFO", "Using cached engine: " + engine_path.string());
            return engine_path.string();
        }
    }

    logger.log("INFO", "Building engine (max_batch=" + std::to_string(max_batch) +
                       ", input_planes=" + std::to_string(input_planes) + "): " +
                       engine_path.string());
    const std::string compile_exe = resolve_trt_compile_exe();
    const int rc = run_trt_compile(compile_exe, onnx_path, engine_path.string(),
                                    max_batch, input_planes);
    if (rc != 0) {
        logger.log("CRITICAL", "talbot_trt_compile failed with exit code " +
                               std::to_string(rc) + " (onnx=" + onnx_path + ")");
        return "";
    }
    logger.log("INFO", "Engine built: " + engine_path.string());
    return engine_path.string();
}

// =============================================================================
// CLI
// =============================================================================
struct CliArgs {
    std::string config_file;
    std::string model_a_path;
    std::string model_b_path;
    std::string run_dir;
    std::string name_a;
    std::string name_b;
    std::vector<std::string> overrides_a;
    std::vector<std::string> overrides_b;
};

static bool parse_args(int argc, char* argv[], CliArgs& out) {
    for (int i = 1; i < argc; ++i) {
        std::string k = argv[i];
        if (k == "--tournament") continue;
        if (i + 1 >= argc) {
            std::cerr << "Missing value for " << k << "\n";
            return false;
        }
        std::string v = argv[++i];
        if      (k == "--config_file") out.config_file = v;
        else if (k == "--model_a")     out.model_a_path = v;
        else if (k == "--model_b")     out.model_b_path = v;
        else if (k == "--run_dir")     out.run_dir = v;
        else if (k == "--name_a")      out.name_a = v;
        else if (k == "--name_b")      out.name_b = v;
        else if (k == "--override_a")  out.overrides_a.push_back(v);
        else if (k == "--override_b")  out.overrides_b.push_back(v);
        else {
            std::cerr << "Unknown arg: " << k << "\n";
            return false;
        }
    }
    if (out.config_file.empty() || out.model_a_path.empty() ||
        out.model_b_path.empty() || out.run_dir.empty()) {
        std::cerr << "Usage: talbot_tournament --config_file <yaml> --model_a <path> --model_b <path> --run_dir <dir>\n"
                     "         [--name_a <label>] [--name_b <label>]\n"
                     "         [--override_a section.key=value]... [--override_b section.key=value]...\n";
        return false;
    }
    if (out.name_a.empty()) out.name_a = fs::path(out.model_a_path).stem().string();
    if (out.name_b.empty()) out.name_b = fs::path(out.model_b_path).stem().string();
    if (out.name_a == out.name_b) {
        std::cerr << "Side names must differ (got '" << out.name_a << "' for both). "
                     "Pass --name_a / --name_b -- the PGN player names are how the "
                     "rating tool tells the two sides apart.\n";
        return false;
    }
    return true;
}

// =============================================================================
// PER-SIDE CONFIG
// =============================================================================
struct SideConfig {
    std::string              name;
    std::string              model_path;
    std::vector<std::string> overrides;

    MctsConfig           mcts;
    ActionSelectorConfig selector;
    bool   use_halving                = false;
    bool   use_puct                   = false;
    double cpuct                      = 0.0;
    double puct_fpu_reduction         = 0.0;
    double early_stop_q_gap           = 0.0;
    int    early_stop_min_visits      = 0;
    bool   early_return_on_forced_win = false;
    bool   tree_reuse                 = false;
};

// Clone `base` and apply "path.to.key=value" overrides, e.g.
//   mcts.search=puct   mcts.gumbel.explore_c=0.2   mcts.puct.cpuct=2.0
// The first path segment must be mcts or selection (the only per-side
// blocks); the full path must already exist in the YAML, so a typo is an
// error instead of a silently ignored setting. Values are stored as YAML
// scalar strings; .as<double>() / .as<int>() / ... parse them as usual.
static YAML::Node apply_overrides(const YAML::Node& base,
                                  const std::vector<std::string>& overrides) {
    YAML::Node out = YAML::Clone(base);
    for (const auto& ov : overrides) {
        const auto eq = ov.find('=');
        if (eq == std::string::npos || eq == 0 || eq + 1 == ov.size()) {
            throw std::runtime_error("bad override '" + ov + "' (want section.key=value)");
        }
        const std::string path  = ov.substr(0, eq);
        const std::string value = ov.substr(eq + 1);

        std::vector<std::string> parts;
        std::stringstream ps(path);
        for (std::string seg; std::getline(ps, seg, '.');) {
            if (seg.empty()) throw std::runtime_error("bad override '" + ov + "' (empty path segment)");
            parts.push_back(seg);
        }
        if (parts.size() < 2) {
            throw std::runtime_error("bad override '" + ov + "' (want section.key=value)");
        }
        if (parts[0] != "mcts" && parts[0] != "selection") {
            throw std::runtime_error("override '" + ov +
                "': only mcts.* and selection.* can differ per side");
        }

        // Existence check on the original YAML. Node::reset rebinds the
        // handle; plain assignment would overwrite the tree (yaml-cpp gotcha).
        YAML::Node probe;
        probe.reset(base);
        for (const auto& seg : parts) {
            const YAML::Node& cprobe = probe;
            if (!cprobe.IsMap() || !cprobe[seg]) {
                throw std::runtime_error("override '" + ov + "' targets unknown key " +
                                         path + " (add it to the YAML first)");
            }
            YAML::Node next = cprobe[seg];
            probe.reset(next);
        }
        if (probe.IsMap()) {
            throw std::runtime_error("override '" + ov + "' targets a section, not a value");
        }

        // Walk the clone to the parent of the final key and assign.
        YAML::Node cur;
        cur.reset(out);
        for (size_t i = 0; i + 1 < parts.size(); ++i) {
            YAML::Node next = cur[parts[i]];
            cur.reset(next);
        }
        cur[parts.back()] = value;
    }
    return out;
}

static SideConfig load_side(const YAML::Node& root,
                            const std::string& name,
                            const std::string& model_path,
                            const std::vector<std::string>& overrides) {
    SideConfig s;
    s.name       = name;
    s.model_path = model_path;
    s.overrides  = overrides;

    const YAML::Node side_root = apply_overrides(root, overrides);
    const YAML::Node m = side_root["mcts"];
    const YAML::Node sel = side_root["selection"];
    if (!m)   throw std::runtime_error("missing 'mcts:' block");
    if (!sel) throw std::runtime_error("missing 'selection:' block");

    // Shared loader with training + UCI (see action_selector.cpp). It reads
    // mcts.search and requires mcts.gumbel (deficit/halving) or mcts.puct
    // (puct). The tournament YAML carries both sections so per-side
    // overrides can switch search type.
    LoadedConfigs lc = load_configs(m, sel);
    s.mcts     = lc.mcts;
    s.selector = lc.selector;

    // HYBRID = SH at the root (use_halving) + PUCT below it (use_puct).
    s.use_halving        = (s.mcts.search == SearchType::HALVING || s.mcts.search == SearchType::HYBRID);
    s.use_puct           = (s.mcts.search == SearchType::PUCT    || s.mcts.search == SearchType::HYBRID);
    s.cpuct              = s.mcts.cpuct;
    s.puct_fpu_reduction = s.mcts.puct_fpu_reduction;

    if (!m["tree_reuse"]) {
        throw std::runtime_error("missing mcts.tree_reuse (true | false)");
    }
    s.tree_reuse = m["tree_reuse"].as<bool>();

    if (s.mcts.search_depth <= 0) {
        throw std::runtime_error("mcts.search_depth must be > 0 for side " + name);
    }

    // POLICY = no search: play argmax of the network policy. Implemented as a
    // 1-simulation PUCT search: after root expansion every child is unvisited
    // with the same FPU, so the single sim goes to argmax(prior) and the
    // max-visits rule plays it. cpuct only needs to be > 0. Reuse off so no
    // old visits can leak in. The budget is 1 regardless of search_depths.
    if (s.mcts.search == SearchType::POLICY) {
        s.use_halving        = false;
        s.use_puct           = true;
        s.cpuct              = 1.0;
        s.puct_fpu_reduction = 0.0;
        s.tree_reuse         = false;
        s.mcts.search_depth  = 1;
    }

    if (m["early_stop_q_gap"])           s.early_stop_q_gap           = m["early_stop_q_gap"].as<double>();
    if (m["early_stop_min_visits"])      s.early_stop_min_visits      = m["early_stop_min_visits"].as<int>();
    if (m["early_return_on_forced_win"]) s.early_return_on_forced_win = m["early_return_on_forced_win"].as<bool>();
    return s;
}

static const char* search_name(const SideConfig& s) {
    return search_type_name(s.mcts.search);
}

static std::string describe_side(const SideConfig& s) {
    std::ostringstream o;
    o << s.name
      << " | model=" << s.model_path
      << " | search=" << search_name(s)
      << " | nodes=" << s.mcts.search_depth
      << " | policy_temp=" << s.mcts.policy_softmax_temp
      << " | early_stop_q_gap=" << s.early_stop_q_gap
      << " | tree_reuse=" << (s.tree_reuse ? (s.use_halving ? "on (ignored: halving)" : "on") : "off");
    if (s.mcts.search == SearchType::POLICY) {
        o << " | (no search: argmax of policy)";
    } else if (s.use_puct) {
        o << " | cpuct=" << s.cpuct << " | fpu_reduction=" << s.puct_fpu_reduction;
    }
    if (s.use_halving || !s.use_puct) {
        o << " | sigma_cap=" << s.mcts.gumbel_sigma_cap
          << " | c_visit=" << s.mcts.gumbel_c_visit
          << " | c_scale=" << s.mcts.gumbel_c_scale;
        if (!s.use_puct)  o << " | explore_c=" << s.mcts.explore_c;
        if (s.use_halving) o << " | m=" << s.mcts.gumbel_m;
    }
    o << " | overrides=[";
    for (size_t i = 0; i < s.overrides.size(); ++i) {
        if (i) o << ", ";
        o << s.overrides[i];
    }
    o << "]";
    return o.str();
}

// =============================================================================
// CONFIG
// =============================================================================
struct TournamentConfig {
    // logging
    std::string base_log_dir;
    int main_logging_level        = 20;
    int worker_logging_level      = 50;
    int batcher_logging_level     = 20;
    int logging_interval_sec      = 10;

    // cpu pinning
    std::vector<int> main_cores;
    std::vector<int> batcher_a_cores;
    std::vector<int> batcher_b_cores;
    std::vector<int> game_worker_cores;
    int workers_per_core          = 1;
    int num_workers               = 0;

    // inference
    int inference_batch_size      = 128;
    int max_batch_size            = 0;
    int batch_timeout_ms          = 10;

    // model dims (shared: both sides must use the same input encoding)
    int input_planes              = 0;
    int board_dim                 = 0;
    int policy_moves              = 0;

    // per-side search config
    SideConfig side_a;
    SideConfig side_b;

    PoolSizingConfig pool_sizing;

    // tournament shape
    int games_per_match           = 100;
    int max_ply_length            = 400;
    std::string opening_file;
    int opening_plies             = 0;     // 0 = play the full book line

    // tablebase
    bool tablebase_enabled        = false;
    std::string tablebase_path;
};

static TournamentConfig load_config(const CliArgs& args) {
    const std::string& yaml_path = args.config_file;
    TournamentConfig cfg;

    YAML::Node root = YAML::LoadFile(yaml_path);
    YAML::Node global = root["global"];
    if (!global) throw std::runtime_error(yaml_path + " missing 'global:' block");

    cfg.base_log_dir       = global["log_dir"].as<std::string>();
    cfg.main_logging_level = global["main_logging_level"].as<int>();

    const std::string model_yaml_path = global["model_file"].as<std::string>();
    YAML::Node model = YAML::LoadFile(model_yaml_path);
    cfg.input_planes = model["model"]["input_planes"].as<int>();
    cfg.board_dim    = model["model"]["board_dim"].as<int>();
    cfg.policy_moves = model["model"]["total_policy_moves"].as<int>();

    YAML::Node cores_n = root["core_pinning"];
    if (!cores_n) throw std::runtime_error(yaml_path + " missing 'core_pinning:' block");
    if (cores_n["main"])
        for (const auto& c : cores_n["main"]) cfg.main_cores.push_back(c.as<int>());

    YAML::Node infer_n = root["inference"];
    if (!infer_n) throw std::runtime_error(yaml_path + " missing 'inference:' block");
    for (const auto& c : cores_n["batcher_a"]) cfg.batcher_a_cores.push_back(c.as<int>());
    for (const auto& c : cores_n["batcher_b"]) cfg.batcher_b_cores.push_back(c.as<int>());
    cfg.inference_batch_size  = infer_n["batch_size"].as<int>();
    cfg.max_batch_size        = cfg.inference_batch_size * infer_n["batch_size_factor"].as<int>();
    cfg.batch_timeout_ms      = infer_n["batch_timeout_ms"].as<int>();
    cfg.batcher_logging_level = infer_n["logging_level"].as<int>();
    cfg.logging_interval_sec  = infer_n["logging_interval_sec"].as<int>();

    cfg.side_a = load_side(root, args.name_a, args.model_a_path, args.overrides_a);
    cfg.side_b = load_side(root, args.name_b, args.model_b_path, args.overrides_b);

    YAML::Node pool_n = root["pool_sizing"];
    if (!pool_n) throw std::runtime_error(yaml_path + " missing 'pool_sizing:' block");
    cfg.pool_sizing.avg_branching       = pool_n["avg_branching"].as<double>();
    cfg.pool_sizing.node_safety_factor  = pool_n["node_safety_factor"].as<double>();
    cfg.pool_sizing.edge_safety_factor  = pool_n["edge_safety_factor"].as<double>();
    cfg.pool_sizing.node_hard_cap_bytes = (size_t)pool_n["node_hard_cap_mb"].as<size_t>() * 1024ull * 1024ull;
    cfg.pool_sizing.edge_hard_cap_bytes = (size_t)pool_n["edge_hard_cap_mb"].as<size_t>() * 1024ull * 1024ull;

    YAML::Node tbn = root["tablebase"];
    if (tbn) {
        cfg.tablebase_enabled = tbn["enabled"].as<bool>();
        if (cfg.tablebase_enabled) cfg.tablebase_path = tbn["path"].as<std::string>();
    }

    YAML::Node t_n = root["tournament"];
    if (!t_n) throw std::runtime_error(yaml_path + " missing 'tournament:' block");
    for (const auto& c : cores_n["game_worker"]) cfg.game_worker_cores.push_back(c.as<int>());
    cfg.workers_per_core     = t_n["workers_per_core"].as<int>();
    cfg.num_workers          = (int)cfg.game_worker_cores.size() * cfg.workers_per_core;
    cfg.games_per_match      = t_n["games_per_match"].as<int>();
    cfg.max_ply_length       = t_n["max_ply_length"].as<int>();
    cfg.opening_file         = t_n["opening_file"].as<std::string>();
    cfg.opening_plies        = t_n["opening_plies"].as<int>();
    cfg.worker_logging_level = t_n["worker_logging_level"].as<int>();

    if (cfg.num_workers <= 0) {
        throw std::runtime_error("no game workers: core_pinning.game_worker is empty or workers_per_core <= 0");
    }
    if (cfg.opening_plies < 0) {
        throw std::runtime_error("tournament.opening_plies must be >= 0 (0 = full book line)");
    }
    if (cfg.games_per_match % 2 != 0) {
        throw std::runtime_error("tournament.games_per_match must be even (each opening plays twice with sides swapped)");
    }

    return cfg;
}

// =============================================================================
// SHARED BUFFERS
// =============================================================================
struct SharedBuffers {
    std::vector<torch::Tensor> input;
    std::vector<torch::Tensor> policy;
    std::vector<torch::Tensor> value;
};
static SharedBuffers make_shared_buffers(const TournamentConfig& cfg) {
    SharedBuffers b;
    auto opts = torch::TensorOptions().dtype(torch::kHalf).device(torch::kCPU);
    for (int i = 0; i < cfg.max_batch_size; ++i) {
        b.input.push_back(torch::zeros({cfg.input_planes, cfg.board_dim, cfg.board_dim}, opts));
        b.policy.push_back(torch::zeros({cfg.policy_moves}, opts));
        b.value.push_back(torch::zeros({3}, opts));
    }
    return b;
}

// =============================================================================
// WORKER STATE
// =============================================================================
struct WorkerState {
    int worker_id = 0;
    std::unique_ptr<Logger>         logger;
    std::unique_ptr<MCTSEngine>     engine_a;
    std::unique_ptr<MCTSEngine>     engine_b;
    std::unique_ptr<ActionSelector> selector_a;
    std::unique_ptr<ActionSelector> selector_b;
};

static std::unique_ptr<MCTSEngine> build_engine(
    const TournamentConfig& cfg,
    const SideConfig& side,
    moodycamel::ConcurrentQueue<std::pair<int, int>>& inf_queue,
    ThreadSafeQueue<std::vector<int>>& result_queue,
    int worker_id,
    SharedBuffers& buf,
    ThreadSafeQueue<int>& free_slots,
    std::atomic<int>* core_wait_count,
    Logger& logger,
    bool tb_ready,
    const chess::Board& dummy_board)
{
    // Size the initial pool for this side's exact per-move budget, so the
    // first reset() doesn't need to grow.
    const PoolTargets initial = MCTSEngine::predict_pool_needs_static(
        side.mcts.search_depth, cfg.pool_sizing);

    auto engine = std::make_unique<MCTSEngine>(
        side.mcts,
        (int)initial.node_target,
        (int)initial.edge_target,
        inf_queue,
        result_queue,
        worker_id,
        dummy_board,
        std::vector<chess::Board>(),
        logger,
        buf.input, buf.policy, buf.value,
        free_slots,
        core_wait_count,
        cfg.workers_per_core,
        tb_ready
    );

    engine->pool_sizing_cfg            = cfg.pool_sizing;
    engine->early_stop_q_gap           = side.early_stop_q_gap;
    engine->early_stop_min_visits      = side.early_stop_min_visits;
    engine->early_return_on_forced_win = side.early_return_on_forced_win;
    engine->use_puct                   = side.use_puct;
    engine->cpuct                      = side.cpuct;
    engine->puct_fpu_reduction         = side.puct_fpu_reduction;
    return engine;
}

// =============================================================================
// MAIN
// =============================================================================
int main(int argc, char* argv[]) {
    CliArgs args;
    if (!parse_args(argc, argv, args)) return 1;

    TournamentConfig cfg;
    try {
        cfg = load_config(args);
    } catch (const std::exception& e) {
        std::cerr << "Fatal: failed to load config " << args.config_file
                  << ": " << e.what() << std::endl;
        return 1;
    }

    // Pairing dir + logging ---------------------------------------------------
    const std::string& name_a = cfg.side_a.name;
    const std::string& name_b = cfg.side_b.name;
    const std::string pairing_dir = args.run_dir + "/" + name_a + "_vs_" + name_b;
    fs::create_directories(pairing_dir);

    Logger main_logger("tournament_main", pairing_dir, cfg.main_logging_level);
    main_logger.rotate(0, 0);
    main_logger.log("INFO", "===== TOURNAMENT: " + name_a + " vs " + name_b + " =====");
    main_logger.log("INFO", "Config: " + args.config_file);
    main_logger.log("INFO", "Pairing dir: " + pairing_dir);
    main_logger.log("INFO", "Side A: " + describe_side(cfg.side_a));
    main_logger.log("INFO", "Side B: " + describe_side(cfg.side_b));
    main_logger.log("INFO", "Workers: " + std::to_string(cfg.num_workers) +
                    " (" + std::to_string(cfg.game_worker_cores.size()) +
                    " cores x " + std::to_string(cfg.workers_per_core) + " per core)");

    // Pin main
    if (!cfg.main_cores.empty()) {
        cpu_set_t s;
        CPU_ZERO(&s);
        for (int c : cfg.main_cores) if (c >= 0 && c < CPU_SETSIZE) CPU_SET(c, &s);
        if (CPU_COUNT(&s) > 0) pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &s);
    }

    // Tablebase --------------------------------------------------------------
    bool tb_ready = false;
    if (cfg.tablebase_enabled) {
        if (tb_init(cfg.tablebase_path.c_str())) {
            tb_ready = (TB_LARGEST > 0);
            main_logger.log("INFO", "Tablebase initialized (TB_LARGEST=" +
                            std::to_string(TB_LARGEST) + ")");
        } else {
            main_logger.log("ERROR", "tb_init failed for " + cfg.tablebase_path);
        }
    }

    // Openings + game specs ---------------------------------------------------
    OpeningBook book;
    {
        std::string err;
        if (!book.load(cfg.opening_file, err)) {
            main_logger.log("CRITICAL", "Failed to load opening file: " + err);
            if (tb_ready) tb_free();
            return 1;
        }
    }
    // Truncate every book line to opening_plies (0 = keep full line), then
    // de-duplicate in file order. Truncation makes many lines identical (e.g.
    // at 2 plies most of the book collapses to a few dozen lines), and with
    // deterministic search a repeated opening replays the same game -- so
    // duplicates would add games without adding information.
    const int num_openings = cfg.games_per_match / 2;
    std::vector<Opening> openings;
    {
        std::unordered_set<std::string> seen;
        for (const Opening& src : book.all()) {
            if ((int)openings.size() >= num_openings) break;
            Opening o = src;
            if (cfg.opening_plies > 0 && (int)o.san_moves.size() > cfg.opening_plies) {
                o.san_moves.resize(cfg.opening_plies);
            }
            std::string key;
            for (const auto& m : o.san_moves) { key += m; key += ' '; }
            if (seen.insert(key).second) openings.push_back(std::move(o));
        }
    }
    main_logger.log("INFO", "Openings: " + std::to_string(openings.size()) +
                    " unique lines at opening_plies=" + std::to_string(cfg.opening_plies) +
                    " (book has " + std::to_string(book.size()) + " lines)");
    if ((int)openings.size() < num_openings) {
        main_logger.log("WARNING", "Requested " + std::to_string(num_openings) +
                        " openings but only " + std::to_string(openings.size()) +
                        " unique lines exist at this depth; playing " +
                        std::to_string(openings.size() * 2) + " games instead.");
    }

    struct GameSpec {
        int game_number;
        Opening opening;
        chess::Color model_a_color;
    };
    std::vector<GameSpec> specs;
    specs.reserve(openings.size() * 2);
    for (size_t i = 0; i < openings.size(); ++i) {
        specs.push_back({(int)(2*i),   openings[i], chess::Color::WHITE});
        specs.push_back({(int)(2*i+1), openings[i], chess::Color::BLACK});
    }
    main_logger.log("INFO", "Total games: " + std::to_string(specs.size()));

    // games.pgn writer -------------------------------------------------------
    // Shared across all worker threads; sessions lock the mutex before writing.
    const std::string pgn_path = pairing_dir + "/games.pgn";
    std::ofstream pgn_out(pgn_path, std::ios::out | std::ios::trunc);
    if (!pgn_out) {
        main_logger.log("CRITICAL", "Could not open games.pgn for write: " + pgn_path);
        if (tb_ready) tb_free();
        return 1;
    }
    std::mutex pgn_mutex;
    PgnFileSink pgn_sink{&pgn_out, &pgn_mutex};

    // Shared buffers ---------------------------------------------------------
    SharedBuffers buf_a = make_shared_buffers(cfg);
    SharedBuffers buf_b = make_shared_buffers(cfg);

    // Inference queues + slots -----------------------------------------------
    moodycamel::ConcurrentQueue<std::pair<int, int>> inf_queue_a;
    moodycamel::ConcurrentQueue<std::pair<int, int>> inf_queue_b;
    std::vector<ThreadSafeQueue<std::vector<int>>> result_queues_a(cfg.num_workers);
    std::vector<ThreadSafeQueue<std::vector<int>>> result_queues_b(cfg.num_workers);
    ThreadSafeQueue<int> free_slots_a, free_slots_b;
    for (int i = 0; i < cfg.max_batch_size; ++i) {
        free_slots_a.push(i);
        free_slots_b.push(i);
    }

    // Ensure TRT engines exist -----------------------------------------------
    const std::string engine_a = ensure_engine(
        cfg.side_a.model_path, cfg.max_batch_size, cfg.input_planes, main_logger);
    const std::string engine_b = ensure_engine(
        cfg.side_b.model_path, cfg.max_batch_size, cfg.input_planes, main_logger);
    if (engine_a.empty() || engine_b.empty()) {
        main_logger.log("CRITICAL", "Engine build failed; aborting pairing.");
        if (tb_ready) tb_free();
        return 1;
    }

    // Batchers ---------------------------------------------------------------
    std::atomic<uint64_t> dummy_step{0};
    InferenceBatcher batcher_a(
        engine_a, cfg.inference_batch_size, cfg.batch_timeout_ms, cfg.num_workers,
        pairing_dir, cfg.batcher_logging_level, cfg.batcher_a_cores,
        0, dummy_step, cfg.logging_interval_sec, "batcher_a");
    InferenceBatcher batcher_b(
        engine_b, cfg.inference_batch_size, cfg.batch_timeout_ms, cfg.num_workers,
        pairing_dir, cfg.batcher_logging_level, cfg.batcher_b_cores,
        0, dummy_step, cfg.logging_interval_sec, "batcher_b");

    std::atomic<bool> stop_event{false};
    std::thread batcher_a_thread([&]() {
        batcher_a.run(inf_queue_a, result_queues_a,
                      buf_a.input, buf_a.policy, buf_a.value,
                      stop_event, &free_slots_a);
    });
    std::thread batcher_b_thread([&]() {
        batcher_b.run(inf_queue_b, result_queues_b,
                      buf_b.input, buf_b.policy, buf_b.value,
                      stop_event, &free_slots_b);
    });

    // Per-core spin_wait counters --------------------------------------------
    std::vector<std::unique_ptr<std::atomic<int>>> core_wait_counts;
    core_wait_counts.reserve(cfg.game_worker_cores.size());
    for (size_t i = 0; i < cfg.game_worker_cores.size(); ++i) {
        core_wait_counts.push_back(std::make_unique<std::atomic<int>>(0));
    }

    // Build worker states ----------------------------------------------------
    chess::Board dummy_board;
    dummy_board.setFen(chess::constants::STARTPOS);

    std::vector<std::unique_ptr<WorkerState>> workers;
    workers.reserve(cfg.num_workers);
    for (int w = 0; w < cfg.num_workers; ++w) {
        auto ws = std::make_unique<WorkerState>();
        ws->worker_id = w;
        ws->logger = std::make_unique<Logger>(
            "game_worker_" + std::to_string(w),
            pairing_dir, cfg.worker_logging_level);

        ws->logger->rotate(0, 0);
        const int core_idx = w / cfg.workers_per_core;
        std::atomic<int>* wait_count = core_wait_counts[core_idx].get();

        ws->engine_a = build_engine(cfg, cfg.side_a, inf_queue_a, result_queues_a[w], w,
                                    buf_a, free_slots_a, wait_count,
                                    *ws->logger, tb_ready, dummy_board);
        ws->engine_b = build_engine(cfg, cfg.side_b, inf_queue_b, result_queues_b[w], w,
                                    buf_b, free_slots_b, wait_count,
                                    *ws->logger, tb_ready, dummy_board);

        ws->selector_a = std::make_unique<ActionSelector>(
            "selector_a_w" + std::to_string(w), w, cfg.side_a.selector, *ws->logger);
        ws->selector_b = std::make_unique<ActionSelector>(
            "selector_b_w" + std::to_string(w), w, cfg.side_b.selector, *ws->logger);

        workers.push_back(std::move(ws));
    }

    // Shared game queue -------------------------------------------------------
    // Workers pull the next unplayed game when they finish one, instead of
    // being handed a fixed list up front. Short games no longer leave a worker
    // idle while another still has several queued, and the match ends as soon
    // as the last game finishes.
    std::atomic<size_t> next_spec{0};

    // Shared counters for the simple summary ---------------------------------
    // W/D/L from side A's perspective. Elo is computed downstream from games.pgn.
    std::atomic<int> a_wins{0}, b_wins{0}, draws{0}, total_plies{0}, games_completed{0};

    // Launch workers ---------------------------------------------------------
    std::vector<std::thread> worker_threads;
    worker_threads.reserve(cfg.num_workers);
    const auto tstart = std::chrono::steady_clock::now();

    // Workers are started kWorkerStaggerMs apart so their first searches
    // don't all hit the batchers at the same instant. Only the start is
    // staggered; after the first game the workers drift apart naturally.
    constexpr int kWorkerStaggerMs = 100;
    for (int w = 0; w < cfg.num_workers; ++w) {
        if (w > 0) std::this_thread::sleep_for(std::chrono::milliseconds(kWorkerStaggerMs));
        worker_threads.emplace_back([&, w]() {
            const int core = cfg.game_worker_cores[w / cfg.workers_per_core];
            if (core >= 0 && core < CPU_SETSIZE) {
                cpu_set_t s;
                CPU_ZERO(&s);
                CPU_SET(core, &s);
                pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &s);
            }
            at::set_num_threads(1);

            WorkerState& ws = *workers[w];

            SearchAgent agent_a{
                *ws.engine_a, *ws.selector_a,
                cfg.side_a.mcts.search_depth,
                cfg.side_a.mcts.gumbel_m,
                cfg.side_a.use_halving,
                cfg.side_a.tree_reuse,
                cfg.side_a.use_puct
            };
            SearchAgent agent_b{
                *ws.engine_b, *ws.selector_b,
                cfg.side_b.mcts.search_depth,
                cfg.side_b.mcts.gumbel_m,
                cfg.side_b.use_halving,
                cfg.side_b.tree_reuse,
                cfg.side_b.use_puct
            };

            // One JSON line per search (see SearchAgent::write_stats). Both
            // agents of this worker run on this thread, so no locking.
            std::ofstream stats_file(pairing_dir + "/search_stats_worker_" +
                                     std::to_string(w) + ".jsonl");
            agent_a.search_label = search_name(cfg.side_a);
            agent_b.search_label = search_name(cfg.side_b);
            agent_a.stats_out = &stats_file;
            agent_a.player    = name_a;
            agent_b.stats_out = &stats_file;
            agent_b.player    = name_b;

            for (size_t idx = next_spec.fetch_add(1); idx < specs.size();
                 idx = next_spec.fetch_add(1)) {
                const GameSpec& spec = specs[idx];
                const chess::Color a_color = spec.model_a_color;
                agent_a.game = spec.game_number;
                agent_b.game = spec.game_number;
                SearchAgent white_agent = (a_color == chess::Color::WHITE) ? agent_a : agent_b;
                SearchAgent black_agent = (a_color == chess::Color::WHITE) ? agent_b : agent_a;
                const chess::Color our_side = a_color;

                // Per-game PGN metadata: player names depend on which colour
                // side A plays this game; everything else is constant across
                // the pairing.
                SessionPgnMetadata pgn_meta;
                pgn_meta.event      = name_a + " vs " + name_b;
                pgn_meta.site       = "Talbot C++ Engine";
                pgn_meta.white_name = (a_color == chess::Color::WHITE) ? name_a : name_b;
                pgn_meta.black_name = (a_color == chess::Color::WHITE) ? name_b : name_a;
                pgn_meta.round      = std::to_string(spec.game_number + 1);
                pgn_meta.config     = PgnConfig::annotated();

                SelfPlaySession session(
                    white_agent, black_agent,
                    our_side, spec.opening, cfg.max_ply_length,
                    *ws.logger, pgn_sink, pgn_meta);

                GameWorker gw(w, agent_a, *ws.logger);
                SessionResult result = gw.run_one_game(session);

                // Tally: white_value is from White's perspective. Convert to
                // an A-score: +1 A won, -1 B won, 0 draw.
                const double a_score = (a_color == chess::Color::WHITE)
                    ? result.white_value : -result.white_value;
                if      (a_score >  0.5) a_wins.fetch_add(1);
                else if (a_score < -0.5) b_wins.fetch_add(1);
                else                     draws.fetch_add(1);
                total_plies.fetch_add(session.total_plies());
                games_completed.fetch_add(1);
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        });
    }

    for (auto& t : worker_threads) t.join();

    const auto tend = std::chrono::steady_clock::now();
    const double wall_s = std::chrono::duration<double>(tend - tstart).count();
    main_logger.log("INFO", "All games complete in " +
                    std::to_string((int)wall_s) + " seconds.");

    // Close PGN cleanly ------------------------------------------------------
    pgn_out.flush();
    pgn_out.close();

    // Summary (counts only, no Elo) ------------------------------------------
    const int wa = a_wins.load();
    const int wb = b_wins.load();
    const int dr = draws.load();
    const int n  = wa + wb + dr;
    const int tp = total_plies.load();
    const double avg_plies = n > 0 ? (double)tp / n : 0.0;

    char buf[512];
    snprintf(buf, sizeof(buf),
        "\n"
        "=================================================\n"
        " %s vs %s\n"
        "-------------------------------------------------\n"
        "  games       : %d\n"
        "  A wins      : %d\n"
        "  draws       : %d\n"
        "  B wins      : %d\n"
        "  avg plies   : %.1f\n"
        "  wall seconds: %d\n"
        "  games.pgn   : %s\n"
        "=================================================\n",
        name_a.c_str(), name_b.c_str(),
        n, wa, dr, wb, avg_plies, (int)wall_s, pgn_path.c_str());
    main_logger.log("CRITICAL", buf);

    // summary.txt for the Python orchestrator / downstream scripts.
    // Grep-friendly key=value. NO Elo -- compute that from games.pgn downstream.
    // model_a / model_b keep their old meaning (the side names, which default
    // to the file stems), so the existing orchestrator keeps working.
    auto join = [](const std::vector<std::string>& v) {
        std::string s;
        for (size_t i = 0; i < v.size(); ++i) { if (i) s += ";"; s += v[i]; }
        return s;
    };
    const std::string summary_path = pairing_dir + "/summary.txt";
    std::ofstream sf(summary_path);
    if (sf) {
        sf << "model_a="      << name_a << "\n";
        sf << "model_b="      << name_b << "\n";
        sf << "mode=fixed\n";
        sf << "path_a="       << cfg.side_a.model_path << "\n";
        sf << "path_b="       << cfg.side_b.model_path << "\n";
        sf << "nodes_a="      << cfg.side_a.mcts.search_depth << "\n";
        sf << "nodes_b="      << cfg.side_b.mcts.search_depth << "\n";
        sf << "search_a="     << search_name(cfg.side_a) << "\n";
        sf << "search_b="     << search_name(cfg.side_b) << "\n";
        sf << "cpuct_a="      << cfg.side_a.cpuct << "\n";
        sf << "cpuct_b="      << cfg.side_b.cpuct << "\n";
        sf << "tree_reuse_a=" << (cfg.side_a.tree_reuse && !cfg.side_a.use_halving) << "\n";
        sf << "tree_reuse_b=" << (cfg.side_b.tree_reuse && !cfg.side_b.use_halving) << "\n";
        sf << "overrides_a="  << join(cfg.side_a.overrides) << "\n";
        sf << "overrides_b="  << join(cfg.side_b.overrides) << "\n";
        sf << "games="        << n  << "\n";
        sf << "a_wins="       << wa << "\n";
        sf << "draws="        << dr << "\n";
        sf << "b_wins="       << wb << "\n";
        sf << "avg_plies="    << avg_plies << "\n";
        sf << "wall_seconds=" << (int)wall_s << "\n";
        sf << "opening_plies=" << cfg.opening_plies << "\n";
        sf << "pgn_file="     << pgn_path << "\n";
    }

    // Shutdown ---------------------------------------------------------------
    stop_event.store(true);
    if (batcher_a_thread.joinable()) batcher_a_thread.join();
    if (batcher_b_thread.joinable()) batcher_b_thread.join();

    if (tb_ready) tb_free();

    return 0;
}