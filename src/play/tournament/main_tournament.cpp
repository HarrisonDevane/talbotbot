// =============================================================================
// main_tournament.cpp
//
// Entry point for talbot_tournament.exe -- one process, one pairing.
//
// LAUNCH CONTRACT (matches the Python orchestrator's subprocess call):
//   talbot_tournament.exe --tournament               # ignored flag, kept for compat
//                          --config_file  <yaml>
//                          --model_a      <path>
//                          --model_b      <path>
//                          --run_dir      <dir>     # top-level; we make a subdir per pairing
//
// WHAT IT DOES:
//   1. Loads YAML config.
//   2. Builds TWO InferenceBatchers, one per model, sharing the GPU.
//   3. Spawns N game-worker threads; each holds TWO MCTSEngines (one per batcher)
//      so it can play either model on either side.
//   4. Generates game specs: games_per_match / 2 openings, each played TWICE
//      with sides swapped so opening effects cancel.
//   5. Distributes specs across workers stride-style; each plays its subset
//      sequentially via SelfPlaySession + GameWorker::run_one_game.
//   6. Mode is fixed-depth OR timed depending on YAML tournament.mode.
//
// OUTPUTS (in <run_dir>/<A>_vs_<B>/):
//   games.pgn   -- every game concatenated as a PGN block. THIS is the file
//                  the downstream Elo/rating script consumes. Sessions write
//                  atomically under a shared mutex; games arrive in completion
//                  order, not launch order.
//   summary.txt -- plain wins/draws/losses counts + total plies. NO Elo --
//                  ratings are computed downstream by whatever script the
//                  operator points at games.pgn.
//   per-worker log files -- for diagnostics.
// =============================================================================

#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>

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

#include "concurrentqueue.h"
#include <torch/torch.h>
#include <yaml-cpp/yaml.h>

#include "chess.hpp"
#include "mcts_engine.hpp"
#include "action_selector.hpp"
#include "inference_batcher.hpp"
#include "logger.hpp"
#include "time_control.hpp"
#include "game_worker.hpp"
#include "self_play_session.hpp"
#include "opening_book.hpp"

#include "tbprobe.h"

namespace fs = std::filesystem;

// =============================================================================
// TRT COMPILE (out-of-process, mirrors main_train.cpp)
//
// Engine building lives in talbot_trt_compile.exe. This exe finds it next to
// itself (same build dir) and shells out synchronously. The engine is written
// to disk at a deterministic path derived from the ONNX path; if it already
// exists we skip the build. Python's tournament orchestrator puts ONNX +
// engine files in a per-run tmp dir, so within a run the SAME checkpoint is
// only compiled once (first pairing that needs it builds; subsequent pairings
// hit cache).
// =============================================================================
static std::string resolve_trt_compile_exe() {
    char buf[MAX_PATH];
    DWORD n = GetModuleFileNameA(nullptr, buf, MAX_PATH);
    if (n == 0 || n == MAX_PATH) return "talbot_trt_compile.exe";
    return (fs::path(std::string(buf, n)).parent_path()
            / "talbot_trt_compile.exe").string();
}

// Run talbot_trt_compile.exe synchronously. Returns exit code, -1 on launch
// failure. Copies the CreateProcess pattern from main_train.cpp verbatim so
// there is one source of truth for how we launch that exe.
static int run_trt_compile(const std::string& compile_exe,
                           const std::string& onnx_path,
                           const std::string& engine_path,
                           int max_batch,
                           int input_planes) {
    std::string cmd = "\"" + compile_exe + "\""
                    + " \"" + onnx_path + "\""
                    + " \"" + engine_path + "\""
                    + " --input-planes " + std::to_string(input_planes)
                    + " --max-batch "    + std::to_string(max_batch)
                    + " --force";

    SECURITY_ATTRIBUTES sa = {};
    sa.nLength        = sizeof(sa);
    sa.bInheritHandle = TRUE;
    HANDLE null_out = CreateFileA("NUL", GENERIC_WRITE,
                                  FILE_SHARE_WRITE | FILE_SHARE_READ,
                                  &sa, OPEN_EXISTING, 0, nullptr);
    if (null_out == INVALID_HANDLE_VALUE) return -1;

    STARTUPINFOA si = {};
    si.cb         = sizeof(si);
    si.dwFlags    = STARTF_USESTDHANDLES;
    si.hStdInput  = GetStdHandle(STD_INPUT_HANDLE);
    si.hStdOutput = null_out;
    si.hStdError  = GetStdHandle(STD_ERROR_HANDLE);
    PROCESS_INFORMATION pi = {};

    std::vector<char> mutable_cmd(cmd.begin(), cmd.end());
    mutable_cmd.push_back('\0');

    BOOL ok = CreateProcessA(
        nullptr, mutable_cmd.data(),
        nullptr, nullptr,
        TRUE, 0,
        nullptr, nullptr,
        &si, &pi);
    if (!ok) {
        CloseHandle(null_out);
        return -1;
    }

    WaitForSingleObject(pi.hProcess, INFINITE);
    DWORD exit_code = 0;
    GetExitCodeProcess(pi.hProcess, &exit_code);
    CloseHandle(pi.hProcess);
    CloseHandle(pi.hThread);
    CloseHandle(null_out);
    return static_cast<int>(exit_code);
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
        logger.log("CRITICAL", "talbot_trt_compile.exe failed with exit code " +
                               std::to_string(rc) + " (onnx=" + onnx_path + ")");
        return "";
    }
    logger.log("INFO", "Engine built: " + engine_path.string());
    return engine_path.string();
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

    // model dims
    int input_planes              = 0;
    int board_dim                 = 0;
    int policy_moves              = 0;

    // selection / mcts
    ActionSelectorConfig selector;
    PoolSizingConfig pool_sizing;

    // early-stop
    double early_stop_q_gap       = 0.0;
    int    early_stop_min_visits  = 0;
    bool   early_return_on_forced_win = false;

    // time control (only used in timed mode; still required in YAML)
    TimeControlConfig time_control;

    // tournament shape
    enum class Mode { FIXED, TIMED };
    Mode mode                     = Mode::FIXED;
    int games_per_match           = 100;
    int max_ply_length            = 400;
    std::string opening_file;

    int64_t initial_time_ms       = 60000;
    int64_t increment_ms          = 1000;

    // tablebase
    bool tablebase_enabled        = false;
    std::string tablebase_path;
};

static TournamentConfig load_config(const std::string& yaml_path) {
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

    YAML::Node eval_n = root["evaluation"];
    if (eval_n && eval_n["main_cores"])
        for (const auto& c : eval_n["main_cores"]) cfg.main_cores.push_back(c.as<int>());

    YAML::Node infer_n = root["inference"];
    if (!infer_n) throw std::runtime_error(yaml_path + " missing 'inference:' block");
    for (const auto& c : infer_n["batcher_a_cores"]) cfg.batcher_a_cores.push_back(c.as<int>());
    for (const auto& c : infer_n["batcher_b_cores"]) cfg.batcher_b_cores.push_back(c.as<int>());
    cfg.inference_batch_size  = infer_n["batch_size"].as<int>();
    cfg.max_batch_size        = cfg.inference_batch_size * infer_n["batch_size_factor"].as<int>();
    cfg.batch_timeout_ms      = infer_n["batch_timeout_ms"].as<int>();
    cfg.batcher_logging_level = infer_n["logging_level"].as<int>();
    cfg.logging_interval_sec  = infer_n["logging_interval_sec"].as<int>();

    YAML::Node mcts_n = root["mcts"];
    if (!mcts_n) throw std::runtime_error(yaml_path + " missing 'mcts:' block");
    ActionSelectorConfig& s = cfg.selector;
    s.virtual_loss           = mcts_n["virtual_loss"].as<double>();
    s.contempt               = mcts_n["contempt"].as<double>();
    s.deficit_eps            = mcts_n["deficit_eps"].as<double>();
    s.two_fold_repetition    = mcts_n["two_fold_repetition"].as<bool>();
    s.gumbel_c_visit         = mcts_n["gumbel_c_visit"].as<double>();
    s.gumbel_c_scale         = mcts_n["gumbel_c_scale"].as<double>();
    s.gumbel_noise           = mcts_n["gumbel_noise"].as<double>();
    s.gumbel_search_depth    = mcts_n["gumbel_search_depth"].as<double>();
    s.gumbel_m               = mcts_n["gumbel_m"].as<double>();
    s.batch_size_per_worker  = mcts_n["worker_minibatch_size"].as<int>();

    YAML::Node sel_n = root["selection"];
    s.draw_cutoff            = sel_n["draw_cutoff"].as<double>();
    s.temperature_ply_cutoff = sel_n["temperature_ply_cutoff"].as<int>();
    s.temperature_q_decay    = sel_n["temperature_q_decay"].as<double>();
    s.resignation_probability= sel_n["resignation_probability"].as<double>();
    s.resignation_cutoff     = sel_n["resignation_cutoff"].as<double>();

    if (mcts_n["early_stop_q_gap"])            cfg.early_stop_q_gap = mcts_n["early_stop_q_gap"].as<double>();
    if (mcts_n["early_stop_min_visits"])       cfg.early_stop_min_visits = mcts_n["early_stop_min_visits"].as<int>();
    if (mcts_n["early_return_on_forced_win"])  cfg.early_return_on_forced_win = mcts_n["early_return_on_forced_win"].as<bool>();

    YAML::Node pool_n = root["pool_sizing"];
    if (!pool_n) throw std::runtime_error(yaml_path + " missing 'pool_sizing:' block");
    cfg.pool_sizing.avg_branching       = pool_n["avg_branching"].as<double>();
    cfg.pool_sizing.node_safety_factor  = pool_n["node_safety_factor"].as<double>();
    cfg.pool_sizing.edge_safety_factor  = pool_n["edge_safety_factor"].as<double>();
    cfg.pool_sizing.node_hard_cap_bytes = (size_t)pool_n["node_hard_cap_mb"].as<size_t>() * 1024ull * 1024ull;
    cfg.pool_sizing.edge_hard_cap_bytes = (size_t)pool_n["edge_hard_cap_mb"].as<size_t>() * 1024ull * 1024ull;

    YAML::Node tc_n = root["time_control"];
    if (!tc_n) throw std::runtime_error(yaml_path + " missing 'time_control:' block");
    TimeControlConfig& tc = cfg.time_control;
    tc.move_horizon       = tc_n["move_horizon"].as<double>();
    tc.increment_fraction = tc_n["increment_fraction"].as<double>();
    tc.base_fraction      = tc_n["base_fraction"].as<double>();
    tc.hard_multiplier    = tc_n["hard_multiplier"].as<double>();
    tc.max_time_fraction  = tc_n["max_time_fraction"].as<double>();
    tc.move_overhead_ms   = tc_n["move_overhead_ms"].as<int64_t>();
    tc.min_think_ms       = tc_n["min_think_ms"].as<int64_t>();
    tc.nps_ewma_alpha     = tc_n["nps_ewma_alpha"].as<double>();
    tc.nps_ewma           = tc_n["nps_ewma_default"].as<double>();

    YAML::Node tbn = root["tablebase"];
    if (tbn) {
        cfg.tablebase_enabled = tbn["enabled"].as<bool>();
        if (cfg.tablebase_enabled) cfg.tablebase_path = tbn["path"].as<std::string>();
    }

    YAML::Node t_n = root["tournament"];
    if (!t_n) throw std::runtime_error(yaml_path + " missing 'tournament:' block");
    for (const auto& c : t_n["game_worker_cores"]) cfg.game_worker_cores.push_back(c.as<int>());
    cfg.workers_per_core   = t_n["workers_per_core"].as<int>();
    cfg.num_workers        = (int)cfg.game_worker_cores.size() * cfg.workers_per_core;
    cfg.games_per_match    = t_n["games_per_match"].as<int>();
    cfg.max_ply_length     = t_n["max_ply_length"].as<int>();
    cfg.opening_file       = t_n["opening_file"].as<std::string>();
    cfg.worker_logging_level = t_n["worker_logging_level"].as<int>();

    const std::string mode_str =
        t_n["mode"] ? t_n["mode"].as<std::string>() : std::string("fixed");
    if (mode_str == "fixed") cfg.mode = TournamentConfig::Mode::FIXED;
    else if (mode_str == "timed") cfg.mode = TournamentConfig::Mode::TIMED;
    else throw std::runtime_error("tournament.mode must be 'fixed' or 'timed', got '" + mode_str + "'");

    if (cfg.mode == TournamentConfig::Mode::TIMED) {
        YAML::Node tt = t_n["time_control_per_game"];
        if (!tt) throw std::runtime_error(
            "tournament.mode=timed requires 'tournament.time_control_per_game' block");
        cfg.initial_time_ms = tt["initial_time_ms"].as<int64_t>();
        cfg.increment_ms    = tt["increment_ms"].as<int64_t>();
    }

    if (cfg.games_per_match % 2 != 0) {
        throw std::runtime_error("tournament.games_per_match must be even (each opening plays twice with sides swapped)");
    }

    return cfg;
}

// =============================================================================
// CLI
// =============================================================================
struct CliArgs {
    std::string config_file;
    std::string model_a_path;
    std::string model_b_path;
    std::string run_dir;
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
        else {
            std::cerr << "Unknown arg: " << k << "\n";
            return false;
        }
    }
    if (out.config_file.empty() || out.model_a_path.empty() ||
        out.model_b_path.empty() || out.run_dir.empty()) {
        std::cerr << "Usage: talbot_tournament --config_file <yaml> --model_a <path> --model_b <path> --run_dir <dir>\n";
        return false;
    }
    return true;
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
    const int initial_sims = (cfg.mode == TournamentConfig::Mode::TIMED)
        ? (int)(cfg.time_control.nps_ewma * 5.0)
        : (int)cfg.selector.gumbel_search_depth;
    const PoolTargets initial = MCTSEngine::predict_pool_needs_static(initial_sims, cfg.pool_sizing);

    auto engine = std::make_unique<MCTSEngine>(
        (int)initial.node_target,
        (int)initial.edge_target,
        cfg.selector.batch_size_per_worker,
        inf_queue,
        result_queue,
        worker_id,
        cfg.selector.deficit_eps,
        cfg.selector.virtual_loss,
        cfg.selector.contempt,
        cfg.selector.draw_cutoff,
        cfg.selector.gumbel_c_visit,
        cfg.selector.gumbel_c_scale,
        cfg.selector.gumbel_noise,
        dummy_board,
        std::vector<chess::Board>(),
        logger,
        buf.input, buf.policy, buf.value,
        free_slots,
        core_wait_count,
        cfg.workers_per_core,
        cfg.selector.two_fold_repetition,
        tb_ready
    );

    engine->pool_sizing_cfg              = cfg.pool_sizing;
    engine->set_nps_alpha(cfg.time_control.nps_ewma_alpha);
    engine->reset_nps_history(cfg.time_control.nps_ewma);
    engine->early_stop_q_gap             = cfg.early_stop_q_gap;
    engine->early_stop_min_visits        = cfg.early_stop_min_visits;
    engine->early_return_on_forced_win   = cfg.early_return_on_forced_win;
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
        cfg = load_config(args.config_file);
    } catch (const std::exception& e) {
        std::cerr << "Fatal: failed to load config " << args.config_file
                  << ": " << e.what() << std::endl;
        return 1;
    }

    // Pairing dir + logging ---------------------------------------------------
    const std::string name_a = fs::path(args.model_a_path).stem().string();
    const std::string name_b = fs::path(args.model_b_path).stem().string();
    const std::string pairing_dir = args.run_dir + "/" + name_a + "_vs_" + name_b;
    fs::create_directories(pairing_dir);

    Logger main_logger("tournament_main", pairing_dir, cfg.main_logging_level);
    main_logger.rotate(0, 0);
    main_logger.log("INFO", "===== TOURNAMENT: " + name_a + " vs " + name_b + " =====");
    main_logger.log("INFO", "Config: " + args.config_file);
    main_logger.log("INFO", "Pairing dir: " + pairing_dir);
    main_logger.log("INFO", std::string("Mode: ") +
                    (cfg.mode == TournamentConfig::Mode::TIMED ? "timed" : "fixed"));
    main_logger.log("INFO", "Workers: " + std::to_string(cfg.num_workers) +
                    " (" + std::to_string(cfg.game_worker_cores.size()) +
                    " cores x " + std::to_string(cfg.workers_per_core) + " per core)");

    // Pin main
    if (!cfg.main_cores.empty()) {
        DWORD_PTR m = 0;
        for (int c : cfg.main_cores) if (c >= 0 && c < 64) m |= (DWORD_PTR{1} << c);
        if (m) SetThreadAffinityMask(GetCurrentThread(), m);
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
    const int num_openings = cfg.games_per_match / 2;
    if ((int)book.size() < num_openings) {
        main_logger.log("WARNING", "Requested " + std::to_string(num_openings) +
                        " openings but book only has " + std::to_string(book.size()) +
                        "; playing " + std::to_string(book.size() * 2) + " games instead.");
    }
    auto openings = book.sample(num_openings);

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
    // Downstream Elo/rating script reads this file.
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
    // Shells out to talbot_trt_compile.exe if the .engine file isn't already
    // cached alongside the .onnx. Cached hits are cheap (one fs::exists call).
    const std::string engine_a = ensure_engine(
        args.model_a_path, cfg.max_batch_size, cfg.input_planes, main_logger);
    const std::string engine_b = ensure_engine(
        args.model_b_path, cfg.max_batch_size, cfg.input_planes, main_logger);
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

    // TimeControl (shared, stateless; only used in timed mode) ---------------
    std::unique_ptr<TimeControl> time_ctrl;
    if (cfg.mode == TournamentConfig::Mode::TIMED) {
        time_ctrl = std::make_unique<TimeControl>(cfg.time_control);
    }

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

        ws->engine_a = build_engine(cfg, inf_queue_a, result_queues_a[w], w,
                                    buf_a, free_slots_a, wait_count,
                                    *ws->logger, tb_ready, dummy_board);
        ws->engine_b = build_engine(cfg, inf_queue_b, result_queues_b[w], w,
                                    buf_b, free_slots_b, wait_count,
                                    *ws->logger, tb_ready, dummy_board);

        ws->selector_a = std::make_unique<ActionSelector>(
            "selector_a_w" + std::to_string(w), w, cfg.selector, *ws->logger);
        ws->selector_b = std::make_unique<ActionSelector>(
            "selector_b_w" + std::to_string(w), w, cfg.selector, *ws->logger);

        workers.push_back(std::move(ws));
    }

    // Distribute specs to workers (stride) -----------------------------------
    std::vector<std::vector<GameSpec>> worker_specs(cfg.num_workers);
    for (int i = 0; i < (int)specs.size(); ++i) {
        worker_specs[i % cfg.num_workers].push_back(specs[i]);
    }

    // Shared counters for the simple summary ---------------------------------
    // W/D/L from Model A's perspective. That's all we track here -- Elo is a
    // separate downstream concern that reads games.pgn.
    std::atomic<int> a_wins{0}, b_wins{0}, draws{0}, total_plies{0}, games_completed{0};

    // Launch workers ---------------------------------------------------------
    std::vector<std::thread> worker_threads;
    worker_threads.reserve(cfg.num_workers);
    const auto tstart = std::chrono::steady_clock::now();

    for (int w = 0; w < cfg.num_workers; ++w) {
        worker_threads.emplace_back([&, w]() {
            const int core = cfg.game_worker_cores[w / cfg.workers_per_core];
            if (core >= 0 && core < 64) {
                SetThreadAffinityMask(GetCurrentThread(), DWORD_PTR{1} << core);
            }
            at::set_num_threads(1);

            WorkerState& ws = *workers[w];

            SearchAgent agent_a{
                *ws.engine_a, *ws.selector_a,
                (int)cfg.selector.gumbel_search_depth,
                (int)cfg.selector.gumbel_m
            };
            SearchAgent agent_b{
                *ws.engine_b, *ws.selector_b,
                (int)cfg.selector.gumbel_search_depth,
                (int)cfg.selector.gumbel_m
            };

            for (const GameSpec& spec : worker_specs[w]) {
                const chess::Color a_color = spec.model_a_color;
                SearchAgent white_agent = (a_color == chess::Color::WHITE) ? agent_a : agent_b;
                SearchAgent black_agent = (a_color == chess::Color::WHITE) ? agent_b : agent_a;
                const chess::Color our_side = a_color;

                std::optional<TimedGameSetup> timed_setup;
                if (cfg.mode == TournamentConfig::Mode::TIMED) {
                    timed_setup = TimedGameSetup{
                        time_ctrl.get(),
                        cfg.initial_time_ms,
                        cfg.increment_ms
                    };
                }

                // Per-game PGN metadata: player names depend on which colour
                // Model A plays this game; everything else is constant across
                // the pairing. Config is annotated -- tournament PGNs get full
                // CCRL-style headers and per-move comments.
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
                    *ws.logger, timed_setup, pgn_sink, pgn_meta);

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
    const std::string summary_path = pairing_dir + "/summary.txt";
    std::ofstream sf(summary_path);
    if (sf) {
        sf << "model_a=" << name_a << "\n";
        sf << "model_b=" << name_b << "\n";
        sf << "mode="    << (cfg.mode == TournamentConfig::Mode::TIMED ? "timed" : "fixed") << "\n";
        sf << "games="   << n     << "\n";
        sf << "a_wins="  << wa    << "\n";
        sf << "draws="   << dr    << "\n";
        sf << "b_wins="  << wb    << "\n";
        sf << "avg_plies=" << avg_plies << "\n";
        sf << "wall_seconds=" << (int)wall_s << "\n";
        sf << "pgn_file=" << pgn_path << "\n";
    }

    // Shutdown ---------------------------------------------------------------
    stop_event.store(true);
    if (batcher_a_thread.joinable()) batcher_a_thread.join();
    if (batcher_b_thread.joinable()) batcher_b_thread.join();

    if (tb_ready) tb_free();

    return 0;
}