// =============================================================================
// main_uci.cpp
//
// Entry point for talbot.exe -- a bare, single-game UCI engine.
//
// UCI clients (chess GUIs, cutechess-cli, tournament tools) launch engines
// with NO arguments and speak UCI on stdin/stdout. There is deliberately no
// CLI here and no env-var overrides. The exe requires three files sitting
// next to it:
//
//   play_uci.yaml   -- all runtime config (paths derived, no external file deps)
//   model.yaml      -- architecture spec (input_planes / board_dim / policy_moves).
//                      Shipped alongside the .engine so dims can't drift.
//   model.engine    -- TensorRT-compiled network (name is hardcoded)
//
// Missing any of the three is fatal at startup.
//
// GUI-tweakable settings are exposed as UCI options (SyzygyPath, MoveOverhead,
// etc.) advertised in the `uci` handshake and applied via `setoption`. Some
// options (Threads, Ponder, UCI_AnalyseMode, Clear Hash, SyzygyProbeLimit)
// are advertised for GUI compatibility but accepted-and-ignored.
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
#include <stdexcept>
#include <chrono>
#include <iomanip>
#include <memory>
#include <algorithm>
#include <mutex>
#include <condition_variable>
#include <cstdlib>

#include "concurrentqueue.h"
#include <torch/torch.h>
#include <yaml-cpp/yaml.h>

#include "chess.hpp"
#include "mcts_engine.hpp"
#include "action_selector.hpp"
#include "inference_batcher.hpp"
#include "logger.hpp"
#include "time_control.hpp"
#include "tbprobe.h"   // Fathom Syzygy probing (tb_init / tb_free / TB_LARGEST)

namespace fs = std::filesystem;

static std::atomic<bool> global_stop_event{false};

// ---- stdout serialisation ---------------------------------------------------
// UCI output comes from three places once we go async: main thread (uciok /
// readyok / echoes), search worker (bestmove), info thread (info lines).
// Each writer takes stdout_mtx around its complete line to prevent torn
// output that would confuse GUI parsers.
static std::mutex stdout_mtx;

static inline void uci_out(const std::string& line) {
    std::lock_guard<std::mutex> lock(stdout_mtx);
    std::cout << line << std::endl;
}

// ---- info-line helpers ------------------------------------------------------
// q -> centipawns via logit map. This is standard for NN engines with WDL
// heads: cp is monotone in q, saturates smoothly near +-1, and roughly
// matches the scale GUIs expect for cp scores. Clamped to a sane range.
static int q_to_cp(double q) {
    // The literal constant 1.5620688421 in the formula is tan⁻¹(10) / 90 * 180/π — it's calibrated so that q = 1.0 (nominal win) maps to cp ≈ +900, 
    // roughly matching Stockfish's "clearly winning but not mate" range. Sensible default.
    const double max_q = 1.569 / 1.5620688421;
    q = std::clamp(q, -max_q, max_q);
    double cp = 90.0 * std::tan(q * 1.5620688421);
    
    return static_cast<int>(std::round(cp));
}

// Root argmax under Gumbel score. Under the edge/node split, iteration is
// over MCTSEdge; the child pointer may be nullptr for unvisited edges (which
// score as v_mix). Returns the winning edge (never nullptr unless the root
// has no children).
static MCTSEdge* get_best_root_edge(MCTSNode* root, MCTSEngine* engine, double contempt) {
    if (!root || root->num_children == 0) return nullptr;

    double max_visits = 0.0;
    for (int i = 0; i < root->num_children; ++i) {
        MCTSNode* c = (root->first_edge + i)->child;
        if (c && c->visits > max_visits) max_visits = c->visits;
    }
    const double v_mix = root->calculate_v_mix(contempt);

    MCTSEdge* best = nullptr;
    double best_score = -1e20;
    for (int i = 0; i < root->num_children; ++i) {
        MCTSEdge* e = root->first_edge + i;
        double noise = (i < (int)engine->root_gumbel_noise.size())
                     ? engine->root_gumbel_noise[i] : 0.0;
        double score = e->calculate_gumbel_score(
            contempt, engine->gumbel_c_visit, engine->gumbel_c_scale,
            max_visits, v_mix, noise);
        if (score > best_score) {
            best_score = score;
            best = e;
        }
    }
    return best;
}

static double best_child_q(MCTSNode* root, MCTSEngine* engine, double contempt) {
    MCTSEdge* best = get_best_root_edge(root, engine, contempt);
    if (!best || !best->child || best->child->visits <= 0) return std::nan("");
    return -best->child->expected_value(contempt);
}

static std::vector<std::string> extract_pv(MCTSNode* root, MCTSEngine* engine, double contempt, int max_depth = 32) {
    std::vector<std::string> pv;
    if (!root) return pv;

    // 1. Root move: pick actual best move (Gumbel Score), not just max visits.
    MCTSEdge* root_best = get_best_root_edge(root, engine, contempt);
    if (!root_best || !root_best->child || root_best->child->visits <= 0) return pv;

    pv.push_back(chess::uci::moveToUci(root_best->move));
    MCTSNode* node = root_best->child;

    // 2. Rest of the tree: walk by max visits. Unvisited edges are skipped
    //    (nullptr child means the sim never descended there).
    while (node && node->is_expanded() && node->num_children > 0 && (int)pv.size() < max_depth) {
        MCTSEdge* next_best = nullptr;
        int max_v = 0;
        for (int i = 0; i < node->num_children; ++i) {
            MCTSEdge* e = node->first_edge + i;
            if (!e->child) continue;
            if (e->child->visits > max_v) {
                max_v = e->child->visits;
                next_best = e;
            }
        }
        if (!next_best || max_v <= 0) break;
        pv.push_back(chess::uci::moveToUci(next_best->move));
        node = next_best->child;
    }
    return pv;
}

// Locate the root edge corresponding to a specific move. Returns nullptr if
// no matching edge is found (shouldn't happen for a move selected from the
// current root, but guarded for safety).
static MCTSEdge* find_root_edge(MCTSNode* root, chess::Move move) {
    if (!root) return nullptr;
    for (int i = 0; i < root->num_children; ++i) {
        MCTSEdge* e = root->first_edge + i;
        if (e->move == move) return e;
    }
    return nullptr;
}

// PV rooted at a specific edge (the move about to be played). Guarantees
// pv[0] == that move's UCI string. Used for the FINAL info line so the PV
// and the following bestmove line agree, which is what UCI GUIs expect.
static std::vector<std::string> extract_pv_from(MCTSEdge* root_edge, int max_depth = 32) {
    std::vector<std::string> pv;
    if (!root_edge) return pv;

    pv.push_back(chess::uci::moveToUci(root_edge->move));

    // Subtree walk: identical to extract_pv's continuation logic. If the
    // played edge has no materialised child (e.g. picked from a mover-has-
    // draw candidate with zero visits), we return just the move itself.
    MCTSNode* node = root_edge->child;
    while (node && node->is_expanded() && node->num_children > 0 && (int)pv.size() < max_depth) {
        MCTSEdge* next_best = nullptr;
        int max_v = 0;
        for (int i = 0; i < node->num_children; ++i) {
            MCTSEdge* e = node->first_edge + i;
            if (!e->child) continue;
            if (e->child->visits > max_v) {
                max_v = e->child->visits;
                next_best = e;
            }
        }
        if (!next_best || max_v <= 0) break;
        pv.push_back(chess::uci::moveToUci(next_best->move));
        node = next_best->child;
    }
    return pv;
}

// -----------------------------------------------------------------------------
static std::vector<std::string> split(const std::string& s, char delimiter) {
    std::vector<std::string> tokens;
    std::string token;
    std::istringstream tokenStream(s);
    while (std::getline(tokenStream, token, delimiter)) {
        if (!token.empty()) tokens.push_back(token);
    }
    return tokens;
}

static DWORD_PTR mask_from_cores(const std::vector<int>& cores) {
    DWORD_PTR m = 0;
    for (int c : cores) {
        if (c >= 0 && c < (int)(sizeof(DWORD_PTR) * 8)) m |= (DWORD_PTR{1} << c);
    }
    return m;
}

// =============================================================================
// CONFIG STRUCT (single source of truth for all UCI runtime knobs)
// =============================================================================
struct UciConfig {
    // paths
    std::string engine_path;
    std::string base_log_dir;

    // logging
    int main_logging_level;

    // cores
    std::vector<int> main_cores;
    std::vector<int> game_worker_cores;
    std::vector<int> inference_worker_cores;

    // inference batcher
    int inference_batch_size;
    int max_batch_size;
    int batch_timeout_ms;
    int batcher_logging_level;
    int logging_interval_sec;

    // model dims (loaded from model.yaml sitting next to the exe)
    int input_planes;
    int board_dim;
    int policy_moves;

    // Split search config. Loaded together via load_configs() so their shared
    // contempt/draw_cutoff fields cannot drift.
    //   mcts     -- fed to MCTSEngine (construction, pool sizing, sim budget)
    //   selector -- fed to ActionSelector (rules A-F in select_move)
    MctsConfig           mcts;
    ActionSelectorConfig selector;

    // time control
    TimeControlConfig time_control;

    // pool sizing (fed into MCTSEngine::pool_sizing_cfg post-construction)
    PoolSizingConfig pool_sizing;

    // tablebase
    bool        tablebase_enabled = false;
    std::string tablebase_path;

    // Inference-only search behaviour: bail out of gumbel phases when the
    // top-two candidates differ in raw Q by at least this much. 0 disables.
    double early_stop_q_gap = 0.0;
    int early_stop_min_visits = 0;
    bool early_return_on_forced_win;
};

// The exe_dir is passed in so we can set engine_path (model.engine sits next
// to the exe by contract; there is no yaml override).
static UciConfig load_config(const std::string& config_file_path,
                             const std::string& exe_dir) {
    UciConfig cfg;

    YAML::Node root    = YAML::LoadFile(config_file_path);
    YAML::Node engine  = root["engine"];        // renamed from "global"
    YAML::Node cores_n = root["core"];
    YAML::Node infer_n = root["inference"];
    YAML::Node mcts_n  = root["mcts"];
    YAML::Node sel_n   = root["selection"];

    if (!engine) throw std::runtime_error("play_uci.yaml missing 'engine:' block");

    cfg.engine_path        = exe_dir + "/model.engine";
    cfg.base_log_dir       = engine["log_dir"] ? engine["log_dir"].as<std::string>() : std::string();
    cfg.main_logging_level = engine["log_level"] ? engine["log_level"].as<int>() : 20;

    if (cores_n["main"])
        for (const auto& c : cores_n["main"]) cfg.main_cores.push_back(c.as<int>());
        
    if (cores_n["game_worker"])
        for (const auto& c : cores_n["game_worker"]) cfg.game_worker_cores.push_back(c.as<int>());

    if (cores_n["inference_worker"])
        for (const auto& c : cores_n["inference_worker"]) cfg.inference_worker_cores.push_back(c.as<int>());

    cfg.inference_batch_size  = infer_n["batch_size"].as<int>();
    cfg.max_batch_size        = cfg.inference_batch_size * infer_n["batch_size_factor"].as<int>();
    cfg.batch_timeout_ms      = infer_n["batch_timeout_ms"].as<int>();
    cfg.batcher_logging_level = infer_n["logging_level"].as<int>();
    cfg.logging_interval_sec  = infer_n["logging_interval_sec"].as<int>();

    // Model dims come from model.yaml sitting next to the exe -- single
    // source of truth with the trained artifact. play_uci.yaml no longer
    // duplicates these, so no drift risk.
    const std::string model_yaml_path = exe_dir + "/model.yaml";
    YAML::Node model = YAML::LoadFile(model_yaml_path);
    cfg.input_planes = model["model"]["input_planes"].as<int>();
    cfg.board_dim    = model["model"]["board_dim"].as<int>();
    cfg.policy_moves = model["model"]["total_policy_moves"].as<int>();

    // Tablebase: no yaml block. TB probing is off at startup; the GUI enables
    // it by sending `setoption name SyzygyPath value <path>`.

    // Shared loader with the training binary (see action_selector.cpp). Adding
    // a new mcts or selection knob means editing load_configs() only -- both
    // binaries pick it up. Pre-refactor this block was a hand-maintained copy
    // of the one in data_generator.cpp; they had already drifted on .as<int>
    // vs .as<double> for the gumbel_* fields.
    LoadedConfigs loaded = load_configs(mcts_n, sel_n, /*require_gumbel_m=*/false);
    cfg.mcts     = loaded.mcts;
    cfg.selector = loaded.selector;

    // Optional: inference early-stop threshold. Missing = disabled (0.0).
    if (mcts_n["early_stop_q_gap"]) {
        cfg.early_stop_q_gap = mcts_n["early_stop_q_gap"].as<double>();
    }

    if (mcts_n["early_stop_min_visits"]) {
        cfg.early_stop_min_visits = mcts_n["early_stop_min_visits"].as<int>();
    }

    if (mcts_n["early_return_on_forced_win"]) {
        cfg.early_return_on_forced_win = mcts_n["early_return_on_forced_win"].as<bool>();
    }


    // time_control is REQUIRED for UCI (clock-based search is the whole point).
    YAML::Node tc_n = root["time_control"];
    if (!tc_n) throw std::runtime_error("play_uci.yaml is missing time_control block");
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

    // pool_sizing is REQUIRED -- MCTSEngine has no fallback. Same 5 knobs as
    // train.yaml but caps typically larger since UCI runs multi-second searches.
    YAML::Node pool_n = root["pool_sizing"];
    if (!pool_n) throw std::runtime_error("play_uci.yaml is missing pool_sizing block");
    cfg.pool_sizing.avg_branching       = pool_n["avg_branching"].as<double>();
    cfg.pool_sizing.node_safety_factor  = pool_n["node_safety_factor"].as<double>();
    cfg.pool_sizing.edge_safety_factor  = pool_n["edge_safety_factor"].as<double>();
    cfg.pool_sizing.node_hard_cap_bytes = (size_t)pool_n["node_hard_cap_mb"].as<size_t>() * 1024ull * 1024ull;
    cfg.pool_sizing.edge_hard_cap_bytes = (size_t)pool_n["edge_hard_cap_mb"].as<size_t>() * 1024ull * 1024ull;

    return cfg;
}

// =============================================================================
// SEARCH WORKER
//
// The worker thread owns the ENTIRE lifecycle of a search:
//   1. wait for start_flag
//   2. mcts.reset(board, history, node_target, edge_target)   // stage 3: grows pools
//   3. spawn info sub-thread (peeks tree ~1Hz -> "info ..." lines)
//   4. run_simulations_fixed / _timed
//   5. join info sub-thread
//   6. agent.select_move + emit "bestmove ..."
//   7. flip done_flag, notify main
//
// This means main thread never blocks on cv_done -- it fires `go`, sets
// start_flag, returns to stdin polling. That's what lets `stop` (or `quit`)
// actually arrive during a search.
// =============================================================================
struct SearchWorker {
    std::thread thread;
    std::mutex  mtx;
    std::condition_variable cv_start;
    std::condition_variable cv_done;
    bool start_flag = false;
    bool quit_flag  = false;
    bool done_flag  = true;

    // Per-search inputs (set by main under mtx before notifying cv_start).
    chess::Board board;
    std::vector<chess::Board> history;
    int search_nodes = 0;
    bool timed       = false;
    int ply_count    = 1;
    std::chrono::steady_clock::time_point soft_deadline;
    std::chrono::steady_clock::time_point hard_deadline;

    // Tree reuse plan. If reuse_moves is non-empty, worker walks it from
    // reuse_start_board / reuse_start_history calling reset_reuse for each
    // move. If any call fails, worker falls back to full reset(board, history).
    // Main leaves reuse_moves empty to force a full reset (ucinewgame, first
    // search of a session, FEN change, opponent-played-outside-top-m case).
    std::vector<chess::Move> reuse_moves;
    chess::Board reuse_start_board;
    std::vector<chess::Board> reuse_start_history;

    // Shared references (bound once by main after construction).
    MCTSEngine*     mcts  = nullptr;
    ActionSelector* agent = nullptr;
    Logger*         logger = nullptr;
    double          contempt = 0.0;    // for score-cp conversion

    // Info-thread coordination. search_active is true from the moment main
    // signals cv_start until the worker has emitted bestmove. The info
    // sub-thread loops while true.
    std::atomic<bool> search_active{false};

    DWORD_PTR core_mask = 0;
};

// Builds the shared CPU half-precision buffers used by a batcher + its engines.
struct SharedBuffers {
    std::vector<torch::Tensor> input;
    std::vector<torch::Tensor> policy;
    std::vector<torch::Tensor> value;
};

static SharedBuffers make_shared_buffers(const UciConfig& cfg) {
    SharedBuffers b;
    auto opts = torch::TensorOptions().dtype(torch::kHalf).device(torch::kCPU);
    for (int i = 0; i < cfg.max_batch_size; ++i) {
        b.input.push_back(torch::zeros({cfg.input_planes, cfg.board_dim, cfg.board_dim}, opts));
        b.policy.push_back(torch::zeros({cfg.policy_moves}, opts));
        b.value.push_back(torch::zeros({3}, opts));
        b.value.push_back(torch::zeros({1}, opts));
    }
    return b;
}

// =============================================================================
// ROOT SYZYGY PROBE
//
// If the position is within the loaded tables and has no castling rights,
// probe DTZ at the root and return the TB-optimal move -- the one that
// preserves the result AND makes progress toward mate under the 50-move rule.
// This is what actually *converts* won endings (KBN-v-K etc.); the in-tree WDL
// probe only scores positions exactly, it cannot order winning moves by
// progress, so on its own it shuffles.
// =============================================================================
static bool probe_root_tablebase(const chess::Board& board,
                                 chess::Move& out,
                                 unsigned& wdl_out,
                                 unsigned& dtz_out) {
    using chess::PieceType;
    using chess::Color;

    const chess::Bitboard wp = board.pieces(PieceType::PAWN,   Color::WHITE);
    const chess::Bitboard wn = board.pieces(PieceType::KNIGHT, Color::WHITE);
    const chess::Bitboard wb = board.pieces(PieceType::BISHOP, Color::WHITE);
    const chess::Bitboard wr = board.pieces(PieceType::ROOK,   Color::WHITE);
    const chess::Bitboard wq = board.pieces(PieceType::QUEEN,  Color::WHITE);
    const chess::Bitboard wk = board.pieces(PieceType::KING,   Color::WHITE);

    const chess::Bitboard bp = board.pieces(PieceType::PAWN,   Color::BLACK);
    const chess::Bitboard bn = board.pieces(PieceType::KNIGHT, Color::BLACK);
    const chess::Bitboard bb = board.pieces(PieceType::BISHOP, Color::BLACK);
    const chess::Bitboard br = board.pieces(PieceType::ROOK,   Color::BLACK);
    const chess::Bitboard bq = board.pieces(PieceType::QUEEN,  Color::BLACK);
    const chess::Bitboard bk = board.pieces(PieceType::KING,   Color::BLACK);

    const chess::Bitboard white_bb = wp | wn | wb | wr | wq | wk;
    const chess::Bitboard black_bb = bp | bn | bb | br | bq | bk;

    if ((white_bb | black_bb).count() > (int)TB_LARGEST) return false;

    const auto& cr = board.castlingRights();
    const bool any_castle =
        cr.has(Color::WHITE, chess::Board::CastlingRights::Side::KING_SIDE)  ||
        cr.has(Color::WHITE, chess::Board::CastlingRights::Side::QUEEN_SIDE) ||
        cr.has(Color::BLACK, chess::Board::CastlingRights::Side::KING_SIDE)  ||
        cr.has(Color::BLACK, chess::Board::CastlingRights::Side::QUEEN_SIDE);
    if (any_castle) return false;

    const chess::Square ep_sq = board.enpassantSq();
    const unsigned ep = (ep_sq == chess::Square::NO_SQ) ? 0u : (unsigned)ep_sq.index();
    const unsigned rule50        = (unsigned)board.halfMoveClock();
    const bool     white_to_move = (board.sideToMove() == Color::WHITE);

    const unsigned res = tb_probe_root(
        white_bb.getBits(), black_bb.getBits(),
        (wk | bk).getBits(), (wq | bq).getBits(), (wr | br).getBits(),
        (wb | bb).getBits(), (wn | bn).getBits(), (wp | bp).getBits(),
        rule50, /*castling=*/0u, ep, white_to_move,
        nullptr /*results array not needed*/);

    if (res == TB_RESULT_FAILED || res == TB_RESULT_CHECKMATE || res == TB_RESULT_STALEMATE)
        return false;

    wdl_out = TB_GET_WDL(res);
    dtz_out = TB_GET_DTZ(res);

    // Decode TB move: from-square, to-square, promotion piece. The library
    // returns fields as raw ints; the mapping to chess::Move is exact match
    // among the legal move list (we don't fabricate a Move from thin air).
    const unsigned from = TB_GET_FROM(res);
    const unsigned to   = TB_GET_TO(res);
    const unsigned promo= TB_GET_PROMOTES(res);

    chess::Movelist moves;
    chess::movegen::legalmoves(moves, board);
    for (const chess::Move& m : moves) {
        if ((unsigned)m.from().index() != from) continue;
        if ((unsigned)m.to().index()   != to)   continue;
        if (m.typeOf() == chess::Move::PROMOTION) {
            // Fathom promo encoding: 1=Q, 2=R, 3=B, 4=N; anything else -> mismatch.
            chess::PieceType want;
            switch (promo) {
                case 1: want = PieceType::QUEEN;  break;
                case 2: want = PieceType::ROOK;   break;
                case 3: want = PieceType::BISHOP; break;
                case 4: want = PieceType::KNIGHT; break;
                default: continue;
            }
            if (m.promotionType() != want) continue;
        } else if (promo != 0) {
            continue;   // TB says promotion but move type isn't
        }
        out = m;
        return true;
    }
    return false;   // decoded move not found among legal moves -> run search
}

// =============================================================================
// CONFIG RESOLUTION
//
// UCI GUIs launch the exe with zero args. The contract is: play_uci.yaml
// and model.engine both sit next to talbot.exe. No overrides, no env vars,
// no CLI flags. Missing either -> fatal at startup with a clear message.
// =============================================================================
static std::string get_exe_dir() {
    char buf[MAX_PATH];
    DWORD n = GetModuleFileNameA(nullptr, buf, MAX_PATH);
    if (n == 0 || n == MAX_PATH) return "";      // shouldn't happen on Windows
    return fs::path(std::string(buf, n)).parent_path().string();
}

// =============================================================================
int main(int argc, char* argv[]) {
    // A GUI launches us with no args and expects UCI on stdio. Do not add flags.

    const std::string exe_dir = get_exe_dir();
    if (exe_dir.empty()) {
        std::cerr << "Fatal: could not determine exe directory (GetModuleFileName failed).\n";
        return 1;
    }

    const std::string config_file_path = exe_dir + "/play_uci.yaml";
    if (!fs::exists(config_file_path)) {
        std::cerr << "Fatal: play_uci.yaml not found at " << config_file_path << "\n"
                  << "It must sit in the same directory as talbot.exe.\n";
        return 1;
    }

    const std::string model_yaml_path = exe_dir + "/model.yaml";
    if (!fs::exists(model_yaml_path)) {
        std::cerr << "Fatal: model.yaml not found at " << model_yaml_path << "\n"
                  << "It must sit in the same directory as talbot.exe.\n";
        return 1;
    }

    UciConfig cfg;
    try {
        cfg = load_config(config_file_path, exe_dir);
    } catch (const std::exception& e) {
        std::cerr << "Fatal: failed to load config (" << config_file_path
                  << "): " << e.what() << std::endl;
        return 1;
    }

    // CLI Override: --cores <comma_separated_ints>
    // Legacy single-set override: applied to both main + game_worker cores.
    // Batcher unchanged (stays on config value).
    //
    // Newer per-role overrides for concurrent-instance affinity pinning:
    //   --worker-cores <csv>   -> main_cores and game_worker_cores
    //   --batcher-cores <csv>  -> inference_worker_cores
    // Both/either can be passed; per-role overrides win over --cores if given.
    auto parse_cores = [](const char* csv, std::vector<int>& out) -> bool {
        std::stringstream ss(csv);
        std::string token;
        while (std::getline(ss, token, ',')) {
            try { out.push_back(std::stoi(token)); }
            catch (...) {
                std::cerr << "Fatal: Invalid core integer: " << token << "\n";
                return false;
            }
        }
        return true;
    };

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--cores" && i + 1 < argc) {
            std::vector<int> override_cores;
            if (!parse_cores(argv[++i], override_cores)) return 1;
            if (!override_cores.empty()) {
                cfg.main_cores = override_cores;
                cfg.game_worker_cores = override_cores;
            }
        }
        else if (arg == "--worker-cores" && i + 1 < argc) {
            std::vector<int> override_cores;
            if (!parse_cores(argv[++i], override_cores)) return 1;
            if (!override_cores.empty()) {
                cfg.main_cores = override_cores;
                cfg.game_worker_cores = override_cores;
            }
        }
        else if (arg == "--batcher-cores" && i + 1 < argc) {
            std::vector<int> override_cores;
            if (!parse_cores(argv[++i], override_cores)) return 1;
            if (!override_cores.empty()) {
                cfg.inference_worker_cores = override_cores;
            }
        }
    }

    if (!fs::exists(cfg.engine_path)) {
        std::cerr << "Fatal: model.engine not found at " << cfg.engine_path << "\n"
                  << "It must sit in the same directory as talbot.exe.\n";
        return 1;
    }

    // ---- logging (optional) -------------------------------------------------
    // If engine.log_dir was empty in the yaml, base_log_dir is empty; we skip
    // creating any directories and let Logger's empty-dir path make every
    // log() call a no-op. This keeps UCI silent by default -- what tournament
    // runners expect from a shipped engine.
    std::string run_log_dir;
    if (!cfg.base_log_dir.empty()) {
        auto now = std::chrono::system_clock::now();
        std::time_t now_time = std::chrono::system_clock::to_time_t(now);
        std::tm* lt = std::localtime(&now_time);
        std::ostringstream time_oss;
        time_oss << std::put_time(lt, "%Y-%m-%d_%H-%M-%S") << "_pid" << GetCurrentProcessId();
        run_log_dir = cfg.base_log_dir + "/" + time_oss.str();
        fs::create_directories(run_log_dir);
    }

    Logger main_logger("uci_main", run_log_dir, cfg.main_logging_level);
    main_logger.rotate(0, 0);
    main_logger.log("INFO", "Booting Talbot UCI Engine...");
    main_logger.log("INFO", "Config: " + config_file_path);
    main_logger.log("INFO", "Engine: " + cfg.engine_path);

    if (!cfg.main_cores.empty()) {
        DWORD_PTR m = mask_from_cores(cfg.main_cores);
        if (m != 0) SetThreadAffinityMask(GetCurrentThread(), m);
    }

    {
        auto join_ints = [](const std::vector<int>& v) {
            std::string s;
            for (size_t i = 0; i < v.size(); ++i) {
                if (i) s += ",";
                s += std::to_string(v[i]);
            }
            return s.empty() ? "<unset>" : s;
        };
        main_logger.log("INFO", "Affinity: main_cores=[" + join_ints(cfg.main_cores)
            + "] game_worker_cores=[" + join_ints(cfg.game_worker_cores)
            + "] inference_worker_cores=[" + join_ints(cfg.inference_worker_cores) + "]");
    }

    // (engine_path existence already validated pre-logger.)

    // Tablebase is disabled at startup. The GUI enables it by sending
    // `setoption name SyzygyPath value <path>`, which triggers tb_init in
    // the setoption handler below.
    bool tb_ready = false;

    // ---- inference plumbing -------------------------------------------------
    SharedBuffers buf = make_shared_buffers(cfg);

    moodycamel::ConcurrentQueue<std::pair<int, int>> inference_queue;
    std::vector<ThreadSafeQueue<std::vector<int>>> result_queues(1);
    ThreadSafeQueue<int> buffer_free_slots;
    for (int i = 0; i < cfg.max_batch_size; ++i) buffer_free_slots.push(i);

    std::atomic<uint64_t> dummy_step{0};

    InferenceBatcher batcher(
        cfg.engine_path, cfg.inference_batch_size, cfg.batch_timeout_ms, 1,
        run_log_dir, cfg.batcher_logging_level, cfg.inference_worker_cores,
        0, dummy_step, cfg.logging_interval_sec);

    std::thread batcher_thread([&]() {
        batcher.run(inference_queue, result_queues,
                    buf.input, buf.policy, buf.value,
                    global_stop_event, &buffer_free_slots);
    });

    // ---- board state (owned by main thread; the search worker gets snapshots)
    chess::Board board;
    board.setFen(chess::constants::STARTPOS);
    std::vector<chess::Board> history;

    // ---- Tree reuse tracking. All updated during position/ucinewgame; read
    // (and prior_* updated) in the go handler when computing the delta.
    // - current_base_fen: "" for startpos, else the FEN string. Used to detect
    //   base-position changes that force a full reset.
    // - current_moves: full move sequence from the current position command.
    // - prior_*: mirror of the state at the last search-triggering go. Only
    //   valid when has_prior_tree is true.
    // We only attempt reuse when the current base matches the prior base and
    // current_moves is a strict extension of prior_moves by 1-4 plies.
    std::string current_base_fen;
    std::vector<chess::Move> current_moves;

    std::string prior_base_fen;
    std::vector<chess::Move> prior_moves;
    chess::Board prior_board;
    std::vector<chess::Board> prior_history;
    bool has_prior_tree = false;

    std::atomic<int> wait_count{0};

    // Prime pool sizing for engine construction. We size the initial pool
    // for a ~5-second first search at the default NPS -- large enough that
    // the very first reset() doesn't need to grow. Subsequent resets grow
    // as needed based on the true measured NPS.
    PoolTargets initial = MCTSEngine::predict_pool_needs_static(
        static_cast<int>(cfg.time_control.nps_ewma * 5.0), cfg.pool_sizing);

    auto mcts_engine = std::make_unique<MCTSEngine>(
        cfg.mcts,
        static_cast<int>(initial.node_target),
        static_cast<int>(initial.edge_target),
        inference_queue, result_queues[0], 0,
        board, history, main_logger,
        buf.input, buf.policy, buf.value,
        buffer_free_slots, &wait_count, 1, tb_ready);

    mcts_engine->pool_sizing_cfg = cfg.pool_sizing;
    mcts_engine->set_nps_alpha(cfg.time_control.nps_ewma_alpha);
    // Prime NPS at construction (not just ucinewgame) so predict_pool_needs_for_time
    // returns a real number on the very first search too.
    mcts_engine->reset_nps_history(cfg.time_control.nps_ewma);
    mcts_engine->early_stop_q_gap = cfg.early_stop_q_gap;
    mcts_engine->early_stop_min_visits = cfg.early_stop_min_visits;
    mcts_engine->early_return_on_forced_win = cfg.early_return_on_forced_win;
    // TimeControl is reconstructed per-go so setoption MoveOverhead changes
    // take effect on the next search.

    // ---- search worker thread ----------------------------------------------
    // Agent must exist before we start the thread since the worker holds a
    // pointer to it (used inside the search-completion select_move call).
    ActionSelector agent("uci_agent", 0, cfg.selector, main_logger);
    int ply_count = 1;

    SearchWorker search_worker;
    search_worker.mcts     = mcts_engine.get();
    search_worker.agent    = &agent;
    search_worker.logger   = &main_logger;
    // contempt is shared -- either cfg.mcts.contempt or cfg.selector.contempt
    // works since load_configs guarantees they're equal. Pick selector since
    // search_worker.contempt feeds ActionSelector.select_move (via q_to_cp /
    // best_child_q) at the end of the search.
    search_worker.contempt = cfg.selector.contempt;
    search_worker.core_mask = mask_from_cores(cfg.game_worker_cores);

    search_worker.thread = std::thread([worker = &search_worker]() {
        if (worker->core_mask != 0)
            SetThreadAffinityMask(GetCurrentThread(), worker->core_mask);

        while (true) {
            std::unique_lock<std::mutex> lock(worker->mtx);
            worker->cv_start.wait(lock, [&]{ return worker->start_flag || worker->quit_flag; });
            if (worker->quit_flag) break;

            // Snapshot inputs so the info sub-thread doesn't have to lock.
            chess::Board board = worker->board;
            std::vector<chess::Board> history = worker->history;
            const bool timed = worker->timed;
            const int  search_nodes = worker->search_nodes;
            const int  ply_snapshot = worker->ply_count;
            const auto soft_dl = worker->soft_deadline;
            const auto hard_dl = worker->hard_deadline;
            std::vector<chess::Move> reuse_moves = worker->reuse_moves;
            chess::Board reuse_start_board = worker->reuse_start_board;
            std::vector<chess::Board> reuse_start_history = worker->reuse_start_history;
            lock.unlock();

            // Predict pool needs BEFORE reset(). For the timed path, we size
            // for (hard_dl - now) at the current EWMA NPS. hard_dl already
            // incorporates time_control.hard_multiplier, so safety_multiplier
            // here is 1.0. For the fixed path (opening plies / go nodes / no
            // clock), sim count is known exactly.
            PoolTargets pt;
            if (timed) {
                double time_s = std::chrono::duration<double>(
                    hard_dl - std::chrono::steady_clock::now()).count();
                pt = worker->mcts->predict_pool_needs_for_time(time_s, 1.0);
            } else {
                pt = worker->mcts->predict_pool_needs(search_nodes);
            }

            // ---- Tree reuse or full reset -----------------------------------
            // If main provided a reuse plan, walk it via reset_reuse (one
            // call per delta ply, reconstructing intermediate boards). Any
            // failure -> fall back to a full reset() from the final position.
            bool reused = false;
            if (!reuse_moves.empty()) {
                chess::Board walk_board = reuse_start_board;
                std::vector<chess::Board> walk_history = reuse_start_history;
                bool all_ok = true;
                for (chess::Move mv : reuse_moves) {
                    chess::Board next_board = walk_board;
                    next_board.makeMove(mv);
                    std::vector<chess::Board> next_history = walk_history;
                    next_history.insert(next_history.begin(), walk_board);
                    if (next_history.size() > 4) next_history.pop_back();

                    if (!worker->mcts->reset_reuse(next_board, next_history, mv,
                                                   pt.node_target, pt.edge_target)) {
                        all_ok = false;
                        break;
                    }
                    walk_board   = next_board;
                    walk_history = next_history;
                }
                reused = all_ok;
                if (!reused) {
                    worker->logger->log("INFO",
                        "Tree reuse aborted mid-walk; falling back to full reset.");
                }
            }
            if (!reused) {
                worker->mcts->reset(board, history, pt.node_target, pt.edge_target);
            }

            // ---- info emission (sub-thread) --------------------------------
            // Ticks ~1Hz; polls the search_active flag every 50ms so it exits
            // promptly when the search finishes. Reads on engine.root are
            // unlocked -- torn stats produce at worst a wobbly info line.
            worker->search_active.store(true, std::memory_order_relaxed);
            const auto search_start = std::chrono::steady_clock::now();

            std::thread info_thread([worker, search_start]() {
                const auto emit_interval = std::chrono::milliseconds(1000);
                const auto poll_interval = std::chrono::milliseconds(50);
                auto next_emit = std::chrono::steady_clock::now() + emit_interval;

                while (worker->search_active.load(std::memory_order_relaxed)) {
                    std::this_thread::sleep_for(poll_interval);
                    auto now = std::chrono::steady_clock::now();
                    if (now < next_emit) continue;
                    next_emit = now + emit_interval;

                    MCTSNode* root = worker->mcts->root;
                    if (!root || root->visits == 0) continue;

                    std::string score_str;
                    if (root->has_forced_outcome() && root->forced_outcome != 0) {
                        int moves_to_mate = (root->distance_to_mate + 1) / 2;
                        if (root->forced_outcome == -1) moves_to_mate = -moves_to_mate;
                        score_str = "score mate " + std::to_string(moves_to_mate);
                    } else {
                        double q = best_child_q(root, worker->mcts, worker->contempt);
                        if (std::isnan(q)) continue;
                        score_str = "score cp " + std::to_string(q_to_cp(q));
                    }

                    int nodes = worker->mcts->simulation_count;
                    long long elapsed_ms =
                        std::chrono::duration_cast<std::chrono::milliseconds>(now - search_start).count();
                    long long nps = elapsed_ms > 0 ? (nodes * 1000LL / elapsed_ms) : 0;

                    std::vector<std::string> pv = extract_pv(root, worker->mcts, worker->contempt);
                    int depth    = worker->mcts->max_selection_depth;
                    int seldepth = std::max(depth, (int)pv.size());

                    std::ostringstream oss;
                    oss << "info depth " << depth
                        << " seldepth " << seldepth
                        << " nodes " << nodes
                        << " nps " << nps
                        << " time " << elapsed_ms
                        << " " << score_str
                        << " pv";
                    for (const auto& m : pv) oss << " " << m;
                    uci_out(oss.str());
                }
            });
            // -----------------------------------------------------------------

            if (timed) {
                worker->mcts->run_simulations_timed_inference( soft_dl);
            } else {
                worker->mcts->run_simulations_fixed_inference(search_nodes);
            }

            // Signal info thread to exit and join it before touching root
            // for the final select_move.
            worker->search_active.store(false, std::memory_order_relaxed);
            if (info_thread.joinable()) info_thread.join();

            // Emit final info line + bestmove.
            {
                MCTSNode* root = worker->mcts->root;
                if (root && root->visits > 0) {
                    long long elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                        std::chrono::steady_clock::now() - search_start).count();
                    long long nps = elapsed_ms > 0 ? (worker->mcts->simulation_count * 1000LL / elapsed_ms) : 0;
                    
                    std::string score_str;
                    bool skip_emit = false;

                    if (root->has_forced_outcome() && root->forced_outcome != 0) {
                        int moves_to_mate = (root->distance_to_mate + 1) / 2;
                        if (root->forced_outcome == -1) moves_to_mate = -moves_to_mate;
                        score_str = "score mate " + std::to_string(moves_to_mate);
                    } else {
                        double q = best_child_q(root, worker->mcts, worker->contempt);
                        if (!std::isnan(q)) {
                            score_str = "score cp " + std::to_string(q_to_cp(q));
                        } else {
                            skip_emit = true;
                        }
                    }

                    if (!skip_emit) {
                        std::vector<std::string> pv = extract_pv(root, worker->mcts, worker->contempt);
                        int depth    = worker->mcts->max_selection_depth;
                        int seldepth = std::max(depth, (int)pv.size());
                        std::ostringstream oss;
                        oss << "info depth " << depth
                            << " seldepth " << seldepth
                            << " nodes " << worker->mcts->simulation_count
                            << " nps " << nps
                            << " time " << elapsed_ms
                            << " " << score_str
                            << " pv";
                        for (const auto& m : pv) oss << " " << m;
                        uci_out(oss.str());
                    }
                }
            }

            SelectionResult result = worker->agent->select_move(worker->mcts->root, ply_snapshot, worker->mcts);
            std::string best_move_str = (result.best_move == chess::Move::NO_MOVE)
                ? "0000" : chess::uci::moveToUci(result.best_move);
            uci_out("bestmove " + best_move_str);
            worker->logger->log("DEBUG", "Engine -> GUI: bestmove " + best_move_str);

            {
                std::lock_guard<std::mutex> l2(worker->mtx);
                worker->start_flag = false;
                worker->done_flag  = true;
            }
            worker->cv_done.notify_one();
        }
    });

    // ---- UCI command loop --------------------------------------------------
    std::string line;
    while (std::getline(std::cin, line)) {
        line.erase(std::remove(line.begin(), line.end(), '\r'), line.end());
        main_logger.log("DEBUG", "GUI -> Engine: " + line);
        std::vector<std::string> tokens = split(line, ' ');
        if (tokens.empty()) continue;
        const std::string& command = tokens[0];

        if (command == "uci") {
            uci_out("id name Talbot UCI");
            uci_out("id author Talbot Dev");

            // Options -- advertised to GUIs so they show up in settings UI.
            // Applied on setoption:
            //   SyzygyPath     -- reinits tablebase probing at the new path.
            //   MoveOverhead   -- mutates cfg.time_control for subsequent gos.
            // Accepted but ignored (kept so GUIs don't complain and so we
            // reserve the names for later):
            //   Threads, Ponder, UCI_AnalyseMode, SyzygyProbeLimit, Clear Hash
            uci_out("option name Threads type spin default 1 min 1 max 1");
            uci_out("option name Ponder type check default false");
            uci_out("option name MoveOverhead type spin default " +
                    std::to_string(cfg.time_control.move_overhead_ms) + " min 0 max 5000");
            uci_out("option name SyzygyPath type string default " +
                    (cfg.tablebase_path.empty() ? std::string("<empty>") : cfg.tablebase_path));
            uci_out("option name SyzygyProbeLimit type spin default 7 min 0 max 7");
            uci_out("option name UCI_AnalyseMode type check default false");
            uci_out("option name Clear Hash type button");
            uci_out("uciok");
        }
        else if (command == "isready") {
            // Return readyok unconditionally -- it's a heartbeat, not a
            // "wait for search to complete" gate.
            uci_out("readyok");
        }
        else if (command == "ucinewgame") {
            board.setFen(chess::constants::STARTPOS);
            history.clear();
            agent.reset_for_new_game();
            mcts_engine->reset_nps_history(cfg.time_control.nps_ewma);
            ply_count = 1;

            // Fresh game -> no tree to reuse across the boundary.
            has_prior_tree = false;
            prior_moves.clear();
            prior_history.clear();
            prior_base_fen.clear();
            current_moves.clear();
            current_base_fen.clear();
        }
        else if (command == "position") {
            history.clear();
            current_moves.clear();
            size_t moves_idx = tokens.size();
            for (size_t i = 1; i < tokens.size(); ++i) {
                if (tokens[i] == "moves") { moves_idx = i; break; }
            }
            if (tokens.size() > 1 && tokens[1] == "startpos") {
                board.setFen(chess::constants::STARTPOS);
                ply_count = 1;
                current_base_fen = "";  // sentinel for startpos
            } else if (tokens.size() > 2 && tokens[1] == "fen") {
                std::string fen;
                for (size_t i = 2; i < moves_idx; ++i)
                    fen += tokens[i] + (i == moves_idx - 1 ? "" : " ");
                board.setFen(fen);
                ply_count = ((board.fullMoveNumber() - 1) * 2) + (board.sideToMove() == chess::Color::BLACK ? 2 : 1);
                current_base_fen = fen;
            }

            for (size_t i = moves_idx + 1; i < tokens.size(); ++i) {
                history.insert(history.begin(), board);
                if (history.size() > 4) history.pop_back();
                chess::Move move = chess::uci::uciToMove(board, tokens[i]);
                current_moves.push_back(move);
                board.makeMove(move);
                ply_count++;
            }
        }
        else if (command == "go") {
            // Root tablebase probe: if the position is inside the loaded tables,
            // play the DTZ-optimal move immediately and skip the search entirely
            // -- this is the path that converts won endings.
            if (tb_ready) {
                chess::Move tb_move;
                unsigned tb_wdl = 0, tb_dtz = 0;
                if (probe_root_tablebase(board, tb_move, tb_wdl, tb_dtz)) {
                    static const char* WDL_NAME[5] =
                        {"loss", "blessed-loss", "draw", "cursed-win", "win"};
                    const char* wdl_str = WDL_NAME[tb_wdl <= 4 ? tb_wdl : 2];
                    const std::string mv = chess::uci::moveToUci(tb_move);
                    main_logger.log("INFO", "Root TB hit: " + mv + " wdl=" + wdl_str +
                                    " dtz=" + std::to_string(tb_dtz) + " -- skipping search.");
                    uci_out("info string syzygy " + std::string(wdl_str) + " dtz " + std::to_string(tb_dtz));
                    uci_out("bestmove " + mv);
                    continue;
                }
            }

            // Parse the subset of UCI `go` params we support.
            int64_t wtime = -1, btime = -1, winc = 0, binc = 0, movetime = -1;
            int movestogo = 0;
            int total_search_nodes = cfg.mcts.gumbel_search_depth;
            for (size_t i = 1; i + 1 < tokens.size(); ++i) {
                if      (tokens[i] == "nodes")     total_search_nodes = std::stoi(tokens[i + 1]);
                else if (tokens[i] == "wtime")     wtime     = std::stoll(tokens[i + 1]);
                else if (tokens[i] == "btime")     btime     = std::stoll(tokens[i + 1]);
                else if (tokens[i] == "winc")      winc      = std::stoll(tokens[i + 1]);
                else if (tokens[i] == "binc")      binc      = std::stoll(tokens[i + 1]);
                else if (tokens[i] == "movestogo") movestogo = std::stoi(tokens[i + 1]);
                else if (tokens[i] == "movetime")  movetime  = std::stoll(tokens[i + 1]);
            }

            bool timed = false;
            std::chrono::steady_clock::time_point soft_dl, hard_dl;
            auto now = std::chrono::steady_clock::now();

            if (ply_count <= 2) {
                // Use fixed nodes at the very start of the game -- NPS estimate
                // is stale from the previous game (or zero) and the trainer's
                // budget is a reasonable proxy.
                timed = false;
                total_search_nodes = cfg.mcts.gumbel_search_depth;
                main_logger.log("INFO", "Opening ply " + std::to_string(ply_count) + " detected. Forcing fixed-node search.");
            } else if (movetime >= 0) {
                // Fixed per-move time: spend (almost) all of it, no soft target.
                timed = true;
                int64_t budget = std::max<int64_t>(1, movetime - cfg.time_control.move_overhead_ms);
                soft_dl = now + std::chrono::milliseconds(budget);
                hard_dl = soft_dl;
            } else if (wtime >= 0 || btime >= 0) {
                // Clock-based: allocate from our side's clock via TimeControl.
                // Constructed fresh each go so setoption MoveOverhead changes
                // take effect on the very next search rather than being frozen
                // at exe-launch.
                bool white = (board.sideToMove() == chess::Color::WHITE);
                ClockState cs;
                cs.time_left_ms = white ? wtime : btime;
                cs.increment_ms = white ? winc  : binc;
                cs.moves_to_go  = movestogo;
                cs.ply          = ply_count;
                TimeControl tc(cfg.time_control);
                TimeBudget tb = tc.allocate(cs);
                timed = true;
                soft_dl = now + std::chrono::milliseconds(tb.target_ms);
                hard_dl = now + std::chrono::milliseconds(tb.hard_limit_ms);
                main_logger.log("INFO", "Time alloc: target=" + std::to_string(tb.target_ms) +
                                "ms hard=" + std::to_string(tb.hard_limit_ms) + "ms");
            }
            // else: no clock, no movetime -> fixed node budget (`go`, `go nodes`, `go infinite`).

            // Defensive: if a previous search is somehow still running (a
            // compliant GUI should have waited for bestmove first), request
            // stop and wait so we don't overwrite the worker's inputs.
            {
                std::unique_lock<std::mutex> lock(search_worker.mtx);
                if (!search_worker.done_flag) {
                    lock.unlock();
                    mcts_engine->request_stop();
                    std::unique_lock<std::mutex> lock2(search_worker.mtx);
                    search_worker.cv_done.wait(lock2, [&]{ return search_worker.done_flag; });
                }
            }

            // Clear the stop flag now (BEFORE signalling start). Doing this
            // inside run_simulations_* would race: a `stop` between our
            // notify_one and the worker's clear would be lost.
            mcts_engine->clear_stop();

            {
                std::lock_guard<std::mutex> lock(search_worker.mtx);
                search_worker.board     = board;
                search_worker.history   = history;
                search_worker.timed     = timed;
                search_worker.ply_count = ply_count;
                if (timed) {
                    search_worker.soft_deadline = soft_dl;
                    search_worker.hard_deadline = hard_dl;
                } else {
                    search_worker.search_nodes = total_search_nodes;
                }

                // ---- Tree reuse plan --------------------------------------
                // Reuse is possible iff we have a prior tree, the current
                // base position matches the prior base, and current_moves is
                // a strict extension of prior_moves by 1-4 plies. The 4-ply
                // cap is defensive: very long descents cost more than a
                // fresh reset by the time you've walked that many levels,
                // and it also indicates something unusual happened (GUI
                // resynced from a saved game etc).
                search_worker.reuse_moves.clear();
                if (has_prior_tree
                    && current_base_fen == prior_base_fen
                    && current_moves.size() > prior_moves.size()
                    && (current_moves.size() - prior_moves.size()) <= 4
                    && std::equal(prior_moves.begin(), prior_moves.end(),
                                  current_moves.begin())) {
                    search_worker.reuse_start_board   = prior_board;
                    search_worker.reuse_start_history = prior_history;
                    for (size_t i = prior_moves.size(); i < current_moves.size(); ++i) {
                        search_worker.reuse_moves.push_back(current_moves[i]);
                    }
                }

                search_worker.done_flag  = false;
                search_worker.start_flag = true;
            }

            // Save state as the new prior for the NEXT go. We store what we
            // are about to search from; the worker's reuse plan (built above)
            // uses the OLD prior_* which we've now overwritten -- correct
            // because the worker already snapshot reuse_* into the struct.
            prior_base_fen = current_base_fen;
            prior_moves    = current_moves;
            prior_board    = board;
            prior_history  = history;
            has_prior_tree = true;

            search_worker.cv_start.notify_one();
            // NB: no wait. Main returns to stdin loop so `stop` / `isready` /
            // `quit` can be received during the search. bestmove is emitted
            // by the worker thread itself when the search finishes.
        }
        else if (command == "stop") {
            // Signal the search to abandon at its next check (~128 sims).
            // Worker will emit bestmove from whatever tree state it has and
            // flip done_flag. Idempotent; no-op if no search is running.
            mcts_engine->request_stop();
        }
        else if (command == "quit") {
            // Interrupt any in-flight search so we can shut down promptly.
            mcts_engine->request_stop();
            {
                std::unique_lock<std::mutex> lock(search_worker.mtx);
                search_worker.cv_done.wait(lock, [&]{ return search_worker.done_flag; });
            }
            break;
        }
        else if (command == "setoption") {
            // Grammar: setoption name <NAME words> [value <VALUE words>]
            // NAME can contain spaces (e.g. "Clear Hash"); VALUE too (paths).
            if (tokens.size() < 3 || tokens[1] != "name") {
                main_logger.log("WARNING", "Malformed setoption: " + line);
                continue;
            }
            size_t value_idx = tokens.size();
            for (size_t i = 2; i < tokens.size(); ++i) {
                if (tokens[i] == "value") { value_idx = i; break; }
            }
            std::string opt_name;
            for (size_t i = 2; i < value_idx; ++i) {
                if (i > 2) opt_name += " ";
                opt_name += tokens[i];
            }
            std::string opt_value;
            for (size_t i = value_idx + 1; i < tokens.size(); ++i) {
                if (i > value_idx + 1) opt_value += " ";
                opt_value += tokens[i];
            }

            if (opt_value.size() >= 2 &&
                ((opt_value.front() == '\'' && opt_value.back() == '\'') ||
                (opt_value.front() == '"'  && opt_value.back() == '"'))) {
                opt_value = opt_value.substr(1, opt_value.size() - 2);
            }

            std::string key = opt_name;
            std::transform(key.begin(), key.end(), key.begin(),
                           [](unsigned char c){ return static_cast<char>(std::tolower(c)); });

            if (key == "syzygypath") {
                // Reinit TB probing at the new path. Empty path disables.
                // Any in-flight search would race with tb_free; UCI protocol
                // says setoption comes between searches, so we don't defend.
                if (tb_ready) { tb_free(); tb_ready = false; }
                cfg.tablebase_path    = opt_value;
                cfg.tablebase_enabled = !opt_value.empty();
                if (cfg.tablebase_enabled && opt_value != "<empty>") {
                    if (tb_init(opt_value.c_str())) {
                        tb_ready = (TB_LARGEST > 0);
                        main_logger.log("INFO", "SyzygyPath -> " + opt_value +
                                        " (TB_LARGEST=" + std::to_string(TB_LARGEST) + ")");
                    } else {
                        main_logger.log("ERROR", "tb_init failed for " + opt_value);
                    }
                } else {
                    main_logger.log("INFO", "SyzygyPath cleared; TB probing disabled.");
                }
                // MCTSEngine captured use_tablebase at construction; push the
                // new value in so the in-tree probe actually fires (or stops).
                // Also affects the root-probe path via tb_ready directly.
                mcts_engine->use_tablebase = tb_ready;
            }
            else if (key == "moveoverhead") {
                try {
                    long long v = std::stoll(opt_value);
                    if (v < 0) v = 0;
                    if (v > 5000) v = 5000;
                    cfg.time_control.move_overhead_ms = v;
                    // TimeControl is reconstructed per-go via the cfg it holds,
                    // so this takes effect on the next `go`. If TimeControl
                    // ever caches the value, we'd need to mutate the instance
                    // here instead.
                    main_logger.log("INFO", "MoveOverhead -> " + std::to_string(v) + "ms");
                } catch (...) {
                    main_logger.log("WARNING", "MoveOverhead: bad value '" + opt_value + "'");
                }
            }
            else if (key == "threads" || key == "ponder" || key == "uci_analysemode" ||
                     key == "syzygyprobelimit" || key == "clear hash") {
                // Advertised for GUI compatibility; not applied.
                main_logger.log("INFO", "setoption '" + opt_name +
                                "' accepted (no effect in this build).");
            }
            else {
                main_logger.log("INFO", "setoption '" + opt_name + "' unknown -- ignored.");
            }
        }
        // Unknown commands (ponderhit, debug, etc.) silently ignored.
    }

    main_logger.log("INFO", "Quit received. Terminating worker...");
    {
        std::lock_guard<std::mutex> lock(search_worker.mtx);
        search_worker.quit_flag = true;
    }
    search_worker.cv_start.notify_one();
    if (search_worker.thread.joinable()) search_worker.thread.join();

    global_stop_event.store(true);
    if (batcher_thread.joinable()) batcher_thread.join();

    if (tb_ready) tb_free();
    return 0;
}