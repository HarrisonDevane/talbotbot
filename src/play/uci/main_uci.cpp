// =============================================================================
// main_uci.cpp
//
// Entry point for talbot.exe -- a bare, single-game UCI engine.
//
// UCI clients (chess GUIs, cutechess-cli, tournament tools) launch engines
// with NO arguments and speak UCI on stdin/stdout. There is deliberately no
// CLI here: the exe reads its config file from
//
//   1. $TALBOT_CONFIG, if set
//   2. play_uci.yaml sitting next to the exe
//
// in that order, and errors out otherwise.
//
// This file was lifted from the old talbot_play --uci path. Tournament code,
// tournament config fields, and the --uci / --tournament subcommand dispatcher
// live in talbot_tournament / talbot_lichess now, not here.
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
    const double EPS = 1e-6;
    if (q >  1.0 - EPS) q =  1.0 - EPS;
    if (q < -1.0 + EPS) q = -1.0 + EPS;
    double cp = 200.0 * std::log10((1.0 + q) / (1.0 - q));
    if (cp >  10000.0) cp =  10000.0;
    if (cp < -10000.0) cp = -10000.0;
    return static_cast<int>(std::round(cp));
}

// Walk from root following max-visits child. Stop at unexpanded/leaf.
// Reads are unlocked (visits/first_child raced against the search thread)
// -- torn reads produce at worst a wobbly PV in one frame, invisible to GUIs.
static std::vector<std::string> extract_pv(MCTSNode* root, int max_depth = 16) {
    std::vector<std::string> pv;
    MCTSNode* node = root;
    while (node && node->is_expanded() && node->num_children > 0 && (int)pv.size() < max_depth) {
        MCTSNode* best = nullptr;
        int best_visits = -1;
        for (int i = 0; i < node->num_children; ++i) {
            MCTSNode* c = node->first_child + i;
            if (c->visits > best_visits) { best_visits = c->visits; best = c; }
        }
        if (!best || best->visits <= 0) break;
        pv.push_back(chess::uci::moveToUci(best->move));
        node = best;
    }
    return pv;
}
// -----------------------------------------------------------------------------

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
    DWORD_PTR mask = 0;
    for (int c : cores) mask |= (static_cast<DWORD_PTR>(1) << c);
    return mask;
}

// =============================================================================
// CONFIG
//
// UCI-only. Fields that used to live in the shared PlayConfig but were only
// touched by the tournament path (opening_file, games_per_match, batcher_a/b
// cores, etc.) are gone. If we need one back later, add it here explicitly.
// =============================================================================
struct UciConfig {
    // paths
    std::string model_file_path;   // model.yaml -- dims live here
    std::string base_log_dir;
    std::string base_model_path;
    std::string engine_path;       // derived: base_model_path + ".engine"
    int  main_logging_level;

    // cores
    std::vector<int> main_cores;
    std::vector<int> game_worker_cores;         // where the search worker pins
    std::vector<int> inference_worker_cores;    // batcher's dispatcher/collector/filler

    // inference / batcher
    int inference_batch_size;
    int max_batch_size;
    int batch_timeout_ms;
    int batcher_logging_level;
    int logging_interval_sec;

    // model dims (loaded from model.yaml)
    int input_planes;
    int board_dim;
    int policy_moves;

    // mcts / selection (single struct; ActionSelector reads what it needs)
    ActionSelectorConfig selector;

    // time control
    TimeControlConfig time_control;

    // tablebase
    bool        tablebase_enabled = false;
    std::string tablebase_path;

    // Inference-only search behaviour: bail out of gumbel phases when the
    // top-two candidates differ in raw Q by at least this much. 0 disables.
    double early_stop_q_gap = 0.0;
    int early_stop_min_visits = 0;
};

static UciConfig load_config(const std::string& config_file_path) {
    UciConfig cfg;

    YAML::Node root   = YAML::LoadFile(config_file_path);
    YAML::Node global = root["global"];
    YAML::Node eval_n = root["evaluation"];
    YAML::Node infer_n= root["inference"];
    YAML::Node mcts_n = root["mcts"];
    YAML::Node sel_n  = root["selection"];

    cfg.model_file_path    = global["model_file"].as<std::string>();
    cfg.base_log_dir       = global["log_dir"].as<std::string>();
    cfg.main_logging_level = global["main_logging_level"].as<int>();
    cfg.base_model_path    = global["model_path"].as<std::string>();
    cfg.engine_path        = cfg.base_model_path + ".engine";

    if (eval_n && eval_n["main_cores"])
        for (const auto& c : eval_n["main_cores"]) cfg.main_cores.push_back(c.as<int>());
    if (eval_n && eval_n["game_worker_cores"])
        for (const auto& c : eval_n["game_worker_cores"]) cfg.game_worker_cores.push_back(c.as<int>());

    if (infer_n["inference_worker_cores"])
        for (const auto& c : infer_n["inference_worker_cores"]) cfg.inference_worker_cores.push_back(c.as<int>());

    cfg.inference_batch_size  = infer_n["batch_size"].as<int>();
    cfg.max_batch_size        = cfg.inference_batch_size * infer_n["batch_size_factor"].as<int>();
    cfg.batch_timeout_ms      = infer_n["batch_timeout_ms"].as<int>();
    cfg.batcher_logging_level = infer_n["logging_level"].as<int>();
    cfg.logging_interval_sec  = infer_n["logging_interval_sec"].as<int>();

    YAML::Node model = YAML::LoadFile(cfg.model_file_path);
    cfg.input_planes = model["model"]["input_planes"].as<int>();
    cfg.board_dim    = model["model"]["board_dim"].as<int>();
    cfg.policy_moves = model["model"]["total_policy_moves"].as<int>();

    if (YAML::Node tb_n = root["tablebase"]) {
        if (tb_n["enabled"] && tb_n["enabled"].as<bool>()) {
            if (!tb_n["path"]) {
                throw std::runtime_error(
                    "tablebase.enabled is true but tablebase.path is missing");
            }
            cfg.tablebase_enabled = true;
            cfg.tablebase_path    = tb_n["path"].as<std::string>();
        }
    }

    ActionSelectorConfig& s = cfg.selector;
    s.node_pool_size         = mcts_n["node_pool_size"].as<int>();
    s.virtual_loss           = mcts_n["virtual_loss"].as<double>();
    s.contempt               = mcts_n["contempt"].as<double>();
    s.deficit_eps                  = mcts_n["deficit_eps"].as<double>();
    s.two_fold_repetition    = mcts_n["two_fold_repetition"].as<bool>();
    s.draw_cutoff            = sel_n["draw_cutoff"].as<double>();
    s.gumbel_c_visit         = mcts_n["gumbel_c_visit"].as<double>();
    s.gumbel_c_scale         = mcts_n["gumbel_c_scale"].as<double>();
    s.gumbel_noise           = mcts_n["gumbel_noise"].as<double>();
    s.gumbel_search_depth    = mcts_n["gumbel_search_depth"].as<double>();
    s.gumbel_m               = mcts_n["gumbel_m"].as<double>();
    s.batch_size_per_worker  = mcts_n["worker_minibatch_size"].as<int>();
    s.temperature_ply_cutoff = sel_n["temperature_ply_cutoff"].as<int>();
    s.temperature_q_decay    = sel_n["temperature_q_decay"].as<double>();
    s.resignation_probability= sel_n["resignation_probability"].as<double>();
    s.resignation_cutoff     = sel_n["resignation_cutoff"].as<double>();

    // Optional: inference early-stop threshold. Missing = disabled (0.0).
    if (mcts_n["early_stop_q_gap"]) {
        cfg.early_stop_q_gap = mcts_n["early_stop_q_gap"].as<double>();
    }

    if (mcts_n["early_stop_min_visits"]) {
        cfg.early_stop_min_visits = mcts_n["early_stop_min_visits"].as<int>();
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

    return cfg;
}

// =============================================================================
// SEARCH WORKER
//
// The worker thread owns the ENTIRE lifecycle of a search:
//   1. wait for start_flag
//   2. mcts.reset(board, history)
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
    int gumbel_m     = 0;
    bool timed       = false;
    int ply_count    = 1;
    std::chrono::steady_clock::time_point soft_deadline;
    std::chrono::steady_clock::time_point hard_deadline;

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
// UCI GUIs launch the exe with zero args. That means the exe has to find its
// own config -- no relative "run from project root" trick like the trainer.
// Resolution order:
//   1. $TALBOT_CONFIG env var (dev / A-B testing convenience)
//   2. play_uci.yaml sitting next to talbot.exe
//   3. fatal error with a clear message
// =============================================================================
static std::string resolve_config_path() {
    if (const char* env = std::getenv("TALBOT_CONFIG")) {
        if (env[0] != '\0') return std::string(env);
    }
    char buf[MAX_PATH];
    DWORD n = GetModuleFileNameA(nullptr, buf, MAX_PATH);
    if (n == 0 || n == MAX_PATH) {
        // Shouldn't happen; fall back to bare filename which resolves against cwd.
        return "play_uci.yaml";
    }
    return (fs::path(std::string(buf, n)).parent_path() / "play_uci.yaml").string();
}

// =============================================================================
int main(int /*argc*/, char* /*argv*/[]) {
    // A GUI launches us with no args and expects UCI on stdio. Do not add flags.

    const std::string config_file_path = resolve_config_path();
    if (!fs::exists(config_file_path)) {
        std::cerr << "Fatal: config not found. Looked at:\n"
                  << "  1. $TALBOT_CONFIG env var (unset or empty)\n"
                  << "  2. " << config_file_path << "\n"
                  << "Place play_uci.yaml next to talbot.exe or set TALBOT_CONFIG.\n";
        return 1;
    }

    UciConfig cfg;
    try {
        cfg = load_config(config_file_path);
    } catch (const std::exception& e) {
        std::cerr << "Fatal: failed to load config (" << config_file_path
                  << "): " << e.what() << std::endl;
        return 1;
    }

    // ---- logging ------------------------------------------------------------
    auto now = std::chrono::system_clock::now();
    std::time_t now_time = std::chrono::system_clock::to_time_t(now);
    std::tm* lt = std::localtime(&now_time);
    std::ostringstream time_oss;
    time_oss << std::put_time(lt, "%Y-%m-%d_%H-%M-%S");
    std::string run_log_dir = cfg.base_log_dir + "/" + time_oss.str();
    fs::create_directories(run_log_dir);

    Logger main_logger("uci_main", run_log_dir, cfg.main_logging_level);
    main_logger.rotate(0, 0);
    main_logger.log("INFO", "Booting Talbot UCI Engine...");
    main_logger.log("INFO", "Config: " + config_file_path);

    if (!cfg.main_cores.empty()) {
        DWORD_PTR m = mask_from_cores(cfg.main_cores);
        if (m != 0) SetThreadAffinityMask(GetCurrentThread(), m);
    }

    if (!fs::exists(cfg.engine_path)) {
        main_logger.log("CRITICAL", "Engine file missing at " + cfg.engine_path);
        std::cerr << "Fatal: TRT engine missing at " << cfg.engine_path << std::endl;
        return 1;
    }

    // ---- Syzygy tablebases --------------------------------------------------
    // tb_init is the one non-thread-safe Fathom call; do it here at startup,
    // before the search worker thread can probe. tb_ready is what we hand to
    // the engine -- false if disabled OR if init found no tables.
    bool tb_ready = false;
    if (cfg.tablebase_enabled) {
        if (tb_init(cfg.tablebase_path.c_str())) {
            tb_ready = (TB_LARGEST > 0);
            main_logger.log("INFO", "Syzygy initialised from " + cfg.tablebase_path +
                            " (TB_LARGEST=" + std::to_string(TB_LARGEST) + ")");
            if (!tb_ready) {
                main_logger.log("WARNING",
                    "tb_init ok but TB_LARGEST=0 -- no tables at path. Probing disabled.");
            }
        } else {
            main_logger.log("ERROR",
                "tb_init failed for " + cfg.tablebase_path + " -- probing disabled.");
        }
    }

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

    std::atomic<int> wait_count{0};

    auto mcts_engine = std::make_unique<MCTSEngine>(
        cfg.selector.node_pool_size, cfg.selector.batch_size_per_worker,
        inference_queue, result_queues[0], 0,
        cfg.selector.deficit_eps, cfg.selector.virtual_loss, cfg.selector.contempt, cfg.selector.draw_cutoff,
        cfg.selector.gumbel_c_visit, cfg.selector.gumbel_c_scale,
        cfg.selector.gumbel_noise, board, history, main_logger,
        buf.input, buf.policy, buf.value,
        buffer_free_slots, &wait_count, 1, cfg.selector.two_fold_repetition, tb_ready);

    mcts_engine->set_nps_alpha(cfg.time_control.nps_ewma_alpha);
    mcts_engine->early_stop_q_gap = cfg.early_stop_q_gap;
    mcts_engine->early_stop_min_visits = cfg.early_stop_min_visits;
    TimeControl time_control(cfg.time_control);

    // ---- search worker thread ----------------------------------------------
    // Agent must exist before we start the thread since the worker holds a
    // pointer to it (used inside the search-completion select_move call).
    ActionSelector agent("uci_agent", 0, cfg.selector, main_logger);
    int ply_count = 1;

    SearchWorker search_worker;
    search_worker.mcts     = mcts_engine.get();
    search_worker.agent    = &agent;
    search_worker.logger   = &main_logger;
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
            const int  gumbel_m     = worker->gumbel_m;
            const int  search_nodes = worker->search_nodes;
            const int  ply_snapshot = worker->ply_count;
            const auto soft_dl = worker->soft_deadline;
            const auto hard_dl = worker->hard_deadline;
            lock.unlock();

            worker->mcts->reset(board, history);

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

                    int nodes = worker->mcts->simulation_count;
                    long long elapsed_ms =
                        std::chrono::duration_cast<std::chrono::milliseconds>(now - search_start).count();
                    long long nps = elapsed_ms > 0 ? (nodes * 1000LL / elapsed_ms) : 0;

                    // Root's expected_value is in own-mover perspective; that
                    // matches UCI's cp convention (positive = side-to-move
                    // stands better). No flip needed.
                    double q = root->expected_value(worker->contempt);
                    int cp = q_to_cp(q);

                    std::vector<std::string> pv = extract_pv(root);
                    std::ostringstream oss;
                    oss << "info depth " << pv.size()
                        << " nodes " << nodes
                        << " nps " << nps
                        << " time " << elapsed_ms
                        << " score cp " << cp
                        << " pv";
                    for (const auto& m : pv) oss << " " << m;
                    uci_out(oss.str());
                }
            });
            // -----------------------------------------------------------------

            if (timed) {
                worker->mcts->run_simulations_timed(gumbel_m, soft_dl, hard_dl);
            } else {
                worker->mcts->run_simulations_fixed(search_nodes, gumbel_m);
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
                    double q = root->expected_value(worker->contempt);
                    int cp = q_to_cp(q);
                    std::vector<std::string> pv = extract_pv(root);
                    std::ostringstream oss;
                    oss << "info depth " << pv.size()
                        << " nodes " << worker->mcts->simulation_count
                        << " nps " << nps
                        << " time " << elapsed_ms
                        << " score cp " << cp
                        << " pv";
                    for (const auto& m : pv) oss << " " << m;
                    uci_out(oss.str());
                }
            }

            SelectionResult result = worker->agent->select_move(worker->mcts->root, ply_snapshot);
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
        }
        else if (command == "position") {
            history.clear();
            size_t moves_idx = tokens.size();
            for (size_t i = 1; i < tokens.size(); ++i) {
                if (tokens[i] == "moves") { moves_idx = i; break; }
            }
            if (tokens.size() > 1 && tokens[1] == "startpos") {
                board.setFen(chess::constants::STARTPOS);
                ply_count = 1;
            } else if (tokens.size() > 2 && tokens[1] == "fen") {
                std::string fen;
                for (size_t i = 2; i < moves_idx; ++i)
                    fen += tokens[i] + (i == moves_idx - 1 ? "" : " ");
                board.setFen(fen);
                ply_count = ((board.fullMoveNumber() - 1) * 2) + (board.sideToMove() == chess::Color::BLACK ? 2 : 1);
            }

            for (size_t i = moves_idx + 1; i < tokens.size(); ++i) {
                history.insert(history.begin(), board);
                if (history.size() > 4) history.pop_back();
                chess::Move move = chess::uci::uciToMove(board, tokens[i]);
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
            int total_search_nodes = static_cast<int>(cfg.selector.gumbel_search_depth);
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
                total_search_nodes = static_cast<int>(cfg.selector.gumbel_search_depth);
                main_logger.log("INFO", "Opening ply " + std::to_string(ply_count) + " detected. Forcing fixed-node search.");
            } else if (movetime >= 0) {
                // Fixed per-move time: spend (almost) all of it, no soft target.
                timed = true;
                int64_t budget = std::max<int64_t>(1, movetime - cfg.time_control.move_overhead_ms);
                soft_dl = now + std::chrono::milliseconds(budget);
                hard_dl = soft_dl;
            } else if (wtime >= 0 || btime >= 0) {
                // Clock-based: allocate from our side's clock via TimeControl.
                bool white = (board.sideToMove() == chess::Color::WHITE);
                ClockState cs;
                cs.time_left_ms = white ? wtime : btime;
                cs.increment_ms = white ? winc  : binc;
                cs.moves_to_go  = movestogo;
                cs.ply          = ply_count;
                TimeBudget tb = time_control.allocate(cs);
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
                search_worker.gumbel_m  = static_cast<int>(cfg.selector.gumbel_m);
                search_worker.timed     = timed;
                search_worker.ply_count = ply_count;
                if (timed) {
                    search_worker.soft_deadline = soft_dl;
                    search_worker.hard_deadline = hard_dl;
                } else {
                    search_worker.search_nodes = total_search_nodes;
                }
                search_worker.done_flag  = false;
                search_worker.start_flag = true;
            }
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
        // Unknown commands (including "setoption", "ponderhit", "debug" for
        // now) are silently ignored -- to be filled in as we iterate.
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