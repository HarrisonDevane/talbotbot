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
#include "puct_mcts.hpp"
#include "puct_action_selector.hpp"
#include "inference_batcher.hpp"
#include "logger.hpp"
#include "time_control.hpp"

#include "tbprobe.h"   // Fathom Syzygy probing (tb_init / tb_free / TB_LARGEST)

namespace fs = std::filesystem;

static std::atomic<bool> global_stop_event{false};

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
// =============================================================================

struct TreeParams {
    // Shared MCTS
    int node_pool_size;
    int batch_size_per_worker;
    double virtual_loss;
    double contempt;
    double policy_softmax_temp;
    bool two_fold_repetition;
    int default_nodes;

    // PUCT-only
    double cpuct;
    double temperature_visits;

    // Shared selection
    double draw_cutoff;
    double resignation_probability;
    double resignation_cutoff;
    int temperature_ply_cutoff;
};

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

    // flat params for MCTS & Action Selection
    TreeParams tree;

    // time control
    TimeControlConfig time_control;

    // tablebase
    bool        tablebase_enabled = false;
    std::string tablebase_path;
};

static UciConfig load_config(const std::string& config_file_path) {
    UciConfig cfg;

    YAML::Node root   = YAML::LoadFile(config_file_path);
    YAML::Node global = root["global"];
    YAML::Node eval_n = root["evaluation"];
    YAML::Node infer_n= root["inference"];
    YAML::Node mcts_n = root["mcts"];
    YAML::Node puct_n = root["puct"];
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

    TreeParams& s = cfg.tree;
    s.node_pool_size         = mcts_n["node_pool_size"].as<int>();
    s.batch_size_per_worker  = mcts_n["worker_minibatch_size"].as<int>();
    s.virtual_loss           = mcts_n["virtual_loss"].as<double>();
    s.contempt               = mcts_n["contempt"].as<double>();
    s.policy_softmax_temp   = mcts_n["policy_softmax_temp"].as<double>();
    s.two_fold_repetition    = mcts_n["two_fold_repetition"].as<bool>();

    s.cpuct                  = puct_n["cpuct"].as<double>();
    s.temperature_visits     = puct_n["temperature_visits"].as<double>();
    s.default_nodes          = puct_n["default_nodes"].as<int>();

    s.draw_cutoff            = sel_n["draw_cutoff"].as<double>();
    s.temperature_ply_cutoff = sel_n["temperature_ply_cutoff"].as<int>();
    s.resignation_probability= sel_n["resignation_probability"].as<double>();
    s.resignation_cutoff     = sel_n["resignation_cutoff"].as<double>();

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
// =============================================================================
struct SearchWorker {
    std::thread thread;
    std::mutex  mtx;
    std::condition_variable cv_start;
    std::condition_variable cv_done;
    bool start_flag = false;
    bool quit_flag  = false;
    bool done_flag  = true;
    chess::Board board;
    std::vector<chess::Board> history;
    int search_nodes = 0;
    bool timed       = false;
    std::chrono::steady_clock::time_point soft_deadline;
    std::chrono::steady_clock::time_point hard_deadline;
    PuctMCTS* mcts = nullptr;
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

    const unsigned from = TB_GET_FROM(res);
    const unsigned to   = TB_GET_TO(res);
    const unsigned promo= TB_GET_PROMOTES(res);

    chess::Movelist moves;
    chess::movegen::legalmoves(moves, board);
    for (const chess::Move& m : moves) {
        if ((unsigned)m.from().index() != from) continue;
        if ((unsigned)m.to().index()   != to)   continue;
        if (m.typeOf() == chess::Move::PROMOTION) {
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
            continue;   
        }
        out = m;
        return true;
    }
    return false;   
}

// =============================================================================
// CONFIG RESOLUTION
// =============================================================================
static std::string resolve_config_path() {
    if (const char* env = std::getenv("TALBOT_CONFIG")) {
        if (env[0] != '\0') return std::string(env);
    }
    char buf[MAX_PATH];
    DWORD n = GetModuleFileNameA(nullptr, buf, MAX_PATH);
    if (n == 0 || n == MAX_PATH) {
        return "play_uci.yaml";
    }
    return (fs::path(std::string(buf, n)).parent_path() / "play_uci.yaml").string();
}

// =============================================================================
int main(int /*argc*/, char* /*argv*/[]) {
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

    auto mcts_engine = std::make_unique<PuctMCTS>(
        cfg.tree.node_pool_size, cfg.tree.batch_size_per_worker,
        inference_queue, result_queues[0], 0,
        cfg.tree.virtual_loss, cfg.tree.contempt, cfg.tree.policy_softmax_temp, cfg.tree.cpuct, board, history, main_logger,
        buf.input, buf.policy, buf.value,
        buffer_free_slots, &wait_count, 1, cfg.tree.two_fold_repetition, tb_ready);

    mcts_engine->set_nps_alpha(cfg.time_control.nps_ewma_alpha);
    TimeControl time_control(cfg.time_control);

    // ---- search worker thread ----------------------------------------------
    SearchWorker search_worker;
    search_worker.mcts = mcts_engine.get();
    search_worker.core_mask = mask_from_cores(cfg.game_worker_cores);

    search_worker.thread = std::thread([worker = &search_worker]() {
        if (worker->core_mask != 0)
            SetThreadAffinityMask(GetCurrentThread(), worker->core_mask);
        while (true) {
            std::unique_lock<std::mutex> lock(worker->mtx);
            worker->cv_start.wait(lock, [&]{ return worker->start_flag || worker->quit_flag; });
            if (worker->quit_flag) break;
            worker->mcts->reset(worker->board, worker->history);
            if (worker->timed) {
                worker->mcts->run_simulations_timed(
                    worker->soft_deadline,
                    worker->hard_deadline);
            } else {
                worker->mcts->run_simulations_fixed(worker->search_nodes);
            }
            worker->start_flag = false;
            worker->done_flag  = true;
            lock.unlock();
            worker->cv_done.notify_one();
        }
    });

    PuctActionSelector::Config asel_cfg{
        { cfg.tree.contempt, cfg.tree.draw_cutoff, cfg.tree.resignation_probability,
          cfg.tree.resignation_cutoff, cfg.tree.temperature_ply_cutoff },
        cfg.tree.temperature_visits
    };
    PuctActionSelector agent("uci_agent", 0, asel_cfg, main_logger);
    int ply_count = 1;

    // ---- UCI command loop --------------------------------------------------
    std::string line;
    while (std::getline(std::cin, line)) {
        line.erase(std::remove(line.begin(), line.end(), '\r'), line.end());
        main_logger.log("DEBUG", "GUI -> Engine: " + line);
        std::vector<std::string> tokens = split(line, ' ');
        if (tokens.empty()) continue;
        const std::string& command = tokens[0];

        if (command == "uci") {
            std::cout << "id name Talbot UCI" << std::endl;
            std::cout << "id author Talbot Dev" << std::endl;
            std::cout << "uciok" << std::endl;
        }
        else if (command == "isready") {
            std::cout << "readyok" << std::endl;
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
                    std::cout << "info string syzygy " << wdl_str
                              << " dtz " << tb_dtz << std::endl;
                    std::cout << "bestmove " << mv << std::endl;
                    continue;
                }
            }

            int64_t wtime = -1, btime = -1, winc = 0, binc = 0, movetime = -1;
            int movestogo = 0;
            int total_search_nodes = cfg.tree.default_nodes;
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
                timed = false;
                total_search_nodes = cfg.tree.default_nodes;
                main_logger.log("INFO", "Opening ply " + std::to_string(ply_count) + " detected. Forcing fixed-node search.");
            } else if (movetime >= 0) {
                timed = true;
                int64_t budget = std::max<int64_t>(1, movetime - cfg.time_control.move_overhead_ms);
                soft_dl = now + std::chrono::milliseconds(budget);
                hard_dl = soft_dl;
            } else if (wtime >= 0 || btime >= 0) {
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

            {
                std::lock_guard<std::mutex> lock(search_worker.mtx);
                search_worker.board    = board;
                search_worker.history  = history;
                search_worker.timed    = timed;
                if (timed) {
                    search_worker.soft_deadline = soft_dl;
                    search_worker.hard_deadline = hard_dl;
                } else {
                    search_worker.search_nodes = total_search_nodes;
                }
                search_worker.done_flag  = false;
                search_worker.start_flag = true;
                search_worker.cv_start.notify_one();
            }
            {
                std::unique_lock<std::mutex> lock(search_worker.mtx);
                search_worker.cv_done.wait(lock, [&]{ return search_worker.done_flag; });
            }
            SelectionResult result = agent.select_move(mcts_engine->root, ply_count);
            std::string best_move_str =
                (result.best_move == chess::Move::NO_MOVE)
                    ? "0000" : chess::uci::moveToUci(result.best_move);
            std::cout << "bestmove " << best_move_str << std::endl;
            main_logger.log("DEBUG", "Engine -> GUI: bestmove " + best_move_str);
        }
        else if (command == "quit") {
            break;
        }
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