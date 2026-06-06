// =============================================================================
// main_play.cpp
//
// Entry point for talbot_play.exe.
//
// Modes (selected by argv):
//   --uci          Single-game UCI engine for a GUI / lichess-bot. WORKING.
//   --tournament   Multi-game self-play match between TWO models. WORKING.
//                  This process plays exactly ONE pairing (model A vs model B)
//                  and exits. Round-robin enumeration / engine building / Elo
//                  fitting are the Python orchestrator's job, not this binary.
//
// CLI:
//   talbot_play --uci         --config_file <yaml>
//   talbot_play --tournament  --config_file <yaml>
//                             --model_a <A.engine> --model_b <B.engine>
//                             --run_dir <orchestrator-created run directory>
//
// Per-pairing model identity and the run directory are per-invocation, so
// --model_a / --model_b / --run_dir are accepted on the CLI. Everything
// structural (cores, openings, counts, search params) lives in the YAML.
// The orchestrator creates one timestamped run directory per tournament and
// passes it to every pairing; each pairing logs into <run_dir>/<A>_vs_<B>/
// and appends to the shared <run_dir>/results.csv.
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

#include "concurrentqueue.h"
#include <torch/torch.h>
#include <yaml-cpp/yaml.h>

#include "chess.hpp"
#include "mcts_engine.hpp"
#include "action_selector.hpp"
#include "inference_batcher.hpp"
#include "logger.hpp"
#include "game_session.hpp"
#include "game_worker.hpp"
#include "self_play_session.hpp"
#include "opening_book.hpp"
#include "trt_builder.hpp"

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
struct PlayConfig {
    // paths
    std::string model_file_path;
    std::string base_log_dir;
    std::string base_model_path;   // used by --uci
    std::string engine_path;       // used by --uci
    int  main_logging_level;

    // cores
    std::vector<int> main_cores;
    std::vector<int> game_worker_cores;
    std::vector<int> inference_worker_cores;   // --uci batcher
    std::vector<int> batcher_a_cores;          // --tournament batcher A
    std::vector<int> batcher_b_cores;          // --tournament batcher B

    // inference / batcher
    int inference_batch_size;
    int max_batch_size;
    int batch_timeout_ms;
    int batcher_logging_level;
    int logging_interval_sec;

    // model dims
    int input_planes;
    int board_dim;
    int policy_moves;

    // tournament
    std::string opening_file;
    int games_per_match;          // total games per pairing; must be even
    int num_openings;             // derived: games_per_match / 2
    int opening_seed;
    int workers_per_core;         // game-worker threads pinned per core
    int concurrent_games;         // derived: len(game_worker_cores) * workers_per_core
    int max_ply_length;
    int worker_logging_level;
    std::vector<int> tournament_worker_cores;  // game-worker cores (tournament section)

    // mcts / selection
    ActionSelectorConfig selector;
};

static PlayConfig load_config(const std::string& config_file_path) {
    PlayConfig cfg;

    YAML::Node root   = YAML::LoadFile(config_file_path);
    YAML::Node global = root["global"];
    YAML::Node eval_n = root["evaluation"];
    YAML::Node infer_n= root["inference"];
    YAML::Node mcts_n = root["mcts"];
    YAML::Node sel_n  = root["selection"];
    YAML::Node tour_n = root["tournament"];

    cfg.model_file_path    = global["model_file"].as<std::string>();
    cfg.base_log_dir       = global["log_dir"].as<std::string>();
    cfg.main_logging_level = global["main_logging_level"].as<int>();
    // model_path is only needed by --uci (to derive the engine path). The
    // tournament config legitimately omits it, so this read is guarded --
    // an unguarded .as<>() on a missing key throws "invalid node".
    if (global["model_path"]) {
        cfg.base_model_path = global["model_path"].as<std::string>();
        cfg.engine_path = cfg.base_model_path + ".engine";
    }

    if (eval_n && eval_n["main_cores"])
        for (const auto& c : eval_n["main_cores"]) cfg.main_cores.push_back(c.as<int>());
    if (eval_n && eval_n["game_worker_cores"])
        for (const auto& c : eval_n["game_worker_cores"]) cfg.game_worker_cores.push_back(c.as<int>());

    if (infer_n["inference_worker_cores"])
        for (const auto& c : infer_n["inference_worker_cores"]) cfg.inference_worker_cores.push_back(c.as<int>());
    if (infer_n["batcher_a_cores"])
        for (const auto& c : infer_n["batcher_a_cores"]) cfg.batcher_a_cores.push_back(c.as<int>());
    if (infer_n["batcher_b_cores"])
        for (const auto& c : infer_n["batcher_b_cores"]) cfg.batcher_b_cores.push_back(c.as<int>());

    cfg.inference_batch_size  = infer_n["batch_size"].as<int>();
    cfg.max_batch_size        = cfg.inference_batch_size * infer_n["batch_size_factor"].as<int>();
    cfg.batch_timeout_ms      = infer_n["batch_timeout_ms"].as<int>();
    cfg.batcher_logging_level = infer_n["logging_level"].as<int>();
    cfg.logging_interval_sec  = infer_n["logging_interval_sec"].as<int>();

    YAML::Node model = YAML::LoadFile(cfg.model_file_path);
    cfg.input_planes = model["chess"]["input_planes"].as<int>();
    cfg.board_dim    = model["chess"]["board_dim"].as<int>();
    cfg.policy_moves = model["chess"]["total_policy_moves"].as<int>();

    if (tour_n) {
        cfg.opening_file      = tour_n["opening_file"].as<std::string>();
        cfg.games_per_match   = tour_n["games_per_match"].as<int>();
        cfg.opening_seed      = tour_n["opening_seed"].as<int>();
        cfg.workers_per_core  = tour_n["workers_per_core"].as<int>();
        cfg.max_ply_length = tour_n["max_ply_length"].as<int>();
        cfg.worker_logging_level  = tour_n["worker_logging_level"].as<int>();

        if (cfg.games_per_match <= 0 || (cfg.games_per_match % 2) != 0) {
            throw std::runtime_error(
                "tournament.games_per_match must be a positive even number "
                "(each opening is played twice); got " +
                std::to_string(cfg.games_per_match));
        }
        cfg.num_openings = cfg.games_per_match / 2;

        // game-worker cores live in the tournament section (mirrors train.yaml's
        // data_generation.game_worker_cores).
        for (const auto& c : tour_n["game_worker_cores"])
            cfg.tournament_worker_cores.push_back(c.as<int>());
        if (cfg.tournament_worker_cores.empty()) {
            throw std::runtime_error("tournament.game_worker_cores is empty");
        }

        // concurrent_games = cores * workers_per_core (mirrors data_generator).
        cfg.concurrent_games =
            static_cast<int>(cfg.tournament_worker_cores.size()) *
            cfg.workers_per_core;
    }

    ActionSelectorConfig& s = cfg.selector;
    s.node_pool_size         = mcts_n["node_pool_size"].as<int>();
    s.virtual_loss           = mcts_n["virtual_loss"].as<double>();
    s.contempt               = mcts_n["contempt"].as<double>();
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

    return cfg;
}

// =============================================================================
// SEARCH WORKER (for --uci) -- kept from the original main_inference.cpp.
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
    int gumbel_m     = 0;
    MCTSEngine* mcts = nullptr;
    DWORD_PTR core_mask = 0;
};

// Builds the shared CPU half-precision buffers used by a batcher + its engines.
struct SharedBuffers {
    std::vector<torch::Tensor> input;
    std::vector<torch::Tensor> policy;
    std::vector<torch::Tensor> value;
};

static SharedBuffers make_shared_buffers(const PlayConfig& cfg) {
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
// UCI MODE
// =============================================================================
static int run_uci(const PlayConfig& cfg) {
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

    if (!cfg.main_cores.empty()) {
        DWORD_PTR m = mask_from_cores(cfg.main_cores);
        if (m != 0) SetThreadAffinityMask(GetCurrentThread(), m);
    }

    if (!fs::exists(cfg.engine_path)) {
        main_logger.log("CRITICAL", "Engine file missing at " + cfg.engine_path);
        std::cerr << "Fatal: TRT engine missing at " << cfg.engine_path << std::endl;
        return 1;
    }

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

    chess::Board board;
    board.setFen(chess::constants::STARTPOS);
    std::vector<chess::Board> history;

    std::atomic<int> wait_count{0};

    auto mcts_engine = std::make_unique<MCTSEngine>(
        cfg.selector.node_pool_size, cfg.selector.batch_size_per_worker,
        inference_queue, result_queues[0], 0,
        cfg.selector.virtual_loss, cfg.selector.contempt, cfg.selector.draw_cutoff,
        cfg.selector.gumbel_c_visit, cfg.selector.gumbel_c_scale,
        cfg.selector.gumbel_noise, board, history, main_logger,
        buf.input, buf.policy, buf.value,
        buffer_free_slots, &wait_count, 1);

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
            worker->mcts->run_simulations(worker->search_nodes, worker->gumbel_m);
            worker->start_flag = false;
            worker->done_flag  = true;
            lock.unlock();
            worker->cv_done.notify_one();
        }
    });

    ActionSelector agent("uci_agent", 0, cfg.selector, main_logger);
    int ply_count = 1;

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
            } else if (tokens.size() > 2 && tokens[1] == "fen") {
                std::string fen;
                for (size_t i = 2; i < moves_idx; ++i)
                    fen += tokens[i] + (i == moves_idx - 1 ? "" : " ");
                board.setFen(fen);
            }
            ply_count = 1;
            for (size_t i = moves_idx + 1; i < tokens.size(); ++i) {
                history.insert(history.begin(), board);
                if (history.size() > 4) history.pop_back();
                chess::Move move = chess::uci::uciToMove(board, tokens[i]);
                board.makeMove(move);
                ply_count++;
            }
        }
        else if (command == "go") {
            int total_search_nodes = static_cast<int>(cfg.selector.gumbel_search_depth);
            for (size_t i = 1; i + 1 < tokens.size(); ++i) {
                if (tokens[i] == "nodes") total_search_nodes = std::stoi(tokens[i + 1]);
            }
            main_logger.log("INFO", "Dispatching search. Budget: " +
                            std::to_string(total_search_nodes));
            {
                std::lock_guard<std::mutex> lock(search_worker.mtx);
                search_worker.board        = board;
                search_worker.history      = history;
                search_worker.search_nodes = total_search_nodes;
                search_worker.gumbel_m     = static_cast<int>(cfg.selector.gumbel_m);
                search_worker.done_flag    = false;
                search_worker.start_flag   = true;
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
    return 0;
}

// =============================================================================
// TOURNAMENT MODE
// =============================================================================

// One game to be played: an opening, and which model takes White.
struct GameSpec {
    int          game_index;     // 0-based, for logging / CSV ordering
    Opening      opening;
    bool         model_a_is_white;
};

// One finished game's result row.
struct GameRecord {
    int         game_index;
    std::string eco;
    bool        model_a_is_white;
    double      white_value;     // +1 White win / 0 draw / -1 Black win
    SessionEndReason reason;
    int         plies;
};

// Everything one game worker needs that is per-worker (its own engines).
// One worker owns FOUR engine objects: A and B engines for when it plays a
// game, but only TWO are live per game (white + black). We give each worker
// exactly two engines per batcher binding is decided per game, so simplest:
// each worker owns one engine bound to batcher A and one bound to batcher B,
// plus one ActionSelector per engine. The white/black assignment per game maps
// model A/B onto these two engines.
struct WorkerEngines {
    std::unique_ptr<MCTSEngine>     engine_a;   // bound to batcher A (model A)
    std::unique_ptr<MCTSEngine>     engine_b;   // bound to batcher B (model B)
    std::unique_ptr<ActionSelector> selector_a;
    std::unique_ptr<ActionSelector> selector_b;
    std::atomic<int>                wait_a{0};
    std::atomic<int>                wait_b{0};
};

std::string ensure_engine_exists(const std::string& model_path, int max_batch_size, Logger& logger) {
    fs::path p(model_path);
    std::string engine_path = (p.parent_path() / (p.stem().string() + ".engine")).string();
    if (fs::exists(engine_path)) {
        logger.log("INFO", "Using existing engine: " + engine_path);
        return engine_path;
    }

    logger.log("INFO", "Engine not found. Building: " + engine_path);

    TRTBuilder builder;
    auto engine = builder.build_engine(model_path, max_batch_size, logger);
    TRTBuilder::save_engine(*engine, engine_path);

    logger.log("INFO", "Engine build successful.");
    return engine_path;
}

static int run_tournament(const PlayConfig& cfg,
                          const std::string& model_a_path,
                          const std::string& model_b_path,
                          const std::string& run_dir) {
    // ---- pairing log directory ----
    // The orchestrator owns the timestamped run directory and passes it in as
    // --run_dir. This process writes its logs into a per-pairing subdirectory
    // <run_dir>/<stemA>_vs_<stemB>/ and appends results to <run_dir>/results.csv.
    // It does NOT mint its own timestamp -- that would scatter one tournament
    // across many directories.
    std::string stem_a = fs::path(model_a_path).stem().string();
    std::string stem_b = fs::path(model_b_path).stem().string();
    std::string pairing_dir = run_dir + "/" + stem_a + "_vs_" + stem_b;
    fs::create_directories(pairing_dir);

    std::string run_log_dir = pairing_dir;   // all loggers below write here
    std::string results_path = run_dir + "/results.csv";

    Logger main_logger("tournament_main", run_log_dir, cfg.main_logging_level);
    main_logger.rotate(0, 0);
    main_logger.log("INFO", "Tournament pairing: A=" + model_a_path +
                            "  B=" + model_b_path);
    main_logger.log("INFO", "Pairing log dir: " + pairing_dir);

    std::string model_a_engine = ensure_engine_exists(model_a_path, cfg.inference_batch_size, main_logger);
    std::string model_b_engine = ensure_engine_exists(model_b_path, cfg.inference_batch_size, main_logger);

    if (!cfg.main_cores.empty()) {
        DWORD_PTR m = mask_from_cores(cfg.main_cores);
        if (m != 0) SetThreadAffinityMask(GetCurrentThread(), m);
    }


    // ---- opening book: load + deterministic seeded subset ----
    OpeningBook book;
    std::string book_error;
    if (!book.load(cfg.opening_file, book_error)) {
        main_logger.log("CRITICAL", "Opening book load failed: " + book_error);
        std::cerr << "Fatal: " << book_error << std::endl;
        return 1;
    }
    std::vector<Opening> chosen =
        book.sample(static_cast<size_t>(cfg.num_openings),
                    static_cast<uint64_t>(cfg.opening_seed));
    main_logger.log("INFO", "Opening book: " + std::to_string(book.size()) +
                            " parsed, " + std::to_string(chosen.size()) +
                            " sampled (seed " + std::to_string(cfg.opening_seed) + ")");

    // ---- build the game list: each opening twice, colours swapped ----
    std::vector<GameSpec> game_list;
    game_list.reserve(chosen.size() * 2);
    int gidx = 0;
    for (const Opening& op : chosen) {
        game_list.push_back({gidx++, op, true});   // model A = White
        game_list.push_back({gidx++, op, false});  // model A = Black
    }
    main_logger.log("INFO", "Game list built: " +
                            std::to_string(game_list.size()) + " games");

    // ---- two batchers: A and B, each with its own queues / buffers ----
    // CRITICAL: the two batchers must NOT share result_queues. Each batcher
    // scatters by worker_id into ITS OWN result_queues vector. A worker's
    // A-engine and B-engine share a worker_id but index different vectors.
    SharedBuffers buf_a = make_shared_buffers(cfg);
    SharedBuffers buf_b = make_shared_buffers(cfg);

    int K = cfg.concurrent_games;

    moodycamel::ConcurrentQueue<std::pair<int, int>> iq_a, iq_b;
    std::vector<ThreadSafeQueue<std::vector<int>>> rq_a(K), rq_b(K);

    ThreadSafeQueue<int> free_a, free_b;
    for (int i = 0; i < cfg.max_batch_size; ++i) { free_a.push(i); free_b.push(i); }

    std::atomic<uint64_t> step_a{0}, step_b{0};

    // Distinct logger names so the two batchers write separate log files in
    // the pairing subdir (e.g. batcher_step_070000_model.log) instead of
    // clobbering a single inference_batcher.log.
    std::string batcher_a_name = "batcher_" + fs::path(model_a_path).stem().string();
    std::string batcher_b_name = "batcher_" + fs::path(model_b_path).stem().string();

    InferenceBatcher batcher_a(
        model_a_engine, cfg.inference_batch_size, cfg.batch_timeout_ms, K,
        run_log_dir, cfg.batcher_logging_level, cfg.batcher_a_cores,
        0, step_a, cfg.logging_interval_sec, batcher_a_name);
    InferenceBatcher batcher_b(
        model_b_engine, cfg.inference_batch_size, cfg.batch_timeout_ms, K,
        run_log_dir, cfg.batcher_logging_level, cfg.batcher_b_cores,
        0, step_b, cfg.logging_interval_sec, batcher_b_name);

    std::thread bt_a([&]() {
        batcher_a.run(iq_a, rq_a, buf_a.input, buf_a.policy, buf_a.value,
                      global_stop_event, &free_a);
    });
    std::thread bt_b([&]() {
        batcher_b.run(iq_b, rq_b, buf_b.input, buf_b.policy, buf_b.value,
                      global_stop_event, &free_b);
    });

    // ---- per-worker engines: each worker owns one A-engine + one B-engine ----
    // worker_id = w indexes rq_a[w] for the A engine and rq_b[w] for the B one.
    chess::Board dummy;
    dummy.setFen(chess::constants::STARTPOS);
    std::vector<chess::Board> empty_hist;

    std::vector<std::unique_ptr<WorkerEngines>> worker_engines;
    std::vector<std::unique_ptr<Logger>>        worker_loggers;
    worker_engines.reserve(K);
    worker_loggers.reserve(K);

    for (int w = 0; w < K; ++w) {
        worker_loggers.push_back(std::make_unique<Logger>(
            "tournament_worker_" + std::to_string(w), run_log_dir, cfg.worker_logging_level));
        worker_loggers.back()->rotate(0, 0);
        Logger& wlog = *worker_loggers.back();

        auto we = std::make_unique<WorkerEngines>();

        we->engine_a = std::make_unique<MCTSEngine>(
            cfg.selector.node_pool_size, cfg.selector.batch_size_per_worker,
            iq_a, rq_a[w], w,
            cfg.selector.virtual_loss, cfg.selector.contempt, cfg.selector.draw_cutoff,
            cfg.selector.gumbel_c_visit, cfg.selector.gumbel_c_scale,
            cfg.selector.gumbel_noise, dummy, empty_hist, wlog,
            buf_a.input, buf_a.policy, buf_a.value,
            free_a, &we->wait_a, 1);

        we->engine_b = std::make_unique<MCTSEngine>(
            cfg.selector.node_pool_size, cfg.selector.batch_size_per_worker,
            iq_b, rq_b[w], w,
            cfg.selector.virtual_loss, cfg.selector.contempt, cfg.selector.draw_cutoff,
            cfg.selector.gumbel_c_visit, cfg.selector.gumbel_c_scale,
            cfg.selector.gumbel_noise, dummy, empty_hist, wlog,
            buf_b.input, buf_b.policy, buf_b.value,
            free_b, &we->wait_b, 1);

        we->selector_a = std::make_unique<ActionSelector>(
            "sel_a_" + std::to_string(w), w, cfg.selector, wlog);
        we->selector_b = std::make_unique<ActionSelector>(
            "sel_b_" + std::to_string(w), w, cfg.selector, wlog);

        worker_engines.push_back(std::move(we));
        std::this_thread::sleep_for(std::chrono::milliseconds(100)); 
    }

    // ---- shared work queue + results ----
    std::atomic<size_t> next_game{0};
    std::vector<GameRecord> records(game_list.size());
    std::mutex records_mtx;

    // Pin each worker to a specific core, mirroring data_generator.cpp:
    // worker w runs on tournament_worker_cores[w / workers_per_core].
    auto worker_fn = [&](int w) {
        int core_index = w / cfg.workers_per_core;
        int core_id = cfg.tournament_worker_cores[core_index];
        SetThreadAffinityMask(GetCurrentThread(),
                              static_cast<DWORD_PTR>(1) << core_id);
        at::set_num_threads(1);

        WorkerEngines& we = *worker_engines[w];
        Logger& wlog = *worker_loggers[w];

        int budget = static_cast<int>(cfg.selector.gumbel_search_depth);
        int gm     = static_cast<int>(cfg.selector.gumbel_m);

        while (true) {
            size_t idx = next_game.fetch_add(1);
            if (idx >= game_list.size()) break;
            const GameSpec& spec = game_list[idx];

            // Map model A/B onto white/black for THIS game.
            // engine_a is model A, engine_b is model B.
            SearchAgent agent_a{*we.engine_a, *we.selector_a, budget, gm};
            SearchAgent agent_b{*we.engine_b, *we.selector_b, budget, gm};

            SearchAgent white = spec.model_a_is_white ? agent_a : agent_b;
            SearchAgent black = spec.model_a_is_white ? agent_b : agent_a;

            // GameWorker's `primary` is the side we nominally drive. For
            // self-play it does not matter which side that is -- the session
            // runs the other side internally. We pick White as primary.
            we.selector_a->reset_for_new_game();
            we.selector_b->reset_for_new_game();

            SelfPlaySession session(white, black, chess::Color::WHITE,
                                    spec.opening, cfg.max_ply_length, wlog);

            GameWorker gw(w, white, wlog);
            SessionResult res = gw.run_one_game(session);

            GameRecord rec;
            rec.game_index       = spec.game_index;
            rec.eco              = spec.opening.eco;
            rec.model_a_is_white = spec.model_a_is_white;
            rec.white_value      = res.white_value;
            rec.reason           = res.reason;
            rec.plies            = session.total_plies();
            {
                std::lock_guard<std::mutex> lk(records_mtx);
                records[idx] = rec;
            }

            wlog.log("INFO", "Finished game " + std::to_string(spec.game_index) +
                             " (" + std::to_string(idx + 1) + "/" +
                             std::to_string(game_list.size()) + ")");
        }
    };

    main_logger.log("INFO", "Spawning " + std::to_string(K) + " game workers...");
    std::vector<std::thread> workers;
    for (int w = 0; w < K; ++w) workers.emplace_back(worker_fn, w);
    for (auto& t : workers) if (t.joinable()) t.join();
    main_logger.log("INFO", "All games complete.");

    // ---- shut the batchers down ----
    global_stop_event.store(true);
    if (bt_a.joinable()) bt_a.join();
    if (bt_b.joinable()) bt_b.join();

    // ---- write results CSV ----
    auto reason_str = [](SessionEndReason r) -> const char* {
        switch (r) {
            case SessionEndReason::CHECKMATE:   return "checkmate";
            case SessionEndReason::DRAW_RULES:  return "draw";
            case SessionEndReason::RESIGNATION: return "resignation";
            case SessionEndReason::PLY_LIMIT:   return "ply_limit";
            case SessionEndReason::ABORTED:     return "aborted";
            default:                            return "unknown";
        }
    };

    std::ofstream csv(results_path, std::ios::app);

    if (!csv) {
        main_logger.log("ERROR", "Could not open results file: " + results_path);
        std::cerr << "Warning: could not write results to " << results_path << std::endl;
    } else {
        // Header only if file is empty
        csv.seekp(0, std::ios::end);

        if (csv.tellp() == 0) {
            csv << "game_index,eco,model_white,model_black,winner,model_a_score,model_b_score,reason,plies\n";
        }

        for (const GameRecord& r : records) {
            std::string model_white = r.model_a_is_white ? stem_a : stem_b;
            std::string model_black = r.model_a_is_white ? stem_b : stem_a;

            std::string winner;
            int model_a_score = 0;
            int model_b_score = 0;

            if (r.reason == SessionEndReason::ABORTED) {
                winner = "aborted";
            }
            else if (r.white_value > 0.0) {
                winner = "white";

                if (r.model_a_is_white) {
                    model_a_score = 1;
                    model_b_score = -1;
                } else {
                    model_a_score = -1;
                    model_b_score = 1;
                }
            }
            else if (r.white_value < 0.0) {
                winner = "black";

                if (r.model_a_is_white) {
                    model_a_score = -1;
                    model_b_score = 1;
                } else {
                    model_a_score = 1;
                    model_b_score = -1;
                }
            }
            else {
                winner = "draw";
            }

            csv << r.game_index << ','
                << r.eco << ','
                << model_white << ','
                << model_black << ','
                << winner << ','
                << model_a_score << ','
                << model_b_score << ','
                << reason_str(r.reason) << ','
                << r.plies << '\n';
        }

        csv.flush();
        main_logger.log("INFO", "Results appended to " + results_path);
    }

    return 0;
}

// =============================================================================
static void print_usage() {
    std::cerr <<
        "Usage:\n"
        "  talbot_play --uci        --config_file <yaml>\n"
        "  talbot_play --tournament --config_file <yaml> "
        "--model_a <A> --model_b <B> --run_dir <dir>\n";
}

int main(int argc, char* argv[]) {
    std::string config_file_path;          // required: no default, must be passed
    std::string mode = "--uci";
    std::string model_a, model_b, run_dir;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--config_file" && i + 1 < argc)       config_file_path = argv[++i];
        else if (arg == "--model_a" && i + 1 < argc)      model_a = argv[++i];
        else if (arg == "--model_b" && i + 1 < argc)      model_b = argv[++i];
        else if (arg == "--run_dir" && i + 1 < argc)      run_dir = argv[++i];
        else if (arg == "--uci" || arg == "--tournament") mode = arg;
        else {
            std::cerr << "Fatal: unrecognised argument: " << arg << "\n";
            print_usage();
            return 1;
        }
    }

    // Config file resolution:
    //   --tournament : --config_file is REQUIRED (orchestrator always passes it).
    //   --uci        : --config_file is OPTIONAL. If omitted it falls back to
    //                  DEFAULT_UCI_CONFIG. This lets a chess GUI launch the bare
    //                  talbot_play.exe with no arguments -- which is what every
    //                  UCI GUI expects and the only reliable way to run under
    //                  CuteChess et al. (GUIs cannot pass startup args cleanly).
    static const char* DEFAULT_UCI_CONFIG =
        "D:/Projects/talbot/config/play_uci.yaml";

    if (config_file_path.empty()) {
        if (mode == "--uci") {
            config_file_path = DEFAULT_UCI_CONFIG;
        } else {
            std::cerr << "Fatal: --tournament requires --config_file\n";
            print_usage();
            return 1;
        }
    }
    if (!fs::exists(config_file_path)) {
        std::cerr << "Fatal: config file not found at " << config_file_path << std::endl;
        return 1;
    }

    PlayConfig cfg;
    try {
        cfg = load_config(config_file_path);
    } catch (const std::exception& e) {
        std::cerr << "Fatal: failed to load config: " << e.what() << std::endl;
        return 1;
    }

    if (mode == "--tournament") {
        if (model_a.empty() || model_b.empty()) {
            std::cerr << "Fatal: --tournament requires --model_a and --model_b\n";
            print_usage();
            return 1;
        }
        if (run_dir.empty()) {
            std::cerr << "Fatal: --tournament requires --run_dir "
                         "(the orchestrator-created run directory)\n";
            print_usage();
            return 1;
        }
        return run_tournament(cfg, model_a, model_b, run_dir);
    }
    return run_uci(cfg);
}