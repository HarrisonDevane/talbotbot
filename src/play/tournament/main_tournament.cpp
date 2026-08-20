// =============================================================================
// main_tournament.cpp
//
// Entry point for tournament pairings.
//
// CLI:
//   talbot_tournament --config_file <yaml> --model_a <A.engine> --model_b <B.engine> --run_dir <dir>
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

static DWORD_PTR mask_from_cores(const std::vector<int>& cores) {
    DWORD_PTR mask = 0;
    for (int c : cores) mask |= (static_cast<DWORD_PTR>(1) << c);
    return mask;
}

// =============================================================================
// CONFIG
// =============================================================================
struct TournamentConfig {
    std::string base_log_dir;
    int  main_logging_level;

    std::vector<int> main_cores;
    std::vector<int> batcher_a_cores;
    std::vector<int> batcher_b_cores;
    std::vector<int> tournament_worker_cores;

    int inference_batch_size;
    int max_batch_size;
    int batch_timeout_ms;
    int batcher_logging_level;
    int logging_interval_sec;

    int input_planes;
    int board_dim;
    int policy_moves;

    std::string opening_file;
    int games_per_match;
    int num_openings;
    int workers_per_core;
    int concurrent_games;
    int max_ply_length;
    int worker_logging_level;

    ActionSelectorConfig selector;
};

static TournamentConfig load_config(const std::string& config_file_path, const std::string& model_file_path) {
    TournamentConfig cfg;

    YAML::Node root   = YAML::LoadFile(config_file_path);
    YAML::Node global = root["global"];
    YAML::Node eval_n = root["evaluation"];
    YAML::Node infer_n= root["inference"];
    YAML::Node mcts_n = root["mcts"];
    YAML::Node sel_n  = root["selection"];
    YAML::Node tour_n = root["tournament"];

    if (!tour_n) throw std::runtime_error("tournament block missing in config");

    cfg.base_log_dir       = global["log_dir"].as<std::string>();
    cfg.main_logging_level = global["main_logging_level"].as<int>();

    if (eval_n && eval_n["main_cores"])
        for (const auto& c : eval_n["main_cores"]) cfg.main_cores.push_back(c.as<int>());

    if (infer_n["batcher_a_cores"])
        for (const auto& c : infer_n["batcher_a_cores"]) cfg.batcher_a_cores.push_back(c.as<int>());
    if (infer_n["batcher_b_cores"])
        for (const auto& c : infer_n["batcher_b_cores"]) cfg.batcher_b_cores.push_back(c.as<int>());

    cfg.inference_batch_size  = infer_n["batch_size"].as<int>();
    cfg.max_batch_size        = cfg.inference_batch_size * infer_n["batch_size_factor"].as<int>();
    cfg.batch_timeout_ms      = infer_n["batch_timeout_ms"].as<int>();
    cfg.batcher_logging_level = infer_n["logging_level"].as<int>();
    cfg.logging_interval_sec  = infer_n["logging_interval_sec"].as<int>();

    YAML::Node model = YAML::LoadFile(model_file_path);
    cfg.input_planes = model["model"]["input_planes"].as<int>();
    cfg.board_dim    = model["model"]["board_dim"].as<int>();
    cfg.policy_moves = model["model"]["total_policy_moves"].as<int>();

    cfg.opening_file      = tour_n["opening_file"].as<std::string>();
    cfg.games_per_match   = tour_n["games_per_match"].as<int>();
    cfg.workers_per_core  = tour_n["workers_per_core"].as<int>();
    cfg.max_ply_length    = tour_n["max_ply_length"].as<int>();
    cfg.worker_logging_level = tour_n["worker_logging_level"].as<int>();

    if (cfg.games_per_match <= 0 || (cfg.games_per_match % 2) != 0) {
        throw std::runtime_error(
            "tournament.games_per_match must be a positive even number "
            "(each opening is played twice); got " +
            std::to_string(cfg.games_per_match));
    }
    cfg.num_openings = cfg.games_per_match / 2;

    for (const auto& c : tour_n["game_worker_cores"])
        cfg.tournament_worker_cores.push_back(c.as<int>());
    if (cfg.tournament_worker_cores.empty()) {
        throw std::runtime_error("tournament.game_worker_cores is empty");
    }

    cfg.concurrent_games = static_cast<int>(cfg.tournament_worker_cores.size()) * cfg.workers_per_core;

    ActionSelectorConfig& s = cfg.selector;
    s.node_pool_size         = mcts_n["node_pool_size"].as<int>();
    s.virtual_loss           = mcts_n["virtual_loss"].as<double>();
    s.contempt               = mcts_n["contempt"].as<double>();
    s.deficit_eps            = mcts_n["deficit_eps"].as<double>();
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
// TOURNAMENT STRUCTURES
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
        b.value.push_back(torch::zeros({1}, opts));
    }
    return b;
}

struct GameSpec {
    int          game_index;
    Opening      opening;
    bool         model_a_is_white;
};

struct GameRecord {
    int         game_index;
    std::string eco;
    bool        model_a_is_white;
    double      white_value;
    SessionEndReason reason;
    int         plies;
};

struct WorkerEngines {
    std::unique_ptr<MCTSEngine>     engine_a;
    std::unique_ptr<MCTSEngine>     engine_b;
    std::unique_ptr<ActionSelector> selector_a;
    std::unique_ptr<ActionSelector> selector_b;
    std::atomic<int>                wait_a{0};
    std::atomic<int>                wait_b{0};
};

std::string ensure_engine_exists(const std::string& model_path, int max_batch_size, int input_planes, Logger& logger) {
    fs::path p(model_path);
    std::string engine_path = (p.parent_path() / (p.stem().string() + ".engine")).string();
    if (fs::exists(engine_path)) {
        logger.log("INFO", "Using existing engine: " + engine_path);
        return engine_path;
    }

    logger.log("INFO", "Engine not found. Building: " + engine_path);
    TRTBuilder builder;
    auto engine = builder.build_engine(model_path, max_batch_size, input_planes, logger);
    TRTBuilder::save_engine(*engine, engine_path);
    logger.log("INFO", "Engine build successful.");
    return engine_path;
}

// =============================================================================
// MAIN TOURNAMENT LOOP
// =============================================================================
static int run_tournament(const TournamentConfig& cfg,
                          const std::string& model_a_path,
                          const std::string& model_b_path,
                          const std::string& run_dir) {

    std::string stem_a = fs::path(model_a_path).stem().string();
    std::string stem_b = fs::path(model_b_path).stem().string();
    std::string pairing_dir = run_dir + "/" + stem_a + "_vs_" + stem_b;
    fs::create_directories(pairing_dir);

    std::string run_log_dir = pairing_dir;
    std::string results_path = run_dir + "/results.csv";

    Logger main_logger("tournament_main", run_log_dir, cfg.main_logging_level);
    main_logger.rotate(0, 0);
    main_logger.log("INFO", "Tournament pairing: A=" + model_a_path + "  B=" + model_b_path);
    main_logger.log("INFO", "Pairing log dir: " + pairing_dir);

    std::string model_a_engine = ensure_engine_exists(model_a_path, cfg.inference_batch_size, cfg.input_planes, main_logger);
    std::string model_b_engine = ensure_engine_exists(model_b_path, cfg.inference_batch_size, cfg.input_planes, main_logger);

    if (!cfg.main_cores.empty()) {
        DWORD_PTR m = mask_from_cores(cfg.main_cores);
        if (m != 0) SetThreadAffinityMask(GetCurrentThread(), m);
    }

    OpeningBook book;
    std::string book_error;
    if (!book.load(cfg.opening_file, book_error)) {
        main_logger.log("CRITICAL", "Opening book load failed: " + book_error);
        std::cerr << "Fatal: " << book_error << std::endl;
        return 1;
    }
    std::vector<Opening> chosen = book.sample(static_cast<size_t>(cfg.num_openings));
    main_logger.log("INFO", "Opening book: " + std::to_string(book.size()) +
                            " parsed, " + std::to_string(chosen.size()) + " chosen.");

    std::vector<GameSpec> game_list;
    game_list.reserve(chosen.size() * 2);
    int gidx = 0;
    for (const Opening& op : chosen) {
        game_list.push_back({gidx++, op, true});
        game_list.push_back({gidx++, op, false});
    }
    main_logger.log("INFO", "Game list built: " + std::to_string(game_list.size()) + " games");

    SharedBuffers buf_a = make_shared_buffers(cfg);
    SharedBuffers buf_b = make_shared_buffers(cfg);
    int K = cfg.concurrent_games;

    moodycamel::ConcurrentQueue<std::pair<int, int>> iq_a, iq_b;
    std::vector<ThreadSafeQueue<std::vector<int>>> rq_a(K), rq_b(K);
    ThreadSafeQueue<int> free_a, free_b;
    for (int i = 0; i < cfg.max_batch_size; ++i) { free_a.push(i); free_b.push(i); }

    std::atomic<uint64_t> step_a{0}, step_b{0};
    std::string batcher_a_name = "batcher_" + stem_a;
    std::string batcher_b_name = "batcher_" + stem_b;

    InferenceBatcher batcher_a(
        model_a_engine, cfg.inference_batch_size, cfg.batch_timeout_ms, K,
        run_log_dir, cfg.batcher_logging_level, cfg.batcher_a_cores,
        0, step_a, cfg.logging_interval_sec, batcher_a_name);
    InferenceBatcher batcher_b(
        model_b_engine, cfg.inference_batch_size, cfg.batch_timeout_ms, K,
        run_log_dir, cfg.batcher_logging_level, cfg.batcher_b_cores,
        0, step_b, cfg.logging_interval_sec, batcher_b_name);

    std::thread bt_a([&]() {
        batcher_a.run(iq_a, rq_a, buf_a.input, buf_a.policy, buf_a.value, global_stop_event, &free_a);
    });
    std::thread bt_b([&]() {
        batcher_b.run(iq_b, rq_b, buf_b.input, buf_b.policy, buf_b.value, global_stop_event, &free_b);
    });

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
            iq_a, rq_a[w], w, cfg.selector.deficit_eps, cfg.selector.virtual_loss, 
            cfg.selector.contempt, cfg.selector.draw_cutoff, cfg.selector.gumbel_c_visit, 
            cfg.selector.gumbel_c_scale, cfg.selector.gumbel_noise, dummy, empty_hist, 
            wlog, buf_a.input, buf_a.policy, buf_a.value, free_a, &we->wait_a, 1);

        we->engine_b = std::make_unique<MCTSEngine>(
            cfg.selector.node_pool_size, cfg.selector.batch_size_per_worker,
            iq_b, rq_b[w], w, cfg.selector.deficit_eps, cfg.selector.virtual_loss, 
            cfg.selector.contempt, cfg.selector.draw_cutoff, cfg.selector.gumbel_c_visit, 
            cfg.selector.gumbel_c_scale, cfg.selector.gumbel_noise, dummy, empty_hist, 
            wlog, buf_b.input, buf_b.policy, buf_b.value, free_b, &we->wait_b, 1);

        we->selector_a = std::make_unique<ActionSelector>("sel_a_" + std::to_string(w), w, cfg.selector, wlog);
        we->selector_b = std::make_unique<ActionSelector>("sel_b_" + std::to_string(w), w, cfg.selector, wlog);

        worker_engines.push_back(std::move(we));
        std::this_thread::sleep_for(std::chrono::milliseconds(100)); 
    }

    std::atomic<size_t> next_game{0};
    std::vector<GameRecord> records(game_list.size());
    std::mutex records_mtx;

    auto worker_fn = [&](int w) {
        int core_index = w / cfg.workers_per_core;
        int core_id = cfg.tournament_worker_cores[core_index];
        SetThreadAffinityMask(GetCurrentThread(), static_cast<DWORD_PTR>(1) << core_id);
        at::set_num_threads(1);

        WorkerEngines& we = *worker_engines[w];
        Logger& wlog = *worker_loggers[w];
        int budget = static_cast<int>(cfg.selector.gumbel_search_depth);
        int gm     = static_cast<int>(cfg.selector.gumbel_m);

        while (true) {
            size_t idx = next_game.fetch_add(1);
            if (idx >= game_list.size()) break;
            const GameSpec& spec = game_list[idx];

            SearchAgent agent_a{*we.engine_a, *we.selector_a, budget, gm};
            SearchAgent agent_b{*we.engine_b, *we.selector_b, budget, gm};
            SearchAgent white = spec.model_a_is_white ? agent_a : agent_b;
            SearchAgent black = spec.model_a_is_white ? agent_b : agent_a;

            we.selector_a->reset_for_new_game();
            we.selector_b->reset_for_new_game();

            SelfPlaySession session(white, black, chess::Color::WHITE, spec.opening, cfg.max_ply_length, wlog);
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
                             " (" + std::to_string(idx + 1) + "/" + std::to_string(game_list.size()) + ")");
        }
    };

    main_logger.log("INFO", "Spawning " + std::to_string(K) + " game workers...");
    std::vector<std::thread> workers;
    for (int w = 0; w < K; ++w) workers.emplace_back(worker_fn, w);
    for (auto& t : workers) if (t.joinable()) t.join();
    main_logger.log("INFO", "All games complete.");

    global_stop_event.store(true);
    if (bt_a.joinable()) bt_a.join();
    if (bt_b.joinable()) bt_b.join();

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
            } else if (r.white_value > 0.0) {
                winner = "white";
                if (r.model_a_is_white) { model_a_score = 1; model_b_score = -1; } 
                else { model_a_score = -1; model_b_score = 1; }
            } else if (r.white_value < 0.0) {
                winner = "black";
                if (r.model_a_is_white) { model_a_score = -1; model_b_score = 1; } 
                else { model_a_score = 1; model_b_score = -1; }
            } else {
                winner = "draw";
            }

            csv << r.game_index << ',' << r.eco << ',' << model_white << ',' << model_black << ','
                << winner << ',' << model_a_score << ',' << model_b_score << ','
                << reason_str(r.reason) << ',' << r.plies << '\n';
        }
        csv.flush();
        main_logger.log("INFO", "Results appended to " + results_path);
    }
    return 0;
}

// =============================================================================
static void print_usage() {
    std::cerr << "Usage: talbot_tournament --config_file <yaml> --model_a <A> --model_b <B> --run_dir <dir>\n";
}

int main(int argc, char* argv[]) {
    std::string config_file_path;
    std::string model_a, model_b, run_dir;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--config_file" && i + 1 < argc)       config_file_path = argv[++i];
        else if (arg == "--model_a" && i + 1 < argc)      model_a = argv[++i];
        else if (arg == "--model_b" && i + 1 < argc)      model_b = argv[++i];
        else if (arg == "--run_dir" && i + 1 < argc)      run_dir = argv[++i];
        else {
            std::cerr << "Fatal: unrecognised argument: " << arg << "\n";
            print_usage();
            return 1;
        }
    }

    if (config_file_path.empty() || model_a.empty() || model_b.empty() || run_dir.empty()) {
        std::cerr << "Fatal: Missing required arguments.\n";
        print_usage();
        return 1;
    }

    if (!fs::exists(config_file_path)) {
        std::cerr << "Fatal: config file not found at " << config_file_path << std::endl;
        return 1;
    }

    TournamentConfig cfg;
    // Load config extracting the model.yaml path from one of the engines (assuming both have same dims)
    std::string model_yaml_path = (fs::path(model_a).parent_path() / "model.yaml").string();
    if (!fs::exists(model_yaml_path)) {
        std::cerr << "Fatal: model.yaml not found at " << model_yaml_path << std::endl;
        return 1;
    }

    try {
        cfg = load_config(config_file_path, model_yaml_path);
    } catch (const std::exception& e) {
        std::cerr << "Fatal: failed to load config: " << e.what() << std::endl;
        return 1;
    }

    return run_tournament(cfg, model_a, model_b, run_dir);
}