#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include <iostream>
#include <vector>
#include <thread>
#include <atomic>
#include <string>
#include <sstream>
#include <filesystem>
#include <unordered_map>
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

namespace fs = std::filesystem;
std::atomic<bool> global_stop_event{false};

struct SearchWorker {
    std::thread thread;
    std::mutex mtx;
    std::condition_variable cv_start;
    std::condition_variable cv_done;

    bool start_flag = false;
    bool quit_flag = false;
    bool done_flag = true;

    chess::Board board;
    std::vector<chess::Board> history;
    int search_nodes = 0;
    int gumbel_m = 0;

    MCTSEngine* mcts = nullptr;
    DWORD_PTR core_mask = 0;
};

std::vector<std::string> split(const std::string& s, char delimiter) {
    std::vector<std::string> tokens;
    std::string token;
    std::istringstream tokenStream(s);
    while (std::getline(tokenStream, token, delimiter)) {
        if (!token.empty()) tokens.push_back(token);
    }
    return tokens;
}

int main(int argc, char* argv[]) {
    std::string config_file_path = "D:/Projects/talbot/config/local_inference.yaml";

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--config_file" && i + 1 < argc) {
            config_file_path = argv[i + 1];
            i++;
        }
    }

    if (!fs::exists(config_file_path)) {
        std::cerr << "Fatal: Master config file not found at " << config_file_path << std::endl;
        return 1;
    }

    YAML::Node config_input = YAML::LoadFile(config_file_path);
    YAML::Node global_config_input = config_input["global"];
    YAML::Node eval_config_input = config_input["evaluation"];
    YAML::Node inference_config_input = config_input["inference"];
    YAML::Node mcts_config_input = config_input["mcts"];
    YAML::Node selector_config_input = config_input["selection"];

    std::string model_file_path = global_config_input["model_file"].as<std::string>();
    std::string base_log_dir = global_config_input["log_dir"].as<std::string>();
    std::string base_model_path = global_config_input["model_path"].as<std::string>();
    std::string engine_path = base_model_path + ".engine";

    if (!fs::exists(model_file_path)) {
        std::cerr << "Fatal: Model config file not found at " << model_file_path << std::endl;
        return 1;
    }
    YAML::Node model = YAML::LoadFile(model_file_path);

    auto now = std::chrono::system_clock::now();
    std::time_t now_time = std::chrono::system_clock::to_time_t(now);
    std::tm* local_time = std::localtime(&now_time);
    
    std::ostringstream time_oss;
    time_oss << std::put_time(local_time, "%Y-%m-%d_%H-%M-%S");
    
    std::string run_log_dir = base_log_dir + "/" + time_oss.str();
    fs::create_directories(run_log_dir);
    
    Logger main_logger("uci_main", run_log_dir, global_config_input["main_logging_level"].as<int>());
    main_logger.rotate(0, 0); 
    main_logger.log("INFO", "Booting Talbot UCI Engine (Single Worker, Unified Logging)...");

    if (eval_config_input["main_cores"]) {
        DWORD_PTR mainMask = 0;
        for (const auto& core : eval_config_input["main_cores"]) {
            mainMask |= (static_cast<DWORD_PTR>(1) << core.as<int>());
        }
        if (mainMask != 0) SetThreadAffinityMask(GetCurrentThread(), mainMask);
    }

    if (!fs::exists(engine_path)) {
        main_logger.log("CRITICAL", "Engine file missing at " + engine_path);
        return 1;
    }

    int inference_batch_size = inference_config_input["batch_size"].as<int>();
    int max_batch_size = inference_batch_size * inference_config_input["batch_size_factor"].as<int>();
    int input_planes = model["chess"]["input_planes"].as<int>();
    int board_dim = model["chess"]["board_dim"].as<int>(); 
    int policy_moves = model["chess"]["total_policy_moves"].as<int>();
    
    ModelConfig model_config{input_planes, board_dim, policy_moves};
    auto options_half = torch::TensorOptions().dtype(torch::kHalf).device(torch::kCPU);
    
    std::vector<torch::Tensor> shared_input_buffer;
    std::vector<torch::Tensor> shared_policy_buffer;
    std::vector<torch::Tensor> shared_value_buffer;

    for (int i = 0; i < max_batch_size; ++i) {
        shared_input_buffer.push_back(torch::zeros({input_planes, board_dim, board_dim}, options_half));
        shared_policy_buffer.push_back(torch::zeros({policy_moves}, options_half));
        shared_value_buffer.push_back(torch::zeros({3}, options_half));
    }

    moodycamel::ConcurrentQueue<std::pair<int, int>> inference_queue;
    std::vector<ThreadSafeQueue<std::vector<int>>> result_queues(1); 
    ThreadSafeQueue<int> buffer_free_slots;
    for (int i = 0; i < max_batch_size; ++i) buffer_free_slots.push(i);

    std::vector<int> batcher_cores;
    for (const auto& core : inference_config_input["inference_worker_cores"]) {
        batcher_cores.push_back(core.as<int>());
    }

    std::atomic<uint64_t> dummy_step{0};

    InferenceBatcher batcher(
        engine_path, inference_batch_size, inference_config_input["batch_timeout_ms"].as<int>(), 1, 
        run_log_dir, inference_config_input["logging_level"].as<int>(), batcher_cores, 0, dummy_step, inference_config_input["logging_interval_sec"].as<int>()
    );
    
    std::thread batcher_thread([&]() {
        batcher.run(inference_queue, result_queues, shared_input_buffer, shared_policy_buffer, shared_value_buffer, global_stop_event, &buffer_free_slots);
    });

    chess::Board board;
    board.setFen(chess::constants::STARTPOS);
    std::vector<chess::Board> history;

    ActionSelectorConfig selector_config;
    selector_config.node_pool_size = mcts_config_input["node_pool_size"].as<int>();
    selector_config.virtual_loss = mcts_config_input["virtual_loss"].as<double>();
    selector_config.contempt = mcts_config_input["contempt"].as<double>();
    selector_config.draw_cutoff = selector_config_input["draw_cutoff"].as<double>();
    selector_config.gumbel_c_visit = mcts_config_input["gumbel_c_visit"].as<double>();
    selector_config.gumbel_c_scale = mcts_config_input["gumbel_c_scale"].as<double>();
    selector_config.gumbel_noise = mcts_config_input["gumbel_noise"].as<double>();
    selector_config.gumbel_search_depth = mcts_config_input["gumbel_search_depth"].as<int>();
    selector_config.gumbel_m = mcts_config_input["gumbel_m"].as<int>();
    selector_config.batch_size_per_worker = mcts_config_input["worker_minibatch_size"].as<int>();
    selector_config.temperature_ply_cutoff = selector_config_input["temperature_ply_cutoff"].as<int>();
    selector_config.temperature_q_decay = selector_config_input["temperature_q_decay"].as<double>();
    selector_config.resignation_probability = selector_config_input["resignation_probability"].as<double>();
    selector_config.resignation_cutoff = selector_config_input["resignation_cutoff"].as<double>();

    std::atomic<int> single_worker_wait_count{0};

    auto mcts_engine = std::make_unique<MCTSEngine>(
        selector_config.node_pool_size, selector_config.batch_size_per_worker, 
        inference_queue, result_queues[0], 0, 
        selector_config.virtual_loss, selector_config.contempt, selector_config.draw_cutoff, 
        selector_config.gumbel_c_visit, selector_config.gumbel_c_scale, 
        selector_config.gumbel_noise, board, history, main_logger, 
        shared_input_buffer, shared_policy_buffer, shared_value_buffer,
        buffer_free_slots, &single_worker_wait_count, 1
    );

    SearchWorker search_worker;
    search_worker.mcts = mcts_engine.get();
    
    if (eval_config_input["game_worker_cores"]) {
        for (const auto& core : eval_config_input["game_worker_cores"]) {
            search_worker.core_mask |= (static_cast<DWORD_PTR>(1) << core.as<int>());
        }
    }

    search_worker.thread = std::thread([worker = &search_worker]() {
        if (worker->core_mask != 0) {
            SetThreadAffinityMask(GetCurrentThread(), worker->core_mask);
        }
        while (true) {
            std::unique_lock<std::mutex> lock(worker->mtx);
            worker->cv_start.wait(lock, [&]{ return worker->start_flag || worker->quit_flag; });
            if (worker->quit_flag) break;
            
            worker->mcts->reset(worker->board, worker->history);
            worker->mcts->run_simulations(worker->search_nodes, worker->gumbel_m);
            
            worker->start_flag = false;
            worker->done_flag = true;
            lock.unlock();
            worker->cv_done.notify_one();
        }
    });

    ActionSelector agent("uci_agent", 0, selector_config, main_logger);
    int ply_count = 1;

    std::string line;
    while (std::getline(std::cin, line)) {
        line.erase(std::remove(line.begin(), line.end(), '\r'), line.end());
        main_logger.log("DEBUG", "GUI -> Engine: " + line);
        std::vector<std::string> tokens = split(line, ' ');
        if (tokens.empty()) continue;

        const std::string& command = tokens[0];

        if (command == "uci") {
            std::cout << "id name Talbot UCI (Single)" << std::endl;
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
                std::string fen = "";
                for (size_t i = 2; i < moves_idx; ++i) fen += tokens[i] + (i == moves_idx - 1 ? "" : " ");
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
            int total_search_nodes = selector_config.gumbel_search_depth;
            for (size_t i = 1; i < tokens.size() - 1; ++i) {
                if (tokens[i] == "nodes") total_search_nodes = std::stoi(tokens[i + 1]);
            }

            main_logger.log("INFO", "Dispatching search to single worker. Budget: " + std::to_string(total_search_nodes));

            auto search_start_time = std::chrono::steady_clock::now();

            {
                std::lock_guard<std::mutex> lock(search_worker.mtx);
                search_worker.board = board;
                search_worker.history = history;
                search_worker.search_nodes = total_search_nodes;
                search_worker.gumbel_m = selector_config.gumbel_m;
                search_worker.done_flag = false;
                search_worker.start_flag = true;
                search_worker.cv_start.notify_one();
            }

            {
                std::unique_lock<std::mutex> lock(search_worker.mtx);
                search_worker.cv_done.wait(lock, [&]{ return search_worker.done_flag; });
            }

            MCTSNode* root = mcts_engine->root;
            double root_v_mix = root->calculate_v_mix(selector_config.contempt);

            SelectionResult result = agent.select_move(root, ply_count);

            std::string best_move_str = (result.best_move == chess::Move::NO_MOVE) ? "0000" : chess::uci::moveToUci(result.best_move);
            
            std::cout << "bestmove " << best_move_str << std::endl;
            main_logger.log("DEBUG", "Engine -> GUI: bestmove " + best_move_str);
        } 
        else if (command == "quit") {
            break;
        }
    }

    main_logger.log("INFO", "Quit command received. Terminating worker...");
    {
        std::lock_guard<std::mutex> lock(search_worker.mtx);
        search_worker.quit_flag = true;
    }
    search_worker.cv_start.notify_one();
    if (search_worker.thread.joinable()) {
        search_worker.thread.join();
    }

    global_stop_event.store(true);
    if (batcher_thread.joinable()) batcher_thread.join();
    return 0;
}