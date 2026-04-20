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
    int core_id = -1;
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

    YAML::Node config = YAML::LoadFile(config_file_path);
    YAML::Node global_cfg = config["global"];
    YAML::Node eval_cfg = config["evaluation"];
    YAML::Node inf_cfg = config["inference"];
    YAML::Node mcts_cfg = config["mcts"];
    YAML::Node sel_cfg = config["selection"];

    std::string model_file_path = global_cfg["model_file"].as<std::string>();
    std::string base_log_dir = global_cfg["log_dir"].as<std::string>();
    std::string base_model_path = global_cfg["model_path"].as<std::string>();
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
    
    Logger main_logger("uci_main", run_log_dir, global_cfg["main_logging_level"].as<int>());
    main_logger.rotate(0, 0); 
    main_logger.log("INFO", "Booting Talbot UCI Engine (Root Parallelism Pool)...");

    if (eval_cfg["main_cores"]) {
        DWORD_PTR mainMask = 0;
        for (const auto& core : eval_cfg["main_cores"]) {
            mainMask |= (static_cast<DWORD_PTR>(1) << core.as<int>());
        }
        if (mainMask != 0) SetThreadAffinityMask(GetCurrentThread(), mainMask);
    }

    if (!fs::exists(engine_path)) {
        main_logger.log("CRITICAL", "Engine file missing at " + engine_path);
        return 1;
    }

    std::vector<int> worker_core_ids;
    for (const auto& core : eval_cfg["game_worker_cores"]) {
        worker_core_ids.push_back(core.as<int>());
    }
    int workers_per_core = eval_cfg["workers_per_core"].as<int>();
    int num_workers = std::max(1, (int)worker_core_ids.size() * workers_per_core);

    main_logger.log("INFO", "Initializing " + std::to_string(num_workers) + " Persistent Ensemble Workers.");

    int inference_batch_size = inf_cfg["batch_size"].as<int>();
    int max_batch_size = inference_batch_size * inf_cfg["batch_size_factor"].as<int>();
    int input_planes = model["model"]["input_planes"].as<int>();
    int board_dim = model["chess"]["board_dim"].as<int>(); 
    int policy_moves = model["chess"]["total_policy_moves"].as<int>();
    
    ModelConfig m_config{input_planes, board_dim, policy_moves};
    auto options_half = torch::TensorOptions().dtype(torch::kHalf).device(torch::kCPU);
    
    std::vector<torch::Tensor> shared_input_buffer;
    std::vector<torch::Tensor> shared_policy_buffer;
    std::vector<torch::Tensor> shared_value_buffer;

    for (int i = 0; i < max_batch_size; ++i) {
        shared_input_buffer.push_back(torch::zeros({input_planes, board_dim, board_dim}, options_half));
        shared_policy_buffer.push_back(torch::zeros({policy_moves}, options_half));
        shared_value_buffer.push_back(torch::zeros({1}, options_half));
    }

    moodycamel::ConcurrentQueue<std::pair<int, int>> inference_queue;
    std::vector<ThreadSafeQueue<std::vector<int>>> result_queues(num_workers);
    ThreadSafeQueue<int> buffer_free_slots;
    for (int i = 0; i < max_batch_size; ++i) buffer_free_slots.push(i);

    std::vector<int> batcher_cores;
    for (const auto& core : inf_cfg["inference_worker_cores"]) {
        batcher_cores.push_back(core.as<int>());
    }

    std::atomic<uint64_t> dummy_step{0};

    InferenceBatcher batcher(
        engine_path, inference_batch_size, inf_cfg["batch_timeout_ms"].as<int>(), num_workers, 
        run_log_dir, inf_cfg["logging_level"].as<int>(), batcher_cores, 0, dummy_step, inf_cfg["logging_interval_sec"].as<int>()
    );
    std::thread batcher_thread([&]() {
        batcher.run(inference_queue, result_queues, shared_input_buffer, shared_policy_buffer, shared_value_buffer, global_stop_event, &buffer_free_slots);
    });

    chess::Board board;
    board.setFen(chess::constants::STARTPOS);
    std::vector<chess::Board> history;
    
    Logger search_logger("uci_search", run_log_dir, mcts_cfg["logging_level"].as<int>());
    search_logger.rotate(0, 0); 

    Logger quiet_logger("quiet_mcts", run_log_dir, 999); 

    ActionSelectorConfig s_config;
    s_config.node_pool_size = mcts_cfg["node_pool_size"].as<int>();
    s_config.virtual_loss = mcts_cfg["virtual_loss"].as<double>();
    s_config.draw_cutoff = sel_cfg["draw_cutoff"].as<double>();
    s_config.gumbel_c_visit = mcts_cfg["gumbel_c_visit"].as<double>();
    s_config.gumbel_c_scale = mcts_cfg["gumbel_c_scale"].as<double>();
    s_config.gumbel_noise = mcts_cfg["gumbel_noise"].as<double>();
    s_config.gumbel_search_depth = mcts_cfg["gumbel_search_depth"].as<int>();
    s_config.gumbel_m = mcts_cfg["gumbel_m"].as<int>();
    s_config.batch_size_per_worker = mcts_cfg["worker_minibatch_size"].as<int>();
    s_config.temperature_ply_cutoff = sel_cfg["temperature_ply_cutoff"].as<int>();
    s_config.top_move_probability = sel_cfg["top_move_probability"].as<double>();
    s_config.temperature_blunder_q_threshold = sel_cfg["temperature_blunder_q_threshold"].as<double>();
    s_config.temperature_blunder_noise_weight = sel_cfg["temperature_blunder_noise_weight"].as<double>();
    s_config.resignation_probability = sel_cfg["resignation_probability"].as<double>();
    s_config.resignation_cutoff = sel_cfg["resignation_cutoff"].as<double>();

    std::vector<std::unique_ptr<MCTSEngine>> mcts_engines;
    for (int w = 0; w < num_workers; ++w) {
        mcts_engines.push_back(std::make_unique<MCTSEngine>(
            s_config.node_pool_size, s_config.batch_size_per_worker, 
            inference_queue, result_queues[w], w,
            s_config.virtual_loss, s_config.draw_cutoff, 
            s_config.gumbel_c_visit, s_config.gumbel_c_scale, 
            s_config.gumbel_noise, board, history, quiet_logger, 
            shared_input_buffer, shared_policy_buffer, shared_value_buffer,
            m_config, buffer_free_slots
        ));
    }

    std::vector<std::unique_ptr<SearchWorker>> search_workers;
    for (int w = 0; w < num_workers; ++w) {
        auto sw = std::make_unique<SearchWorker>();
        sw->mcts = mcts_engines[w].get();
        sw->core_id = worker_core_ids[w / workers_per_core];

        sw->thread = std::thread([worker = sw.get()]() {
            SetThreadAffinityMask(GetCurrentThread(), (static_cast<DWORD_PTR>(1) << worker->core_id));
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
        search_workers.push_back(std::move(sw));
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    ActionSelector agent("uci_agent", 0, s_config, search_logger);
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
                std::string fen = "";
                for (size_t i = 2; i < moves_idx; ++i) fen += tokens[i] + (i == moves_idx - 1 ? "" : " ");
                board.setFen(fen);
            }

            ply_count = 1;

            // FIX: Removed the forced STARTPOS duplication block.
            // This loop now accurately mirrors data_generator.cpp temporal history sequences.
            for (size_t i = moves_idx + 1; i < tokens.size(); ++i) {
                history.insert(history.begin(), board);
                if (history.size() > 4) history.pop_back();
                
                chess::Move move = chess::uci::uciToMove(board, tokens[i]); 
                board.makeMove(move);
                ply_count++;
            }
        }
        else if (command == "go") {
            int total_search_nodes = s_config.gumbel_search_depth;
            for (size_t i = 1; i < tokens.size() - 1; ++i) {
                if (tokens[i] == "nodes") total_search_nodes = std::stoi(tokens[i + 1]);
            }

            int worker_nodes = total_search_nodes / num_workers;
            if (worker_nodes < 1) worker_nodes = 1;

            search_logger.log("INFO", "Dispatching search to pool. Budget per worker: " + std::to_string(worker_nodes));

            auto search_start_time = std::chrono::steady_clock::now();

            for (int w = 0; w < num_workers; ++w) {
                std::lock_guard<std::mutex> lock(search_workers[w]->mtx);
                search_workers[w]->board = board;
                search_workers[w]->history = history;
                search_workers[w]->search_nodes = worker_nodes;
                search_workers[w]->gumbel_m = s_config.gumbel_m;
                search_workers[w]->done_flag = false;
                search_workers[w]->start_flag = true;
                search_workers[w]->cv_start.notify_one();
            }

            for (int w = 0; w < num_workers; ++w) {
                std::unique_lock<std::mutex> lock(search_workers[w]->mtx);
                search_workers[w]->cv_done.wait(lock, [&]{ return search_workers[w]->done_flag; });
            }

            auto search_end_time = std::chrono::steady_clock::now();
            double duration = std::chrono::duration<double>(search_end_time - search_start_time).count();
            double speed = (duration > 0) ? ((worker_nodes * num_workers) / duration) : 0.0;

            MCTSNode* agg_root = mcts_engines[0]->root;
            for (int i = 0; i < agg_root->num_children; ++i) {
                MCTSNode* agg_child = agg_root->first_child + i;
                for (int w = 1; w < num_workers; ++w) {
                    MCTSNode* w_child = mcts_engines[w]->root->first_child + i;
                    agg_child->visits += w_child->visits;
                    agg_child->value_sum += w_child->value_sum;
                    if (w_child->forced_outcome.has_value()) {
                        if (!agg_child->forced_outcome.has_value()) {
                            agg_child->forced_outcome = w_child->forced_outcome;
                            agg_child->distance_to_mate = w_child->distance_to_mate;
                        } else {
                            if (w_child->forced_outcome.value() == -1 && agg_child->forced_outcome.value() == -1) {
                                agg_child->distance_to_mate = std::min(agg_child->distance_to_mate.value(), w_child->distance_to_mate.value());
                            } else if (w_child->forced_outcome.value() == 1 && agg_child->forced_outcome.value() == 1) {
                                agg_child->distance_to_mate = std::max(agg_child->distance_to_mate.value(), w_child->distance_to_mate.value());
                            } else if (w_child->forced_outcome.value() == -1) {
                                agg_child->forced_outcome = -1;
                                agg_child->distance_to_mate = w_child->distance_to_mate;
                            }
                        }
                    }
                }
            }

            agg_root->visits = 0;
            agg_root->value_sum = 0;
            for (int w = 0; w < num_workers; ++w) {
                agg_root->visits += mcts_engines[w]->root->visits;
                agg_root->value_sum += mcts_engines[w]->root->value_sum;
            }

            double agg_v_mix = agg_root->calculate_v_mix();
            double max_visits = 1.0;
            for (int i = 0; i < agg_root->num_children; ++i) {
                if (agg_root->first_child[i].visits > max_visits) max_visits = agg_root->first_child[i].visits;
            }
            for (int i = 0; i < agg_root->num_children; ++i) {
                agg_root->first_child[i].calculate_gumbel_score(s_config.gumbel_c_visit, s_config.gumbel_c_scale, max_visits, agg_v_mix);
            }

            search_logger.log("INFO", "");
            search_logger.log("INFO", "--- Ensemble Search Results ---");
            
            std::stringstream rss;
            rss << "Tree Stats: Root v_mix=" << std::fixed << std::setprecision(4) << agg_v_mix 
                << " | Sims: " << (worker_nodes * num_workers) << " | Time: " << duration << "s | Speed: " << std::fixed << std::setprecision(1) << speed << " sim/s";
            search_logger.log("INFO", rss.str());

            char table_header[256];
            snprintf(table_header, sizeof(table_header), 
                "%-8s %8s %8s %8s %8s %8s %8s", 
                "Move", "Visits", "Logit", "Raw Q", "Score", "Outcome", "DTM");
            search_logger.log("INFO", table_header);
            search_logger.log("INFO", std::string(70, '-'));

            std::vector<MCTSNode*> children;
            for (int i = 0; i < agg_root->num_children; ++i) {
                children.push_back(agg_root->first_child + i);
            }

            std::sort(children.begin(), children.end(), [](MCTSNode* a, MCTSNode* b) {
                if (a->visits != b->visits) return a->visits > b->visits;
                return a->gumbel_score > b->gumbel_score;
            });

            for (MCTSNode* node : children) {
                if (node->visits == 0) continue; 
                char line[512];
                std::string outcome_str = node->forced_outcome.has_value() ? std::to_string(node->forced_outcome.value()) : "None";
                std::string dtm_str = node->distance_to_mate.has_value() ? std::to_string(node->distance_to_mate.value()) : "None";
                double q_val = (node->visits > 0) ? (-node->value_sum / node->visits) : agg_v_mix;

                snprintf(line, sizeof(line), 
                    "%-8s %8d %8.4f %8.4f %8.4f %8s %8s", 
                    chess::uci::moveToUci(node->move).c_str(), node->visits, node->raw_logit, 
                    q_val, node->gumbel_score, outcome_str.c_str(), dtm_str.c_str()
                );
                search_logger.log("INFO", line);
            }
            search_logger.log("INFO", std::string(70, '-'));
            search_logger.log("INFO", ""); 

            SelectionResult result = agent.select_move(agg_root, ply_count);
            
            std::string best_move_str = (result.best_move == chess::Move::NO_MOVE) ? "0000" : chess::uci::moveToUci(result.best_move);
            
            std::cout << "bestmove " << best_move_str << std::endl;
            main_logger.log("DEBUG", "Engine -> GUI: bestmove " + best_move_str);
        } 
        else if (command == "quit") {
            break;
        }
    }

    main_logger.log("INFO", "Quit command received. Terminating workers...");
    for (int w = 0; w < num_workers; ++w) {
        {
            std::lock_guard<std::mutex> lock(search_workers[w]->mtx);
            search_workers[w]->quit_flag = true;
        }
        search_workers[w]->cv_start.notify_one();
        if (search_workers[w]->thread.joinable()) {
            search_workers[w]->thread.join();
        }
    }

    global_stop_event.store(true);
    if (batcher_thread.joinable()) batcher_thread.join();
    return 0;
}