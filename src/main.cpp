#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include <iostream>
#include <vector>
#include <thread>
#include <atomic>
#include <string>
#include <chrono>
#include <csignal>
#include <unordered_map>
#include <stdexcept>
#include <filesystem>

#include <torch/torch.h>
#include <yaml-cpp/yaml.h>

#include "data_generator.hpp"
#include "board_utils.hpp"
#include "logger.hpp"
#include "inference_batcher.hpp" 
#include "trt_builder.hpp"

namespace fs = std::filesystem;

std::atomic<bool> global_stop_event{false};
std::atomic<bool> trt_build_in_progress{false};

void signal_handler(int signum) {
    global_stop_event.store(true);
}

int main(int argc, char* argv[]) {
    std::signal(SIGINT, signal_handler);

    std::unordered_map<std::string, std::string> args;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg.substr(0, 2) == "--" && i + 1 < argc) {
            args[arg.substr(2)] = argv[i + 1];
            i++;
        }
    }

    try {
        const std::string config_file = args["config_file"];
        const std::string model_file = args["model_file"];
        YAML::Node config = YAML::LoadFile(config_file);
        YAML::Node model = YAML::LoadFile(model_file);
        
        auto orch_cores = config["data_generation"]["data_generator_cores"];
        int num_orch_threads = orch_cores.size();
        at::set_num_threads(num_orch_threads); 

        DWORD_PTR mainMask = 0;
        for (const auto& core : orch_cores) {
            mainMask |= (static_cast<DWORD_PTR>(1) << core.as<int>());
        }
        SetThreadAffinityMask(GetCurrentThread(), mainMask);
        
        const std::string rl_dir     = args["rl_dir"];
        const std::string state_file = args["state_file"];
        const std::string db_path    = args["db_path"]; 
        const int rot_interval       = config["global"]["logging_rotation_steps"].as<int>();
        const size_t max_buffer_size = config["global"]["max_buffer_size"].as<size_t>();

        const int input_planes = model["model"]["input_planes"].as<int>();
        const int board_dim = model["chess"]["board_dim"].as<int>(); 
        const int policy_moves = model["chess"]["total_policy_moves"].as<int>();
                
        int main_log_level = config["data_generation"]["main_logging_level"].as<int>();
        Logger main_logger("orchestrator_c", rl_dir, main_log_level); 
        
        int current_step = get_step_from_yaml(state_file, 0);
        main_logger.rotate(current_step, rot_interval);
        
        main_logger.log("INFO", "=== C++ ORCHESTRATOR STARTED ===");
        
        std::string core_list = "";
        for(const auto& core : orch_cores) core_list += std::to_string(core.as<int>()) + " ";
        main_logger.log("INFO", "Orchestrator pinned to core(s): " + core_list);

        int num_cores = config["data_generation"]["game_worker_cores"].size();
        int workers_per_core = config["data_generation"]["workers_per_core"].as<int>();
        int inference_batch_size = config["inference"]["batch_size"].as<int>();
        int batch_factor = config["inference"]["batch_size_factor"].as<int>();
        
        int num_workers = num_cores * workers_per_core;
        int total_slots = inference_batch_size * batch_factor;

        // --- SHARDED QUEUE INITIALIZATION ---
        int workers_per_queue = config["data_generation"]["workers_per_queue"].as<int>();
        int num_shards = (num_workers + workers_per_queue - 1) / workers_per_queue;
        main_logger.log("INFO", "Allocating " + std::to_string(num_shards) + " inference queue shards (" + std::to_string(workers_per_queue) + " workers/shard).");

        std::vector<ThreadSafeQueue<std::vector<std::pair<int, int>>>*> inference_shards;
        for (int i = 0; i < num_shards; ++i) {
            inference_shards.push_back(new ThreadSafeQueue<std::vector<std::pair<int, int>>>());
        }

        main_logger.log("INFO", "Allocating " + std::to_string(total_slots) + " shared memory slots.");

        auto options_half = torch::TensorOptions().dtype(torch::kHalf).device(torch::kCPU);

        std::vector<torch::Tensor> shared_input_buffer;
        std::vector<torch::Tensor> shared_policy_buffer;
        std::vector<torch::Tensor> shared_value_buffer;

        for (int i = 0; i < total_slots; ++i) {
            shared_input_buffer.push_back(torch::zeros({input_planes, board_dim, board_dim}, options_half));
            shared_policy_buffer.push_back(torch::zeros({policy_moves}, options_half));
            shared_value_buffer.push_back(torch::zeros({1}, options_half));
        }

        std::vector<ThreadSafeQueue<std::vector<int>>> result_queues(num_workers);
        ThreadSafeQueue<int> buffer_free_slots;
        for (int i = 0; i < total_slots; ++i) buffer_free_slots.push(i);

        std::atomic<size_t> write_head(0);
        std::atomic<size_t> buffer_count(0);

        int trt_build_interval = config["global"]["new_model_interval_steps"].as<int>();
        std::string onnx_path = config["global"]["model_path"].as<std::string>() + ".onnx";
        std::string engine_path = config["global"]["model_path"].as<std::string>() + ".engine";
        int last_build_step = current_step;

        if (!fs::exists(engine_path) && fs::exists(onnx_path)) {
            main_logger.log("CRITICAL", "Cold start: No Engine found. Starting synchronous TRT build...");
            std::cout << "[BUILD] TensorRT is cooking. Terminal will be silent for ~60s..." << std::endl;
            
            try {
                auto result = TRTBuilder::build_engine(onnx_path, inference_batch_size, main_logger);
                if (result) {
                    TRTBuilder::save_engine(*result, engine_path);
                    main_logger.log("INFO", "Initial TRT Engine deployed and saved.");
                } else {
                    main_logger.log("CRITICAL", "Initial TRT Build returned null result. Aborting.");
                    return 1;
                }
            } catch (const std::exception& e) {
                main_logger.log("CRITICAL", std::string("Initial TRT Build Failed: ") + e.what());
                return 1;
            }
        } else if (!fs::exists(engine_path) && !fs::exists(onnx_path)) {
            main_logger.log("CRITICAL", "Fatal: Neither ONNX nor Engine file exists. Cannot start.");
            return 1;
        }

        std::vector<int> batcher_cores;
        for (const auto& core : config["inference"]["inference_worker_cores"]) {
            batcher_cores.push_back(core.as<int>());
        }
        
        int batcher_log_level = config["inference"]["logging_level"].as<int>();
        int log_interval_sec = config["inference"]["logging_interval_sec"].as<int>();
        std::string model_path = config["global"]["model_path"].as<std::string>() + ".pt";
        int batch_timeout = config["inference"]["batch_timeout_ms"].as<int>();

        InferenceBatcher batcher(
            model_path, 
            inference_batch_size, 
            batch_timeout, 
            num_workers, 
            rl_dir, 
            batcher_log_level, 
            batcher_cores,
            rot_interval,
            current_step,
            log_interval_sec
        );

        std::thread batcher_thread([&]() {
            try {
                batcher.run(
                    inference_shards, result_queues,
                    shared_input_buffer, shared_policy_buffer, shared_value_buffer,
                    global_stop_event
                );
                std::this_thread::sleep_for(std::chrono::seconds(1));
            } catch (const std::exception& e) {
                main_logger.log("CRITICAL", std::string("Batcher Thread Fatal Error: ") + e.what());
                global_stop_event.store(true);
            }
        });

        DataGenerator generator(
            config["data_generation"], config["mcts"], config["selection"],
            model, rl_dir, state_file, db_path, rot_interval,
            main_logger,
            inference_shards, result_queues,
            shared_input_buffer, shared_policy_buffer, shared_value_buffer,
            buffer_free_slots, write_head, buffer_count, max_buffer_size
        );
        generator.start();

        main_logger.log("INFO", "Monitor loop live. TRT background build triggers every " + std::to_string(trt_build_interval) + " steps.");

        while (!global_stop_event.load()) {
            std::this_thread::sleep_for(std::chrono::seconds(1));
            
            int new_step = get_step_from_yaml(state_file, current_step);
            
            if (new_step >= (last_build_step + trt_build_interval) && !trt_build_in_progress.load()) {
                main_logger.log("CRITICAL", "Step threshold reached (" + 
                    std::to_string(new_step) + "). Launching Background TRT Cook...");
                
                last_build_step = new_step;
                trt_build_in_progress.store(true);

                std::thread([&batcher, &main_logger, onnx_path, new_step, inference_batch_size]() {
                    try {
                        if (fs::exists(onnx_path)) {
                            auto result = TRTBuilder::build_engine(onnx_path, inference_batch_size, main_logger);
                            if (result) {
                                batcher.signal_trt_reload(result->serialized_data, new_step);
                                
                                std::string save_path = onnx_path;
                                save_path.replace(save_path.find(".onnx"), 5, ".engine");
                                TRTBuilder::save_engine(*result, save_path);
                                
                                main_logger.log("INFO", "New TRT Engine deployed and saved for step " + std::to_string(new_step));
                            } else {
                                main_logger.log("CRITICAL", "TRT Build returned null result.");
                            }
                        } else {
                            main_logger.log("CRITICAL", "TRT Build skipped: ONNX file not found at " + onnx_path);
                        }
                    } catch (const std::exception& e) {
                        main_logger.log("CRITICAL", std::string("TRT Background Build Failed: ") + e.what());
                    }
                    trt_build_in_progress.store(false);
                }).detach(); 
            }

            if (new_step > current_step) {
                current_step = new_step;
                main_logger.rotate(current_step, rot_interval);
            }

            main_logger.log("INFO", "Orchestrator Heartbeat | LMDB Buffer: " + 
                std::to_string(buffer_count.load()) + " | Global Step: " + std::to_string(current_step));
        }

        generator.stop();
        if (batcher_thread.joinable()) batcher_thread.join();
        
        for (auto shard : inference_shards) delete shard;
        
        main_logger.log("INFO", "=== C++ ORCHESTRATOR SHUTDOWN COMPLETE ===");

    } catch (const std::exception& e) {
        std::cerr << "FATAL ERROR: " << e.what() << std::endl;
        return 1;
    }
    return 0;
}