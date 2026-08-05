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

#include "concurrentqueue.h"
#include <torch/torch.h>
#include <yaml-cpp/yaml.h>

#include "data_generator.hpp"
#include "io_writer.hpp"
#include "board_utils.hpp"
#include "logger.hpp"
#include "inference_batcher.hpp" 
#include "trt_builder.hpp"

namespace fs = std::filesystem;

std::atomic<bool> global_stop_event{false};
std::atomic<bool> trt_update_in_progress{false};

void signal_handler(int signum) {
    global_stop_event.store(true);
}

// Unified LMDB Signal Reader
bool read_lmdb_signal(MDB_env* env, MDB_dbi dbi, const char* key_name, uint64_t& out_val) {
    bool found = false;
    MDB_txn* txn;
    if (mdb_txn_begin(env, nullptr, MDB_RDONLY, &txn) == MDB_SUCCESS) {
        MDB_val key = { strlen(key_name), (void*)key_name }; 
        MDB_val data;
        if (mdb_get(txn, dbi, &key, &data) == MDB_SUCCESS) {
            out_val = *static_cast<uint64_t*>(data.mv_data);
            found = true;
        }
        mdb_txn_commit(txn);
    }
    return found;
}

// Unified LMDB Signal Writer
void write_lmdb_signal(MDB_env* env, MDB_dbi dbi, const char* key_name, uint64_t val) {
    MDB_txn* txn;
    if (mdb_txn_begin(env, nullptr, 0, &txn) == MDB_SUCCESS) {
        MDB_val key = { strlen(key_name), (void*)key_name };
        MDB_val data = { sizeof(uint64_t), &val };
        mdb_put(txn, dbi, &key, &data, 0);
        mdb_txn_commit(txn);
    }
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

    MDB_env* lmdb_env = nullptr;

    const std::string config_file = args["config_file"];
    const std::string model_file = args["model_file"];
    const std::string train_dir = args["train_dir"];
    const std::string db_path = args["db_path"]; 

    YAML::Node config = YAML::LoadFile(config_file);
    YAML::Node model = YAML::LoadFile(model_file);
    
    int main_log_level = config["data_generation"]["main_logging_level"].as<int>();
    Logger main_logger("orchestrator_c", train_dir, main_log_level); 
    const int rot_interval = config["global"]["logging_rotation_steps"].as<int>();
    
    main_logger.log("INFO", "[MAIN] Booting up C++ Engine...");

    auto orch_cores = config["data_generation"]["data_generator_cores"];
    int num_orch_threads = orch_cores.size();
    at::set_num_threads(num_orch_threads); 

    DWORD_PTR mainMask = 0;
    for (const auto& core : orch_cores) {
        mainMask |= (static_cast<DWORD_PTR>(1) << core.as<int>());
    }
    SetThreadAffinityMask(GetCurrentThread(), mainMask);
    SetProcessWorkingSetSize(GetCurrentProcess(),
    (SIZE_T)8ULL * 1024 * 1024 * 1024,
    (SIZE_T)16ULL * 1024 * 1024 * 1024);
    
    const double sampling_ratio = config["training"]["sampling_ratio"].as<double>();
    const size_t batch_size = config["training"]["batch_size"].as<size_t>();
    const size_t flush_threshold = static_cast<size_t>(batch_size / sampling_ratio);

    const int input_planes = model["model"]["input_planes"].as<int>();
    const int board_dim = model["model"]["board_dim"].as<int>(); 
    const int policy_moves = model["model"]["total_policy_moves"].as<int>();
            
    mdb_env_create(&lmdb_env);
    mdb_env_set_mapsize(lmdb_env, (size_t)1024 * 1024 * 1024 * config["global"]["buffer_size_gb"].as<int>()); 
    mdb_env_open(lmdb_env, db_path.c_str(), MDB_NOSYNC | MDB_NOTLS, 0664);

    MDB_dbi shared_dbi;
    MDB_txn* init_txn;
    mdb_txn_begin(lmdb_env, nullptr, 0, &init_txn);
    mdb_dbi_open(init_txn, nullptr, 0, &shared_dbi);
    mdb_txn_commit(init_txn);

    uint64_t init_step = 0;
    size_t init_games = 0, init_samples = 0, write_head = 0, buffer_count = 0, buffer_wraps = 0;
    double init_entropy = 0.0;

    MDB_txn* txn;
    mdb_txn_begin(lmdb_env, nullptr, MDB_RDONLY, &txn);

    MDB_val py_key = { 10, (void*)"__PY_STATE" };
    MDB_val py_data;
    if (mdb_get(txn, shared_dbi, &py_key, &py_data) == MDB_SUCCESS) {
        PyState* s = static_cast<PyState*>(py_data.mv_data);
        init_step = s->training_steps;
    }

    MDB_val cpp_key = { 11, (void*)"__CPP_STATE" };
    MDB_val cpp_data;
    if (mdb_get(txn, shared_dbi, &cpp_key, &cpp_data) == MDB_SUCCESS) {
        CppState* s = static_cast<CppState*>(cpp_data.mv_data);
        init_games = s->games_played;
        init_samples = s->samples_generated;
        init_entropy = s->lifetime_entropy;
        write_head = s->buffer_head_ptr;
        buffer_count = s->buffer_count;
        buffer_wraps = s->buffer_wraps;
    }
    mdb_txn_commit(txn);

    std::atomic<uint64_t> current_step(init_step);
    std::atomic<size_t> atomic_write_head(write_head);
    std::atomic<size_t> atomic_buffer_count(buffer_count);
    std::atomic<size_t> atomic_buffer_wraps(buffer_wraps);
    
    main_logger.rotate(init_step, rot_interval);
    main_logger.log("INFO", "=== C++ ORCHESTRATOR STARTED ===");
    
    int num_cores = config["data_generation"]["game_worker_cores"].size();
    int workers_per_core = config["data_generation"]["workers_per_core"].as<int>();
    int inference_batch_size = config["inference"]["batch_size"].as<int>();
    int batch_factor = config["inference"]["batch_size_factor"].as<int>();
    
    int num_workers = num_cores * workers_per_core;
    int total_slots = inference_batch_size * batch_factor;

    moodycamel::ConcurrentQueue<std::pair<int, int>> inference_queue;
    ThreadSafeQueue<CompletedGame> completed_games_queue;

    auto options_half = torch::TensorOptions().dtype(torch::kHalf).device(torch::kCPU);
    std::vector<torch::Tensor> shared_input_buffer;
    std::vector<torch::Tensor> shared_policy_buffer;
    std::vector<torch::Tensor> shared_value_buffer;

    for (int i = 0; i < total_slots; ++i) {
        shared_input_buffer.push_back(torch::zeros({input_planes, board_dim, board_dim}, options_half));
        shared_policy_buffer.push_back(torch::zeros({policy_moves}, options_half));
        shared_value_buffer.push_back(torch::zeros({3}, options_half));
    }

    std::vector<ThreadSafeQueue<std::vector<int>>> result_queues(num_workers);
    ThreadSafeQueue<int> buffer_free_slots;
    for (int i = 0; i < total_slots; ++i) buffer_free_slots.push(i);

    std::string onnx_path = config["global"]["model_path"].as<std::string>() + ".onnx";
    std::string engine_path = config["global"]["model_path"].as<std::string>() + ".engine";

    main_logger.log("INFO", "[MAIN] Synchronizing with LMDB TRT IPC Flags...");
    
    uint64_t export_signal = 0;
    bool has_export = read_lmdb_signal(lmdb_env, shared_dbi, "__TRT_EXPORT_SIGNAL", export_signal);
    
    uint64_t ready_signal = 0;
    bool has_ready = read_lmdb_signal(lmdb_env, shared_dbi, "__TRT_ENGINE_READY", ready_signal);

    uint64_t last_handled_export_step = has_ready ? ready_signal : 0; 
    bool needs_initial_build = false;

    if (has_export) {
        if (!has_ready || ready_signal < export_signal) {
            main_logger.log("INFO", "Export signal (" + std::to_string(export_signal) + 
                            ") exceeds Ready signal. Forcing Synchronous Initial TRT build.");
            needs_initial_build = true;
        }
    } else {
        throw std::runtime_error("Fatal: __TRT_EXPORT_SIGNAL missing in DB. Python failed to seed protocol.");
    }

    if (needs_initial_build) {
        main_logger.log("INFO", "[BUILD] TensorRT is cooking. Terminal will be silent for ~60s...");
        auto result = TRTBuilder::build_engine(onnx_path, inference_batch_size, input_planes, main_logger);
        if (result) {
            TRTBuilder::save_engine(*result, engine_path);
            write_lmdb_signal(lmdb_env, shared_dbi, "__TRT_ENGINE_READY", export_signal);
            last_handled_export_step = export_signal;
            main_logger.log("INFO", "[BUILD] Initial Engine Build complete. IPC Ready signal written.");
        } else {
            throw std::runtime_error("Initial TRT Build returned null result.");
        }
    }

    std::vector<int> batcher_cores;
    for (const auto& core : config["inference"]["inference_worker_cores"]) {
        batcher_cores.push_back(core.as<int>());
    }
    
    int batcher_log_level = config["inference"]["logging_level"].as<int>();
    int log_interval_sec = config["inference"]["logging_interval_sec"].as<int>();
    std::string model_path = config["global"]["model_path"].as<std::string>() + ".pt";
    int batch_timeout = config["inference"]["batch_timeout_ms"].as<int>();

    main_logger.log("INFO", "[MAIN] Initializing Batcher...");
    InferenceBatcher batcher(
        model_path, inference_batch_size, batch_timeout, num_workers,
        train_dir, batcher_log_level, batcher_cores, rot_interval, current_step, log_interval_sec
    );

    std::thread batcher_thread([&]() {
        batcher.run(inference_queue, result_queues, shared_input_buffer, shared_policy_buffer, shared_value_buffer, global_stop_event, &buffer_free_slots);
    });

    main_logger.log("INFO", "[MAIN] Initializing Data Generator Workers...");
    DataGenerator generator(
        config["global"], config["data_generation"], config["mcts"], config["selection"],
        model, train_dir, rot_interval, main_logger,
        inference_queue, result_queues, shared_input_buffer, shared_policy_buffer, shared_value_buffer,
        buffer_free_slots, completed_games_queue, init_games + 1, current_step
    );

    std::vector<int> io_write_cores;
    for (const auto& core : config["data_generation"]["io_write_cores"]) {
        io_write_cores.push_back(core.as<int>());
    }

    main_logger.log("INFO", "[MAIN] Initializing I/O Writer...");
    IOWriter io_writer(
        lmdb_env, config["global"]["min_buffer_size"].as<size_t>(), config["global"]["max_buffer_size"].as<size_t>(),
        config["global"]["buffer_ramp_steps"].as<int>(), io_write_cores, train_dir, flush_threshold,
        config["data_generation"]["io_logging_level"].as<int>(), rot_interval, 
        ModelConfig{input_planes, board_dim, policy_moves}, completed_games_queue,
        current_step, atomic_write_head, atomic_buffer_count, atomic_buffer_wraps,
        init_games, init_samples, init_entropy
    );

    main_logger.log("INFO", "[MAIN] System successfully booted. Handing control to monitor loop.");
    io_writer.start();
    generator.start();

    while (!global_stop_event.load()) {
        std::this_thread::sleep_for(std::chrono::seconds(1));
        
        uint64_t new_step = current_step.load(std::memory_order_relaxed);
        mdb_txn_begin(lmdb_env, nullptr, MDB_RDONLY, &txn);
        if (mdb_get(txn, shared_dbi, &py_key, &py_data) == MDB_SUCCESS) {
            PyState* s = static_cast<PyState*>(py_data.mv_data);
            new_step = s->training_steps;
            if (new_step > current_step.load(std::memory_order_relaxed)) {
                current_step.store(new_step, std::memory_order_relaxed);
                main_logger.rotate(new_step, rot_interval);
            }
        }
        mdb_txn_commit(txn);

        uint64_t exported_step = 0;
        if (read_lmdb_signal(lmdb_env, shared_dbi, "__TRT_EXPORT_SIGNAL", exported_step)) {
            if (exported_step > last_handled_export_step && !trt_update_in_progress.load()) {
                last_handled_export_step = exported_step;
                trt_update_in_progress.store(true);

                main_logger.log("INFO", "ONNX export signal received for step " + 
                                std::to_string(exported_step) + ". Initiating full engine rebuild...");

                uint64_t current_train_step = current_step.load(std::memory_order_relaxed);
                uint64_t current_export_signal = exported_step;

                std::thread([&batcher, &main_logger, onnx_path, engine_path, inference_batch_size, 
                                lmdb_env, shared_dbi, current_train_step, current_export_signal, input_planes]() {
                    
                    auto start_time = std::chrono::steady_clock::now();
                    
                    try {
                        // Request pause and wait for pipeline drain
                        main_logger.log("INFO", "Initiating Synchronous Pipeline Drain...");
                        batcher.request_pause();
                        
                        while (!batcher.is_fully_paused() && !global_stop_event.load()) {
                            std::this_thread::sleep_for(std::chrono::milliseconds(5));
                        }
                        
                        if (global_stop_event.load()) {
                            batcher.cancel_pause();
                            trt_update_in_progress.store(false);
                            return;
                        }
                        
                        main_logger.log("INFO", "Pipeline drained.");
                        main_logger.log("INFO", "Executing FULL ENGINE REBUILD (this will take ~60s)...");

                        auto result = TRTBuilder::build_engine(
                            onnx_path, 
                            inference_batch_size,
                            input_planes, 
                            main_logger
                        );
                        
                        if (result) {
                            batcher.signal_trt_reload(result->serialized_data, static_cast<int>(current_train_step));
                            
                            while (batcher.is_fully_paused() && !global_stop_event.load()) {
                                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                            }
                            
                            TRTBuilder::save_engine(*result, engine_path);
                            
                            double duration = std::chrono::duration<double>(
                                std::chrono::steady_clock::now() - start_time
                            ).count();
                            main_logger.log("INFO", "Full rebuild completed in " + 
                                            std::to_string(duration) + " seconds.");

                            write_lmdb_signal(lmdb_env, shared_dbi, "__TRT_ENGINE_READY", current_export_signal);
                        } else {
                            main_logger.log("CRITICAL", "Full rebuild FAILED. Resuming with old engine.");
                            batcher.cancel_pause();
                        }
                        
                    } catch (const std::exception& e) {
                        main_logger.log("CRITICAL", std::string("TRT Update Failed: ") + e.what());
                        batcher.cancel_pause();
                    }
                    
                    trt_update_in_progress.store(false);
                }).detach();
            }
        }
    }

    generator.stop();
    io_writer.stop();
    if (batcher_thread.joinable()) batcher_thread.join();

    if (lmdb_env) mdb_env_close(lmdb_env);
    return 0;
}