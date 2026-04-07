#pragma once

#include <torch/torch.h>
#include <NvInfer.h>
#include <vector>
#include <string>
#include <atomic>
#include <optional>
#include <chrono>
#include <mutex>
#include <c10/cuda/CUDAStream.h>
#include "logger.hpp"
#include "mcts_engine.hpp" 
#include "concurrentqueue.h"

class InferenceBatcher {
private:
    std::string model_path; 
    int batch_size;
    int timeout_ms;
    int num_workers;
    
    std::string rl_dir;
    int logging_level;
    std::vector<int> core_ids;
    
    int rotation_interval;
    std::atomic<int> current_global_step;
    int logging_interval_sec;

    torch::Device device;

    nvinfer1::IRuntime* trt_runtime = nullptr;
    nvinfer1::ICudaEngine* trt_engine = nullptr;
    nvinfer1::IExecutionContext* trt_contexts[3] = {nullptr, nullptr, nullptr};

    std::optional<c10::cuda::CUDAStream> stream_a;
    std::optional<c10::cuda::CUDAStream> stream_b;
    
    std::atomic<bool> pending_trt_reload{false}; 
    std::vector<uint8_t> pending_engine_data;
    std::mutex reload_mutex;

    // --- NEW: Deterministic Drain Handshake ---
    std::atomic<bool> pause_requested{false};
    std::atomic<bool> is_paused{false};
    // ------------------------------------------

    double interval_total_processing_duration = 0.0;
    int interval_batches_processed = 0;
    int interval_total_inferences = 0;
    std::chrono::steady_clock::time_point last_report_time;

    void load_initial_engine(Logger& logger);

    std::vector<std::pair<int, int>> collect_batch(
        moodycamel::ConcurrentQueue<std::pair<int, int>>& queue,
        Logger& logger,
        std::atomic<bool>& stop_event
    );

public:
    InferenceBatcher(
        const std::string& path, int b_size, int timeout, int workers, 
        const std::string& rl_dir, int log_level, const std::vector<int>& cores,
        int rot_interval, int initial_step, int log_interval_sec
    );

    ~InferenceBatcher();

    void signal_trt_reload(const std::vector<uint8_t>& engine_data, int new_step) { 
        std::lock_guard<std::mutex> lock(reload_mutex);
        current_global_step.store(new_step);
        pending_engine_data = engine_data;
        pending_trt_reload.store(true); 
    }

    // --- NEW: Handshake Methods ---
    void request_pause() { pause_requested.store(true); }
    bool is_fully_paused() const { return is_paused.load(); }
    void cancel_pause() { 
        pause_requested.store(false); 
        is_paused.store(false); 
    }
    // ------------------------------

    void run(
        moodycamel::ConcurrentQueue<std::pair<int, int>>& queue,    
        std::vector<ThreadSafeQueue<std::vector<int>>>& result_queues,
        std::vector<torch::Tensor>& shared_input_buffer,
        std::vector<torch::Tensor>& shared_policy_buffer,
        std::vector<torch::Tensor>& shared_value_buffer,
        std::atomic<bool>& stop_event
    );
};