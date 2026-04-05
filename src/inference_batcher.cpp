#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include "inference_batcher.hpp"
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <thread>
#include <fstream>

class BatcherTRTLogger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::cerr << "[Batcher TRT] " << msg << std::endl;
        }
    }
} gBatcherLogger;

struct PipelineJob {
    int slot;
    std::vector<std::pair<int, int>> requests;
    std::chrono::steady_clock::time_point batch_start_time;
};

InferenceBatcher::InferenceBatcher(
    const std::string& path, int b_size, int timeout, int workers, 
    const std::string& r_dir, int log_level, const std::vector<int>& cores,
    int rot_interval, int initial_step, int log_interval_sec
) : model_path(path), batch_size(b_size), timeout_ms(timeout), 
    num_workers(workers), rl_dir(r_dir), logging_level(log_level), core_ids(cores),
    rotation_interval(rot_interval), current_global_step(initial_step), 
    logging_interval_sec(log_interval_sec),
    device(torch::kCUDA) 
{
    if (model_path.find(".pt") != std::string::npos) {
        model_path.replace(model_path.find(".pt"), 3, ".engine");
    }
    stream_a = c10::cuda::getStreamFromPool();
    stream_b = c10::cuda::getStreamFromPool();
}

InferenceBatcher::~InferenceBatcher() {
    for (int i = 0; i < 3; ++i) {
        if (trt_contexts[i]) delete trt_contexts[i];
    }
    if (trt_engine) delete trt_engine;
    if (trt_runtime) delete trt_runtime;
}

void InferenceBatcher::load_initial_engine(Logger& logger) {
    std::ifstream file(model_path, std::ios::binary | std::ios::ate);
    if (!file) {
        logger.log("WARNING", "No initial TRT engine found at " + model_path + ". Waiting for background build...");
        return;
    }

    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::vector<uint8_t> buffer(size);
    if (file.read(reinterpret_cast<char*>(buffer.data()), size)) {
        if (!trt_runtime) trt_runtime = nvinfer1::createInferRuntime(gBatcherLogger);
        trt_engine = trt_runtime->deserializeCudaEngine(buffer.data(), buffer.size());
        
        for (int i = 0; i < 3; ++i) {
            trt_contexts[i] = trt_engine->createExecutionContext();
        }
        logger.log("INFO", "Initial TRT Engine successfully loaded from disk.");
    }
}

std::vector<std::pair<int, int>> InferenceBatcher::collect_batch(
    std::vector<ThreadSafeQueue<std::vector<std::pair<int, int>>>*>& shards,
    Logger& logger,
    std::atomic<bool>& stop_event
) {
    std::vector<std::pair<int, int>> requests;
    requests.reserve(batch_size);
    auto start_poll = std::chrono::steady_clock::now();
    static int last_shard = 0;

    while ((int)requests.size() < batch_size) {
        if (stop_event.load()) break;
        bool activity = false;
        
        for (size_t i = 0; i < shards.size(); ++i) {
            int idx = (last_shard + i) % shards.size();
            std::vector<std::pair<int, int>> incoming;

            if (shards[idx]->try_pop(incoming)) {
                activity = true;
                size_t space = (size_t)batch_size - requests.size();
                
                if (incoming.size() <= space) {
                    requests.insert(requests.end(), incoming.begin(), incoming.end());
                } else {
                    requests.insert(requests.end(), incoming.begin(), incoming.begin() + space);
                    std::vector<std::pair<int, int>> left(incoming.begin() + space, incoming.end());
                    shards[idx]->push(std::move(left));
                }
                
                if ((int)requests.size() >= batch_size) {
                    last_shard = (idx + 1) % shards.size();
                    return requests;
                }
            }
        }

        if (!requests.empty()) {
            double elapsed_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start_poll).count();
            if (elapsed_ms >= (double)timeout_ms) break;
        }

        if (!activity) {
            _mm_pause();
        }
    }
    return requests;
}

void InferenceBatcher::run(
    std::vector<ThreadSafeQueue<std::vector<std::pair<int, int>>>*>& shards,
    std::vector<ThreadSafeQueue<std::vector<int>>>& result_queues,
    std::vector<torch::Tensor>& shared_input_buffer,
    std::vector<torch::Tensor>& shared_policy_buffer,
    std::vector<torch::Tensor>& shared_value_buffer,
    std::atomic<bool>& stop_event
) {
    at::set_num_threads(1);

    Logger logger("inference_batcher", rl_dir, logging_level);
    logger.rotate(current_global_step.load(), rotation_interval);
    logger.log("INFO", "=== INFERENCE BATCHER STARTED ===");
    
    DWORD_PTR frontendMask = 0;
    DWORD_PTR backendMask = 0;
    DWORD_PTR fillerMask = 0;

    if (core_ids.size() >= 2) {
        frontendMask = (static_cast<DWORD_PTR>(1) << core_ids[0]);
        backendMask = (static_cast<DWORD_PTR>(1) << core_ids[1]);
        fillerMask = frontendMask; 
        
        if (core_ids.size() >= 3) {
            fillerMask = (static_cast<DWORD_PTR>(1) << core_ids[2]);
            logger.log("INFO", "Dispatcher pinned to core " + std::to_string(core_ids[0]) + 
                               ", Collector pinned to core " + std::to_string(core_ids[1]) + 
                               ", Filler pinned to core " + std::to_string(core_ids[2]));
        } else {
            logger.log("INFO", "Dispatcher and Filler sharing core " + std::to_string(core_ids[0]) + 
                               ", Collector pinned to core " + std::to_string(core_ids[1]));
        }
    } else {
        logger.log("CRITICAL", "Insufficient cores provided for pipelined batcher.");
        return;
    }

    SetThreadAffinityMask(GetCurrentThread(), frontendMask);
    load_initial_engine(logger);

    auto input_shape = shared_input_buffer[0].sizes();
    std::vector<int64_t> batch_shape = {batch_size};
    batch_shape.insert(batch_shape.end(), input_shape.begin(), input_shape.end());

    auto policy_shape = shared_policy_buffer[0].sizes();
    std::vector<int64_t> policy_batch_shape = {batch_size, policy_shape[0]};
    auto value_shape = shared_value_buffer[0].sizes();
    std::vector<int64_t> value_batch_shape = {batch_size, value_shape[0]};

    auto pinned_opts = torch::TensorOptions().dtype(torch::kHalf).pinned_memory(true);
    auto gpu_opts_fp16 = torch::TensorOptions().dtype(torch::kHalf).device(device);

    const int NUM_SLOTS = 3;
    
    torch::Tensor pinned_staging[NUM_SLOTS] = { torch::empty(batch_shape, pinned_opts), torch::empty(batch_shape, pinned_opts), torch::empty(batch_shape, pinned_opts) };
    torch::Tensor policy_cpu[NUM_SLOTS]     = { torch::empty(policy_batch_shape, pinned_opts), torch::empty(policy_batch_shape, pinned_opts), torch::empty(policy_batch_shape, pinned_opts) };
    torch::Tensor value_cpu[NUM_SLOTS]      = { torch::empty(value_batch_shape, pinned_opts), torch::empty(value_batch_shape, pinned_opts), torch::empty(value_batch_shape, pinned_opts) };

    torch::Tensor gpu_policy_fp16[NUM_SLOTS] = { torch::empty(policy_batch_shape, gpu_opts_fp16), torch::empty(policy_batch_shape, gpu_opts_fp16), torch::empty(policy_batch_shape, gpu_opts_fp16) };
    torch::Tensor gpu_value_fp16[NUM_SLOTS]  = { torch::empty(value_batch_shape, gpu_opts_fp16), torch::empty(value_batch_shape, gpu_opts_fp16), torch::empty(value_batch_shape, gpu_opts_fp16) };
    torch::Tensor gpu_input_fp16[NUM_SLOTS]  = { torch::empty(batch_shape, gpu_opts_fp16), torch::empty(batch_shape, gpu_opts_fp16), torch::empty(batch_shape, gpu_opts_fp16) };

    size_t input_bytes_per_tensor = shared_input_buffer[0].numel() * sizeof(uint16_t);
    size_t policy_bytes_per_tensor = shared_policy_buffer[0].numel() * sizeof(uint16_t);
    size_t value_bytes_per_tensor = shared_value_buffer[0].numel() * sizeof(uint16_t);

    size_t input_numel = shared_input_buffer[0].numel();
    size_t policy_numel = shared_policy_buffer[0].numel();
    size_t value_numel = shared_value_buffer[0].numel();

    c10::cuda::CUDAStream* streams[NUM_SLOTS] = {
        &stream_a.value(), 
        &stream_b.value(), 
        new c10::cuda::CUDAStream(c10::cuda::getStreamFromPool())
    };
    cudaStream_t raw_streams[NUM_SLOTS] = {streams[0]->stream(), streams[1]->stream(), streams[2]->stream()};
    
    cudaEvent_t fw_start_events[NUM_SLOTS], fw_stop_events[NUM_SLOTS], compute_start[NUM_SLOTS], compute_stop[NUM_SLOTS];
    for (int i = 0; i < NUM_SLOTS; ++i) {
        cudaEventCreate(&fw_start_events[i]); cudaEventCreate(&fw_stop_events[i]);
        cudaEventCreate(&compute_start[i]); cudaEventCreate(&compute_stop[i]);
    }

    ThreadSafeQueue<PipelineJob> dispatch_queue;
    ThreadSafeQueue<PipelineJob> scatter_queue;
    std::atomic<bool> slot_free[NUM_SLOTS] = {true, true, true};

    std::thread collector_thread([&]() {
        SetThreadAffinityMask(GetCurrentThread(), backendMask);
        last_report_time = std::chrono::steady_clock::now();
        while (true) {
            PipelineJob job;
            if (scatter_queue.try_pop(job)) {
                cudaStreamSynchronize(raw_streams[job.slot]); 
                
                if (logging_level <= 10) {
                    float total_ms = 0.0f, compute_ms = 0.0f;
                    cudaEventElapsedTime(&total_ms, fw_start_events[job.slot], fw_stop_events[job.slot]);
                    cudaEventElapsedTime(&compute_ms, compute_start[job.slot], compute_stop[job.slot]);
                    logger.log("DEBUG", "[Slot " + std::to_string(job.slot) + "] GPU Math: " + std::to_string(compute_ms) + "ms | GPU Total: " + std::to_string(total_ms) + "ms");
                }

                c10::Half* src_policy_base = policy_cpu[job.slot].data_ptr<c10::Half>();
                c10::Half* src_value_base = value_cpu[job.slot].data_ptr<c10::Half>();
                std::vector<std::vector<int>> worker_notifications(num_workers);

                for (size_t i = 0; i < job.requests.size(); ++i) {
                    int w_id = job.requests[i].first;
                    int s_idx = job.requests[i].second;

                    std::memcpy(shared_policy_buffer[s_idx].data_ptr<c10::Half>(), src_policy_base + (i * policy_numel), policy_bytes_per_tensor);
                    std::memcpy(shared_value_buffer[s_idx].data_ptr<c10::Half>(), src_value_base + (i * value_numel), value_bytes_per_tensor);
                    worker_notifications[w_id].push_back(s_idx);
                }
                for (int w_id = 0; w_id < num_workers; ++w_id) {
                    if (!worker_notifications[w_id].empty()) result_queues[w_id].push(worker_notifications[w_id]);
                }

                interval_batches_processed++;
                interval_total_processing_duration += std::chrono::duration<double>(std::chrono::steady_clock::now() - job.batch_start_time).count();
                interval_total_inferences += (int)job.requests.size();
                
                slot_free[job.slot].store(true); 
            } else if (stop_event.load()) break;
            else _mm_pause();

            auto current_time = std::chrono::steady_clock::now();
            double elapsed_interval_time = std::chrono::duration<double>(current_time - last_report_time).count();

            if (elapsed_interval_time >= (double)logging_interval_sec) {
                if (logging_level <= 20) {
                    char buffer[256];
                    snprintf(buffer, sizeof(buffer), "--- Inference Batcher Performance Report (%.2fs interval) ---", elapsed_interval_time);
                    logger.log("INFO", buffer);

                    if (interval_batches_processed > 0) {
                        double util_pct = (interval_total_processing_duration / NUM_SLOTS / elapsed_interval_time) * 100.0;
                        logger.log("INFO", "  Batches processed:        " + std::to_string(interval_batches_processed));
                        logger.log("INFO", "  Total inferences:         " + std::to_string(interval_total_inferences));
                        logger.log("INFO", "  Avg batch process time:   " + std::to_string(interval_total_processing_duration / interval_batches_processed) + "s");
                        logger.log("INFO", "  Overall Inferences/Sec:   " + std::to_string(interval_total_inferences / elapsed_interval_time));
                        logger.log("INFO", "  Batcher Utilization:      " + std::to_string(util_pct) + "%");
                    }
                }
                last_report_time = current_time;
                interval_batches_processed = 0;
                interval_total_processing_duration = 0.0;
                interval_total_inferences = 0;
            }
        }
    });

    std::thread filler_thread([&]() {
        SetThreadAffinityMask(GetCurrentThread(), fillerMask);
        int current_slot = 0;
        
        while (!stop_event.load()) {
            if (pending_trt_reload.load()) {
                std::this_thread::sleep_for(std::chrono::microseconds(100));
                continue;
            }

            while (!slot_free[current_slot].load() && !stop_event.load()) {
                _mm_pause();
            }
            if (stop_event.load()) break;

            auto requests = collect_batch(shards, logger, stop_event);
            if (requests.empty()) continue;

            auto batch_start = std::chrono::steady_clock::now();
            int req_size = (int)requests.size();

            slot_free[current_slot].store(false); 

            // --- QUEUE LOGGING ADDED HERE ---
            if (logging_level <= 10) {
                std::ostringstream q_log;
                q_log << "Batch Size: " << req_size << " | Shard Backlogs: [";
                for (size_t i = 0; i < shards.size(); ++i) {
                    q_log << shards[i]->size() << (i == shards.size() - 1 ? "" : ", ");
                }
                q_log << "] | DispatchQ: " << dispatch_queue.size() << " | ScatterQ: " << scatter_queue.size();
                logger.log("DEBUG", q_log.str());
            }
            // --------------------------------

            c10::Half* staging_ptr = pinned_staging[current_slot].data_ptr<c10::Half>();
            for (size_t i = 0; i < req_size; ++i) {
                std::memcpy(staging_ptr + (i * input_numel), shared_input_buffer[requests[i].second].data_ptr<c10::Half>(), input_bytes_per_tensor);
            }

            dispatch_queue.push({current_slot, std::move(requests), batch_start});
            current_slot = (current_slot + 1) % NUM_SLOTS;
        }
    });

    while (!stop_event.load()) {
        if (pending_trt_reload.load()) {
            bool all_free = true;
            for (int i = 0; i < NUM_SLOTS; ++i) {
                if (!slot_free[i].load()) all_free = false;
            }
            
            if (all_free && dispatch_queue.empty()) {
                std::lock_guard<std::mutex> lock(reload_mutex);
                for (int i = 0; i < 3; ++i) {
                    if (trt_contexts[i]) { delete trt_contexts[i]; trt_contexts[i] = nullptr; }
                }
                if (trt_engine) { delete trt_engine; trt_engine = nullptr; }
                if (!trt_runtime) trt_runtime = nvinfer1::createInferRuntime(gBatcherLogger);

                trt_engine = trt_runtime->deserializeCudaEngine(pending_engine_data.data(), pending_engine_data.size());
                
                for (int i = 0; i < 3; ++i) {
                    trt_contexts[i] = trt_engine->createExecutionContext();
                }
                
                logger.rotate(current_global_step.load(), rotation_interval);
                logger.log("INFO", "Hot swap complete. Executing on new TRT Engine.");
                pending_trt_reload.store(false);
            } else {
                std::this_thread::sleep_for(std::chrono::microseconds(100));
            }
            continue;
        }

        if (!trt_contexts[0]) {
            std::this_thread::sleep_for(std::chrono::milliseconds(10));
            continue;
        }

        PipelineJob job;
        if (dispatch_queue.try_pop(job)) {
            int current_slot = job.slot;
            int req_size = job.requests.size();

            auto t_launch_start = std::chrono::steady_clock::now();
            c10::cuda::CUDAStreamGuard guard(*streams[current_slot]);
            cudaEventRecord(fw_start_events[current_slot], raw_streams[current_slot]);
            
            gpu_input_fp16[current_slot].slice(0, 0, req_size).copy_(pinned_staging[current_slot].slice(0, 0, req_size), true);

            int padded_size = (req_size + 63) & ~63;
            if (padded_size == 0) padded_size = 64;
            if (padded_size > batch_size) padded_size = batch_size;

            cudaEventRecord(compute_start[current_slot], raw_streams[current_slot]);
            
            trt_contexts[current_slot]->setInputShape("input", nvinfer1::Dims4{padded_size, 69, 8, 8});
            trt_contexts[current_slot]->setTensorAddress("input", gpu_input_fp16[current_slot].data_ptr());
            trt_contexts[current_slot]->setTensorAddress("policy", gpu_policy_fp16[current_slot].data_ptr());
            trt_contexts[current_slot]->setTensorAddress("value", gpu_value_fp16[current_slot].data_ptr());

            trt_contexts[current_slot]->enqueueV3(raw_streams[current_slot]);
            
            cudaEventRecord(compute_stop[current_slot], raw_streams[current_slot]);

            cudaMemcpyAsync(policy_cpu[current_slot].data_ptr(), gpu_policy_fp16[current_slot].data_ptr(), req_size * policy_bytes_per_tensor, cudaMemcpyDeviceToHost, raw_streams[current_slot]);
            cudaMemcpyAsync(value_cpu[current_slot].data_ptr(), gpu_value_fp16[current_slot].data_ptr(), req_size * value_bytes_per_tensor, cudaMemcpyDeviceToHost, raw_streams[current_slot]);
            cudaEventRecord(fw_stop_events[current_slot], raw_streams[current_slot]);

            if (logging_level <= 10) {
                double launch_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_launch_start).count();
                logger.log("DEBUG", "[Slot " + std::to_string(current_slot) + "] GPU Instruction Overhead: " + std::to_string(launch_ms) + "ms");
            }

            scatter_queue.push(std::move(job));
        } else {
            _mm_pause();
        }
    }

    if (filler_thread.joinable()) filler_thread.join();
    if (collector_thread.joinable()) collector_thread.join();
    delete streams[2];
}