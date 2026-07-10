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
#include <atomic>

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
    const std::string& path, int b_size, int timeout, int workers, int input_planes,
    const std::string& r_dir, int log_level, const std::vector<int>& cores,
    int rot_interval, std::atomic<uint64_t>& initial_step, int log_interval_sec,
    const std::string& lg_name
) : model_path(path), batch_size(b_size), timeout_ms(timeout), 
    num_workers(workers), input_planes(input_planes), rl_dir(r_dir), 
    logging_level(log_level), core_ids(cores), logger_name(lg_name),
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
        if (logger.get_level() <= 30) {
            logger.log("WARNING", "No initial TRT engine found at " + model_path + ". Waiting for background build...");
        }
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
        if (logger.get_level() <= 20) {
            logger.log("INFO", "Initial TRT Engine successfully loaded from disk.");
        }
    }
}

std::vector<std::pair<int, int>> InferenceBatcher::collect_batch(
    moodycamel::ConcurrentQueue<std::pair<int, int>>& queue,
    Logger& logger,
    std::atomic<bool>& stop_event
) {
    std::vector<std::pair<int, int>> requests;
    requests.reserve(batch_size);
    auto start_poll = std::chrono::steady_clock::now();

    while ((int)requests.size() < batch_size) {
        if (stop_event.load()) break;
        
        size_t current_size = requests.size();
        size_t space_left = batch_size - current_size;
        
        requests.resize(batch_size); 
        size_t dequeued = queue.try_dequeue_bulk(requests.data() + current_size, space_left);
        requests.resize(current_size + dequeued); 

        if (dequeued > 0) {
            if ((int)requests.size() >= batch_size) {
                return requests;
            }
            start_poll = std::chrono::steady_clock::now();
        } else {
            double elapsed_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start_poll).count();
            if (elapsed_ms >= (double)timeout_ms && !requests.empty()) break;
            _mm_pause();
        }
    }
    return requests;
}

void InferenceBatcher::run(
    moodycamel::ConcurrentQueue<std::pair<int, int>>& queue,    
    std::vector<ThreadSafeQueue<std::vector<int>>>& result_queues,
    std::vector<torch::Tensor>& shared_input_buffer,
    std::vector<torch::Tensor>& shared_policy_buffer,
    std::vector<torch::Tensor>& shared_value_buffer,
    std::atomic<bool>& stop_event,
    ThreadSafeQueue<int>* buffer_free_slots
) {
    at::set_num_threads(1);

    Logger logger(logger_name, rl_dir, logging_level);
    logger.rotate(current_global_step.load(), rotation_interval);
    
    if (logger.get_level() <= 20) {
        logger.log("INFO", "=== INFERENCE BATCHER STARTED ===");
    }
    
    std::atomic<double> local_idle_time_sec{0.0};

    // Distribute cores round-robin across 3 roles: dispatcher(0), collector(1), filler(2)
    DWORD_PTR frontendMask = 0;
    DWORD_PTR backendMask = 0;
    DWORD_PTR fillerMask = 0;
    std::vector<std::string> role_cores(3);

    for (size_t i = 0; i < core_ids.size(); ++i) {
        DWORD_PTR bit = static_cast<DWORD_PTR>(1) << core_ids[i];
        int role = i % 3;
        if (role == 0) frontendMask |= bit;
        else if (role == 1) backendMask |= bit;
        else fillerMask |= bit;
        if (!role_cores[role].empty()) role_cores[role] += ",";
        role_cores[role] += std::to_string(core_ids[i]);
    }

    if (logger.get_level() <= 20) {
        logger.log("INFO", "Dispatcher pinned to cores [" + role_cores[0] + "]"
                           ", Collector pinned to cores [" + role_cores[1] + "]"
                           ", Filler pinned to cores [" + role_cores[2] + "]");
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
    
    torch::Tensor pinned_staging[NUM_SLOTS] = { torch::zeros(batch_shape, pinned_opts), torch::zeros(batch_shape, pinned_opts), torch::zeros(batch_shape, pinned_opts) };
    torch::Tensor policy_cpu[NUM_SLOTS]     = { torch::zeros(policy_batch_shape, pinned_opts), torch::zeros(policy_batch_shape, pinned_opts), torch::zeros(policy_batch_shape, pinned_opts) };
    torch::Tensor value_cpu[NUM_SLOTS]      = { torch::zeros(value_batch_shape, pinned_opts), torch::zeros(value_batch_shape, pinned_opts), torch::zeros(value_batch_shape, pinned_opts) };

    torch::Tensor gpu_policy_fp16[NUM_SLOTS] = { torch::zeros(policy_batch_shape, gpu_opts_fp16), torch::zeros(policy_batch_shape, gpu_opts_fp16), torch::zeros(policy_batch_shape, gpu_opts_fp16) };
    torch::Tensor gpu_value_fp16[NUM_SLOTS]  = { torch::zeros(value_batch_shape, gpu_opts_fp16), torch::zeros(value_batch_shape, gpu_opts_fp16), torch::zeros(value_batch_shape, gpu_opts_fp16) };
    torch::Tensor gpu_input_fp16[NUM_SLOTS]  = { torch::zeros(batch_shape, gpu_opts_fp16), torch::zeros(batch_shape, gpu_opts_fp16), torch::zeros(batch_shape, gpu_opts_fp16) };

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
        
        uint64_t previousTotalTicks = 0;
        uint64_t previousIdleTicks = 0;

        while (true) {
            PipelineJob job;
            if (scatter_queue.try_pop(job)) {
                cudaStreamSynchronize(raw_streams[job.slot]); 
                
                if (logger.get_level() <= 10) {
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
                if (logger.get_level() <= 20) {
                    
                    size_t free_byte = 0, total_byte = 0;
                    cudaError_t cuda_status = cudaMemGetInfo(&free_byte, &total_byte);
                    double free_db = (double)free_byte / (1024.0 * 1024.0);
                    double total_db = (double)total_byte / (1024.0 * 1024.0);
                    double used_db = total_db - free_db;

                    double idle_sec = local_idle_time_sec.exchange(0.0);
                    double idle_pct = (idle_sec / elapsed_interval_time) * 100.0;
                    
                    double avg_batch_size = (interval_batches_processed > 0) ? (double)interval_total_inferences / interval_batches_processed : 0.0;
                    double fill_rate_pct = (avg_batch_size / batch_size) * 100.0;
                    
                    double avg_process_time = (interval_batches_processed > 0) ? (interval_total_processing_duration / interval_batches_processed) : 0.0;
                    double util_pct = (interval_total_processing_duration / NUM_SLOTS / elapsed_interval_time) * 100.0;
                    double inf_per_sec = interval_total_inferences / elapsed_interval_time;

                    // System CPU tracking
                    FILETIME idleTime, kernelTime, userTime;
                    float sys_cpu_pct = 0.0f;
                    if (GetSystemTimes(&idleTime, &kernelTime, &userTime)) {
                        auto FileTimeToInt64 = [](const FILETIME& ft) {
                            return (((uint64_t)ft.dwHighDateTime) << 32) | ((uint64_t)ft.dwLowDateTime);
                        };
                        uint64_t idleTicks = FileTimeToInt64(idleTime);
                        uint64_t totalTicks = FileTimeToInt64(kernelTime) + FileTimeToInt64(userTime);
                        
                        uint64_t totalTicksSinceLastTime = totalTicks - previousTotalTicks;
                        uint64_t idleTicksSinceLastTime  = idleTicks - previousIdleTicks;
                        
                        if (previousTotalTicks > 0 && totalTicksSinceLastTime > 0) {
                            sys_cpu_pct = 100.0f * (1.0f - ((float)idleTicksSinceLastTime) / totalTicksSinceLastTime);
                        }
                        previousTotalTicks = totalTicks;
                        previousIdleTicks = idleTicks;
                    }

                    // System RAM tracking
                    MEMORYSTATUSEX memInfo;
                    memInfo.dwLength = sizeof(MEMORYSTATUSEX);
                    GlobalMemoryStatusEx(&memInfo);
                    double sys_ram_used_gb = (memInfo.ullTotalPhys - memInfo.ullAvailPhys) / (1024.0 * 1024.0 * 1024.0);
                    double sys_ram_total_gb = memInfo.ullTotalPhys / (1024.0 * 1024.0 * 1024.0);
                    
                    // Queue profiling
                    size_t input_queue_size = queue.size_approx();
                    size_t dispatch_q_size = dispatch_queue.size();
                    size_t scatter_q_size = scatter_queue.size();
                    int current_free_slots = buffer_free_slots ? (int)buffer_free_slots->size() : -1;
                    
                    size_t total_result_items = 0;
                    int active_result_queues = 0;
                    for (auto& rq : result_queues) {
                        size_t qs = rq.size();
                        total_result_items += qs;
                        if (qs > 0) active_result_queues++;
                    }
                    logger.rotate(current_global_step.load(), rotation_interval);

                    logger.log("INFO", "============================================================");
                    logger.log("INFO", " INFERENCE BATCHER DIAGNOSTICS (" + std::to_string(elapsed_interval_time) + "s interval)");
                    logger.log("INFO", "============================================================");
                    
                    logger.log("INFO", "  [SYSTEM HEALTH]");
                    logger.log("INFO", "    System CPU Usage       : " + std::to_string(sys_cpu_pct) + "%");
                    logger.log("INFO", "    System RAM Used        : " + std::to_string(sys_ram_used_gb) + " GB / " + std::to_string(sys_ram_total_gb) + " GB");
                    logger.log("INFO", "    Buffer Free Slots      : " + std::to_string(current_free_slots));

                    logger.log("INFO", "  [QUEUE HEALTH]");
                    logger.log("INFO", "    Inference Q (Approx)   : " + std::to_string(input_queue_size));
                    logger.log("INFO", "    Dispatch Q (To GPU)    : " + std::to_string(dispatch_q_size));
                    logger.log("INFO", "    Scatter Q (From GPU)   : " + std::to_string(scatter_q_size));
                    logger.log("INFO", "    Result Qs (To Workers) : " + std::to_string(total_result_items) + " items across " + std::to_string(active_result_queues) + " active worker queues");

                    logger.log("INFO", "  [THROUGHPUT]");
                    logger.log("INFO", "    Overall Inferences/Sec : " + std::to_string(inf_per_sec));
                    logger.log("INFO", "    Batches Processed      : " + std::to_string(interval_batches_processed));
                    logger.log("INFO", "    Total Inferences       : " + std::to_string(interval_total_inferences));
                    
                    logger.log("INFO", "  [BATCH HEALTH]");
                    logger.log("INFO", "    Avg Batch Process Time : " + std::to_string(avg_process_time) + "s");
                    logger.log("INFO", "    Avg Actual Batch Size  : " + std::to_string(avg_batch_size) + " / " + std::to_string(batch_size));
                    logger.log("INFO", "    Batch Fill Rate        : " + std::to_string(fill_rate_pct) + "%");
                    
                    logger.log("INFO", "  [GPU PIPELINE]");
                    logger.log("INFO", "    Batcher Utilization    : " + std::to_string(util_pct) + "%");
                    logger.log("INFO", "    Filler IDLE Time       : " + std::to_string(idle_pct) + "% (Waiting for Workers)");
                    
                    if (cuda_status == cudaSuccess) {
                        logger.log("INFO", "  [VRAM STATUS]");
                        logger.log("INFO", "    VRAM Used : " + std::to_string(used_db) + " MB");
                        logger.log("INFO", "    VRAM Free : " + std::to_string(free_db) + " MB");
                    } else {
                        logger.log("ERROR", "   [VRAM STATUS] Failed to query CUDA Memory.");
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
            if (pending_trt_reload.load() || pause_requested.load()) {
                std::this_thread::yield();
                continue;
            }

            while (!slot_free[current_slot].load() && !stop_event.load()) {
                _mm_pause();
            }
            if (stop_event.load()) break;

            auto wait_start = std::chrono::steady_clock::now();
            auto requests = collect_batch(queue, logger, stop_event);
            auto wait_end = std::chrono::steady_clock::now();
            
            double wait_duration = std::chrono::duration<double>(wait_end - wait_start).count();
            
            double current_idle = local_idle_time_sec.load();
            while(!local_idle_time_sec.compare_exchange_weak(current_idle, current_idle + wait_duration));

            if (requests.empty()) continue;

            auto batch_start = std::chrono::steady_clock::now();
            int req_size = (int)requests.size();

            slot_free[current_slot].store(false); 

            if (logger.get_level() <= 10) {
                std::ostringstream q_log;
                q_log << "Batch Size: " << req_size << " | InferenceQ Approx: " << queue.size_approx() 
                      << " | DispatchQ: " << dispatch_queue.size() << " | ScatterQ: " << scatter_queue.size();
                logger.log("DEBUG", q_log.str());
            }

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
            std::lock_guard<std::mutex> lock(reload_mutex);
            
            cudaDeviceSynchronize();

            for (int i = 0; i < 3; ++i) {
                if (trt_contexts[i]) { delete trt_contexts[i]; trt_contexts[i] = nullptr; }
            }
            if (trt_engine) { delete trt_engine; trt_engine = nullptr; }
            if (trt_runtime) { delete trt_runtime; trt_runtime = nullptr; }

            trt_runtime = nvinfer1::createInferRuntime(gBatcherLogger);
            trt_engine = trt_runtime->deserializeCudaEngine(pending_engine_data.data(), pending_engine_data.size());
            
            pending_engine_data.clear();
            pending_engine_data.shrink_to_fit();

            for (int i = 0; i < 3; ++i) {
                trt_contexts[i] = trt_engine->createExecutionContext();
            }
            
            if (logger.get_level() <= 20) {
                logger.log("INFO", "Hot swap complete. Resuming Inference.");
            }
            
            pending_trt_reload.store(false);
            is_paused.store(false); 
            pause_requested.store(false); 
            continue;
        }

        bool all_free = true;
        for (int i = 0; i < NUM_SLOTS; ++i) {
            if (!slot_free[i].load()) all_free = false;
        }

        if (pause_requested.load()) {
            if (all_free && dispatch_queue.empty()) {
                is_paused.store(true);
                std::this_thread::yield();
                continue; 
            }
        }

        if (!trt_contexts[0]) {
            std::this_thread::yield();
            continue;
        }

        PipelineJob job;
        if (dispatch_queue.try_pop(job)) {
            int current_slot = job.slot;
            int req_size = job.requests.size();

            auto t_launch_start = std::chrono::steady_clock::now();
            c10::cuda::CUDAStreamGuard guard(*streams[current_slot]);
            cudaEventRecord(fw_start_events[current_slot], raw_streams[current_slot]);

            int padded_size = (req_size + 63) & ~63;

            if (padded_size == 0) padded_size = 64;
            if (padded_size > batch_size) padded_size = batch_size;

            gpu_input_fp16[current_slot].slice(0, 0, padded_size).zero_();
            gpu_input_fp16[current_slot].slice(0, 0, req_size).copy_(pinned_staging[current_slot].slice(0, 0, req_size), true);


            cudaEventRecord(compute_start[current_slot], raw_streams[current_slot]);
            
            trt_contexts[current_slot]->setInputShape("input", nvinfer1::Dims4{padded_size, 111, 8, 8});
            trt_contexts[current_slot]->setTensorAddress("input", gpu_input_fp16[current_slot].data_ptr());
            trt_contexts[current_slot]->setTensorAddress("policy", gpu_policy_fp16[current_slot].data_ptr());
            trt_contexts[current_slot]->setTensorAddress("value", gpu_value_fp16[current_slot].data_ptr());

            trt_contexts[current_slot]->enqueueV3(raw_streams[current_slot]);
            
            cudaEventRecord(compute_stop[current_slot], raw_streams[current_slot]);

            cudaMemcpyAsync(policy_cpu[current_slot].data_ptr(), gpu_policy_fp16[current_slot].data_ptr(), req_size * policy_bytes_per_tensor, cudaMemcpyDeviceToHost, raw_streams[current_slot]);
            cudaMemcpyAsync(value_cpu[current_slot].data_ptr(), gpu_value_fp16[current_slot].data_ptr(), req_size * value_bytes_per_tensor, cudaMemcpyDeviceToHost, raw_streams[current_slot]);
            cudaEventRecord(fw_stop_events[current_slot], raw_streams[current_slot]);

            if (logger.get_level() <= 10) {
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