#pragma once

#include <atomic>
#include <vector>
#include <thread>
#include <string>
#include <cstdint>
#include <torch/torch.h>
#include "chess.hpp"
#include "logger.hpp"
#include "gumbel_mcts.hpp" 
#include "data_generator.hpp" 
#include <zstd.h>

extern "C" {
    #include "lmdb.h"
}

#pragma pack(push, 1)
struct CppState {
    uint64_t games_played;
    uint64_t samples_generated;
    double lifetime_entropy;
    uint64_t buffer_count;
    uint64_t buffer_head_ptr;
    uint64_t buffer_wraps;
};

struct PyState {
    uint64_t training_steps;
    double hours_training;
};
#pragma pack(pop)

class IOWriter {
public:
    IOWriter(
        MDB_env* env,
        size_t min_buffer, 
        size_t max_buffer, 
        int ramp_steps,
        const std::vector<int>& core_ids,
        const std::string& rl_dir,
        size_t flush_threshold,
        int logging_level,
        int rot_interval,
        const ModelConfig& model_cfg,
        ThreadSafeQueue<CompletedGame>& queue,
        std::atomic<uint64_t>& current_step,
        std::atomic<size_t>& write_head,
        std::atomic<size_t>& buffer_count,
        std::atomic<size_t>& buffer_wraps,
        size_t start_games,
        size_t start_samples,
        double start_entropy
    );

    ~IOWriter();

    void start();
    void stop();

private:
    MDB_env* lmdb_env;
    std::atomic<bool> stop_event{false};
    std::thread writer_thread;

    size_t min_buffer_size;
    size_t max_buffer_size;
    int buffer_ramp_steps;
    std::vector<int> io_cores;
    size_t flush_threshold;
    int rotation_interval;
    
    Logger logger;
    ModelConfig model_config;

    ThreadSafeQueue<CompletedGame>& completed_games_queue;
    std::atomic<uint64_t>& current_step;
    std::atomic<size_t>& write_head;
    std::atomic<size_t>& buffer_count;
    std::atomic<size_t>& buffer_wraps;

    size_t lifetime_games;
    size_t lifetime_samples;
    double lifetime_entropy;

    void run();
    size_t get_dynamic_buffer_limit(int current_step);
    
    void pack_bits_into(const std::vector<c10::Half>& data, std::vector<uint8_t>& out); 
    void pack_bits_bool_into(const uint8_t* data, size_t size, std::vector<uint8_t>& out);
};