#pragma once

#include <atomic>
#include <vector>
#include <thread>
#include <string>
#include <cstdint>
#include <memory>
#include <torch/torch.h>
#include <yaml-cpp/yaml.h>
#include "chess.hpp"
#include "mcts_engine.hpp"
#include "action_selector.hpp"
#include "logger.hpp"

extern "C" {
    #include "lmdb.h"
}

// 1. STRICT ISOLATION: Only Game/Worker orchestration settings
struct DataGenConfig {
    int num_cores;
    int workers_per_core; 
    int total_workers;
    std::vector<int> core_ids;
    int max_ply_length;
    int worker_logging_level;
    int rotation_interval;
    std::string rl_dir;
    std::string state_file;
};

struct GameTransition {
    std::vector<c10::Half> board_state;
    std::vector<float> policy;      
    std::vector<uint8_t> legal_mask;
    chess::Color turn;
    chess::Move move; // Track move for PGN
};

class DataGenerator {
private:
    DataGenConfig config;
    ActionSelectorConfig selector_config;
    ModelConfig model_config; 
    
    Logger& main_logger; 

    std::string lmdb_path;
    MDB_env* lmdb_env;
    std::atomic<bool> stop_event;
    std::atomic<int> game_counter;

    std::vector<ThreadSafeQueue<std::vector<std::pair<int, int>>>*>& inference_shards;
    std::vector<ThreadSafeQueue<std::vector<int>>>& result_queues;
    std::vector<torch::Tensor>& shared_input_buffer;
    std::vector<torch::Tensor>& shared_policy_buffer;
    std::vector<torch::Tensor>& shared_value_buffer;
    ThreadSafeQueue<int>& buffer_free_slots;

    std::atomic<size_t>& write_head;
    std::atomic<size_t>& buffer_count;
    size_t max_buffer_size;

    std::vector<std::thread> workers;

    void worker_main(int logical_idx, int core_id);
    void write_game_to_lmdb(const std::vector<GameTransition>& game_data, double final_game_value);
    void _generate_pgn(int game_number, const std::vector<GameTransition>& transitions, const std::string& result_str, Logger& logger);

    std::vector<uint8_t> pack_bits(const std::vector<c10::Half>& data); // FIXED: Signature updated
    std::vector<uint8_t> pack_bits_bool(const uint8_t* data, size_t size);

public:
    DataGenerator(
        const YAML::Node& data_gen_cfg,
        const YAML::Node& mcts_cfg,
        const YAML::Node& sel_cfg,
        const YAML::Node& model_cfg,
        const std::string& rl_dir,
        const std::string& state_file,
        const std::string& db_path,
        int rot_interval,
        Logger& logger,
        std::vector<ThreadSafeQueue<std::vector<std::pair<int, int>>>*>& i_shards,
        std::vector<ThreadSafeQueue<std::vector<int>>>& r_queues,
        std::vector<torch::Tensor>& in_buffer,
        std::vector<torch::Tensor>& p_buffer,
        std::vector<torch::Tensor>& v_buffer,
        ThreadSafeQueue<int>& free_slots,
        std::atomic<size_t>& w_head,
        std::atomic<size_t>& b_count,
        size_t max_buffer_size
    );

    ~DataGenerator();

    void start();
    void stop();
};