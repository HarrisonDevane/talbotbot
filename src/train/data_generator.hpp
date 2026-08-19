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
#include "gumbel_mcts.hpp"              // pulls in mcts_base.hpp transitively
#include "gumbel_action_selector.hpp"   // pulls in action_selector_base.hpp
#include "logger.hpp"
#include "target_generator.hpp"
#include "concurrentqueue.h"

struct DataGenConfig {
    int num_cores;
    int workers_per_core; 
    int total_workers;
    std::vector<int> core_ids;
    int max_ply_length;
    int worker_logging_level;
    int rotation_interval;
    double target_shrinkage_k;
    std::string rl_dir;
};

struct GameTransition {
    std::vector<c10::Half> board_state;
    std::vector<float> policy;      
    std::vector<uint8_t> legal_mask;
    chess::Color turn;
    chess::Move move;
};

struct CompletedGame {
    int game_number;
    std::vector<GameTransition> transitions;
    double final_game_value;
    int local_step;
    double game_entropy_sum;
};

class DataGenerator {
public:
    std::atomic<int> game_counter;
    std::atomic<size_t> interval_games{0};
    std::atomic<size_t> interval_samples{0};
    std::atomic<double> interval_entropy{0.0};

    DataGenerator(
        const YAML::Node& global_cfg,
        const YAML::Node& data_gen_cfg,
        const YAML::Node& mcts_cfg,
        const YAML::Node& gumbel_cfg,   // new: gumbel: block in train.yaml
        const YAML::Node& sel_cfg,
        const YAML::Node& model_cfg,
        const std::string& rl_dir,
        int rot_interval,
        Logger& logger,
        moodycamel::ConcurrentQueue<std::pair<int, int>>& i_queue,
        std::vector<ThreadSafeQueue<std::vector<int>>>& r_queues,
        std::vector<torch::Tensor>& in_buffer,
        std::vector<torch::Tensor>& p_buffer,
        std::vector<torch::Tensor>& v_buffer,
        ThreadSafeQueue<int>& free_slots,
        ThreadSafeQueue<CompletedGame>& completed_games_queue,
        int start_game_id,
        std::atomic<uint64_t>& current_step
    );

    ~DataGenerator();

    void start();
    void stop();

private:
    DataGenConfig config;

    // Flat yaml-derived parameters used to construct each worker's
    // GumbelMCTS + GumbelActionSelector + TargetGenerator::Config. Internal
    // to DataGenerator so no other TU depends on this shape.
    struct TreeParams {
        // Shared MCTS (mcts: block)
        int    node_pool_size;
        int    batch_size_per_worker;
        double virtual_loss;
        double contempt;
        double policy_softmax_temp;
        bool   two_fold_repetition;

        // Gumbel-specific (gumbel: block)
        double gumbel_c_visit;
        double gumbel_c_scale;
        double gumbel_noise;
        int    gumbel_search_depth;
        int    gumbel_m;
        double temperature_q_decay;

        // Shared action-selection (selection: block)
        int    temperature_ply_cutoff;
        double draw_cutoff;
        double resignation_probability;
        double resignation_cutoff;
    };
    TreeParams tree_params;

    ModelConfig model_config;
    
    Logger& main_logger; 
    std::atomic<bool> stop_event;
    std::atomic<uint64_t>& current_step;

    moodycamel::ConcurrentQueue<std::pair<int, int>>& inference_queue;
    std::vector<ThreadSafeQueue<std::vector<int>>>& result_queues;
    std::vector<torch::Tensor>& shared_input_buffer;
    std::vector<torch::Tensor>& shared_policy_buffer;
    std::vector<torch::Tensor>& shared_value_buffer;
    std::vector<std::unique_ptr<std::atomic<int>>> core_wait_counts;
    ThreadSafeQueue<int>& buffer_free_slots;

    ThreadSafeQueue<CompletedGame>& completed_games_queue;
    std::vector<std::thread> workers;

    void worker_main(int logical_idx, int core_id);
    void _generate_pgn(int game_number, const std::vector<GameTransition>& transitions, const std::string& result_str, Logger& logger);
};