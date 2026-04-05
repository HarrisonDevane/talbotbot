#pragma once

#include <vector>
#include <string>
#include <memory>
#include <random>
#include "chess.hpp"
#include "mcts_engine.hpp"
#include "logger.hpp"

struct ActionSelectorConfig {
    int node_pool_size;
    double virtual_loss;
    double draw_cutoff;
    double gumbel_c_visit;
    double gumbel_c_scale;
    double gumbel_noise;
    double gumbel_search_depth;
    double gumbel_m;                   
    
    double minimax_smoothing_factor;
    int temperature_ply_cutoff;
    double temperature_blunder_threshold;
    double temperature_top_move;
    
    double resignation_probability;
    double resignation_cutoff;
    int batch_size_per_worker;
};

struct SelectionResult {
    chess::Move best_move = chess::Move::NO_MOVE;
    std::vector<float> policy_vector;
    int simulation_count = 0;
    double entropy = 0.0;
    bool resigned = false;
};

class ActionSelector {
private:
    std::string name;
    int worker_id;
    ActionSelectorConfig config;
    bool use_resignation;
    Logger& logger; 

    ModelConfig model_config;
    std::mt19937 rng; 
    
    // Persistent Engine to stop allocator thrashing
    std::unique_ptr<MCTSEngine> mcts;

    ThreadSafeQueue<std::vector<std::pair<int, int>>>& inference_queue;
    ThreadSafeQueue<std::vector<int>>& result_queue;
    std::vector<torch::Tensor>& shared_input_buffer;
    std::vector<torch::Tensor>& shared_policy_buffer;
    std::vector<torch::Tensor>& shared_value_buffer;
    ThreadSafeQueue<int>& buffer_free_slots;

public:
    ActionSelector(
        std::string name,
        int worker_id,
        ActionSelectorConfig config,
        const ModelConfig& model_cfg, 
        Logger& logger, 
        ThreadSafeQueue<std::vector<std::pair<int, int>>>& i_queue,
        ThreadSafeQueue<std::vector<int>>& r_queue,
        std::vector<torch::Tensor>& in_buffer,
        std::vector<torch::Tensor>& p_buffer,
        std::vector<torch::Tensor>& v_buffer,
        ThreadSafeQueue<int>& free_slots
    );

    void reset_for_new_game();
    void set_name(const std::string& new_name) { name = new_name; }

    SelectionResult select_action(
        const chess::Board& board, 
        const std::vector<chess::Board>& history, 
        int ply_count, 
        int gumbel_search_depth, 
        int gumbel_m
    );
};