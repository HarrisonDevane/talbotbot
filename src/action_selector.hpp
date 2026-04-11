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
    
    double minimax_win_target;
    double minimax_loss_target;
    int temperature_ply_cutoff;
    double temperature_blunder_threshold;
    double temperature_top_move;
    
    double resignation_probability;
    double resignation_cutoff;
    int batch_size_per_worker;
};

struct SelectionResult {
    chess::Move best_move = chess::Move::NO_MOVE;
    bool resigned = false;
};

class ActionSelector {
private:
    std::string name;
    int worker_id;
    ActionSelectorConfig config;
    bool use_resignation;
    Logger& logger; 
    std::mt19937 rng; 

public:
    ActionSelector(std::string name, int worker_id, ActionSelectorConfig config, Logger& logger);
    
    void reset_for_new_game();
    void set_name(const std::string& new_name) { name = new_name; }

    SelectionResult select_move(MCTSNode* root, double root_v_mix, int ply_count);
};