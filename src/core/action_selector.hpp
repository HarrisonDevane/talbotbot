#pragma once

#include <vector>
#include <string>
#include <memory>
#include <random>
#include "chess.hpp"
#include "mcts_engine.hpp"
#include "logger.hpp"

struct ActionSelectorConfig {
    // Unused since stage 3 -- kept only so any callers still populating this
    // field don't fail to compile. Pool sizing lives on MCTSEngine::pool_sizing_cfg
    // now; nothing here reads it.
    int node_pool_size = 0;

    double contempt;
    double deficit_eps;
    double policy_softmax_temp;
    double virtual_loss;
    double draw_cutoff;
    double gumbel_c_visit;
    double gumbel_c_scale;
    double gumbel_noise;
    double gumbel_search_depth;
    double gumbel_m;
    
    int temperature_ply_cutoff;
    double temperature_q_decay;
    
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

    // Iterates root's edges (MCTSEdge*). move and raw_logit live on the edge;
    // visits / expected_value / forced_outcome live on edge->child (nullable
    // -- unmaterialised children treated the same as visits==0 was pre-split).
    // Gumbel tie-break sort computes score inline via MCTSEdge::calculate_gumbel_score
    // with noise pulled from engine->root_gumbel_noise, same formula as before.
    SelectionResult select_move(MCTSNode* root, int ply_count, MCTSEngine* engine);
};