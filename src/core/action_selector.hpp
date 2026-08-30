#pragma once

#include <vector>
#include <string>
#include <memory>
#include <random>
#include "chess.hpp"
#include "mcts_engine.hpp"
#include "logger.hpp"

namespace YAML { class Node; }

struct ActionSelectorConfig {
    // Shared with MctsConfig
    double contempt;
    double draw_cutoff;

    // Selection-only.
    int    temperature_ply_cutoff;
    double temperature_q_decay;
    double resignation_probability;
    double resignation_cutoff;
};

struct LoadedConfigs {
    MctsConfig mcts;
    ActionSelectorConfig selector;
};
LoadedConfigs load_configs(const YAML::Node& mcts_n, const YAML::Node& sel_n,
                           bool require_gumbel_m);

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