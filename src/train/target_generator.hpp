#pragma once

#include <vector>
#include "chess.hpp"
#include "mcts_engine.hpp"
#include "action_selector.hpp"
#include "logger.hpp"

struct TargetResult {
    std::vector<float> policy_vector;
    double entropy = 0.0;
};

class TargetGenerator {
public:
    // Pure stateless mathematical observer
    static TargetResult generate_targets(
        MCTSNode* root, 
        const chess::Board& board,
        const ActionSelectorConfig& config, 
        const ModelConfig& model_config, 
        Logger& logger
    );
};