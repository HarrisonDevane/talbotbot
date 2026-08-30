#pragma once

#include <vector>
#include "chess.hpp"
#include "mcts_engine.hpp"
#include "logger.hpp"

struct TargetResult {
    std::vector<float> policy_vector;
    double entropy = 0.0;
};

class TargetGenerator {
public:
    static TargetResult generate_targets(
        MCTSNode* root,
        const chess::Board& board,
        const MctsConfig& config,
        const ModelConfig& model_config,
        const double target_shrinkage_k,
        Logger& logger
    );
};