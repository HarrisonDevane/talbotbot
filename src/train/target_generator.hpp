#pragma once

#include <vector>
#include "chess.hpp"
#include "mcts_base.hpp"    // MCTSNode + ModelConfig live here now
#include "logger.hpp"

struct TargetResult {
    std::vector<float> policy_vector;
    double entropy = 0.0;
};

// Stateless helper: turn a finished gumbel-search tree into training targets.
// The three knobs it needs -- contempt for value framing, gumbel_c_visit /
// gumbel_c_scale for the σ scale on the completed-Q softmax -- live in
// TargetGenerator::Config below rather than piggy-backing on a wider config.
class TargetGenerator {
public:
    struct Config {
        double contempt;
        double gumbel_c_visit;
        double gumbel_c_scale;
    };

    static TargetResult generate_targets(
        MCTSNode* root,
        const chess::Board& board,
        const Config& cfg,
        const ModelConfig& model_config,
        const double target_shrinkage_k,
        Logger& logger
    );
};