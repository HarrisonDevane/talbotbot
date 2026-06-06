#include "target_generator.hpp"
#include "board_utils.hpp"
#include <cmath>
#include <algorithm>

TargetResult TargetGenerator::generate_targets(
    MCTSNode* root, const chess::Board& board,
    const ActionSelectorConfig& config, const ModelConfig& model_config, Logger& logger) 
{
    TargetResult result;
    result.policy_vector.resize(model_config.policy_moves, 0.0f);
    
    int num_children = root->num_children;
    if (num_children == 0) return result;

    std::vector<MCTSNode*> all_children;
    chess::Movelist all_moves;
    for(int i = 0; i < num_children; ++i) {
        all_children.push_back(root->first_child + i);
        all_moves.add(all_children.back()->move);
    }

    std::vector<float> base_logits(num_children);
    float max_logit = -1e20f;

    for (int i = 0; i < num_children; ++i) {
        base_logits[i] = static_cast<float>(all_children[i]->gumbel_score - all_children[i]->gumbel_noise);
        if (base_logits[i] > max_logit) max_logit = base_logits[i];
    }

    std::vector<float> base_probs(num_children, 0.0f);
    float sum_exp = 0.0f;
    for (int i = 0; i < num_children; ++i) {
        base_probs[i] = std::exp(base_logits[i] - max_logit);
        sum_exp += base_probs[i];
    }
    for (int i = 0; i < num_children; ++i) base_probs[i] /= sum_exp;

    std::vector<MCTSNode*> losing_nodes;
    for (MCTSNode* child : all_children) {
        if (child->forced_outcome.has_value() && child->forced_outcome.value() == 1) {
            losing_nodes.push_back(child);
        }
    }

    std::vector<float> final_probs(num_children, 0.0f);
    
    // Zero out forced losses, renormalize the rest.
    // Wins and draws keep their smooth improved-policy probabilities.
    if (!losing_nodes.empty()) {
        float non_loss_mass = 0.0f;

        for (int i = 0; i < num_children; ++i) {
            bool is_loss = all_children[i]->forced_outcome.has_value() && 
                           all_children[i]->forced_outcome.value() == 1;
            if (is_loss) {
                final_probs[i] = 0.0f;
            } else {
                non_loss_mass += base_probs[i];
            }
        }

        for (int i = 0; i < num_children; ++i) {
            bool is_loss = all_children[i]->forced_outcome.has_value() && 
                           all_children[i]->forced_outcome.value() == 1;
            if (!is_loss) {
                if (non_loss_mass > 0.0f) {
                    final_probs[i] = base_probs[i] / non_loss_mass;
                } else {
                    final_probs[i] = 1.0f / (num_children - (int)losing_nodes.size());
                }
            }
        }

        logger.log("INFO", std::to_string(losing_nodes.size()) + " forced loss(es) zeroed out and mass redistributed.");
    } else {
        final_probs = base_probs;
    }

    map_policy_to_global_vector(all_moves, final_probs.data(), board, result.policy_vector.data());
    
    result.entropy = 0.0;
    for (float p : final_probs) {
        if (p > 0) result.entropy -= (p * std::log(p + 1e-10f));
    }

    return result;
}