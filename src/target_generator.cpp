#include "target_generator.hpp"
#include "board_utils.hpp"
#include <cmath>
#include <algorithm>

TargetResult TargetGenerator::generate_targets(
    MCTSNode* root, double root_v_mix, const chess::Board& board,
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

    std::vector<MCTSNode*> winning_nodes, losing_nodes, draw_nodes, non_forced_nodes;
    for (MCTSNode* child : all_children) {
        if (child->forced_outcome.has_value()) {
            if (child->forced_outcome.value() == -1) winning_nodes.push_back(child);
            else if (child->forced_outcome.value() == 1) losing_nodes.push_back(child);
            else draw_nodes.push_back(child);
        } else {
            non_forced_nodes.push_back(child);
        }
    }

    std::vector<float> final_probs(num_children, 0.0f);
    
    auto apply_reallocation = [&](const std::vector<int>& target_indices, float target_mass, bool distribute_targets_evenly) {
        if (target_indices.size() == (size_t)num_children) {
            for (int i = 0; i < num_children; ++i) final_probs[i] = base_probs[i];
            return;
        }
        
        float sum_targets = 0.0f;
        float sum_others = 0.0f;
        std::vector<bool> is_target(num_children, false);
        
        for (int idx : target_indices) {
            is_target[idx] = true;
            sum_targets += base_probs[idx];
        }
        for (int i = 0; i < num_children; ++i) {
            if (!is_target[i]) sum_others += base_probs[i];
        }
        
        for (int i = 0; i < num_children; ++i) {
            if (is_target[i]) {
                if (distribute_targets_evenly) {
                    final_probs[i] = target_mass / target_indices.size();
                } else {
                    final_probs[i] = (sum_targets > 0.0f) ? target_mass * (base_probs[i] / sum_targets) : target_mass / target_indices.size();
                }
            } else {
                if (sum_others > 0.0f) {
                    final_probs[i] = (1.0f - target_mass) * (base_probs[i] / sum_others);
                } else {
                    final_probs[i] = (1.0f - target_mass) / (num_children - target_indices.size());
                }
            }
        }
    };
    
    if (!winning_nodes.empty()) {
        int min_dtm = 999999;
        for (MCTSNode* child : winning_nodes) {
            if (child->distance_to_mate.value() < min_dtm) min_dtm = child->distance_to_mate.value();
        }
        std::vector<int> target_indices;
        for (int i = 0; i < num_children; ++i) {
            if (all_children[i]->forced_outcome.has_value() && 
                all_children[i]->forced_outcome.value() == -1 && 
                all_children[i]->distance_to_mate.value() == min_dtm) {
                target_indices.push_back(i);
            }
        }
        apply_reallocation(target_indices, config.minimax_win_target, true);
        logger.log("INFO", std::to_string(target_indices.size()) + " fastest win(s) found. Reallocating " + std::to_string(config.minimax_win_target) + " mass.");

    } else if (!draw_nodes.empty() && root_v_mix <= config.draw_cutoff) {
        std::vector<int> target_indices;
        for (int i = 0; i < num_children; ++i) {
            if (all_children[i]->forced_outcome.has_value() && all_children[i]->forced_outcome.value() == 0) {
                target_indices.push_back(i);
            }
        }
        apply_reallocation(target_indices, config.minimax_win_target, true);
        logger.log("INFO", "Forced draw condition met. Reallocating " + std::to_string(config.minimax_win_target) + " mass.");

    } else if (!losing_nodes.empty() && !non_forced_nodes.empty()) {
        std::vector<int> target_indices;
        for (int i = 0; i < num_children; ++i) {
            if (all_children[i]->forced_outcome.has_value() && all_children[i]->forced_outcome.value() == 1) {
                target_indices.push_back(i);
            }
        }
        apply_reallocation(target_indices, config.minimax_loss_target, false);
        logger.log("INFO", std::to_string(losing_nodes.size()) + " forced loss(es) squashed to " + std::to_string(config.minimax_loss_target) + " mass.");

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