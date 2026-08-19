#include "target_generator.hpp"
#include "board_utils.hpp"
#include <cmath>
#include <algorithm>

TargetResult TargetGenerator::generate_targets(
    MCTSNode* root, const chess::Board& board,
    const ActionSelectorConfig& config, const ModelConfig& model_config, const double target_shrinkage_k, Logger& logger) 
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

    // Improved policy: pi' = softmax(raw_logit + sigma(completedQ))
    //
    // completedQ uses visit-weighted shrinkage toward v_mix rather than the
    // paper's step function (Eq. 10). A node's empirical Q is blended with
    // v_mix in proportion to its visits:
    //
    //   completedQ(a) = (N(a) * q_hat(a) + k * v_mix) / (N(a) + k)
    //
    // N=0 reduces to pure v_mix (identical to Eq. 10); as N grows the
    // empirical Q dominates. This prevents a confidently-misevaluated
    // 1-visit node from presenting its full raw Q to the training target.
    //
    // Forced outcomes are exact, not estimates -- shrinkage does not apply.
    // A proven win enters the softmax at q = +1, a proven loss at q = -1,
    // a proven draw at its exact draw value, regardless of visit count.
    //
    // Proven losses are NOT hard-zeroed: at sigma_scale >= c_visit, the
    // q_norm floor of 0.0 suppresses them by tens of logits, so their
    // target mass is effectively zero while the target remains a smooth
    // softmax over exact values (paper-canonical Eq. 11 behaviour).


    double v_mix = root->calculate_v_mix(config.contempt);

    int max_visits = 0;
    for (int i = 0; i < num_children; ++i) {
        max_visits = std::max(max_visits, all_children[i]->visits);
    }
    double sigma_scale = (config.gumbel_c_visit + max_visits) * config.gumbel_c_scale;

    std::vector<float> base_logits(num_children);
    float max_logit = -1e20f;

    for (int i = 0; i < num_children; ++i) {
        MCTSNode* c = all_children[i];

        double q;
        if (c->has_forced_outcome()) {
            // Proven subtree: exact value, no shrinkage, visits irrelevant.
            // forced_outcome is from the child's perspective; negate for ours.
            int fo = c->forced_outcome;
            if      (fo == -1) q =  1.0;              // proven win for us
            else if (fo ==  1) q = -1.0;              // proven loss
            else               q =  config.contempt;  // proven draw
        } else {
            q = (c->visits > 0)
                ? (c->visits * -c->expected_value(config.contempt) + target_shrinkage_k * v_mix) / (c->visits + target_shrinkage_k)
                : v_mix;
        }

        double q_norm = (q + 1.0) / 2.0;
        base_logits[i] = static_cast<float>(c->raw_logit + sigma_scale * q_norm);
        if (base_logits[i] > max_logit) max_logit = base_logits[i];
    }

    std::vector<float> final_probs(num_children, 0.0f);
    float sum_exp = 0.0f;
    for (int i = 0; i < num_children; ++i) {
        final_probs[i] = std::exp(base_logits[i] - max_logit);
        sum_exp += final_probs[i];
    }
    for (int i = 0; i < num_children; ++i) final_probs[i] /= sum_exp;

    map_policy_to_global_vector(all_moves, final_probs.data(), board, result.policy_vector.data());
    
    result.entropy = 0.0;
    for (float p : final_probs) {
        if (p > 0) result.entropy -= (p * std::log(p + 1e-10f));
    }

    return result;
}