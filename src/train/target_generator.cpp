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

    double v_mix = root->calculate_v_mix(config.contempt);

    int max_visits = 0;
    for (int i = 0; i < num_children; ++i) {
        max_visits = std::max(max_visits, all_children[i]->visits);
    }
    double sigma_scale = (config.gumbel_c_visit + max_visits) * config.gumbel_c_scale;

    std::vector<float> base_logits(num_children);
    float max_logit = -1e20f;

    // MLH steering gate + direction. Winning (v_mix >= start): sign -1, prefer
    // faster. Losing (v_mix <= -start): sign +1, prefer slower (drag out). Linear
    // ramp start->full; the two branches are mutually exclusive since start > 0.
    double mlh_gate = 0.0;
    double mlh_sign = 0.0;
    if (config.mlh_lambda > 0.0 && config.mlh_gate_full > config.mlh_gate_start) {
        double denom = config.mlh_gate_full - config.mlh_gate_start;
        if (v_mix >= config.mlh_gate_start) {
            mlh_gate = std::min(1.0, (v_mix - config.mlh_gate_start) / denom);
            mlh_sign = -1.0;
        } else if (v_mix <= -config.mlh_gate_start) {
            mlh_gate = std::min(1.0, (-v_mix - config.mlh_gate_start) / denom);
            mlh_sign = +1.0;
        }
    }
    const bool apply_mlh = (mlh_gate > 0.0);
    double min_mean_mlh = 1e18;
    if (apply_mlh) {
        for (int i = 0; i < num_children; ++i) {
            MCTSNode* c = all_children[i];
            if (c->visits > 0) min_mean_mlh = std::min(min_mean_mlh, c->mlh_sum / c->visits);
        }
    }

    for (int i = 0; i < num_children; ++i) {
        MCTSNode* c = all_children[i];

        double q;
        if (c->forced_outcome.has_value()) {
            int fo = c->forced_outcome.value();
            if      (fo == -1) q =  1.0;              // proven win for us
            else if (fo ==  1) q = -1.0;              // proven loss
            else               q =  config.contempt;  // proven draw
        } else {
            q = (c->visits > 0)
                ? (c->visits * -c->expected_value(config.contempt) + target_shrinkage_k * v_mix) / (c->visits + target_shrinkage_k)
                : v_mix;
        }

        if (apply_mlh && c->visits > 0) {
            q += mlh_sign * config.mlh_lambda * mlh_gate * (c->mlh_sum / c->visits - min_mean_mlh);
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