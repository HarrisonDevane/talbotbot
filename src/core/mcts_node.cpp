#include "mcts_node.hpp"
#include <algorithm>

MCTSNode::MCTSNode(MCTSNode* p, chess::Move m) : parent(p), move(m) {}

MCTSNode* MCTSNode::get_child(chess::Move m) const {
    for (int i = 0; i < num_children; ++i) {
        MCTSNode* child = first_child + i; // Pointer arithmetic
        if (child->move == m) return child;
    }
    return nullptr;
}

double MCTSNode::expected_value(double contempt) const {
    if (visits == 0) return (raw_w - raw_l) + (contempt * raw_d);
    return (w_sum - l_sum + (contempt * d_sum)) / visits;
}

double MCTSNode::calculate_gumbel_score(double contempt, double gumbel_c_visit, double gumbel_c_scale, double max_visits, double v_mix,
                                        double min_sibling_mlh, double mlh_lambda, double mlh_gate_start, double mlh_gate_full) {
    double q_val = (visits > 0) ? -expected_value(contempt) : v_mix;

    // Moves-left steering, both directions, relative to the fastest sibling
    // (rel >= 0 = how much slower this move is). Winning (v_mix >= start): lower Q
    // of slower moves -> converge faster. Losing (v_mix <= -start): raise Q of
    // slower moves -> drag the game out. Linear ramp start->full on |v_mix|; the
    // two branches are mutually exclusive since start > 0.
    if (mlh_lambda > 0.0 && mlh_gate_full > mlh_gate_start && visits > 0) {
        double denom = mlh_gate_full - mlh_gate_start;
        double rel = mlh_sum / visits - min_sibling_mlh;
        if (v_mix >= mlh_gate_start) {
            double gate = std::min(1.0, (v_mix - mlh_gate_start) / denom);
            q_val -= mlh_lambda * gate * rel;   // winning: prefer faster finish
        } else if (v_mix <= -mlh_gate_start) {
            double gate = std::min(1.0, (-v_mix - mlh_gate_start) / denom);
            q_val += mlh_lambda * gate * rel;   // losing: prefer to prolong
        }
    }

    double q_norm = (q_val + 1.0) / 2.0;
    double sigma = (gumbel_c_visit + max_visits) * gumbel_c_scale;
    gumbel_score = raw_logit + gumbel_noise + (sigma * q_norm);
    
    return gumbel_score;
}

double MCTSNode::calculate_v_mix(double contempt) const {
    double raw_q = (raw_w - raw_l) + (contempt * raw_d);
    
    if (num_children == 0) {
        return raw_q;
    }

    // 1. Find max logit for numerical stability in the softmax
    double max_logit = -1e20;
    for (int i = 0; i < num_children; ++i) {
        if ((first_child + i)->raw_logit > max_logit) {
            max_logit = (first_child + i)->raw_logit;
        }
    }

    // 2. Calculate the denominator for the softmax distribution
    double sum_exp = 0.0;
    for (int i = 0; i < num_children; ++i) {
        sum_exp += std::exp((first_child + i)->raw_logit - max_logit);
    }

    // 3. Accumulate components for Equation 33
    double sum_visits = 0.0;
    double sum_pi_visited = 0.0;
    double sum_pi_q = 0.0;

    for (int i = 0; i < num_children; ++i) {
        MCTSNode* child = first_child + i;
        if (child->visits > 0) {
            // Calculate pi(a) for this specific visited action
            double pi_a = std::exp(child->raw_logit - max_logit) / sum_exp;
            double child_q = -child->expected_value(contempt);
            
            sum_visits += child->visits;
            sum_pi_visited += pi_a;
            sum_pi_q += (pi_a * child_q);
        }
    }

    // 4. Guard against division by zero if no nodes have been visited
    if (sum_visits == 0.0 || sum_pi_visited == 0.0) {
        return raw_q;
    }

    // 5. Compute the final v_mix 
    double scaling_factor = sum_visits / sum_pi_visited;
    double weighted_q_term = scaling_factor * sum_pi_q;
    
    return (raw_q + weighted_q_term) / (1.0 + sum_visits);
}