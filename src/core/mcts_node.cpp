#include "mcts_node.hpp"

MCTSNode::MCTSNode(MCTSNode* p, chess::Move m) : parent(p), move(m) {}

MCTSNode* MCTSNode::get_child(chess::Move m) const {
    for (int i = 0; i < num_children; ++i) {
        MCTSNode* child = first_child + i;
        if (child->move == m) return child;
    }
    return nullptr;
}

double MCTSNode::expected_value(double contempt) const {
    if (visits == 0) return (raw_w - raw_l()) + (contempt * raw_d);
    return (w_sum - l_sum + (contempt * d_sum)) / visits;
}

// No caching -- callers store if they need to reuse. noise: non-root
// descendants pass 0; root children pass their entry from
// MCTSEngine::root_gumbel_noise[].
double MCTSNode::calculate_gumbel_score(double contempt, double gumbel_c_visit,
                                        double gumbel_c_scale, double max_visits,
                                        double v_mix, double noise) const {
    double q_val = (visits > 0) ? -expected_value(contempt) : v_mix;
    double q_norm = (q_val + 1.0) / 2.0;
    double sigma = (gumbel_c_visit + max_visits) * gumbel_c_scale;
    return raw_logit + noise + (sigma * q_norm);
}

double MCTSNode::calculate_v_mix(double contempt) const {
    double sum_visits = 0.0;
    double sum_q_weighted = 0.0;

    for (int i = 0; i < num_children; ++i) {
        MCTSNode* child = first_child + i;
        if (child->visits > 0) {
            double child_q = -child->expected_value(contempt);
            sum_visits    += child->visits;
            sum_q_weighted += (child->visits * child_q);
        }
    }
    double raw_q = (raw_w - raw_l()) + (contempt * raw_d);
    return (raw_q + sum_q_weighted) / (1.0 + sum_visits);
}