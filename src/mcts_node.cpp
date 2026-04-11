#include "mcts_node.hpp"

MCTSNode::MCTSNode(MCTSNode* p, chess::Move m) : parent(p), move(m) {}

MCTSNode* MCTSNode::get_child(chess::Move m) const {
    for (int i = 0; i < num_children; ++i) {
        MCTSNode* child = first_child + i; // Pointer arithmetic
        if (child->move == m) return child;
    }
    return nullptr;
}

double MCTSNode::calculate_gumbel_score(double gumbel_c_visit, double gumbel_c_scale, double max_visits, double v_mix) {
    double q_val = (visits > 0) ? (-value_sum / visits) : v_mix;
    double q_norm = (q_val + 1.0) / 2.0;
    
    double sigma = (gumbel_c_visit + max_visits) * gumbel_c_scale;
    gumbel_score = raw_logit + gumbel_noise + (sigma * q_norm);
    
    return gumbel_score;
}

double MCTSNode::calculate_v_mix() const {
    double sum_visits = 0.0;
    double sum_q_weighted = 0.0;

    for (int i = 0; i < num_children; ++i) {
        MCTSNode* child = first_child + i;
        if (child->visits > 0) {
            double child_q = -child->value_sum / child->visits;
            sum_visits += child->visits;
            sum_q_weighted += (child->visits * child_q);
        }
    }
    return (raw_value + sum_q_weighted) / (1.0 + sum_visits);
}