#include "mcts_node.hpp"

MCTSNode::MCTSNode(MCTSNode* p) : parent(p) {}

MCTSEdge* MCTSNode::get_edge(chess::Move m) const {
    for (int i = 0; i < num_children; ++i) {
        MCTSEdge* e = first_edge + i;
        if (e->move == m) return e;
    }
    return nullptr;
}

double MCTSNode::expected_value(double contempt) const {
    // Proven outcomes dominate: forced_outcome is a proof, contempt does
    // not apply. Perspective is this node's mover.
    if (has_forced_outcome()) {
        if (forced_outcome == -1) return -1.0;
        if (forced_outcome ==  0) return  0.0;
        return 1.0;   // forced_outcome == 1
    }
    // Mover-has-draw override: mhd asserts this node's mover realizes
    // exactly 0 (via PV preference to draw or no-escape). Contempt does
    // not apply -- the flag says the drawn value IS the realized value,
    // not an unresolved estimate to bias.
    if (mover_has_draw()) return 0.0;
    // Unresolved: NN-averaged Q, with contempt bias on the draw fraction.
    if (visits == 0) return (raw_w - raw_l()) + (contempt * raw_d);
    return (w_sum - l_sum + (contempt * d_sum)) / visits;
}

// Moved from MCTSNode -- the score is now a property of the edge from parent
// to child. Formula identical to the previous MCTSNode::calculate_gumbel_score,
// with visits/Q read through the child pointer. Null child == 0 visits ==
// fall back to v_mix, matching the pre-refactor 0-visit case.
double MCTSEdge::calculate_gumbel_score(double contempt, double gumbel_c_visit,
                                        double gumbel_c_scale, double max_visits,
                                        double v_mix, double noise) const {
    int32_t v = (child != nullptr) ? child->visits : 0;
    double q_val = (v > 0) ? -child->expected_value(contempt) : v_mix;
    double q_norm = (q_val + 1.0) / 2.0;
    double sigma = (gumbel_c_visit + max_visits) * gumbel_c_scale;
    return raw_logit + noise + (sigma * q_norm);
}

double MCTSNode::calculate_v_mix(double contempt) const {
    double sum_visits = 0.0;
    double sum_q_weighted = 0.0;

    for (int i = 0; i < num_children; ++i) {
        MCTSEdge* edge = first_edge + i;
        MCTSNode* c = edge->child;
        if (c != nullptr && c->visits > 0) {
            double child_q = -c->expected_value(contempt);
            sum_visits     += c->visits;
            sum_q_weighted += (c->visits * child_q);
        }
    }
    double raw_q = (raw_w - raw_l()) + (contempt * raw_d);
    return (raw_q + sum_q_weighted) / (1.0 + sum_visits);
}