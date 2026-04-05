#pragma once

#include <vector>
#include <optional>
#include "chess.hpp"

struct MCTSNode {
    MCTSNode* parent = nullptr;
    
    // THE FIX: Only point to the first child. The rest are contiguous in the NodePool.
    MCTSNode* first_child = nullptr;
    int num_children = 0;
    
    chess::Move move = chess::Move::NO_MOVE;
    int policy_flat_index = -1;

    int visits = 0; 
    int num_unselected_children = 0;
    double value_sum = 0.0;
    double raw_logit = 0.0;
    double raw_value = 0.0;
    double q_val = 0.0;
    double q_norm = 0.0;

    double gumbel_noise = 0.0;
    double gumbel_score = 0.0;
    
    std::optional<int> forced_outcome = std::nullopt;
    std::optional<int> distance_to_mate = std::nullopt;
    
    bool expanded = false;
    bool selected = false;

    MCTSNode(MCTSNode* p = nullptr, chess::Move m = chess::Move::NO_MOVE);

    MCTSNode* get_child(chess::Move m) const;
    double calculate_gumbel_score(double gumbel_c_visit, double gumbel_c_scale, double max_visits, double v_mix);
    double calculate_v_mix() const;
};