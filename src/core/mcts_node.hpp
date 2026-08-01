#pragma once

#include <vector>
#include <optional>
#include "chess.hpp"

struct MCTSNode {
    MCTSNode* parent = nullptr;
    
    MCTSNode* first_child = nullptr;
    int num_children = 0;
    
    chess::Move move = chess::Move::NO_MOVE;
    int policy_flat_index = -1;

    int visits = 0; 
    int num_available_children = 0;
    
    double w_sum = 0.0;
    double d_sum = 0.0;
    double l_sum = 0.0;

    double raw_logit = 0.0;
    double raw_w = 0.0;
    double raw_d = 0.0;
    double raw_l = 0.0;

    double raw_mlh = 0.0;
    double mlh_sum = 0.0;

    double gumbel_noise = 0.0;
    double gumbel_score = 0.0;
    
    std::optional<int> forced_outcome = std::nullopt;
    std::optional<int> distance_to_mate = std::nullopt;
    
    bool expanded = false;
    bool unavailable_for_selection = false;

    MCTSNode(MCTSNode* p = nullptr, chess::Move m = chess::Move::NO_MOVE);

    MCTSNode* get_child(chess::Move m) const;
    
    double expected_value(double contempt) const;
    double calculate_gumbel_score(double contempt, double gumbel_c_visit, double gumbel_c_scale, double max_visits, double v_mix);
    double calculate_v_mix(double contempt) const;
};