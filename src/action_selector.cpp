#include "action_selector.hpp"
#include <iostream>
#include <cmath>
#include <numeric>
#include <algorithm>
#include <chrono>

ActionSelector::ActionSelector(
    std::string name, int worker_id, ActionSelectorConfig config, Logger& logger
) : name(name), worker_id(worker_id), config(config), logger(logger) {
    std::random_device rd;
    auto time_seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    rng.seed(rd() ^ worker_id ^ time_seed);
    reset_for_new_game();
}

void ActionSelector::reset_for_new_game() {
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    use_resignation = dist(rng) < config.resignation_probability;
    logger.log("DEBUG", "Agent state reset. Resignation allowed: " + std::string(use_resignation ? "True" : "False"));
}

SelectionResult ActionSelector::select_move(MCTSNode* root, int ply_count) {
    SelectionResult result;
    
    int num_children = root->num_children;
    if (num_children == 0) return result;

    std::vector<MCTSNode*> all_children;
    for(int i = 0; i < num_children; ++i) {
        all_children.push_back(root->first_child + i);
    }

    std::vector<MCTSNode*> winning_nodes, losing_nodes, draw_nodes, non_forced_visited;
    for (MCTSNode* child : all_children) {
        if (child->forced_outcome.has_value()) {
            if (child->forced_outcome.value() == -1) winning_nodes.push_back(child);
            else if (child->forced_outcome.value() == 1) losing_nodes.push_back(child);
            else draw_nodes.push_back(child);
        } else {
            if (child->visits > 0) non_forced_visited.push_back(child);
        }
    }

    MCTSNode* top_node;
    double best_q;

    if (!non_forced_visited.empty()) {
        std::sort(non_forced_visited.begin(), non_forced_visited.end(), [](MCTSNode* a, MCTSNode* b) {
            if (a->visits != b->visits) return a->visits > b->visits;
            return a->gumbel_score > b->gumbel_score;
        });
        MCTSNode* m1 = non_forced_visited[0];
        MCTSNode* m2 = (non_forced_visited.size() > 1) ? non_forced_visited[1] : m1;
        top_node = (m1->gumbel_score > m2->gumbel_score) ? m1 : m2;
        best_q = -top_node->calculate_v_mix();
    }

    // Rule A: Win
    if (!winning_nodes.empty()) {
        int min_dtm = 999999;
        for (MCTSNode* c : winning_nodes) if (c->distance_to_mate.value() < min_dtm) min_dtm = c->distance_to_mate.value();
        std::vector<chess::Move> best_moves;
        for (MCTSNode* c : winning_nodes) if (c->distance_to_mate.value() == min_dtm) best_moves.push_back(c->move);
        
        std::uniform_int_distribution<> dist(0, best_moves.size() - 1);
        result.best_move = best_moves[dist(rng)];

    // Rule B: Draw
    } else if (!draw_nodes.empty() && best_q <= config.draw_cutoff) {
        std::uniform_int_distribution<> dist(0, draw_nodes.size() - 1);
        result.best_move = draw_nodes[dist(rng)]->move;

    // Rule C: Temperature / Safe Moves
    } else if (!non_forced_visited.empty()) {
        if (ply_count <= config.temperature_ply_cutoff) {

            std::vector<MCTSNode*> valid_nodes;
            
            double sum_other_visits = 0.0;
            double total_valid_visits = top_node->visits;

            for (MCTSNode* node : non_forced_visited) {
                // 1. Calculate the absolute Q-drop for this candidate
                double q_drop = best_q - (-node->calculate_v_mix());
                
                // 2. Calculate the dynamic threshold using the candidate's noise
                double dynamic_threshold = config.temperature_blunder_q_threshold + 
                                           (config.temperature_blunder_noise_weight * node->gumbel_noise);
                
                // 3. Clamp it so negative noise doesn't artificially shrink the threshold below baseline
                dynamic_threshold = std::max(config.temperature_blunder_q_threshold, dynamic_threshold);

                // 4. Evaluate against the dynamic threshold
                if (q_drop <= dynamic_threshold) {
                    valid_nodes.push_back(node);
                    if (node != top_node) {
                        sum_other_visits += node->visits;
                        total_valid_visits += node->visits;
                    }
                }
            }

            std::vector<double> act_probs(valid_nodes.size(), 0.0);
            int top_idx_in_valid = -1;
            for(size_t i = 0; i < valid_nodes.size(); ++i) {
                if(valid_nodes[i] == top_node) top_idx_in_valid = i;
            }

            if (valid_nodes.size() > 1 && sum_other_visits > 0.0) {
                // NEW LOGIC: Calculate natural probability, apply config as a floor
                double natural_top_prob = static_cast<double>(top_node->visits) / total_valid_visits;
                double actual_top_prob = std::max(config.top_move_probability, natural_top_prob);
                double remaining_prob = 1.0 - actual_top_prob;

                act_probs[top_idx_in_valid] = actual_top_prob;
                for (size_t i = 0; i < valid_nodes.size(); ++i) {
                    if (i != top_idx_in_valid) {
                        act_probs[i] = (valid_nodes[i]->visits / sum_other_visits) * remaining_prob;
                    }
                }
            } else {
                act_probs[top_idx_in_valid] = 1.0;
            }

            std::discrete_distribution<> d(act_probs.begin(), act_probs.end());
            result.best_move = valid_nodes[d(rng)]->move;
        } else {
            // [Existing non-temperature logic remains the same]
            std::sort(non_forced_visited.begin(), non_forced_visited.end(), [](MCTSNode* a, MCTSNode* b) {
                if (a->visits != b->visits) return a->visits > b->visits;
                return a->gumbel_score > b->gumbel_score;
            });
            MCTSNode* m1 = non_forced_visited[0];
            MCTSNode* m2 = (non_forced_visited.size() > 1) ? non_forced_visited[1] : m1;
            result.best_move = (m1->gumbel_score > m2->gumbel_score) ? m1->move : m2->move;
        }
        
    // Rule D: Delay Mate
    } else {
        if (!draw_nodes.empty()) {
            std::uniform_int_distribution<> dist(0, draw_nodes.size() - 1);
            result.best_move = draw_nodes[dist(rng)]->move;
        } else if (!losing_nodes.empty()) {
            MCTSNode* best_delay = losing_nodes[0];
            for (MCTSNode* c : losing_nodes) {
                if (c->distance_to_mate.value() > best_delay->distance_to_mate.value()) best_delay = c;
            }
            result.best_move = best_delay->move;
        } else {
            std::uniform_int_distribution<> dist(0, num_children - 1);
            result.best_move = all_children[dist(rng)]->move;
        }
    }

    if (use_resignation && best_q < config.resignation_cutoff) {
        logger.log("INFO", "Best Value (" + std::to_string(best_q) + ") is below cutoff. Triggering Resignation.");
        result.resigned = true;
        result.best_move = chess::Move::NO_MOVE;
    }

    return result;
}