#include "action_selector.hpp"
#include <iostream>
#include <cmath>
#include <numeric>
#include <algorithm>
#include "board_utils.hpp"
#include <chrono>

ActionSelector::ActionSelector(
    std::string name, int worker_id, ActionSelectorConfig config, 
    const ModelConfig& model_cfg, 
    Logger& logger,
    ThreadSafeQueue<std::vector<std::pair<int, int>>>& i_queue,
    ThreadSafeQueue<std::vector<int>>& r_queue,
    std::vector<torch::Tensor>& in_buffer,
    std::vector<torch::Tensor>& p_buffer,
    std::vector<torch::Tensor>& v_buffer,
    ThreadSafeQueue<int>& free_slots
) : name(name), worker_id(worker_id), config(config), logger(logger),
    model_config(model_cfg),
    inference_queue(i_queue), result_queue(r_queue),
    shared_input_buffer(in_buffer), shared_policy_buffer(p_buffer),
    shared_value_buffer(v_buffer), buffer_free_slots(free_slots) 
{
    std::random_device rd;
    rng.seed(rd());
    
    chess::Board dummy;
    dummy.setFen(chess::constants::STARTPOS);
    mcts = std::make_unique<MCTSEngine>(
        config.node_pool_size, config.batch_size_per_worker, inference_queue, result_queue, worker_id,
        config.virtual_loss, config.draw_cutoff, config.gumbel_c_visit, 
        config.gumbel_c_scale, config.gumbel_noise, dummy, std::vector<chess::Board>(), logger,
        shared_input_buffer, shared_policy_buffer, shared_value_buffer,
        model_config, buffer_free_slots
    );

    reset_for_new_game();
}

void ActionSelector::reset_for_new_game() {
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    use_resignation = dist(rng) < config.resignation_probability;
    logger.log("DEBUG", "Agent state reset. Resignation allowed: " + std::string(use_resignation ? "True" : "False"));
}

SelectionResult ActionSelector::select_action(const chess::Board& board, const std::vector<chess::Board>& history, int ply_count, int gumbel_search_depth, int gumbel_m) {
    SelectionResult result;
    result.policy_vector.resize(model_config.policy_moves, 0.0f);

    int move_number = ((ply_count - 1) / 2) + 1;
    std::string side_str = (board.sideToMove() == chess::Color::WHITE) ? "White" : "Black";
    
    char banner[512];
    snprintf(banner, sizeof(banner), 
        "\n============================================================\n"
        "                    --- MOVE %d: %s, PLY %d STARTED ---\n"
        "============================================================", 
        move_number, side_str.c_str(), ply_count);
    
    logger.log("INFO", banner);
    logger.log("INFO", "Current player: " + name);
    
    auto move_start_time = std::chrono::high_resolution_clock::now();

    mcts->reset(board, history);
    result.simulation_count = mcts->run_simulations(gumbel_search_depth, gumbel_m);

    // FETCHING THE CHILDREN CONTIGUOUSLY
    std::vector<MCTSNode*> all_children;
    for(int i = 0; i < mcts->root->num_children; ++i) {
        all_children.push_back(mcts->root->first_child + i);
    }

    int num_children = all_children.size();
    if (num_children == 0) return result; 

    chess::Movelist all_moves;
    std::vector<float> base_logits(num_children);
    float max_logit = -1e20f;

    for (int i = 0; i < num_children; ++i) {
        all_moves.add(all_children[i]->move);
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

    double root_v_mix = mcts->root->calculate_v_mix();
    std::vector<MCTSNode*> winning_nodes, losing_nodes, draw_nodes, non_forced_nodes, non_forced_visited;
    
    for (MCTSNode* child : all_children) {
        if (child->forced_outcome.has_value()) {
            if (child->forced_outcome.value() == -1) winning_nodes.push_back(child);
            else if (child->forced_outcome.value() == 1) losing_nodes.push_back(child);
            else draw_nodes.push_back(child);
        } else {
            non_forced_nodes.push_back(child);
            if (child->visits > 0) non_forced_visited.push_back(child);
        }
    }

    std::vector<float> final_probs(num_children, 0.0f);
    std::vector<float> minimax_probs(num_children, 0.0f);
    
    if (!winning_nodes.empty()) {
        int min_dtm = 999999;
        for (MCTSNode* child : winning_nodes) {
            if (child->distance_to_mate.value() < min_dtm) min_dtm = child->distance_to_mate.value();
        }
        
        int count_best = 0;
        for (MCTSNode* child : winning_nodes) if (child->distance_to_mate.value() == min_dtm) count_best++;
        
        float prob_per_best = 1.0f / count_best;
        for (int i = 0; i < num_children; ++i) {
            if (all_children[i]->forced_outcome.has_value() && 
                all_children[i]->forced_outcome.value() == -1 && 
                all_children[i]->distance_to_mate.value() == min_dtm) {
                minimax_probs[i] = prob_per_best;
            }
        }
        logger.log("INFO", std::to_string(count_best) + " fastest win(s) found (DTM " + std::to_string(min_dtm) + ").");
    } else if (!draw_nodes.empty() && root_v_mix <= config.draw_cutoff) {
        float prob_per_best = 1.0f / draw_nodes.size();
        for (int i = 0; i < num_children; ++i) {
            if (all_children[i]->forced_outcome.has_value() && all_children[i]->forced_outcome.value() == 0) {
                minimax_probs[i] = prob_per_best;
            }
        }
        logger.log("INFO", "Forced draw condition met for " + std::to_string(draw_nodes.size()) + " nodes.");
    } else if (!losing_nodes.empty() && !non_forced_nodes.empty()) {
        float prob_per_best = 1.0f / non_forced_nodes.size();
        for (int i = 0; i < num_children; ++i) {
            if (!all_children[i]->forced_outcome.has_value()) {
                minimax_probs[i] = prob_per_best;
            }
        }
        logger.log("INFO", std::to_string(losing_nodes.size()) + " forced loss(es) found. Smoothing over safe moves.");
    } else {
        minimax_probs = base_probs; 
    }

    for (int i = 0; i < num_children; ++i) {
        final_probs[i] = (1.0f - config.minimax_smoothing_factor) * base_probs[i] + 
                         (config.minimax_smoothing_factor * minimax_probs[i]);
    }

    if (!winning_nodes.empty()) {
        logger.log("DEBUG", "Applying Move Rule A: Selecting lowest DTM win.");
        int min_dtm = 999999;
        for (MCTSNode* c : winning_nodes) if (c->distance_to_mate.value() < min_dtm) min_dtm = c->distance_to_mate.value();
        std::vector<chess::Move> best_moves;
        for (MCTSNode* c : winning_nodes) if (c->distance_to_mate.value() == min_dtm) best_moves.push_back(c->move);
        
        std::uniform_int_distribution<> dist(0, best_moves.size() - 1);
        result.best_move = best_moves[dist(rng)];

    } else if (!draw_nodes.empty() && root_v_mix <= config.draw_cutoff) {
        logger.log("DEBUG", "Applying Move Rule B: Selecting forced draw to avoid loss.");
        std::uniform_int_distribution<> dist(0, draw_nodes.size() - 1);
        result.best_move = draw_nodes[dist(rng)]->move;

    } else if (!non_forced_visited.empty()) {
        logger.log("DEBUG", "Applying Move Rule C: Normal Selection from safe visited nodes.");
        if (ply_count <= config.temperature_ply_cutoff) {
            std::sort(non_forced_visited.begin(), non_forced_visited.end(), [](MCTSNode* a, MCTSNode* b) {
                if (a->visits != b->visits) return a->visits > b->visits;
                return a->gumbel_score > b->gumbel_score;
            });

            MCTSNode* top_node = non_forced_visited[0];
            if (non_forced_visited.size() > 1 && non_forced_visited[1]->gumbel_score > top_node->gumbel_score) {
                top_node = non_forced_visited[1];
            }

            double best_q_val = -top_node->calculate_v_mix();
            std::vector<MCTSNode*> valid_nodes;
            double sum_other_visits = 0.0;

            for (MCTSNode* node : non_forced_visited) {
                if ((best_q_val - (-node->calculate_v_mix())) <= config.temperature_blunder_threshold) {
                    valid_nodes.push_back(node);
                    if (node != top_node) sum_other_visits += node->visits;
                }
            }

            std::vector<double> act_probs(valid_nodes.size(), 0.0);
            double top_prob = config.temperature_top_move;
            double remaining_prob = 1.0 - top_prob;
            
            int top_idx_in_valid = -1;
            for(size_t i = 0; i < valid_nodes.size(); ++i) {
                if(valid_nodes[i] == top_node) top_idx_in_valid = i;
            }

            if (remaining_prob > 0.0 && valid_nodes.size() > 1) {
                if (sum_other_visits > 0.0) {
                    act_probs[top_idx_in_valid] = top_prob;
                    for (size_t i = 0; i < valid_nodes.size(); ++i) {
                        if (i != top_idx_in_valid) {
                            act_probs[i] = (valid_nodes[i]->visits / sum_other_visits) * remaining_prob;
                        }
                    }
                } else {
                    act_probs[top_idx_in_valid] = 1.0;
                }
            } else {
                act_probs[top_idx_in_valid] = 1.0;
            }

            std::discrete_distribution<> d(act_probs.begin(), act_probs.end());
            result.best_move = valid_nodes[d(rng)]->move;
        } else {
            logger.log("DEBUG", "Late game detected. Applying greedy selection.");
            std::sort(non_forced_visited.begin(), non_forced_visited.end(), [](MCTSNode* a, MCTSNode* b) {
                if (a->visits != b->visits) return a->visits > b->visits;
                return a->gumbel_score > b->gumbel_score;
            });
            MCTSNode* m1 = non_forced_visited[0];
            MCTSNode* m2 = (non_forced_visited.size() > 1) ? non_forced_visited[1] : m1;
            result.best_move = (m1->gumbel_score > m2->gumbel_score) ? m1->move : m2->move;
        }
    } else {
        logger.log("DEBUG", "Applying Move Rule D: Forced into bad state.");
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

    if (use_resignation && root_v_mix < config.resignation_cutoff) {
        logger.log("INFO", "Root Value (" + std::to_string(root_v_mix) + ") is below cutoff. Triggering Resignation.");
        result.resigned = true;
        result.best_move = chess::Move::NO_MOVE;
        std::fill(result.policy_vector.begin(), result.policy_vector.end(), 0.0f);
        return result;
    }

    map_policy_to_global_vector(all_moves, final_probs.data(), board, result.policy_vector.data());
    
    result.entropy = 0.0;
    for (float p : final_probs) {
        if (p > 0) result.entropy -= (p * std::log(p + 1e-10f));
    }

    auto move_end_time = std::chrono::high_resolution_clock::now();
    double total_move_time = std::chrono::duration<double>(move_end_time - move_start_time).count();
    double sim_speed = (total_move_time > 0) ? (result.simulation_count / total_move_time) : 0.0;

    char buffer[256];
    snprintf(buffer, sizeof(buffer), "Time: %.4fs | Speed: %.1f sim/s | Entropy: %.4f | Value: %.4f", 
             total_move_time, sim_speed, result.entropy, root_v_mix);
    
    logger.log("INFO", buffer);

    return result;
}