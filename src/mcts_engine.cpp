#include "mcts_engine.hpp"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <algorithm>
#include <random>
#include <thread>
#include <sstream>
#include "board_utils.hpp"

#define NOW() std::chrono::high_resolution_clock::now()
#define ELAPSED(start, end) std::chrono::duration<double>(end - start).count()

MCTSEngine::MCTSEngine(
    int node_pool_capacity, int worker_batch_size, ThreadSafeQueue<std::vector<std::pair<int, int>>>& my_inference_shard, 
    ThreadSafeQueue<std::vector<int>>& result_queue, int worker_id, double virtual_loss,
    double draw_cutoff, double gumbel_c_visit, double gumbel_c_scale, double gumbel_noise, 
    const chess::Board& board, const std::vector<chess::Board>& base_history, Logger& logger, 
    std::vector<torch::Tensor>& shared_input_buffer, 
    std::vector<torch::Tensor>& shared_policy_buffer, std::vector<torch::Tensor>& shared_value_buffer,
    const ModelConfig& model_cfg, 
    ThreadSafeQueue<int>& buffer_free_slots
) : worker_batch_size(worker_batch_size), worker_id(worker_id), virtual_loss(virtual_loss),
    draw_cutoff(draw_cutoff), gumbel_c_visit(gumbel_c_visit), gumbel_c_scale(gumbel_c_scale), 
    gumbel_noise(gumbel_noise), root_board(board), base_history(base_history), 
    node_pool(node_pool_capacity), logger(logger), my_inference_shard(my_inference_shard), result_queue(result_queue), 
    buffer_free_slots(buffer_free_slots), shared_input_buffer(shared_input_buffer), 
    shared_policy_buffer(shared_policy_buffer), shared_value_buffer(shared_value_buffer),
    model_config(model_cfg) 
{
    torch::set_num_threads(1);

    device = torch::kCUDA;
    policy_logits_dtype = torch::kFloat16;

    in_flight_nodes.resize(shared_input_buffer.size(), nullptr);
    root = node_pool.allocate();
    simulation_count = 0;
    inference_sent = 0;
    inference_received = 0;
}

void MCTSEngine::reset(const chess::Board& board, const std::vector<chess::Board>& history) {
    if (!batch_buffer.empty()) {
        _submit_batch();
    }
    
    while (inference_received < inference_sent) {
        std::vector<int> completed_indices = result_queue.pop_wait();
        for (int buffer_index : completed_indices) {
            buffer_free_slots.push(buffer_index);
            inference_received++;
            in_flight_nodes[buffer_index] = nullptr;
        }
    }

    std::vector<int> stray;
    while (result_queue.try_pop(stray)) {
        for (int idx : stray) buffer_free_slots.push(idx);
    }

    std::fill(in_flight_nodes.begin(), in_flight_nodes.end(), nullptr);

    node_pool.reset();
    root_board = board;
    base_history = history;
    root = node_pool.allocate();
    
    simulation_count = 0;
    inference_sent = 0;
    inference_received = 0;
    batch_buffer.clear();

    time_selection = 0.0;
    time_expansion = 0.0;
    time_backpropagation = 0.0;
    time_retrieval = 0.0;
    time_queueing = 0.0;
    time_misc = 0.0;
    time_wait_for_inference = 0.0;
}

void MCTSEngine::_wait_for_inference() {
    auto start_time = NOW();

    if (!batch_buffer.empty()) {
        _submit_batch();
    }

    while (inference_received < inference_sent) {
        std::vector<int> completed_indices = result_queue.pop_wait();

        logger.log("DEBUG", "Received " + std::to_string(completed_indices.size()) + " inferences from batcher.");

        for (int buffer_index : completed_indices) {
            MCTSNode* node = in_flight_nodes[buffer_index];
            in_flight_nodes[buffer_index] = nullptr;
            inference_received++;

            c10::Half* policy_ptr = shared_policy_buffer[buffer_index].data_ptr<c10::Half>();
            float value_output = (float)shared_value_buffer[buffer_index].data_ptr<c10::Half>()[0];

            buffer_free_slots.push(buffer_index);

            if (node != nullptr) {
                if (!node->expanded) {
                    auto exp_start = NOW();
                    for (int i = 0; i < node->num_children; ++i) {
                        MCTSNode* child = node->first_child + i;
                        child->raw_logit = policy_ptr[child->policy_flat_index];
                    }
                    node->expanded = true;
                    time_expansion += ELAPSED(exp_start, NOW());
                }

                _backpropagate(node, (double)value_output, false);
            }
        }
    }
    time_wait_for_inference += ELAPSED(start_time, NOW());
}

MCTSNode* MCTSEngine::_select(MCTSNode* start_node, std::vector<MCTSNode*>& simulation_path) {
    auto start_time = NOW();
    MCTSNode* node = start_node;
    double exp_cache[256];

    while (true) {
        if (node->num_children == 0 || !node->expanded || node->selected) break;

        MCTSNode* best_child = nullptr;
        double best_deficit = -1e20;
        double max_visits = 0.0;
        double sum_visits = 0.0;
        int num_children = node->num_children;

        for (int i = 0; i < num_children; ++i) {
            MCTSNode* child = node->first_child + i;
            if (child->forced_outcome.has_value() && child->forced_outcome.value() == 1) continue;
            if (child->visits > max_visits) max_visits = child->visits;
            sum_visits += child->visits;
        }
        
        double v_mix = node->calculate_v_mix();
        double max_score_logit = -1e20;

        for (int i = 0; i < num_children; ++i) {
            MCTSNode* child = node->first_child + i;
            if (child->forced_outcome.has_value() && child->forced_outcome.value() == 1) continue;
            double score = child->calculate_gumbel_score(gumbel_c_visit, gumbel_c_scale, max_visits, v_mix);
            if (score > max_score_logit) max_score_logit = score;
        }

        double sum_score_exp = 0.0;
        for (int i = 0; i < num_children; ++i) {
            MCTSNode* child = node->first_child + i;
            if (child->forced_outcome.has_value() && child->forced_outcome.value() == 1) {
                exp_cache[i] = 0.0;
                continue;
            }
            exp_cache[i] = std::exp(child->gumbel_score - max_score_logit);
            sum_score_exp += exp_cache[i];
        }

        double inv_sum_visits = 1.0 / (1.0 + sum_visits);
        for (int i = 0; i < num_children; ++i) {
            MCTSNode* child = node->first_child + i;
            if (exp_cache[i] == 0.0) continue;

            double pi_prime = exp_cache[i] / sum_score_exp;
            double child_n_norm = child->visits * inv_sum_visits;
            double deficit = pi_prime - child_n_norm;

            if (deficit > best_deficit) {
                best_deficit = deficit;
                best_child = child;
            }
        }

        if (best_child == nullptr) break;

        root_board.makeMove(best_child->move);
        simulation_path.push_back(best_child);
        node = best_child;
    }
    time_selection += ELAPSED(start_time, NOW());
    return node;
}

void MCTSEngine::_mark_selected(MCTSNode* node) {
    MCTSNode* current_node = node;
    current_node->selected = true;
    MCTSNode* parent = current_node->parent;

    while (parent != nullptr) {
        parent->num_unselected_children -= 1;
        if (parent->num_unselected_children > 0) break;
        parent->selected = true;
        current_node = parent;
        parent = current_node->parent;
    }
}

void MCTSEngine::_unmark_selected(MCTSNode* node) {
    MCTSNode* current_node = node;
    current_node->selected = false;
    MCTSNode* parent = current_node->parent;

    while (parent != nullptr) {
        parent->num_unselected_children += 1;
        if (parent->num_unselected_children == 1) {
            parent->selected = false;
            current_node = parent;
            parent = current_node->parent;
        } else break;
    }
}

void MCTSEngine::_retrieve_inference(bool block) {
    auto start_time = NOW();
    std::vector<int> completed_indices;

    while (true) {
        if (block) {
            completed_indices = result_queue.pop_wait();
            block = false;
        } else {
            if (!result_queue.try_pop(completed_indices)) break;
        }

        for (int buffer_index : completed_indices) {
            MCTSNode* node = in_flight_nodes[buffer_index];
            in_flight_nodes[buffer_index] = nullptr;
            inference_received++;

            c10::Half* policy_ptr = shared_policy_buffer[buffer_index].data_ptr<c10::Half>();
            float value_output = (float)shared_value_buffer[buffer_index].data_ptr<c10::Half>()[0];

            buffer_free_slots.push(buffer_index);

            if (node != nullptr) {
                if (!node->expanded) {
                    auto exp_start = NOW();
                    for (int i = 0; i < node->num_children; ++i) {
                        MCTSNode* child = node->first_child + i;
                        child->raw_logit = policy_ptr[child->policy_flat_index];
                    }
                    node->expanded = true;
                    time_expansion += ELAPSED(exp_start, NOW());
                }
                _backpropagate(node, (double)value_output, false);
            }
        }
    }
    time_retrieval += ELAPSED(start_time, NOW());
}

void MCTSEngine::_submit_batch() {
    auto start_time = NOW();
    int batch_size = batch_buffer.size();
    if (batch_size == 0) return;

    logger.log("DEBUG", "Submitting batch of " + std::to_string(batch_size) + " states to inference queue.");
    my_inference_shard.push(batch_buffer);
    inference_sent += batch_size;
    batch_buffer.clear();

    time_queueing += ELAPSED(start_time, NOW());
}

void MCTSEngine::_handle_terminal_node(MCTSNode* leaf) {
    auto start_time = NOW();
    auto result = root_board.isGameOver(); 
    
    double value = 0.0;
    std::string term_type = "Draw";

    if (result.second == chess::GameResult::WIN) {
        value = -1.0; 
        term_type = "Loss (Mate)";
    } else if (result.second == chess::GameResult::DRAW) {
        value = 0.0;
    }

    logger.log("DEBUG", "Terminal node reached during search. Result: " + term_type);

    _mark_selected(leaf);
    time_expansion += ELAPSED(start_time, NOW());
    
    _backpropagate(leaf, value, true);
    simulation_count++;
}

void MCTSEngine::_queue_leaf_for_inference(MCTSNode* leaf, const std::vector<MCTSNode*>& simulation_path) {
    auto start_time = NOW();
    int buffer_index;

    // 1. Acquire a slot in the shared buffer
    while (!buffer_free_slots.try_pop(buffer_index)) {
        _retrieve_inference(false);
        if (!batch_buffer.empty()) _submit_batch();
        // Reduced sleep to minimize latency now that we are pushing for 100k+
        std::this_thread::sleep_for(std::chrono::microseconds(100)); 
    }

    in_flight_nodes[buffer_index] = leaf;
    _mark_selected(leaf);
    
    // 2. Expand Node (Generate legal moves)
    auto exp_start = NOW();
    chess::Movelist moves;
    chess::movegen::legalmoves(moves, root_board);
    leaf->num_children = moves.size();
    leaf->num_unselected_children = leaf->num_children;

    if (leaf->num_children > 0) {
        leaf->first_child = node_pool.allocate(leaf, moves[0]);
        PolicyComponent pc = move_to_policy_components(moves[0], root_board);
        leaf->first_child->policy_flat_index = policy_components_to_flat_index(pc.row, pc.col, pc.channel);

        for (int i = 1; i < leaf->num_children; ++i) {
            MCTSNode* child = node_pool.allocate(leaf, moves[i]);
            pc = move_to_policy_components(moves[i], root_board);
            child->policy_flat_index = policy_components_to_flat_index(pc.row, pc.col, pc.channel);
        }
    }
    time_expansion += ELAPSED(exp_start, NOW());

    // 3. Construct History (Unmake moves to get past boards)
    std::vector<chess::Board> combined_history;
    std::vector<chess::Move> unmade_moves;

    for (int i = (int)simulation_path.size() - 1; i >= 0 && combined_history.size() < 4; --i) {
        root_board.unmakeMove(simulation_path[i]->move);
        unmade_moves.push_back(simulation_path[i]->move);
        combined_history.push_back(root_board);
    }

    for (size_t i = 0; i < base_history.size() && combined_history.size() < 4; ++i) {
        combined_history.push_back(base_history[i]);
    }

    // Remake moves to restore current board state
    for (int i = (int)unmade_moves.size() - 1; i >= 0; --i) {
        root_board.makeMove(unmade_moves[i]);
    }

    // 4. DIRECT MEMORY WRITE (The "Turbo" switch)
    // No temporary vectors, no LibTorch tensor wrapping.
    c10::Half* destination_ptr = shared_input_buffer[buffer_index].data_ptr<c10::Half>();
    board_to_tensor_69(root_board, combined_history, destination_ptr);

    // 5. Queue for Batcher
    batch_buffer.push_back({worker_id, buffer_index});
    _virtual_loss(leaf, true);

    if (batch_buffer.size() >= (size_t)worker_batch_size) {
        _submit_batch();
    }

    time_misc += ELAPSED(start_time, NOW());
    simulation_count++;
}

void MCTSEngine::_run_single_async_simulation(MCTSNode* start_node) {
    std::vector<MCTSNode*> simulation_path;
    root_board.makeMove(start_node->move);
    simulation_path.push_back(start_node);
    
    int start_path_len = simulation_path.size();

    while (true) {
        _retrieve_inference(false);
        if (batch_buffer.size() >= (size_t)worker_batch_size) _submit_batch();

        if (start_node->selected) {
            if (!batch_buffer.empty()) _submit_batch();
            if (inference_received >= inference_sent) break;
            
            std::this_thread::sleep_for(std::chrono::microseconds(1000));
            continue;
        }

        MCTSNode* leaf = _select(start_node, simulation_path);

        if (root_board.isGameOver().second != chess::GameResult::NONE) {
            _handle_terminal_node(leaf);
            break;
        }

        if (leaf->selected || buffer_free_slots.empty()) {
            if (!batch_buffer.empty()) _submit_batch();

            if (leaf->selected && inference_received >= inference_sent) {
                _backpropagate(leaf, leaf->calculate_v_mix(), false);
                simulation_count++;
                break;
            }

            while (simulation_path.size() > start_path_len) {
                root_board.unmakeMove(simulation_path.back()->move);
                simulation_path.pop_back();
            }
            std::this_thread::sleep_for(std::chrono::microseconds(1000));
            continue;
        }

        _queue_leaf_for_inference(leaf, simulation_path);
        break;
    }

    while (!simulation_path.empty()) {
        root_board.unmakeMove(simulation_path.back()->move);
        simulation_path.pop_back();
    }
}

void MCTSEngine::_log_tournament_results(const std::vector<MCTSNode*>& candidates, const std::string& phase_name) {
    double root_v_mix = root->calculate_v_mix();
    
    logger.log("INFO", ""); 
    logger.log("INFO", "--- " + phase_name + " ---");
    
    std::stringstream rss;
    rss << "Tree Stats: Root v_mix=" << std::fixed << std::setprecision(4) << root_v_mix;
    logger.log("INFO", rss.str());

    char table_header[256];
    snprintf(table_header, sizeof(table_header), 
        "%-8s %8s %8s %8s %8s %8s %8s %8s %8s", 
        "Move", "Visits", "Logit", "Noise", "Raw Q", "Norm Q", "Score", "Outcome", "DTM");
    logger.log("INFO", table_header);
    logger.log("INFO", std::string(95, '-'));
    
    std::vector<MCTSNode*> sorted_cands = candidates;
    std::sort(sorted_cands.begin(), sorted_cands.end(), [](MCTSNode* a, MCTSNode* b) {
        if (a->visits != b->visits) return a->visits > b->visits;
        return a->gumbel_score > b->gumbel_score;
    });

    for (MCTSNode* node : sorted_cands) {
        char line[512];
        std::string outcome_str = node->forced_outcome.has_value() ? std::to_string(node->forced_outcome.value()) : "None";
        std::string dtm_str = node->distance_to_mate.has_value() ? std::to_string(node->distance_to_mate.value()) : "None";

        snprintf(line, sizeof(line), 
            "%-8s %8d %8.4f %8.4f %8.4f %8.4f %8.4f %8s %8s", 
            chess::uci::moveToUci(node->move).c_str(), node->visits, node->raw_logit, node->gumbel_noise,
            node->q_val, node->q_norm, node->gumbel_score, outcome_str.c_str(), dtm_str.c_str()
        );
        logger.log("INFO", line);
    }
    
    logger.log("INFO", std::string(95, '-'));
    logger.log("INFO", ""); 
}

int MCTSEngine::run_simulations(int search_depth, int max_m) {
    logger.log("INFO", "Starting Sequential Halving MCTS. Budget: " + std::to_string(search_depth));

    _queue_leaf_for_inference(root, {}); 
    _submit_batch();
    _wait_for_inference();

    std::vector<MCTSNode*> all_nodes;
    for(int i = 0; i < root->num_children; ++i) {
        all_nodes.push_back(root->first_child + i);
    }
    std::vector<MCTSNode*> active_candidates;

    std::mt19937 gen(1337); 
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    for (MCTSNode* child : all_nodes) {
        double u = dist(gen);
        child->gumbel_noise = -gumbel_noise * std::log(-std::log(u));
        child->gumbel_score = child->gumbel_noise + child->raw_logit;

        root_board.makeMove(child->move);
        if (root_board.isGameOver().second != chess::GameResult::NONE) {
            _handle_terminal_node(child);
        } else {
            active_candidates.push_back(child);
        }
        root_board.unmakeMove(child->move);
    }

    int m = std::min(max_m, (int)active_candidates.size());
    if (m == 0) return simulation_count;

    std::sort(active_candidates.begin(), active_candidates.end(), [](MCTSNode* a, MCTSNode* b) {
        return a->gumbel_score > b->gumbel_score;
    });
    active_candidates.resize(m);

    int num_phases = (m <= 1) ? 1 : std::ceil(std::log2(m));
    int phase_budget = search_depth / num_phases;
    int remaining_search_depth = search_depth;

    for (int phase_idx = 0; phase_idx < num_phases; ++phase_idx) {
        int num_cands = active_candidates.size();
        if (num_cands <= 1) break;

        logger.log("DEBUG", "Starting Phase " + std::to_string(phase_idx) + " with " + std::to_string(num_cands) + " candidates.");

        if (phase_idx == 0) {
            for (MCTSNode* child : active_candidates) {
                remaining_search_depth -= 1;
                root_board.makeMove(child->move);
                if (root_board.isGameOver().second == chess::GameResult::NONE) {
                    _queue_leaf_for_inference(child, {child}); 
                }
                root_board.unmakeMove(child->move);
            }
            _submit_batch();
            _wait_for_inference();
        }

        int current_phase_budget = (phase_idx == num_phases - 1) ? remaining_search_depth : phase_budget;
        if (phase_idx == 0) current_phase_budget = std::max(0, current_phase_budget - num_cands);
        current_phase_budget = std::max(0, std::min(current_phase_budget, remaining_search_depth));

        int active_idx = 0;
        while (current_phase_budget > 0 && num_cands > 0) {
            MCTSNode* child = active_candidates[active_idx];
            
            if (child->forced_outcome.has_value()) {
                active_candidates.erase(active_candidates.begin() + active_idx);
                num_cands = active_candidates.size();
                if (num_cands == 0) break;
                if (active_idx >= num_cands) active_idx = 0;
                continue;
            }

            _run_single_async_simulation(child);
            remaining_search_depth -= 1;
            current_phase_budget -= 1;
            
            active_idx++;
            if (active_idx >= num_cands) active_idx = 0;
        }

        _wait_for_inference();

        double max_visits_phase = 1.0;
        for (MCTSNode* child : active_candidates) {
            if (child->visits > max_visits_phase) max_visits_phase = child->visits;
        }

        double root_v_mix = root->calculate_v_mix();
        for (MCTSNode* child : active_candidates) {
            child->calculate_gumbel_score(gumbel_c_visit, gumbel_c_scale, max_visits_phase, root_v_mix);
        }

        _log_tournament_results(active_candidates, "Phase " + std::to_string(phase_idx) + " End");

        if (num_cands > 1 && phase_idx < (num_phases - 1)) {
            active_candidates.erase(
                std::remove_if(active_candidates.begin(), active_candidates.end(), 
                [](MCTSNode* c) { return c->forced_outcome.has_value() && c->forced_outcome.value() == 1; }),
                active_candidates.end()
            );
            
            if (active_candidates.size() > 1) {
                std::sort(active_candidates.begin(), active_candidates.end(), [](MCTSNode* a, MCTSNode* b) {
                    return a->gumbel_score > b->gumbel_score;
                });
                int cutoff = (active_candidates.size() + 1) / 2;
                active_candidates.resize(cutoff);
            }
        }
    }

    double max_visits_final = 1.0;
    for (MCTSNode* child : all_nodes) {
        if (child->visits > max_visits_final) max_visits_final = child->visits;
    }
    double root_v_mix = root->calculate_v_mix();
    for (MCTSNode* child : all_nodes) {
        child->calculate_gumbel_score(gumbel_c_visit, gumbel_c_scale, max_visits_final, root_v_mix);
    }

    _log_tournament_results(all_nodes, "Final scores");
    logger.log("INFO", "Simulations complete. Total: " + std::to_string(simulation_count));
    logger.log("INFO", "--- Gumbel Search (" + std::to_string(simulation_count) + " sims) Timings ---");
    
    char buffer[128];
    auto log_timer = [&](const char* label, double value) {
        snprintf(buffer, sizeof(buffer), "%-35s %.4f", label, value);
        logger.log("INFO", buffer);
    };

    log_timer("Selection time:", time_selection);
    log_timer("Queueing time:", time_queueing);
    log_timer("Retrieving time:", time_retrieval);
    log_timer("Expansion time:", time_expansion);
    log_timer("Backpropagation time:", time_backpropagation);
    log_timer("Forced waiting for inference time:", time_wait_for_inference);

    if (!batch_buffer.empty() || inference_received < inference_sent) {
        _wait_for_inference();
    }

    return simulation_count;
}

void MCTSEngine::_backpropagate_minimax(MCTSNode* node) {
    if (node->num_children == 0) return;

    int best_win_dtm = 999999;
    int worst_loss_dtm = -1;
    bool all_children_are_wins = true;
    bool has_winning_child = false;

    for (int i = 0; i < node->num_children; ++i) {
        MCTSNode* child = node->first_child + i;
        if (child->forced_outcome.has_value()) {
            if (child->forced_outcome.value() == -1) {
                has_winning_child = true;
                if (child->distance_to_mate.value() < best_win_dtm) best_win_dtm = child->distance_to_mate.value();
            }
            if (child->forced_outcome.value() != 1) {
                all_children_are_wins = false;
            } else {
                if (child->distance_to_mate.value() > worst_loss_dtm) worst_loss_dtm = child->distance_to_mate.value();
            }
        } else {
            all_children_are_wins = false;
        }
    }

    if (has_winning_child) {
        node->forced_outcome = 1;
        node->distance_to_mate = best_win_dtm + 1;
    } else if (all_children_are_wins) {
        node->forced_outcome = -1;
        node->distance_to_mate = worst_loss_dtm + 1;
    } else {
        node->forced_outcome = std::nullopt;
        node->distance_to_mate = std::nullopt;
    }
}

void MCTSEngine::_backpropagate(MCTSNode* node, double value, bool is_terminal) {
    auto start_time = NOW();
    
    if (is_terminal) {
        node->forced_outcome = static_cast<int>(value);
        node->distance_to_mate = 0;
    } else {
        _virtual_loss(node, false);
        _unmark_selected(node);
    }

    MCTSNode* current_node = node;
    double value_for_backprop = value;
    current_node->raw_value = value;

    while (current_node != nullptr) {
        current_node->visits += 1;
        current_node->value_sum += value_for_backprop;
        
        _backpropagate_minimax(current_node);

        value_for_backprop = -value_for_backprop;
        current_node = current_node->parent;
    }
    time_backpropagation += ELAPSED(start_time, NOW());
}

void MCTSEngine::_virtual_loss(MCTSNode* node, bool is_applying) {
    int multiplier = is_applying ? 1 : -1;
    MCTSNode* current_node = node;

    while (current_node != nullptr) {
        current_node->visits += (1 * multiplier);
        current_node->value_sum += (virtual_loss * multiplier);
        current_node = current_node->parent;
    }
}