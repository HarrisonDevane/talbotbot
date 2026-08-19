// =============================================================================
// gumbel_mcts.cpp
//
// Gumbel-specific MCTS implementation. Every method here was lifted from the
// original mcts_engine.cpp with one substantive edit: _select no longer has
// a PUCT branch. That branch has moved to PuctMCTS. Since cpuct is no longer
// a field on this engine, there's nothing to branch on.
// =============================================================================

#include "gumbel_mcts.hpp"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <algorithm>
#include <random>
#include <thread>
#include <sstream>
#include <chrono>
#include "board_utils.hpp"

#define NOW() std::chrono::high_resolution_clock::now()
#define ELAPSED(start, end) std::chrono::duration<double>(end - start).count()

GumbelMCTS::GumbelMCTS(
    int node_pool_capacity,
    int worker_batch_size,
    moodycamel::ConcurrentQueue<std::pair<int, int>>& inference_queue,
    ThreadSafeQueue<std::vector<int>>& result_queue,
    int worker_id,
    double virtual_loss,
    double contempt,
    double policy_softmax_temp,
    double gumbel_c_visit,
    double gumbel_c_scale,
    double gumbel_noise,
    const chess::Board& board,
    const std::vector<chess::Board>& base_history,
    Logger& logger,
    std::vector<torch::Tensor>& shared_input_buffer,
    std::vector<torch::Tensor>& shared_policy_buffer,
    std::vector<torch::Tensor>& shared_value_buffer,
    ThreadSafeQueue<int>& buffer_free_slots,
    std::atomic<int>* core_wait_count,
    int workers_per_core,
    bool two_fold_repetition,
    bool use_tablebase
) : MctsBase(
        node_pool_capacity, worker_batch_size, inference_queue, result_queue,
        worker_id, virtual_loss, contempt, policy_softmax_temp, board, base_history, logger,
        shared_input_buffer, shared_policy_buffer, shared_value_buffer,
        buffer_free_slots, core_wait_count, workers_per_core,
        two_fold_repetition, use_tablebase
    ),
    gumbel_c_visit(gumbel_c_visit),
    gumbel_c_scale(gumbel_c_scale),
    gumbel_noise(gumbel_noise)
{
    std::random_device rd;
    rng.seed(rd() ^ worker_id ^ std::chrono::high_resolution_clock::now().time_since_epoch().count());
}

// -----------------------------------------------------------------------------
// _select: deficit-Gumbel selection at every node.
//
// pi'(a) = softmax(gumbel_score(a))
// choose argmax_a { pi'(a) - N(a) / (1 + sum_N) }
// -----------------------------------------------------------------------------
MCTSNode* GumbelMCTS::_select(MCTSNode* start_node, std::vector<MCTSNode*>& simulation_path) {
    auto start_time = NOW();
    MCTSNode* node = start_node;
    double exp_cache[256];

    while (true) {
        if (node->num_children == 0 || !node->is_expanded() ||
            node->is_unavailable() || node->has_forced_outcome()) break;

        MCTSNode* best_child   = nullptr;
        double    best_deficit = -1e20;
        double    max_visits   = 0.0;
        double    sum_visits   = 0.0;
        int       num_children = node->num_children;

        for (int i = 0; i < num_children; ++i) {
            MCTSNode* child = node->first_child + i;
            if (child->has_forced_outcome() || child->is_unavailable()) continue;
            if (child->visits > max_visits) max_visits = child->visits;
            sum_visits += child->visits;
        }

        double v_mix           = node->calculate_v_mix(contempt);
        double max_score_logit = -1e20;

        for (int i = 0; i < num_children; ++i) {
            MCTSNode* child = node->first_child + i;
            if (child->has_forced_outcome() || child->is_unavailable()) continue;
            double score = child->calculate_gumbel_score(contempt, gumbel_c_visit,
                                                         gumbel_c_scale, max_visits, v_mix);
            if (score > max_score_logit) max_score_logit = score;
        }

        double sum_score_exp = 0.0;
        for (int i = 0; i < num_children; ++i) {
            MCTSNode* child = node->first_child + i;
            if (child->has_forced_outcome() || child->is_unavailable()) {
                exp_cache[i] = 0.0;
                continue;
            }
            exp_cache[i] = std::exp(child->gumbel_score - max_score_logit);
            sum_score_exp += exp_cache[i];
        }

        double inv_sum_visits = 1.0 / (1.0 + sum_visits);
        double inv_sum_score  = 1.0 / sum_score_exp;
        for (int i = 0; i < num_children; ++i) {
            MCTSNode* child = node->first_child + i;
            if (exp_cache[i] == 0.0) continue;

            double pi_prime     = exp_cache[i] * inv_sum_score;
            double child_n_norm = child->visits * inv_sum_visits;
            double deficit      = pi_prime - child_n_norm;

            if (deficit > best_deficit) {
                best_deficit = deficit;
                best_child   = child;
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

// -----------------------------------------------------------------------------
// _run_single_async_simulation: one simulation from `start_node`. Wraps
// selection + terminal/tablebase handling + leaf queueing. Returns true iff
// a simulation was actually performed (caller only charges budget then).
// -----------------------------------------------------------------------------
bool GumbelMCTS::_run_single_async_simulation(MCTSNode* start_node) {
    std::vector<MCTSNode*> simulation_path;
    root_board.makeMove(start_node->move);
    simulation_path.push_back(start_node);

    bool completed = false;

    int loop_iterations = 0;
    int unavailable_continues = 0;
    int select_unavailable_continues = 0;

    while (true) {
        loop_iterations++;
        _retrieve_inference(false);
        if (batch_buffer.size() >= (size_t)worker_batch_size) {
            _spin_wait(
                [&]() { return inference_sent > inference_received; },
                [&]() { _retrieve_inference(true); }
            );
            _submit_batch();
        }

        if (start_node->is_unavailable() || buffer_free_slots.empty()) {
            if (start_node->is_unavailable()) unavailable_continues++;
            if (!batch_buffer.empty()) _submit_batch();
            if (inference_received >= inference_sent) {
                logger.log("WARNING", "No-op sim exit: unavailable=" +
                           std::to_string(start_node->is_unavailable()) +
                           " slots_empty=" + std::to_string(buffer_free_slots.empty()) +
                           " unavailable_continues=" + std::to_string(unavailable_continues) +
                           " select_unavailable_continues=" + std::to_string(select_unavailable_continues));
                break;
            }
            _retrieve_inference(true);
            continue;
        }

        MCTSNode* leaf = _select(start_node, simulation_path);

        if (logger.get_level() <= 10) {
            std::string path_str = "";
            std::string root_move = "";
            MCTSNode* curr = leaf;
            while (curr != nullptr && curr->move != chess::Move::NO_MOVE) {
                std::string uci = chess::uci::moveToUci(curr->move);
                path_str = uci + (path_str.empty() ? "" : " ") + path_str;
                root_move = uci;
                curr = curr->parent;
            }
            if (root_move == "e3h6") {
                logger.log("DEBUG", "Selected path: " + path_str);
            }
        }

        if (root_board.isGameOver().second != chess::GameResult::NONE || root_board.isRepetition(two_fold_repetition ? 1 : 2)) {
            _handle_terminal_node(leaf);
            completed = true;
            break;
        }

        if (use_tablebase && _try_tablebase(leaf)) {
            completed = true;
            break;
        }

        if (leaf->is_expanded()) {
            logger.log("WARNING", "_select returned an already-expanded interior node (" +
                       chess::uci::moveToUci(leaf->move) + "); skipping re-queue.");
            while (simulation_path.size() > 1) {
                root_board.unmakeMove(simulation_path.back()->move);
                simulation_path.pop_back();
            }
            if (!batch_buffer.empty()) _submit_batch();
            if (inference_received >= inference_sent) break;
            _retrieve_inference(true);
            continue;
        }

        if (start_node->is_unavailable()) {
            select_unavailable_continues++;
            while (simulation_path.size() > 1) {
                root_board.unmakeMove(simulation_path.back()->move);
                simulation_path.pop_back();
            }
            continue;
        }

        _queue_leaf_for_inference(leaf, simulation_path);
        completed = true;
        break;
    }

    while (!simulation_path.empty()) {
        root_board.unmakeMove(simulation_path.back()->move);
        simulation_path.pop_back();
    }
    return completed;
}

// -----------------------------------------------------------------------------
// Assign gumbel noise/scores to every root child, route terminals out, and
// build the active candidate set sorted+truncated to m. Returns m (0 => none).
// -----------------------------------------------------------------------------
int GumbelMCTS::_build_candidates(int max_m, std::vector<MCTSNode*>& all_nodes,
                                  std::vector<MCTSNode*>& active_candidates) {
    all_nodes.clear();
    for (int i = 0; i < root->num_children; ++i) {
        all_nodes.push_back(root->first_child + i);
    }
    active_candidates.clear();

    std::uniform_real_distribution<double> dist(0.0, 1.0);
    for (MCTSNode* child : all_nodes) {
        double u = dist(rng);
        child->gumbel_noise = -gumbel_noise * std::log(-std::log(u));
        child->gumbel_score = child->gumbel_noise + child->raw_logit;

        root_board.makeMove(child->move);
        if (root_board.isGameOver().second != chess::GameResult::NONE || root_board.isRepetition(two_fold_repetition ? 1 : 2)) {
            _handle_terminal_node(child);
        } else {
            active_candidates.push_back(child);
        }
        root_board.unmakeMove(child->move);
    }

    int m = std::min(max_m, (int)active_candidates.size());
    if (m == 0) return 0;

    std::sort(active_candidates.begin(), active_candidates.end(), [](MCTSNode* a, MCTSNode* b) {
        return a->gumbel_score > b->gumbel_score;
    });
    active_candidates.resize(m);
    return m;
}

// -----------------------------------------------------------------------------
// Round 0: one simulation against each candidate, drain, log. Decrements the
// caller's remaining budget by the number of candidates touched.
// -----------------------------------------------------------------------------
void GumbelMCTS::_run_round0(std::vector<MCTSNode*>& active_candidates, int& remaining_search_depth) {
    int ply_count = ((root_board.fullMoveNumber() - 1) * 2) + (root_board.sideToMove() == chess::Color::BLACK ? 2 : 1);

    int current_move = (ply_count + 1) / 2;
    std::string color = (root_board.sideToMove() == chess::Color::WHITE) ? "WHITE" : "BLACK";

    logger.log("INFO", "===============================================================================================");
    logger.log("INFO", " MOVE " + std::to_string(current_move) + " | PLY " + std::to_string(ply_count) + " | " + color);
    logger.log("INFO", "===============================================================================================");

    for (MCTSNode* child : active_candidates) {
        remaining_search_depth -= 1;
        root_board.makeMove(child->move);
        if (root_board.isGameOver().second == chess::GameResult::NONE && !root_board.isRepetition(two_fold_repetition ? 1 : 2)) {
            // Syzygy probe for root children -- see original comment.
            if (!(use_tablebase && _try_tablebase(child))) {
                _queue_leaf_for_inference(child, {child});
            }
        }
        root_board.unmakeMove(child->move);
    }
    _submit_batch();
    while (inference_received < inference_sent) {
        _retrieve_inference(true);
    }
}

// -----------------------------------------------------------------------------
// Recompute gumbel scores for a node set against its current max-visit count.
// -----------------------------------------------------------------------------
void GumbelMCTS::_rescore(std::vector<MCTSNode*>& nodes) {
    double max_visits = 1.0;
    for (MCTSNode* child : nodes) {
        if (child->visits > max_visits) max_visits = child->visits;
    }
    double root_v_mix = root->calculate_v_mix(contempt);
    for (MCTSNode* child : nodes) {
        child->calculate_gumbel_score(contempt, gumbel_c_visit, gumbel_c_scale, max_visits, root_v_mix);
    }
}

// -----------------------------------------------------------------------------
// Sequential-halving cut: drop proven-loss-for-us candidates, then keep the
// top half by gumbel score.
// -----------------------------------------------------------------------------
void GumbelMCTS::_halve(std::vector<MCTSNode*>& active_candidates) {
    active_candidates.erase(
        std::remove_if(active_candidates.begin(), active_candidates.end(),
        [](MCTSNode* c) { return c->has_forced_outcome(); }),
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

// -----------------------------------------------------------------------------
// _log_tournament_results / _log_node_by_path: diagnostic dumps that read
// gumbel_score. Kept in this file because the interpretation is gumbel-specific
// (PUCT wouldn't have a meaningful "Score" column here).
// -----------------------------------------------------------------------------
void GumbelMCTS::_log_tournament_results(const std::vector<MCTSNode*>& candidates,
                                         const std::string& phase_name,
                                         int remaining_search_depth,
                                         int phase_budget,
                                         int sims_completed) {
    if (logger.get_level() > 20) return;

    double root_v_mix = root->calculate_v_mix(contempt);

    logger.log("INFO", "");
    logger.log("INFO", "--- " + phase_name + " ---");

    std::stringstream rss;
    rss << "Tree Stats: Root v_mix=" << std::fixed << std::setprecision(4) << root_v_mix;
    logger.log("INFO", rss.str());

    {
        int active = 0, forced = 0, total_visits = 0;
        for (MCTSNode* n : candidates) {
            if (n->has_forced_outcome()) forced++; else active++;
            total_visits += n->visits;
        }
        char bud[256];
        snprintf(bud, sizeof(bud),
            "Budget: remaining=%d phase_budget=%d sims_completed=%d | "
            "cands=%d (active=%d forced=%d) sum_visits=%d",
            remaining_search_depth, phase_budget, sims_completed,
            (int)candidates.size(), active, forced, total_visits);
        logger.log("INFO", bud);
    }

    char table_header[256];
    snprintf(table_header, sizeof(table_header),
        "%-8s %8s %8s %8s %8s %8s %8s %8s %8s %8s",
        "Move", "Logit", "Visits", "Win%", "Draw%", "Loss%", "Norm Q", "Score", "Outcome", "DTM");
    logger.log("INFO", table_header);
    logger.log("INFO", std::string(95, '-'));

    std::vector<MCTSNode*> sorted_cands = candidates;
    std::sort(sorted_cands.begin(), sorted_cands.end(), [](MCTSNode* a, MCTSNode* b) {
        if (a->visits != b->visits) return a->visits > b->visits;
        return a->gumbel_score > b->gumbel_score;
    });

    for (MCTSNode* node : sorted_cands) {
        char line[512];
        std::string outcome_str = node->has_forced_outcome() ? std::to_string(node->forced_outcome) : "None";
        std::string dtm_str = node->has_forced_outcome() ? std::to_string(node->distance_to_mate) : "None";

        double w_pct = (node->visits > 0) ? (node->l_sum / node->visits) * 100.0 : node->raw_l * 100.0;
        double d_pct = (node->visits > 0) ? (node->d_sum / node->visits) * 100.0 : node->raw_d * 100.0;
        double l_pct = (node->visits > 0) ? (node->w_sum / node->visits) * 100.0 : node->raw_w * 100.0;

        double q_val = (node->visits > 0) ? -node->expected_value(contempt) : root_v_mix;
        double q_norm = (q_val + 1.0) / 2.0;

        snprintf(line, sizeof(line),
            "%-8s %8.4f %8d %8.1f %8.1f %8.1f %8.4f %8.4f %8s %8s",
            chess::uci::moveToUci(node->move).c_str(), node->raw_logit, node->visits,
            w_pct, d_pct, l_pct, q_norm, node->gumbel_score, outcome_str.c_str(), dtm_str.c_str());
        logger.log("INFO", line);
    }

    logger.log("INFO", std::string(95, '-'));
    logger.log("INFO", "");
}

// _log_node_by_path lives in MctsBase (shared debug helper).

// -----------------------------------------------------------------------------
// Self-play / fixed-budget path. Full sequential-halving schedule to completion.
// -----------------------------------------------------------------------------
int GumbelMCTS::run_simulations_fixed(int search_depth, int max_m) {
    if (logger.get_level() <= 20) {
        logger.log("INFO", "Starting Sequential Halving MCTS. Budget: " + std::to_string(search_depth));
    }

    const auto wall_start = std::chrono::steady_clock::now();
    _expand_root();

    std::vector<MCTSNode*> all_nodes;
    std::vector<MCTSNode*> active_candidates;
    int m = _build_candidates(max_m, all_nodes, active_candidates);
    if (m == 0) return simulation_count;

    int remaining_search_depth = search_depth;
    bool did_round0 = false;
    int r0_spent = 0;
    int phase_idx = 0;

    while (active_candidates.size() > 1 && remaining_search_depth > 0) {
        int num_cands = active_candidates.size();

        if (!did_round0) {
            int before = remaining_search_depth;
            _run_round0(active_candidates, remaining_search_depth);
            r0_spent = before - remaining_search_depth;
            did_round0 = true;
            active_candidates.erase(
                std::remove_if(active_candidates.begin(), active_candidates.end(),
                    [](MCTSNode* c){ return c->has_forced_outcome(); }),
                active_candidates.end());
            num_cands = active_candidates.size();
            if (num_cands <= 1) break;
        }

        int phases_left = std::max(1, (int)std::ceil(std::log2((double)num_cands)));
        int current_phase_budget;
        if (phases_left <= 1) {
            current_phase_budget = remaining_search_depth;
        } else {
            int pool = remaining_search_depth + (phase_idx == 0 ? r0_spent : 0);
            current_phase_budget = pool / phases_left;
            if (phase_idx == 0) current_phase_budget -= r0_spent;
        }
        current_phase_budget = std::max(0, std::min(current_phase_budget, remaining_search_depth));

        int active_idx = 0;
        int no_progress_streak = 0;
        while (current_phase_budget > 0 && num_cands > 0) {
            MCTSNode* child = active_candidates[active_idx];

            if (child->has_forced_outcome()) {
                active_candidates.erase(active_candidates.begin() + active_idx);
                num_cands = active_candidates.size();
                if (num_cands == 0) break;
                if (active_idx >= num_cands) active_idx = 0;
                continue;
            }

            if (_run_single_async_simulation(child)) {
                remaining_search_depth -= 1;
                current_phase_budget -= 1;
                no_progress_streak = 0;
            } else {
                no_progress_streak += 1;
                if (no_progress_streak >= num_cands) {
                    logger.log("WARNING", "Phase stalled: all " + std::to_string(num_cands) +
                               " candidates returned no-op with nothing in flight. Ending phase early (budget left: " +
                               std::to_string(current_phase_budget) + ").");
                    break;
                }
            }

            active_idx++;
            if (active_idx >= num_cands) active_idx = 0;
        }

        while (inference_received < inference_sent) {
            _retrieve_inference(true);
        }

        _rescore(active_candidates);
        _log_tournament_results(active_candidates,
                        "Phase " + std::to_string(phase_idx) + " End",
                        remaining_search_depth, current_phase_budget, simulation_count);

        if (active_candidates.size() > 2) {
            _halve(active_candidates);
        } else {
            break;
        }
        phase_idx++;
    }

    _rescore(all_nodes);
    _log_tournament_results(all_nodes, "Final scores");

    if (logger.get_level() <= 20) {
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
    }

    _flush_inflight();
    _record_nps(simulation_count, std::chrono::duration<double>(std::chrono::steady_clock::now() - wall_start).count());
    return simulation_count;
}

// Gumbel is training-only; there is no timed variant. The play path (PUCT)
// owns clock-based search entirely.