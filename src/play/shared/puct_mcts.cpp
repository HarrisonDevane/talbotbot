// =============================================================================
// puct_mcts.cpp
//
// PUCT selection + flat (non-halving) simulation loops.
//
// _select's body is lifted from the PUCT branch of the original
// mcts_engine.cpp::_select (which contained both PUCT and gumbel branches).
// _run_single_async_simulation mirrors GumbelMCTS's, but starts from root
// (not from a candidate child) since there is no candidate set here.
// =============================================================================

#include "puct_mcts.hpp"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <algorithm>
#include <thread>
#include <sstream>
#include <chrono>
#include "board_utils.hpp"

#define NOW() std::chrono::high_resolution_clock::now()
#define ELAPSED(start, end) std::chrono::duration<double>(end - start).count()

PuctMCTS::PuctMCTS(
    int node_pool_capacity,
    int worker_batch_size,
    moodycamel::ConcurrentQueue<std::pair<int, int>>& inference_queue,
    ThreadSafeQueue<std::vector<int>>& result_queue,
    int worker_id,
    double virtual_loss,
    double contempt,
    double policy_softmax_temp,
    double cpuct,
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
    cpuct(cpuct)
{}

// -----------------------------------------------------------------------------
// _select: PUCT selection at every node.
//
// UCB(a) = Q(a) + cpuct * P(a) * sqrt(sum_N_available) / (1 + N(a))
// P(a)   = softmax(raw_logit) over available children
// Q(a)   = -expected_value(child) if visited, v_mix(parent) if not (FPU)
//
// Skips children that are proven (has_forced_outcome) or currently in flight
// (is_unavailable). Walks down until a leaf or a dead-end.
// -----------------------------------------------------------------------------
MCTSNode* PuctMCTS::_select(MCTSNode* start_node, std::vector<MCTSNode*>& simulation_path) {
    auto start_time = NOW();
    MCTSNode* node = start_node;
    double exp_cache[256];

    while (true) {
        if (node->num_children == 0 || !node->is_expanded() ||
            node->is_unavailable() || node->has_forced_outcome()) break;

        int num_children = node->num_children;
        double sum_visits_p = 0.0;
        double max_logit    = -1e20;
        for (int i = 0; i < num_children; ++i) {
            MCTSNode* child = node->first_child + i;
            if (child->has_forced_outcome() || child->is_unavailable()) continue;
            sum_visits_p += child->visits;
            if (child->raw_logit > max_logit) max_logit = child->raw_logit;
        }

        // Softmax priors over available children.
        double sum_exp = 0.0;
        for (int i = 0; i < num_children; ++i) {
            MCTSNode* child = node->first_child + i;
            if (child->has_forced_outcome() || child->is_unavailable()) {
                exp_cache[i] = 0.0;
                continue;
            }
            exp_cache[i] = std::exp(child->raw_logit - max_logit);
            sum_exp += exp_cache[i];
        }

        double v_mix_p     = node->calculate_v_mix(contempt);
        double sqrt_sum    = std::sqrt(std::max(0.0, sum_visits_p));
        double inv_sum_exp = 1.0 / sum_exp;

        MCTSNode* best_child_p = nullptr;
        double best_ucb = -1e20;
        for (int i = 0; i < num_children; ++i) {
            MCTSNode* child = node->first_child + i;
            if (exp_cache[i] == 0.0) continue;
            double prior = exp_cache[i] * inv_sum_exp;
            double q     = (child->visits > 0) ? -child->expected_value(contempt) : v_mix_p;
            double u     = cpuct * prior * sqrt_sum / (1.0 + child->visits);
            double ucb   = q + u;
            if (ucb > best_ucb) {
                best_ucb     = ucb;
                best_child_p = child;
            }
        }

        if (best_child_p == nullptr) break;
        root_board.makeMove(best_child_p->move);
        simulation_path.push_back(best_child_p);
        node = best_child_p;
    }
    time_selection += ELAPSED(start_time, NOW());
    return node;
}

// -----------------------------------------------------------------------------
// _run_single_async_simulation: one PUCT simulation starting from root.
//
// Unlike GumbelMCTS's version (which starts from a candidate child), PUCT
// walks from the root down through the tree. simulation_path is filled by
// _select; we unmake all its moves before returning.
// -----------------------------------------------------------------------------
bool PuctMCTS::_run_single_async_simulation() {
    std::vector<MCTSNode*> simulation_path;
    bool completed = false;

    while (true) {
        _retrieve_inference(false);
        if (batch_buffer.size() >= (size_t)worker_batch_size) {
            _spin_wait(
                [&]() { return inference_sent > inference_received; },
                [&]() { _retrieve_inference(true); }
            );
            _submit_batch();
        }

        if (root->is_unavailable() || buffer_free_slots.empty()) {
            if (!batch_buffer.empty()) _submit_batch();
            if (inference_received >= inference_sent) {
                // No-op exit: nothing simulated, nothing in flight. Caller
                // must not charge budget.
                break;
            }
            _retrieve_inference(true);
            continue;
        }

        MCTSNode* leaf = _select(root, simulation_path);

        if (root_board.isGameOver().second != chess::GameResult::NONE ||
            root_board.isRepetition(two_fold_repetition ? 1 : 2)) {
            _handle_terminal_node(leaf);
            completed = true;
            break;
        }

        // Syzygy WDL: exact resolution before we spend an NN eval.
        if (use_tablebase && _try_tablebase(leaf)) {
            completed = true;
            break;
        }

        if (leaf->is_expanded()) {
            logger.log("WARNING", "_select returned an already-expanded interior node (" +
                       chess::uci::moveToUci(leaf->move) + "); skipping re-queue.");
            while (!simulation_path.empty()) {
                root_board.unmakeMove(simulation_path.back()->move);
                simulation_path.pop_back();
            }
            if (!batch_buffer.empty()) _submit_batch();
            if (inference_received >= inference_sent) break;
            _retrieve_inference(true);
            continue;
        }

        if (root->is_unavailable()) {
            // Root itself was marked unavailable while we were selecting.
            while (!simulation_path.empty()) {
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
void PuctMCTS::_log_search_summary() {
    if (logger.get_level() > 20) return;

    if (root != nullptr && root->is_expanded()) {
        double root_v_mix = root->calculate_v_mix(contempt);

        logger.log("INFO", "");
        logger.log("INFO", "--- Root Candidate Stats ---");
        std::stringstream rss;
        rss << "Tree Stats: Root v_mix=" << std::fixed << std::setprecision(4) << root_v_mix;
        logger.log("INFO", rss.str());

        char table_header[256];
        snprintf(table_header, sizeof(table_header),
            "%-8s %8s %8s %8s %8s %8s %8s %8s %8s",
            "Move", "Logit", "Visits", "Win%", "Draw%", "Loss%", "Q(own)", "Outcome", "DTM");
        logger.log("INFO", table_header);
        logger.log("INFO", std::string(85, '-'));

        std::vector<MCTSNode*> children;
        for (int i = 0; i < root->num_children; ++i) {
            children.push_back(root->first_child + i);
        }

        std::sort(children.begin(), children.end(), [](MCTSNode* a, MCTSNode* b) {
            if (a->visits != b->visits) return a->visits > b->visits;
            return a->raw_logit > b->raw_logit;
        });

        for (MCTSNode* node : children) {
            char line[512];
            std::string outcome_str = node->has_forced_outcome() ? std::to_string(node->forced_outcome) : "None";
            std::string dtm_str = node->has_forced_outcome() ? std::to_string(node->distance_to_mate) : "None";

            double w_pct = (node->visits > 0) ? (node->l_sum / node->visits) * 100.0 : node->raw_l * 100.0;
            double d_pct = (node->visits > 0) ? (node->d_sum / node->visits) * 100.0 : node->raw_d * 100.0;
            double l_pct = (node->visits > 0) ? (node->w_sum / node->visits) * 100.0 : node->raw_w * 100.0;

            double q_val = (node->visits > 0) ? -node->expected_value(contempt) : root_v_mix;

            snprintf(line, sizeof(line),
                "%-8s %8.4f %8d %8.1f %8.1f %8.1f %8.4f %8s %8s",
                chess::uci::moveToUci(node->move).c_str(), node->raw_logit, node->visits,
                w_pct, d_pct, l_pct, q_val, outcome_str.c_str(), dtm_str.c_str());
            logger.log("INFO", line);
        }
        logger.log("INFO", std::string(85, '-'));
        logger.log("INFO", "");
    }

    logger.log("INFO", "PUCT search complete. Sims: " + std::to_string(simulation_count));
    logger.log("INFO", "--- PUCT Search (" + std::to_string(simulation_count) + " sims) Timings ---");

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

// -----------------------------------------------------------------------------
// Fixed-budget entry point. Flat loop until the budget is exhausted.
// -----------------------------------------------------------------------------
int PuctMCTS::run_simulations_fixed(int max_nodes) {
    if (logger.get_level() <= 20) {
        logger.log("INFO", "Starting PUCT MCTS. Fixed budget: " + std::to_string(max_nodes) + " nodes");
    }

    const auto wall_start = std::chrono::steady_clock::now();
    _expand_root();

    if (root->num_children == 0) {
        _flush_inflight();
        _record_nps(simulation_count,
                    std::chrono::duration<double>(std::chrono::steady_clock::now() - wall_start).count());
        return simulation_count;
    }

    int no_progress_streak = 0;
    while (simulation_count < max_nodes) {
        if (_run_single_async_simulation()) {
            no_progress_streak = 0;
        } else {
            no_progress_streak++;
            if (no_progress_streak >= 32) {
                logger.log("WARNING", "PUCT search stalled: 32 consecutive no-op sims. Ending.");
                break;
            }
        }
    }

    _flush_inflight();
    _log_search_summary();
    _record_nps(simulation_count,
                std::chrono::duration<double>(std::chrono::steady_clock::now() - wall_start).count());
    return simulation_count;
}

// -----------------------------------------------------------------------------
// Timed entry point. Checks hard_deadline every iteration. soft_deadline is
// unused for now -- reserved for the early-termination path.
// -----------------------------------------------------------------------------
int PuctMCTS::run_simulations_timed(std::chrono::steady_clock::time_point soft_deadline,
                                    std::chrono::steady_clock::time_point hard_deadline) {
    (void)soft_deadline;   // will drive early-termination when it lands

    const auto wall_start = std::chrono::steady_clock::now();

    if (logger.get_level() <= 20) {
        double budget_s = std::chrono::duration<double>(hard_deadline - wall_start).count();
        logger.log("INFO", "Starting PUCT MCTS. Time budget: " +
                   std::to_string(budget_s) + "s (nps~" +
                   std::to_string((long long)nps_ewma_) + ")");
    }

    _expand_root();

    if (root->num_children == 0) {
        _flush_inflight();
        _record_nps(simulation_count,
                    std::chrono::duration<double>(std::chrono::steady_clock::now() - wall_start).count());
        return simulation_count;
    }

    int since_check = 0;
    int no_progress_streak = 0;
    while (true) {
        // Cheap: check the deadline once every 128 sims. NN inference dwarfs
        // the wall-clock cost of a single sim, so this keeps checks off the
        // hot path without letting us overspend meaningfully.
        if (++since_check >= 128) {
            since_check = 0;
            if (std::chrono::steady_clock::now() >= hard_deadline) break;
        }

        if (_run_single_async_simulation()) {
            no_progress_streak = 0;
        } else {
            no_progress_streak++;
            if (no_progress_streak >= 32) {
                logger.log("WARNING", "PUCT search stalled: 32 consecutive no-op sims. Ending.");
                break;
            }
        }
    }

    _flush_inflight();
    _log_search_summary();
    _record_nps(simulation_count,
                std::chrono::duration<double>(std::chrono::steady_clock::now() - wall_start).count());
    return simulation_count;
}