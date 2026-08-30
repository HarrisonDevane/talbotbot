#include "mcts_engine.hpp"
#include <iostream>
#include <iomanip>
#include <cmath>
#include <algorithm>
#include <random>
#include <thread>
#include <sstream>
#include "board_utils.hpp"
#include "tbprobe.h"

#define NOW() std::chrono::high_resolution_clock::now()
#define ELAPSED(start, end) std::chrono::duration<double>(end - start).count()

// Debug helper -- find the edge in n->parent that points to n so we can print
// the move that led to n. O(branching); invoked only from DEBUG-level logging
// paths in _backpropagate.
static chess::Move _incoming_move(const MCTSNode* n) {
    if (n == nullptr || n->parent == nullptr) return chess::Move::NO_MOVE;
    for (int i = 0; i < n->parent->num_children; ++i) {
        MCTSEdge* e = n->parent->first_edge + i;
        if (e->child == n) return e->move;
    }
    return chess::Move::NO_MOVE;
}

MCTSEngine::MCTSEngine(
    const MctsConfig& cfg,
    int node_pool_capacity, int edge_pool_capacity,
    moodycamel::ConcurrentQueue<std::pair<int, int>>& inference_queue,
    ThreadSafeQueue<std::vector<int>>& result_queue,
    int worker_id,
    const chess::Board& board, const std::vector<chess::Board>& base_history, Logger& logger,
    std::vector<torch::Tensor>& shared_input_buffer, std::vector<torch::Tensor>& shared_policy_buffer, std::vector<torch::Tensor>& shared_value_buffer,
    ThreadSafeQueue<int>& buffer_free_slots, std::atomic<int>* core_wait_count, int workers_per_core, bool use_tablebase

) : worker_batch_size(cfg.batch_size_per_worker), worker_id(worker_id),
    deficit_eps(cfg.deficit_eps), policy_softmax_temp(cfg.policy_softmax_temp),
    virtual_loss(cfg.virtual_loss), contempt(cfg.contempt), draw_cutoff(cfg.draw_cutoff),
    gumbel_c_visit(cfg.gumbel_c_visit), gumbel_c_scale(cfg.gumbel_c_scale),
    gumbel_noise(cfg.gumbel_noise), root_board(board), base_history(base_history),
    node_pool(node_pool_capacity), edge_pool(edge_pool_capacity),
    scratch_node_pool(1), scratch_edge_pool(1), logger(logger),
    inference_queue(inference_queue), result_queue(result_queue),
    buffer_free_slots(buffer_free_slots), shared_input_buffer(shared_input_buffer),
    shared_policy_buffer(shared_policy_buffer), shared_value_buffer(shared_value_buffer),
    core_wait_count(core_wait_count), workers_per_core(workers_per_core), use_tablebase(use_tablebase)
{
    torch::set_num_threads(1);

    device = torch::kCUDA;
    policy_logits_dtype = torch::kFloat16;

    in_flight_nodes.resize(shared_input_buffer.size(), nullptr);
    root = node_pool.allocate();
    simulation_count = 0;
    inference_sent = 0;
    inference_received = 0;
    nps_ewma_ = 0.0;
    nps_alpha_ = 0.4;   // sensible default; call set_nps_alpha to override
    std::random_device rd;
    rng.seed(rd() ^ worker_id ^ std::chrono::high_resolution_clock::now().time_since_epoch().count());
}

// Reset with optional pool sizing. Grow-only: if targets exceed current pool
// capacity, we resize the underlying vector -- which is safe because at this
// point every MCTSNode*/MCTSEdge* the engine held has already been drained
// (in_flight -> nullptr, batch_buffer cleared) and next_idx has been reset,
// so no one is looking at a pointer that could be invalidated by the resize.
// We then re-allocate root from the grown pool.
void MCTSEngine::reset(const chess::Board& board,
                       const std::vector<chess::Board>& history,
                       size_t node_target,
                       size_t edge_target) {
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
    edge_pool.reset();

    // Grow pools BEFORE allocating root. Safe here because next_idx == 0
    // and there are no live pointers into either pool.
    if (node_target > 0) node_pool.grow_to_at_least(node_target);
    if (edge_target > 0) edge_pool.grow_to_at_least(edge_target);

    root_board = board;
    base_history = history;
    root = node_pool.allocate();
    
    simulation_count = 0;
    inference_sent = 0;
    inference_received = 0;
    batch_buffer.clear();
    max_selection_depth = 0;

    root_gumbel_noise.clear();
    pool_exhausted.store(false, std::memory_order_relaxed);

    time_selection = 0.0;
    time_expansion = 0.0;
    time_backpropagation = 0.0;
    time_retrieval = 0.0;
    time_queueing = 0.0;
    time_misc = 0.0;

    time_select_gscore  = 0.0;
    time_select_softmax = 0.0;
    time_select_other   = 0.0;
    time_backprop_stat_update = 0.0;
    time_backprop_minimax     = 0.0;
    time_backprop_other       = 0.0;
    time_retrieve_pop     = 0.0;
    time_retrieve_process = 0.0;
}

// -----------------------------------------------------------------------------
// reset_reuse: tree reuse via copy-into-scratch + swap.
//
// Preconditions:
//   - Called between searches (previous run_simulations_* has returned).
//   - Any in-flight inference from the prior search is drained here.
//   - contempt has NOT changed since the prior search (cached_q on reused
//     nodes was computed with the old contempt; changing contempt silently
//     staled all of them).
//
// Failure modes (return false, engine state unchanged):
//   - root has no children yet (never expanded).
//   - played_move not among root's children (opponent surprised us; the
//     Gumbel top-m sampling didn't consider it).
//   - Matching edge's child is null or unexpanded (no useful info stored).
//   - Copy raised (pool grow failed etc.) -- caught, logged, returned false.
//
// Caller pattern in main_uci / self_play:
//   if (!engine.reset_reuse(new_board, new_history, played_move, node_t, edge_t)) {
//       engine.reset(new_board, new_history, node_t, edge_t);
//   }
// -----------------------------------------------------------------------------
bool MCTSEngine::reset_reuse(const chess::Board& new_board,
                             const std::vector<chess::Board>& new_history,
                             chess::Move played_move,
                             size_t node_target,
                             size_t edge_target) {
    // Bail early if root has no tree to reuse.
    if (root == nullptr || root->num_children == 0 || root->first_edge == nullptr) {
        return false;
    }

    // Find the edge for played_move.
    MCTSEdge* matched_edge = nullptr;
    for (uint16_t i = 0; i < root->num_children; ++i) {
        MCTSEdge* e = root->first_edge + i;
        if (e->move == played_move) {
            matched_edge = e;
            break;
        }
    }
    if (matched_edge == nullptr) return false;

    MCTSNode* old_subtree_root = matched_edge->child;
    if (old_subtree_root == nullptr) return false;

    if (!old_subtree_root->is_expanded()
        || old_subtree_root->has_forced_outcome()
        || old_subtree_root->is_unavailable()
        || old_subtree_root->num_children == 0) {
        if (logger.get_level() <= 20) {
            logger.log("INFO",
                "reset_reuse: played move leads to non-reusable node "
                "(expanded=" + std::to_string(old_subtree_root->is_expanded())
                + " fo=" + (old_subtree_root->has_forced_outcome()
                            ? std::to_string(old_subtree_root->forced_outcome)
                            : "none")
                + " unavailable=" + std::to_string(old_subtree_root->is_unavailable())
                + " num_children=" + std::to_string(old_subtree_root->num_children)
                + "); falling back to full reset.");
        }
        return false;
    }

    // ---- Drain in-flight inference (same as reset). MUST happen before we
    //      touch pools -- otherwise a late completion would write into memory
    //      we're about to swap. ----
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

    // ---- Prep scratch pools. Grow to at least the current active-pool used
    //      count (guaranteed >= subtree size) or to the caller-supplied
    //      targets, whichever is larger. Reset ensures next_idx = 0. ----
    scratch_node_pool.reset();
    scratch_edge_pool.reset();
    const size_t node_min = std::max(node_target, node_pool.used());
    const size_t edge_min = std::max(edge_target, edge_pool.used());
    if (node_min > 0) scratch_node_pool.grow_to_at_least(node_min);
    if (edge_min > 0) scratch_edge_pool.grow_to_at_least(edge_min);

    // ---- Copy the subtree into scratch. ----
    MCTSNode* new_root = nullptr;
    try {
        new_root = _copy_subtree(old_subtree_root, /*new_parent=*/nullptr,
                                 scratch_node_pool, scratch_edge_pool);
    } catch (const std::exception& e) {
        logger.log("WARNING", std::string("reset_reuse: copy failed (") +
                              e.what() + "); falling back to full reset.");
        // Leave engine state as-is; caller will invoke reset() on false.
        return false;
    }

    // ---- Swap active <-> scratch. After swap:
    //        node_pool         holds the copied live subtree
    //        scratch_node_pool holds the (now dead) old tree
    //      Same for edges. MCTSNode* pointers returned by _copy_subtree
    //      remain valid because std::swap on std::vector just swaps the
    //      underlying data() pointers -- the actual heap allocations don't
    //      move. ----
    std::swap(node_pool, scratch_node_pool);
    std::swap(edge_pool, scratch_edge_pool);

    // ---- Update root / board / history. ----
    root = new_root;
    root_board = new_board;
    base_history = new_history;

    // ---- Per-search counters reset (tree stats intentionally preserved). ----
    simulation_count = 0;
    inference_sent = 0;
    inference_received = 0;
    batch_buffer.clear();
    max_selection_depth = 0;
    root_gumbel_noise.clear();
    pool_exhausted.store(false, std::memory_order_relaxed);

    time_selection = 0.0;
    time_expansion = 0.0;
    time_backpropagation = 0.0;
    time_retrieval = 0.0;
    time_queueing = 0.0;
    time_misc = 0.0;
    time_wait_for_inference = 0.0;
    time_select_gscore  = 0.0;
    time_select_softmax = 0.0;
    time_select_other   = 0.0;
    time_backprop_stat_update = 0.0;
    time_backprop_minimax     = 0.0;
    time_backprop_other       = 0.0;
    time_retrieve_pop     = 0.0;
    time_retrieve_process = 0.0;

    if (logger.get_level() <= 20) {
        logger.log("INFO", "reset_reuse: subtree " +
                   std::to_string(node_pool.used()) + " nodes, " +
                   std::to_string(edge_pool.used()) + " edges (root visits=" +
                   std::to_string(root->visits) + ", played=" +
                   chess::uci::moveToUci(played_move) + ")");
    }

    return true;
}

// -----------------------------------------------------------------------------
// _copy_subtree: recursive DFS copy from (old_node, in some source pool) into
// (np, ep). Each node/edge is a memberwise copy; parent + first_edge pointers
// are patched to point inside (np, ep).
//
// Recursion depth is bounded by search tree depth (typically < 100 for chess
// searches; well within default 1MB thread stack).
//
// Throws std::runtime_error if either pool exceeds capacity mid-copy. Caller
// catches and treats as reuse failure.
// -----------------------------------------------------------------------------
MCTSNode* MCTSEngine::_copy_subtree(MCTSNode* old_node, MCTSNode* new_parent,
                                    NodePool& np, EdgePool& ep) {
    // allocate() default-constructs at (new_parent); we then overwrite with
    // memberwise copy from old_node and re-patch parent / first_edge below.
    // The default construction is wasteful but not hot-path.
    MCTSNode* new_node = np.allocate(new_parent);
    *new_node = *old_node;
    new_node->parent     = new_parent;
    new_node->first_edge = nullptr;   // patched below if children exist

    if (old_node->num_children > 0 && old_node->first_edge != nullptr) {
        MCTSEdge* new_edges = ep.allocate_block(old_node->num_children);
        new_node->first_edge = new_edges;

        for (uint16_t i = 0; i < old_node->num_children; ++i) {
            MCTSEdge* old_edge = old_node->first_edge + i;
            MCTSEdge* new_edge = new_edges + i;
            *new_edge = *old_edge;   // copies raw_logit, policy_flat_index, move, etc.

            if (old_edge->child != nullptr) {
                new_edge->child = _copy_subtree(old_edge->child, new_node, np, ep);
            } else {
                new_edge->child = nullptr;   // unmaterialised child stays unmaterialised
            }
        }
    }

    return new_node;
}

// -----------------------------------------------------------------------------
// Pool sizing helpers.
// -----------------------------------------------------------------------------
PoolTargets MCTSEngine::predict_pool_needs_static(int predicted_sims,
                                                  const PoolSizingConfig& cfg) {
    int sims = std::max(predicted_sims, 1);
    size_t node_target = static_cast<size_t>(
        static_cast<double>(sims) * cfg.node_safety_factor);
    size_t edge_target = static_cast<size_t>(
        static_cast<double>(sims) * cfg.avg_branching * cfg.edge_safety_factor);

    // Convert byte caps to element counts and clamp.
    size_t node_cap = cfg.node_hard_cap_bytes / sizeof(MCTSNode);
    size_t edge_cap = cfg.edge_hard_cap_bytes / sizeof(MCTSEdge);
    if (node_cap == 0) node_cap = 1;   // guard against absurd cap
    if (edge_cap == 0) edge_cap = 1;

    if (node_target > node_cap) node_target = node_cap;
    if (edge_target > edge_cap) edge_target = edge_cap;

    // Floor at 1 so the pool always has at least room for root.
    if (node_target < 1) node_target = 1;
    if (edge_target < 1) edge_target = 1;

    return { node_target, edge_target };
}

PoolTargets MCTSEngine::predict_pool_needs_for_time(double time_s,
                                                    double safety_multiplier) const {
    double time_clamped = std::max(0.0, time_s);
    double sims_d = estimated_nps() * time_clamped * safety_multiplier;
    if (sims_d < 1.0) sims_d = 1.0;
    // Cap to int range defensively.
    int sims = (sims_d > static_cast<double>(std::numeric_limits<int>::max()))
             ? std::numeric_limits<int>::max()
             : static_cast<int>(sims_d);
    return predict_pool_needs(sims);
}


MCTSNode* MCTSEngine::_select(MCTSNode* start_node, std::vector<MCTSEdge*>& simulation_path) {
    const bool profile = logger.get_level() <= 20;
    auto start_time = NOW();
    MCTSNode* node = start_node;

    // Stack caches -- outer scope so they're allocated once per _select call,
    // not once per descent step. 256 slots covers chess (max legal moves ~218).
    // Total ~14 KB stack; fits comfortably in L1.
    MCTSEdge* edge_cache[256];
    MCTSNode* child_cache[256];
    int       visits_cache[256];
    float     raw_logit_cache[256];
    bool      active_cache[256];
    double    score_cache[256];
    double    exp_cache[256];
    double    prior_cache[256];

    while (true) {
        auto other_start = profile ? NOW() : start_time;
        if (node->num_children == 0 || !node->is_expanded() || node->is_unavailable() || node->has_forced_outcome()) {
            if (profile) time_select_other += ELAPSED(other_start, NOW());
            break;
        }

        const int  num_edges = node->num_children;
        const bool use_prior = (deficit_eps > 0.0);

        double max_visits = 0.0;
        double sum_visits = 0.0;

        // ===== H: snapshot pass (only pass that touches child heap) =====
        for (int i = 0; i < num_edges; ++i) {
            MCTSEdge* edge = node->first_edge + i;
            MCTSNode* child = edge->child;
            edge_cache[i]      = edge;
            child_cache[i]     = child;
            raw_logit_cache[i] = edge->raw_logit;

            if (child == nullptr) {
                // Unmaterialised: selectable, treated as 0 visits.
                active_cache[i] = true;
                visits_cache[i] = 0;
                continue;
            }
            if (child->has_forced_outcome() || child->is_unavailable()) {
                active_cache[i] = false;
                visits_cache[i] = 0;
                continue;
            }
            active_cache[i] = true;
            const int v = child->visits;
            visits_cache[i] = v;
            if (v > max_visits) max_visits = v;
            sum_visits += v;
        }

        const double v_mix       = node->calculate_v_mix(contempt);
        const double sigma_scale = (gumbel_c_visit + max_visits) * gumbel_c_scale;
        if (profile) time_select_other += ELAPSED(other_start, NOW());

        // ===== S1: gscore per active edge (stack-only, cached_q direct read) =====
        auto gscore_start = profile ? NOW() : start_time;
        double max_score_logit = -1e20;
        double max_raw_logit   = -1e20;
        for (int i = 0; i < num_edges; ++i) {
            if (!active_cache[i]) continue;
            // q from child: cached_q was written from child's perspective in
            // backprop, so negate for the parent-selection perspective. Fresh
            // 0-visit nodes use v_mix, same as pre-refactor.
            double q = (visits_cache[i] > 0)
                     ? -static_cast<double>(child_cache[i]->cached_q)
                     : v_mix;
            // Draw cap: if the child has mover_has_draw set, that mover can
            // guarantee >= 0 for themselves, so parent's value for this move
            // is <= 0. Cap parent-perspective Q from above at 0.0. Prevents
            // noisy positive NN estimates from beating a certain drawing line.
            if (child_cache[i] != nullptr && child_cache[i]->mover_has_draw()) {
                q = 0.0;
            }
            const double q_norm = (q + 1.0) / 2.0;
            const double score = raw_logit_cache[i] + sigma_scale * q_norm;
            score_cache[i] = score;
            if (score > max_score_logit) max_score_logit = score;
            if (use_prior && raw_logit_cache[i] > max_raw_logit) {
                max_raw_logit = raw_logit_cache[i];
            }
        }
        if (profile) time_select_gscore += ELAPSED(gscore_start, NOW());

        // ===== S2: softmax exp (skip prior branch entirely when eps == 0) =====
        auto softmax_start = profile ? NOW() : start_time;
        double sum_score_exp = 0.0;
        double sum_prior_exp = 0.0;
        if (use_prior) {
            for (int i = 0; i < num_edges; ++i) {
                if (!active_cache[i]) {
                    exp_cache[i]   = 0.0;
                    prior_cache[i] = 0.0;
                    continue;
                }
                const double e = std::exp(score_cache[i] - max_score_logit);
                exp_cache[i]    = e;
                sum_score_exp  += e;
                const double p = std::exp(raw_logit_cache[i] - max_raw_logit);
                prior_cache[i]  = p;
                sum_prior_exp  += p;
            }
        } else {
            // eps == 0: prior softmax is multiplied by 0 downstream; skip it.
            for (int i = 0; i < num_edges; ++i) {
                if (!active_cache[i]) {
                    exp_cache[i] = 0.0;
                    continue;
                }
                const double e = std::exp(score_cache[i] - max_score_logit);
                exp_cache[i]   = e;
                sum_score_exp += e;
            }
        }
        if (profile) time_select_softmax += ELAPSED(softmax_start, NOW());

        // ===== S3: deficit argmax =====
        auto other2_start = profile ? NOW() : start_time;
        const double inv_sum_visits    = 1.0 / (1.0 + sum_visits);
        const double inv_sum_score_exp = (sum_score_exp > 0.0) ? (1.0 / sum_score_exp) : 0.0;

        double best_deficit = -1e20;
        int    best_i       = -1;

        if (use_prior) {
            const double inv_sum_prior_exp = (sum_prior_exp > 0.0) ? (1.0 / sum_prior_exp) : 0.0;
            const double one_minus_eps     = 1.0 - deficit_eps;
            for (int i = 0; i < num_edges; ++i) {
                if (!active_cache[i]) continue;
                const double pi_prime = one_minus_eps * (exp_cache[i]   * inv_sum_score_exp)
                                      +  deficit_eps  * (prior_cache[i] * inv_sum_prior_exp);
                const double child_n_norm = static_cast<double>(visits_cache[i]) * inv_sum_visits;
                const double deficit      = pi_prime - child_n_norm;
                if (deficit > best_deficit) {
                    best_deficit = deficit;
                    best_i       = i;
                }
            }
        } else {
            for (int i = 0; i < num_edges; ++i) {
                if (!active_cache[i]) continue;
                const double pi_prime     = exp_cache[i] * inv_sum_score_exp;
                const double child_n_norm = static_cast<double>(visits_cache[i]) * inv_sum_visits;
                const double deficit      = pi_prime - child_n_norm;
                if (deficit > best_deficit) {
                    best_deficit = deficit;
                    best_i       = i;
                }
            }
        }

        if (best_i < 0) {
            if (profile) time_select_other += ELAPSED(other2_start, NOW());
            break;
        }

        MCTSEdge* best_edge = edge_cache[best_i];
        MCTSNode* next_node = child_cache[best_i];
        if (next_node == nullptr) {
            if (!node_pool.has_capacity(1)) {
                if (!pool_exhausted.exchange(true, std::memory_order_relaxed)) {
                    logger.log("WARNING", "Node pool exhausted during _select (remaining=" +
                               std::to_string(node_pool.remaining()) + " of " +
                               std::to_string(node_pool.capacity()) + ").");
                }
                if (profile) time_select_other += ELAPSED(other2_start, NOW());
                break;
            }
            next_node = node_pool.allocate(node);
            best_edge->child = next_node;
        }

        root_board.makeMove(best_edge->move);
        simulation_path.push_back(best_edge);
        node = next_node;
        if (profile) time_select_other += ELAPSED(other2_start, NOW());
    }
    time_selection += ELAPSED(start_time, NOW());
    int depth = static_cast<int>(simulation_path.size()) +
                (start_node == root ? 0 : 1);
    if (depth > max_selection_depth) max_selection_depth = depth;
    return node;
}

void MCTSEngine::_propagate_unavailability_upward(MCTSNode* node) {
    node->set_unavailable(true);
    MCTSNode* current = node;
    MCTSNode* parent = current->parent;

    while (parent != nullptr) {
        parent->num_available_children -= 1;
        if (parent->num_available_children > 0) break;
        parent->set_unavailable(true);
        current = parent;
        parent = current->parent;
    }
}

void MCTSEngine::_mark_selected(MCTSNode* node) {
    _propagate_unavailability_upward(node);
}

void MCTSEngine::_unmark_selected(MCTSNode* node) {
    MCTSNode* current_node = node;
    current_node->set_unavailable(false);
    MCTSNode* parent = current_node->parent;

    while (parent != nullptr) {
        parent->num_available_children += 1;
        if (parent->num_available_children == 1) {
            parent->set_unavailable(false);
            current_node = parent;
            parent = current_node->parent;
        } else break;
    }
}

void MCTSEngine::_dispatch_selected_leaf(MCTSNode* leaf,
                                        std::vector<MCTSEdge*>& simulation_path) {
    if (root_board.isGameOver().second != chess::GameResult::NONE
        || root_board.isRepetition(1)) {
        _handle_terminal_node(leaf);
    }
    else if (use_tablebase && _try_tablebase(leaf)) {
        // TB backprop inside.
    }
    else if (leaf->is_expanded()) {
        // Diagnostic dump only under DEBUG; string-building is otherwise skipped.
        if (logger.get_level() <= 10) {
            std::string path_str;
            for (MCTSEdge* e : simulation_path) {
                if (!path_str.empty()) path_str += " ";
                path_str += chess::uci::moveToUci(e->move);
            }
            MCTSNode* only_child = (leaf->num_children == 1) ? (leaf->first_edge + 0)->child : nullptr;
            std::string child_state;
            if (only_child == nullptr) child_state = "nullptr";
            else {
                child_state = "unavailable=" + std::to_string(only_child->is_unavailable())
                            + " fo=" + (only_child->has_forced_outcome() ? std::to_string(only_child->forced_outcome) : "none")
                            + " expanded=" + std::to_string(only_child->is_expanded())
                            + " num_children=" + std::to_string(only_child->num_children);
            }
            logger.log("DEBUG", "_select returned already-expanded interior node; skipping. path=[" + path_str
                + "] leaf_children=" + std::to_string(leaf->num_children)
                + " leaf_unavailable=" + std::to_string(leaf->is_unavailable())
                + " leaf_num_available=" + std::to_string(leaf->num_available_children)
                + " leaf_fo=" + (leaf->has_forced_outcome() ? std::to_string(leaf->forced_outcome) : "none")
                + " child=[" + child_state + "]");
        }
    }
    else {
        _queue_leaf_for_inference(leaf, simulation_path);
    }

    while (!simulation_path.empty()) {
        root_board.unmakeMove(simulation_path.back()->move);
        simulation_path.pop_back();
    }
}

// End-of-search drain. Flush any partially-filled batch and block until every
// outstanding inference has returned. Called at the tail of both inference-only
// sim loops before final logging.
void MCTSEngine::_drain_pending_inference() {
    if (!batch_buffer.empty()) _submit_batch();
    while (inference_received < inference_sent) {
        _retrieve_inference(true);
    }
}

template <typename Predicate, typename WorkFn>
void MCTSEngine::_spin_wait(Predicate should_keep_waiting, WorkFn work_fn) {
    if (workers_per_core <= 1) {
        while (should_keep_waiting()) {
            work_fn();
        }
        return;
    }

    core_wait_count->fetch_add(1, std::memory_order_acquire);
    while (should_keep_waiting()) {
        work_fn();
        if (core_wait_count->load(std::memory_order_relaxed) == workers_per_core) {
            _mm_pause();
        } else {
            std::this_thread::yield();
        }
    }
    core_wait_count->fetch_sub(1, std::memory_order_release);
}

// Sub-buckets time_retrieve_pop and time_retrieve_process cover only the
// retrieve-specific work; expansion (time_expansion) and backprop
// (time_backpropagation) called from here have their own outer counters
// and are deliberately not counted here to avoid double-attribution.
// time_retrieval ~= pop + process + (expansion within retrieve) + (backprop within retrieve).
void MCTSEngine::_retrieve_inference(bool block) {
    const bool profile = logger.get_level() <= 20;
    auto start_time = NOW();
    std::vector<int> completed_indices;

    while (true) {
        auto pop_start = profile ? NOW() : start_time;
        if (block) {
            completed_indices = result_queue.pop_wait();
            block = false;
        } else {
            if (!result_queue.try_pop(completed_indices)) {
                if (profile) time_retrieve_pop += ELAPSED(pop_start, NOW());
                break;
            }
        }
        if (profile) time_retrieve_pop += ELAPSED(pop_start, NOW());

        if (logger.get_level() <= 10) {
            logger.log("DEBUG", "Received " + std::to_string(completed_indices.size()) + " inferences from batcher.");
        }

        for (int buffer_index : completed_indices) {
            auto process_start = profile ? NOW() : start_time;

            MCTSNode* node = in_flight_nodes[buffer_index];
            in_flight_nodes[buffer_index] = nullptr;
            inference_received++;

            c10::Half* policy_ptr = shared_policy_buffer[buffer_index].data_ptr<c10::Half>();
            c10::Half* wdl_ptr = shared_value_buffer[buffer_index].data_ptr<c10::Half>();
            float p_win = (float)wdl_ptr[0];
            float p_draw = (float)wdl_ptr[1];
            float p_loss = (float)wdl_ptr[2];

            buffer_free_slots.push(buffer_index);
            if (profile) time_retrieve_process += ELAPSED(process_start, NOW());

            if (node != nullptr) {
                if (!node->is_expanded()) {
                    auto exp_start = NOW();
                    for (int i = 0; i < node->num_children; ++i) {
                        MCTSEdge* edge = node->first_edge + i;
                        edge->raw_logit = policy_ptr[edge->policy_flat_index] / policy_softmax_temp;
                    }
                    node->set_expanded(true);
                    time_expansion += ELAPSED(exp_start, NOW());
                }
                _backpropagate(node, p_win, p_draw, p_loss, false);
            }
        }
    }
    time_retrieval += ELAPSED(start_time, NOW());
}

void MCTSEngine::_submit_batch() {
    auto start_time = NOW();
    int b_size = batch_buffer.size();
    if (b_size == 0) return;

    if (logger.get_level() <= 10) {
        logger.log("DEBUG", "Submitting batch of " + std::to_string(b_size) + " states to inference queue.");
    }
    
    inference_queue.enqueue_bulk(batch_buffer.data(), b_size);
    
    inference_sent += b_size;
    batch_buffer.clear();

    time_queueing += ELAPSED(start_time, NOW());
}

void MCTSEngine::_handle_terminal_node(MCTSNode* leaf) {
    auto start_time = NOW();
    auto result = root_board.isGameOver(); 
    
    double w = 0.0, d = 0.0, l = 0.0;
    std::string term_type = "Draw";

    if (result.second == chess::GameResult::LOSE) {
        l = 1.0; 
        term_type = "Loss (Mate)";
    } else if (result.second == chess::GameResult::DRAW || root_board.isRepetition(1)) {
        d = 1.0;
    }

    if (logger.get_level() <= 10) {
        logger.log("DEBUG", "Terminal node reached during search. Result: " + term_type);
    }

    _mark_selected(leaf);
    time_expansion += ELAPSED(start_time, NOW());

    // Draw leaves seed mover_has_draw = true directly. Non-draw terminals
    // (wins/losses) leave it default-false; minimax will clear it explicitly
    // anyway when it sees the forced outcome.
    if (d == 1.0) {
        leaf->set_mover_has_draw(true);
    }

    _backpropagate(leaf, w, d, l, true);
    simulation_count++;
}

bool MCTSEngine::_try_tablebase(MCTSNode* leaf) {
    using chess::PieceType;
    using chess::Color;

    const chess::Bitboard wp = root_board.pieces(PieceType::PAWN,   Color::WHITE);
    const chess::Bitboard wn = root_board.pieces(PieceType::KNIGHT, Color::WHITE);
    const chess::Bitboard wb = root_board.pieces(PieceType::BISHOP, Color::WHITE);
    const chess::Bitboard wr = root_board.pieces(PieceType::ROOK,   Color::WHITE);
    const chess::Bitboard wq = root_board.pieces(PieceType::QUEEN,  Color::WHITE);
    const chess::Bitboard wk = root_board.pieces(PieceType::KING,   Color::WHITE);

    const chess::Bitboard bp = root_board.pieces(PieceType::PAWN,   Color::BLACK);
    const chess::Bitboard bn = root_board.pieces(PieceType::KNIGHT, Color::BLACK);
    const chess::Bitboard bb = root_board.pieces(PieceType::BISHOP, Color::BLACK);
    const chess::Bitboard br = root_board.pieces(PieceType::ROOK,   Color::BLACK);
    const chess::Bitboard bq = root_board.pieces(PieceType::QUEEN,  Color::BLACK);
    const chess::Bitboard bk = root_board.pieces(PieceType::KING,   Color::BLACK);

    const chess::Bitboard white_bb = wp | wn | wb | wr | wq | wk;
    const chess::Bitboard black_bb = bp | bn | bb | br | bq | bk;

    if ((white_bb | black_bb).count() > (int)TB_LARGEST) return false;

    const auto& cr = root_board.castlingRights();
    const bool any_castle =
        cr.has(Color::WHITE, chess::Board::CastlingRights::Side::KING_SIDE)  ||
        cr.has(Color::WHITE, chess::Board::CastlingRights::Side::QUEEN_SIDE) ||
        cr.has(Color::BLACK, chess::Board::CastlingRights::Side::KING_SIDE)  ||
        cr.has(Color::BLACK, chess::Board::CastlingRights::Side::QUEEN_SIDE);
    if (any_castle) return false;

    const chess::Square ep_sq = root_board.enpassantSq();
    const unsigned ep = (ep_sq == chess::Square::NO_SQ) ? 0u
                                                        : (unsigned)ep_sq.index();
    const unsigned rule50        = (unsigned)root_board.halfMoveClock();
    const bool     white_to_move = (root_board.sideToMove() == Color::WHITE);

    const unsigned wdl = tb_probe_wdl(
        white_bb.getBits(), black_bb.getBits(),
        (wk | bk).getBits(), (wq | bq).getBits(), (wr | br).getBits(),
        (wb | bb).getBits(), (wn | bn).getBits(), (wp | bp).getBits(),
        rule50, /*castling=*/0u, ep, white_to_move);

    if (wdl == TB_RESULT_FAILED) return false;

    double w = 0.0, d = 0.0, l = 0.0;
    switch (wdl) {
        case TB_WIN:          w = 1.0; break;
        case TB_LOSS:         l = 1.0; break;
        case TB_CURSED_WIN:
        case TB_BLESSED_LOSS:
        case TB_DRAW:         d = 1.0; break;
        default:              return false;
    }

    _mark_selected(leaf);
    if (d == 1.0) {
        leaf->set_mover_has_draw(true);
    }
    _backpropagate(leaf, w, d, l, true);
    simulation_count++;
    return true;
}

void MCTSEngine::_queue_leaf_for_inference(MCTSNode* leaf, const std::vector<MCTSEdge*>& simulation_path) {
    auto start_time = NOW();

    chess::Movelist moves;
    chess::movegen::legalmoves(moves, root_board);

    if (!edge_pool.has_capacity(moves.size())) {
        if (!pool_exhausted.exchange(true, std::memory_order_relaxed)) {
            logger.log("WARNING", "Edge pool exhausted (remaining=" +
                       std::to_string(edge_pool.remaining()) + " of " +
                       std::to_string(edge_pool.capacity()) +
                       "). Truncating search early.");
        }
        return;
    }

    int buffer_index;

    _spin_wait(
        [&]() { return !buffer_free_slots.try_pop(buffer_index); },
        [&]() { _retrieve_inference(false); if (!batch_buffer.empty()) _submit_batch(); }
    );

    in_flight_nodes[buffer_index] = leaf;
    _mark_selected(leaf);

    auto exp_start = NOW();
    leaf->num_children = moves.size();
    leaf->num_available_children = leaf->num_children;

    if (leaf->num_children > 0) {
        leaf->first_edge = edge_pool.allocate_block(leaf->num_children);
        for (int i = 0; i < leaf->num_children; ++i) {
            MCTSEdge* edge = leaf->first_edge + i;
            edge->move = moves[i];
            PolicyComponent pc = move_to_policy_components(moves[i], root_board);
            edge->policy_flat_index = policy_components_to_flat_index(pc.row, pc.col, pc.channel);
        }
    }
    time_expansion += ELAPSED(exp_start, NOW());

    std::vector<chess::Board> combined_history;
    std::vector<chess::Move> unmade_moves;

    for (int i = (int)simulation_path.size() - 1; i >= 0 && combined_history.size() < 7; --i) {
        root_board.unmakeMove(simulation_path[i]->move);
        unmade_moves.push_back(simulation_path[i]->move);
        combined_history.push_back(root_board);
    }

    for (size_t i = 0; i < base_history.size() && combined_history.size() < 7; ++i) {
        combined_history.push_back(base_history[i]);
    }

    for (int i = (int)unmade_moves.size() - 1; i >= 0; --i) {
        root_board.makeMove(unmade_moves[i]);
    }

    c10::Half* destination_ptr = shared_input_buffer[buffer_index].data_ptr<c10::Half>();
    board_to_tensor(root_board, combined_history, destination_ptr);

    batch_buffer.push_back({worker_id, buffer_index});
    _virtual_loss(leaf, true);

    if (batch_buffer.size() >= (size_t)worker_batch_size) { 
        _submit_batch();
        _spin_wait(
            [&]() { return inference_sent > inference_received; },
            [&]() { _retrieve_inference(true); }
        );
    }

    time_misc += ELAPSED(start_time, NOW());
    simulation_count++;
}


bool MCTSEngine::_run_single_async_simulation(MCTSEdge* start_edge) {
    std::vector<MCTSEdge*> simulation_path;
    root_board.makeMove(start_edge->move);
    simulation_path.push_back(start_edge);

    if (start_edge->child == nullptr) {
        if (!node_pool.has_capacity(1)) {
            if (!pool_exhausted.exchange(true, std::memory_order_relaxed)) {
                logger.log("WARNING", "Node pool exhausted materialising candidate start edge.");
            }
            root_board.unmakeMove(start_edge->move);
            return false;
        }
        start_edge->child = node_pool.allocate(root);
    }
    MCTSNode* start_node = start_edge->child;

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

        if (pool_exhausted.load(std::memory_order_relaxed)) {
            completed = false;
            break;
        }

        if (logger.get_level() <= 10) {
            std::string path_str = "";
            std::string root_move_str = "";
            for (MCTSEdge* e : simulation_path) {
                std::string uci = chess::uci::moveToUci(e->move);
                if (path_str.empty()) root_move_str = uci;
                if (!path_str.empty()) path_str += " ";
                path_str += uci;
            }
            if (root_move_str == "e3h6") {
                logger.log("DEBUG", "Selected path: " + path_str);
            }
        }

        if (root_board.isGameOver().second != chess::GameResult::NONE || root_board.isRepetition(1)) {
            _handle_terminal_node(leaf);
            completed = true;
            break;
        }

        if (use_tablebase && _try_tablebase(leaf)) {
            completed = true;
            break;
        }

        if (leaf->is_expanded()) {
            logger.log("DEBUG", "_select returned an already-expanded interior node; skipping re-queue.");
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
        if (pool_exhausted.load(std::memory_order_relaxed)) {
            completed = false;
            break;
        }
        completed = true;
        break;
    }

    while (!simulation_path.empty()) {
        root_board.unmakeMove(simulation_path.back()->move);
        simulation_path.pop_back();
    }
    return completed;
}

void MCTSEngine::_log_tournament_results(const std::vector<MCTSEdge*>& candidates,
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
        for (MCTSEdge* e : candidates) {
            MCTSNode* c = e->child;
            if (c != nullptr && c->has_forced_outcome()) forced++; else active++;
            if (c != nullptr) total_visits += c->visits;
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
        "%-8s %8s %8s %8s %8s %8s %8s %8s %8s %8s %4s",
        "Move", "Logit", "Visits", "Win%", "Draw%", "Loss%", "Norm Q", "Score", "Outcome", "DTM", "MHD");
    logger.log("INFO", table_header);
    logger.log("INFO", std::string(100, '-'));

    double log_max_visits = 0.0;
    for (MCTSEdge* e : candidates) {
        MCTSNode* c = e->child;
        if (c != nullptr && c->visits > log_max_visits) log_max_visits = c->visits;
    }
    auto score_of = [&](MCTSEdge* e) -> double {
        int idx = static_cast<int>(e - root->first_edge);
        double noise = (idx >= 0 && idx < (int)root_gumbel_noise.size())
                     ? root_gumbel_noise[idx] : 0.0;
        return e->calculate_gumbel_score(contempt, gumbel_c_visit, gumbel_c_scale,
                                         log_max_visits, root_v_mix, noise);
    };

    std::vector<MCTSEdge*> sorted_cands = candidates;
    std::sort(sorted_cands.begin(), sorted_cands.end(), [&](MCTSEdge* a, MCTSEdge* b) {
        int va = (a->child != nullptr) ? a->child->visits : 0;
        int vb = (b->child != nullptr) ? b->child->visits : 0;
        if (va != vb) return va > vb;
        return score_of(a) > score_of(b);
    });

    for (MCTSEdge* e : sorted_cands) {
        MCTSNode* c = e->child;
        char line[512];

        std::string outcome_str, dtm_str;
        int c_visits = 0;
        double w_pct = 0.0, d_pct = 0.0, l_pct = 0.0;
        double q_val = root_v_mix;
        int mhd = 0;
        if (c != nullptr) {
            outcome_str = c->has_forced_outcome() ? std::to_string(c->forced_outcome) : "None";
            dtm_str     = c->has_forced_outcome() ? std::to_string(c->distance_to_mate) : "None";
            c_visits = c->visits;
            if (c->visits > 0) {
                w_pct = (c->l_sum / c->visits) * 100.0;
                d_pct = (c->d_sum / c->visits) * 100.0;
                l_pct = (c->w_sum / c->visits) * 100.0;
                q_val = -c->expected_value(contempt);
            } else {
                w_pct = c->raw_l() * 100.0;
                d_pct = c->raw_d   * 100.0;
                l_pct = c->raw_w   * 100.0;
            }
            mhd = c->mover_has_draw() ? 1 : 0;
        } else {
            outcome_str = "None";
            dtm_str     = "None";
        }
        double q_norm = (q_val + 1.0) / 2.0;

        snprintf(line, sizeof(line),
            "%-8s %8.4f %8d %8.1f %8.1f %8.1f %8.4f %8.4f %8s %8s %4d",
            chess::uci::moveToUci(e->move).c_str(), e->raw_logit, c_visits,
            w_pct, d_pct, l_pct, q_norm, score_of(e), outcome_str.c_str(), dtm_str.c_str(),
            mhd);
        logger.log("INFO", line);
    }

    logger.log("INFO", std::string(100, '-'));
    logger.log("INFO", "");
}

void MCTSEngine::_log_node_by_path(const std::vector<std::string>& uci_path, int top_n) {
    if (logger.get_level() > 20) return;

    MCTSNode* node = root;
    MCTSEdge* incoming = nullptr;
    std::string walked;
    for (const std::string& uci : uci_path) {
        if (node == nullptr) break;
        MCTSEdge* found_edge = nullptr;
        for (int i = 0; i < node->num_children; ++i) {
            MCTSEdge* e = node->first_edge + i;
            if (chess::uci::moveToUci(e->move) == uci) { found_edge = e; break; }
        }
        if (found_edge == nullptr || found_edge->child == nullptr) {
            logger.log("INFO", "[path dump] move '" + uci + "' not found or unvisited below '" + walked + "'");
            return;
        }
        walked += (walked.empty() ? "" : " ") + uci;
        incoming = found_edge;
        node = found_edge->child;
    }
    if (node == nullptr) return;

    logger.log("INFO", "");
    logger.log("INFO", "=== Node-by-path dump: [" + walked + "] ===");

    char head[512];
    double tgt_q_own = (node->visits > 0) ? node->expected_value(contempt)
                                          : ((node->raw_w - node->raw_l()) + contempt * node->raw_d);
    double tgt_vmix  = node->is_expanded() ? node->calculate_v_mix(contempt) : 0.0;
    snprintf(head, sizeof(head),
        "TARGET raw network WDL (own mover persp): W=%.4f D=%.4f L=%.4f  -> raw_value(own)=%+.4f",
        node->raw_w, node->raw_d, node->raw_l(), (node->raw_w - node->raw_l()));
    logger.log("INFO", head);
    float incoming_logit = (incoming != nullptr) ? incoming->raw_logit : 0.0f;
    snprintf(head, sizeof(head),
        "TARGET visits=%d  expected_value(own)=%+.4f  v_mix(own)=%+.4f  incoming_logit=%.3f  expanded=%d  outcome=%s",
        node->visits, tgt_q_own, tgt_vmix, incoming_logit, node->is_expanded() ? 1 : 0,
        node->has_forced_outcome() ? std::to_string(node->forced_outcome).c_str() : "None");
    logger.log("INFO", head);

    if (!node->is_expanded() || node->num_children == 0) {
        logger.log("INFO", "  (target is a leaf / unexpanded -- no children)");
        logger.log("INFO", "=== end node-by-path dump ===");
        logger.log("INFO", "");
        return;
    }

    std::vector<MCTSEdge*> kids;
    for (int i = 0; i < node->num_children; ++i) kids.push_back(node->first_edge + i);
    std::sort(kids.begin(), kids.end(), [&](MCTSEdge* a, MCTSEdge* b) {
        int va = (a->child != nullptr) ? a->child->visits : 0;
        int vb = (b->child != nullptr) ? b->child->visits : 0;
        if (va != vb) return va > vb;
        double qa = (a->child != nullptr && a->child->visits > 0)
                    ? a->child->expected_value(contempt) : 1e9;
        double qb = (b->child != nullptr && b->child->visits > 0)
                    ? b->child->expected_value(contempt) : 1e9;
        return qa < qb;
    });
    int shown = (top_n > 0 && (int)kids.size() > top_n) ? top_n : (int)kids.size();

    char gh[256];
    snprintf(gh, sizeof(gh), "  %-8s %8s %8s %10s %10s %8s",
             "reply", "logit", "visits", "Q(target)", "rawV(child)", "outcome");
    logger.log("INFO", gh);

    for (int j = 0; j < shown; ++j) {
        MCTSEdge* e = kids[j];
        MCTSNode* c = e->child;
        int c_visits = (c != nullptr) ? c->visits : 0;
        double q_target, child_rawV;
        std::string oc;
        if (c != nullptr && c->visits > 0) {
            q_target   = -c->expected_value(contempt);
            child_rawV = (c->raw_w - c->raw_l());
        } else if (c != nullptr) {
            q_target   = -((c->raw_w - c->raw_l()) + contempt * c->raw_d);
            child_rawV = (c->raw_w - c->raw_l());
        } else {
            q_target   = 0.0;
            child_rawV = 0.0;
        }
        oc = (c != nullptr && c->has_forced_outcome()) ? std::to_string(c->forced_outcome) : "None";

        char line[512];
        snprintf(line, sizeof(line),
            "  %-8s %8.3f %8d %+10.4f %+10.4f %8s",
            chess::uci::moveToUci(e->move).c_str(), e->raw_logit, c_visits,
            q_target, child_rawV, oc.c_str());
        logger.log("INFO", line);
    }
    if (shown < (int)kids.size()) {
        char more[64];
        snprintf(more, sizeof(more), "  ... (%d more replies)", (int)kids.size() - shown);
        logger.log("INFO", more);
    }
    logger.log("INFO", "=== end node-by-path dump ===");
    logger.log("INFO", "");
}

// End-of-search summary: outer timings, profiling sub-buckets (only meaningful
// if the corresponding _select / _backpropagate / _retrieve_inference calls
// happened while INFO log was on), and pool occupancy. Self-gated -- callers
// don't need to wrap in a level check.
void MCTSEngine::_log_system_stats() {
    if (logger.get_level() > 20) return;

    logger.log("INFO", "--- System Stats (" + std::to_string(simulation_count) + " sims) ---");

    char buffer[256];
    auto log_timer = [&](const char* label, double value) {
        snprintf(buffer, sizeof(buffer), "%-35s %.4f", label, value);
        logger.log("INFO", buffer);
    };

    log_timer("Selection time:", time_selection);
    log_timer("  -> gumbel score:", time_select_gscore);
    log_timer("  -> softmax exp:", time_select_softmax);
    log_timer("  -> other (scan/argmax/move):", time_select_other);
    log_timer("Queueing time:", time_queueing);
    log_timer("Retrieving time:", time_retrieval);
    log_timer("  -> pop (queue wait):", time_retrieve_pop);
    log_timer("  -> process (tensor+writeback):", time_retrieve_process);
    log_timer("Expansion time:", time_expansion);
    log_timer("Backpropagation time:", time_backpropagation);
    log_timer("  -> stat update (visits/sums):", time_backprop_stat_update);
    log_timer("  -> minimax (sibling scan):", time_backprop_minimax);
    log_timer("  -> other (setup/vloss/flip):", time_backprop_other);
    log_timer("Forced waiting for inference time:", time_wait_for_inference);

    const size_t n_used = node_pool.used();
    const size_t n_cap  = node_pool.capacity();
    const size_t e_used = edge_pool.used();
    const size_t e_cap  = edge_pool.capacity();

    snprintf(buffer, sizeof(buffer),
        "Node pool: %zu / %zu (%.1f%%) x %zuB = %.2f / %.2f MB",
        n_used, n_cap, n_cap ? (100.0 * n_used / n_cap) : 0.0,
        sizeof(MCTSNode),
        (n_used * sizeof(MCTSNode)) / (1024.0 * 1024.0),
        (n_cap  * sizeof(MCTSNode)) / (1024.0 * 1024.0));
    logger.log("INFO", buffer);

    snprintf(buffer, sizeof(buffer),
        "Edge pool: %zu / %zu (%.1f%%) x %zuB = %.2f / %.2f MB",
        e_used, e_cap, e_cap ? (100.0 * e_used / e_cap) : 0.0,
        sizeof(MCTSEdge),
        (e_used * sizeof(MCTSEdge)) / (1024.0 * 1024.0),
        (e_cap  * sizeof(MCTSEdge)) / (1024.0 * 1024.0));
    logger.log("INFO", buffer);
}

// ---------------------------------------------------------------------------
// Shared sequential-halving building blocks.
// ---------------------------------------------------------------------------

void MCTSEngine::_expand_root() {
    if (root != nullptr && root->is_expanded()) return; // Prevent overwriting reused edges
    
    _queue_leaf_for_inference(root, {});
    _submit_batch();
    while (inference_received < inference_sent) {
        _retrieve_inference(true);
    }
}

int MCTSEngine::_build_candidates(int max_m, std::vector<MCTSEdge*>& all_edges,
                                  std::vector<MCTSEdge*>& active_candidates) {
    all_edges.clear();
    for (int i = 0; i < root->num_children; ++i) {
        all_edges.push_back(root->first_edge + i);
    }
    active_candidates.clear();

    root_gumbel_noise.assign(root->num_children, 0.0f);

    std::uniform_real_distribution<double> dist(0.0, 1.0);
    for (int i = 0; i < (int)all_edges.size(); ++i) {
        MCTSEdge* edge = all_edges[i];
        double u = dist(rng);
        root_gumbel_noise[i] = static_cast<float>(-gumbel_noise * std::log(-std::log(u)));

        root_board.makeMove(edge->move);
        bool is_terminal = root_board.isGameOver().second != chess::GameResult::NONE
                        || root_board.isRepetition(1);
        if (is_terminal) {
            if (edge->child == nullptr) {
                if (!node_pool.has_capacity(1)) {
                    pool_exhausted.store(true, std::memory_order_relaxed);
                    root_board.unmakeMove(edge->move);
                    return 0;
                }
                edge->child = node_pool.allocate(root);
            }
            _handle_terminal_node(edge->child);
        } else {
            active_candidates.push_back(edge);
        }
        root_board.unmakeMove(edge->move);
    }

    int m = std::min(max_m, (int)active_candidates.size());
    if (m == 0) return 0;

    std::sort(active_candidates.begin(), active_candidates.end(), [&](MCTSEdge* a, MCTSEdge* b) {
        int ia = static_cast<int>(a - root->first_edge);
        int ib = static_cast<int>(b - root->first_edge);
        double sa = root_gumbel_noise[ia] + a->raw_logit;
        double sb = root_gumbel_noise[ib] + b->raw_logit;
        return sa > sb;
    });
    active_candidates.resize(m);
    return m;
}

void MCTSEngine::_run_round0(std::vector<MCTSEdge*>& active_candidates, int& remaining_search_depth) {
    int ply_count = ((root_board.fullMoveNumber() - 1) * 2) + (root_board.sideToMove() == chess::Color::BLACK ? 2 : 1);
    int current_move = (ply_count + 1) / 2;
    std::string color = (root_board.sideToMove() == chess::Color::WHITE) ? "WHITE" : "BLACK";

    logger.log("INFO", "===============================================================================================");
    logger.log("INFO", " MOVE " + std::to_string(current_move) + " | PLY " + std::to_string(ply_count) + " | " + color);
    logger.log("INFO", "===============================================================================================");    

    for (MCTSEdge* edge : active_candidates) {
        remaining_search_depth -= 1;
        root_board.makeMove(edge->move);
        if (root_board.isGameOver().second == chess::GameResult::NONE && !root_board.isRepetition(1)) {
            if (edge->child == nullptr) {
                if (!node_pool.has_capacity(1)) {
                    if (!pool_exhausted.exchange(true, std::memory_order_relaxed)) {
                        logger.log("WARNING", "Node pool exhausted during Round 0 materialisation.");
                    }
                    root_board.unmakeMove(edge->move);
                    break;
                }
                edge->child = node_pool.allocate(root);
            }
            MCTSNode* child = edge->child;
            if (!(use_tablebase && _try_tablebase(child))) {
                _queue_leaf_for_inference(child, {edge});
            }
        }
        root_board.unmakeMove(edge->move);
    }
    _submit_batch();
    while (inference_received < inference_sent) {
        _retrieve_inference(true);
    }
}

void MCTSEngine::_halve(std::vector<MCTSEdge*>& active_candidates) {
    active_candidates.erase(
        std::remove_if(active_candidates.begin(), active_candidates.end(),
        [](MCTSEdge* e) { return e->child != nullptr && e->child->has_forced_outcome(); }),
        active_candidates.end()
    );
    if (active_candidates.size() > 1) {
        double max_visits = 1.0;
        for (MCTSEdge* e : active_candidates) {
            MCTSNode* c = e->child;
            if (c != nullptr && c->visits > max_visits) max_visits = c->visits;
        }
        double root_v_mix = root->calculate_v_mix(contempt);
        std::sort(active_candidates.begin(), active_candidates.end(), [&](MCTSEdge* a, MCTSEdge* b) {
            int ia = static_cast<int>(a - root->first_edge);
            int ib = static_cast<int>(b - root->first_edge);
            double noise_a = (ia >= 0 && ia < (int)root_gumbel_noise.size()) ? root_gumbel_noise[ia] : 0.0;
            double noise_b = (ib >= 0 && ib < (int)root_gumbel_noise.size()) ? root_gumbel_noise[ib] : 0.0;
            double sa = a->calculate_gumbel_score(contempt, gumbel_c_visit, gumbel_c_scale, max_visits, root_v_mix, noise_a);
            double sb = b->calculate_gumbel_score(contempt, gumbel_c_visit, gumbel_c_scale, max_visits, root_v_mix, noise_b);
            return sa > sb;
        });
        int cutoff = (active_candidates.size() + 1) / 2;
        active_candidates.resize(cutoff);
    }
}

void MCTSEngine::_flush_inflight() {
    if (!batch_buffer.empty()) _submit_batch();
    while (inference_received < inference_sent) {
        _retrieve_inference(true);
    }
}

void MCTSEngine::_record_nps(int sims, double seconds) {
    if (sims <= 0 || seconds <= 0.0) return;
    const double inst = static_cast<double>(sims) / seconds;
    nps_ewma_ = (nps_ewma_ <= 0.0) ? inst : (nps_alpha_ * inst + (1.0 - nps_alpha_) * nps_ewma_);
}

bool MCTSEngine::_should_return_on_forced_win() const {
    if (!early_return_on_forced_win) return false;
    for (int i = 0; i < root->num_children; ++i) {
        MCTSNode* c = (root->first_edge + i)->child;
        if (c != nullptr && c->has_forced_outcome() && c->forced_outcome == -1) return true;
    }
    return false;
}

bool MCTSEngine::_should_early_stop(const std::vector<MCTSEdge*>& candidates) const {
    if (early_stop_q_gap <= 0.0) return false;
    if (candidates.size() < 2)   return false;

    double best_q   = -2.0;
    double second_q = -2.0;
    MCTSNode* best_c = nullptr;
    int    visited  = 0;
    for (MCTSEdge* e : candidates) {
        MCTSNode* c = e->child;
        if (c == nullptr || c->visits <= early_stop_min_visits) continue;
        ++visited;
        double q = -c->expected_value(contempt);
        if (q > best_q) { second_q = best_q; best_q = q; best_c = c; }
        else if (q > second_q) { second_q = q; }
    }
    if (visited < 2) return false;
    if (best_c != nullptr && best_c->mover_has_draw()) return false;

    return (best_q - second_q) >= early_stop_q_gap;
}

int MCTSEngine::run_simulations_fixed(int search_depth, int max_m) {
    if (logger.get_level() <= 20) {
        logger.log("INFO", "Starting Sequential Halving MCTS. Budget: " + std::to_string(search_depth));
    }

    const auto wall_start = std::chrono::steady_clock::now();
    _expand_root();

    std::vector<MCTSEdge*> all_edges;
    std::vector<MCTSEdge*> active_candidates;
    int m = _build_candidates(max_m, all_edges, active_candidates);
    if (m == 0) return simulation_count;

    int remaining_search_depth = search_depth;
    bool did_round0 = false;
    bool aborted = false;
    int r0_spent = 0;
    int phase_idx = 0;

    while (!active_candidates.empty() && remaining_search_depth > 0) {
        if (stop_requested.load(std::memory_order_relaxed) || pool_exhausted.load(std::memory_order_relaxed)) { aborted = true; break; }

        int num_cands = active_candidates.size();

        if (!did_round0) {
            int before = remaining_search_depth;
            _run_round0(active_candidates, remaining_search_depth);
            r0_spent = before - remaining_search_depth;
            did_round0 = true;
            active_candidates.erase(
                std::remove_if(active_candidates.begin(), active_candidates.end(),
                    [](MCTSEdge* e){ return e->child != nullptr && e->child->has_forced_outcome(); }),
                active_candidates.end());
            num_cands = active_candidates.size();
            if (num_cands == 0) break;
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
        int since_check = 0;
        while (current_phase_budget > 0 && num_cands > 0) {
            MCTSEdge* edge = active_candidates[active_idx];
            MCTSNode* child = edge->child;

            if (child != nullptr && child->has_forced_outcome()) {
                active_candidates.erase(active_candidates.begin() + active_idx);
                num_cands = active_candidates.size();
                if (num_cands == 0) break;
                if (active_idx >= num_cands) active_idx = 0;
                continue;
            }

            if (_run_single_async_simulation(edge)) {
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

            if (++since_check >= 128) {
                since_check = 0;
                if (stop_requested.load(std::memory_order_relaxed) || pool_exhausted.load(std::memory_order_relaxed)) { aborted = true; break; }
            }

            active_idx++;
            if (active_idx >= num_cands) active_idx = 0;
        }

        while (inference_received < inference_sent) {
            _retrieve_inference(true);
        }

        _log_tournament_results(active_candidates,
                        "Phase " + std::to_string(phase_idx) + " End",
                        remaining_search_depth, current_phase_budget, simulation_count);

        if (aborted) break;

        if (_should_early_stop(active_candidates)) {
            logger.log("INFO", "Early stop at phase " + std::to_string(phase_idx) +
                       ": Q gap >= " + std::to_string(early_stop_q_gap));
            break;
        }

        if (_should_return_on_forced_win()) {
            logger.log("INFO", "Early return at phase " + std::to_string(phase_idx) + ": forced win found.");
            break;
        }

        if (active_candidates.size() > 2) {
            _halve(active_candidates);
        }
        phase_idx++;
    }

    _log_tournament_results(all_edges, "Final scores");
    _log_system_stats();

    _flush_inflight();
    _record_nps(simulation_count, std::chrono::duration<double>(std::chrono::steady_clock::now() - wall_start).count());
    return simulation_count;
}

int MCTSEngine::run_simulations_timed(int max_m,
                                      std::chrono::steady_clock::time_point soft_deadline,
                                      std::chrono::steady_clock::time_point hard_deadline) {
    const auto wall_start = std::chrono::steady_clock::now();

    double target_s = std::chrono::duration<double>(soft_deadline - wall_start).count();
    if (target_s < 0.0) target_s = 0.0;
    int search_depth = nps_ewma_ * target_s;

    if (logger.get_level() <= 20) {
        logger.log("INFO", "Starting Timed Sequential Halving MCTS. Planned budget: " +
                   std::to_string(search_depth) + " (nps~" + std::to_string((long long)nps_ewma_) + ")");
    }

    _expand_root();

    std::vector<MCTSEdge*> all_edges;
    std::vector<MCTSEdge*> active_candidates;
    int m = _build_candidates(max_m, all_edges, active_candidates);
    if (m == 0) {
        _record_nps(simulation_count, std::chrono::duration<double>(std::chrono::steady_clock::now() - wall_start).count());
        return simulation_count;
    }

    int remaining_search_depth = search_depth;
    bool did_round0 = false;
    bool aborted = false;
    int r0_spent = 0;
    int phase_idx = 0;

    while (!active_candidates.empty() && remaining_search_depth > 0) {
        if (did_round0 && std::chrono::steady_clock::now() >= soft_deadline) break;
        if (stop_requested.load(std::memory_order_relaxed) || pool_exhausted.load(std::memory_order_relaxed)) { aborted = true; break; }

        int num_cands = active_candidates.size();

        if (!did_round0) {
            int before = remaining_search_depth;
            _run_round0(active_candidates, remaining_search_depth);
            r0_spent = before - remaining_search_depth;
            did_round0 = true;
            active_candidates.erase(
                std::remove_if(active_candidates.begin(), active_candidates.end(),
                    [](MCTSEdge* e){ return e->child != nullptr && e->child->has_forced_outcome(); }),
                active_candidates.end());
            num_cands = active_candidates.size();
            if (num_cands == 0) break;
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
        int since_check = 0;
        int no_progress_streak = 0;
        while (current_phase_budget > 0 && num_cands > 0) {
            MCTSEdge* edge = active_candidates[active_idx];
            MCTSNode* child = edge->child;

            if (child != nullptr && child->has_forced_outcome()) {
                active_candidates.erase(active_candidates.begin() + active_idx);
                num_cands = active_candidates.size();
                if (num_cands == 0) break;
                if (active_idx >= num_cands) active_idx = 0;
                continue;
            }

            if (_run_single_async_simulation(edge)) {
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

            if (++since_check >= 128) {
                since_check = 0;
                if (std::chrono::steady_clock::now() >= hard_deadline
                    || stop_requested.load(std::memory_order_relaxed)
                    || pool_exhausted.load(std::memory_order_relaxed)) {
                    aborted = true; break;
                }
            }

            active_idx++;
            if (active_idx >= num_cands) active_idx = 0;
        }

        while (inference_received < inference_sent) {
            _retrieve_inference(true);
        }

        _log_tournament_results(active_candidates, "Phase " + std::to_string(phase_idx) + " End");

        if (aborted) break;

        if (_should_early_stop(active_candidates)) {
            logger.log("INFO", "Early stop at phase " + std::to_string(phase_idx) +
                       ": Q gap >= " + std::to_string(early_stop_q_gap));
            break;
        }

        if (_should_return_on_forced_win()) {
            logger.log("INFO", "Early return at phase " + std::to_string(phase_idx) + ": forced win found.");
            break;
        }

        if (active_candidates.size() > 2) {
            _halve(active_candidates);
        }
        phase_idx++;
    }

    _log_tournament_results(all_edges, "Final scores");
    _log_system_stats();

    _flush_inflight();
    _record_nps(simulation_count, std::chrono::duration<double>(std::chrono::steady_clock::now() - wall_start).count());
    return simulation_count;
}

int MCTSEngine::run_simulations_timed_inference(std::chrono::steady_clock::time_point target) {
    // ---- INFERENCE-ONLY: deficit-selection at root, soft-deadline stop ----
    // Sequential halving is not used here. Every sim descends from root using
    // the same _select() call used at non-root, which applies the Gumbel-score
    // pi_prime + deficit-with-prior-mixing formula uniformly at every level.
    //
    // Stops at target (target budget). hard_deadline is accepted for
    // signature symmetry with the training variant but not consulted -- soft
    // is authoritative for inference.
    //
    // Do NOT call this from training: it breaks the SH-based policy target
    // machinery. max_m is ignored for the same reason.
    //
    // Batch-management pattern mirrors _run_single_async_simulation exactly.
    // -------------------------------------------------------------------
    const auto wall_start = std::chrono::steady_clock::now();

    if (logger.get_level() <= 20) {
        double budget_s = std::chrono::duration<double>(target - wall_start).count();
        if (budget_s < 0.0) budget_s = 0.0;
        logger.log("INFO", "Starting Timed Inference MCTS (deficit-at-root). Soft budget: "
                   + std::to_string(budget_s) + "s (nps~"
                   + std::to_string((long long)nps_ewma_) + ")");
    }

    _expand_root();
    if (root->num_children == 0) {
        _record_nps(simulation_count,
                    std::chrono::duration<double>(std::chrono::steady_clock::now() - wall_start).count());
        return simulation_count;
    }

    bool aborted = false;
    int since_check = 0;
    int no_progress_streak = 0;
    const int deadline_check_interval = 128;
    const int max_no_progress = 4096;

    std::vector<MCTSEdge*> simulation_path;
    simulation_path.reserve(64);

    std::vector<MCTSEdge*> all_edges;
    all_edges.reserve(root->num_children);
    for (int i = 0; i < root->num_children; ++i) {
        all_edges.push_back(root->first_edge + i);
    }

    while (true) {
        // Periodic soft-deadline / stop / pool check.
        if (++since_check >= deadline_check_interval) {
            since_check = 0;
            if (std::chrono::steady_clock::now() >= target
                || stop_requested.load(std::memory_order_relaxed)
                || pool_exhausted.load(std::memory_order_relaxed)) {
                aborted = true;
                break;
            }

            if (_should_early_stop(all_edges)) {
                logger.log("INFO", "Early stop: Q gap >= " + std::to_string(early_stop_q_gap));
                break;
            }

            if (_should_return_on_forced_win()) {
                logger.log("INFO", "Early return: forced win found.");
                break;
            }
        }



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
                // root->is_unavailable() becomes true when num_available_children
                // reaches 0 -- i.e. every root child has a proven forced_outcome.
                // That's legitimate search completion, not a stall. Exit cleanly.
                if (root->is_unavailable()) {
                    if (logger.get_level() <= 20) {
                        logger.log("INFO",
                            "Search complete: all root children have proven outcomes.");
                    }
                    aborted = true;
                    break;
                }
                if (++no_progress_streak > max_no_progress) {
                    logger.log("WARNING",
                        "Timed inference loop stalled: no progress and nothing in flight. "
                        "unavailable=" + std::to_string(root->is_unavailable()) +
                        " slots_empty=" + std::to_string(buffer_free_slots.empty()));
                    aborted = true;
                    break;
                }
                continue;
            }
            _retrieve_inference(true);
            continue;
        }

        simulation_path.clear();
        MCTSNode* leaf = _select(root, simulation_path);

        if (pool_exhausted.load(std::memory_order_relaxed)) {
            aborted = true;
            break;
        }

        if (simulation_path.empty()) {
            logger.log("WARNING", "_select did not descend from root; ending search.");
            break;
        }

        no_progress_streak = 0;

        _dispatch_selected_leaf(leaf, simulation_path);
    }

    _drain_pending_inference();

    _log_tournament_results(all_edges, "Final scores");

    _log_system_stats();
    _flush_inflight();
    _record_nps(simulation_count,
                std::chrono::duration<double>(std::chrono::steady_clock::now() - wall_start).count());
    return simulation_count;
}

int MCTSEngine::run_simulations_fixed_inference(int search_depth) {
    // ---- INFERENCE-ONLY: deficit-selection at root, fixed sim budget ----
    // Same shape as run_simulations_timed_inference but stops at a target
    // simulation count instead of a wall-clock deadline. Every sim descends
    // from root via _select() (deficit + Gumbel score + prior mixing).
    //
    // Do NOT call this from training: it breaks the SH-based policy target
    // machinery.
    // -------------------------------------------------------------------
    const auto wall_start = std::chrono::steady_clock::now();

    if (logger.get_level() <= 20) {
        logger.log("INFO", "Starting Fixed Inference MCTS (deficit-at-root). Budget: "
                   + std::to_string(search_depth));
    }

    _expand_root();
    if (root->num_children == 0) {
        _record_nps(simulation_count,
                    std::chrono::duration<double>(std::chrono::steady_clock::now() - wall_start).count());
        return simulation_count;
    }

    const int start_count = simulation_count;
    bool aborted = false;
    int since_check = 0;
    int no_progress_streak = 0;
    const int stop_check_interval = 128;
    const int max_no_progress = 4096;

    std::vector<MCTSEdge*> simulation_path;
    simulation_path.reserve(64);

    while ((simulation_count - start_count) < search_depth) {
        // Periodic stop / pool check.
        if (++since_check >= stop_check_interval) {
            since_check = 0;
            if (stop_requested.load(std::memory_order_relaxed)
                || pool_exhausted.load(std::memory_order_relaxed)) {
                aborted = true;
                break;
            }
        }

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
                // Legitimate completion: all root children proven-forced.
                if (root->is_unavailable()) {
                    if (logger.get_level() <= 20) {
                        logger.log("INFO",
                            "Search complete: all root children have proven outcomes.");
                    }
                    aborted = true;
                    break;
                }
                if (++no_progress_streak > max_no_progress) {
                    logger.log("WARNING",
                        "Fixed inference loop stalled: no progress and nothing in flight. "
                        "unavailable=" + std::to_string(root->is_unavailable()) +
                        " slots_empty=" + std::to_string(buffer_free_slots.empty()));
                    aborted = true;
                    break;
                }
                continue;
            }
            _retrieve_inference(true);
            continue;
        }

        simulation_path.clear();
        MCTSNode* leaf = _select(root, simulation_path);

        if (pool_exhausted.load(std::memory_order_relaxed)) {
            aborted = true;
            break;
        }

        if (simulation_path.empty()) {
            logger.log("WARNING", "_select did not descend from root; ending search.");
            break;
        }

        no_progress_streak = 0;

        _dispatch_selected_leaf(leaf, simulation_path);
    }

    _drain_pending_inference();

    std::vector<MCTSEdge*> all_edges;
    all_edges.reserve(root->num_children);
    for (int i = 0; i < root->num_children; ++i) {
        all_edges.push_back(root->first_edge + i);
    }
    _log_tournament_results(all_edges, "Final scores");

    _log_system_stats();
    _flush_inflight();
    _record_nps(simulation_count,
                std::chrono::duration<double>(std::chrono::steady_clock::now() - wall_start).count());
    return simulation_count;
}

void MCTSEngine::_backpropagate_minimax(MCTSNode* node) {
    if (node->num_children == 0) return;

    int best_win_dtm = 999999;
    int worst_loss_dtm = -1;
    int best_draw_dtm = 999999;

    bool has_winning_child = false;
    bool has_drawing_child = false;
    bool all_children_proven = true;
    bool all_children_are_losses = true;

    // For mover_has_draw's "no escape" case: every non-losing child (i.e. every
    // child parent might actually play) is either forced_outcome==0 or has
    // mover_has_draw set. Proven-losing children (fo==1 from parent's view)
    // are ignored -- parent won't play them, so they don't affect the argument
    // that parent's value is <= 0.
    bool has_non_losing_child = false;
    bool all_non_losing_are_draw_capped = true;

    bool had_outcome = node->has_forced_outcome();

    for (int i = 0; i < node->num_children; ++i) {
        MCTSEdge* edge = node->first_edge + i;
        MCTSNode* child = edge->child;

        if (child == nullptr) {
            all_children_proven = false;
            all_children_are_losses = false;
            has_non_losing_child = true;
            all_non_losing_are_draw_capped = false;
            continue;
        }

        if (child->has_forced_outcome()) {
            const int outcome = child->forced_outcome;
            if (outcome == -1) {
                has_winning_child = true;
                if (child->distance_to_mate < best_win_dtm) best_win_dtm = child->distance_to_mate;
                has_non_losing_child = true;
                all_non_losing_are_draw_capped = false;
            } else if (outcome == 0) {
                has_drawing_child = true;
                all_children_are_losses = false;
                has_non_losing_child = true;
                if (child->distance_to_mate < best_draw_dtm) best_draw_dtm = child->distance_to_mate;
            } else { // outcome == 1: proven loss for parent
                if (child->distance_to_mate > worst_loss_dtm) worst_loss_dtm = child->distance_to_mate;
            }
        } else {
            all_children_proven = false;
            all_children_are_losses = false;
            has_non_losing_child = true;
            if (!child->mover_has_draw()) {
                all_non_losing_are_draw_capped = false;
            }
        }
    }

    // ---- Forced-outcome proof (monotonic, provable). --------------------------
    // Interior draws (fo=0) restored: all children proven, no winning child,
    // at least one drawing child -> mover picks the draw. DTM is min drawing
    // child's DTM + 1 (shortest path to the certain draw).
    if (has_winning_child) {
        node->forced_outcome = 1;
        node->distance_to_mate = static_cast<int16_t>(best_win_dtm + 1);
    } else if (all_children_proven) {
        if (all_children_are_losses) {
            node->forced_outcome = -1;
            node->distance_to_mate = static_cast<int16_t>(worst_loss_dtm + 1);
        } else if (has_drawing_child) {
            node->forced_outcome = 0;
            node->distance_to_mate = static_cast<int16_t>(best_draw_dtm + 1);
        }
    }

    // Decrement parent's available-child count for ANY newly-proven outcome
    // via the shared cascade helper.
    if (!had_outcome && node->has_forced_outcome() && node->parent != nullptr) {
        if (!node->is_unavailable()) {
            _propagate_unavailability_upward(node);
        }
    }

    // ---- mover_has_draw: non-monotonic PV-based + no-escape rule. -------------
    if (node->has_forced_outcome()) {
        node->set_mover_has_draw(node->forced_outcome == 0);
        return;
    }

    const bool no_escape_case = has_non_losing_child && all_non_losing_are_draw_capped;

    int    best_i = -1;
    double best_q = -1e20;
    for (int i = 0; i < node->num_children; ++i) {
        MCTSNode* c = (node->first_edge + i)->child;
        if (c == nullptr) continue;
        if (c->visits == 0) continue;
        if (c->has_forced_outcome() && c->forced_outcome == 1) continue;

        const double q = -c->expected_value(contempt);
        if (q > best_q) { best_q = q; best_i = i; }
    }
    bool pv_is_draw_capped = false;
    if (best_i >= 0) {
        MCTSNode* pv_child = (node->first_edge + best_i)->child;
        pv_is_draw_capped = (pv_child->forced_outcome == 0) || pv_child->mover_has_draw();
    }

    node->set_mover_has_draw(pv_is_draw_capped || no_escape_case);
}

// Sub-buckets time_backprop_stat_update / _minimax / _other approximate their
// parent time_backpropagation. Gated on INFO log level via `profile` bool;
// zero overhead when off.
void MCTSEngine::_backpropagate(MCTSNode* node, double w, double d, double l, bool is_terminal) {
    const bool profile = logger.get_level() <= 20;
    auto start_time = NOW();

    auto other_start = profile ? NOW() : start_time;
    if (is_terminal) {
        node->forced_outcome = static_cast<int8_t>((w > 0.0) ? 1 : ((l > 0.0) ? -1 : 0));
        node->distance_to_mate = 0;
    } else {
        _virtual_loss(node, false);
        _unmark_selected(node);
    }

    MCTSNode* current_node = node;
    current_node->raw_w = w;
    current_node->raw_d = d;

    double current_w = w;
    double current_d = d;
    double current_l = l;

    if (logger.get_level() <= 10) {
        logger.log("DEBUG",
            chess::uci::moveToUci(_incoming_move(current_node))
            + " raw WDL: " + std::to_string(w) + "/" + std::to_string(d) + "/" + std::to_string(l));
    }
    if (profile) time_backprop_other += ELAPSED(other_start, NOW());

    while (current_node != nullptr) {
        // -- stat update: visit/sum increments --
        auto stat_start = profile ? NOW() : start_time;
        current_node->visits += 1;
        current_node->w_sum += static_cast<float>(current_w);
        current_node->d_sum += static_cast<float>(current_d);
        current_node->l_sum += static_cast<float>(current_l);
        current_node->update_cached_q(contempt);   // keep cache coherent for _select

        if (logger.get_level() <= 10) {
            logger.log("DEBUG",
                chess::uci::moveToUci(_incoming_move(current_node))
                + " updated WDL sums: " + std::to_string(current_node->w_sum)
                + "/" + std::to_string(current_node->d_sum)
                + "/" + std::to_string(current_node->l_sum));
        }
        if (profile) time_backprop_stat_update += ELAPSED(stat_start, NOW());

        // -- minimax: loops over parent's children looking for forced outcomes.
        //    Suspected hog because it scans siblings at every ancestor level. --
        auto minimax_start = profile ? NOW() : start_time;
        _backpropagate_minimax(current_node);
        if (profile) time_backprop_minimax += ELAPSED(minimax_start, NOW());

        // -- perspective flip + walk to parent --
        auto flip_start = profile ? NOW() : start_time;
        double temp_w = current_w;
        current_w = current_l;
        current_l = temp_w;

        current_node = current_node->parent;
        if (profile) time_backprop_other += ELAPSED(flip_start, NOW());
    }
    time_backpropagation += ELAPSED(start_time, NOW());
}

void MCTSEngine::_virtual_loss(MCTSNode* node, bool is_applying) {
    int multiplier = is_applying ? 1 : -1;
    MCTSNode* current_node = node;

    while (current_node != nullptr) {
        current_node->visits += (1 * multiplier);
        current_node->l_sum += static_cast<float>(virtual_loss * multiplier);
        current_node->update_cached_q(contempt);   // keep cache coherent for _select
        current_node = current_node->parent;
    }
}