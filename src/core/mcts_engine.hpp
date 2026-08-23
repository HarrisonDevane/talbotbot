#pragma once

#include <vector>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <memory>
#include <chrono>
#include <atomic>
#include <torch/torch.h>
#include <random>
#include "chess.hpp"
#include "mcts_node.hpp"
#include "logger.hpp"
#include "concurrentqueue.h"

struct ModelConfig {
    int input_planes;
    int board_dim;
    int policy_moves;
};

// -----------------------------------------------------------------------------
// Pool sizing knobs. Fed from YAML block `pool_sizing` in both train.yaml
// and play_uci.yaml. Set on MCTSEngine post-construction (like early_stop_*).
// -----------------------------------------------------------------------------
struct PoolSizingConfig {
    // Average legal moves per position -- ~35 for chess. Used as the
    // per-node edge multiplier when translating "predicted sims" into
    // "predicted edges".
    double avg_branching = 35.0;

    // Multiplicative slack over the predicted count. Node factor typically
    // 1.5 (accounts for lazy materialisation touching some children more
    // than once via re-descent); edge factor 1.2 (expansion is one-shot).
    double node_safety_factor = 1.5;
    double edge_safety_factor = 1.2;

    // Absolute per-worker ceilings in bytes. predict_pool_needs converts
    // to element counts via sizeof(MCTSNode) / sizeof(MCTSEdge). These are
    // the "run away NPS estimate can't OOM the box" backstop.
    size_t node_hard_cap_bytes = 100ull * 1024 * 1024;
    size_t edge_hard_cap_bytes = 100ull * 1024 * 1024;
};

struct PoolTargets {
    size_t node_target;
    size_t edge_target;
};

template <typename T>
class ThreadSafeQueue {
private:
    std::queue<T> queue_;
    std::mutex mutex_;
    std::condition_variable cond_;

public:
    void push(T item) {
        std::unique_lock<std::mutex> lock(mutex_);
        queue_.push(std::move(item));
        cond_.notify_one();
    }

    bool try_pop(T& item) {
        std::unique_lock<std::mutex> lock(mutex_);
        if (queue_.empty()) return false;
        item = std::move(queue_.front());
        queue_.pop();
        return true;
    }

    T pop_wait() {
        std::unique_lock<std::mutex> lock(mutex_);
        cond_.wait(lock, [this]() { return !queue_.empty(); });
        T item = std::move(queue_.front());
        queue_.pop();
        return item;
    }

    bool empty() {
        std::unique_lock<std::mutex> lock(mutex_);
        return queue_.empty();
    }

    size_t size() {
        std::lock_guard<std::mutex> lock(mutex_);
        return queue_.size();
    }
};

class NodePool {
private:
    std::vector<MCTSNode> pool;
    size_t next_idx = 0;

public:
    explicit NodePool(size_t capacity) { pool.resize(capacity); }

    void reset() { next_idx = 0; }

    // Grow the underlying storage. ONLY SAFE when no external pointers into
    // pool are live -- callers must invoke this only inside reset(), between
    // reset()-ing next_idx and re-allocating root. std::vector::resize can
    // move the storage, invalidating every MCTSNode* the tree holds.
    void grow_to_at_least(size_t n) {
        if (n > pool.size()) pool.resize(n);
    }

    MCTSNode* allocate(MCTSNode* parent = nullptr) {
        if (next_idx >= pool.size()) {
            throw std::runtime_error("NodePool capacity exceeded! Increase initial capacity.");
        }
        MCTSNode* ptr = &pool[next_idx++];
        *ptr = MCTSNode(parent);
        return ptr;
    }

    bool   has_capacity(size_t n) const { return next_idx + n <= pool.size(); }
    size_t remaining()             const { return pool.size() - next_idx; }
    size_t capacity()              const { return pool.size(); }
    size_t used()                  const { return next_idx; }
};

class EdgePool {
private:
    std::vector<MCTSEdge> pool;
    size_t next_idx = 0;

public:
    explicit EdgePool(size_t capacity) { pool.resize(capacity); }

    void reset() { next_idx = 0; }

    // Same safety contract as NodePool::grow_to_at_least.
    void grow_to_at_least(size_t n) {
        if (n > pool.size()) pool.resize(n);
    }

    MCTSEdge* allocate_block(size_t n) {
        if (next_idx + n > pool.size()) {
            throw std::runtime_error("EdgePool capacity exceeded! Increase initial capacity.");
        }
        MCTSEdge* ptr = &pool[next_idx];
        for (size_t i = 0; i < n; ++i) ptr[i] = MCTSEdge{};
        next_idx += n;
        return ptr;
    }

    bool   has_capacity(size_t n) const { return next_idx + n <= pool.size(); }
    size_t remaining()             const { return pool.size() - next_idx; }
    size_t capacity()              const { return pool.size(); }
    size_t used()                  const { return next_idx; }
};

class MCTSEngine {
public:
    int worker_batch_size;
    int worker_id;
    bool two_fold_repetition;
    bool early_terminal_return;
    double deficit_eps;
    double virtual_loss;
    double contempt;
    double draw_cutoff;
    int simulation_count;
    int inference_sent;
    int inference_received;

    int max_selection_depth = 0;
    double gumbel_c_visit;
    double gumbel_c_scale;
    double gumbel_noise;
    std::mt19937 rng;

    double time_selection = 0.0;
    double time_expansion = 0.0;
    double time_backpropagation = 0.0;
    double time_retrieval = 0.0;
    double time_queueing = 0.0;
    double time_misc = 0.0;
    double time_wait_for_inference = 0.0;

    // ---- Diagnostic sub-buckets. Collected only when logger.get_level() <= 20
    // (INFO). Zero cost when info is off: guarded by a locally-cached bool at
    // the top of each profiled function. Sums approximate their parent bucket
    // minus a small chrono-call overhead; interpret as ratios, not absolutes.
    //
    // _select breakdown:
    double time_select_gscore  = 0.0;   // MCTSEdge::calculate_gumbel_score loop
    double time_select_softmax = 0.0;   // std::exp for score and prior softmax
    double time_select_other   = 0.0;   // visit-stat scan + deficit argmax + makeMove

    // _backpropagate breakdown:
    double time_backprop_stat_update = 0.0;   // visits/w_sum/d_sum/l_sum increments
    double time_backprop_minimax     = 0.0;   // _backpropagate_minimax calls
    double time_backprop_other       = 0.0;   // terminal setup / virtual_loss / unmark / perspective flip

    // _retrieve_inference breakdown (expansion/backprop have their own outer
    // counters and are NOT counted here to avoid double-counting):
    double time_retrieve_pop     = 0.0;   // pop_wait / try_pop on result_queue
    double time_retrieve_process = 0.0;   // data_ptr + wdl read + policy writeback + free-slot push

    chess::Board root_board;
    std::vector<chess::Board> base_history;
    MCTSNode* root;
    NodePool node_pool;
    EdgePool edge_pool;
    Logger& logger;

    // Pool sizing config. Set post-construction from YAML. Used by
    // predict_pool_needs() / predict_pool_needs_for_time().
    PoolSizingConfig pool_sizing_cfg;

    moodycamel::ConcurrentQueue<std::pair<int, int>>& inference_queue;
    ThreadSafeQueue<std::vector<int>>& result_queue;
    ThreadSafeQueue<int>& buffer_free_slots;

    std::vector<torch::Tensor>& shared_input_buffer;
    std::vector<torch::Tensor>& shared_policy_buffer;
    std::vector<torch::Tensor>& shared_value_buffer;

    std::vector<MCTSNode*> in_flight_nodes;
    std::vector<std::pair<int, int>> batch_buffer;
    
    torch::DeviceType device;
    torch::ScalarType policy_logits_dtype;

    std::atomic<bool> stop_requested{false};
    void request_stop() { stop_requested.store(true, std::memory_order_relaxed); }
    void clear_stop()   { stop_requested.store(false, std::memory_order_relaxed); }

    double early_stop_q_gap = 0.0;
    int early_stop_min_visits = 0;
    bool early_return_on_forced_win = false;

    std::vector<float> root_gumbel_noise;
    std::atomic<bool> pool_exhausted{false};

    std::atomic<int>* core_wait_count;
    int workers_per_core;
    bool use_tablebase;

    // Constructor takes INITIAL pool capacities. Under the pool_sizing
    // scheme these should be sized for the expected first search (or set
    // to any minimum >= 1 and let the first reset() grow them). See
    // predict_pool_needs() below and the call sites in data_generator /
    // main_uci for the intended usage.
    MCTSEngine(
        int node_pool_capacity,
        int edge_pool_capacity,
        int worker_batch_size, 
        moodycamel::ConcurrentQueue<std::pair<int, int>>& inference_queue,
        ThreadSafeQueue<std::vector<int>>& result_queue, 
        int worker_id, 
        double deficit_eps,
        double virtual_loss,
        double contempt,
        double draw_cutoff, 
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
        bool use_tablebase = false
    );

    // Reset with optional pool sizing. If node_target > current node_pool
    // capacity, node_pool is grown to that size (never shrunk). Same for
    // edge_target / edge_pool. Passing 0 for either target leaves that
    // pool's capacity unchanged. Grow-only: peak-across-lifetime is the
    // committed RAM per worker.
    void reset(const chess::Board& board,
               const std::vector<chess::Board>& history,
               size_t node_target = 0,
               size_t edge_target = 0);

    int run_simulations_fixed(int search_depth, int max_m);
    int run_simulations_timed(int max_m,
                              std::chrono::steady_clock::time_point soft_deadline,
                              std::chrono::steady_clock::time_point hard_deadline);

    double estimated_nps() const { return nps_ewma_; }
    void   reset_nps_history(double e) { nps_ewma_ = e; }
    void   set_nps_alpha(double a) { nps_alpha_ = a; }

    // ---- Pool sizing helpers ---------------------------------------------
    //
    // Convert a predicted simulation count into pool sizes. Applies safety
    // factors, then clamps to the byte caps in pool_sizing_cfg. Floors
    // predicted_sims at 1 internally so a bogus 0 estimate can never zero
    // the pool.
    PoolTargets predict_pool_needs(int predicted_sims) const {
        return predict_pool_needs_static(predicted_sims, pool_sizing_cfg);
    }
    static PoolTargets predict_pool_needs_static(int predicted_sims,
                                                 const PoolSizingConfig& cfg);

    // Convenience for the timed path. Predicts sims from
    //   sims = estimated_nps() * time_s * safety_multiplier
    // then delegates. safety_multiplier is typically time_control.hard_multiplier
    // so we budget enough for a hard-deadline overrun. Callers MUST have
    // primed estimated_nps() via reset_nps_history(default) at least once
    // before the first search (main_uci does this at construction).
    PoolTargets predict_pool_needs_for_time(double time_s,
                                            double safety_multiplier) const;
    // ----------------------------------------------------------------------

private:
    void _wait_for_inference();
    MCTSNode* _select(MCTSNode* start_node, std::vector<MCTSEdge*>& simulation_path);
    void _backpropagate_minimax(MCTSNode* node);
    void _backpropagate(MCTSNode* node, double w, double d, double l, bool is_terminal);
    void _virtual_loss(MCTSNode* node, bool is_applying);

    bool _should_early_stop(const std::vector<MCTSEdge*>& candidates) const;
    
    void _mark_selected(MCTSNode* node);
    void _unmark_selected(MCTSNode* node);
    void _retrieve_inference(bool block);
    void _submit_batch();
    void _handle_terminal_node(MCTSNode* leaf);
    bool _try_tablebase(MCTSNode* leaf);
    void _queue_leaf_for_inference(MCTSNode* leaf, const std::vector<MCTSEdge*>& simulation_path);
    bool _run_single_async_simulation(MCTSEdge* start_edge);

    void _log_tournament_results(const std::vector<MCTSEdge*>& candidates,
                                const std::string& phase_name,
                                int remaining_search_depth = -1,
                                int phase_budget = -1,
                                int sims_completed = -1);
    void _log_node_by_path(const std::vector<std::string>& uci_path, int top_n);
    // End-of-search summary: timing breakdown + pool occupancy. Self-gated
    // on INFO log level (no need for caller to guard). Called from both
    // run_simulations_fixed and run_simulations_timed.
    void _log_system_stats();

    void _expand_root();
    int  _build_candidates(int max_m, std::vector<MCTSEdge*>& all_edges,
                           std::vector<MCTSEdge*>& active_candidates);
    void _run_round0(std::vector<MCTSEdge*>& active_candidates, int& remaining_search_depth);
    void _halve(std::vector<MCTSEdge*>& active_candidates);
    void _flush_inflight();

    void _record_nps(int sims, double seconds);
    double nps_ewma_;
    double nps_alpha_;

    template <typename Predicate, typename WorkFn>
    void _spin_wait(Predicate should_keep_waiting, WorkFn work_fn);
    bool _should_return_on_forced_win() const;
};