#pragma once

// =============================================================================
// mcts_base.hpp
//
// Shared MCTS machinery for the two search variants:
//   * GumbelMCTS  -- sequential-halving with Gumbel-top-k root selection.
//                    Used by training (data_generator).
//   * PuctMCTS    -- AlphaZero-style PUCT selection at every node.
//                    Used by play (main_uci, tournament external).
//
// What lives here (shared):
//   - Node pool, root, board, history
//   - Batched-inference plumbing (queues, buffers, in-flight tracking)
//   - Leaf expansion, terminal handling, tablebase probing
//   - Backpropagation (WDL accumulator, minimax proven-outcome propagation)
//   - Virtual loss application
//   - NPS estimator (for both engines' timed runs)
//
// What lives in the derived classes:
//   - _select        -- selection strategy differs entirely between variants
//   - _run_single_async_simulation -- calls _select; per-variant
//   - run_simulations_fixed / run_simulations_timed -- top-level scheduling
//   - Variant-specific state (gumbel_c_*, cpuct, etc.)
//   - Variant-specific display/logging
//
// Extraction rule: this header depends only on chess, torch, logger, and
// the queue types. It does NOT include gumbel_mcts.hpp or puct_mcts.hpp.
// The relationship is one-way: derived reaches down into base's protected
// state; base never reaches up.
// =============================================================================

#include <vector>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <memory>
#include <chrono>
#include <atomic>
#include <thread>          // for std::this_thread::yield in _spin_wait
#include <immintrin.h>     // for _mm_pause in _spin_wait
#include <torch/torch.h>
#include "chess.hpp"
#include "mcts_node.hpp"
#include "logger.hpp"
#include "concurrentqueue.h"

// -----------------------------------------------------------------------------
// Model shape config -- consumed by callers that build the shared tensor
// buffers. Kept here (rather than in a standalone header) because it has
// historically lived alongside the engine and every caller of the engine
// needs it too.
// -----------------------------------------------------------------------------
struct ModelConfig {
    int input_planes;
    int board_dim;
    int policy_moves;
};

// -----------------------------------------------------------------------------
// Bounded MPSC-ish queue used for inference result plumbing. Lives here for
// legacy reasons; nothing in the queue is MCTS-specific.
// -----------------------------------------------------------------------------
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

// -----------------------------------------------------------------------------
// Fixed-capacity node pool. Callers size this generously; overflow throws.
// -----------------------------------------------------------------------------
class NodePool {
private:
    std::vector<MCTSNode> pool;
    size_t next_idx = 0;

public:
    NodePool(size_t capacity) {
        pool.resize(capacity);
    }

    void reset() {
        next_idx = 0;
    }

    MCTSNode* allocate(MCTSNode* parent = nullptr, chess::Move move = chess::Move::NO_MOVE) {
        if (next_idx >= pool.size()) {
            throw std::runtime_error("NodePool capacity exceeded! Increase initial capacity.");
        }
        MCTSNode* ptr = &pool[next_idx++];
        *ptr = MCTSNode(parent, move);
        return ptr;
    }
};

// =============================================================================
// MctsBase: state + methods shared by every MCTS variant.
// =============================================================================
class MctsBase {
public:
    // ---- shared state (public: read directly by ActionSelector, callers) ----
    int worker_batch_size;
    int worker_id;
    bool two_fold_repetition;
    double virtual_loss;
    double contempt;
    double policy_softmax_temp;

    int simulation_count = 0;
    int inference_sent = 0;
    int inference_received = 0;

    // Timing counters -- read by derived logging code.
    double time_selection = 0.0;
    double time_expansion = 0.0;
    double time_backpropagation = 0.0;
    double time_retrieval = 0.0;
    double time_queueing = 0.0;
    double time_misc = 0.0;
    double time_wait_for_inference = 0.0;

    chess::Board root_board;
    std::vector<chess::Board> base_history;
    MCTSNode* root;
    NodePool node_pool;
    Logger& logger;

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

    std::atomic<int>* core_wait_count;
    int workers_per_core;
    bool use_tablebase;

    // ---- lifecycle ----
    MctsBase(
        int node_pool_capacity,
        int worker_batch_size,
        moodycamel::ConcurrentQueue<std::pair<int, int>>& inference_queue,
        ThreadSafeQueue<std::vector<int>>& result_queue,
        int worker_id,
        double virtual_loss,
        double contempt,
        double policy_softmax_temp,
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

    virtual ~MctsBase() = default;

    // Reset per-search state: drain in-flight batches, reset the node pool,
    // re-seat root, clear counters. Derived classes may override to add
    // variant-specific reset behaviour (e.g. PUCT tree reuse) but must call
    // this base implementation.
    virtual void reset(const chess::Board& board, const std::vector<chess::Board>& history);

    // ---- NPS estimator (shared -- both variants have timed search) ----
    double estimated_nps() const { return nps_ewma_; }
    void   reset_nps_history(double e) { nps_ewma_ = e; }
    void   set_nps_alpha(double a)    { nps_alpha_ = a; }

protected:
    // ---- shared machinery: leaf/queue/backprop ----
    void _mark_selected(MCTSNode* node);
    void _unmark_selected(MCTSNode* node);
    void _retrieve_inference(bool block);
    void _submit_batch();
    void _handle_terminal_node(MCTSNode* leaf);
    bool _try_tablebase(MCTSNode* leaf);
    void _queue_leaf_for_inference(MCTSNode* leaf, const std::vector<MCTSNode*>& simulation_path);
    void _expand_root();
    void _flush_inflight();
    void _backpropagate_minimax(MCTSNode* node);
    void _backpropagate(MCTSNode* node, double w, double d, double l, bool is_terminal);
    void _virtual_loss(MCTSNode* node, bool is_applying);

    // Fold one completed search's (sims, seconds) into the trailing NPS EWMA.
    void _record_nps(int sims, double seconds);

    // Debug: navigate from root following a UCI move path and dump the target
    // node's raw network value plus its children. For "why is this node's Q
    // wrong" questions -- the raw value tells you if the value head is blind.
    // Reads gumbel_score on nodes (a shared MCTSNode field), but the semantic
    // interpretation of that column is variant-specific; harmless in PUCT
    // output (just prints 0.0).
    void _log_node_by_path(const std::vector<std::string>& uci_path, int top_n);

    // Spin-wait helper -- template must stay in the header. Derived classes
    // use it when they need to spin on a condition while draining inference.
    // Definition is inline below the class body.
    template <typename Predicate, typename WorkFn>
    void _spin_wait(Predicate should_keep_waiting, WorkFn work_fn);

    // ---- NPS state ----
    double nps_ewma_  = 0.0;    // trailing nodes/sec; NOT touched by reset()
    double nps_alpha_ = 0.4;    // EWMA smoothing; override via set_nps_alpha()
};

// -----------------------------------------------------------------------------
// _spin_wait: template definition. Kept out-of-class only for readability;
// every TU that instantiates it (gumbel_mcts.cpp, puct_mcts.cpp, mcts_base.cpp)
// needs the full body visible here.
// -----------------------------------------------------------------------------
template <typename Predicate, typename WorkFn>
inline void MctsBase::_spin_wait(Predicate should_keep_waiting, WorkFn work_fn) {
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