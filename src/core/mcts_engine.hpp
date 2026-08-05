#pragma once

#include <vector>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <memory>
#include <chrono>
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

class MCTSEngine {
public:
    int worker_batch_size;
    int worker_id;
    double deficit_eps;
    double virtual_loss;
    double contempt;
    double draw_cutoff;
    int simulation_count;
    int inference_sent;
    int inference_received;
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

    MCTSEngine(
        int node_pool_capacity, 
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
        bool use_tablebase = false
    );

    void reset(const chess::Board& board, const std::vector<chess::Board>& history);

    // Self-play / fixed-budget path. Runs the full sequential-halving schedule
    // to completion. Body is UNCHANGED from the original run_simulations.
    int run_simulations_fixed(int search_depth, int max_m);

    // Clocked path. Plans the schedule from the trailing NPS estimate, stops at
    // the soft deadline (phase boundary) or hard deadline (mid-phase, with drain).
    // Deadlines are steady_clock (monotonic). max_nodes is the pool-safety cap.
    int run_simulations_timed(int max_m,
                              std::chrono::steady_clock::time_point soft_deadline,
                              std::chrono::steady_clock::time_point hard_deadline);

    // Trailing nodes/sec estimate (EWMA). 0.0 == no data yet. Survives reset();
    // cleared per game via reset_nps_history().
    double estimated_nps() const { return nps_ewma_; }
    void   reset_nps_history(double e) { nps_ewma_ = e; }
    void   set_nps_alpha(double a) { nps_alpha_ = a; }   // wired from time_control.nps_ewma_alpha

private:
    void _wait_for_inference();
    MCTSNode* _select(MCTSNode* start_node, std::vector<MCTSNode*>& simulation_path);
    void _backpropagate_minimax(MCTSNode* node);
    void _backpropagate(MCTSNode* node, double w, double d, double l, bool is_terminal);
    void _virtual_loss(MCTSNode* node, bool is_applying);
    
    void _mark_selected(MCTSNode* node);
    void _unmark_selected(MCTSNode* node);
    void _retrieve_inference(bool block);
    void _submit_batch();
    void _handle_terminal_node(MCTSNode* leaf);
    // Syzygy WDL probe for the current leaf (UCI only). Returns true iff the
    // leaf was resolved as a proven terminal here; false falls through to NN.
    bool _try_tablebase(MCTSNode* leaf);
    void _queue_leaf_for_inference(MCTSNode* leaf, const std::vector<MCTSNode*>& simulation_path); 
    // Returns true iff one simulation was actually performed (a leaf was
    // queued for inference, or a terminal node was handled). Returns false
    // on the no-op exit path (candidate subtree unavailable / no free buffer
    // slots, with nothing in flight for this worker). Callers must only
    // charge search budget when this returns true.
    bool _run_single_async_simulation(MCTSNode* start_node);
    
    void _log_tournament_results(const std::vector<MCTSNode*>& candidates,
                                const std::string& phase_name,
                                int remaining_search_depth = -1,
                                int phase_budget = -1,
                                int sims_completed = -1);
    // Debug: navigate from root following a UCI move path and dump the target
    // node's RAW network value plus its children. For "why is this node's Q
    // wrong" questions -- the raw value tells you if the value head is blind.
    void _log_node_by_path(const std::vector<std::string>& uci_path, int top_n);

    // Shared sequential-halving building blocks (used by both _fixed and _timed).
    void _expand_root();
    int  _build_candidates(int max_m, std::vector<MCTSNode*>& all_nodes,
                           std::vector<MCTSNode*>& active_candidates);
    void _run_round0(std::vector<MCTSNode*>& active_candidates, int& remaining_search_depth);
    void _rescore(std::vector<MCTSNode*>& nodes);
    void _halve(std::vector<MCTSNode*>& active_candidates);
    void _flush_inflight();

    // Fold one completed search's (sims, seconds) into the trailing NPS EWMA.
    void _record_nps(int sims, double seconds);
    double nps_ewma_;   // trailing nodes/sec; NOT touched by reset()
    double nps_alpha_;   // EWMA smoothing; default preserves prior behaviour, override via set_nps_alpha()

    template <typename Predicate, typename WorkFn>
    void _spin_wait(Predicate should_keep_waiting, WorkFn work_fn);
};