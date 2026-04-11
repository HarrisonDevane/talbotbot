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
    double virtual_loss;
    double draw_cutoff;
    int simulation_count;
    int inference_sent;
    int inference_received;
    double gumbel_c_visit;
    double gumbel_c_scale;
    double gumbel_noise;
    std::mt19937 rng;

    ModelConfig model_config;

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

    MCTSEngine(
        int node_pool_capacity, 
        int worker_batch_size, 
        moodycamel::ConcurrentQueue<std::pair<int, int>>& inference_queue,
        ThreadSafeQueue<std::vector<int>>& result_queue, 
        int worker_id, 
        double virtual_loss,
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
        const ModelConfig& model_config, 
        ThreadSafeQueue<int>& buffer_free_slots
    );

    void reset(const chess::Board& board, const std::vector<chess::Board>& history);
    int run_simulations(int search_depth, int max_m);

private:
    MCTSNode* _select(MCTSNode* start_node, std::vector<MCTSNode*>& simulation_path);
    void _backpropagate_minimax(MCTSNode* node);
    void _backpropagate(MCTSNode* node, double value, bool is_terminal);
    void _virtual_loss(MCTSNode* node, bool is_applying);
    
    void _mark_selected(MCTSNode* node);
    void _unmark_selected(MCTSNode* node);
    void _retrieve_inference(bool block);
    void _submit_batch();
    void _handle_terminal_node(MCTSNode* leaf);
    void _queue_leaf_for_inference(MCTSNode* leaf, const std::vector<MCTSNode*>& simulation_path); 
    void _run_single_async_simulation(MCTSNode* start_node);
    
    void _log_tournament_results(const std::vector<MCTSNode*>& candidates, const std::string& phase_name);
};