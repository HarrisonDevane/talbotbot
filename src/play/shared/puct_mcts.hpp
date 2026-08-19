#pragma once

// =============================================================================
// puct_mcts.hpp
//
// AlphaZero-style PUCT MCTS. Used by play (main_uci, tournament external
// engines, lichess).
//
// Selection strategy: at every node, argmax of
//   UCB(a) = Q(a) + cpuct * P(a) * sqrt(sum_N) / (1 + N(a))
//
// where P(a) is the softmax of raw policy logits over currently-available
// children and Q(a) uses v_mix as the FPU (First Play Urgency) for unvisited
// children. No sequential halving, no phase structure, no gumbel noise.
//
// Two entry points:
//   run_simulations_fixed(max_nodes)
//       Flat loop: run sims until the budget is spent. Used for UCI
//       `go nodes N` and for opening plies where the NPS estimate is stale.
//
//   run_simulations_timed(soft, hard)
//       Flat loop with a deadline check each iteration. Hard deadline is
//       the actual stop. Soft is currently unused; it becomes meaningful
//       when early-termination lands (best-move-uncatchable early exit).
//
// Everything below `MctsBase` here is PUCT-specific -- one selection
// function, one simulation wrapper, two top-level entry points. All the
// batched-inference plumbing, backpropagation, virtual loss, tablebase
// probing, node-by-path debug, and NPS estimator lives in MctsBase.
// =============================================================================

#include "mcts_base.hpp"

class PuctMCTS : public MctsBase {
public:
    // ---- PUCT-specific state (public: read by callers / logging) ----
    double cpuct;

    PuctMCTS(
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
        bool use_tablebase = false
    );

    // Fixed node budget. Used by UCI for `go nodes N` and early plies where
    // the NPS estimate has not warmed up yet.
    int run_simulations_fixed(int max_nodes);

    // Deadline-based. hard_deadline is the actual stop. soft_deadline is
    // accepted for API stability and future early-termination; presently
    // ignored.
    int run_simulations_timed(std::chrono::steady_clock::time_point soft_deadline,
                              std::chrono::steady_clock::time_point hard_deadline);

private:
    // PUCT selection: walks from `start_node` down to a leaf, choosing
    // argmax { Q + cpuct * P * sqrt(sum_N) / (1 + N) } at every node.
    MCTSNode* _select(MCTSNode* start_node, std::vector<MCTSNode*>& simulation_path);

    // One simulation from root. Wraps selection + terminal/tablebase handling
    // + leaf queueing. Returns true iff a simulation was actually performed
    // (caller only charges budget then).
    bool _run_single_async_simulation();

    // Per-search summary line at end of run_simulations_*.
    void _log_search_summary();
};