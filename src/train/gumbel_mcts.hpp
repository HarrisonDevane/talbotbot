#pragma once

// =============================================================================
// gumbel_mcts.hpp
//
// Gumbel-top-k + sequential-halving MCTS. Used by training (data_generator).
//
// Selection strategy: at every non-root node, argmax of the deficit
//   pi'(a) - N(a) / (1 + sum_N)
// where pi' is a softmax over cached gumbel_scores (raw_logit + noise +
// sigma * v_mix normalisation). This is the "policy-target-network"
// perspective from the Danihelka paper's Appendix; it does not use a cpuct.
//
// Root scheduling: sequential halving. Round 0 evaluates every candidate
// once, then successive phases halve the survivor set. Budget is either a
// fixed sim count (self-play) or a soft/hard deadline pair (analysis).
//
// Everything below `MctsBase` here is gumbel-specific -- selection function,
// candidate builder, phase halving, per-phase rescoring, and the top-level
// run_simulations_* orchestration. All the batched-inference plumbing,
// backpropagation, virtual loss, tablebase probing, and NPS estimator lives
// in MctsBase and is used unmodified.
// =============================================================================

#include "mcts_base.hpp"
#include <random>

class GumbelMCTS : public MctsBase {
public:
    // ---- gumbel-specific state (public: kept where a caller might read them
    //      for logging, matching the old MCTSEngine's convention) ----
    double gumbel_c_visit;
    double gumbel_c_scale;
    double gumbel_noise;
    std::mt19937 rng;

    GumbelMCTS(
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
        bool use_tablebase = false
    );

    // Self-play / fixed-budget path. Runs the full sequential-halving schedule
    // to completion. Gumbel is training-only; the timed path lives in
    // PuctMCTS (which is what play uses).
    int run_simulations_fixed(int search_depth, int max_m);

private:
    // Gumbel deficit selection: walks from start_node down to a leaf, chosen
    // by max { pi'(a) - N(a) / (1 + sum_N) } at every node.
    MCTSNode* _select(MCTSNode* start_node, std::vector<MCTSNode*>& simulation_path);

    // One simulation from `start_node`. Calls _select, then either handles a
    // terminal / tablebase resolution or queues the leaf for NN inference.
    // Returns true iff a simulation was actually performed.
    bool _run_single_async_simulation(MCTSNode* start_node);

    // Sequential-halving helpers.
    int  _build_candidates(int max_m,
                           std::vector<MCTSNode*>& all_nodes,
                           std::vector<MCTSNode*>& active_candidates);
    void _run_round0(std::vector<MCTSNode*>& active_candidates,
                     int& remaining_search_depth);
    void _rescore(std::vector<MCTSNode*>& nodes);
    void _halve(std::vector<MCTSNode*>& active_candidates);

    // Phase-boundary tournament dump -- reads gumbel_score, so it's
    // gumbel-flavoured. _log_node_by_path lives in MctsBase since both
    // engines can use it.
    void _log_tournament_results(const std::vector<MCTSNode*>& candidates,
                                 const std::string& phase_name,
                                 int remaining_search_depth = -1,
                                 int phase_budget = -1,
                                 int sims_completed = -1);
};