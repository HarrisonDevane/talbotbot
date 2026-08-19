#pragma once

// =============================================================================
// gumbel_action_selector.hpp
//
// Training-time action selection. Rule D:
//   * ply <= temperature_ply_cutoff -> Boltzmann sample on q_tilde with
//     σ correction (search-averaged WDL variance). Weight is
//       w(a) = N(a) * exp(-q_drop(a) / temperature_q_decay)
//   * else -> deterministic "gumbel winner": sort by (visits desc,
//     gumbel_score desc), take top-2, pick whichever of the two has the
//     higher gumbel_score.
//
// _best_q is also overridden to preserve the pre-split top-node formula so
// Rules B (draw) and C (resign) trigger identically to the old code.
// =============================================================================

#include "action_selector_base.hpp"

class GumbelActionSelector : public ActionSelectorBase {
public:
    // Config inherits SharedConfig; add gumbel-only fields below.
    // Aggregate init (C++17): GumbelActionSelector::Config{{contempt, ...}, temperature_q_decay}
    struct Config : public SharedConfig {
        double temperature_q_decay;
    };

    GumbelActionSelector(std::string name, int worker_id, Config cfg, Logger& logger);

protected:
    // Preserves the pre-split formula: sort non_forced_visited by
    // (visits desc, gumbel_score desc); take top-2; reference node is
    // whichever of the two has the higher gumbel_score; return
    // -expected_value(contempt) of that node.
    double _best_q(std::vector<MCTSNode*>& non_forced_visited) override;

    chess::Move _select_from_visited(
        const std::vector<MCTSNode*>& non_forced_visited,
        int ply_count) override;

private:
    double temperature_q_decay_;
};