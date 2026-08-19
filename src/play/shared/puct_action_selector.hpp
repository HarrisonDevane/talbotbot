#pragma once

// =============================================================================
// puct_action_selector.hpp
//
// Play-time action selection. Rule D:
//   * ply <= temperature_ply_cutoff -> sample proportional to
//         w(a) = N(a) ^ (1 / temperature_visits)
//     temperature_visits = 1.0 gives sampling proportional to visits;
//     temperature_visits -> 0 approaches argmax N (matches deterministic).
//   * else -> deterministic argmax by visits, tie-break by Q.
//
// Rules A/B/C/E and _best_q come from ActionSelectorBase unchanged. _best_q
// uses the natural max-Q formulation there; there is no gumbel_score to
// consult in a PUCT tree.
// =============================================================================

#include "action_selector_base.hpp"

class PuctActionSelector : public ActionSelectorBase {
public:
    // Config inherits SharedConfig; add puct-only fields below.
    // Aggregate init (C++17): PuctActionSelector::Config{{contempt, ...}, temperature_visits}
    struct Config : public SharedConfig {
        double temperature_visits;
    };

    PuctActionSelector(std::string name, int worker_id, Config cfg, Logger& logger);

protected:
    chess::Move _select_from_visited(
        const std::vector<MCTSNode*>& non_forced_visited,
        int ply_count) override;

private:
    double temperature_visits_;
};