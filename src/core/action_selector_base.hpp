#pragma once

// =============================================================================
// action_selector_base.hpp
//
// Base class for the post-search action selection cascade. The cascade is:
//
//   Rule A: WIN         -- one or more children have a proven forced win for
//                          us. Pick the shortest mate (uniform among ties).
//   Rule B: DRAW        -- a forced-draw child exists AND best_q <= draw_cutoff
//                          (we're not winning). Pick a draw child at random.
//   Rule C: RESIGN      -- best_q below resignation_cutoff AND this game rolled
//                          resignation-enabled. Return resigned=true.
//   Rule D: PLAY        -- normal case. Delegate to _select_from_visited().
//                          This is where Gumbel and PUCT diverge.
//   Rule E: DELAY MATE  -- no visited non-forced options; pick the losing move
//                          with the longest DTM. Fallback to random.
//
// Rules A/B/C/E and the categorization/best-Q computation are shared. Rule D
// is variant-specific: derived classes override _select_from_visited().
//
// Config split: SharedConfig here holds the fields both variants need.
// Variant-specific configs (GumbelActionSelector::Config, PuctActionSelector::Config)
// carry the shared fields plus their own knobs and decompose into SharedConfig
// in the derived constructor.
// =============================================================================

#include <vector>
#include <string>
#include <random>
#include "chess.hpp"
#include "mcts_node.hpp"
#include "logger.hpp"

struct SelectionResult {
    chess::Move best_move = chess::Move::NO_MOVE;
    bool resigned = false;
};

class ActionSelectorBase {
public:
    struct SharedConfig {
        double contempt;
        double draw_cutoff;
        double resignation_probability;
        double resignation_cutoff;
        int    temperature_ply_cutoff;
    };

    ActionSelectorBase(std::string name, int worker_id, SharedConfig cfg, Logger& logger);
    virtual ~ActionSelectorBase() = default;

    // Roll the per-game dice for resignation. Call at the start of each game.
    void reset_for_new_game();

    void set_name(const std::string& new_name) { name = new_name; }

    // Runs the full A/B/C/D/E cascade.
    SelectionResult select_move(MCTSNode* root, int ply_count);

protected:
    // Categorized root children. Population happens once in select_move() and
    // is handed to _select_from_visited() for Rule D.
    struct CategorizedChildren {
        std::vector<MCTSNode*> winning;             // child proven-lost (opp loses) -> we win
        std::vector<MCTSNode*> losing;              // child proven-won (opp wins)   -> we lose
        std::vector<MCTSNode*> drawing;             // child proven draw OR practical draw
        std::vector<MCTSNode*> non_forced_visited;  // real search results
    };

    // Fill the four buckets from root's children. "Practical draw" detection
    // (opponent's best reply is a forced draw) folds a non-forced child into
    // drawing.
    CategorizedChildren _categorize(MCTSNode* root);

    // Max of -expected_value(contempt) over visited non-forced children.
    // Returns -2.0 if the set is empty (a sentinel below any real Q).
    //
    // VIRTUAL because Gumbel preserves its historical formula: sort by
    // (visits desc, gumbel_score desc), take top-of-top-2 as the reference
    // node, return -expected_value of that node. That formula slightly
    // differs from max-Q at unconverged searches and is training-tuned;
    // GumbelActionSelector overrides to keep behaviour identical to the
    // pre-split code. PUCT uses this default (true max Q).
    //
    // Passed by non-const ref so overrides may sort in place if they want.
    virtual double _best_q(std::vector<MCTSNode*>& non_forced_visited);

    // ---- variant-specific: Rule D ----
    // Called with a non-empty non_forced_visited set. Return the move to play.
    virtual chess::Move _select_from_visited(
        const std::vector<MCTSNode*>& non_forced_visited,
        int ply_count) = 0;

    // ---- shared state, accessible to derived classes ----
    std::string      name;
    int              worker_id;
    SharedConfig     shared_cfg;
    bool             use_resignation;
    Logger&          logger;
    std::mt19937     rng;
};