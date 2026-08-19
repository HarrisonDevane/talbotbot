// =============================================================================
// action_selector_base.cpp
//
// Shared rule cascade: A (win) / B (draw) / C (resign) / D (delegate) / E
// (delay mate). Categorization + practical-draw detection also lives here.
// Only Rule D is variant-specific and gets punted to _select_from_visited().
// =============================================================================

#include "action_selector_base.hpp"
#include <algorithm>
#include <chrono>
#include <random>
#include <string>

ActionSelectorBase::ActionSelectorBase(
    std::string name, int worker_id, SharedConfig cfg, Logger& logger
) : name(std::move(name)),
    worker_id(worker_id),
    shared_cfg(cfg),
    logger(logger)
{
    std::random_device rd;
    auto time_seed = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    rng.seed(rd() ^ worker_id ^ time_seed);
    reset_for_new_game();
}

void ActionSelectorBase::reset_for_new_game() {
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    use_resignation = dist(rng) < shared_cfg.resignation_probability;
    logger.log("DEBUG",
        "Agent state reset. Resignation allowed: " +
        std::string(use_resignation ? "True" : "False"));
}

ActionSelectorBase::CategorizedChildren
ActionSelectorBase::_categorize(MCTSNode* root) {
    CategorizedChildren out;
    int num_children = root->num_children;

    std::vector<MCTSNode*> all_children;
    all_children.reserve(num_children);
    for (int i = 0; i < num_children; ++i) all_children.push_back(root->first_child + i);

    for (MCTSNode* child : all_children) {
        if (child->has_forced_outcome()) {
            if      (child->forced_outcome == -1) out.winning.push_back(child);
            else if (child->forced_outcome ==  1) out.losing.push_back(child);
            else                                  out.drawing.push_back(child);
        } else {
            // Practical-draw detection: if the opponent's best reply to this
            // move is a forced draw, this move IS a draw in value regardless
            // of its Q. Skip grandchildren that are proven wins for us (the
            // opponent would avoid them anyway).
            bool is_practical_draw = false;
            if (child->is_expanded() && child->num_children > 0) {
                MCTSNode* best_grandchild = nullptr;
                double    best_gq = -2.0;
                for (int i = 0; i < child->num_children; ++i) {
                    MCTSNode* gc = child->first_child + i;
                    if (gc->has_forced_outcome() && gc->forced_outcome == -1) continue;
                    if (gc->visits == 0) continue;
                    double gq = -gc->expected_value(shared_cfg.contempt);
                    if (gq > best_gq) { best_gq = gq; best_grandchild = gc; }
                }
                if (best_grandchild != nullptr &&
                    best_grandchild->has_forced_outcome() &&
                    best_grandchild->forced_outcome == 0) {
                    is_practical_draw = true;
                }
            }

            if (is_practical_draw) {
                out.drawing.push_back(child);
                logger.log("INFO",
                    "Practical draw detected: " + chess::uci::moveToUci(child->move)
                    + " (opponent's best response is a forced draw)");
            } else if (child->visits > 0) {
                out.non_forced_visited.push_back(child);
            }
        }
    }
    return out;
}

// Default: true max Q over visited non-forced children. PUCT uses this.
// Gumbel overrides to preserve its historical top-node formula.
double ActionSelectorBase::_best_q(std::vector<MCTSNode*>& non_forced_visited) {
    if (non_forced_visited.empty()) return -2.0;
    double best = -2.0;
    for (MCTSNode* c : non_forced_visited) {
        double q = -c->expected_value(shared_cfg.contempt);
        if (q > best) best = q;
    }
    return best;
}

SelectionResult ActionSelectorBase::select_move(MCTSNode* root, int ply_count) {
    SelectionResult result;

    if (root->num_children == 0) return result;

    CategorizedChildren cats = _categorize(root);
    double best_q = _best_q(cats.non_forced_visited);

    // ---- Rule A: Win ----
    // Shortest mate. Uniform among ties.
    if (!cats.winning.empty()) {
        int min_dtm = 999999;
        for (MCTSNode* c : cats.winning) if (c->distance_to_mate < min_dtm) min_dtm = c->distance_to_mate;
        std::vector<chess::Move> best_moves;
        for (MCTSNode* c : cats.winning) if (c->distance_to_mate == min_dtm) best_moves.push_back(c->move);
        std::uniform_int_distribution<> dist(0, (int)best_moves.size() - 1);
        result.best_move = best_moves[dist(rng)];
    }
    // ---- Rule B: Draw ----
    // Prefer forced draw when we're not winning (best_q at/below draw_cutoff).
    else if (!cats.drawing.empty() && best_q <= shared_cfg.draw_cutoff) {
        std::uniform_int_distribution<> dist(0, (int)cats.drawing.size() - 1);
        result.best_move = cats.drawing[dist(rng)]->move;
    }
    // ---- Rule C: Resign ----
    // Only when we have a visited non-forced move (otherwise let Rule E play
    // out the forced loss to mate) AND resignation was rolled on for this game.
    else if (use_resignation && !cats.non_forced_visited.empty()
             && best_q < shared_cfg.resignation_cutoff) {
        logger.log("INFO",
            "Best Value (" + std::to_string(best_q) +
            ") is below cutoff. Triggering Resignation.");
        result.resigned = true;
        result.best_move = chess::Move::NO_MOVE;
    }
    // ---- Rule D: Play ----
    // The variant-specific selection.
    else if (!cats.non_forced_visited.empty()) {
        result.best_move = _select_from_visited(cats.non_forced_visited, ply_count);
    }
    // ---- Rule E: Delay Mate ----
    // No visited non-forced options; every child is a proven loss. Pick the
    // longest DTM to give the opponent the most rope. Fallback: uniform
    // over all children (defensive; shouldn't be reachable).
    else {
        if (!cats.losing.empty()) {
            MCTSNode* best_delay = cats.losing[0];
            for (MCTSNode* c : cats.losing) {
                if (c->distance_to_mate > best_delay->distance_to_mate) best_delay = c;
            }
            result.best_move = best_delay->move;
        } else {
            std::uniform_int_distribution<> dist(0, root->num_children - 1);
            result.best_move = (root->first_child + dist(rng))->move;
        }
    }

    logger.log("INFO", "Move selected: " + chess::uci::moveToUci(result.best_move));
    return result;
}