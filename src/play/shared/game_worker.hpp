#pragma once

// =============================================================================
// game_worker.hpp
//
// GameWorker drives ONE real, alternating-turn game to completion.
//
// SCOPE (post Option-2 decision):
//   GameWorker serves deployments that ARE genuine continuous games with a
//   persistent position and real termination -- i.e. SelfPlaySession now, and
//   LichessSession later. It deliberately does NOT serve UCI: UCI is a
//   stateless query/response oracle, not an alternating game, and forcing it
//   through this loop bent three interface methods into vestigial stubs. The
//   UCI path is a separate plain loop in main_uci.cpp.
//
//   So GameSession has exactly two real shapes, and that is what justifies the
//   abstraction: two genuinely-similar deployments, not three forced ones.
//
// THE LOOP:
//   while not over:
//     our turn  -> search with our model, submit (session advances its board)
//     their turn-> block on the session for the opponent's move
//   Termination is reported either by is_over() (ends we cause on the board)
//   or by await_opponent_move() returning game_continues=false (opponent
//   resigns / aborts / disconnects).
//
// SEARCH:
//   Fixed node budget only. Each SearchAgent carries its own budget and its
//   own root strategy (deficit = the UCI inference search, halving = the
//   training sequential-halving search), so the two sides of a game can differ.
//   Tree reuse is per side (mcts.tree_reuse). When on, a side's next search
//   starts from the subtree of its previous search that matches the moves
//   played since (its own move + the opponent's reply), exactly like UCI.
//   Halving (training SH) sides never reuse: reset_reuse is deficit-only.
//
// WHAT GameWorker DOES NOT DO:
//   * It does not know which model the opponent uses. For self-play, the
//     SelfPlaySession internally owns the opponent's SearchAgent and runs it
//     inside await_opponent_move(). GameWorker only ever touches `primary`,
//     the agent for OUR colour. This keeps the two-engine A/B routing entirely
//     inside SelfPlaySession, where it can be tested in isolation.
//   * It does not own engine/selector lifetime -- those are referenced.
//
// THREADING: one GameWorker per thread. The tournament host spawns K of them.
// =============================================================================

#include <vector>
#include <string>
#include <ostream>
#include "chess.hpp"
#include "mcts_engine.hpp"
#include "action_selector.hpp"
#include "logger.hpp"
#include "game_session.hpp"

// -----------------------------------------------------------------------------
// SearchAgent: one model's ability to choose a move =
//   (engine, selector, node budget, root strategy).
// References are non-owning; the host keeps the real objects alive.
// -----------------------------------------------------------------------------
struct SearchAgent {
    // Prior positions fed to the network (current + this many). Must match
    // data_generator / main_uci. Shared with SelfPlaySession.
    static constexpr size_t kHistoryPlies = 7;
    // Longest move gap we will walk for reuse (normally 2: our move + reply).
    // Same cap as main_uci.
    static constexpr size_t kMaxReuseWalk = 4;

    MCTSEngine&     engine;
    ActionSelector& selector;
    int             search_budget;   // simulations per move
    int             gumbel_m;        // sequential-halving m (ignored by deficit search)
    bool            use_halving;     // true: run_simulations_fixed (training SH)
                                     // false: run_simulations_fixed_inference (UCI search)
    bool            tree_reuse;      // mcts.tree_reuse (ignored when use_halving)
    bool            use_puct = false;// label only: the engine itself was set to PUCT
                                     // selection by the host (MCTSEngine::use_puct)

    // Search-stats logging (tournament only). When stats_out is set, every
    // search appends one JSON line: identification, the move played, PV
    // length, and every root move's prior / visits / Q. Statistics are
    // computed offline from these lines.
    std::ostream* stats_out = nullptr;
    std::string   player    = {};
    int           game      = -1;

    // Search-type name for logs/JSON ("deficit", "halving", "puct", "hybrid",
    // "policy"). Set by the host from search_type_name(mcts.search).
    std::string   search_label = "deficit";

    // Reuse state: where this agent last searched. Cleared by new_game().
    bool                      has_tree    = false;
    size_t                    searched_at = 0;   // moves.size() at last search
    chess::Board              last_board = chess::Board();   // explicit init: Board ctor is explicit, aggregate init would warn
    std::vector<chess::Board> last_history = {};

    bool reuse_active() const { return tree_reuse && !use_halving; }
    void new_game() { has_tree = false; searched_at = 0; }

    // (reuse walk | reset) -> run_simulations -> select_move. ply is 1-based.
    // `moves` = every move played this game (GameSession::moves_played()).
    SelectionResult think(const chess::Board& board,
                          const std::vector<chess::Board>& history,
                          int ply,
                          const std::vector<chess::Move>& moves);

private:
    void write_stats(const chess::Board& board, int ply, const SelectionResult& r,
                     double search_ms) const;

    // Walk the previous tree down through moves[searched_at..]. Returns false
    // (caller must reset) if there is no tree, the gap is too long, or any
    // played move is missing from the old tree.
    bool try_reuse(const chess::Board& board,
                   const std::vector<chess::Board>& history,
                   const std::vector<chess::Move>& moves,
                   const PoolTargets& pt);
};

class GameWorker {
public:
    GameWorker(int worker_id, SearchAgent primary, Logger& logger);

    // Drive `session` from its current position to termination; return result.
    SessionResult run_one_game(GameSession& session);

private:
    int        worker_id;
    SearchAgent primary;             // the model playing OUR colour
    Logger&    logger;
};