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
//   UCI path is a separate plain loop in main_play.cpp.
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
#include "chess.hpp"
#include "mcts_engine.hpp"
#include "action_selector.hpp"
#include "logger.hpp"
#include "game_session.hpp"

// -----------------------------------------------------------------------------
// SearchAgent: one model's ability to choose a move = (engine, selector, budget).
// References are non-owning; the host keeps the real objects alive.
// -----------------------------------------------------------------------------
struct SearchAgent {
    MCTSEngine&     engine;
    ActionSelector& selector;
    int             search_budget;   // node budget per move (gumbel_search_depth)
    int             gumbel_m;        // sequential-halving m

    // reset -> run_simulations -> select_move. ply is 1-based.
    SelectionResult think(const chess::Board& board,
                          const std::vector<chess::Board>& history,
                          int ply);
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