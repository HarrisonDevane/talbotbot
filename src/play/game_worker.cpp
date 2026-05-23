// =============================================================================
// game_worker.cpp
// =============================================================================

#include "game_worker.hpp"

// -----------------------------------------------------------------------------
// SearchAgent::think -- one move of search + selection.
// Identical 3-call sequence to data_generator.cpp's worker_main:
//     reset -> run_simulations -> select_move
// -----------------------------------------------------------------------------
SelectionResult SearchAgent::think(const chess::Board& board,
                                   const std::vector<chess::Board>& history,
                                   int ply) {
    engine.reset(board, history);
    engine.run_simulations(search_budget, gumbel_m);
    return selector.select_move(engine.root, ply);
}

// -----------------------------------------------------------------------------
GameWorker::GameWorker(int worker_id, SearchAgent primary, Logger& logger)
    : worker_id(worker_id), primary(primary), logger(logger) {}

// -----------------------------------------------------------------------------
// run_one_game
//
// Termination ordering matters. We check is_over() at the TOP of the loop so
// that a game which ended because of the OPPONENT's last move (mate/draw the
// session detected after await_opponent_move advanced its board) is caught
// before we try to search a finished position.
//
// Ends we can reach:
//   1. is_over() true            -> board-detected end (mate / draw rules /
//                                   ply limit). Session fills `result`.
//   2. our resignation           -> ActionSelector returns resigned/NO_MOVE.
//                                   We score it and stop.
//   3. opponent non-move outcome -> await_opponent_move returns
//                                   game_continues=false (opponent resigned,
//                                   aborted, disconnected). Session fills it.
// -----------------------------------------------------------------------------
SessionResult GameWorker::run_one_game(GameSession& session) {
    session.on_game_start();

    SessionResult result;
    result.reason = SessionEndReason::NOT_OVER;

    while (true) {
        // (1) Has the game already ended on the board?
        if (session.is_over(result)) break;

        if (session.our_turn()) {
            const chess::Board& pos = session.current_position();
            int ply = session.ply_count();

            SelectionResult choice =
                primary.think(pos, session.history(), ply);

            // (2) ActionSelector resigned on our behalf.
            if (choice.resigned || choice.best_move == chess::Move::NO_MOVE) {
                chess::Color us = pos.sideToMove();
                result.reason      = SessionEndReason::RESIGNATION;
                // We resign => the other side wins.
                result.white_value = (us == chess::Color::WHITE) ? -1.0 : 1.0;
                logger.log("INFO",
                    "[worker " + std::to_string(worker_id) +
                    "] resigned at ply " + std::to_string(ply));
                break;
            }

            logger.log("DEBUG",
                "[worker " + std::to_string(worker_id) + "] our move: " +
                chess::uci::moveToUci(choice.best_move));

            // submit_our_move advances the session's internal position.
            session.submit_our_move(choice.best_move);
        }
        else {
            // Block until the opponent acts. For SelfPlaySession this runs the
            // opponent model's MCTS internally and advances the board.
            MoveOutcome outcome = session.await_opponent_move();

            if (!outcome.game_continues) {
                // (3) Opponent resigned / aborted / disconnected.
                result = outcome.result;
                logger.log("INFO",
                    "[worker " + std::to_string(worker_id) +
                    "] game ended via opponent path; reason=" +
                    std::to_string(static_cast<int>(result.reason)));
                break;
            }
            // Normal opponent move -- session already advanced its board.
        }
    }

    session.on_game_end(result);
    return result;
}