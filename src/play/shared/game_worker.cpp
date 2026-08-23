// =============================================================================
// game_worker.cpp
// =============================================================================

#include "game_worker.hpp"
#include <chrono>

// -----------------------------------------------------------------------------
// SearchAgent::think -- one move of search + selection.
//
// Dispatches on the presence of a TimeBudget:
//   nullopt  -> run_simulations_fixed(search_budget, gumbel_m). Pool sized to
//               that exact sim count (grow-only in the engine's reset()).
//   present  -> run_simulations_timed(gumbel_m, soft_dl, hard_dl). Pool sized
//               to (estimated_nps * hard_limit) via predict_pool_needs_for_time.
//
// The pool_sizing_cfg must have been set on the engine at construction time
// (host is responsible). Both branches call reset() with pool targets so the
// pool grows on demand as the search deepens.
// -----------------------------------------------------------------------------
SelectionResult SearchAgent::think(const chess::Board& board,
                                   const std::vector<chess::Board>& history,
                                   int ply,
                                   std::optional<TimeBudget> budget) {
    if (budget.has_value()) {
        const auto now = std::chrono::steady_clock::now();
        const auto soft_dl = now + std::chrono::milliseconds(budget->target_ms);
        const auto hard_dl = now + std::chrono::milliseconds(budget->hard_limit_ms);

        // Sizing: total time we might spend is up to hard_limit_ms. Convert to
        // seconds and let the engine multiply by its EWMA'd NPS. safety_mult=1
        // because hard_limit_ms already incorporates hard_multiplier upstream.
        const double hard_s = static_cast<double>(budget->hard_limit_ms) / 1000.0;
        PoolTargets pt = engine.predict_pool_needs_for_time(hard_s, 1.0);
        engine.reset(board, history, pt.node_target, pt.edge_target);
        engine.run_simulations_timed(gumbel_m, soft_dl, hard_dl);
    } else {
        PoolTargets pt = engine.predict_pool_needs(search_budget);
        engine.reset(board, history, pt.node_target, pt.edge_target);
        engine.run_simulations_fixed(search_budget, gumbel_m);
    }
    return selector.select_move(engine.root, ply, &engine);
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
//                                   aborted, disconnected, timed out). Session
//                                   fills it.
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

            // Ask session for a time budget. nullopt means fixed-depth mode.
            auto budget = session.our_time_budget();

            SelectionResult choice =
                primary.think(pos, session.history(), ply, budget);

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

            // submit_our_move advances the session's internal position AND
            // (for timed sessions) deducts elapsed time from our clock.
            session.submit_our_move(choice.best_move);

            // Post-submit end-check: if the session finished the game inside
            // submit_our_move (mate, draw, OR time-loss detected on clock
            // deduction), pick that up now rather than trying to continue.
            if (session.is_over(result)) break;
        }
        else {
            // Block until the opponent acts. For SelfPlaySession this runs the
            // opponent model's MCTS internally and advances the board.
            MoveOutcome outcome = session.await_opponent_move();

            if (!outcome.game_continues) {
                // (3) Opponent resigned / aborted / disconnected / flagged.
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