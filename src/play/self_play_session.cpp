// =============================================================================
// self_play_session.cpp
// =============================================================================

#include "self_play_session.hpp"

// -----------------------------------------------------------------------------
SelfPlaySession::SelfPlaySession(SearchAgent white_agent,
                                 SearchAgent black_agent,
                                 chess::Color our_side,
                                 const Opening& opening,
                                 int max_ply,
                                 Logger& logger)
    : agent_white(white_agent),
      agent_black(black_agent),
      our_side(our_side),
      opening(opening),
      max_ply(max_ply),
      logger(logger) {
    board.setFen(chess::constants::STARTPOS);
}

// -----------------------------------------------------------------------------
// advance: apply a move, maintain history (most-recent-first, capped 4) and the
// ply counter. Single choke point so our-move and opponent-move stay identical.
// -----------------------------------------------------------------------------
void SelfPlaySession::advance(chess::Move move) {
    hist.insert(hist.begin(), board);
    if (hist.size() > 4) hist.pop_back();
    board.makeMove(move);
    ply++;
}

// -----------------------------------------------------------------------------
// on_game_start: build the starting position by replaying the opening's SAN
// moves. parseSan throws on malformed/ambiguous SAN -- if that happens the
// opening is unusable, so we latch setup_failed and the game ends ABORTED.
// -----------------------------------------------------------------------------
void SelfPlaySession::on_game_start() {
    board.setFen(chess::constants::STARTPOS);
    hist.clear();
    ply = 1;
    finished = false;
    setup_failed = false;

    for (const std::string& san : opening.san_moves) {
        chess::Move mv;
        try {
            mv = chess::uci::parseSan(board, san);
        } catch (const std::exception& e) {
            logger.log("ERROR",
                "Opening replay failed (ECO " + opening.eco + ") on SAN '" +
                san + "': " + e.what());
            setup_failed = true;
            break;
        }
        if (mv == chess::Move::NO_MOVE) {
            logger.log("ERROR",
                "Opening replay produced NO_MOVE (ECO " + opening.eco +
                ") on SAN '" + san + "'");
            setup_failed = true;
            break;
        }
        advance(mv);
    }

    if (setup_failed) {
        finished = true;
        final_result.reason      = SessionEndReason::ABORTED;
        final_result.white_value = 0.0;
        return;
    }

    // The opening itself may already be a finished position (rare, but a long
    // forced line could mate or hit a draw rule). Check before play begins.
    SessionResult r;
    if (detect_board_termination(r)) {
        finished = true;
        final_result = r;
    }

    logger.log("INFO",
        "Game start: opening ECO " + opening.eco + ", " +
        std::to_string(opening.san_moves.size()) + " opening plies, " +
        "we play " + (our_side == chess::Color::WHITE ? "White" : "Black"));
}

// -----------------------------------------------------------------------------
// detect_board_termination: classify the CURRENT board.
//   * isGameOver() -> mate or a draw rule (stalemate / 50-move / insufficient /
//     repetition, depending on chess.hpp's implementation).
//   * ply limit    -> scored as a draw, matching data_generator.cpp.
// white_value is always from White's perspective.
// -----------------------------------------------------------------------------
bool SelfPlaySession::detect_board_termination(SessionResult& result) const {
    auto over = board.isGameOver();   // {GameResultReason, GameResult}

    if (over.second != chess::GameResult::NONE) {
        if (over.second == chess::GameResult::LOSE) {
            // The side to move has been mated -> the OTHER side won.
            chess::Color loser = board.sideToMove();
            result.reason      = SessionEndReason::CHECKMATE;
            result.white_value = (loser == chess::Color::WHITE) ? -1.0 : 1.0;
        } else {
            // DRAW (stalemate, 50-move, repetition, insufficient material).
            result.reason      = SessionEndReason::DRAW_RULES;
            result.white_value = 0.0;
        }
        return true;
    }

    // Hard ply cap -> forced draw.
    if (ply > max_ply) {
        result.reason      = SessionEndReason::PLY_LIMIT;
        result.white_value = 0.0;
        return true;
    }

    return false;
}

// -----------------------------------------------------------------------------
// is_over: the game is over if (a) it has been latched finished, or (b) the
// current board is terminal. GameWorker calls this at the top of its loop.
// -----------------------------------------------------------------------------
bool SelfPlaySession::is_over(SessionResult& result) const {
    if (finished) {
        result = final_result;
        return true;
    }
    return detect_board_termination(result);
}

// -----------------------------------------------------------------------------
// submit_our_move: GameWorker chose a move with its `primary` agent. Apply it,
// then check whether OUR move just ended the game (mate / draw) so the latched
// result is ready for the next is_over() call.
// -----------------------------------------------------------------------------
void SelfPlaySession::submit_our_move(chess::Move move) {
    advance(move);

    SessionResult r;
    if (detect_board_termination(r)) {
        finished = true;
        final_result = r;
    }
}

// -----------------------------------------------------------------------------
// await_opponent_move -- THE A/B ROUTING.
//
// It is the opponent's turn. The opponent is whichever agent is assigned the
// side NOT equal to our_side. We run THAT agent's MCTS search here, internally.
// GameWorker never touches the opponent agent -- this is the only place the
// second engine is used.
//
// After the opponent's move we must detect termination ourselves: GameWorker
// checks is_over() only at loop top, so a game ended by the opponent's move
// has to come back as game_continues = false here.
//
// Opponent resignation: if the opponent agent's ActionSelector resigns, the
// opponent loses. We surface that as a finished game, not a normal move.
// -----------------------------------------------------------------------------
MoveOutcome SelfPlaySession::await_opponent_move() {
    MoveOutcome outcome;

    // Pick the agent for the side to move (always the opponent here, since
    // GameWorker only calls this when !our_turn()).
    chess::Color mover = board.sideToMove();
    SearchAgent& opp = (mover == chess::Color::WHITE) ? agent_white : agent_black;

    SelectionResult choice = opp.think(board, hist, ply);

    // Opponent resigned -> opponent (the side to move) loses.
    if (choice.resigned || choice.best_move == chess::Move::NO_MOVE) {
        outcome.game_continues   = false;
        outcome.result.reason    = SessionEndReason::RESIGNATION;
        outcome.result.white_value = (mover == chess::Color::WHITE) ? -1.0 : 1.0;
        finished = true;
        final_result = outcome.result;
        logger.log("INFO",
            "Opponent (" + std::string(mover == chess::Color::WHITE ? "White" : "Black") +
            ") resigned at ply " + std::to_string(ply));
        return outcome;
    }

    advance(choice.best_move);
    logger.log("DEBUG",
        "Opponent move: " + chess::uci::moveToUci(choice.best_move));

    // Did the opponent's move end the game?
    SessionResult r;
    if (detect_board_termination(r)) {
        outcome.game_continues = false;
        outcome.result = r;
        finished = true;
        final_result = r;
        return outcome;
    }

    outcome.game_continues = true;
    outcome.move = choice.best_move;
    return outcome;
}

// -----------------------------------------------------------------------------
void SelfPlaySession::on_game_end(const SessionResult& result) {
    const char* reason = "?";
    switch (result.reason) {
        case SessionEndReason::CHECKMATE:   reason = "checkmate";   break;
        case SessionEndReason::DRAW_RULES:  reason = "draw";        break;
        case SessionEndReason::RESIGNATION: reason = "resignation"; break;
        case SessionEndReason::PLY_LIMIT:   reason = "ply limit";   break;
        case SessionEndReason::ABORTED:     reason = "aborted";     break;
        default:                            reason = "not over";    break;
    }
    logger.log("INFO",
        "Game end: ECO " + opening.eco + " | reason=" + reason +
        " | white_value=" + std::to_string(result.white_value) +
        " | plies=" + std::to_string(ply - 1));
}