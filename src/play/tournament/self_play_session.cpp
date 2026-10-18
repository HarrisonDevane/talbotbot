// =============================================================================
// self_play_session.cpp
// =============================================================================

#include "self_play_session.hpp"
#include <sstream>
#include <algorithm>

// -----------------------------------------------------------------------------
SelfPlaySession::SelfPlaySession(SearchAgent white_agent,
                                 SearchAgent black_agent,
                                 chess::Color our_side,
                                 const Opening& opening,
                                 int max_ply,
                                 Logger& logger,
                                 std::optional<PgnFileSink>        pgn_sink,
                                 std::optional<SessionPgnMetadata> pgn_meta_in)
    : agent_white(white_agent),
      agent_black(black_agent),
      our_side(our_side),
      opening(opening),
      max_ply(max_ply),
      logger(logger),
      pgn_sink(pgn_sink),
      pgn_meta(pgn_meta_in.value_or(SessionPgnMetadata{})) {
    board.setFen(chess::constants::STARTPOS);
}

// -----------------------------------------------------------------------------
void SelfPlaySession::advance(chess::Move move) {
    hist.insert(hist.begin(), board);
    if (hist.size() > kHistoryPlies) hist.pop_back();
    board.makeMove(move);
    game_moves.push_back(move);
    ply++;
}

// -----------------------------------------------------------------------------
void SelfPlaySession::on_game_start() {
    board.setFen(chess::constants::STARTPOS);
    hist.clear();
    game_moves.clear();
    opening_move_count = 0;
    ply = 1;
    finished = false;
    setup_failed = false;
    agent_white.new_game();
    agent_black.new_game();

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
    opening_move_count = static_cast<int>(game_moves.size());

    if (setup_failed) {
        finished = true;
        final_result.reason      = SessionEndReason::ABORTED;
        final_result.white_value = 0.0;
        return;
    }

    SessionResult r;
    if (detect_board_termination(r)) {
        finished = true;
        final_result = r;
    }

    std::string opening_line;
    for (int i = 0; i < opening_move_count; ++i) {
        if (i) opening_line += " ";
        opening_line += chess::uci::moveToUci(game_moves[i]);
    }
    logger.log("INFO",
        "Game start: opening ECO " + opening.eco + " | " +
        std::to_string(opening_move_count) + " opening plies | moves: " +
        opening_line +
        " | budgets W=" + std::to_string(agent_white.search_budget) +
        " (" + agent_white.search_label +
        (agent_white.reuse_active() ? "+reuse" : "") + ")" +
        " B=" + std::to_string(agent_black.search_budget) +
        " (" + agent_black.search_label +
        (agent_black.reuse_active() ? "+reuse" : "") + ")");
}

// -----------------------------------------------------------------------------
bool SelfPlaySession::detect_board_termination(SessionResult& result) const {
    auto over = board.isGameOver();

    if (over.second != chess::GameResult::NONE) {
        if (over.second == chess::GameResult::LOSE) {
            chess::Color loser = board.sideToMove();
            result.reason      = SessionEndReason::CHECKMATE;
            result.white_value = (loser == chess::Color::WHITE) ? -1.0 : 1.0;
        } else {
            result.reason      = SessionEndReason::DRAW_RULES;
            result.white_value = 0.0;
        }
        return true;
    }

    if (ply > max_ply) {
        result.reason      = SessionEndReason::PLY_LIMIT;
        result.white_value = 0.0;
        return true;
    }

    return false;
}

// -----------------------------------------------------------------------------
bool SelfPlaySession::is_over(SessionResult& result) const {
    if (finished) {
        result = final_result;
        return true;
    }
    return detect_board_termination(result);
}

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
MoveOutcome SelfPlaySession::await_opponent_move() {
    MoveOutcome outcome;

    chess::Color mover = board.sideToMove();
    SearchAgent& opp = (mover == chess::Color::WHITE) ? agent_white : agent_black;

    SelectionResult choice = opp.think(board, hist, ply, game_moves);

    if (choice.resigned || choice.best_move == chess::Move::NO_MOVE) {
        outcome.game_continues     = false;
        outcome.result.reason      = SessionEndReason::RESIGNATION;
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
// on_game_end: log the summary line, then build the PGN via pgn_writer and
// ship it to both sinks. The session fills in the game-derived fields
// (result_str, ECO, opening_plies, termination); the host pre-loaded the rest
// (event, site, player names, round, config).
// -----------------------------------------------------------------------------
void SelfPlaySession::on_game_end(const SessionResult& result) {
    const char* reason = "?";
    switch (result.reason) {
        case SessionEndReason::CHECKMATE:   reason = "checkmate";   break;
        case SessionEndReason::DRAW_RULES:  reason = "draw";        break;
        case SessionEndReason::RESIGNATION: reason = "resignation"; break;
        case SessionEndReason::PLY_LIMIT:   reason = "ply limit";   break;
        case SessionEndReason::TIME_LOSS:   reason = "time loss";   break;
        case SessionEndReason::ABORTED:     reason = "aborted";     break;
        default:                            reason = "not over";    break;
    }
    logger.log("INFO",
        "Game end: ECO " + opening.eco + " | reason=" + reason +
        " | white_value=" + std::to_string(result.white_value) +
        " | plies=" + std::to_string(ply - 1));

    // Determine result string from reason + white_value.
    std::string result_str;
    switch (result.reason) {
        case SessionEndReason::CHECKMATE:
        case SessionEndReason::RESIGNATION:
        case SessionEndReason::TIME_LOSS:
            result_str = (result.white_value > 0.0) ? "1-0" : "0-1";
            break;
        case SessionEndReason::DRAW_RULES:
        case SessionEndReason::PLY_LIMIT:
            result_str = "1/2-1/2";
            break;
        default:
            result_str = "*";
            break;
    }

    // Assemble PgnHeader: host-provided metadata + session-derived fields.
    PgnHeader hdr;
    hdr.event         = pgn_meta.event;
    hdr.site          = pgn_meta.site;
    hdr.round         = pgn_meta.round;
    hdr.white         = pgn_meta.white_name;
    hdr.black         = pgn_meta.black_name;
    hdr.eco           = opening.eco;
    hdr.opening_plies = opening_move_count;

    if (result.reason == SessionEndReason::TIME_LOSS) {
        hdr.termination = "Time forfeit";
    } else if (result.reason == SessionEndReason::CHECKMATE ||
               result.reason == SessionEndReason::DRAW_RULES ||
               result.reason == SessionEndReason::RESIGNATION) {
        hdr.termination = "Normal";
    } else if (result.reason == SessionEndReason::PLY_LIMIT) {
        hdr.termination = "Adjudication";
    }

    // No per-move annotations yet -- populating those requires extending
    // SelectionResult with eval/depth/elapsed and having the session capture
    // them per move. When that lands, populate a std::vector<PgnMoveAnnotation>
    // parallel to game_moves and pass it here instead of {}.
    const std::string pgn = build_pgn(hdr, game_moves, {}, result_str, pgn_meta.config);

    logger.log("CRITICAL", "Game PGN:\n" + pgn);
    if (pgn_sink && pgn_sink->out && pgn_sink->mutex) {
        std::lock_guard<std::mutex> lock(*pgn_sink->mutex);
        (*pgn_sink->out) << pgn;
        pgn_sink->out->flush();
    }
}