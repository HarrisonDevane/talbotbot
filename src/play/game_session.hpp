#pragma once

// =============================================================================
// game_session.hpp
//
// The single abstraction seam for the play engine (talbot_play).
//
// A GameSession answers exactly the questions that DIFFER between deployments:
//   - where does the opponent's move come from?
//   - where does our move go?
//   - is the game over, and what was the result?
//   - what is the current position?
//
// Everything else (MCTS search, action selection, budget, ply counting,
// history management) lives in GameWorker and is identical across all
// deployments. That is the entire point of this file: it names the ~3 things
// that vary so the rest can be written once.
//
// Implementations:
//   UciSession      -- opponent move from stdin ("position ... moves ..."),
//                      our move to stdout ("bestmove ...").
//   SelfPlaySession -- no external opponent; the "opponent move" is produced
//                      by running the other model's MCTS search internally.
//   LichessSession  -- opponent move from the Lichess game stream (HTTPS),
//                      our move POSTed to the Lichess API. (Added later.)
//
// DEPENDENCY RULE: this header lives in core/ conceptually -- it must not
// include anything from train/ or play/ implementation files. It depends only
// on the chess types.
// =============================================================================

#include <vector>
#include <optional>
#include "chess.hpp"

// -----------------------------------------------------------------------------
// Result of a finished game, from White's point of view, plus a reason.
// final_value is +1.0 White win, -1.0 Black win, 0.0 draw -- matching the
// convention already used in data_generator.cpp (final_game_value).
// -----------------------------------------------------------------------------
enum class SessionEndReason {
    NOT_OVER,        // game still in progress
    CHECKMATE,       // normal mate on the board
    DRAW_RULES,      // stalemate / 50-move / repetition / insufficient material
    RESIGNATION,     // a side resigned (ActionSelector resignation path)
    PLY_LIMIT,       // hit the max_ply_length hard cutoff -> scored as draw
    OPPONENT_LEFT,   // remote opponent aborted/disconnected (Lichess)
    ABORTED          // session torn down externally (quit, error, timeout)
};

struct SessionResult {
    SessionEndReason reason = SessionEndReason::NOT_OVER;
    double white_value = 0.0;  // +1 / 0 / -1 from White's perspective
};

// -----------------------------------------------------------------------------
// MoveOutcome: what await_opponent_move() returns.
//
// A remote opponent does not only ever "make a move" -- it can resign, abort,
// or the stream can end. await_opponent_move() must be able to report that
// without overloading chess::Move::NO_MOVE to mean five different things.
// -----------------------------------------------------------------------------
struct MoveOutcome {
    bool        game_continues = true;     // false => game is over, see result
    chess::Move move = chess::Move::NO_MOVE;
    SessionResult result;                  // populated only when !game_continues
};

// -----------------------------------------------------------------------------
// The interface.
//
// Threading contract: a GameSession is owned and driven by exactly ONE
// GameWorker thread. It is not required to be internally thread-safe.
// Blocking is allowed and expected: await_opponent_move() may block on stdin
// or a network socket; submit_our_move() may block on a network POST.
// -----------------------------------------------------------------------------
class GameSession {
public:
    virtual ~GameSession() = default;

    // The position the worker should think about / play from.
    // Must reflect every move applied so far (ours and the opponent's).
    virtual const chess::Board& current_position() const = 0;

    // History of prior positions, most-recent-first, capped at the 4 plies
    // board_to_tensor_69 consumes. Returned so GameWorker can hand it straight
    // to MCTSEngine::reset(board, history) without each session re-deriving it.
    virtual const std::vector<chess::Board>& history() const = 0;

    // True when it is OUR turn to search and submit a move.
    // (UciSession knows this from the position command; SelfPlaySession from
    // the color assigned to this worker; LichessSession from the game stream.)
    virtual bool our_turn() const = 0;

    // Whose colour we are playing in this game. Needed by SelfPlaySession to
    // decide which model (batcher A or B) searches, and for result scoring.
    virtual chess::Color our_color() const = 0;

    // The 1-based ply counter, for ActionSelector temperature logic
    // (temperature_ply_cutoff). Matches ply_count in data_generator.cpp.
    virtual int ply_count() const = 0;

    // ---- the three things that actually differ ----

    // Push the move we chose to wherever it belongs (stdout / network /
    // internal board) AND advance the session's internal position by it.
    // After this returns, current_position() reflects the new move.
    virtual void submit_our_move(chess::Move move) = 0;

    // Block until the opponent has moved (or resigned / aborted / disconnected).
    // On a normal move: advances the session's internal position and returns
    // game_continues = true. Otherwise returns game_continues = false with a
    // populated SessionResult.
    virtual MoveOutcome await_opponent_move() = 0;

    // Has the game ended? Checked after every applied move. When true, *result
    // is filled. This covers ends WE detect on the board (mate/draw after our
    // move); ends the opponent causes come back through await_opponent_move().
    virtual bool is_over(SessionResult& result) const = 0;

    // ---- lifecycle ----

    // Optional: called once before the first move. Lets a session do I/O it
    // would rather not do in its constructor (open a stream, send a handshake).
    virtual void on_game_start() {}

    // Optional: called once after the game ends, for cleanup / final logging /
    // sending a result acknowledgement.
    virtual void on_game_end(const SessionResult& result) {}
};