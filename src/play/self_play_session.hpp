#pragma once

// =============================================================================
// self_play_session.hpp
//
// GameSession implementation for tournament play: ONE game between two models,
// model A and model B, started from a fixed opening line.
//
// THE A/B ROUTING -- THE ONE GENUINELY NEW MECHANIC
// -------------------------------------------------
// A SelfPlaySession owns TWO SearchAgents:
//     agent_white -- the model assigned the White pieces this game
//     agent_black -- the model assigned the Black pieces this game
// Each agent's MCTSEngine is bound (at construction, by the host) to a
// different InferenceBatcher -- so White's searches hit batcher A's GPU model
// and Black's hit batcher B's. The session does not know or care which batcher
// is which; it only knows "white agent" and "black agent".
//
// GameWorker only ever drives ONE side via its `primary` agent (the side the
// worker is nominally "playing"). The OTHER side is driven entirely inside
// this session's await_opponent_move(), which calls the opponent agent's
// think() directly. That is why all two-engine logic lives here and nowhere
// else: it is contained, and testable in isolation.
//
// IMPORTANT: because await_opponent_move() runs a full MCTS search internally,
// it is NOT cheap and NOT non-blocking -- a single call may take as long as a
// normal move. That is expected and fine; GameWorker is built to block here.
//
// OPENING SETUP
// -------------
// The session is constructed with an Opening (SAN token list from OpeningBook).
// At on_game_start() it replays those SAN moves through chess::uci::parseSan
// onto a fresh board. parseSan THROWS on bad/ambiguous SAN; a bad opening makes
// the whole game unplayable, so setup failure is surfaced as a finished game
// with reason = ABORTED rather than silently swallowed -- the host then sees
// the opening was bad in the results.
//
// COLOUR / RESULT CONVENTION
// --------------------------
// our_color() is whatever colour GameWorker's `primary` agent plays. The host
// sets this when it builds the GameSpec. white_value in SessionResult is
// always from White's perspective (+1 White win / 0 draw / -1 Black win),
// matching data_generator.cpp's final_game_value.
// =============================================================================

#include <vector>
#include <string>
#include "chess.hpp"
#include "logger.hpp"
#include "game_session.hpp"
#include "game_worker.hpp"     // for SearchAgent
#include "opening_book.hpp"    // for Opening

class SelfPlaySession : public GameSession {
public:
    // white_agent / black_agent : the two models, already bound to their
    //                             respective batchers by the host.
    // our_side                  : the colour GameWorker's `primary` agent is
    //                             playing -- determines our_turn()/our_color().
    // opening                   : the SAN line to start from.
    // max_ply                   : hard draw cutoff (config.max_ply_length).
    SelfPlaySession(SearchAgent white_agent,
                    SearchAgent black_agent,
                    chess::Color our_side,
                    const Opening& opening,
                    int max_ply,
                    Logger& logger);

    // ---- GameSession interface ----
    const chess::Board& current_position() const override { return board; }
    const std::vector<chess::Board>& history() const override { return hist; }
    bool our_turn() const override { return board.sideToMove() == our_side; }
    chess::Color our_color() const override { return our_side; }
    int  ply_count() const override { return ply; }

    void submit_our_move(chess::Move move) override;
    MoveOutcome await_opponent_move() override;
    bool is_over(SessionResult& result) const override;

    void on_game_start() override;
    void on_game_end(const SessionResult& result) override;

    // ---- tournament-specific accessors (read by the host after the game) ----
    const std::string& opening_eco() const { return opening.eco; }
    int total_plies() const { return ply - 1; }

private:
    // Apply a move to the internal board, advance history + ply counter.
    // Shared by submit_our_move and the opponent path.
    void advance(chess::Move move);

    // Evaluate the current board for a finished game. Fills `result` and
    // returns true if the game is over (mate / draw rules / ply limit).
    bool detect_board_termination(SessionResult& result) const;

    // Emit the full game as a PGN at CRITICAL level (mirrors data_generator).
    void log_pgn(const SessionResult& result);

    SearchAgent agent_white;
    SearchAgent agent_black;
    chess::Color our_side;
    Opening opening;
    int max_ply;
    Logger& logger;

    chess::Board board;
    std::vector<chess::Board> hist;   // most-recent-first, capped at 4

    // Every move applied this game, in order -- opening moves first, then
    // played moves. Used to emit a PGN at game end (mirrors data_generator).
    std::vector<chess::Move> game_moves;
    int opening_move_count = 0;       // how many leading entries are the opening

    int  ply = 1;                     // 1-based, matches data_generator.cpp
    bool setup_failed = false;        // true if opening replay threw
    bool finished = false;            // latched once the game has ended
    SessionResult final_result;       // cached result once finished
};