#pragma once

// =============================================================================
// self_play_session.hpp
//
// GameSession implementation for tournament play: ONE game between two
// players A and B (different models, or the same model with different search
// parameters), started from a fixed opening line.
//
// A/B ROUTING
// -----------
// Owns TWO SearchAgents (agent_white / agent_black), each already bound to its
// batcher by the host. GameWorker drives its `primary` side; the opponent side
// runs internally inside await_opponent_move via the other agent's think().
//
// SEARCH BUDGET
// -------------
// Fixed node budget only; each agent carries its own budget and root strategy.
// There are no clocks.
//
// PGN OUTPUT
// ----------
// PGN emission delegated to pgn_writer.hpp (shared with data_generator).
// SessionPgnMetadata holds the fields the host knows (player names, event,
// round, PgnConfig) that the session doesn't; the session fills in the rest
// (ECO, opening plies, termination) from game state at end.
//
// Two sinks (independent):
//   (1) logger at CRITICAL -- per-worker log file (diagnostics)
//   (2) optional PgnFileSink -- shared games.pgn across all workers
// =============================================================================

#include <vector>
#include <string>
#include <optional>
#include <fstream>
#include <mutex>
#include "chess.hpp"
#include "logger.hpp"
#include "game_session.hpp"
#include "game_worker.hpp"     // for SearchAgent
#include "opening_book.hpp"    // for Opening
#include "pgn_writer.hpp"      // for PgnConfig / PgnHeader (metadata + build_pgn)

// -----------------------------------------------------------------------------
// PgnFileSink: shared PGN output file + guard mutex. Both pointers must
// outlive the session.
// -----------------------------------------------------------------------------
struct PgnFileSink {
    std::ofstream* out;
    std::mutex*    mutex;
};

// -----------------------------------------------------------------------------
// SessionPgnMetadata: the PGN fields the HOST knows about (player names,
// event, round, format config). The session fills in the rest (ECO from the
// opening, termination from result) at end.
//
// The host constructs one of these per game before building the session:
//   SessionPgnMetadata meta;
//   meta.event      = "Talbot Tournament";
//   meta.white_name = (a_color == WHITE) ? name_a : name_b;
//   meta.black_name = (a_color == WHITE) ? name_b : name_a;
//   meta.round      = std::to_string(game_number + 1);
//   meta.config     = PgnConfig::annotated();
// -----------------------------------------------------------------------------
struct SessionPgnMetadata {
    std::string event      = "Talbot Tournament";
    std::string site       = "Talbot C++ Engine";
    std::string white_name = "?";
    std::string black_name = "?";
    std::string round;                                // empty -> "?" in output
    PgnConfig   config     = PgnConfig::annotated();  // tournament default
};

class SelfPlaySession : public GameSession {
public:
    SelfPlaySession(SearchAgent white_agent,
                    SearchAgent black_agent,
                    chess::Color our_side,
                    const Opening& opening,
                    int max_ply,
                    Logger& logger,
                    std::optional<PgnFileSink>        pgn_sink = std::nullopt,
                    std::optional<SessionPgnMetadata> pgn_meta = std::nullopt);

    // ---- GameSession interface ----
    const chess::Board& current_position() const override { return board; }
    const std::vector<chess::Board>& history() const override { return hist; }
    const std::vector<chess::Move>& moves_played() const override { return game_moves; }
    bool our_turn() const override { return board.sideToMove() == our_side; }
    chess::Color our_color() const override { return our_side; }
    int  ply_count() const override { return ply; }

    void submit_our_move(chess::Move move) override;
    MoveOutcome await_opponent_move() override;
    bool is_over(SessionResult& result) const override;

    void on_game_start() override;
    void on_game_end(const SessionResult& result) override;

    // ---- accessors ----
    const std::string& opening_eco() const { return opening.eco; }
    int total_plies() const { return ply - 1; }

private:
    // Prior positions kept for the network input (current + this many).
    // Must match data_generator / main_uci, which both cap at 7.
    static constexpr size_t kHistoryPlies = SearchAgent::kHistoryPlies;

    void advance(chess::Move move);
    bool detect_board_termination(SessionResult& result) const;

    SearchAgent agent_white;
    SearchAgent agent_black;
    chess::Color our_side;
    Opening opening;
    int max_ply;
    Logger& logger;

    std::optional<PgnFileSink>     pgn_sink;
    SessionPgnMetadata             pgn_meta;   // always present (defaults if not supplied)

    chess::Board board;
    std::vector<chess::Board> hist;

    std::vector<chess::Move> game_moves;
    int opening_move_count = 0;

    int  ply = 1;
    bool setup_failed = false;
    bool finished = false;
    SessionResult final_result;
};