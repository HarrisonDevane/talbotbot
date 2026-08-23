#pragma once

// =============================================================================
// pgn_writer.hpp
//
// Shared PGN construction module. Used by:
//   - main_tournament.cpp    -> annotated PGNs for BayesElo / analysis
//   - data_generator.cpp     -> minimal PGNs for self-play training data
//   - (future) LichessSession, external UCI drivers, etc.
//
// This module knows only about PGN format. It has no dependencies on
// MCTSEngine, SelfPlaySession, InferenceBatcher -- it takes plain data in and
// returns a string. Trivially unit-testable in isolation.
//
// FORMAT
// ------
// Output matches cutechess-cli conventions so PGNs from any of the above
// sources are interchangeable with cutechess PGNs and interoperable with
// BayesElo, Ordo, chess GUIs, and analysis tools.
//
// USAGE
// -----
//   PgnHeader hdr;
//   hdr.event = "Talbot Self-Play";
//   hdr.white = "checkpoint_50000";
//   hdr.black = "checkpoint_40000";
//   hdr.eco   = "B90";
//   hdr.time_control = "60+1";
//
//   std::vector<PgnMoveAnnotation> anns;
//   for (each move played) {
//       PgnMoveAnnotation a;
//       a.cp         = q_to_cp(root_q);
//       a.depth      = max_pv_depth;
//       a.elapsed_ms = move_wall_ms;
//       anns.push_back(a);
//   }
//
//   std::string pgn = build_pgn(hdr, moves, anns, "1-0", PgnConfig::annotated());
// =============================================================================

#include <string>
#include <vector>
#include <cstdint>
#include "chess.hpp"

// -----------------------------------------------------------------------------
// PgnHeader
//
// All game-level metadata. Fields left empty are omitted from output (except
// Event/Site/White/Black/Result which are always emitted -- PGN requires them
// per the Seven Tag Roster).
//
// starting_fen is empty for standard startpos. If set to any non-startpos FEN,
// build_pgn emits [SetUp "1"] and [FEN "..."] tags per PGN spec.
// -----------------------------------------------------------------------------
struct PgnHeader {
    std::string event = "?";
    std::string site  = "?";
    std::string date;                  // "yyyy.mm.dd"; empty -> today's UTC date
    std::string round;                 // e.g. "3.2"
    std::string white = "?";
    std::string black = "?";
    std::string eco;
    std::string time_control;          // cutechess format: "60+1", "40/900", "inf"
    std::string termination;           // "Normal" | "Time forfeit" | "Adjudication" | ...
    int         opening_plies = 0;     // custom tag; 0 = omitted
    std::string starting_fen;          // empty = standard startpos
};

// -----------------------------------------------------------------------------
// PgnMoveAnnotation
//
// Per-move analysis emitted as a {cp/depth time} comment after the move in
// cutechess format. Provide one PgnMoveAnnotation per move in the game, or
// pass an empty vector to build_pgn if annotations aren't desired.
//
// Any single move can be individually skipped (e.g. opening-book moves with
// no search) by setting skip=true; the move token still appears but with no
// comment attached.
//
// cp is centipawns from side-to-move perspective (positive = side to move is
// winning). Use q_to_cp() to convert from a Q value in [-1, 1].
// -----------------------------------------------------------------------------
struct PgnMoveAnnotation {
    int     cp         = 0;
    int     depth      = 0;         // typically max PV depth for MCTS
    int64_t elapsed_ms = 0;
    bool    skip       = false;
};

// -----------------------------------------------------------------------------
// PgnConfig
//
// Which tags and annotation fields to include. All optional features are
// individual booleans so they compose freely (an enum with tiers would lock
// you into combinations you'll eventually want to break).
//
// Two presets cover the common cases:
//   minimal()   : self-play / training data. Just STR + ECO + moves + result.
//                 No annotations, no extra tags -- millions of games, so we
//                 keep it small.
//   annotated() : tournament / external play. Full CCRL-style headers plus
//                 per-move {eval/depth time} comments. Match count is small,
//                 verbosity is free.
//
// Annotation fields are only ever emitted if the annotations vector passed to
// build_pgn is non-empty AND matches moves.size(). Otherwise they're silently
// skipped -- the config just says "please emit these if you have them".
// -----------------------------------------------------------------------------
struct PgnConfig {
    // Optional header tags
    bool include_date          = true;
    bool include_round         = false;
    bool include_eco           = true;
    bool include_time_control  = false;
    bool include_termination   = false;
    bool include_opening_plies = false;
    bool include_ply_count     = false;

    // Per-move annotation fields. Eval is a prerequisite -- if include_eval
    // is false, the whole {...} comment is skipped regardless of the depth
    // and time flags. This matches cutechess convention (eval-first format).
    bool include_eval  = false;
    bool include_depth = false;
    bool include_time  = false;

    // Approximate character width for movetext line wrapping. Won't split
    // mid-move, so lines may exceed slightly.
    int max_line_width = 80;

    static PgnConfig minimal();
    static PgnConfig annotated();
};

// -----------------------------------------------------------------------------
// build_pgn
//
// Assemble one complete PGN block, ready to write to disk or log.
//
//   header      : all header info; missing fields omitted per config.
//   moves       : moves in order, applied to header.starting_fen (or standard
//                 startpos if that's empty). Not modified.
//   annotations : per-move analysis. Either empty (no comments) or exactly
//                 moves.size() entries. Size mismatch -> treated as empty
//                 (defensive; caller bug rather than crash).
//   result_str  : one of "1-0", "0-1", "1/2-1/2", "*".
//   config      : see PgnConfig above.
//
// Returns the full PGN block followed by a trailing blank line, so
// concatenating multiple build_pgn results produces a valid multi-game PGN.
// -----------------------------------------------------------------------------
std::string build_pgn(
    const PgnHeader& header,
    const std::vector<chess::Move>& moves,
    const std::vector<PgnMoveAnnotation>& annotations,
    const std::string& result_str,
    const PgnConfig& config
);

// -----------------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------------

// Lc0-style Q -> centipawn mapping. Q in [-1, 1]; cp is signed centipawns.
// This is one convention among several; if you need Stockfish-scale or a
// different mapping, do the conversion yourself before setting annotation.cp.
int q_to_cp(double q);

// "yyyy.mm.dd" of the current UTC date. Used when PgnHeader::date is empty.
std::string today_utc_date();