// =============================================================================
// pgn_writer.cpp
// =============================================================================

#include "pgn_writer.hpp"
#include <sstream>
#include <cmath>
#include <ctime>
#include <cstdio>
#include <algorithm>

// -----------------------------------------------------------------------------
// Presets
// -----------------------------------------------------------------------------
PgnConfig PgnConfig::minimal() {
    PgnConfig c;
    c.include_date          = false;
    c.include_round         = false;
    c.include_eco           = true;    // trivial to include, useful for filtering
    c.include_time_control  = false;
    c.include_termination   = false;
    c.include_opening_plies = false;
    c.include_ply_count     = false;
    c.include_eval          = false;
    c.include_depth         = false;
    c.include_time          = false;
    return c;
}

PgnConfig PgnConfig::annotated() {
    PgnConfig c;
    c.include_date          = true;
    c.include_round         = true;
    c.include_eco           = true;
    c.include_time_control  = true;
    c.include_termination   = true;
    c.include_opening_plies = true;
    c.include_ply_count     = true;
    c.include_eval          = true;
    c.include_depth         = true;
    c.include_time          = true;
    return c;
}

// -----------------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------------
int q_to_cp(double q) {
    // Lc0 mapping. Sharp near |q|=1, roughly linear near 0.
    q = std::clamp(q, -0.999, 0.999);
    return static_cast<int>(std::round(111.7 * std::tan(1.5620688421 * q)));
}

std::string today_utc_date() {
    std::time_t now = std::time(nullptr);
    std::tm tm_utc{};
#ifdef _WIN32
    gmtime_s(&tm_utc, &now);
#else
    gmtime_r(&now, &tm_utc);
#endif
    char buf[16];
    std::snprintf(buf, sizeof(buf), "%04d.%02d.%02d",
                  tm_utc.tm_year + 1900, tm_utc.tm_mon + 1, tm_utc.tm_mday);
    return std::string(buf);
}

// -----------------------------------------------------------------------------
// Emit "[Tag \"value\"]\n" if value is non-empty. Overload for int always
// emits (numeric tags typically want 0 to appear rather than be omitted --
// callers control by not calling if 0 is meaningless).
// -----------------------------------------------------------------------------
static void emit_tag(std::ostream& os, const char* tag, const std::string& value) {
    if (value.empty()) return;
    os << "[" << tag << " \"" << value << "\"]\n";
}
static void emit_tag(std::ostream& os, const char* tag, int value) {
    os << "[" << tag << " \"" << value << "\"]\n";
}

// -----------------------------------------------------------------------------
// format_annotation: build the "{eval/depth time}" comment for one move.
//
// Format follows cutechess convention:
//   {+0.34}            eval only
//   {+0.34/15}         eval + depth
//   {+0.34 1.20s}      eval + time
//   {+0.34/15 1.20s}   eval + depth + time
//
// Eval is required. If include_eval is false OR the move is individually
// skipped, we emit no comment at all -- depth/time don't stand on their own
// in cutechess format, and inventing a placeholder like "?/15" is worse than
// omitting.
// -----------------------------------------------------------------------------
static std::string format_annotation(const PgnMoveAnnotation& ann,
                                     const PgnConfig& cfg) {
    if (ann.skip) return "";
    if (!cfg.include_eval) return "";

    char buf[64];
    std::string s = "{";

    const double pawns = ann.cp / 100.0;
    std::snprintf(buf, sizeof(buf), "%+.2f", pawns);
    s += buf;

    if (cfg.include_depth && ann.depth > 0) {
        s += "/" + std::to_string(ann.depth);
    }

    if (cfg.include_time) {
        const double seconds = ann.elapsed_ms / 1000.0;
        std::snprintf(buf, sizeof(buf), " %.2fs", seconds);
        s += buf;
    }

    s += "}";
    return s;
}

// -----------------------------------------------------------------------------
std::string build_pgn(
    const PgnHeader& header,
    const std::vector<chess::Move>& moves,
    const std::vector<PgnMoveAnnotation>& annotations,
    const std::string& result_str,
    const PgnConfig& config)
{
    std::ostringstream out;

    // ---- Headers -----------------------------------------------------------
    // Seven Tag Roster: Event, Site, Date, Round, White, Black, Result.
    // Event/Site/White/Black/Result are always emitted; Date and Round are
    // gated by config since minimal PGNs skip them.
    emit_tag(out, "Event", header.event);
    emit_tag(out, "Site",  header.site);
    if (config.include_date) {
        emit_tag(out, "Date", header.date.empty() ? today_utc_date() : header.date);
    }
    if (config.include_round) {
        emit_tag(out, "Round", header.round.empty() ? std::string("?") : header.round);
    }
    emit_tag(out, "White",  header.white);
    emit_tag(out, "Black",  header.black);
    emit_tag(out, "Result", result_str);

    // Optional tags -- each individually gated.
    if (config.include_eco)          emit_tag(out, "ECO",         header.eco);
    if (config.include_time_control) emit_tag(out, "TimeControl", header.time_control);
    if (config.include_termination)  emit_tag(out, "Termination", header.termination);
    if (config.include_opening_plies && header.opening_plies > 0) {
        emit_tag(out, "OpeningPlies", header.opening_plies);
    }
    if (config.include_ply_count) {
        emit_tag(out, "PlyCount", static_cast<int>(moves.size()));
    }

    // SetUp / FEN tags for non-standard starting positions.
    if (!header.starting_fen.empty() &&
        header.starting_fen != std::string(chess::constants::STARTPOS)) {
        emit_tag(out, "SetUp", std::string("1"));
        emit_tag(out, "FEN",   header.starting_fen);
    }

    out << "\n";

    // ---- Movetext ----------------------------------------------------------
    chess::Board board;
    board.setFen(header.starting_fen.empty()
                 ? chess::constants::STARTPOS
                 : header.starting_fen);

    // Annotations must exactly match the move count to be usable. Any other
    // size (including empty) is treated as "no annotations". Defensive rather
    // than assert -- a caller bug shouldn't corrupt output.
    const bool annotations_valid =
        !annotations.empty() && annotations.size() == moves.size();

    // Line accumulator + wrap logic. We emit one token at a time (a token is
    // "1. Nf3" or "Nf3", optionally with " {annotation}" attached) and flush
    // to the stream when the line would exceed max_line_width.
    std::string line;
    line.reserve(128);

    auto flush_line = [&]() {
        if (line.empty()) return;
        out << line << "\n";
        line.clear();
    };

    // Determine whose move comes first based on the starting position, so the
    // move-number prefix is placed correctly. Standard startpos and any
    // opening line ending on White-to-move give first_is_white=true; a FEN
    // starting on Black would flip this.
    const bool first_is_white = (board.sideToMove() == chess::Color::WHITE);

    for (size_t i = 0; i < moves.size(); ++i) {
        std::string token;

        // Move-number prefix. For White moves: "N. ". For the very first move
        // if Black is on move (uncommon; only happens with a custom FEN), use
        // "N... " ellipsis form per PGN convention.
        const bool this_is_white = first_is_white ? (i % 2 == 0) : (i % 2 == 1);
        if (this_is_white) {
            const int move_number = first_is_white
                ? static_cast<int>(i / 2 + 1)
                : static_cast<int>((i + 1) / 2 + 1);
            token = std::to_string(move_number) + ". ";
        } else if (i == 0 && !first_is_white) {
            token = "1... ";
        }

        token += chess::uci::moveToSan(board, moves[i]);
        board.makeMove(moves[i]);

        if (annotations_valid) {
            const std::string ann = format_annotation(annotations[i], config);
            if (!ann.empty()) token += " " + ann;
        }

        // Wrap: flush before adding if this token would push us over.
        if (!line.empty() &&
            static_cast<int>(line.size() + 1 + token.size()) > config.max_line_width) {
            flush_line();
        }
        if (!line.empty()) line += " ";
        line += token;
    }

    // Result terminator on its own -- wrap first if needed so it's not
    // orphaned mid-line at edge cases.
    if (!line.empty() &&
        static_cast<int>(line.size() + 1 + result_str.size()) > config.max_line_width) {
        flush_line();
    }
    if (!line.empty()) line += " ";
    line += result_str;
    flush_line();

    out << "\n";  // trailing blank line between games (PGN spec)
    return out.str();
}