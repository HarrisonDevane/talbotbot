#pragma once

// =============================================================================
// opening_book.hpp
//
// Loads a file of PGN "opening lines" and provides a deterministic, seeded
// random subset of them for tournament play.
//
// INPUT FORMAT (as supplied):
//   A text file of concatenated PGN games. Each game is a [Event ...]-headed
//   block of header lines, then SAN movetext that MAY wrap across several
//   physical lines, terminated by a result token (1-0 / 0-1 / 1/2-1/2 / *).
//   Each block is a short opening line (~8 full moves) -- not a full game.
//
// WHAT THIS CLASS DOES:
//   * Parses the file into a list of Opening records.
//   * Each Opening stores the raw SAN tokens ("Nf3", "d5", "g3", ...) with
//     move numbers ("1.", "2.") and the result token stripped out.
//   * Provides a SEEDED, deterministic random subset: same (file, seed, count)
//     always yields the same openings, in the same order. No sample file is
//     persisted -- the subset is fully derivable from those three inputs.
//
// WHAT THIS CLASS DELIBERATELY DOES NOT DO:
//   * It does NOT convert SAN strings into chess::Move objects. SAN is
//     position-dependent, so that conversion needs a live chess::Board and is
//     done later (when a session sets up its starting position). Keeping this
//     class pure text-processing means it has zero dependency on the chess
//     move generator and is trivially testable in isolation.
//
// DETERMINISM:
//   load() reads every opening in file order. sample(count, seed) returns the
//   first `count` of them in that order -- a plain sequential read, no shuffle.
//   The opening file is already pre-shuffled on disk, so the first N lines are
//   a fixed, representative set. `seed` is ignored; see opening_book.cpp::sample.
// =============================================================================

#include <string>
#include <vector>
#include <cstdint>

// One parsed opening line.
struct Opening {
    std::string eco;                  // ECO code from [Eco "..."], "" if absent
    std::vector<std::string> san_moves; // SAN tokens in order, no move numbers
    int source_index = -1;            // 0-based position in the original file
};

class OpeningBook {
public:
    OpeningBook() = default;

    // Parse `pgn_path` into the internal opening list.
    // Returns true on success. On failure (file missing/unreadable) returns
    // false and leaves the book empty; `error` is filled with a reason.
    bool load(const std::string& pgn_path, std::string& error);

    // Total openings parsed from the file.
    size_t size() const { return openings_.size(); }

    // All parsed openings, in file order.
    const std::vector<Opening>& all() const { return openings_; }

    // Sequential subset: the first `count` openings in file order. No shuffle.
    //   count : how many openings to return. If count >= size(), returns ALL.
    //   seed  : IGNORED. Kept only for call-site compatibility. The opening
    //           file is already pre-shuffled on disk, so a straight first-N
    //           read is the fixed set. See opening_book.cpp.
    // The returned vector is a copy; callers may keep it for the tournament.
    std::vector<Opening> sample(size_t count) const;

private:
    std::vector<Opening> openings_;

    // Parse one PGN block (headers + wrapped movetext) into an Opening.
    // Returns false if the block contains no usable moves.
    static bool parse_block(const std::string& block, int source_index,
                            Opening& out);
};