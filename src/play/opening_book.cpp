// =============================================================================
// opening_book.cpp
// =============================================================================

#include "opening_book.hpp"
#include <fstream>
#include <sstream>
#include <random>
#include <numeric>
#include <algorithm>
#include <cctype>

namespace {

// Trim leading/trailing ASCII whitespace.
std::string trim(const std::string& s) {
    size_t a = 0, b = s.size();
    while (a < b && std::isspace(static_cast<unsigned char>(s[a]))) ++a;
    while (b > a && std::isspace(static_cast<unsigned char>(s[b - 1]))) --b;
    return s.substr(a, b - a);
}

bool starts_with(const std::string& s, const char* prefix) {
    return s.rfind(prefix, 0) == 0;
}

// Is this token a PGN result marker?
bool is_result_token(const std::string& tok) {
    return tok == "1-0" || tok == "0-1" || tok == "1/2-1/2" || tok == "*";
}

// Is this token a move number like "1." or "23." or (rare) "12..."?
// A move-number token is all digits followed by one or more dots.
bool is_move_number(const std::string& tok) {
    if (tok.empty()) return false;
    size_t i = 0;
    while (i < tok.size() && std::isdigit(static_cast<unsigned char>(tok[i]))) ++i;
    if (i == 0) return false;                 // must start with a digit
    if (i == tok.size()) return false;        // pure number, no dot -> not this
    for (size_t j = i; j < tok.size(); ++j) {
        if (tok[j] != '.') return false;      // trailing chars must all be dots
    }
    return true;
}

// Extract the [Eco "..."] value from a header line, or "" if not that header.
std::string parse_eco_header(const std::string& line) {
    // Expected form: [Eco "A07"]
    if (!starts_with(line, "[Eco ")) return "";
    size_t q1 = line.find('"');
    if (q1 == std::string::npos) return "";
    size_t q2 = line.find('"', q1 + 1);
    if (q2 == std::string::npos) return "";
    return line.substr(q1 + 1, q2 - q1 - 1);
}

} // namespace

// -----------------------------------------------------------------------------
// parse_block
//
// A block is everything belonging to one game: header lines (each starting
// with '[') followed by movetext lines that may wrap. We:
//   * scan header lines, grabbing [Eco ...] if present;
//   * concatenate all non-header lines into one movetext string;
//   * tokenise on whitespace;
//   * drop move-number tokens and the result token;
//   * keep the rest as SAN moves.
// -----------------------------------------------------------------------------
bool OpeningBook::parse_block(const std::string& block, int source_index,
                              Opening& out) {
    out = Opening{};
    out.source_index = source_index;

    std::istringstream iss(block);
    std::string line;
    std::string movetext;

    while (std::getline(iss, line)) {
        std::string t = trim(line);
        if (t.empty()) continue;

        if (t[0] == '[') {
            // Header line. Only [Eco ...] is of interest.
            std::string eco = parse_eco_header(t);
            if (!eco.empty()) out.eco = eco;
        } else {
            // Movetext line (possibly a wrapped continuation). Join with a
            // space so a wrap mid-token ("8.\nQd2") still tokenises correctly.
            if (!movetext.empty()) movetext += ' ';
            movetext += t;
        }
    }

    // Tokenise movetext on whitespace.
    std::istringstream mss(movetext);
    std::string tok;
    while (mss >> tok) {
        if (is_result_token(tok)) continue;
        if (is_move_number(tok)) continue;

        // Some PGNs glue a move number to the first move ("1.Nf3"). Split it.
        size_t dot = tok.find_last_of('.');
        if (dot != std::string::npos && dot + 1 < tok.size()) {
            bool prefix_numeric = true;
            for (size_t i = 0; i < dot; ++i) {
                if (!std::isdigit(static_cast<unsigned char>(tok[i]))) {
                    prefix_numeric = false;
                    break;
                }
            }
            if (prefix_numeric && dot > 0) {
                tok = tok.substr(dot + 1);     // keep only the SAN part
            }
        }

        if (tok.empty()) continue;
        out.san_moves.push_back(tok);
    }

    return !out.san_moves.empty();
}

// -----------------------------------------------------------------------------
// load
//
// Splits the file into blocks. A new block begins at an [Event ...] header.
// Everything from one [Event to just before the next is one block.
// -----------------------------------------------------------------------------
bool OpeningBook::load(const std::string& pgn_path, std::string& error) {
    openings_.clear();

    std::ifstream file(pgn_path);
    if (!file) {
        error = "could not open opening file: " + pgn_path;
        return false;
    }

    std::string line;
    std::string current_block;
    int next_index = 0;

    auto flush_block = [&]() {
        if (trim(current_block).empty()) return;
        Opening op;
        if (parse_block(current_block, next_index, op)) {
            openings_.push_back(std::move(op));
            ++next_index;
        }
        current_block.clear();
    };

    while (std::getline(file, line)) {
        std::string t = trim(line);
        // A new [Event ...] marks the start of the next game.
        if (starts_with(t, "[Event ") && !trim(current_block).empty()) {
            flush_block();
        }
        current_block += line;
        current_block += '\n';
    }
    flush_block();   // last block

    if (openings_.empty()) {
        error = "opening file parsed but contained no usable openings: "
                + pgn_path;
        return false;
    }

    error.clear();
    return true;
}

// -----------------------------------------------------------------------------
// sample
//
// SEQUENTIAL read: take the first `count` openings in file order. No shuffling.
//
// The opening file (8moves_v3.pgn) is already pre-shuffled on disk, so the
// first `count` lines are a representative, fixed set. Reading them straight
// through means the internal tournament and any external tournament pointed at
// the SAME file (in file order) play the identical opening set -- which is the
// whole reason for not shuffling here.
//
// -----------------------------------------------------------------------------
std::vector<Opening> OpeningBook::sample(size_t count) const {
    if (count > openings_.size()) count = openings_.size();

    std::vector<Opening> result;
    result.reserve(count);
    for (size_t i = 0; i < count; ++i) {
        result.push_back(openings_[i]);
    }
    return result;
}