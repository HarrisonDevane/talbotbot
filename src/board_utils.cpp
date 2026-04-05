// src_shared_c/board_utils.cpp

#include "chess.hpp"
#include <vector>
#include <stdexcept>
#include <cmath>
#include <string>
#include <c10/util/Half.h>
#include "board_utils.hpp"

using namespace chess;

// --- High-Performance Mapping Lookups ---
inline int get_sliding_dir_idx(int dr, int df) {
    if (dr == -1 && df == 0) return 0;
    if (dr == -1 && df == 1) return 1;
    if (dr == 0  && df == 1) return 2;
    if (dr == 1  && df == 1) return 3;
    if (dr == 1  && df == 0) return 4;
    if (dr == 1  && df == -1) return 5;
    if (dr == 0  && df == -1) return 6;
    if (dr == -1 && df == -1) return 7;
    return -1;
}

constexpr int SLIDING_DIRS_LIST[8][2] = {
    {-1, 0}, {-1, 1}, {0, 1}, {1, 1}, {1, 0}, {1, -1}, {0, -1}, {-1, -1}
};

inline int get_knight_offset_idx(int dr, int df) {
    if (dr == -2 && df == -1) return 0;
    if (dr == -2 && df == 1) return 1;
    if (dr == -1 && df == -2) return 2;
    if (dr == -1 && df == 2) return 3;
    if (dr == 1  && df == -2) return 4;
    if (dr == 1  && df == 2) return 5;
    if (dr == 2  && df == -1) return 6;
    if (dr == 2  && df == 1) return 7;
    return -1;
}

constexpr int KNIGHT_OFFSETS_LIST[8][2] = {
    {-2, -1}, {-2, 1}, {-1, -2}, {-1, 2}, {1, -2}, {1, 2}, {2, -1}, {2, 1}
};

inline int get_pawn_promo_move_type_idx(int df) {
    if (df == 0) return 0;
    if (df == -1) return 1;
    if (df == 1) return 2;
    return -1;
}

constexpr int PAWN_PROMO_MOVE_TYPES_LIST[3] = {0, -1, 1};

// --- Helper Functions ---

void _fill_piece_planes(const Board& board, Color orientation_color, c10::Half* out_view, int start_plane_idx) {
    const PieceType PIECE_TYPES[6] = {
        PieceType::PAWN, PieceType::KNIGHT, PieceType::BISHOP,
        PieceType::ROOK, PieceType::QUEEN, PieceType::KING
    };

    // Create a half-precision 1.0 constant
    const c10::Half one = c10::Half(1.0f);

    for (int color_idx = 0; color_idx < 2; ++color_idx) {
        Color color = static_cast<Color>(color_idx);
        bool is_me = (color == orientation_color);
        int base_plane = start_plane_idx + (is_me ? 0 : 6);

        for (int pt_idx = 0; pt_idx < 6; ++pt_idx) {
            PieceType pt = PIECE_TYPES[pt_idx];
            Bitboard bb = board.pieces(pt, color);
            int plane_idx = base_plane + pt_idx;

            while (bb) {
                int square = bb.pop(); 
                int rank = square / 8;
                int col = square % 8;
                
                // Vertical mirror logic based on side to move
                int row = (orientation_color == Color::WHITE) ? (7 - rank) : rank;
                
                // Write as FP16 1.0
                out_view[(plane_idx * 64) + (row * 8) + col] = one;
            }
        }
    }
}

void board_to_tensor_69(const Board& board, const std::vector<Board>& history_boards, c10::Half* planes_out) {
    // 0. Clear the buffer (crucial since we no longer use memset for bytes)
    // We use std::fill for type safety with c10::Half
    std::fill(planes_out, planes_out + TOTAL_INPUT_SIZE, c10::Half(0.0f));

    // 1. Current Piece Planes (0-11)
    _fill_piece_planes(board, board.sideToMove(), planes_out, 0);

    // 2. Side to Move (12)
    c10::Half turn_val = (board.sideToMove() == Color::WHITE) ? c10::Half(1.0f) : c10::Half(0.0f);
    for(int i=0; i<64; ++i) planes_out[12 * 64 + i] = turn_val;

    // 3. Castling Rights (13-16)
    Color us = board.sideToMove();
    const auto rights = board.castlingRights();

    c10::Half us_ks_val   = rights.has(us,  Board::CastlingRights::Side::KING_SIDE) ? c10::Half(1.0f) : c10::Half(0.0f);
    c10::Half us_qs_val   = rights.has(us,  Board::CastlingRights::Side::QUEEN_SIDE) ? c10::Half(1.0f) : c10::Half(0.0f);
    c10::Half them_ks_val = rights.has(~us, Board::CastlingRights::Side::KING_SIDE) ? c10::Half(1.0f) : c10::Half(0.0f);
    c10::Half them_qs_val = rights.has(~us, Board::CastlingRights::Side::QUEEN_SIDE) ? c10::Half(1.0f) : c10::Half(0.0f);

    for(int i=0; i<64; ++i) {
        planes_out[13 * 64 + i] = us_ks_val;
        planes_out[14 * 64 + i] = us_qs_val;
        planes_out[15 * 64 + i] = them_ks_val;
        planes_out[16 * 64 + i] = them_qs_val;
    }

    // 4. En Passant (17)
    Square ep_sq = board.enpassantSq();
    if (ep_sq != Square::NO_SQ) {
        int ep_file = ep_sq.index() % 8;
        c10::Half one = c10::Half(1.0f);
        for (int row = 0; row < 8; ++row) {
            planes_out[(17 * 64) + (row * 8) + ep_file] = one;
        }
    }

    // 5. History (18-65)
    for (size_t i = 0; i < 4; ++i) {
        int start_plane_idx = 18 + (i * 12);
        if (i < history_boards.size()) {
            _fill_piece_planes(history_boards[i], board.sideToMove(), planes_out, start_plane_idx);
        }
    }

    // 6. Repetition (66-67)
    c10::Half rep_2 = board.isRepetition(2) ? c10::Half(1.0f) : c10::Half(0.0f);
    c10::Half rep_3 = board.isRepetition(3) ? c10::Half(1.0f) : c10::Half(0.0f);
    for(int i=0; i<64; ++i) {
        planes_out[66 * 64 + i] = rep_2;
        planes_out[67 * 64 + i] = rep_3;
    }

    // 7. 50-Move Rule (68) - Normalized FP16
    c10::Half clock_val = c10::Half(static_cast<float>(board.halfMoveClock()) / 100.0f);
    for(int i=0; i<64; ++i) {
        planes_out[68 * 64 + i] = clock_val;
    }
}

PolicyComponent move_to_policy_components(const Move& move, const Board& board) {
    // Extract integer indexes explicitly to avoid conversion failures
    int from_sq = move.from().index();
    int to_sq = move.to().index();

    int from_rank = from_sq / 8;
    int from_file = from_sq % 8;
    int to_rank = to_sq / 8;
    int to_file = to_sq % 8;

    int from_row_norm, from_col_norm, to_row_norm, to_col_norm;

    if (board.sideToMove() == Color::WHITE) {
        from_row_norm = 7 - from_rank;
        from_col_norm = from_file;
        to_row_norm = 7 - to_rank;
        to_col_norm = to_file;
    } else {
        from_row_norm = from_rank;
        from_col_norm = from_file;
        to_row_norm = to_rank;
        to_col_norm = to_file;
    }

    int dr = to_row_norm - from_row_norm;
    int df = to_col_norm - from_col_norm;

    if (move.typeOf() == Move::PROMOTION && move.promotionType() != PieceType::QUEEN) {
        int promo_piece_idx = -1;
        if (move.promotionType() == PieceType::KNIGHT) promo_piece_idx = 0;
        else if (move.promotionType() == PieceType::BISHOP) promo_piece_idx = 1;
        else if (move.promotionType() == PieceType::ROOK) promo_piece_idx = 2;

        int pawn_move_type_idx = get_pawn_promo_move_type_idx(df);
        int channel = 64 + (promo_piece_idx * 3) + pawn_move_type_idx;
        return {from_row_norm, from_col_norm, channel};
    }

    PieceType piece_type = board.at<Piece>(move.from()).type();

    if (piece_type == PieceType::KNIGHT) {
        int offset_idx = get_knight_offset_idx(dr, df);
        if (offset_idx == -1) throw std::runtime_error("Invalid knight move offset");
        int channel = 56 + offset_idx;
        return {from_row_norm, from_col_norm, channel};
    }

    if (move.typeOf() == Move::CASTLING) {
            const auto rights = board.castlingRights();
            const auto side   = rights.closestSide(move.to().file(), move.from().file());
            const bool is_king_side = (side == Board::CastlingRights::Side::KING_SIDE);

            int df_norm  = is_king_side ? 1 : -1;
            int dir_idx  = get_sliding_dir_idx(0, df_norm);
            
            int distance = 2; 

            int channel = (dir_idx * 7) + (distance - 1);
            return {from_row_norm, from_col_norm, channel};
        }

    int dir_idx = -1;
    int distance = 0;

    if (dr == 0) {
        dir_idx = get_sliding_dir_idx(0, df > 0 ? 1 : -1);
        distance = std::abs(df);
    } else if (df == 0) {
        dir_idx = get_sliding_dir_idx(dr > 0 ? 1 : -1, 0);
        distance = std::abs(dr);
    } else if (std::abs(dr) == std::abs(df)) {
        if (dr < 0 && df > 0) dir_idx = get_sliding_dir_idx(-1, 1);
        else if (dr < 0 && df < 0) dir_idx = get_sliding_dir_idx(-1, -1);
        else if (dr > 0 && df > 0) dir_idx = get_sliding_dir_idx(1, 1);
        else if (dr > 0 && df < 0) dir_idx = get_sliding_dir_idx(1, -1);
        distance = std::abs(dr);
    } else {
        throw std::runtime_error("Could not classify move into policy head channels.");
    }

    if (distance < 1 || distance > 7) throw std::runtime_error("Sliding move distance out of bounds.");

    int channel = (dir_idx * 7) + (distance - 1);
    return {from_row_norm, from_col_norm, channel};
}

Move policy_components_to_move(int from_row_norm, int from_col_norm, int channel, const Board& board) {
    int actual_from_rank, actual_from_file;
    if (board.sideToMove() == Color::WHITE) {
        actual_from_rank = 7 - from_row_norm;
        actual_from_file = from_col_norm;
    } else {
        actual_from_rank = from_row_norm;
        actual_from_file = from_col_norm;
    }

    int actual_from_square = actual_from_rank * 8 + actual_from_file;
    int to_row_norm = 0, to_col_norm = 0;
    
    int expected_to_square = -1;
    PieceType expected_promo = PieceType::NONE;
    bool is_underpromotion = false;

    // 1. Underpromotions
    if (channel >= 64 && channel <= 72) {
        int relative_channel = channel - 64;
        int promo_piece_idx = relative_channel / 3;
        int pawn_move_type_idx = relative_channel % 3;

        const PieceType PROMO_TYPES[3] = {PieceType::KNIGHT, PieceType::BISHOP, PieceType::ROOK};
        expected_promo = PROMO_TYPES[promo_piece_idx];
        is_underpromotion = true;

        int df_norm = PAWN_PROMO_MOVE_TYPES_LIST[pawn_move_type_idx];
        to_row_norm = 0;
        to_col_norm = from_col_norm + df_norm;
    }
    // 2. Knight Moves
    else if (channel >= 56 && channel <= 63) {
        int offset_idx = channel - 56;
        to_row_norm = from_row_norm + KNIGHT_OFFSETS_LIST[offset_idx][0];
        to_col_norm = from_col_norm + KNIGHT_OFFSETS_LIST[offset_idx][1];
    }
    // 3. Sliding Moves
    else if (channel >= 0 && channel <= 55) {
        int dir_idx = channel / 7;
        int distance = (channel % 7) + 1;
        to_row_norm = from_row_norm + (SLIDING_DIRS_LIST[dir_idx][0] * distance);
        to_col_norm = from_col_norm + (SLIDING_DIRS_LIST[dir_idx][1] * distance);
    } else {
        return Move::NO_MOVE;
    }

    if (to_row_norm < 0 || to_row_norm > 7 || to_col_norm < 0 || to_col_norm > 7) return Move::NO_MOVE;

    int actual_to_rank = (board.sideToMove() == Color::WHITE) ? (7 - to_row_norm) : to_row_norm;
    int actual_to_file = to_col_norm;
    expected_to_square = actual_to_rank * 8 + actual_to_file;

    // Match against strictly generated legal moves to avoid constructor API incompatibilities
    Movelist moves;
    movegen::legalmoves(moves, board);
    
    for (const Move& legal_move : moves) {
        // --- NEW: Intercept Castling ---
        if (legal_move.typeOf() == Move::CASTLING) {
            const auto rights = board.castlingRights();
            const auto side = rights.closestSide(legal_move.to().file(), legal_move.from().file());
            const bool is_king_side = (side == Board::CastlingRights::Side::KING_SIDE);
            
            // Calculate where the King ACTUALLY lands in standard chess
            int king_land_file = is_king_side ? 6 : 2; // g-file or c-file
            int king_land_rank = (board.sideToMove() == Color::WHITE) ? 0 : 7;
            int standard_to_sq = king_land_rank * 8 + king_land_file;

            if (legal_move.from().index() == actual_from_square && standard_to_sq == expected_to_square) {
                return legal_move;
            }
            continue;
        }

        // Standard move matching
        if (legal_move.from().index() == actual_from_square && 
            legal_move.to().index() == expected_to_square) {
            
            if (is_underpromotion) {
                if (legal_move.promotionType() == expected_promo) return legal_move;
            } else {
                if (legal_move.typeOf() == Move::PROMOTION) {
                    if (legal_move.promotionType() == PieceType::QUEEN) return legal_move;
                } else {
                    return legal_move;
                }
            }
        }
    }

    return Move::NO_MOVE;
}

inline int policy_components_to_flat_index(int from_row, int from_col, int channel) {
    return from_row * (BOARD_DIM * POLICY_CHANNELS) + from_col * POLICY_CHANNELS + channel;
}

PolicyComponent policy_flat_index_to_components(int flat_index) {
    int channel = flat_index % POLICY_CHANNELS;
    int remaining = flat_index / POLICY_CHANNELS;
    int from_col = remaining % BOARD_DIM;
    int from_row = remaining / BOARD_DIM;
    return {from_row, from_col, channel};
}

void get_legal_move_mask(const Board& board, bool* mask_out) {
    Movelist moves;
    movegen::legalmoves(moves, board);
    
    for(int i = 0; i < TOTAL_POLICY_MOVES; ++i) {
        mask_out[i] = false;
    }

    for (const Move& move : moves) {
        PolicyComponent pc = move_to_policy_components(move, board);
        int flat_index = policy_components_to_flat_index(pc.row, pc.col, pc.channel);
        mask_out[flat_index] = true;
    }
}

void map_policy_to_global_vector(const Movelist& moves, const float* probs, const Board& board, float* policy_vector_out) {
    for(int i = 0; i < TOTAL_POLICY_MOVES; ++i) {
        policy_vector_out[i] = 0.0f;
    }

    for (int i = 0; i < moves.size(); ++i) {
        PolicyComponent pc = move_to_policy_components(moves[i], board);
        int flat_index = policy_components_to_flat_index(pc.row, pc.col, pc.channel);
        policy_vector_out[flat_index] = probs[i];
    }
}