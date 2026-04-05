#pragma once

#include <vector>
#include "chess.hpp"

// --- Core Constants ---
constexpr int BOARD_DIM = 8;
constexpr int INPUT_CHANNELS = 69;
constexpr int TOTAL_INPUT_SIZE = INPUT_CHANNELS * BOARD_DIM * BOARD_DIM;
constexpr int POLICY_CHANNELS = 73;
constexpr int TOTAL_POLICY_MOVES = POLICY_CHANNELS * BOARD_DIM * BOARD_DIM;

// --- Structs ---
struct PolicyComponent {
    int row;
    int col;
    int channel;
};

// --- Main External Functions ---

// Fills the 69-channel float array with the current board state and history
void board_to_tensor_69(const chess::Board& board, const std::vector<chess::Board>& history_boards, c10::Half* planes_out);

// Creates a boolean mask of valid moves aligned with the 4672-length policy vector
void get_legal_move_mask(const chess::Board& board, bool* mask_out);

// Maps a list of probabilities to the global 4672-length policy vector
void map_policy_to_global_vector(const chess::Movelist& moves, const float* probs, const chess::Board& board, float* policy_vector_out);

// --- Move/Policy Conversion Utilities ---
PolicyComponent move_to_policy_components(const chess::Move& move, const chess::Board& board);
chess::Move policy_components_to_move(int from_row_norm, int from_col_norm, int channel, const chess::Board& board);

int policy_components_to_flat_index(int from_row, int from_col, int channel);
PolicyComponent policy_flat_index_to_components(int flat_index);