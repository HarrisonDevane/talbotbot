# src_shared/utils.pyx

# Use standard Python import for modules accessed via Python API calls
import chess
import numpy as np
import torch
# We use Cython's view for NumPy array access, but keep numpy import for object creation
from cython.view cimport array as c_array
import cython

# C-level imports for types, functions, and constants
cimport numpy as cnp  # For c-typed access to NumPy data structures/types
from libc.math cimport tanh

# Add this to the top of utils.pyx alongside the imports
cdef extern from *:
    """
    // MSVC (Windows) compatibility
    #if defined(_MSC_VER)
    #include <intrin.h>
    #pragma intrinsic(_BitScanForward64)
    static __inline int __builtin_ctzll(unsigned long long bb) {
        unsigned long index;
        _BitScanForward64(&index, bb);
        return index;
    }
    #endif
    // GCC/Clang already have __builtin_ctzll defined natively
    """
    int __builtin_ctzll(unsigned long long x) nogil

cdef int _BOARD_DIM = 8
cdef int _INPUT_CHANNELS = 69
cdef int _TOTAL_INPUT_SIZE = _INPUT_CHANNELS * _BOARD_DIM * _BOARD_DIM
cdef int _POLICY_CHANNELS = 73
cdef int _TOTAL_POLICY_MOVES = _POLICY_CHANNELS * _BOARD_DIM * _BOARD_DIM

# Expose as Python globals
BOARD_DIM = _BOARD_DIM
INPUT_CHANNELS = _INPUT_CHANNELS
TOTAL_INPUT_SIZE = _TOTAL_INPUT_SIZE
POLICY_CHANNELS = _POLICY_CHANNELS
TOTAL_POLICY_MOVES = _TOTAL_POLICY_MOVES
BOARD_BYTES = _TOTAL_INPUT_SIZE // 8
MASK_BYTES  = _TOTAL_POLICY_MOVES // 8 


# --- Python-Level Constants (must remain Python for dict/list lookups) ---

SLIDING_DIRS_MAPPING = {
    (-1, 0): 0, (-1, 1): 1, (0, 1): 2, (1, 1): 3,
    (1, 0): 4, (1, -1): 5, (0, -1): 6, (-1, -1): 7
}
SLIDING_DIRS_LIST = list(SLIDING_DIRS_MAPPING.keys())

KNIGHT_OFFSETS_MAPPING = {
    (-2, -1): 0, (-2, 1): 1, (-1, -2): 2, (-1, 2): 3,
    (1, -2): 4, (1, 2): 5, (2, -1): 6, (2, 1): 7
}
KNIGHT_OFFSETS_LIST = list(KNIGHT_OFFSETS_MAPPING.keys())

PROMOTION_PIECES_ORDER = [chess.KNIGHT, chess.BISHOP, chess.ROOK]

PAWN_PROMO_MOVE_TYPES_MAPPING = {
    0: 0, -1: 1, 1: 2
}
PAWN_PROMO_MOVE_TYPES_LIST = list(PAWN_PROMO_MOVE_TYPES_MAPPING.keys())

# --- Helper Functions (C-typed) ---

cdef cnp.ndarray _get_piece_planes(object board_state, bint orientation_color):
    """
    Fills planes relative to orientation_color using fast bitboard scanning.
    """
    cdef cnp.ndarray piece_planes = np.zeros((12, _BOARD_DIM, _BOARD_DIM), dtype=np.float32)
    
    # C-typed variables for the fast loop
    cdef unsigned long long bb, color_mask
    cdef int square, row, col, base_plane, plane_idx, pt_idx
    cdef bint is_me, color

    # Map python-chess internal piece bitboards (1-indexed in python-chess: PAWN=1...KING=6)
    cdef tuple piece_masks = (
        board_state.pawns,
        board_state.knights,
        board_state.bishops,
        board_state.rooks,
        board_state.queens,
        board_state.kings
    )

    # Iterate over both colors: True (White) and False (Black)
    for color in (True, False):
        is_me = (color == orientation_color)
        base_plane = 0 if is_me else 6
        color_mask = board_state.occupied_co[color]
        
        for pt_idx in range(6):
            # Bitwise AND: Intersect the piece type bitboard with the color bitboard
            bb = piece_masks[pt_idx] & color_mask
            plane_idx = base_plane + pt_idx
            
            # Fast bit-scan loop: Iterates only as many times as there are pieces
            while bb:
                # Find the index of the lowest set bit (0-63)
                square = __builtin_ctzll(bb)
                
                # Convert 1D square to 2D tensor coordinates (Rank 7 -> Row 0)
                # bitwise right-shift by 3 is division by 8; bitwise AND 7 is modulo 8
                row = 7 - (square >> 3) 
                col = square & 7
                
                piece_planes[plane_idx, row, col] = 1.0
                
                # Clear the lowest set bit to move to the next piece
                bb &= (bb - 1)

    return piece_planes


cpdef cnp.ndarray board_to_tensor_69(object board):
    """
    Encode a python-chess Board into a (69, 8, 8) numpy float32 tensor.
    Fully Relative Representation with Spatial Invariance (Vertical Mirror).
    """
    cdef cnp.ndarray planes = np.zeros((_INPUT_CHANNELS, _BOARD_DIM, _BOARD_DIM), dtype=np.float32)
    
    # --- Local C-Typed variables ---
    cdef int ep_file, start_plane_idx, i
    cdef object temp_board, hist_piece_planes, current_planes
    cdef bint us, them

    # 1. Current Board State (Planes 0-11)
    current_planes = _get_piece_planes(board, board.turn)
    planes[0:12, :, :] = current_planes

    # 2. Auxiliary Planes (12-17)
    planes[12, :, :] = 1.0 if board.turn == chess.WHITE else 0.0

    # Planes 13-16: Castling Rights
    # Since we strictly vertically mirror, Kingside is ALWAYS right (+x) 
    # and Queenside is ALWAYS left (-x) for both colors.
    us = board.turn
    them = not board.turn

    planes[13, :, :] = 1.0 if board.has_kingside_castling_rights(us) else 0.0
    planes[14, :, :] = 1.0 if board.has_queenside_castling_rights(us) else 0.0
    planes[15, :, :] = 1.0 if board.has_kingside_castling_rights(them) else 0.0
    planes[16, :, :] = 1.0 if board.has_queenside_castling_rights(them) else 0.0

    # Plane 17: En Passant
    # We DO NOT pre-flip. The global flip later handles this correctly.
    if board.ep_square is not None:
        ep_file = chess.square_file(board.ep_square) 
        planes[17, :, ep_file] = 1.0

    # 3. Historical Board States (Planes 18-65)
    temp_board = board.copy()
    for i in range(4):
        if not temp_board.move_stack: 
            break
        temp_board.pop()
        
        # History relative to ROOT player (trajectory consistency)
        hist_piece_planes = _get_piece_planes(temp_board, board.turn)
        
        start_plane_idx = 18 + (i * 12)
        planes[start_plane_idx : start_plane_idx + 12, :, :] = hist_piece_planes

    # 4. Repetition Channels (Planes 66-67)
    planes[66, :, :] = 1.0 if board.is_repetition(count=2) else 0.0
    planes[67, :, :] = 1.0 if board.is_repetition(count=3) else 0.0

    # 5. Add 50-move rule counter
    planes[68, :, :] = board.halfmove_clock / 100.0

    # 6. Spatial Flip (Vertical Mirror ONLY)
    if board.turn == chess.BLACK:
        # axis=1 mirrors ranks, leaving files identical. 
        planes = np.flip(planes, axis=1).copy()

    return planes


# Use 'object' for chess.Move and chess.Board
cpdef tuple move_to_policy_components(object move, object board):
    """Converts a chess.Move object to a (row, col, channel) index."""
    
    # --- Local C-Typed variables for speed ---
    cdef int from_rank, from_file, to_rank, to_file
    cdef int from_row_norm, from_col_norm, to_row_norm, to_col_norm
    cdef int dr, df, channel
    cdef object offset_key, direction_key
    cdef int promo_piece_idx, pawn_move_type_idx
    cdef int distance, dir_idx
    cdef object piece_type 

    # Python-chess calls (remain Python)
    from_rank, from_file = chess.square_rank(move.from_square), chess.square_file(move.from_square)
    to_rank, to_file = chess.square_rank(move.to_square), chess.square_file(move.to_square)

    # Coordinate normalization (Strict Vertical Symmetry Logic)
    if board.turn == chess.WHITE:
        # White Perspective (Standard Matrix: Top-Left is a8)
        from_row_norm = 7 - from_rank
        from_col_norm = from_file
        to_row_norm = 7 - to_rank
        to_col_norm = to_file
    else:
        # Black Perspective (Vertical Mirror Matrix: Top-Left is a1)
        from_row_norm = from_rank
        from_col_norm = from_file
        to_row_norm = to_rank
        to_col_norm = to_file

    dr = to_row_norm - from_row_norm
    df = to_col_norm - from_col_norm

    # 1. Handle Underpromotions
    if move.promotion and move.promotion != chess.QUEEN:
        promo_piece_idx = PROMOTION_PIECES_ORDER.index(move.promotion)
        pawn_move_type_idx = PAWN_PROMO_MOVE_TYPES_MAPPING[df] # df is pawn_col_diff_type
        
        channel = 64 + (promo_piece_idx * 3) + pawn_move_type_idx
        return from_row_norm, from_col_norm, channel

    piece_type = board.piece_at(move.from_square).piece_type

    # 2. Handle Knight Moves
    if piece_type == chess.KNIGHT:
        offset_key = (dr, df)
        channel = 56 + KNIGHT_OFFSETS_MAPPING[offset_key]
        return from_row_norm, from_col_norm, channel

    # 3. Handle Castling (treated as King moves for 2 squares)
    if board.is_castling(move):
        if df == 2:
            direction_key = (0, 1) # East (Kingside)
        elif df == -2:
            direction_key = (0, -1) # West (Queenside)
        else:
            raise ValueError(f"Unexpected castling move displacement: {move}")

        distance = 2
        dir_idx = SLIDING_DIRS_MAPPING[direction_key]
        channel = (dir_idx * 7) + (distance - 1)
        return from_row_norm, from_col_norm, channel

    # 4. Handle General Sliding Moves
    direction_key = (0,0)
    distance = 0

    if dr == 0:
        direction_key = (0, 1) if df > 0 else (0, -1)
        distance = abs(df)
    elif df == 0:
        direction_key = (1, 0) if dr > 0 else (-1, 0)
        distance = abs(dr)
    elif abs(dr) == abs(df):
        if dr < 0 and df > 0: direction_key = (-1, 1) # NE
        elif dr < 0 and df < 0: direction_key = (-1, -1) # NW
        elif dr > 0 and df > 0: direction_key = (1, 1) # SE
        elif dr > 0 and df < 0: direction_key = (1, -1) # SW
        distance = abs(dr)
    else:
        raise ValueError(f"Could not classify move {move} into policy head channels.")

    if distance < 1 or distance > 7:
        raise ValueError(f"Sliding move distance {distance} out of bounds (1-7) for move {move}")

    dir_idx = SLIDING_DIRS_MAPPING[direction_key]
    channel = (dir_idx * 7) + (distance - 1)
    return from_row_norm, from_col_norm, channel


cpdef object policy_components_to_move(int from_row_norm, int from_col_norm, int channel, object board):
    """Converts a (row, col, channel) index back to a chess.Move."""
    
    # --- Local C-Typed variables for speed ---
    cdef int actual_from_rank, actual_from_file, actual_from_square
    cdef int actual_to_rank, actual_to_file, actual_to_square
    cdef int relative_channel, promo_piece_idx, pawn_move_type_idx
    cdef int df_norm, to_row_norm, to_col_norm
    cdef object promotion_piece, move # chess.Move object
    cdef int offset_idx, dr_norm, dir_idx, distance, sr_norm, sf_norm
    cdef object moving_piece # chess.Piece object

    # Inverse coordinate normalization (Matches the explicit geometry above)
    if board.turn == chess.WHITE:
        actual_from_rank = 7 - from_row_norm
        actual_from_file = from_col_norm
    else:
        actual_from_rank = from_row_norm
        actual_from_file = from_col_norm

    actual_from_square = chess.square(actual_from_file, actual_from_rank)

    # 1. Underpromotions (channels 64-72)
    if 64 <= channel <= 72:
        relative_channel = channel - 64
        promo_piece_idx = relative_channel // 3
        pawn_move_type_idx = relative_channel % 3

        if promo_piece_idx >= len(PROMOTION_PIECES_ORDER): return None
        promotion_piece = PROMOTION_PIECES_ORDER[promo_piece_idx]
        
        if pawn_move_type_idx >= len(PAWN_PROMO_MOVE_TYPES_LIST): return None
        df_norm = PAWN_PROMO_MOVE_TYPES_LIST[pawn_move_type_idx]

        to_row_norm = 0
        to_col_norm = from_col_norm + df_norm

        if not (0 <= to_col_norm <= 7): return None
        
        # Inverse logic for destination
        if board.turn == chess.WHITE:
            actual_to_rank = 7 - to_row_norm
            actual_to_file = to_col_norm
        else:
            actual_to_rank = to_row_norm
            actual_to_file = to_col_norm

        actual_to_square = chess.square(actual_to_file, actual_to_rank)
        move = chess.Move(actual_from_square, actual_to_square, promotion=promotion_piece)

        if move in board.legal_moves:
            return move
        return None

    # 2. Knight Moves (channels 56-63)
    elif 56 <= channel <= 63:
        offset_idx = channel - 56
        if offset_idx >= len(KNIGHT_OFFSETS_LIST): return None

        dr_norm, df_norm = KNIGHT_OFFSETS_LIST[offset_idx]
        to_row_norm = from_row_norm + dr_norm
        to_col_norm = from_col_norm + df_norm

        if not (0 <= to_row_norm <= 7 and 0 <= to_col_norm <= 7): return None
        
        if board.turn == chess.WHITE:
            actual_to_rank = 7 - to_row_norm
            actual_to_file = to_col_norm
        else:
            actual_to_rank = to_row_norm
            actual_to_file = to_col_norm

        actual_to_square = chess.square(actual_to_file, actual_to_rank)
        move = chess.Move(actual_from_square, actual_to_square)

        if move in board.legal_moves:
            return move
        return None

    # 3. Queen-like (Sliding) Moves (channels 0-55)
    elif 0 <= channel <= 55:
        dir_idx = channel // 7
        distance = (channel % 7) + 1

        if dir_idx >= len(SLIDING_DIRS_LIST): return None

        sr_norm, sf_norm = SLIDING_DIRS_LIST[dir_idx]
        dr_norm, df_norm = sr_norm * distance, sf_norm * distance
        
        to_row_norm = from_row_norm + dr_norm
        to_col_norm = from_col_norm + df_norm

        if not (0 <= to_row_norm <= 7 and 0 <= to_col_norm <= 7): return None
        
        if board.turn == chess.WHITE:
            actual_to_rank = 7 - to_row_norm
            actual_to_file = to_col_norm
        else:
            actual_to_rank = to_row_norm
            actual_to_file = to_col_norm
        
        actual_to_square = chess.square(actual_to_file, actual_to_rank)
        move = chess.Move(actual_from_square, actual_to_square)

        # Handle Queen promotion for pawn moves
        moving_piece = board.piece_at(actual_from_square)
        if moving_piece and moving_piece.piece_type == chess.PAWN:
            if (board.turn == chess.WHITE and actual_to_rank == 7) or \
               (board.turn == chess.BLACK and actual_to_rank == 0):
                move.promotion = chess.QUEEN

        if move in board.legal_moves:
            return move
        return None
    
    return None


@cython.cdivision(True)
cpdef inline int policy_components_to_flat_index(int from_row, int from_col, int channel):
    """Converts a (from_row, from_col, channel) tuple into a single integer index."""
    cdef int index
    index = from_row * (_BOARD_DIM * _POLICY_CHANNELS) + \
            from_col * _POLICY_CHANNELS + \
            channel
    
    return index


@cython.cdivision(True)
cpdef inline tuple policy_flat_index_to_components(int flat_index):
    """Converts a single integer index back into its (from_row, from_col, channel) tuple."""
    cdef int channel, remaining, from_col, from_row
    
    channel = flat_index % _POLICY_CHANNELS
    
    remaining = flat_index // _POLICY_CHANNELS
    from_col = remaining % _BOARD_DIM
    
    from_row = remaining // _BOARD_DIM
    
    return from_row, from_col, channel


# Using 'object' for PyTorch tensors, as we don't have a C-API cimport for torch
cpdef object policy_components_to_flat_index_torch(object from_row_tensor, 
                                                    object from_col_tensor, 
                                                    object channel_tensor):
    """Converts batched (from_row, from_col, channel) tensors into a single flat integer index tensor."""
    
    from_row_tensor = from_row_tensor.long()
    from_col_tensor = from_col_tensor.long()
    channel_tensor = channel_tensor.long()

    # Use C constants for strides (they are imported from pxd)
    cdef int col_stride = _POLICY_CHANNELS
    cdef int row_stride = _BOARD_DIM * _POLICY_CHANNELS

    flat_index_tensor = (from_row_tensor * row_stride) + \
                        (from_col_tensor * col_stride) + \
                        channel_tensor
    
    return flat_index_tensor


@cython.cdivision(True)
cpdef tuple policy_flat_index_to_components_torch(object flat_index_tensor):
    """Converts a 1D tensor of flat integer indices back into (from_row, from_col, channel) tensors."""
    
    flat_index_tensor = flat_index_tensor.long()

    if not torch.all((flat_index_tensor >= 0) & (flat_index_tensor < _TOTAL_POLICY_MOVES)):
        raise ValueError("Invalid flat_index_tensor contains values outside 0 to %d." % (_TOTAL_POLICY_MOVES-1))

    channel_tensor = flat_index_tensor % _POLICY_CHANNELS
    
    remaining_tensor = flat_index_tensor // _POLICY_CHANNELS
    from_col_tensor = remaining_tensor % _BOARD_DIM
    
    from_row_tensor = remaining_tensor // _BOARD_DIM
    
    return from_row_tensor, from_col_tensor, channel_tensor


cpdef cnp.ndarray get_legal_move_mask(object board):
    """Generates a 4672-length boolean array where True indicates a legal move."""
    cdef cnp.ndarray mask = np.zeros(_TOTAL_POLICY_MOVES, dtype=np.bool_)
    cdef list legal_moves = list(board.legal_moves)
    cdef int from_row, from_col, channel, flat_index
    cdef object move
    
    for move in legal_moves:
        from_row, from_col, channel = move_to_policy_components(move, board)
        flat_index = policy_components_to_flat_index(from_row, from_col, channel)
        mask[flat_index] = True
        
    return mask