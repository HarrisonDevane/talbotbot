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

cdef int _BOARD_DIM = 8
cdef int _INPUT_CHANNELS = 68
cdef int _TOTAL_INPUT_SIZE = 68 * 8 * 8
cdef int _POLICY_CHANNELS = 73
cdef int _TOTAL_POLICY_MOVES = 8 * 8 * 73

# Expose as Python globals
BOARD_DIM = _BOARD_DIM
INPUT_CHANNELS = _INPUT_CHANNELS
TOTAL_INPUT_SIZE = _TOTAL_INPUT_SIZE
POLICY_CHANNELS = _POLICY_CHANNELS
TOTAL_POLICY_MOVES = _TOTAL_POLICY_MOVES


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

# UPDATED: Now accepts orientation_color to handle "Me" vs "Opponent" logic
cdef cnp.ndarray _get_piece_planes(board_state: object, orientation_color: bint):
    """Helper for board_to_tensor_68. Fills planes relative to orientation_color."""
    cdef cnp.ndarray piece_planes = np.zeros((12, _BOARD_DIM, _BOARD_DIM), dtype=np.float32)
    cdef dict piece_to_plane = {
        chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2,
        chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5,
    }
    cdef int row, col, base_plane, plane_idx
    cdef object square, piece

    for square, piece in board_state.piece_map().items():
        # Standard Matrix coordinates (Rank 7 -> Row 0)
        row = 7 - chess.square_rank(square)
        col = chess.square_file(square)
        
        # Relative Logic: "Me" (0-5) vs "Opponent" (6-11)
        if piece.color == orientation_color:
            base_plane = 0 
        else:
            base_plane = 6 
        
        plane_idx = base_plane + piece_to_plane[piece.piece_type]
        piece_planes[plane_idx, row, col] = 1.0
        
    return piece_planes


cpdef cnp.ndarray board_to_tensor_68(object board):
    """
    Encode a python-chess Board into a (68, 8, 8) numpy float32 tensor.
    Fully Relative Representation with Spatial Invariance.
    """
    cdef int num_input_planes = 18 + (4 * 12) + 2 # = 68
    cdef cnp.ndarray planes = np.zeros((num_input_planes, _BOARD_DIM, _BOARD_DIM), dtype=np.float32)
    
    # --- Local C-Typed variables ---
    cdef int ep_file, start_plane_idx, i
    cdef object temp_board, hist_piece_planes, current_planes
    cdef bint us, them

    # 1. Current Board State (Planes 0-11)
    current_planes = _get_piece_planes(board, board.turn)
    planes[0:12, :, :] = current_planes

    # 2. Auxiliary Planes (12-17)
    planes[12, :, :] = 1.0 if board.turn == chess.WHITE else 0.0

    # Planes 13-16: Castling Rights (Relative + Spatial)
    us = board.turn
    them = not board.turn

    if board.turn == chess.WHITE:
        # White: Kingside is Right (Spatial), Queenside is Left
        # Ch 13/14 = Mine, Ch 15/16 = Theirs
        planes[13, :, :] = 1.0 if board.has_kingside_castling_rights(us) else 0.0
        planes[14, :, :] = 1.0 if board.has_queenside_castling_rights(us) else 0.0
        planes[15, :, :] = 1.0 if board.has_kingside_castling_rights(them) else 0.0
        planes[16, :, :] = 1.0 if board.has_queenside_castling_rights(them) else 0.0
    else:
        # Black: Queenside is Right (Spatial), Kingside is Left
        # We want Ch 13 to always mean "Castle Right"
        # So we map Black's Queenside rights to Ch 13
        planes[13, :, :] = 1.0 if board.has_queenside_castling_rights(us) else 0.0
        planes[14, :, :] = 1.0 if board.has_kingside_castling_rights(us) else 0.0
        planes[15, :, :] = 1.0 if board.has_queenside_castling_rights(them) else 0.0
        planes[16, :, :] = 1.0 if board.has_kingside_castling_rights(them) else 0.0

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

    # 5. Spatial Flip
    if board.turn == chess.BLACK:
        planes = np.flip(planes, axis=(1, 2)).copy()

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

    # Coordinate normalization (Explicit Geometry Logic)
    if board.turn == chess.WHITE:
        # White Perspective (Standard Matrix: Top-Left is a8)
        # Rank 7 -> Row 0, File 0 -> Col 0
        from_row_norm = 7 - from_rank
        from_col_norm = from_file
        to_row_norm = 7 - to_rank
        to_col_norm = to_file
    else:
        # Black Perspective (Rotated 180 Matrix: Top-Left is h1)
        # Rank 0 -> Row 0, File 7 -> Col 0
        # This matches np.flip(axis=(1,2)) used in input tensor
        from_row_norm = from_rank
        from_col_norm = 7 - from_file
        to_row_norm = to_rank
        to_col_norm = 7 - to_file

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
            direction_key = (0, 1) # East
        elif df == -2:
            direction_key = (0, -1) # West
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


# FIX: Changed return type from 'chess.Move' to 'object'
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
        # Inverse of (7 - rank) is (7 - row)
        actual_from_rank = 7 - from_row_norm
        actual_from_file = from_col_norm
    else:
        # Inverse of (rank) is (row)
        actual_from_rank = from_row_norm
        # Inverse of (7 - file) is (7 - col) -- FIX APPLIED
        actual_from_file = 7 - from_col_norm

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
            actual_to_file = 7 - to_col_norm # FIX APPLIED

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
            actual_to_file = 7 - to_col_norm # FIX APPLIED

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
            actual_to_file = 7 - to_col_norm # FIX APPLIED
        
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