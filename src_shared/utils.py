import chess
import numpy as np
import torch
import math

# --- Policy Head Channel Mappings ---

# Relative directions for Queen-like moves (channels 0-55)
# Order: N, NE, E, SE, S, SW, W, NW
SLIDING_DIRS_MAPPING = {
    (-1, 0): 0,  # N
    (-1, 1): 1,  # NE
    (0, 1): 2,   # E
    (1, 1): 3,   # SE
    (1, 0): 4,   # S
    (1, -1): 5,  # SW
    (0, -1): 6,  # W
    (-1, -1): 7  # NW
}
SLIDING_DIRS_LIST = list(SLIDING_DIRS_MAPPING.keys()) # For reverse lookup

# Relative offsets for Knight moves (channels 56-63)
# Order matters for consistent channel indexing
KNIGHT_OFFSETS_MAPPING = {
    (-2, -1): 0, (-2, 1): 1, (-1, -2): 2, (-1, 2): 3,
    (1, -2): 4, (1, 2): 5, (2, -1): 6, (2, 1): 7
}
KNIGHT_OFFSETS_LIST = list(KNIGHT_OFFSETS_MAPPING.keys()) # For reverse lookup

# Promotion pieces (for underpromotions, channels 64-72)
PROMOTION_PIECES_ORDER = [chess.KNIGHT, chess.BISHOP, chess.ROOK] # Queen promotions handled as sliding moves

# Relative pawn move type for underpromotions
# (col_diff for pawn advance)
PAWN_PROMO_MOVE_TYPES_MAPPING = {
    0: 0,   # Straight push
    -1: 1,  # Left capture
    1: 2    # Right capture
}

PAWN_PROMO_MOVE_TYPES_LIST = list(PAWN_PROMO_MOVE_TYPES_MAPPING.keys()) # For reverse lookup

BOARD_DIM = 8
POLICY_CHANNELS = 73 # This is the '73' in 8x8x73 (move types per square)
TOTAL_POLICY_MOVES = BOARD_DIM * BOARD_DIM * POLICY_CHANNELS # 8 * 8 * 73 = 4672

# --- Helper Functions ---

def convert_coords(rank, file):
    # Converts python-chess (rank, file) to (row, col) where (0,0) is a8 and (7,7) is h1
    # python-chess ranks: 0=rank 1, 7=rank 8
    # python-chess files: 0=file a, 7=file h
    return 7 - rank, file # row = 7 - rank, col = file

def board_to_tensor_18(board: chess.Board) -> np.ndarray:
    """
    Encode a python-chess Board into a (18, 8, 8) numpy float32 tensor.
    Planes:
      0-5: White pieces [P, N, B, R, Q, K]
      6-11: Black pieces [p, n, b, r, q, k]
      12:  White to move (all ones or zeros)
      13-16: Castling rights (Wk, Wq, Bk, Bq)
      17: En passant file (1 at file of ep square, else 0)
    """
    piece_map = board.piece_map()
    planes = np.zeros((18, 8, 8), dtype=np.float32)

    piece_to_plane = {
        chess.PAWN: 0,
        chess.KNIGHT: 1,
        chess.BISHOP: 2,
        chess.ROOK: 3,
        chess.QUEEN: 4,
        chess.KING: 5,
    }

    for square, piece in piece_map.items():
        row, col = convert_coords(chess.square_rank(square), chess.square_file(square))
        base_plane = 0 if piece.color == chess.WHITE else 6
        plane_idx = base_plane + piece_to_plane[piece.piece_type]
        planes[plane_idx, row, col] = 1.0

    # White to move
    planes[12, :, :] = 1.0 if board.turn == chess.WHITE else 0.0

    # Castling rights
    planes[13, :, :] = 1.0 if board.has_kingside_castling_rights(chess.WHITE) else 0.0
    planes[14, :, :] = 1.0 if board.has_queenside_castling_rights(chess.WHITE) else 0.0
    planes[15, :, :] = 1.0 if board.has_kingside_castling_rights(chess.BLACK) else 0.0
    planes[16, :, :] = 1.0 if board.has_queenside_castling_rights(chess.BLACK) else 0.0

    # En passant file
    if board.ep_square is not None:
        # The file is relative to the board, not the piece
        ep_file = chess.square_file(board.ep_square)
        # convert_coords isn't needed here as it's a file-wide plane
        planes[17, :, ep_file] = 1.0

    return planes

def _get_piece_planes(board_state: chess.Board) -> np.ndarray:
    piece_planes = np.zeros((12, BOARD_DIM, BOARD_DIM), dtype=np.float32)
    piece_to_plane = {
        chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2,
        chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5,
    }

    for square, piece in board_state.piece_map().items():
        row, col = convert_coords(chess.square_rank(square), chess.square_file(square))
        base_plane = 0 if piece.color == chess.WHITE else 6
        plane_idx = base_plane + piece_to_plane[piece.piece_type]
        piece_planes[plane_idx, row, col] = 1.0
    return piece_planes



def board_to_tensor_68(board: chess.Board) -> np.ndarray:
    """
    Encode a python-chess Board into a (68, 8, 8) numpy float32 tensor.
    Planes:
      0-5: White pieces [P, N, B, R, Q, K]
      6-11: Black pieces [p, n, b, r, q, k]
      12:  White to move (all ones or zeros)
      13-16: Castling rights (Wk, Wq, Bk, Bq)
      17: En passant file (1 at file of ep square, else 0)
      18-65: Last 4 historic half moves (2 full moves) - 4 * 12 = 48 planes
      66-67: Three move repetition counter (2 planes)
    """
    # 18 current planes + 4*12 historical piece planes + 2 repetition planes
    num_input_planes = 18 + (4 * 12) + 2 # = 68
    planes = np.zeros((num_input_planes, BOARD_DIM, BOARD_DIM), dtype=np.float32)

    # Current Board State (Planes 0-17)
    for square, piece in board.piece_map().items():
        row, col = convert_coords(chess.square_rank(square), chess.square_file(square))
        base_plane = 0 if piece.color == chess.WHITE else 6
        plane_idx = base_plane + {
            chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2,
            chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5,
        }[piece.piece_type]
        planes[plane_idx, row, col] = 1.0

    planes[12, :, :] = 1.0 if board.turn == chess.WHITE else 0.0
    planes[13, :, :] = 1.0 if board.has_kingside_castling_rights(chess.WHITE) else 0.0
    planes[14, :, :] = 1.0 if board.has_queenside_castling_rights(chess.WHITE) else 0.0
    planes[15, :, :] = 1.0 if board.has_kingside_castling_rights(chess.BLACK) else 0.0
    planes[16, :, :] = 1.0 if board.has_queenside_castling_rights(chess.BLACK) else 0.0

    if board.ep_square is not None:
        ep_file = chess.square_file(board.ep_square)
        planes[17, :, ep_file] = 1.0

    # Historical Board States (Planes 18-65) - Last 4 half-moves (2 full moves)
    temp_board = board.copy()
    # Loop 4 times for 4 half-moves (2 full moves)
    for i in range(4):
        if not temp_board.move_stack: # Check if there are moves to pop
            break
        temp_board.pop() # Go back one half-move
        hist_piece_planes = _get_piece_planes(temp_board)
        # Calculate start_plane_idx based on current historical plane iteration
        start_plane_idx = 18 + (i * 12)
        planes[start_plane_idx : start_plane_idx + 12, :, :] = hist_piece_planes

    # Repetition Channels (Planes 66-67)
    if board.is_repetition(count=2): # Checks if current position has appeared once before (2-fold)
        planes[66, :, :] = 1.0
    if board.is_repetition(count=3): # Checks if current position has appeared twice before (3-fold)
        planes[67, :, :] = 1.0


    if board.turn == chess.BLACK:
        # Spatially flip ALL 8x8 planes (axis 1 for rows/ranks, axis 2 for columns/files).
        planes = np.flip(planes, axis=(1, 2)).copy()

    return planes


def move_to_policy_components(move: chess.Move, board: chess.Board):
    """
    Converts a chess.Move object to a (row, col, channel) index for the 73x8x8 policy head.
    The (row, col) refers to the from_square of the move.
    """

    # These are the actual ranks and files
    from_rank, from_file = chess.square_rank(move.from_square), chess.square_file(move.from_square)
    to_rank, to_file = chess.square_rank(move.to_square), chess.square_file(move.to_square)

    # Convert to internal (row, col) representation (0,0 is a8) if white.
    # If black, do nothing, as we want the policy index to output from the perspective of 
    # The player's turn
    if board.turn == chess.WHITE:
        from_row_norm, from_col_norm = convert_coords(from_rank, from_file)
        to_row_norm, to_col_norm = convert_coords(to_rank, to_file)
    else:
        from_row_norm, from_col_norm = from_rank, from_file
        to_row_norm, to_col_norm = to_rank, to_file

    dr = to_row_norm - from_row_norm # row delta
    df = to_col_norm - from_col_norm # col delta

    # 1. Handle Underpromotions (Queen promotions are handled as normal sliding moves)
    if move.promotion and move.promotion != chess.QUEEN:
        if move.promotion not in PROMOTION_PIECES_ORDER:
            raise ValueError(f"Unexpected promotion piece: {move.promotion}")

        promo_piece_idx = PROMOTION_PIECES_ORDER.index(move.promotion)

        # Determine pawn move type (straight, left capture, right capture)
        # The 'col_diff' for black is inverted relative to standard board,
        # but the move.to_square - move.from_square will naturally give the correct
        # relative change on the board.
        # We need the horizontal component of the pawn move relative to its own file.
        # Example: white pawn e7xe8 becomes e8. If from_file=4, to_file=5, col_diff=1 (right capture)
        pawn_col_diff_type = df # This works directly for both white and black
                                # as we are measuring horizontal displacement.

        if pawn_col_diff_type not in PAWN_PROMO_MOVE_TYPES_MAPPING:
            raise ValueError(f"Unexpected pawn promotion horizontal displacement: {pawn_col_diff_type}")

        pawn_move_type_idx = PAWN_PROMO_MOVE_TYPES_MAPPING[pawn_col_diff_type]
        
        channel = 64 + (promo_piece_idx * 3) + pawn_move_type_idx
        return from_row_norm, from_col_norm, channel

    # 2. Handle Knight Moves
    if board.piece_at(move.from_square).piece_type == chess.KNIGHT:
        offset_key = (dr, df)
        if offset_key not in KNIGHT_OFFSETS_MAPPING:
            # This should ideally not happen for a legal knight move
            raise ValueError(f"Invalid knight move offset: {offset_key} for move {move}")
        channel = 56 + KNIGHT_OFFSETS_MAPPING[offset_key]
        return from_row_norm, from_col_norm, channel

    # 3. Handle Castling (treated as King moves for 2 squares)
    if board.is_castling(move):
        # Determine direction and distance for the King's actual move
        # King moves 2 squares horizontally during castling
        if df == 2: # Kingside castling
            direction_key = (0, 1) # East
            distance = 2
        elif df == -2: # Queenside castling
            direction_key = (0, -1) # West
            distance = 2
        else:
            raise ValueError(f"Unexpected castling move displacement: {move}")

        if direction_key not in SLIDING_DIRS_MAPPING:
            raise ValueError(f"Castling direction not found in SLIDING_DIRS_MAPPING: {direction_key}")

        dir_idx = SLIDING_DIRS_MAPPING[direction_key]
        channel = (dir_idx * 7) + (distance - 1)
        return from_row_norm, from_col_norm, channel


    # 4. Handle General Sliding Moves (Rook, Bishop, Queen, King, Pawn non-promo moves, En passant)
    # The 'from_square' contains the piece. Check piece type for appropriate handling.
    # King moves (non-castling) are also handled here as they are like "sliding" one square

    # Calculate direction and distance for sliding moves
    direction_key = (0,0)
    distance = 0

    if dr == 0: # Horizontal
        direction_key = (0, 1) if df > 0 else (0, -1)
        distance = abs(df)
    elif df == 0: # Vertical
        direction_key = (1, 0) if dr > 0 else (-1, 0) # Note: dr is row delta, +1 means moving DOWN (increasing row index)
        distance = abs(dr)
    elif abs(dr) == abs(df): # Diagonal
        if dr < 0 and df > 0: direction_key = (-1, 1) # NE
        elif dr < 0 and df < 0: direction_key = (-1, -1) # NW
        elif dr > 0 and df > 0: direction_key = (1, 1) # SE
        elif dr > 0 and df < 0: direction_key = (1, -1) # SW
        distance = abs(dr)
    else:
        # This case should be caught by Knight/Promotion checks if the move is legal.
        # If it's a legal move and reaches here, it implies a problem in logic.
        raise ValueError(f"Could not classify move {move} into policy head channels.")

    if distance < 1 or distance > 7:
        raise ValueError(f"Sliding move distance {distance} out of bounds (1-7) for move {move}")

    if direction_key not in SLIDING_DIRS_MAPPING:
        raise ValueError(f"Sliding direction {direction_key} not found in SLIDING_DIRS_MAPPING for move {move}")

    dir_idx = SLIDING_DIRS_MAPPING[direction_key]
    channel = (dir_idx * 7) + (distance - 1)
    return from_row_norm, from_col_norm, channel



def policy_components_to_move(from_row_norm: int, from_col_norm: int, channel: int, board: chess.Board) -> chess.Move | None:
    """
    Converts a (row, col, channel) index from the 73x8x8 policy head output back to a chess.Move.
    from_row_norm, from_col_norm are the (row, col) of the from_square in the normalized (player-agnostic) space.
    Returns None if the reconstructed move is not legal or mapping is invalid.
    """
    # Inverse of the coordinate normalization
    if board.turn == chess.WHITE:
        # For White, the normalized (from_row_norm) came from 7 - actual_rank
        # So, actual_rank = 7 - from_row_norm
        actual_from_rank = 7 - from_row_norm
        actual_from_file = from_col_norm # col_norm is always actual_file
    else: # board.turn == chess.BLACK
        # For Black, the normalized (from_row_norm) was directly the actual_rank
        # So, actual_rank = from_row_norm
        actual_from_rank = from_row_norm
        actual_from_file = from_col_norm

    actual_from_square = chess.square(actual_from_file, actual_from_rank)

    # 1. Underpromotions (channels 64-72)
    if 64 <= channel <= 72:
        relative_channel = channel - 64
        promo_piece_idx = relative_channel // 3
        pawn_move_type_idx = relative_channel % 3

        if promo_piece_idx >= len(PROMOTION_PIECES_ORDER):
            return None # Invalid promotion piece index
        promotion_piece = PROMOTION_PIECES_ORDER[promo_piece_idx]
        
        if pawn_move_type_idx >= len(PAWN_PROMO_MOVE_TYPES_LIST):
            return None # Invalid pawn move type index
        df_norm = PAWN_PROMO_MOVE_TYPES_LIST[pawn_move_type_idx] # This is the normalized col_diff

        # Normalized target row for promotion is always 0 (top of normalized board)
        to_row_norm = 0
        to_col_norm = from_col_norm + df_norm

        # Check bounds for to_col_norm
        if not (0 <= to_col_norm <= 7):
            return None
        
        # Inverse of the coordinate normalization for to_square
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
        if offset_idx >= len(KNIGHT_OFFSETS_LIST):
            return None # Invalid knight offset index

        dr_norm, df_norm = KNIGHT_OFFSETS_LIST[offset_idx] # Normalized deltas
        to_row_norm = from_row_norm + dr_norm
        to_col_norm = from_col_norm + df_norm

        # Check if normalized target square is on board
        if not (0 <= to_row_norm <= 7 and 0 <= to_col_norm <= 7):
            return None
        
        # Inverse of the coordinate normalization for to_square
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
        distance = (channel % 7) + 1 # +1 because distance is 1-indexed

        if dir_idx >= len(SLIDING_DIRS_LIST):
            return None # Invalid direction index

        sr_norm, sf_norm = SLIDING_DIRS_LIST[dir_idx] # Normalized step deltas
        dr_norm, df_norm = sr_norm * distance, sf_norm * distance # Total normalized deltas
        
        to_row_norm = from_row_norm + dr_norm
        to_col_norm = from_col_norm + df_norm

        # Check if normalized target square is on board
        if not (0 <= to_row_norm <= 7 and 0 <= to_col_norm <= 7):
            return None
        
        # Inverse of the coordinate normalization for to_square
        if board.turn == chess.WHITE:
            actual_to_rank = 7 - to_row_norm
            actual_to_file = to_col_norm
        else:
            actual_to_rank = to_row_norm
            actual_to_file = to_col_norm
        
        actual_to_square = chess.square(actual_to_file, actual_to_rank)
        move = chess.Move(actual_from_square, actual_to_square)

        # Handle Queen promotion for pawn moves ending on the promotion rank
        moving_piece = board.piece_at(actual_from_square) # Use actual_from_square
        if moving_piece and moving_piece.piece_type == chess.PAWN:
            if (board.turn == chess.WHITE and actual_to_rank == 7) or \
               (board.turn == chess.BLACK and actual_to_rank == 0): # Use actual_to_rank
                move.promotion = chess.QUEEN

        if move in board.legal_moves:
            return move
        return None
    
    return None # Invalid channel


def policy_components_to_flat_index(from_row: int, from_col: int, channel: int) -> int:
    """
    Converts a (from_row, from_col, channel) tuple representing a chess move
    into a single integer index (0 to 4671).

    Args:
        from_row (int): Row of the starting square (0-7).
        from_col (int): Column of the starting square (0-7).
        channel (int): Type of move from that square (0-72).

    Returns:
        int: A unique integer index corresponding to the move (0 to 4671).
    """
    if not (0 <= from_row < BOARD_DIM and
            0 <= from_col < BOARD_DIM and
            0 <= channel < POLICY_CHANNELS):
        raise ValueError(f"Invalid policy components: ({from_row}, {from_col}, {channel}). "
                         f"Rows/cols must be 0-{BOARD_DIM-1}, channel 0-{POLICY_CHANNELS-1}.")

    # The formula flattens the 3D coordinates into a 1D index
    # Think of it like converting a multi-digit number to a single number
    # (e.g., from_row is the 'hundreds' place, from_col is 'tens', channel is 'units')
    index = from_row * (BOARD_DIM * POLICY_CHANNELS) + \
            from_col * POLICY_CHANNELS + \
            channel
    
    return index


def policy_flat_index_to_components(flat_index: int) -> tuple[int, int, int]:
    """
    Converts a single integer index (0 to 4671) back into its
    (from_row, from_col, channel) tuple representation.

    Args:
        flat_index (int): A unique integer index corresponding to a move (0 to 4671).

    Returns:
        tuple[int, int, int]: The (from_row, from_col, channel) tuple.
    """
    if not (0 <= flat_index < TOTAL_POLICY_MOVES):
        raise ValueError(f"Invalid flat_index: {flat_index}. Must be 0 to {TOTAL_POLICY_MOVES-1}.")

    # Reverse the flattening logic
    channel = flat_index % POLICY_CHANNELS
    
    remaining = flat_index // POLICY_CHANNELS
    from_col = remaining % BOARD_DIM
    
    from_row = remaining // BOARD_DIM
    
    return from_row, from_col, channel




def policy_components_to_flat_index_torch(from_row_tensor: torch.Tensor, 
                                          from_col_tensor: torch.Tensor, 
                                          channel_tensor: torch.Tensor) -> torch.Tensor:
    """
    Converts batched (from_row, from_col, channel) tensors into a single flat integer index tensor.
    All input tensors must be on the same device.

    Args:
        from_row_tensor (torch.Tensor): Tensor of rows (e.g., shape [N], values 0-7).
        from_col_tensor (torch.Tensor): Tensor of columns (e.g., shape [N], values 0-7).
        channel_tensor (torch.Tensor): Tensor of channels (e.g., shape [N], values 0-72).

    Returns:
        torch.Tensor: A 1D tensor of unique integer indices.
    """
    # Ensure all inputs are long tensors for indexing
    from_row_tensor = from_row_tensor.long()
    from_col_tensor = from_col_tensor.long()
    channel_tensor = channel_tensor.long()

    # Precompute strides
    col_stride = POLICY_CHANNELS
    row_stride = BOARD_DIM * POLICY_CHANNELS

    flat_index_tensor = (from_row_tensor * row_stride) + \
                        (from_col_tensor * col_stride) + \
                        channel_tensor
    
    return flat_index_tensor


def policy_flat_index_to_components_torch(flat_index_tensor: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Converts a 1D tensor of flat integer indices back into
    (from_row, from_col, channel) tensors.

    Args:
        flat_index_tensor (torch.Tensor): A 1D tensor of unique integer indices.

    Returns:
        tuple[torch.Tensor, torch.Tensor, torch.Tensor]: Tensors for from_row, from_col, and channel.
    """
    # Ensure input is a long tensor
    flat_index_tensor = flat_index_tensor.long()

    # Total number of possible policy moves
    TOTAL_POLICY_MOVES = BOARD_DIM * BOARD_DIM * POLICY_CHANNELS # 8 * 8 * 73 = 4672

    if not torch.all((flat_index_tensor >= 0) & (flat_index_tensor < TOTAL_POLICY_MOVES)):
        raise ValueError(f"Invalid flat_index_tensor contains values outside 0 to {TOTAL_POLICY_MOVES-1}.")

    channel_tensor = flat_index_tensor % POLICY_CHANNELS
    
    remaining_tensor = flat_index_tensor // POLICY_CHANNELS
    from_col_tensor = remaining_tensor % BOARD_DIM
    
    from_row_tensor = remaining_tensor // BOARD_DIM
    
    return from_row_tensor, from_col_tensor, channel_tensor


def get_win_probability(value_output: torch.Tensor) -> float:
    """
    Converts the raw value_output (from tanh activation, range [-1, 1])
    into a win probability (range [0, 1]).
    
    Args:
        value_output: A scalar torch.Tensor output from the value head,
                        typically in the range [-1, 1].
                        -1 signifies a certain loss, 1 signifies a certain win,
                        0 signifies a draw.
    
    Returns:
        A float representing the estimated win probability for the current
        player, between 0 and 1.
    """
    # Ensure value_output is a scalar before calling .item()
    if value_output.numel() != 1:
        raise ValueError(f"Expected scalar tensor for value_output, but got shape {value_output.shape}")
        
    value_scalar = value_output.item()
    
    # Linear scaling from [-1, 1] to [0, 1]
    win_probability = (value_scalar + 1) / 2
    
    return win_probability


def centipawn_to_normalized_value(cp_eval, cp_scale_factor=350): # Adjusted default scale factor
    """
    Converts a centipawn evaluation (from White's perspective) to a normalized
    value between -1 (Black wins) and 1 (White wins) using tanh.
    A value of 0 indicates an equal position.

    Args:
        cp_eval (int): The centipawn evaluation from Stockfish.
        cp_scale_factor (float): A factor to scale the centipawns before applying tanh.
                                 Determines how steep the curve is.
                                 Higher values make the curve flatter (larger cp needed for strong score).
                                 Lower values make the curve steeper (smaller cp needed for strong score).
                                 Typical values for this mapping range from 200 to 600.
                                 A factor of 350-400 often works well, meaning ~400cp maps towards 0.7-0.8.
    Returns:
        float: Normalized value between -1 and 1.
    """
    if cp_eval is None:
        return 0.0

    # Handle mate scores: assign clear +1 or -1
    # Stockfish usually reports mate in N as a very high positive/negative number (e.g., 32767 for mate in 1)
    if cp_eval > 20000: # Use a threshold large enough to catch all mate scores
        return 1.0
    if cp_eval < -20000:
        return -1.0

    # Apply tanh
    return math.tanh(cp_eval / cp_scale_factor)