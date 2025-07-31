import numpy as np
import sys
import os
import chess
import chess.engine
import h5py
import glob
import logging
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
import random
import time

# --- Logging Configuration ---
current_script_dir = os.path.dirname(os.path.abspath(__file__))
# Adjust parent_dir to point to the root of your project where 'utils.py' is
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
sys.path.insert(0, parent_dir)

log_dir = os.path.join(current_script_dir, "../logs/data_processing/stockfish_endgames")
os.makedirs(log_dir, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
main_log_file_path = os.path.join(log_dir, f"stockfish_data_processing_{timestamp}.log")

logging.basicConfig(
    level=logging.INFO, # Keep main logger at INFO for overall progress
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(main_log_file_path),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Import utils after setting up sys.path
import utils

# --- Configuration Constants ---
HDF5_CHUNK_SIZE = 1000 # Number of moves to buffer before writing to HDF5 (applies to total positions recorded)
TOTAL_POSITIONS_TO_GENERATE = 1_000_000 # Total positions across all workers (roughly, as game length varies)
MAX_PIECES_FOR_ENDGAME = 7 # Max pieces for generated endgames (e.g., up to 7 pieces)
MIN_PIECES_FOR_ENDGAME = 3 # Min pieces for generated endgames (e.g., at least 3 pieces)
MAX_ATTEMPTS_PER_POSITION_INIT = 200 # Max attempts to generate a legal initial position
NUM_PRE_MOVES_TO_PLAY = 4 # Number of best legal moves to play to construct history
STOCKFISH_ANALYSIS_TIME = 0.02 # Stockfish analysis time per position
MAX_ENGINE_RESTARTS_PER_WORKER = 10 # Max times a worker will attempt to restart the engine
ENGINE_RESTART_DELAY = 2 # Seconds to wait before attempting an engine restart (increased)

# New: Stockfish engine options for each instance
STOCKFISH_THREADS = 1 # Use 1 thread per worker to reduce CPU contention
STOCKFISH_HASH_MB = 16 # Small hash table to reduce memory per worker (in MB)

# --- Piece types for random placement with weighted probability ---
# Order: Pawn > Rook > Bishop/Knight > Queen
# More repetitions = higher probability
WEIGHTED_PIECE_TYPES = [
    chess.PAWN, chess.PAWN, chess.PAWN, chess.PAWN, chess.PAWN, # High probability for pawns
    chess.ROOK, chess.ROOK, # Medium-high for rooks
    chess.BISHOP, chess.BISHOP, # Medium for bishops
    chess.KNIGHT, chess.KNIGHT, # Medium for knights
    chess.QUEEN # Low for queens
]

# All standard piece types (excluding kings, as they are placed separately)
PIECE_TYPES = [
    chess.PAWN, chess.ROOK, chess.KNIGHT, chess.BISHOP, chess.QUEEN
]

# --- Max counts for individual piece types (excluding kings) per side ---
MAX_PAWNS_PER_SIDE = 8 # Normal max
MAX_ROOKS_PER_SIDE = 2
MAX_BISHOPS_PER_SIDE = 2 # Max 2 bishops per side
MAX_KNIGHTS_PER_SIDE = 2
MAX_QUEENS_PER_SIDE = 1 # Prevent unrealistic multiple queens

# Pre-defined pawn ranks to avoid back-rank spawns
PAWN_RANKS = [
    chess.square(f, r)
    for r in range(1, 7)    # Ranks 2 to 7 (0-indexed 1 to 6)
    for f in range(8)      # Files a to h (0-indexed 0 to 7)
]
def _get_stockfish_analysis(board, stockfish_engine, worker_logger):
    """
    Gets Stockfish's best move and evaluation for a given board.
    Returns (best_move, centipawn_score) or (None, None) if analysis fails.
    """
    if not stockfish_engine:
        worker_logger.debug(f"Stockfish engine not available for board {board.fen()}.")
        return None, None

    try:
        if board.is_game_over():
            outcome = board.outcome()
            if outcome:
                if outcome.winner == chess.WHITE:
                    return None, 30000 # White wins
                elif outcome.winner == chess.BLACK:
                    return None, -30000 # Black wins
                else: # Draw
                    return None, 0
            return None, None

        info = stockfish_engine.analyse(board, chess.engine.Limit(time=STOCKFISH_ANALYSIS_TIME))
        
        # --- NEW DEBUGGING LOGGING FOR PV PARSING ---
        # Log the raw 'info' dictionary content for inspection
        if "pv" in info:
            worker_logger.debug(f"Raw PV received for board {board.fen()}: {info['pv']}")
        else:
            worker_logger.debug(f"No PV in info for board {board.fen()}. Full info: {info}")
        # --- END NEW DEBUGGING LOGGING ---

        best_move = None
        # The 'pv' key in 'info' should contain a list of chess.Move objects if parsing was successful.
        # The 'Exception parsing pv from info' message implies the issue happens internally
        # before this list is reliably formed, or if the list contains non-Move objects.
        if "pv" in info and info["pv"]:
            try:
                # Ensure the first element of pv is a chess.Move object
                best_move = info["pv"][0]
                if not isinstance(best_move, chess.Move):
                    worker_logger.debug(f"PV info contains non-Move object for board {board.fen()}. PV: {info['pv']}") # Changed to DEBUG
                    best_move = None
            except Exception as e:
                # This exception would catch if info["pv"] was not a list or similar unexpected structure
                worker_logger.debug(f"Error accessing or validating info['pv'][0] for board {board.fen()}. PV content: {info.get('pv', 'N/A')}. Error: {e}") # Changed to DEBUG
                best_move = None # If accessing or validating fails, treat as no best move found
        else:
            worker_logger.debug(f"Stockfish returned empty or missing PV for board {board.fen()}. Full info: {info}") # Changed to DEBUG
            # If Stockfish doesn't provide a PV, we treat it as no best move found
            best_move = None


        score = info["score"].white()

        if score.is_mate():
            centipawn_score = 30000 if score.mate() > 0 else -30000
        else:
            centipawn_score = score.cp
        
        return best_move, centipawn_score
    except chess.engine.EngineError as e:
        worker_logger.debug(f"Stockfish engine error for {board.fen()}: {e}. Marking engine as dead.") # Changed to DEBUG
        return None, None
    except Exception as e:
        worker_logger.debug(f"Unexpected error during Stockfish analysis for {board.fen()}: {e}. Skipping analysis.") # Changed to DEBUG
        return None, None

def _generate_and_play_game(
    stockfish_engine,
    worker_logger
):
    """
    Generates an initial chess position, plays NUM_PRE_MOVES_TO_PLAY optimal moves,
    and then plays the game to its conclusion using Stockfish's best moves.
    Yields (input_tensor, policy_flat_index, current_player_value_target) for each position in the game.
    Returns the count of positions generated for this game.
    """
    initial_attempts = 0
    board_found = False

    while initial_attempts < MAX_ATTEMPTS_PER_POSITION_INIT and not board_found:
        board = chess.Board()
        board.clear() # Start with an empty board

        # --- Piece Counts for current board generation attempt ---
        white_piece_counts = {p_type: 0 for p_type in PIECE_TYPES}
        black_piece_counts = {p_type: 0 for p_type in PIECE_TYPES}
        
        # Track bishop colors for each side: True for light square, False for dark square
        white_bishop_square_colors = []
        black_bishop_square_colors = []

        # Place kings first
        white_king_square = random.choice(chess.SQUARES)
        board.set_piece_at(white_king_square, chess.Piece(chess.KING, chess.WHITE))

        black_king_square = random.choice(chess.SQUARES)
        # Ensure black king is not too close to white king (min 2 squares apart)
        while chess.square_distance(white_king_square, black_king_square) < 2:
            black_king_square = random.choice(chess.SQUARES)
        board.set_piece_at(black_king_square, chess.Piece(chess.KING, chess.BLACK))

        # Randomly add other pieces until within MAX_PIECES_FOR_ENDGAME
        pieces_to_add_target = MAX_PIECES_FOR_ENDGAME - 2 # Target max additional pieces
        current_pieces_on_board = 2 # Kings
        
        # Iterate and add pieces one by one, respecting max limits and piece probabilities
        attempt_add_piece = 0
        while current_pieces_on_board < MAX_PIECES_FOR_ENDGAME and attempt_add_piece < 100: # Limit attempts to prevent infinite loops if impossible
            attempt_add_piece += 1

            piece_type = random.choice(WEIGHTED_PIECE_TYPES)
            color = random.choice([chess.WHITE, chess.BLACK])
            
            # Get current piece counts and bishop square colors tracker for the chosen side
            if color == chess.WHITE:
                current_counts = white_piece_counts
                bishop_colors_tracker = white_bishop_square_colors
            else: # Black
                current_counts = black_piece_counts
                bishop_colors_tracker = black_bishop_square_colors

            can_place_this_type = False
            if piece_type == chess.PAWN and current_counts[chess.PAWN] < MAX_PAWNS_PER_SIDE:
                can_place_this_type = True
            elif piece_type == chess.ROOK and current_counts[chess.ROOK] < MAX_ROOKS_PER_SIDE:
                can_place_this_type = True
            elif piece_type == chess.KNIGHT and current_counts[chess.KNIGHT] < MAX_KNIGHTS_PER_SIDE:
                can_place_this_type = True
            elif piece_type == chess.QUEEN and current_counts[chess.QUEEN] < MAX_QUEENS_PER_SIDE:
                can_place_this_type = True
            elif piece_type == chess.BISHOP and current_counts[chess.BISHOP] < MAX_BISHOPS_PER_SIDE:
                can_place_this_type = True
            
            if can_place_this_type:
                # Filter available squares based on piece type and bishop color constraint
                available_squares = []
                if piece_type == chess.PAWN:
                    squares_to_check = PAWN_RANKS # Only pawn ranks (2-7)
                else:
                    squares_to_check = chess.SQUARES # Any square for other pieces

                for s in squares_to_check:
                    # Ensure square is empty and not a king's square
                    if board.piece_at(s) is None and s != white_king_square and s != black_king_square:
                        if piece_type == chess.BISHOP:
                            # If we already have a bishop of this color, ensure the new one is on the opposite square color
                            if current_counts[chess.BISHOP] == 1:
                                existing_bishop_color = bishop_colors_tracker[0] # The color of the square of the first bishop
                                current_square_is_light = (chess.BB_LIGHT_SQUARES & chess.BB_SQUARES[s]) != 0
                                if current_square_is_light != existing_bishop_color: # Check if the square color is different
                                    available_squares.append(s)
                            else: # If no bishops yet, any color square is fine for the first bishop
                                available_squares.append(s)
                        else: # For non-bishops, just check emptiness
                            available_squares.append(s)

                if available_squares:
                    square = random.choice(available_squares)
                    board.set_piece_at(square, chess.Piece(piece_type, color))
                    current_counts[piece_type] += 1
                    current_pieces_on_board += 1
                    if piece_type == chess.BISHOP:
                        # Record the square color (True for light, False for dark) of the newly placed bishop
                        bishop_colors_tracker.append((chess.BB_LIGHT_SQUARES & chess.BB_SQUARES[square]) != 0)
                # else: No available square for this piece type that meets constraints, try another in the next iteration
            # else: Cannot place this piece type due to max limit, try another piece in next iteration

        # Ensure piece count is within range (after all attempts to add pieces)
        piece_count = sum(1 for _ in board.piece_map())
        if not (MIN_PIECES_FOR_ENDGAME <= piece_count <= MAX_PIECES_FOR_ENDGAME):
            initial_attempts += 1
            continue

        # Ensure the generated position is legal
        if not board.is_valid():
            initial_attempts += 1
            continue
        
        # Randomly set whose turn it is
        board.turn = random.choice([chess.WHITE, chess.BLACK])
        board_found = True

    if not board_found:
        worker_logger.debug(f"Could not generate a legal initial endgame board after {MAX_ATTEMPTS_PER_POSITION_INIT} attempts. Skipping game generation.") # Changed to DEBUG
        return # Return nothing, not 0, if no game generated

    current_game_positions = [] # To store all positions for this game
    
    # --- Play NUM_PRE_MOVES_TO_PLAY "optimal" legal moves to build history ---
    history_gen_successful = True
    for move_num in range(NUM_PRE_MOVES_TO_PLAY):
        if board.is_game_over(): # Game might end even during pre-moves
            break

        if not board.legal_moves:
            history_gen_successful = False
            worker_logger.debug(f"History gen failed: No legal moves for board {board.fen()}. Skipping this game.") # Changed to DEBUG
            break

        best_move_sf, _ = _get_stockfish_analysis(board, stockfish_engine, worker_logger)
        
        if best_move_sf is None: # Stockfish failed to provide a move (engine error or no move found)
            history_gen_successful = False
            worker_logger.debug(f"Stockfish failed to provide a best move for history generation at step {move_num+1} for board {board.fen()}. Signalling engine issue to worker.") # Changed to DEBUG
            # Important: Raise EngineError here to signal to the worker process that the engine died
            raise chess.engine.EngineError("Stockfish engine died during history generation.")

        try:
            board.push(best_move_sf)
        except ValueError:
            worker_logger.debug(f"History gen failed: Error pushing move {best_move_sf.uci()} for board {board.fen()}. Skipping this game.") # Changed to DEBUG
            history_gen_successful = False
            break
            
    if not history_gen_successful:
        return # Return nothing if history generation failed

    # --- Play the game to the end, recording each position ---
    game_play_successful = True
    
    while not board.is_game_over():
        # Get Stockfish analysis for the current board state
        best_move_for_policy, sf_eval_cp = _get_stockfish_analysis(board, stockfish_engine, worker_logger)
        
        if sf_eval_cp is None or best_move_for_policy is None:
            worker_logger.debug(f"Stockfish analysis failed during game play for board {board.fen()}. Terminating this game early and signalling engine issue.") # Changed to DEBUG
            game_play_successful = False
            # This is where we need to tell the calling worker to restart the engine
            raise chess.engine.EngineError("Stockfish engine died during game play.") 

        # Convert board to input tensor
        input_tensor = utils.board_to_tensor_68(board)

        # Value target: Normalized Stockfish centipawn score
        normalized_eval = utils.centipawn_to_normalized_value(sf_eval_cp)
        # Value target should always be from the perspective of the player to move
        current_player_value_target = normalized_eval if board.turn == chess.WHITE else -normalized_eval

        # Policy target: One-hot encoding of Stockfish's best move
        try:
            from_row, from_col, channel = utils.move_to_policy_components(best_move_for_policy, board)
            policy_flat_index = utils.policy_components_to_flat_index(from_row, from_col, channel)
            current_game_positions.append((input_tensor, policy_flat_index, current_player_value_target))
        except ValueError:
            worker_logger.debug(f"Skipping best move {best_move_for_policy.uci()} for policy generation due to encoding error. Board: {board.fen()}. Terminating this game early.") # Changed to DEBUG
            game_play_successful = False
            break

        # Make the move to advance the game
        try:
            board.push(best_move_for_policy)
        except ValueError:
            worker_logger.debug(f"Attempted to push illegal move {best_move_for_policy.uci()} during game play for board {board.fen()}. Terminating this game unexpectedly.") # Changed to DEBUG
            game_play_successful = False
            break
            
    for data_point in current_game_positions:
        yield data_point
    
    # Return nothing, as the caller can count yielded items if needed
    # The crucial part is that it either yields positions or raises an error.


def worker_process_stockfish_data(
    worker_id,
    num_positions_for_worker,
    output_hdf5_dir,
    stockfish_engine_path
):
    """
    Worker function to generate Stockfish endgame data (full games) and save it to a single HDF5 file.
    Includes robust engine re-initialization.
    """
    worker_logger = logging.getLogger(f"Stockfish Worker {worker_id}") # Changed logger name for clarity
    worker_log_file_path = os.path.join(log_dir, f"stockfish_worker_{worker_id}_log_{timestamp}.log")
    worker_logger.addHandler(logging.FileHandler(worker_log_file_path))
    worker_logger.setLevel(logging.INFO) # Set worker logger to DEBUG for detailed logs

    input_tensors_batch = []
    policy_targets_batch = []
    value_targets_batch = []

    total_positions_processed_in_worker = 0
    output_hdf5_path = os.path.join(output_hdf5_dir, f"stockfish_worker_{worker_id}.h5")
    
    stockfish_engine = None
    
    # Flag to indicate if the engine needs a restart *before* the next game attempt
    engine_needs_restart = True

    try:
        # Create HDF5 file and datasets
        with h5py.File(output_hdf5_path, 'w') as hf:
            dummy_board_for_shape = chess.Board()
            for _ in range(NUM_PRE_MOVES_TO_PLAY):
                if dummy_board_for_shape.legal_moves:
                    dummy_board_for_shape.push(random.choice(list(dummy_board_for_shape.legal_moves)))
                else:
                    break
            dummy_input_tensor = utils.board_to_tensor_68(dummy_board_for_shape)
            board_tensor_shape = dummy_input_tensor.shape

            boards_dset = hf.create_dataset(
                'inputs', shape=(0, *board_tensor_shape), maxshape=(None, *board_tensor_shape),
                dtype=np.float16, compression='gzip', chunks=True
            )
            policies_dset = hf.create_dataset(
                'policies', shape=(0,), maxshape=(None,),
                dtype=np.int32, compression='gzip', chunks=True
            )
            values_dset = hf.create_dataset(
                'values', shape=(0,), maxshape=(None,),
                dtype=np.float16, compression='gzip', chunks=True
            )
            worker_logger.info(f"Stockfish Worker {worker_id}: Created new HDF5 file: {output_hdf5_path}")
            worker_logger.info(f"Stockfish Worker {worker_id}: Attempting to generate approx. {num_positions_for_worker} positions (full games).")

            games_generated = 0
            while total_positions_processed_in_worker < num_positions_for_worker:
                # --- Engine health check and re-initialization for next game ---
                if engine_needs_restart:
                    if stockfish_engine: # If an old engine object exists, try to quit it cleanly
                        worker_logger.info(f"Stockfish Worker {worker_id}: Attempting to quit existing engine instance...")
                        try:
                            # Use a timeout for quitting
                            stockfish_engine.quit()
                            worker_logger.info(f"Stockfish Worker {worker_id}: Old Stockfish engine quit successfully.")
                        except chess.engine.EngineTerminatedError:
                            worker_logger.debug(f"Stockfish Worker {worker_id}: Old engine already terminated or failed to quit cleanly.") # Changed to DEBUG
                        except Exception as e:
                            worker_logger.debug(f"Stockfish Worker {worker_id}: Error terminating old Stockfish engine: {e}") # Changed to DEBUG
                        stockfish_engine = None # Ensure it's explicitly None
                        time.sleep(ENGINE_RESTART_DELAY) # Give OS some time to clean up
                        worker_logger.info(f"Stockfish Worker {worker_id}: Paused for {ENGINE_RESTART_DELAY}s before re-initializing.")

                    worker_logger.info(f"Stockfish Worker {worker_id}: Attempting to initialize/re-initialize Stockfish engine...")
                    restart_attempts = 0
                    while stockfish_engine is None and restart_attempts <= MAX_ENGINE_RESTARTS_PER_WORKER:
                        try:
                            if stockfish_engine_path and os.path.exists(stockfish_engine_path):
                                stockfish_engine = chess.engine.SimpleEngine.popen_uci(stockfish_engine_path)
                                # Set Stockfish options
                                stockfish_engine.configure({"Threads": STOCKFISH_THREADS, "Hash": STOCKFISH_HASH_MB})
                                worker_logger.info(f"Stockfish Worker {worker_id}: Stockfish engine initialized/re-initialized successfully. (Attempt {restart_attempts + 1})")
                                worker_logger.info(f"Stockfish Worker {worker_id}: Engine configured with Threads={STOCKFISH_THREADS}, Hash={STOCKFISH_HASH_MB}MB.")
                                engine_needs_restart = False # Engine is alive, no longer needs restart
                            else:
                                worker_logger.debug(f"Stockfish Worker {worker_id}: Stockfish engine path not found or invalid: '{stockfish_engine_path}'. Cannot initialize engine.") # Changed to DEBUG
                                break # Break from inner retry loop if path is bad
                        except Exception as e:
                            worker_logger.debug(f"Stockfish Worker {worker_id}: Failed to initialize/re-initialize Stockfish engine: {e}. Retrying in {ENGINE_RESTART_DELAY}s... (Attempt {restart_attempts + 1})") # Changed to DEBUG
                            stockfish_engine = None
                            restart_attempts += 1
                            time.sleep(ENGINE_RESTART_DELAY) # Delay before next retry
                    
                    if stockfish_engine is None:
                        worker_logger.critical(f"Stockfish Worker {worker_id}: Failed to initialize Stockfish engine after {MAX_ENGINE_RESTARTS_PER_WORKER} attempts. Cannot continue data generation. Terminating worker.")
                        break # Break out of the main while loop, stopping this worker
                
                # --- Generate and Play One Game ---
                if not engine_needs_restart: # Only try to generate if engine is confirmed to be running
                    current_game_position_count = 0
                    try:
                        game_positions_generator = _generate_and_play_game(stockfish_engine, worker_logger)
                        game_yielded_data = False
                        for input_tensor, policy_flat_index, value_target in game_positions_generator:
                            game_yielded_data = True
                            input_tensors_batch.append(input_tensor)
                            policy_targets_batch.append(policy_flat_index)
                            value_targets_batch.append(value_target)
                            
                            total_positions_processed_in_worker += 1
                            current_game_position_count += 1

                            if len(input_tensors_batch) >= HDF5_CHUNK_SIZE:
                                current_len = boards_dset.shape[0]
                                boards_dset.resize(current_len + len(input_tensors_batch), axis=0)
                                policies_dset.resize(current_len + len(policy_targets_batch), axis=0)
                                values_dset.resize(current_len + len(value_targets_batch), axis=0)

                                boards_dset[current_len:] = np.array(input_tensors_batch, dtype=np.float16)
                                policies_dset[current_len:] = np.array(policy_targets_batch, dtype=np.int32)
                                values_dset[current_len:] = np.array(value_targets_batch, dtype=np.float16)

                                worker_logger.info(f"Stockfish Worker {worker_id}: Saved {len(input_tensors_batch)} positions. Total in this HDF5: {boards_dset.shape[0]}. Overall processed: {total_positions_processed_in_worker}")

                                input_tensors_batch = []
                                policy_targets_batch = []
                                value_targets_batch = []
                        
                        if game_yielded_data:
                            games_generated += 1
                            worker_logger.info(f"Stockfish Worker {worker_id}: Generated {current_game_position_count} positions from one game. Total games: {games_generated}. Total positions: {total_positions_processed_in_worker}")

                    except chess.engine.EngineError as e:
                        worker_logger.debug(f"Stockfish Worker {worker_id}: Engine error during game generation: {e}. Signalling engine for restart.") # Changed to DEBUG
                        engine_needs_restart = True # This will trigger engine restart at the top of the next while loop iteration
                    except Exception as e:
                        worker_logger.debug(f"Stockfish Worker {worker_id}: Unexpected error during _generate_and_play_game (not engine error): {e}", exc_info=True) # Changed to DEBUG
                        # For non-engine errors, we assume the engine is still okay, but the game is aborted.
                        # We continue the main loop to try generating another game with the same engine.
                else: 
                    # This path is taken if engine_needs_restart was True and re-initialization failed.
                    # The critical error has been logged, and the main loop will eventually break.
                    pass 

            # Save any remaining data in the batch after all positions are generated
            if input_tensors_batch:
                current_len = boards_dset.shape[0]
                boards_dset.resize(current_len + len(input_tensors_batch), axis=0)
                policies_dset.resize(current_len + len(policy_targets_batch), axis=0)
                values_dset.resize(current_len + len(value_targets_batch), axis=0)

                boards_dset[current_len:] = np.array(input_tensors_batch, dtype=np.float16)
                policies_dset[current_len:] = np.array(policy_targets_batch, dtype=np.int32)
                values_dset[current_len:] = np.array(value_targets_batch, dtype=np.float16)

                worker_logger.info(f"Stockfish Worker {worker_id}: Saved {len(input_tensors_batch)} remaining positions. Final total in this HDF5: {boards_dset.shape[0]}. Overall processed: {total_positions_processed_in_worker}")

    except Exception as e:
        worker_logger.critical(f"Stockfish Worker {worker_id}: CRITICAL ERROR during worker processing: {e}", exc_info=True)
        if os.path.exists(output_hdf5_path):
            os.remove(output_hdf5_path) # Clean up partially created file if an error occurred during HDF5 ops
        return output_hdf5_path, 0, e

    finally:
        if stockfish_engine:
            try:
                stockfish_engine.quit(timeout=1.0) # Attempt to quit with timeout
                worker_logger.info(f"Stockfish Worker {worker_id}: Stockfish engine quit successfully on final cleanup.")
            except chess.engine.EngineTerminatedError:
                worker_logger.debug(f"Stockfish Worker {worker_id}: Stockfish engine already terminated on final cleanup.") # Changed to DEBUG
            except Exception as e:
                worker_logger.debug(f"Stockfish Worker {worker_id}: Error terminating Stockfish engine cleanly on final cleanup: {e}") # Changed to DEBUG

    worker_logger.info(f"Stockfish Worker {worker_id}: Finished. Positions processed: {total_positions_processed_in_worker}")
    
    return output_hdf5_path, total_positions_processed_in_worker, None


if __name__ == "__main__":
    # --- IMPORTANT: Configure this path ---
    # Make sure this path correctly points to your Stockfish executable.
    # On Windows, it's typically a .exe file. On Linux/macOS, it might be just 'stockfish'
    # within the stockfish folder.
    STOCKFISH_ENGINE_PATH = os.path.join(parent_dir, "../data", "engine", "stockfish", "stockfish-windows-x86-64-avx2.exe") # Adjust as needed

    output_hdf5_dir = os.path.join(current_script_dir, "../data", "stockfish_endgames")
    os.makedirs(output_hdf5_dir, exist_ok=True)
    
    num_workers = os.cpu_count() or 4 # Use all CPU cores by default, or 4 if cannot determine
    
    # CONSIDER REDUCING num_workers IF CRASHES PERSIST, especially if you have many cores
    # and Stockfish is still struggling even with 1 thread/instance.
    # For example, on a 16-core machine, you might try num_workers = 8 or even 4.
    # Each Stockfish instance will run on 1 thread with the new config.

    positions_per_worker = TOTAL_POSITIONS_TO_GENERATE // num_workers

    logger.info(f"Starting parallel Stockfish full-game data generation.")
    logger.info(f"Stockfish engine path: {STOCKFISH_ENGINE_PATH}")
    logger.info(f"Saving ONE HDF5 file per worker in: {output_hdf5_dir}")
    logger.info(f"Using {num_workers} parallel workers.")
    logger.info(f"Each worker will generate approximately {positions_per_worker} positions (from multiple full games).")
    logger.info(f"Total expected positions: {TOTAL_POSITIONS_TO_GENERATE}")
    logger.info(f"Generating games by starting from a random endgame, playing {NUM_PRE_MOVES_TO_PLAY} *Stockfish-optimal* legal moves to construct history, and then playing the game to completion using Stockfish's best moves.")
    logger.info(f"Value and Policy targets will be derived from Stockfish evaluations at each step.")
    logger.info(f"MAX_PIECES_FOR_ENDGAME: {MAX_PIECES_FOR_ENDGAME}")
    logger.info(f"Piece selection probability: Pawns > Rooks > Bishops/Knights > Queens")
    logger.info(f"Individual piece limits: Pawns: {MAX_PAWNS_PER_SIDE}/side, Queens: {MAX_QUEENS_PER_SIDE}/side, others: 2/side")
    logger.info(f"Max engine restarts per worker: {MAX_ENGINE_RESTARTS_PER_WORKER} with {ENGINE_RESTART_DELAY}s delay.")
    logger.info(f"Stockfish engine configured with Threads={STOCKFISH_THREADS}, Hash={STOCKFISH_HASH_MB}MB per instance.")


    if not os.path.exists(STOCKFISH_ENGINE_PATH):
        logger.error(f"Stockfish engine not found at: {STOCKFISH_ENGINE_PATH}. Please ensure the path is correct. Exiting.")
        sys.exit(1)

    finished_files = []
    total_moves_processed_overall = 0

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {}
        for i in range(num_workers):
            future = executor.submit(
                worker_process_stockfish_data,
                i,
                positions_per_worker,
                output_hdf5_dir,
                STOCKFISH_ENGINE_PATH
            )
            futures[future] = i

        for future in as_completed(futures):
            worker_id = futures[future]
            try:
                output_file, moves_count, error = future.result()
                if error:
                    logger.error(f"Stockfish Worker {worker_id} failed to process data: {error}")
                else:
                    finished_files.append(output_file)
                    total_moves_processed_overall += moves_count
                    logger.info(f"Worker {worker_id} completed. Generated {moves_count} positions.")
            except Exception as e:
                logger.critical(f"A worker process (ID: {worker_id}) encountered an unhandled exception: {e}", exc_info=True)
    
    logger.info(f"\n--- Data Generation Summary ---")
    logger.info(f"Total positions generated across all workers: {total_moves_processed_overall}")
    logger.info(f"Successfully generated files: {finished_files}")
    logger.info(f"All workers finished processing.")