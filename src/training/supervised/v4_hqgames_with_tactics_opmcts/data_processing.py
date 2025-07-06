import numpy as np
import sys
import os
import chess
import chess.pgn
import chess.engine
import h5py
import glob
import logging
from datetime import datetime
import pandas as pd
import io

# --- Logging Configuration ---
current_script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, parent_dir)

log_dir = os.path.join(current_script_dir, "logs")
os.makedirs(log_dir, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file_path = os.path.join(log_dir, f"tactical_data_processing_{timestamp}.log")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file_path),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

import utils # Assuming utils module is correctly set up for board_to_tensor, move_to_policy_components, etc.

# Define a batch size for processing and saving to HDF5
HDF5_CHUNK_SIZE = 1000 # Keep this reasonable for memory management


def process_all_moves_from_tactical_csv(
    csv_file_path,
    output_hdf5_path="tactical_data_all_moves.h5",
    max_total_positions=None, # Now counts ALL positions (both players' moves)
    min_rating_filter=None,
    stockfish_analysis_time=0.01
):
    """
    Processes tactical puzzles from a CSV file, including ALL moves in the solution sequence
    (both the initiating player's moves and the opponent's responses).
    Saves positions and their corresponding moves/evaluations into a single HDF5 file.
    Filters puzzles by rating.
    Stops processing once max_total_positions is reached.

    Args:
        csv_file_path (str): Path to the input CSV file.
        output_hdf5_path (str): Path to the output HDF5 file.
        max_total_positions (int, optional): Maximum total board positions to process (from both players).
                                             Defaults to None (process all available positions).
        min_rating_filter (int, optional): Minimum rating for puzzles to include.
                                           Puzzles below this rating will be skipped.
                                           Defaults to None (no rating filtering).
        stockfish_analysis_time (float): Time limit for Stockfish analysis per position in seconds.
    """

    input_tensors_batch = []
    policy_targets_batch = []
    value_targets_batch = []

    total_positions_processed = 0 # Now counts successfully processed positions for ALL moves
    puzzles_skipped_rating = 0
    total_puzzles_read = 0

    logger.info(f"Reading CSV file: {csv_file_path}")
    try:
        df = pd.read_csv(csv_file_path, usecols=[0, 1, 2, 3], header=None)
        df.columns = ['PuzzleId', 'FEN', 'Moves', 'Rating'] 
    except FileNotFoundError:
        logger.error(f"CSV file not found: {csv_file_path}")
        return
    except Exception as e:
        logger.error(f"An error occurred while reading the CSV: {e}")
        return
    
    logger.info(f"Loaded {len(df)} puzzles from CSV.")

    # Initialize Stockfish engine
    engine_path = os.path.join(parent_dir, "data/engine/stockfish/stockfish-windows-x86-64-avx2.exe")
    if not os.path.exists(engine_path):
        logger.error(f"Stockfish engine not found at {engine_path}. Please check the path.")
        return
    
    engine = chess.engine.SimpleEngine.popen_uci(engine_path)
    logger.info("Stockfish engine initialized.")
    logger.info(f"Stockfish analysis time limit per position: {stockfish_analysis_time} seconds.")

    # --- Initialize HDF5 file and datasets ---
    with h5py.File(output_hdf5_path, 'w') as hf:
        dummy_board = chess.Board()
        dummy_input_tensor = utils.board_to_tensor(dummy_board)
        board_tensor_shape = dummy_input_tensor.shape

        policies_dset = hf.create_dataset(
            'policies',
            shape=(0,),
            maxshape=(None,),
            dtype=np.int32,
            compression='gzip',
            chunks=True
        )
        values_dset = hf.create_dataset(
            'values',
            shape=(0,),
            maxshape=(None,),
            dtype=np.float16,
            compression='gzip',
            chunks=True
        )
        boards_dset = hf.create_dataset(
            'inputs',
            shape=(0, *board_tensor_shape),
            maxshape=(None, *board_tensor_shape),
            dtype=np.float16,
            compression='gzip',
            chunks=True
        )

        logger.info(f"Opened HDF5 file: {output_hdf5_path} and created datasets.")
        if min_rating_filter is not None:
            logger.info(f"Filtering puzzles: Only including puzzles with Rating >= {min_rating_filter}.")

        for index, row in df.iterrows():
            if max_total_positions is not None and total_positions_processed >= max_total_positions:
                logger.info(f"Reached {max_total_positions} total positions. Stopping processing.")
                break # Break outer loop

            total_puzzles_read += 1
            
            puzzle_id = row['PuzzleId']
            fen_start = row['FEN']
            moves_str = row['Moves']
            rating = row['Rating']
            
            # --- Rating Filter ---
            if min_rating_filter is not None:
                try:
                    rating = int(rating)
                    if rating < min_rating_filter:
                        puzzles_skipped_rating += 1
                        logger.debug(f"Skipping puzzle {puzzle_id} (Rating: {rating}) due to rating filter ({min_rating_filter}).")
                        continue
                except ValueError:
                    logger.warning(f"Skipping puzzle {puzzle_id}: Could not parse rating '{rating}'.")
                    puzzles_skipped_rating += 1
                    continue

            # --- Puzzle Setup ---
            try:
                board = chess.Board(fen_start)
                
                # Parse all moves in the solution sequence
                solution_moves_uci = moves_str.split(' ')
                
                if not solution_moves_uci or solution_moves_uci[0] == '':
                    logger.warning(f"Skipping puzzle {puzzle_id}: No valid moves in solution string '{moves_str}'.")
                    continue

            except Exception as e:
                logger.warning(f"Skipping puzzle {puzzle_id} (FEN: {fen_start}, Moves: {moves_str}) due to initial parsing error: {e}")
                continue

            # --- Iterate through each position/move in the puzzle's solution ---
            # Loop will process position BEFORE move, then make move for next iteration
            for i, uci_move_str in enumerate(solution_moves_uci):
                if max_total_positions is not None and total_positions_processed >= max_total_positions:
                    logger.info(f"Reached {max_total_positions} total positions. Stopping processing.")
                    break # Break inner loop
                
                try:
                    current_move = chess.Move.from_uci(uci_move_str)
                    if current_move not in board.legal_moves:
                        logger.warning(f"Skipping move {uci_move_str} in puzzle {puzzle_id}: Illegal move from FEN '{board.fen()}'. Stopping processing this puzzle.")
                        break # Stop processing this puzzle if an illegal move is encountered
                except ValueError:
                    logger.warning(f"Skipping invalid UCI move string '{uci_move_str}' in puzzle {puzzle_id}. Stopping processing this puzzle.")
                    break # Stop processing this puzzle
                
                # *** Removed the `if board.turn == initial_turn:` condition ***
                # Now, all moves in the solution sequence are recorded as training data.
                
                # 1. Capture current board state
                input_tensor = utils.board_to_tensor(board)

                # 2. Determine the policy target (the move to be played from current position)
                try:
                    from_row, from_col, channel = utils.move_to_policy_components(current_move, board)
                except ValueError as e:
                    logger.warning(f"Skipping position/move in puzzle {puzzle_id} ({board.fen()} -> {current_move}): Policy encoding error: {e}")
                    board.push(current_move) # Still make the move to advance the board
                    continue 
                except Exception as e:
                    logger.error(f"UNEXPECTED ERROR: Skipping position/move in puzzle {puzzle_id} ({board.fen()} -> {current_move}): Unknown policy encoding error: {e}")
                    board.push(current_move)
                    continue

                policy_flat_index = utils.policy_components_to_flat_index(from_row, from_col, channel)
                
                # 3. Get Stockfish evaluation for the current board state (from the current player's perspective)
                sf_eval_cp = None
                try:
                    info = engine.analyse(board, chess.engine.Limit(time=stockfish_analysis_time))
                    score = info["score"].white() # Evaluation from White's perspective

                    if score.is_mate():
                        sf_eval_cp = 30000 if score.mate() > 0 else -30000 
                    else:
                        sf_eval_cp = score.cp
                except chess.engine.EngineError as e:
                    logger.warning(f"Stockfish engine error for puzzle {puzzle_id}, position {i+1} ({board.fen()}): {e}. Skipping Stockfish eval for this position.")
                    sf_eval_cp = None
                except Exception as e:
                    logger.error(f"Unexpected error during Stockfish analysis for puzzle {puzzle_id}, position {i+1} ({board.fen()}): {e}. Skipping Stockfish eval for this position.")
                    sf_eval_cp = None
                
                # Determine value target from the perspective of the player to move
                current_player_value_target = 0.0
                if sf_eval_cp is not None:
                    normalized_eval = utils.centipawn_to_normalized_value(sf_eval_cp)
                    # Value is from the perspective of the player whose turn it is
                    current_player_value_target = normalized_eval if board.turn == chess.WHITE else -normalized_eval
                else:
                    logger.debug(f"Using neutral value (0.0) for puzzle {puzzle_id}, position {i+1} (no Stockfish eval available).")
                    current_player_value_target = 0.0

                # Add processed data for this position (unconditionally now)
                input_tensors_batch.append(input_tensor)
                policy_targets_batch.append(policy_flat_index)
                value_targets_batch.append(current_player_value_target)

                total_positions_processed += 1 # Increment for ALL moves processed
                
                # Save batch to HDF5
                if len(input_tensors_batch) >= HDF5_CHUNK_SIZE:
                    current_len = boards_dset.shape[0]

                    boards_dset.resize(current_len + len(input_tensors_batch), axis=0)
                    policies_dset.resize(current_len + len(policy_targets_batch), axis=0)
                    values_dset.resize(current_len + len(value_targets_batch), axis=0)

                    boards_dset[current_len:] = np.array(input_tensors_batch, dtype=np.float16)
                    policies_dset[current_len:] = np.array(policy_targets_batch, dtype=np.int32)
                    values_dset[current_len:] = np.array(value_targets_batch, dtype=np.float16)

                    logger.info(f"    Saved {len(input_tensors_batch)} positions to HDF5. Total positions in HDF5: {boards_dset.shape[0]} / {max_total_positions or 'All'}")

                    input_tensors_batch = []
                    policy_targets_batch = []
                    value_targets_batch = []
                
                # Always make the move to advance the board for the next iteration
                board.push(current_move)
            
            # After inner loop for puzzle moves, check if max_total_positions was reached
            if max_total_positions is not None and total_positions_processed >= max_total_positions:
                break


        # Save any remaining data in the batch after all puzzles
        if input_tensors_batch:
            current_len = boards_dset.shape[0]
            boards_dset.resize(current_len + len(input_tensors_batch), axis=0)
            policies_dset.resize(current_len + len(policy_targets_batch), axis=0)
            values_dset.resize(current_len + len(value_targets_batch), axis=0)

            boards_dset[current_len:] = np.array(input_tensors_batch, dtype=np.float16)
            policies_dset[current_len:] = np.array(policy_targets_batch, dtype=np.int32)
            values_dset[current_len:] = np.array(value_targets_batch, dtype=np.float16)

            logger.info(f"Saved {len(input_tensors_batch)} remaining positions to HDF5. Total positions in HDF5: {boards_dset.shape[0]}")

    logger.info(f"Finished processing. Final total puzzles read: {total_puzzles_read}")
    if min_rating_filter is not None:
        logger.info(f"Puzzles skipped due to rating filter (< {min_rating_filter}): {puzzles_skipped_rating}")
    logger.info(f"Final total ALL positions processed (saved to HDF5): {total_positions_processed}")
    logger.info(f"Data saved to: {output_hdf5_path}")

    if engine:
        engine.quit()
        logger.info("Stockfish engine terminated.")


if __name__ == "__main__":
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Path to your tactical puzzles CSV file (e.g., from Lichess puzzles dataset)
    # Adjust this path based on where you store your CSV
    tactical_csv_file_path = os.path.join(parent_dir, "data", "lichess_tactics", "lichess_db_puzzle.csv") # Example path

    # Define the output HDF5 file path for tactical data
    output_hdf5_file_tactics = os.path.join(current_script_dir, "data", "tactical_puzzles_all_moves.h5") 

    # Set the maximum number of tactical positions to process
    max_total_moves_limit = 1000000 # Target 1M positions (now ALL moves)
    
    # Set the minimum puzzle rating filter
    min_puzzle_rating = 2420 

    logger.info(f"Starting tactical data processing from CSV: {tactical_csv_file_path}")
    logger.info(f"Saving processed tactical data (including all moves) to: {output_hdf5_file_tactics}")
    logger.info(f"Targeting a maximum of {max_total_moves_limit} total positions (all moves).")
    logger.info(f"Applying puzzle rating filter: only puzzles with Rating >= {min_puzzle_rating} will be included.")

    process_all_moves_from_tactical_csv(
        csv_file_path=tactical_csv_file_path,
        output_hdf5_path=output_hdf5_file_tactics,
        max_total_positions=max_total_moves_limit,
        min_rating_filter=min_puzzle_rating
    )

    logger.info("Tactical data processing complete!")