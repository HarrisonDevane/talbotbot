import numpy as np
import sys
import os
import chess
import chess.engine
import h5py
import glob
import logging
from datetime import datetime
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed

# --- Logging Configuration ---
current_script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
sys.path.insert(0, parent_dir)

# Ensure log directory is specific to this tactical processing
log_dir = os.path.join(current_script_dir, "../logs/data_processing/tactical_puzzles")
os.makedirs(log_dir, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
main_log_file_path = os.path.join(log_dir, f"main_tactical_processing_{timestamp}.log")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(main_log_file_path),
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)

# Import utils after setting up sys.path
import utils

# Define constants for processing
HDF5_CHUNK_SIZE = 1000 # Number of moves to buffer before writing to HDF5
# Target 10M tactical positions in total, split across 8 workers
# So, 10,000,000 / 8 = 1,250,000 positions per worker
MAX_MOVES_PER_WORKER = 312_500 


def _extract_puzzle_data(row, engine, stockfish_analysis_time, worker_logger):
    """
    Extracts input tensors, policy targets, and value targets for each move in a single puzzle solution.
    Yields (input_tensor, policy_flat_index, current_player_value_target) for each move.
    """
    puzzle_id = row['PuzzleId']
    fen_start = row['FEN']
    moves_str = row['Moves']

    try:
        board = chess.Board(fen_start)
        solution_moves_uci = moves_str.split(' ')
        
        if not solution_moves_uci or solution_moves_uci[0] == '':
            worker_logger.debug(f"Skipping puzzle {puzzle_id}: No valid moves in solution string '{moves_str}'.")
            return # Exit this generator for the current puzzle

        # New: Check if there are at least 4 half-moves (2 full moves)
        if len(solution_moves_uci) < 4:
            worker_logger.debug(f"Skipping puzzle {puzzle_id}: Solution has less than 6 half-moves ({len(solution_moves_uci)}).")
            return

    except Exception as e:
        worker_logger.warning(f"Skipping puzzle {puzzle_id} (FEN: {fen_start}, Moves: {moves_str}) due to initial parsing error: {e}")
        return # Exit this generator for the current puzzle

    # New: Push the first 2 full moves (4 half-moves)
    for i in range(4):
        uci_move_str = solution_moves_uci[i]
        try:
            current_move = chess.Move.from_uci(uci_move_str)
            if current_move not in board.legal_moves:
                worker_logger.debug(f"Skipping puzzle {puzzle_id}: Illegal move '{uci_move_str}' during initial 3 full moves from FEN '{board.fen()}'. Stopping processing this puzzle.")
                return # Stop processing this puzzle if an illegal move is encountered
            board.push(current_move)
        except ValueError:
            worker_logger.debug(f"Skipping puzzle {puzzle_id}: Invalid UCI move string '{uci_move_str}' during initial 3 full moves. Stopping processing this puzzle.")
            return # Stop processing this puzzle

    # Continue processing moves *after* the initial 2 full moves
    for i in range(4, len(solution_moves_uci)):
        uci_move_str = solution_moves_uci[i]
        try:
            current_move = chess.Move.from_uci(uci_move_str)
            if current_move not in board.legal_moves:
                worker_logger.warning(f"Skipping move {uci_move_str} in puzzle {puzzle_id}: Illegal move from FEN '{board.fen()}'. Stopping processing this puzzle.")
                break # Stop processing this puzzle if an illegal move is encountered
        except ValueError:
            worker_logger.warning(f"Skipping invalid UCI move string '{uci_move_str}' in puzzle {puzzle_id}. Stopping processing this puzzle.")
            break # Stop processing this puzzle
        
        # 1. Capture current board state
        input_tensor = utils.board_to_tensor_68(board)

        # 2. Determine the policy target (the move to be played from current position)
        try:
            from_row, from_col, channel = utils.move_to_policy_components(current_move, board)
        except ValueError as e:
            worker_logger.warning(f"Skipping position/move in puzzle {puzzle_id} ({board.fen()} -> {current_move}): Policy encoding error: {e}")
            board.push(current_move) # Still make the move to advance the board
            continue 
        except Exception as e:
            worker_logger.error(f"UNEXPECTED ERROR: Skipping position/move in puzzle {puzzle_id} ({board.fen()} -> {current_move}): Unknown policy encoding error: {e}")
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
            worker_logger.warning(f"Stockfish engine error for puzzle {puzzle_id}, position {i+1} ({board.fen()}): {e}. Skipping Stockfish eval for this position.")
            sf_eval_cp = None
        except Exception as e:
            worker_logger.error(f"Unexpected error during Stockfish analysis for puzzle {puzzle_id}, position {i+1} ({board.fen()}): {e}. Skipping Stockfish eval for this position.")
            sf_eval_cp = None
        
        # Determine value target from the perspective of the player to move
        current_player_value_target = 0.0
        if sf_eval_cp is not None:
            normalized_eval = utils.centipawn_to_normalized_value(sf_eval_cp)
            # Value is from the perspective of the player whose turn it is
            current_player_value_target = normalized_eval if board.turn == chess.WHITE else -normalized_eval
        else:
            worker_logger.debug(f"Using neutral value (0.0) for puzzle {puzzle_id}, position {i+1} (no Stockfish eval available).")
            current_player_value_target = 0.0

        yield input_tensor, policy_flat_index, current_player_value_target
        
        # Always make the move to advance the board for the next iteration
        board.push(current_move)


def worker_process_tactical_csv_chunk(
    worker_id,
    csv_file_paths_for_worker, # List of CSV chunk file paths assigned to this worker
    output_hdf5_dir,
    min_rating_filter,
    stockfish_analysis_time,
    engine_path,
    max_moves_per_worker
):
    """
    Worker function to process assigned tactical CSV chunks and save data to a single HDF5 file.
    Stops when max_moves_per_worker limit is reached.
    """
    worker_logger = logging.getLogger(f"tactical_worker_{worker_id}")
    worker_log_file_path = os.path.join(log_dir, f"tactical_worker_{worker_id}_log_{timestamp}.log")
    worker_logger.addHandler(logging.FileHandler(worker_log_file_path))
    worker_logger.setLevel(logging.INFO)

    input_tensors_batch = []
    policy_targets_batch = []
    value_targets_batch = []

    total_positions_processed_in_worker = 0
    puzzles_skipped_rating = 0
    total_puzzles_read = 0

    output_hdf5_path = os.path.join(output_hdf5_dir, f"tactical_worker_{worker_id}.h5")
    engine = None

    try:
        if engine_path:
            try:
                engine = chess.engine.SimpleEngine.popen_uci(engine_path)
                worker_logger.info(f"Tactical Worker {worker_id}: Stockfish engine initialized.")
                worker_logger.info(f"Tactical Worker {worker_id}: Stockfish analysis time limit per position: {stockfish_analysis_time} seconds.")
            except Exception as e:
                worker_logger.error(f"Tactical Worker {worker_id}: Failed to initialize Stockfish engine: {e}. Proceeding without engine analysis.")
                engine = None 

        # Initialize HDF5 file and datasets
        with h5py.File(output_hdf5_path, 'w') as hf:
            dummy_board = chess.Board()
            # Ensure this uses the correct board_to_tensor function from utils
            dummy_input_tensor = utils.board_to_tensor_68(dummy_board) 
            board_tensor_shape = dummy_input_tensor.shape

            boards_dset = hf.create_dataset(
                'inputs',
                shape=(0, *board_tensor_shape),
                maxshape=(None, *board_tensor_shape),
                dtype=np.float16,
                compression='gzip',
                chunks=True
            )
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
            worker_logger.info(f"Tactical Worker {worker_id}: Created new HDF5 file: {output_hdf5_path}")
            if min_rating_filter is not None:
                worker_logger.info(f"Tactical Worker {worker_id}: Filtering puzzles: Rating >= {min_rating_filter}.")
            worker_logger.info(f"Tactical Worker {worker_id}: Max moves to process for this worker: {max_moves_per_worker}")

            for csv_file_path in csv_file_paths_for_worker:
                if total_positions_processed_in_worker >= max_moves_per_worker:
                    worker_logger.info(f"Tactical Worker {worker_id}: Reached {max_moves_per_worker} moves. Stopping processing new CSV chunks.")
                    break # Stop processing new CSV chunks

                worker_logger.info(f"Tactical Worker {worker_id}: Processing CSV chunk: {os.path.basename(csv_file_path)}")
                
                try:
                    # Read the CSV chunk. Assuming it does not have a header.
                    df_chunk = pd.read_csv(csv_file_path, usecols=[0, 1, 2, 3], header=None, on_bad_lines='skip', low_memory=False)
                    df_chunk.columns = ['PuzzleId', 'FEN', 'Moves', 'Rating']
                except FileNotFoundError:
                    worker_logger.error(f"Tactical Worker {worker_id}: CSV chunk not found: {csv_file_path}. Skipping.")
                    continue
                except Exception as e:
                    worker_logger.error(f"Tactical Worker {worker_id}: Error reading CSV chunk {csv_file_path}: {e}. Skipping.")
                    continue

                for index, row in df_chunk.iterrows():
                    if total_positions_processed_in_worker >= max_moves_per_worker:
                        worker_logger.info(f"Tactical Worker {worker_id}: Reached {max_moves_per_worker} moves mid-chunk. Stopping.")
                        break # Break inner loop (processing rows in current chunk)

                    total_puzzles_read += 1
                    
                    # --- Rating Filter ---
                    rating = row['Rating']
                    puzzle_id = row['PuzzleId'] # For logging
                    if min_rating_filter is not None:
                        try:
                            rating = int(rating)
                            if rating < min_rating_filter:
                                puzzles_skipped_rating += 1
                                worker_logger.debug(f"Tactical Worker {worker_id}: Skipping puzzle {puzzle_id} (Rating: {rating}) due to rating filter ({min_rating_filter}).")
                                continue
                        except ValueError:
                            worker_logger.warning(f"Tactical Worker {worker_id}: Skipping puzzle {puzzle_id}: Could not parse rating '{rating}'.")
                            puzzles_skipped_rating += 1
                            continue

                    # --- Extract and store data for each move in the puzzle's solution ---
                    for input_tensor, policy_flat_index, value_target in _extract_puzzle_data(row, engine, stockfish_analysis_time, worker_logger):
                        if total_positions_processed_in_worker >= max_moves_per_worker:
                            worker_logger.info(f"Tactical Worker {worker_id}: Reached {max_moves_per_worker} moves. Stopping processing moves in current puzzle.")
                            break # Stop processing moves in current puzzle

                        input_tensors_batch.append(input_tensor)
                        policy_targets_batch.append(policy_flat_index)
                        value_targets_batch.append(value_target)
                        
                        total_positions_processed_in_worker += 1

                        if len(input_tensors_batch) >= HDF5_CHUNK_SIZE:
                            current_len = boards_dset.shape[0]

                            boards_dset.resize(current_len + len(input_tensors_batch), axis=0)
                            policies_dset.resize(current_len + len(policy_targets_batch), axis=0)
                            values_dset.resize(current_len + len(value_targets_batch), axis=0)

                            boards_dset[current_len:] = np.array(input_tensors_batch, dtype=np.float16)
                            policies_dset[current_len:] = np.array(policy_targets_batch, dtype=np.int32)
                            values_dset[current_len:] = np.array(value_targets_batch, dtype=np.float16)

                            worker_logger.info(f"Tactical Worker {worker_id}: Saved {len(input_tensors_batch)} positions. Total in this HDF5: {boards_dset.shape[0]}. Overall processed: {total_positions_processed_in_worker}")

                            input_tensors_batch = []
                            policy_targets_batch = []
                            value_targets_batch = []
                    
                    if total_positions_processed_in_worker >= max_moves_per_worker:
                        break # Break from inner loop (processing rows/puzzles in current CSV chunk)

                if total_positions_processed_in_worker >= max_moves_per_worker:
                    break # Break from outer loop (processing CSV chunks)

            # Save any remaining data in the batch after all files/limits are processed
            if input_tensors_batch:
                current_len = boards_dset.shape[0]
                boards_dset.resize(current_len + len(input_tensors_batch), axis=0)
                policies_dset.resize(current_len + len(policy_targets_batch), axis=0)
                values_dset.resize(current_len + len(value_targets_batch), axis=0)

                boards_dset[current_len:] = np.array(input_tensors_batch, dtype=np.float16)
                policies_dset[current_len:] = np.array(policy_targets_batch, dtype=np.int32)
                values_dset[current_len:] = np.array(value_targets_batch, dtype=np.float16)

                worker_logger.info(f"Tactical Worker {worker_id}: Saved {len(input_tensors_batch)} remaining positions. Final total in this HDF5: {boards_dset.shape[0]}. Overall processed: {total_positions_processed_in_worker}")

    except Exception as e:
        worker_logger.error(f"Tactical Worker {worker_id}: CRITICAL ERROR processing files: {e}", exc_info=True)
        # Clean up partially created HDF5 file if an error occurred during processing
        if os.path.exists(output_hdf5_path):
            os.remove(output_hdf5_path)
        return output_hdf5_path, 0, e # Return error to main process

    finally:
        if engine:
            engine.quit()
            worker_logger.info(f"Tactical Worker {worker_id}: Stockfish engine terminated.")

    worker_logger.info(f"Tactical Worker {worker_id}: Finished. Puzzles read: {total_puzzles_read}, Skipped (Rating): {puzzles_skipped_rating}, Positions processed: {total_positions_processed_in_worker}")
    
    return output_hdf5_path, total_positions_processed_in_worker, None


if __name__ == "__main__":
    # Path to the directory containing your 8 split tactical puzzles CSV files
    # This should be the same directory where you saved the output from the splitting script
    TACTICAL_CSV_CHUNKS_DIR = os.path.join(parent_dir, "../data", "lichess_tactics")

    # Define the output HDF5 directory for tactical data
    output_hdf5_dir_tactics = os.path.join(current_script_dir, "../data", "processed_tactical_h5") 
    os.makedirs(output_hdf5_dir_tactics, exist_ok=True)
    
    num_workers = os.cpu_count() or 4 # Use all available CPU cores, or default to 4

    logger.info(f"Starting parallel tactical data processing from directory: {TACTICAL_CSV_CHUNKS_DIR}")
    logger.info(f"Saving ONE HDF5 file per worker in: {output_hdf5_dir_tactics}")
    logger.info(f"Using {num_workers} parallel workers.")
    logger.info(f"Each worker will process up to {MAX_MOVES_PER_WORKER} positions (all moves in solution).")
    logger.info(f"Total expected positions: {num_workers * MAX_MOVES_PER_WORKER}")

    # Find all the split CSV files
    # Assumes files are named 'lichess_db_puzzle_part_X_of_Y.csv'
    all_tactical_csv_chunks = sorted(glob.glob(os.path.join(TACTICAL_CSV_CHUNKS_DIR, "lichess_db_puzzle_part_*.csv")))
    
    if not all_tactical_csv_chunks:
        logger.warning(f"No tactical CSV chunk files found in: {TACTICAL_CSV_CHUNKS_DIR}. Please run the CSV splitting script first. Exiting.")
        sys.exit(0)
    
    logger.info(f"Found {len(all_tactical_csv_chunks)} tactical CSV chunks to process.")

    # Distribute CSV chunks among workers
    worker_csv_assignments = [[] for _ in range(num_workers)]
    for i, csv_file_path in enumerate(all_tactical_csv_chunks):
        worker_csv_assignments[i % num_workers].append(csv_file_path)

    finished_files = []
    total_positions_processed_overall = 0

    # Set the minimum puzzle rating filter for all workers
    min_puzzle_rating = None 
    stockfish_analysis_time = 0.02 # Stockfish analysis time per position

    # Stockfish engine path
    stockfish_engine_path = os.path.join(parent_dir, "../data/engine/stockfish/stockfish-windows-x86-64-avx2.exe")


    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {}
        for i in range(num_workers):
            # Only submit a task if there are files assigned to this worker
            if worker_csv_assignments[i]:
                future = executor.submit(
                    worker_process_tactical_csv_chunk,
                    i, # worker_id
                    worker_csv_assignments[i], # CSV chunk file paths for this worker
                    output_hdf5_dir_tactics,
                    min_puzzle_rating,
                    stockfish_analysis_time,
                    stockfish_engine_path,
                    MAX_MOVES_PER_WORKER
                )
                futures[future] = i # Store the worker_id with the future
            else:
                logger.info(f"Worker {i} has no CSV chunks assigned. Skipping submission.")


        # Collect results as they complete
        for future in as_completed(futures):
            worker_id = futures[future]
            try:
                output_file, positions_count, error = future.result()
                if error:
                    logger.error(f"Tactical Worker {worker_id} failed to process files: {error}")
                else:
                    finished_files.append(output_file)
                    total_positions_processed_overall += positions_count
                    logger.info(f"Tactical Worker {worker_id} completed. Output: {os.path.basename(output_file)}, Positions: {positions_count}. Overall processed: {total_positions_processed_overall}")

            except Exception as exc:
                logger.error(f"An unexpected error occurred for Tactical Worker {worker_id}: {exc}", exc_info=True)

    logger.info("Parallel tactical data processing complete!")
    logger.info(f"Total individual HDF5 files created: {len(finished_files)}")
    logger.info(f"Overall total positions processed: {total_positions_processed_overall}")
    logger.info(f"Individual HDF5 files are located in: {output_hdf5_dir_tactics}")
    logger.info(f"You can now combine these files later using a separate script if needed.")