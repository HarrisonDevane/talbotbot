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
from concurrent.futures import ProcessPoolExecutor, as_completed

# --- Logging Configuration ---
current_script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, parent_dir)

log_dir = os.path.join(current_script_dir, "logs")
os.makedirs(log_dir, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file_path = os.path.join(log_dir, f"main_data_processing_{timestamp}.log")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file_path),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

import utils

HDF5_CHUNK_SIZE = 1000

# This function will process a single PGN file
def process_single_pgn_file(
    pgn_file_path,
    output_hdf5_dir, # Directory for output HDF5 files
    worker_id, # Identifier for the worker process (for logging)
    min_elo_filter=None,
    stockfish_analysis_time=0.01,
    engine_path=None # Pass engine path to worker
):
    """
    Processes a single PGN file and saves its data to a unique HDF5 file.
    """
    # Create a worker-specific logger for this task
    worker_logger = logging.getLogger(f"worker_{worker_id}_file_{os.path.basename(pgn_file_path)}")
    # Configure worker-specific logger to output to a separate log file
    worker_log_file_path = os.path.join(log_dir, f"worker_{worker_id}_file_log_{os.path.basename(pgn_file_path).replace('.pgn', '')}_{timestamp}.log")
    worker_logger.addHandler(logging.FileHandler(worker_log_file_path))
    worker_logger.setLevel(logging.INFO) # Set level for worker logs

    input_tensors_batch = []
    policy_targets_batch = []
    value_targets_batch = []

    total_moves_processed_in_file = 0
    games_skipped_elo = 0
    total_games_read = 0

    # Determine the unique output HDF5 path for this PGN file
    base_pgn_name = os.path.splitext(os.path.basename(pgn_file_path))[0]
    output_hdf5_path = os.path.join(output_hdf5_dir, f"{base_pgn_name}.h5")

    engine = None
    if engine_path:
        try:
            engine = chess.engine.SimpleEngine.popen_uci(engine_path)
            worker_logger.info(f"Worker {worker_id} processing {os.path.basename(pgn_file_path)}: Stockfish engine initialized.")
            worker_logger.info(f"Worker {worker_id} processing {os.path.basename(pgn_file_path)}: Stockfish analysis time limit per move: {stockfish_analysis_time} seconds.")
        except Exception as e:
            worker_logger.error(f"Worker {worker_id} processing {os.path.basename(pgn_file_path)}: Failed to initialize Stockfish engine: {e}. Proceeding without engine analysis.")
            return output_hdf5_path, 0, e # Return error to main process

    try:
        # --- Create HDF5 file and datasets in write mode ('w') for this single PGN ---
        with h5py.File(output_hdf5_path, 'w') as hf: # Use 'w' mode for a new file for each PGN
            dummy_board = chess.Board()
            dummy_input_tensor = utils.board_to_tensor(dummy_board)
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
            worker_logger.info(f"Worker {worker_id} processing {os.path.basename(pgn_file_path)}: Created new HDF5 file: {output_hdf5_path}")
            if min_elo_filter is not None:
                worker_logger.info(f"Worker {worker_id} processing {os.path.basename(pgn_file_path)}: Filtering games: Both Elo >= {min_elo_filter}.")

            # Process games within this single PGN file
            with open(pgn_file_path, 'r', encoding='utf-8', errors='ignore') as pgn_file:
                while True:
                    game = chess.pgn.read_game(pgn_file)
                    if game is None:
                        break

                    total_games_read += 1
                    
                    if min_elo_filter is not None:
                        try:
                            white_elo = int(game.headers.get("WhiteElo", 0))
                            black_elo = int(game.headers.get("BlackElo", 0))

                            if white_elo < min_elo_filter or black_elo < min_elo_filter:
                                games_skipped_elo += 1
                                worker_logger.debug(f"Worker {worker_id} processing {os.path.basename(pgn_file_path)}: Skipping game {total_games_read} (WhiteElo: {white_elo}, BlackElo: {black_elo}) due to Elo filter ({min_elo_filter}).")
                                continue
                        except (ValueError, TypeError):
                            games_skipped_elo += 1
                            worker_logger.debug(f"Worker {worker_id} processing {os.path.basename(pgn_file_path)}: Skipping game {total_games_read} due to unparseable Elo in headers: {game.headers.get('WhiteElo')}/{game.headers.get('BlackElo')}")
                            continue

                    board = game.board()

                    result = game.headers.get("Result")
                    if result == "1-0":
                        game_value_fallback = 1.0
                    elif result == "0-1":
                        game_value_fallback = -1.0
                    else:
                        game_value_fallback = 0.0

                    for move_num, move in enumerate(game.mainline_moves()):
                        input_tensor = utils.board_to_tensor(board)

                        try:
                            from_row, from_col, channel = utils.move_to_policy_components(move, board)
                        except ValueError as e:
                            worker_logger.warning(f"Worker {worker_id} processing {os.path.basename(pgn_file_path)}: Skipping move {move_num+1} ({move}) in game {total_games_read} due to encoding error: {e}")
                            board.push(move)
                            continue
                        except Exception as e:
                            worker_logger.error(f"Worker {worker_id} processing {os.path.basename(pgn_file_path)}: UNEXPECTED ERROR: Skipping move {move_num+1} ({move}) in game {total_games_read} due to unknown error: {e}")
                            board.push(move)
                            continue

                        policy_flat_index = utils.policy_components_to_flat_index(from_row, from_col, channel)
                        
                        sf_eval_cp = None
                        if engine:
                            try:
                                info = engine.analyse(board, chess.engine.Limit(time=stockfish_analysis_time))
                                score = info["score"].white()

                                if score.is_mate():
                                    sf_eval_cp = 30000 if score.mate() > 0 else -30000
                                else:
                                    sf_eval_cp = score.cp
                            except chess.engine.EngineError as e:
                                worker_logger.warning(f"Worker {worker_id} processing {os.path.basename(pgn_file_path)}: Stockfish engine error for game {total_games_read}, move {move}: {e}. Falling back to game result value.")
                                sf_eval_cp = None
                            except Exception as e:
                                worker_logger.error(f"Worker {worker_id} processing {os.path.basename(pgn_file_path)}: Unexpected error during Stockfish analysis for game {total_games_read}, move {move}: {e}. Falling back to game result value.")
                                sf_eval_cp = None
                        
                        current_player_value_target = 0.0
                        if sf_eval_cp is not None:
                            normalized_eval = utils.centipawn_to_normalized_value(sf_eval_cp)
                            current_player_value_target = normalized_eval if board.turn == chess.WHITE else -normalized_eval
                        else:
                            current_player_value_target = game_value_fallback if board.turn == chess.WHITE else -game_value_fallback
                            worker_logger.debug(f"Worker {worker_id} processing {os.path.basename(pgn_file_path)}: Using game result for value target for game {total_games_read}, move {move} (no Stockfish eval available).")


                        input_tensors_batch.append(input_tensor)
                        policy_targets_batch.append(policy_flat_index)
                        value_targets_batch.append(current_player_value_target)

                        board.push(move)
                        total_moves_processed_in_file += 1

                        if len(input_tensors_batch) >= HDF5_CHUNK_SIZE:
                            current_len = boards_dset.shape[0]

                            boards_dset.resize(current_len + len(input_tensors_batch), axis=0)
                            policies_dset.resize(current_len + len(policy_targets_batch), axis=0)
                            values_dset.resize(current_len + len(value_targets_batch), axis=0)

                            boards_dset[current_len:] = np.array(input_tensors_batch, dtype=np.float16)
                            policies_dset[current_len:] = np.array(policy_targets_batch, dtype=np.int32)
                            values_dset[current_len:] = np.array(value_targets_batch, dtype=np.float16)

                            worker_logger.info(f"Worker {worker_id} processing {os.path.basename(pgn_file_path)}: Saved {len(input_tensors_batch)} moves. Total in this HDF5: {boards_dset.shape[0]}")

                            input_tensors_batch = []
                            policy_targets_batch = []
                            value_targets_batch = []

            if input_tensors_batch: # Save any remaining data in the batch
                current_len = boards_dset.shape[0]
                boards_dset.resize(current_len + len(input_tensors_batch), axis=0)
                policies_dset.resize(current_len + len(policy_targets_batch), axis=0)
                values_dset.resize(current_len + len(value_targets_batch), axis=0)

                boards_dset[current_len:] = np.array(input_tensors_batch, dtype=np.float16)
                policies_dset[current_len:] = np.array(policy_targets_batch, dtype=np.int32)
                values_dset[current_len:] = np.array(value_targets_batch, dtype=np.float16)

                worker_logger.info(f"Worker {worker_id} processing {os.path.basename(pgn_file_path)}: Saved {len(input_tensors_batch)} remaining moves. Final total in this HDF5: {boards_dset.shape[0]}")

    except Exception as e:
        worker_logger.error(f"Worker {worker_id} processing {os.path.basename(pgn_file_path)}: CRITICAL ERROR processing file: {e}")
        # Clean up partially created HDF5 file if an error occurred during processing
        if os.path.exists(output_hdf5_path):
            os.remove(output_hdf5_path)
        return output_hdf5_path, 0, e # Return error to main process

    finally:
        if engine:
            engine.quit()
            worker_logger.info(f"Worker {worker_id} processing {os.path.basename(pgn_file_path)}: Stockfish engine terminated.")

    worker_logger.info(f"Worker {worker_id} processing {os.path.basename(pgn_file_path)}: Finished. Games read: {total_games_read}, Skipped (Elo): {games_skipped_elo}, Moves processed: {total_moves_processed_in_file}")
    
    return output_hdf5_path, total_moves_processed_in_file, None # Return file path, count, and no error


if __name__ == "__main__":
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    pgn_data_directory = os.path.join(parent_dir, "data", "gm_games")
    stockfish_engine_path = os.path.join(parent_dir, "data/engine/stockfish/stockfish-windows-x86-64-avx2.exe")

    # Directory where individual HDF5 files will be saved
    output_hdf5_dir = os.path.join(current_script_dir, "data", "processed_pgn_h5")
    os.makedirs(output_hdf5_dir, exist_ok=True)
    
    # max_moves_limit has been removed, so this will process all available data.
    num_workers = os.cpu_count() or 4 # Use all available CPU cores, or default to 4

    logger.info(f"Starting parallel data processing from directory: {pgn_data_directory}")
    logger.info(f"Saving EACH processed PGN file to its own HDF5 file in: {output_hdf5_dir}")
    logger.info(f"Using {num_workers} parallel workers.")

    all_pgn_files = sorted(glob.glob(os.path.join(pgn_data_directory, "*.pgn")), reverse=True)
    if not all_pgn_files:
        logger.warning(f"No PGN files found in: {pgn_data_directory}. Exiting.")
        sys.exit(0)
    
    logger.info(f"Found {len(all_pgn_files)} PGN files to process.")

    finished_files = []
    total_moves_processed_overall = 0
    worker_id_counter = 0 # To assign unique IDs to tasks for logging

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {}
        # Submit a task for each PGN file
        for pgn_file_path in all_pgn_files:
            future = executor.submit(
                process_single_pgn_file,
                pgn_file_path,
                output_hdf5_dir,
                worker_id_counter % num_workers, # Recycle worker IDs for logging distinction
                None, # min_elo_filter (set as needed)
                0.01, # stockfish_analysis_time (set as needed)
                stockfish_engine_path
            )
            futures[future] = pgn_file_path # Store the original PGN path with the future
            worker_id_counter += 1

        # Collect results as they complete
        for future in as_completed(futures):
            pgn_original_path = futures[future]
            try:
                output_file, moves_count, error = future.result()
                if error:
                    logger.error(f"Failed to process {os.path.basename(pgn_original_path)}: {error}")
                else:
                    finished_files.append(output_file)
                    total_moves_processed_overall += moves_count
                    logger.info(f"Processed {os.path.basename(pgn_original_path)}. Output: {os.path.basename(output_file)}, Moves: {moves_count}. Total moves so far: {total_moves_processed_overall}")

            except Exception as exc:
                logger.error(f"An unexpected error occurred while processing {os.path.basename(pgn_original_path)}: {exc}")

    logger.info("Parallel data processing complete!")
    logger.info(f"Total individual HDF5 files created: {len(finished_files)}")
    logger.info(f"Overall total moves processed: {total_moves_processed_overall}")
    logger.info(f"Individual HDF5 files are located in: {output_hdf5_dir}")
    logger.info(f"You can now combine these files later using a separate script.")