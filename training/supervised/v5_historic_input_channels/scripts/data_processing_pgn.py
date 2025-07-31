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
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
sys.path.insert(0, parent_dir)

log_dir = os.path.join(current_script_dir, "../logs/data_processing/lichess")
os.makedirs(log_dir, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
main_log_file_path = os.path.join(log_dir, f"main_data_processing_{timestamp}.log")

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

HDF5_CHUNK_SIZE = 1000 # Number of moves to buffer before writing to HDF5
MIN_GAME_MOVES = 30    # Minimum number of full moves for a game to be processed (60 ply)
MAX_MOVES_PER_WORKER = 625_000 # Each worker will process up to 4 million positions


def _extract_game_data(game, engine, stockfish_analysis_time, worker_logger):
    """
    Extracts input tensors, policy targets, and value targets for each move in a single game.
    Yields (input_tensor, policy_flat_index, current_player_value_target) for each move.
    """
    board = game.board()
    
    result = game.headers.get("Result")
    if result == "1-0":
        game_value_fallback = 1.0
    elif result == "0-1":
        game_value_fallback = -1.0
    else:
        game_value_fallback = 0.0

    for move_num, move in enumerate(game.mainline_moves()):
        input_tensor = utils.board_to_tensor_68(board)

        try:
            from_row, from_col, channel = utils.move_to_policy_components(move, board)
        except ValueError as e:
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
                worker_logger.warning(f"Stockfish engine error for move {move_num+1}: {e}. Falling back to game result value.")
                sf_eval_cp = None
            except Exception as e:
                worker_logger.error(f"Unexpected error during Stockfish analysis for move {move_num+1}: {e}. Falling back to game result value.")
                sf_eval_cp = None
        
        current_player_value_target = 0.0
        if sf_eval_cp is not None:
            normalized_eval = utils.centipawn_to_normalized_value(sf_eval_cp)
            current_player_value_target = normalized_eval if board.turn == chess.WHITE else -normalized_eval
        else:
            current_player_value_target = game_value_fallback if board.turn == chess.WHITE else -game_value_fallback

        yield input_tensor, policy_flat_index, current_player_value_target
        board.push(move)


def worker_process_files(
    worker_id,
    pgn_file_paths, # List of PGN files assigned to this worker
    output_hdf5_dir,
    min_elo_filter,
    stockfish_analysis_time,
    engine_path,
    max_moves_per_worker
):
    """
    Worker function to process a list of PGN files and save data to a single HDF5 file.
    Stops when max_moves_per_worker limit is reached.
    """
    worker_logger = logging.getLogger(f"worker_{worker_id}")
    worker_log_file_path = os.path.join(log_dir, f"worker_{worker_id}_log_{timestamp}.log")
    worker_logger.addHandler(logging.FileHandler(worker_log_file_path))
    worker_logger.setLevel(logging.INFO)

    input_tensors_batch = []
    policy_targets_batch = []
    value_targets_batch = []

    total_moves_processed_in_worker = 0
    games_skipped_elo = 0
    games_skipped_short = 0
    total_games_read = 0

    output_hdf5_path = os.path.join(output_hdf5_dir, f"worker_{worker_id}.h5")
    engine = None

    try:
        if engine_path:
            try:
                engine = chess.engine.SimpleEngine.popen_uci(engine_path)
                worker_logger.info(f"Worker {worker_id}: Stockfish engine initialized.")
                worker_logger.info(f"Worker {worker_id}: Stockfish analysis time limit per move: {stockfish_analysis_time} seconds.")
            except Exception as e:
                worker_logger.error(f"Worker {worker_id}: Failed to initialize Stockfish engine: {e}. Proceeding without engine analysis.")
                # If engine fails, set it to None so analysis is skipped
                engine = None 

        # Create HDF5 file and datasets in write mode ('w')
        with h5py.File(output_hdf5_path, 'w') as hf:
            dummy_board = chess.Board()
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
            worker_logger.info(f"Worker {worker_id}: Created new HDF5 file: {output_hdf5_path}")
            if min_elo_filter is not None:
                worker_logger.info(f"Worker {worker_id}: Filtering games: Both Elo >= {min_elo_filter}.")
            worker_logger.info(f"Worker {worker_id}: Filtering games: Min {MIN_GAME_MOVES} full moves.")
            worker_logger.info(f"Worker {worker_id}: Max moves to process for this worker: {max_moves_per_worker}")

            for pgn_file_path in pgn_file_paths:
                if total_moves_processed_in_worker >= max_moves_per_worker:
                    worker_logger.info(f"Worker {worker_id}: Reached {max_moves_per_worker} moves. Stopping processing new PGN files.")
                    break # Stop processing new PGN files

                worker_logger.info(f"Worker {worker_id}: Processing PGN file: {os.path.basename(pgn_file_path)}")
                
                with open(pgn_file_path, 'r', encoding='utf-8', errors='ignore') as pgn_file:
                    while True:
                        if total_moves_processed_in_worker >= max_moves_per_worker:
                            worker_logger.info(f"Worker {worker_id}: Reached {max_moves_per_worker} moves mid-file. Stopping.")
                            break # Stop processing moves within current PGN file

                        game = chess.pgn.read_game(pgn_file)
                        if game is None:
                            break # No more games in this PGN file

                        total_games_read += 1
                        
                        # --- Game Length Filter ---
                        # game.end().board().ply() gives total half-moves (plies)
                        # MIN_GAME_MOVES is in full moves, so multiply by 2 for plies
                        if game.end().board().ply() < (MIN_GAME_MOVES * 2):
                            games_skipped_short += 1
                            worker_logger.debug(f"Worker {worker_id}: Skipping game {total_games_read} (from {os.path.basename(pgn_file_path)}) due to short length (<{MIN_GAME_MOVES} moves).")
                            continue

                        # --- Elo Filter ---
                        if min_elo_filter is not None:
                            try:
                                white_elo = int(game.headers.get("WhiteElo", 0))
                                black_elo = int(game.headers.get("BlackElo", 0))

                                if white_elo < min_elo_filter or black_elo < min_elo_filter:
                                    games_skipped_elo += 1
                                    worker_logger.debug(f"Worker {worker_id}: Skipping game {total_games_read} (from {os.path.basename(pgn_file_path)}) (WhiteElo: {white_elo}, BlackElo: {black_elo}) due to Elo filter ({min_elo_filter}).")
                                    continue
                            except (ValueError, TypeError):
                                games_skipped_elo += 1
                                worker_logger.debug(f"Worker {worker_id}: Skipping game {total_games_read} (from {os.path.basename(pgn_file_path)}) due to unparseable Elo.")
                                continue

                        # --- Extract and store data for each move in the game ---
                        moves_in_game_processed = 0
                        for input_tensor, policy_flat_index, value_target in _extract_game_data(game, engine, stockfish_analysis_time, worker_logger):
                            if total_moves_processed_in_worker >= max_moves_per_worker:
                                worker_logger.info(f"Worker {worker_id}: Reached {max_moves_per_worker} moves. Stopping processing moves in current game.")
                                break # Stop processing moves in current game

                            input_tensors_batch.append(input_tensor)
                            policy_targets_batch.append(policy_flat_index)
                            value_targets_batch.append(value_target)
                            
                            total_moves_processed_in_worker += 1
                            moves_in_game_processed += 1

                            if len(input_tensors_batch) >= HDF5_CHUNK_SIZE:
                                current_len = boards_dset.shape[0]
                                boards_dset.resize(current_len + len(input_tensors_batch), axis=0)
                                policies_dset.resize(current_len + len(policy_targets_batch), axis=0)
                                values_dset.resize(current_len + len(value_targets_batch), axis=0)

                                boards_dset[current_len:] = np.array(input_tensors_batch, dtype=np.float16)
                                policies_dset[current_len:] = np.array(policy_targets_batch, dtype=np.int32)
                                values_dset[current_len:] = np.array(value_targets_batch, dtype=np.float16)

                                worker_logger.info(f"Worker {worker_id}: Saved {len(input_tensors_batch)} moves. Total in this HDF5: {boards_dset.shape[0]}. Overall processed: {total_moves_processed_in_worker}")

                                input_tensors_batch = []
                                policy_targets_batch = []
                                value_targets_batch = []
                        
                        if total_moves_processed_in_worker >= max_moves_per_worker:
                            break # Break from inner while loop (processing games in current PGN file)

            # Save any remaining data in the batch after all files/limits are processed
            if input_tensors_batch:
                current_len = boards_dset.shape[0]
                boards_dset.resize(current_len + len(input_tensors_batch), axis=0)
                policies_dset.resize(current_len + len(policy_targets_batch), axis=0)
                values_dset.resize(current_len + len(value_targets_batch), axis=0)

                boards_dset[current_len:] = np.array(input_tensors_batch, dtype=np.float16)
                policies_dset[current_len:] = np.array(policy_targets_batch, dtype=np.int32)
                values_dset[current_len:] = np.array(value_targets_batch, dtype=np.float16)

                worker_logger.info(f"Worker {worker_id}: Saved {len(input_tensors_batch)} remaining moves. Final total in this HDF5: {boards_dset.shape[0]}. Overall processed: {total_moves_processed_in_worker}")

    except Exception as e:
        worker_logger.error(f"Worker {worker_id}: CRITICAL ERROR processing files: {e}", exc_info=True)
        # Clean up partially created HDF5 file if an error occurred during processing
        if os.path.exists(output_hdf5_path):
            os.remove(output_hdf5_path)
        return output_hdf5_path, 0, e # Return error to main process

    finally:
        if engine:
            engine.quit()
            worker_logger.info(f"Worker {worker_id}: Stockfish engine terminated.")

    worker_logger.info(f"Worker {worker_id}: Finished. Games read: {total_games_read}, Skipped (Elo): {games_skipped_elo}, Skipped (Short): {games_skipped_short}, Moves processed: {total_moves_processed_in_worker}")
    
    return output_hdf5_path, total_moves_processed_in_worker, None


if __name__ == "__main__":
    pgn_data_directory = os.path.join(parent_dir, "../data", "lichess_elite_db")
    stockfish_engine_path = os.path.join(parent_dir, "../data/engine/stockfish/stockfish-windows-x86-64-avx2.exe")

    output_hdf5_dir = os.path.join(current_script_dir, "../data", "lichess_elite_db")
    os.makedirs(output_hdf5_dir, exist_ok=True)
    
    num_workers = os.cpu_count() or 4 # Use all available CPU cores, or default to 4

    logger.info(f"Starting parallel data processing from directory: {pgn_data_directory}")
    logger.info(f"Saving ONE HDF5 file per worker in: {output_hdf5_dir}")
    logger.info(f"Using {num_workers} parallel workers.")
    logger.info(f"Each worker will process up to {MAX_MOVES_PER_WORKER} moves.")
    logger.info(f"Total expected moves: {num_workers * MAX_MOVES_PER_WORKER}")


    all_pgn_files = sorted(glob.glob(os.path.join(pgn_data_directory, "*.pgn")), reverse=True)
    if not all_pgn_files:
        logger.warning(f"No PGN files found in: {pgn_data_directory}. Exiting.")
        sys.exit(0)
    
    logger.info(f"Found {len(all_pgn_files)} PGN files to process.")

    # Distribute PGN files among workers
    worker_pgn_assignments = [[] for _ in range(num_workers)]
    for i, pgn_file_path in enumerate(all_pgn_files):
        worker_pgn_assignments[i % num_workers].append(pgn_file_path)

    finished_files = []
    total_moves_processed_overall = 0

    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        futures = {}
        for i in range(num_workers):
            future = executor.submit(
                worker_process_files,
                i, # worker_id
                worker_pgn_assignments[i], # PGN files for this worker
                output_hdf5_dir,
                2600, # min_elo_filter (set as needed in this line if you want to filter)
                0.02, # stockfish_analysis_time (set as needed)
                stockfish_engine_path,
                MAX_MOVES_PER_WORKER
            )
            futures[future] = i # Store the worker_id with the future

        # Collect results as they complete
        for future in as_completed(futures):
            worker_id = futures[future]
            try:
                output_file, moves_count, error = future.result()
                if error:
                    logger.error(f"Worker {worker_id} failed to process files: {error}")
                else:
                    finished_files.append(output_file)
                    total_moves_processed_overall += moves_count
                    logger.info(f"Worker {worker_id} completed. Output: {os.path.basename(output_file)}, Moves: {moves_count}. Overall processed: {total_moves_processed_overall}")

            except Exception as exc:
                logger.error(f"An unexpected error occurred for worker {worker_id}: {exc}", exc_info=True)

    logger.info("Parallel data processing complete!")
    logger.info(f"Total individual HDF5 files created: {len(finished_files)}")
    logger.info(f"Overall total moves processed: {total_moves_processed_overall}")
    logger.info(f"Individual HDF5 files are located in: {output_hdf5_dir}")
    logger.info(f"You can now combine these files later using a separate script if needed.")