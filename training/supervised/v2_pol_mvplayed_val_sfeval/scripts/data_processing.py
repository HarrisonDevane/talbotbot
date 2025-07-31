import numpy as np
import sys
import os
import chess
import chess.pgn
import chess.engine
import h5py
import glob
import logging # Import the logging module
from datetime import datetime # For timestamping log files

# --- Logging Configuration ---
# Get the parent directory path for imports and log file location
current_script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, parent_dir)

# Ensure the logs directory exists
log_dir = os.path.join(current_script_dir, "logs")
os.makedirs(log_dir, exist_ok=True)

# Generate a timestamp for the log file
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file_path = os.path.join(log_dir, f"data_processing_{timestamp}.log")

# Configure the logger
logging.basicConfig(
    level=logging.INFO, # Set the minimum level of messages to capture (INFO, WARNING, ERROR, DEBUG)
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file_path), # Output to a file
        logging.StreamHandler(sys.stdout)   # Also output to console
    ]
)

# Use the logger instance
logger = logging.getLogger(__name__)

# Assuming utils.py now contains policy_components_to_flat_index
import utils

# Define a batch size for processing and saving to HDF5
HDF5_CHUNK_SIZE = 1000

def process_pgn_data_from_directory(
    pgn_directory_path,
    output_hdf5_path="processed_data.h5",
    max_total_moves=None,
    min_elo_filter=None ,
    stockfish_analysis_time=0.01
):
    """
    Processes PGN data from multiple files in a directory and saves it into a single HDF5 file.
    Filters games to include only those where both White and Black Elo are above min_elo_filter.
    Stops processing once max_total_moves is reached.

    Args:
        pgn_directory_path (str): Path to the directory containing PGN files.
        output_hdf5_path (str): Path to the output HDF5 file.
        max_total_moves (int, optional): Maximum total moves (positions) to process across all games.
                                         Defaults to None (process all available moves).
        min_elo_filter (int, optional): Minimum Elo rating for both White and Black players.
                                        Games with either player below this or missing Elo will be skipped.
                                        Defaults to None (no Elo filtering).
    """

    input_tensors_batch = []
    policy_targets_batch = []
    value_targets_batch = []

    total_moves_processed = 0
    games_skipped_elo = 0 # <-- NEW: Counter for skipped games
    total_games_read = 0  # <-- NEW: Counter for total games read

    # Get a list of all PGN files in the directory
    pgn_files = sorted(glob.glob(os.path.join(pgn_directory_path, "*.pgn")), reverse=True)
    if not pgn_files:
        logger.warning(f"No PGN files found in: {pgn_directory_path}")
        return
    
    engine = chess.engine.SimpleEngine.popen_uci(os.path.join(parent_dir, "data/engine/stockfish/stockfish-windows-x86-64-avx2.exe")) 
    print(engine)
    logger.info("Stockfish engine initialized.")
    logger.info(f"Stockfish analysis time limit per move: {stockfish_analysis_time} seconds.")


    # --- Initialize HDF5 file and datasets ---
    with h5py.File(output_hdf5_path, 'w') as hf:
        dummy_board = chess.Board()
        dummy_input_tensor = utils.board_to_tensor(dummy_board)
        board_tensor_shape = dummy_input_tensor.shape

        policies_dset = hf.create_dataset(
            'policies',
            shape=(0, ), # Simplified from (*policy_shape) as it's just a scalar
            maxshape=(None, ),
            dtype=np.int32,
            compression='gzip',
            chunks=True
        )
        values_dset = hf.create_dataset(
            'values',
            shape=(0, ), # Simplified from (*value_shape) as it's just a scalar
            maxshape=(None, ),
            dtype=np.float16,
            compression='gzip',
            chunks=True
        )
        # Re-create boards_dset with correct shape to avoid potential issues if it was ()
        boards_dset = hf.create_dataset(
            'inputs',
            shape=(0, *board_tensor_shape),
            maxshape=(None, *board_tensor_shape),
            dtype=np.float16,
            compression='gzip',
            chunks=True
        )

        logger.info(f"Opened HDF5 file: {output_hdf5_path} and created datasets.")
        logger.info(f"Processing PGN files from: {pgn_directory_path}")
        if min_elo_filter is not None:
            logger.info(f"Filtering games: Only including games where BOTH White and Black Elo are >= {min_elo_filter}.")

        for pgn_file_path in pgn_files:
            if max_total_moves is not None and total_moves_processed >= max_total_moves:
                logger.info(f"Reached {max_total_moves} moves. Stopping processing.")
                break

            logger.info(f"\nProcessing file: {os.path.basename(pgn_file_path)}")
            with open(pgn_file_path) as pgn_file:
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
                                logger.debug(f"Skipping game {total_games_read} (WhiteElo: {white_elo}, BlackElo: {black_elo}) due to Elo filter ({min_elo_filter}).")
                                continue
                        except (ValueError, TypeError):
                            games_skipped_elo += 1
                            logger.debug(f"Skipping game {total_games_read} due to unparseable Elo in headers: {game.headers.get('WhiteElo')}/{game.headers.get('BlackElo')}")
                            continue

                    if max_total_moves is not None and total_moves_processed >= max_total_moves:
                        break

                    board = game.board()

                    result = game.headers.get("Result")
                    if result == "1-0":
                        game_value_fallback = 1.0
                    elif result == "0-1":
                        game_value_fallback = -1.0
                    else:
                        game_value_fallback = 0.0

                    for move in game.mainline_moves():
                        if max_total_moves is not None and total_moves_processed >= max_total_moves:
                            break

                        input_tensor = utils.board_to_tensor(board)

                        try:
                            from_row, from_col, channel = utils.move_to_policy_components(move, board)
                        except ValueError as e:
                            logger.warning(f"Skipping move {move} in {os.path.basename(pgn_file_path)} (game {total_games_read}) due to encoding error: {e}")
                            board.push(move)
                            continue
                        except Exception as e:
                            logger.error(f"UNEXPECTED ERROR: Skipping move {move} in {os.path.basename(pgn_file_path)} (game {total_games_read}) due to unknown error: {e}")
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
                                logger.warning(f"Stockfish engine error for game {total_games_read}, move {move}: {e}. Falling back to game result value.")
                                sf_eval_cp = None
                            except Exception as e:
                                logger.error(f"Unexpected error during Stockfish analysis for game {total_games_read}, move {move}: {e}. Falling back to game result value.")
                                sf_eval_cp = None
                        
                        current_player_value_target = 0.0
                        if sf_eval_cp is not None:
                            normalized_eval = utils.centipawn_to_normalized_value(sf_eval_cp)
                            current_player_value_target = normalized_eval if board.turn == chess.WHITE else -normalized_eval
                        else:
                            current_player_value_target = game_value_fallback if board.turn == chess.WHITE else -game_value_fallback
                            logger.debug(f"Using game result for value target for game {total_games_read}, move {move} (no Stockfish eval available).")


                        input_tensors_batch.append(input_tensor)
                        policy_targets_batch.append(policy_flat_index)
                        value_targets_batch.append(current_player_value_target)

                        board.push(move)
                        total_moves_processed += 1

                        if len(input_tensors_batch) >= HDF5_CHUNK_SIZE:
                            current_len = boards_dset.shape[0]

                            boards_dset.resize(current_len + len(input_tensors_batch), axis=0)
                            policies_dset.resize(current_len + len(policy_targets_batch), axis=0)
                            values_dset.resize(current_len + len(value_targets_batch), axis=0)

                            boards_dset[current_len:] = np.array(input_tensors_batch, dtype=np.float16)
                            policies_dset[current_len:] = np.array(policy_targets_batch, dtype=np.int32)
                            values_dset[current_len:] = np.array(value_targets_batch, dtype=np.float16)

                            logger.info(f"    Saved {len(input_tensors_batch)} moves to HDF5. Total moves in HDF5: {boards_dset.shape[0]} / {max_total_moves or 'All'}")

                            input_tensors_batch = []
                            policy_targets_batch = []
                            value_targets_batch = []

            if max_total_moves is not None and total_moves_processed >= max_total_moves:
                break

        if input_tensors_batch:
            current_len = boards_dset.shape[0]
            boards_dset.resize(current_len + len(input_tensors_batch), axis=0)
            policies_dset.resize(current_len + len(policy_targets_batch), axis=0)
            values_dset.resize(current_len + len(value_targets_batch), axis=0)

            boards_dset[current_len:] = np.array(input_tensors_batch, dtype=np.float16)
            policies_dset[current_len:] = np.array(policy_targets_batch, dtype=np.int32)
            values_dset[current_len:] = np.array(value_targets_batch, dtype=np.float16)

            logger.info(f"Saved {len(input_tensors_batch)} remaining moves to HDF5. Total moves in HDF5: {boards_dset.shape[0]}")

    logger.info(f"Finished processing. Final total games read: {total_games_read}")
    if min_elo_filter is not None:
        logger.info(f"Games skipped due to Elo filter (< {min_elo_filter}): {games_skipped_elo}")
    logger.info(f"Final total moves processed (saved to HDF5): {total_moves_processed}")
    logger.info(f"Data saved to: {output_hdf5_path}")

    if engine:
        engine.quit()
        logger.info("Stockfish engine terminated.")


if __name__ == "__main__":
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    pgn_data_directory = os.path.join(parent_dir, "data", "lichess_elite_db")

    # Define the output HDF5 file path
    # This will create 'chess_data.h5' in your 'your_project/data/' directory
    output_hdf5_file = os.path.join(current_script_dir, "data", "chess_data.h5") # <-- NEW: Changed output file name

    # Set the maximum number of moves to process
    max_moves_limit = 5000000 # 5 million moves
    elo_threshold = 2600 # Set elo threshold

    logger.info(f"Starting data processing from directory: {pgn_data_directory}")
    logger.info(f"Saving processed data to: {output_hdf5_file}")
    logger.info(f"Targeting a maximum of {max_moves_limit} moves.")
    logger.info(f"Applying Elo filter: only games with both players >= {elo_threshold} will be included.")

    process_pgn_data_from_directory(
        pgn_directory_path=pgn_data_directory,
        output_hdf5_path=output_hdf5_file,
        max_total_moves=max_moves_limit,
        min_elo_filter=elo_threshold
    )

    logger.info("Data processing complete!")