import numpy as np
import sys, os
import chess
import chess.pgn
import h5py
import glob # Import glob for finding files

# Get the parent directory path to help with relative imports and paths
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.insert(0, parent_dir)

import utils # Assuming utils.py now contains policy_components_to_flat_index

# Define a batch size for processing and saving to HDF5
HDF5_CHUNK_SIZE = 10000 

def process_pgn_data_from_directory(
    pgn_directory_path,
    output_hdf5_path="processed_data.h5",
    max_total_moves=None
):
    """
    Processes PGN data from multiple files in a directory and saves it into a single HDF5 file.
    Stops processing once max_total_moves is reached.

    Args:
        pgn_directory_path (str): Path to the directory containing PGN files.
        output_hdf5_path (str): Path to the output HDF5 file.
        max_total_moves (int, optional): Maximum total moves (positions) to process across all games.
                                         Defaults to None (process all available moves).
    """

    input_tensors_batch = []
    policy_targets_batch = [] 
    value_targets_batch = []

    total_moves_processed = 0

    # Get a list of all PGN files in the directory
    pgn_files = sorted(glob.glob(os.path.join(pgn_directory_path, "*.pgn")))
    if not pgn_files:
        print(f"No PGN files found in: {pgn_directory_path}")
        return

    # --- Initialize HDF5 file and datasets ---
    with h5py.File(output_hdf5_path, 'w') as hf:
        # Determine board tensor shape from a dummy call
        dummy_board = chess.Board()
        dummy_input_tensor = utils.board_to_tensor(dummy_board)
        board_tensor_shape = dummy_input_tensor.shape # e.g., (18, 8, 8)

        policy_shape = () # Single integer index per sample
        value_shape = ()  # Single float value per sample (scalar)

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
            shape=(0, *policy_shape), 
            maxshape=(None, *policy_shape), 
            dtype=np.int32, 
            compression='gzip', 
            chunks=True
        )
        values_dset = hf.create_dataset(
            'values', 
            shape=(0, *value_shape), 
            maxshape=(None, *value_shape), 
            dtype=np.float16, 
            compression='gzip', 
            chunks=True
        )
        print(f"Opened HDF5 file: {output_hdf5_path} and created datasets.")
        print(f"Processing PGN files from: {pgn_directory_path}")

        # Loop through each PGN file found
        for pgn_file_path in pgn_files:
            if max_total_moves is not None and total_moves_processed >= max_total_moves:
                print(f"Reached {max_total_moves} moves. Stopping processing.")
                break # Exit the file loop

            print(f"\nProcessing file: {os.path.basename(pgn_file_path)}")
            with open(pgn_file_path) as pgn_file:
                game_count = 0
                while True:
                    game = chess.pgn.read_game(pgn_file)
                    if game is None:
                        break # End of current PGN file

                    game_count += 1
                    
                    # Check if total moves limit is reached before processing next game
                    if max_total_moves is not None and total_moves_processed >= max_total_moves:
                        break 

                    board = game.board()
                    
                    result = game.headers.get("Result")
                    if result == "1-0":
                        game_value = 1.0
                    elif result == "0-1":
                        game_value = -1.0
                    else:
                        game_value = 0.0

                    for move in game.mainline_moves():
                        if max_total_moves is not None and total_moves_processed >= max_total_moves:
                            break # Exit the move loop if total moves limit is hit mid-game

                        input_tensor = utils.board_to_tensor(board)

                        try:
                            from_row, from_col, channel = utils.move_to_policy_components(move, board)
                        except ValueError as e:
                            print(f"Skipping move {move} in {os.path.basename(pgn_file_path)} due to encoding error: {e}")
                            board.push(move)
                            continue
                        except Exception as e:
                            print(f"UNEXPECTED ERROR: Skipping move {move} in {os.path.basename(pgn_file_path)} due to unknown error: {e}")
                            board.push(move)
                            continue
                        
                        policy_flat_index = utils.policy_components_to_flat_index(from_row, from_col, channel)
                        
                        input_tensors_batch.append(input_tensor)
                        policy_targets_batch.append(policy_flat_index) 
                        
                        current_player_value = game_value if board.turn == chess.WHITE else -game_value
                        value_targets_batch.append(current_player_value)

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
                            
                            print(f"  Saved {len(input_tensors_batch)} moves to HDF5. Total moves in HDF5: {boards_dset.shape[0]} / {max_total_moves or 'All'}")
                            
                            input_tensors_batch = []
                            policy_targets_batch = []
                            value_targets_batch = []
            
            # After processing all games in a file, check if limit reached
            if max_total_moves is not None and total_moves_processed >= max_total_moves:
                break # Exit the file loop (redundant, but good for clarity)

        # Save any remaining data in the last partial batch (after all files or limit hit)
        if input_tensors_batch:
            current_len = boards_dset.shape[0]
            boards_dset.resize(current_len + len(input_tensors_batch), axis=0)
            policies_dset.resize(current_len + len(policy_targets_batch), axis=0)
            values_dset.resize(current_len + len(value_targets_batch), axis=0)
            
            boards_dset[current_len:] = np.array(input_tensors_batch, dtype=np.float16)
            policies_dset[current_len:] = np.array(policy_targets_batch, dtype=np.int32)
            values_dset[current_len:] = np.array(value_targets_batch, dtype=np.float16)
            
            print(f"Saved {len(input_tensors_batch)} remaining moves to HDF5. Total moves in HDF5: {boards_dset.shape[0]}")

    print(f"Finished processing. Final total moves processed: {total_moves_processed}")
    print(f"Data saved to: {output_hdf5_path}")


if __name__ == "__main__":
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    
    pgn_data_directory = os.path.join(parent_dir, "data", "lichess_elite_db")
    
    # Define the output HDF5 file path
    # This will create 'chess_data_5M_moves.h5' in your 'your_project/' directory
    output_hdf5_file = os.path.join(current_script_dir, "data", "chess_data.h5") 

    # Set the maximum number of moves to process
    max_moves_limit = 5000000 # 5 million moves

    print(f"Starting data processing from directory: {pgn_data_directory}")
    print(f"Saving processed data to: {output_hdf5_file}")
    print(f"Targeting a maximum of {max_moves_limit} moves.")

    process_pgn_data_from_directory(
        pgn_directory_path=pgn_data_directory,
        output_hdf5_path=output_hdf5_file,
        max_total_moves=max_moves_limit
    )

    print("Data processing complete!")