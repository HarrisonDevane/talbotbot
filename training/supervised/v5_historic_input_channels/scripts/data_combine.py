import os
import h5py
import glob
import sys
import numpy as np

def combine_hdf5_files(input_dirs, output_file_path):
    """
    Combines multiple HDF5 files from one or more input directories into a single HDF5 file.
    Reads individual files in chunks to manage memory usage.

    Args:
        input_dirs (list): A list of directories containing the individual HDF5 files.
        output_file_path (str): The full path for the consolidated HDF5 file.
    """
    print(f"Starting HDF5 file combination process.")
    print(f"Input directories: {input_dirs}")
    print(f"Output consolidated file: {output_file_path}")

    all_hdf5_files = []
    for input_dir in input_dirs:
        if not os.path.isdir(input_dir):
            print(f"Warning: Input directory '{input_dir}' does not exist or is not a directory. Skipping.")
            continue
        all_hdf5_files.extend(sorted(glob.glob(os.path.join(input_dir, "*.h5"))))

    if not all_hdf5_files:
        print(f"Warning: No HDF5 files found in any of the specified input directories. Nothing to combine.")
        return

    print(f"Found {len(all_hdf5_files)} individual HDF5 files across all input directories to combine.")

    # Get the shape of the 'inputs' dataset from the first non-empty file to initialize the consolidated file
    dummy_input_shape = None
    first_valid_file_path = None

    for file_path in all_hdf5_files:
        try:
            with h5py.File(file_path, 'r') as f:
                if 'inputs' in f and f['inputs'].shape[0] > 0:
                    dummy_input_shape = f['inputs'].shape[1:]
                    first_valid_file_path = file_path
                    print(f"Detected input tensor shape: {dummy_input_shape} from {os.path.basename(first_valid_file_path)}")
                    break
                else:
                    print(f"Warning: File '{os.path.basename(file_path)}' is missing 'inputs' dataset or it's empty. Skipping for shape detection.")
        except Exception as e:
            print(f"Error opening or reading HDF5 file '{os.path.basename(file_path)}' for shape detection: {e}")

    if dummy_input_shape is None:
        print("Error: Could not determine 'inputs' dataset shape from any of the HDF5 files. Aborting combination.")
        return

    total_moves_combined = 0
    errors_encountered = 0
    progress_counter_for_print = 0 # Counter for progress reporting every N records

    # Define the chunk size for reading data from individual files
    # Adjust this value based on your available RAM and the size of your records
    # A larger chunk size means less overhead but more temporary memory usage.
    # 100,000 records might be a good starting point, adjust as needed.
    READ_CHUNK_SIZE = 100000 
    
    # Define how often to print progress (e.g., every 10,000 combined records)
    PRINT_PROGRESS_INTERVAL = 10000

    # Create the new, consolidated HDF5 file
    with h5py.File(output_file_path, 'w') as hf_combined:
        boards_dset = hf_combined.create_dataset(
            'inputs',
            shape=(0, *dummy_input_shape),
            maxshape=(None, *dummy_input_shape),
            dtype=np.float16,
            compression='lzf',
            chunks=True # Let h5py determine optimal chunking for the combined file
        )
        policies_dset = hf_combined.create_dataset(
            'policies',
            shape=(0,),
            maxshape=(None,),
            dtype=np.int32,
            compression='lzf',
            chunks=True
        )
        values_dset = hf_combined.create_dataset(
            'values',
            shape=(0,),
            maxshape=(None,),
            dtype=np.float16,
            compression='lzf',
            chunks=True
        )
        print(f"Created consolidated HDF5 file '{output_file_path}' with empty datasets.")

        for i, file_path in enumerate(all_hdf5_files):
            file_name = os.path.basename(file_path)
            try:
                with h5py.File(file_path, 'r') as hf_individual:
                    # Get dataset objects, not full data
                    inputs_src = hf_individual['inputs']
                    policies_src = hf_individual['policies']
                    values_src = hf_individual['values']

                    num_records_in_file = inputs_src.shape[0]

                    if num_records_in_file == 0:
                        print(f"Warning: Skipping empty file: {file_name}")
                        continue

                    print(f"Processing file {i+1}/{len(all_hdf5_files)}: '{file_name}' ({num_records_in_file} records)")

                    for start_idx in range(0, num_records_in_file, READ_CHUNK_SIZE):
                        end_idx = min(start_idx + READ_CHUNK_SIZE, num_records_in_file)
                        
                        # Read a chunk into memory
                        inputs_chunk = inputs_src[start_idx:end_idx]
                        policies_chunk = policies_src[start_idx:end_idx]
                        values_chunk = values_src[start_idx:end_idx]
                        
                        chunk_records_read = inputs_chunk.shape[0]

                        current_len = boards_dset.shape[0]

                        # Resize and append data to the combined datasets
                        boards_dset.resize(current_len + chunk_records_read, axis=0)
                        policies_dset.resize(current_len + chunk_records_read, axis=0)
                        values_dset.resize(current_len + chunk_records_read, axis=0)

                        boards_dset[current_len:] = inputs_chunk
                        policies_dset[current_len:] = policies_chunk
                        values_dset[current_len:] = values_chunk

                        total_moves_combined += chunk_records_read
                        progress_counter_for_print += chunk_records_read

                        # Print progress update
                        if progress_counter_for_print >= PRINT_PROGRESS_INTERVAL:
                            print(f"  ... Appended {total_moves_combined} records so far.")
                            progress_counter_for_print = 0 # Reset counter

                    # After processing all chunks in a file, print final status for that file
                    print(f"  Finished processing all {num_records_in_file} records from '{file_name}'.")

            except KeyError as ke:
                print(f"Error: Missing dataset in file '{file_name}': {ke}. Skipping this file.")
                errors_encountered += 1
            except Exception as e:
                print(f"Error processing file '{file_name}': {e}. Skipping this file.")
                errors_encountered += 1
        
    print("\n--- Combination Summary ---")
    print(f"Finished combining HDF5 files.")
    print(f"Total moves written to '{output_file_path}': {total_moves_combined}")
    if errors_encountered > 0:
        print(f"Warning: Encountered {errors_encountered} errors while processing individual files. Some data might be missing.")
    else:
        print("All files combined successfully without errors.")


if __name__ == "__main__":
    # --- Configuration ---
    current_script_dir = os.path.dirname(os.path.abspath(__file__))

    input_hdf5_directories = [
        os.path.join(current_script_dir, "../data", "gm_games"),
        os.path.join(current_script_dir, "../data", "lichess_elite_db"),
        os.path.join(current_script_dir, "../data", "processed_tactical_h5"),
        os.path.join(current_script_dir, "../data", "stockfish_endgames"),
        os.path.join(current_script_dir, "../data", "stockfish_endgames_2"),
    ]

    # Output file: The name and path for the single consolidated HDF5 file
    consolidated_hdf5_file = os.path.join(current_script_dir, "../data", "all_chess_data.h5")

    # Run the combination process
    combine_hdf5_files(input_hdf5_directories, consolidated_hdf5_file)


