import os
import h5py
import glob
import sys
import numpy as np

def combine_hdf5_files(input_dir, output_file_path):
    """
    Combines multiple HDF5 files from an input directory into a single HDF5 file.

    Args:
        input_dir (str): The directory containing the individual HDF5 files.
        output_file_path (str): The full path for the consolidated HDF5 file.
    """
    print(f"Starting HDF5 file combination process.")
    print(f"Input directory: {input_dir}")
    print(f"Output consolidated file: {output_file_path}")

    all_hdf5_files = sorted(glob.glob(os.path.join(input_dir, "*.h5")))

    if not all_hdf5_files:
        print(f"Warning: No HDF5 files found in '{input_dir}'. Nothing to combine.")
        return

    print(f"Found {len(all_hdf5_files)} individual HDF5 files to combine.")

    # Get the shape of the 'inputs' dataset from the first file to initialize the consolidated file
    # This assumes all 'inputs' datasets have the same inner dimensions.
    first_file_path = all_hdf5_files[0]
    try:
        with h5py.File(first_file_path, 'r') as f:
            if 'inputs' not in f:
                print(f"Error: First HDF5 file '{first_file_path}' does not contain 'inputs' dataset. Cannot determine shape.")
                return
            dummy_input_shape = f['inputs'].shape[1:] # Get (channels, rows, cols)
            print(f"Detected input tensor shape: {dummy_input_shape} from {os.path.basename(first_file_path)}")
    except Exception as e:
        print(f"Error opening or reading first HDF5 file '{first_file_path}': {e}")
        return

    total_moves_combined = 0
    errors_encountered = 0

    # Create the new, consolidated HDF5 file
    with h5py.File(output_file_path, 'w') as hf_combined:
        # Create datasets with resizable (maxshape=None) attributes
        # Use existing data types and chunking for efficiency, though chunks will be adjusted by h5py
        boards_dset = hf_combined.create_dataset(
            'inputs',
            shape=(0, *dummy_input_shape), # Start with 0 length, correct dimensions
            maxshape=(None, *dummy_input_shape),
            dtype=np.float16, # Assuming float16 from your original script
            compression='gzip',
            chunks=True # Let h5py determine optimal chunking based on data
        )
        policies_dset = hf_combined.create_dataset(
            'policies',
            shape=(0,),
            maxshape=(None,),
            dtype=np.int32, # Assuming int32 from your original script
            compression='gzip',
            chunks=True
        )
        values_dset = hf_combined.create_dataset(
            'values',
            shape=(0,),
            maxshape=(None,),
            dtype=np.float16, # Assuming float16 from your original script
            compression='gzip',
            chunks=True
        )
        print(f"Created consolidated HDF5 file '{output_file_path}' with empty datasets.")

        for i, file_path in enumerate(all_hdf5_files):
            file_name = os.path.basename(file_path)
            try:
                with h5py.File(file_path, 'r') as hf_individual:
                    inputs = hf_individual['inputs'][:]
                    policies = hf_individual['policies'][:]
                    values = hf_individual['values'][:]

                    num_records = inputs.shape[0]

                    if num_records == 0:
                        print(f"Warning: Skipping empty file: {file_name}")
                        continue

                    current_len = boards_dset.shape[0]

                    # Resize and append data
                    boards_dset.resize(current_len + num_records, axis=0)
                    policies_dset.resize(current_len + num_records, axis=0)
                    values_dset.resize(current_len + num_records, axis=0)

                    boards_dset[current_len:] = inputs
                    policies_dset[current_len:] = policies
                    values_dset[current_len:] = values

                    total_moves_combined += num_records
                    print(f"Appended {num_records} records from {file_name}. Total combined: {total_moves_combined}")

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
    # Determine the current script's directory
    current_script_dir = os.path.dirname(os.path.abspath(__file__))

    # Input directory: Where your individual HDF5 files are stored
    # This assumes 'processed_pgn_h5' is a subfolder of 'data' which is in
    # the same directory as your script. Adjust if your structure is different.
    input_hdf5_directory = os.path.join(current_script_dir, "data")

    # Output file: The name and path for the single consolidated HDF5 file
    consolidated_hdf5_file = os.path.join(current_script_dir, "data", "all_chess_data.h5")

    # Run the combination process
    combine_hdf5_files(input_hdf5_directory, consolidated_hdf5_file)