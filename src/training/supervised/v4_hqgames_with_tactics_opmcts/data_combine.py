import h5py
import os
import numpy as np

# --- Configuration ---
TACTICAL_H5_PATH = r"C:\Users\User\Projects\talbot\training\supervised\v4_hqgames_with_tactics_opmcts\data\tactical_puzzles_all_moves.h5"
GM_H5_PATH = r"C:\Users\User\Projects\talbot\training\supervised\v4_hqgames_with_tactics_opmcts\data\gm_moves.h5"
OUTPUT_H5_PATH = r"C:\Users\User\Projects\talbot\training\supervised\v4_hqgames_with_tactics_opmcts\data\combined_data_1M_GM.h5" # Changed output filename for clarity

# List of dataset names to combine
DATASET_NAMES = ['inputs', 'policies', 'values']

# Limit for GM moves
MAX_GM_MOVES_LIMIT = 1000000 # Only take 1 million GM moves

def combine_hdf5_data(tactical_file_path, gm_file_path, output_file_path, dataset_names, max_gm_moves_limit):
    print(f"Combining data from:\n  - {tactical_file_path}\n  - {gm_file_path}")
    print(f"Output will be saved to: {output_file_path}")

    with h5py.File(tactical_file_path, 'r') as f_tactical, \
         h5py.File(gm_file_path, 'r') as f_gm, \
         h5py.File(output_file_path, 'w') as f_combined:

        # Get the number of examples from tactical file
        num_tactical_examples = f_tactical[dataset_names[0]].shape[0]
        
        # Get the number of examples from GM file, respecting the limit
        actual_gm_examples_to_use = min(f_gm[dataset_names[0]].shape[0], max_gm_moves_limit)
        
        total_examples = num_tactical_examples + actual_gm_examples_to_use

        print(f"Tactical examples (from file): {num_tactical_examples}")
        print(f"GM examples (from file, max {max_gm_moves_limit} used): {actual_gm_examples_to_use}")
        print(f"Total combined examples expected: {total_examples}")

        # Create datasets in the combined file with resizable (unlimited) first dimension
        for name in dataset_names:
            # Get shape and dtype from the first file (assuming consistency)
            original_shape = f_tactical[name].shape
            original_dtype = f_tactical[name].dtype

            # Define maxshape with None for the first dimension to allow resizing
            max_shape = (None,) + original_shape[1:]

            # Create the dataset in the combined file
            f_combined.create_dataset(name, shape=(0,) + original_shape[1:], 
                                      maxshape=max_shape, dtype=original_dtype, 
                                      chunks=True, compression='gzip')

        # --- Append data from tactical puzzles file ---
        print("\nAppending tactical puzzle data...")
        for name in dataset_names:
            source_dset = f_tactical[name]
            target_dset = f_combined[name]

            current_size = target_dset.shape[0]
            new_size = current_size + source_dset.shape[0]
            target_dset.resize(new_size, axis=0)

            chunk_size = 10000
            for i in range(0, source_dset.shape[0], chunk_size):
                end_idx = min(i + chunk_size, source_dset.shape[0])
                data_chunk = source_dset[i:end_idx]
                target_dset[current_size + i : current_size + end_idx] = data_chunk
                print(f"  - Appended {end_idx} tactical examples to '{name}'...", end='\r')
            print(f"  - Finished appending {source_dset.shape[0]} tactical examples to '{name}'.")


        # --- Append data from GM moves file (respecting the limit) ---
        print("\nAppending GM moves data (limited to 1M)...")
        for name in dataset_names:
            source_dset = f_gm[name]
            target_dset = f_combined[name]

            current_size = target_dset.shape[0]
            # The number of GM examples to copy is actual_gm_examples_to_use
            new_size = current_size + actual_gm_examples_to_use
            target_dset.resize(new_size, axis=0)

            chunk_size = 10000
            # Iterate only up to the actual_gm_examples_to_use limit
            for i in range(0, actual_gm_examples_to_use, chunk_size):
                end_idx = min(i + chunk_size, actual_gm_examples_to_use)
                data_chunk = source_dset[i:end_idx]
                target_dset[current_size + i : current_size + end_idx] = data_chunk
                print(f"  - Appended {end_idx} GM examples to '{name}'...", end='\r')
            print(f"  - Finished appending {actual_gm_examples_to_use} GM examples to '{name}'.")

    print(f"\nData combining complete. Total examples in '{output_file_path}': {total_examples}")

if __name__ == "__main__":
    combine_hdf5_data(TACTICAL_H5_PATH, GM_H5_PATH, OUTPUT_H5_PATH, DATASET_NAMES, MAX_GM_MOVES_LIMIT)