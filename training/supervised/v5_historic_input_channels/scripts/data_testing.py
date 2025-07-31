import h5py
import numpy as np
import os
import sys
import logging
from datetime import datetime

# --- Logging Configuration ---
current_script_dir = os.path.dirname(os.path.abspath(__file__))
# Adjust parent_dir to point to the root of your project if 'utils.py' is needed,
# though for simply reading and printing, utils might not be strictly necessary.
# If you want to decode policy flat index back to moves, you'll need utils.
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../.."))
sys.path.insert(0, parent_dir)

log_dir = os.path.join(current_script_dir, "../logs/data_inspection")
os.makedirs(log_dir, exist_ok=True)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file_path = os.path.join(log_dir, f"inspect_hdf5_data_{timestamp}.log")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file_path),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Try to import utils for board/move conversion.
# This part is optional but useful if you want to see actual moves.
try:
    import utils
    logger.info("Successfully imported 'utils' module. Move decoding will be available.")
    can_decode_moves = True
except ImportError:
    logger.warning("Could not import 'utils' module. Policy indices will be shown as raw numbers.")
    can_decode_moves = False

# --- Channel Names Definition (from your utils.py board_to_tensor_68) ---
CHANNEL_NAMES = [
    "White Pawns", "White Knights", "White Bishops", "White Rooks", "White Queens", "White King",
    "Black Pawns", "Black Knights", "Black Bishops", "Black Rooks", "Black Queens", "Black King",
    "White to Move (1.0) / Black to Move (0.0)",
    "White Kingside Castling Rights", "White Queenside Castling Rights",
    "Black Kingside Castling Rights", "Black Queenside Castling Rights",
    "En Passant File (1.0 at file of ep square)",
]

# Add names for historical planes
for i in range(4): # 4 historical half-moves
    CHANNEL_NAMES.extend([
        f"Hist {i+1} half-move: White Pawns", f"Hist {i+1} half-move: White Knights",
        f"Hist {i+1} half-move: White Bishops", f"Hist {i+1} half-move: White Rooks",
        f"Hist {i+1} half-move: White Queens", f"Hist {i+1} half-move: White King",
        f"Hist {i+1} half-move: Black Pawns", f"Hist {i+1} half-move: Black Knights",
        f"Hist {i+1} half-move: Black Bishops", f"Hist {i+1} half-move: Black Rooks",
        f"Hist {i+1} half-move: Black Queens", f"Hist {i+1} half-move: Black King",
    ])

CHANNEL_NAMES.extend([
    "Two-fold Repetition (1.0 if current position has appeared once before)",
    "Three-fold Repetition (1.0 if current position has appeared twice before)",
])


def print_board_plane(plane_data, channel_name):
    """Prints a single 8x8 plane with a descriptive header."""
    logger.info(f"    Channel: {channel_name}")
    # Adjust formatting based on your data (e.g., if values are not just 0 or 1)
    # For binary presence (0 or 1), using integers looks cleaner.
    # For float values, consider using ':.1f' or similar.
    for r in range(8):
        # Format each row: '1 0 0 1 ...' or '0.0 1.0 0.0 ...'
        logger.info("      " + " ".join([f"{x:.0f}" for x in plane_data[r, :]]))
    logger.info("") # Add a blank line for separation

def inspect_hdf5_data(file_path, num_rows_to_print=1, start_row=0):
    """
    Reads an HDF5 file and prints a specified number of rows from 'inputs', 'policies',
    and 'values' datasets, starting from a given row, with detailed input tensor display.

    Args:
        file_path (str): The path to the HDF5 file.
        num_rows_to_print (int): The number of rows to print from each dataset.
        start_row (int): The starting row index from which to begin inspection (0-indexed).
    """
    if not os.path.exists(file_path):
        logger.error(f"Error: HDF5 file not found at '{file_path}'.")
        return

    logger.info(f"Attempting to read HDF5 file: '{file_path}'")
    logger.info(f"Will print {num_rows_to_print} rows, starting from row {start_row}.")

    try:
        with h5py.File(file_path, 'r') as hf:
            if 'inputs' not in hf or 'policies' not in hf or 'values' not in hf:
                logger.error(f"Error: HDF5 file '{file_path}' does not contain "
                             "expected datasets ('inputs', 'policies', 'values').")
                logger.info(f"Available datasets: {list(hf.keys())}")
                return

            inputs_dset = hf['inputs']
            policies_dset = hf['policies']
            values_dset = hf['values']

            total_entries = inputs_dset.shape[0]
            logger.info(f"Total entries in the HDF5 file: {total_entries}")

            if total_entries == 0:
                logger.warning("The HDF5 file is empty (contains 0 entries).")
                return
            
            if start_row >= total_entries:
                logger.warning(f"Start row ({start_row}) is beyond the total number of entries ({total_entries}). No data to display.")
                return
            
            # Determine the actual number of rows to read to avoid going out of bounds
            end_row = min(start_row + num_rows_to_print, total_entries)
            rows_to_read_actual = end_row - start_row

            if rows_to_read_actual <= 0:
                logger.warning(f"Calculated 0 or fewer rows to read from start_row {start_row} to end_row {end_row}. Adjust num_rows_to_print or start_row.")
                return

            logger.info(f"Reading {rows_to_read_actual} rows for inspection (from index {start_row} to {end_row-1}).")

            # Read the specified range of entries
            first_inputs = inputs_dset[start_row : end_row]
            first_policies = policies_dset[start_row : end_row]
            first_values = values_dset[start_row : end_row]

            logger.info("\n" + "="*80)
            logger.info(f"         HDF5 Data Inspection: Rows {start_row} to {end_row-1}")
            logger.info("="*80 + "\n")

            for i in range(rows_to_read_actual):
                # The index within the slice is 'i', but the original row number is 'start_row + i'
                original_row_number = start_row + i
                logger.info(f"\n{'='*20} Row {original_row_number+1}/{total_entries} (Displaying entry {i+1}/{rows_to_read_actual}) {'='*20}\n")
                
                # --- Print Input Tensor Nicely ---
                logger.info(f"Input Tensor (Shape: {first_inputs[i].shape})")
                
                if first_inputs[i].shape[0] != len(CHANNEL_NAMES):
                    logger.warning(f"  Mismatch: Input tensor has {first_inputs[i].shape[0]} channels, but {len(CHANNEL_NAMES)} channel names are defined. Channel names might be misaligned.")

                for c in range(first_inputs[i].shape[0]): # Iterate through channels
                    channel_name = CHANNEL_NAMES[c] if c < len(CHANNEL_NAMES) else f"Unknown Channel {c}"
                    print_board_plane(first_inputs[i, c, :, :], channel_name)
                
                # --- Print Policy and Value Targets ---
                policy_flat_index = first_policies[i]
                logger.info(f"Policy Target (Flat Index): {policy_flat_index}")
                if can_decode_moves:
                    try:
                        from_row, from_col, channel_idx = utils.policy_flat_index_to_components(policy_flat_index)
                        
                        logger.info(f"  Decoded Policy Components: From Rank: {from_row}, File: {from_col}, Move Type Channel: {channel_idx}")
                    except Exception as e:
                        logger.warning(f"  Could not decode policy index to components/move: {e}")
                
                value_target = first_values[i]
                logger.info(f"Value Target: {value_target:.4f}")

            logger.info("\n" + "="*80)
            logger.info("          HDF5 Data Inspection Complete")
            logger.info("="*80 + "\n")

    except Exception as e:
        logger.error(f"An unexpected error occurred while reading the HDF5 file: {e}", exc_info=True)

if __name__ == "__main__":
    # --- IMPORTANT: Configure the path to the HDF5 file you want to inspect ---
    
    # Path to the directory where your HDF5 files are saved
    data_dir = os.path.join(current_script_dir, "../data", "stockfish_endgames")
    
    # Specify the HDF5 file to inspect
    HDF5_FILE_TO_INSPECT = os.path.join(data_dir, "stockfish_worker_0.h5")

    print(f"Configured HDF5 file to inspect: {HDF5_FILE_TO_INSPECT}")

    # Number of rows to display
    NUM_ROWS_TO_DISPLAY = 5

    # New parameter: Starting row index (0-indexed)
    # For example, to start from the 10th row, set START_ROW = 9
    START_ROW = 20000

    logger.info(f"Starting HDF5 data inspection for file: {HDF5_FILE_TO_INSPECT}")
    logger.info(f"Displaying {NUM_ROWS_TO_DISPLAY} rows, starting from row {START_ROW}.")
    
    inspect_hdf5_data(HDF5_FILE_TO_INSPECT, NUM_ROWS_TO_DISPLAY, START_ROW)