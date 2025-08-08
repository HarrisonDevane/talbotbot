import torch
import os

# --- User Configuration ---
# IMPORTANT: Replace this with the actual path to your checkpoint file.
# This file is expected to contain a dictionary, with 'model_state_dict' as one of its keys.
YOUR_CHECKPOINT_PATH = "/Users/User/Projects/talbot/training/reinforcement/best_models/best_model.pth"

# The path where you want to save the extracted model state dictionary.
# This file will contain ONLY the model's parameters (a dictionary of tensors).
OUTPUT_MODEL_STATE_PATH = "/Users/User/Projects/talbot/training/reinforcement/best_models/best_model_corrected.pth"
# --- End User Configuration ---

try:
    # Load the full checkpoint dictionary from your file
    # Use map_location='cpu' to ensure it loads correctly regardless of CUDA availability
    checkpoint = torch.load(YOUR_CHECKPOINT_PATH, map_location='cpu')
    print(f"Successfully loaded the checkpoint dictionary from: {YOUR_CHECKPOINT_PATH}")

    # Check if 'model_state_dict' key exists in the loaded checkpoint
    if 'model_state_dict' in checkpoint:
        # Directly extract the model_state_dict
        model_state_dict = checkpoint['model_state_dict']
        print("Successfully extracted 'model_state_dict' from the checkpoint.")

        # Save ONLY the extracted model_state_dict to a new file
        torch.save(model_state_dict, OUTPUT_MODEL_STATE_PATH)
        print(f"Model state dictionary saved directly to: {OUTPUT_MODEL_STATE_PATH}")

        # Optional: Verify by loading the new file to confirm its content
        loaded_state = torch.load(OUTPUT_MODEL_STATE_PATH, map_location='cpu')
        print(f"Verification: Loaded '{OUTPUT_MODEL_STATE_PATH}'. It contains {len(loaded_state)} parameter tensors.")

    else:
        print(f"Error: 'model_state_dict' key not found in the checkpoint loaded from {YOUR_CHECKPOINT_PATH}.")
        print("Please ensure your checkpoint file contains a dictionary with this key.")

except FileNotFoundError:
    print(f"Error: Checkpoint file not found at {YOUR_CHECKPOINT_PATH}.")
    print("Please ensure the path is correct and the file exists.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")