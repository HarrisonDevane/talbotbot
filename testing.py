import chess
import torch
import numpy as np
import sys
import os
import utils # Assuming utils is here
from training.supervised.model import ChessAIModel

# Adjust path to import your model and utils
# Assuming your project structure is something like:
# project_root/
# ├── data/
# ├── models/
# │   └── model.py  # Contains ChessAIModel
# └── scripts/
#     ├── mcts.py
#     ├── utils.py # Contains board_to_tensor, get_win_probability
#     └── temp_test_script.py (this script)

# Adjust sys.path to ensure you can import your modules
current_script_dir = os.path.dirname(os.path.abspath(__file__))


# --- Load your trained model ---
model_path = os.path.abspath(os.path.join(current_script_dir, "training/supervised/v2_pol_mvplayed_val_sfeval/model/best_chess_ai_model.pth"))
try:
    # Ensure ChessAIModel is instantiated with correct parameters
    # These should match the arguments passed to ChessAIModel in train.py's `train_model` function call
    model_instance = ChessAIModel(
        num_input_planes=18,        # Match your train.py
        num_residual_blocks=16,     # Match your train.py
        num_filters=128             # IMPORTANT: You need to include num_filters if your ChessAIModel constructor requires it!
    )
    
    # Load the entire checkpoint dictionary
    checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
    
    # Load only the model's state_dict from the checkpoint
    model_instance.load_state_dict(checkpoint['model_state_dict'])
    
    model_instance.eval() # Set model to evaluation mode

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_instance.to(device)
    
    print(f"Model loaded successfully from {model_path}. Using device: {device}")
    # You can also print useful info from the checkpoint, like epoch or best_val_loss
    print(f"Loaded from epoch: {checkpoint.get('epoch', 'N/A')}")
    print(f"Loaded best_val_loss: {checkpoint.get('best_val_loss', 'N/A'):.4f}")

except FileNotFoundError:
    print(f"Error: Model file not found at {model_path}")
    print("Please ensure the 'model_path' variable points to your actual trained model (.pth) file.")
    # You might want to exit here or handle the error gracefully
except Exception as e:
    print(f"Error loading model or state_dict: {e}")
    print("Please check your ChessAIModel instantiation parameters (num_input_planes, num_residual_blocks, num_filters) match your training script.")
    # Provide more context if you suspect the model architecture is mismatched


# Helper function to run inference and print results
def analyze_board(board, test_name=""):
    print(f"\n--- {test_name} ---")
    print(f"FEN: {board.fen()}")
    print(f"Turn: {'White' if board.turn == chess.WHITE else 'Black'}")

    board_input = torch.from_numpy(utils.board_to_tensor(board)).unsqueeze(0).float().to(device)
    with torch.no_grad():
        _, value_output = model_instance(board_input)

    raw_value = value_output.item()
    win_prob = utils.get_win_probability(value_output)

    print(f"Model's raw value_output.item() (from tanh): {raw_value:.4f}")
    print(f"Win probability (from utils.get_win_probability): {win_prob:.4f}")
    if raw_value > 0.5:
        print(f"  -> Model seems to think current player is doing well.")
    elif raw_value < -0.5:
        print(f"  -> Model seems to think current player is doing poorly.")
    else:
        print(f"  -> Model seems to think it's a relatively even position.")

# --- Test Case 1: White is overwhelmingly winning, White's Turn ---
blunder_before_fen = "2kr1bnr/ppp1pppp/2n5/3q4/3P2b1/2N1BN2/PPP2PPP/R2QKB1R b KQ - 6 6"
board_before_blunder = chess.Board(blunder_before_fen)
analyze_board(board_before_blunder, "Test Case 1: BEFORE Black's Queen Blunder (Black to move)")

# --- NEW Test Case 2: AFTER Black's Queen Blunder (White to move, Queen on f3) ---
# This is the board state immediately after Black's Queen captured on f3 (d5f3)
# It's now White's turn, and White can capture the Queen.
blunder_after_fen = "2kr1bnr/ppp1pppp/2n5/8/3P2b1/2N1Bq2/PPP2PPP/R2QKB1R w KQ - 0 7"
board_after_blunder = chess.Board(blunder_after_fen)
analyze_board(board_after_blunder, "Test Case 2: AFTER Black's Queen Blunder (White to move, Queen on f3)")

print("\n--- End of Model Perspective Test ---")