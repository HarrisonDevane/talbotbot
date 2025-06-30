import chess
import os
import torch
import torch.nn.functional as F
import sys, os
import random

# Get the parent directory path to help with relative imports and paths
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, parent_dir)

import utils
import training.supervised.model as model

class TalbotPlayer:
    def __init__(self, model_path: str, name="Talbot"):
        self.name = name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize the model with the correct number of input planes (18 as per board_to_tensor)
        self.model = model.ChessAIModel(num_input_planes=18, num_residual_blocks=20, num_filters=128)
        
        # Load the entire checkpoint dictionary, then extract the model_state_dict
        # The `map_location` argument is important for loading models trained on GPU onto CPU, or vice versa.
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict']) 
        
        self.model.to(self.device)
        self.model.eval() # Set the model to evaluation mode

    def get_move(self, board: chess.Board) -> chess.Move:
        """
        Determines the best move for the current board state using the trained ChessAIModel.
        """
        # 1. Convert board to tensor
        board_tensor_np = utils.board_to_tensor(board)
        # Add a batch dimension and convert to torch.Tensor
        board_input = torch.from_numpy(board_tensor_np).unsqueeze(0).float().to(self.device)

        # 2. Perform forward pass
        with torch.no_grad(): # No need to calculate gradients for inference
            policy_logits, value_output = self.model(board_input)

        winning_prob = utils.get_win_probability(value_output)

        # The policy_logits are (1, TOTAL_POLICY_MOVES)
        # We need to reshape them to (BOARD_DIM, BOARD_DIM, POLICY_CHANNELS) conceptually
        # or just work with the flattened logits and map them back.

        # Convert policy_logits to probabilities
        policy_probs = F.softmax(policy_logits, dim=1).squeeze(0) # Squeeze to remove batch dimension

        # Get legal moves from the current board state
        legal_moves = list(board.legal_moves)
        
        # If there are no legal moves, it's a game end state (checkmate/stalemate)
        if not legal_moves:
            return None 

        # Map legal moves to their policy head flat indices
        legal_move_indices = []
        for move in legal_moves:
            try:
                # Ensure the move_to_policy_components handles current board's turn correctly
                # (which it does by normalizing from_row, from_col based on board.turn)
                from_row_norm, from_col_norm, channel = utils.move_to_policy_components(move, board)
                flat_index = utils.policy_components_to_flat_index(from_row_norm, from_col_norm, channel)
                legal_move_indices.append(flat_index)
            except ValueError as e:
                # This can happen for very unusual moves or if the mapping logic has a bug
                # For robustness, we can print a warning and skip such moves.
                print(f"Warning: Could not convert legal move {move.uci()} to policy components: {e}")
                continue
        
        if not legal_move_indices:
            print("No legal moves could be mapped to policy indices. Falling back to random move.")
            return random.choice(legal_moves)

        # Create a mask for legal moves
        legal_move_mask = torch.zeros_like(policy_probs, dtype=torch.bool)
        legal_move_mask[legal_move_indices] = True

        # Apply the mask: set probabilities of illegal moves to a very small number or 0
        # to ensure they are not picked. Using -inf for log-softmax, or 0 for softmax.
        # Since we already applied softmax, setting to 0 is appropriate.
        masked_policy_probs = policy_probs * legal_move_mask.float()

        # If all masked probabilities are zero (e.g., due to floating point underflow 
        # or issues with the mapping), fall back to a random legal move.
        if masked_policy_probs.sum() == 0:
            print("All masked policy probabilities are zero. Falling back to random move.")
            return random.choice(legal_moves)
            
        # Select the move with the highest probability among legal moves
        # Get the index of the maximum probability
        best_flat_index = torch.argmax(masked_policy_probs).item()

        # Convert the flat index back to policy components and then to a chess.Move
        from_row_norm, from_col_norm, channel = utils.policy_flat_index_to_components(best_flat_index)
        predicted_move = utils.policy_components_to_move(from_row_norm, from_col_norm, channel, board)

        # Double-check if the predicted_move is actually legal (it should be due to masking)
        if predicted_move and predicted_move in legal_moves:
            print(f'Probability Talbot is winning: {winning_prob}')
            return predicted_move
        else:
            # Fallback: if the predicted move is somehow illegal or None,
            # select the best legal move by iterating through all legal moves
            # and finding the one with the highest *original* probability.
            print(f"Warning: Model predicted an illegal or None move ({predicted_move}). Falling back to best legal move search.")
            best_move_from_legal_list = None
            max_prob = -1
            for move in legal_moves:
                try:
                    frn, fcn, ch = utils.move_to_policy_components(move, board)
                    flat_idx = utils.policy_components_to_flat_index(frn, fcn, ch)
                    current_prob = policy_probs[flat_idx].item()
                    if current_prob > max_prob:
                        max_prob = current_prob
                        best_move_from_legal_list = move
                except ValueError:
                    continue # Skip moves that cannot be mapped

            if best_move_from_legal_list:
                return best_move_from_legal_list
            else:
                # Last resort: random legal move if nothing else works
                print("Could not find any legal move with a valid policy index. Choosing a random legal move.")
                return random.choice(legal_moves)


    def is_human(self):
        return False

    def close(self):
        # No engine to close for a pure NN player
        pass