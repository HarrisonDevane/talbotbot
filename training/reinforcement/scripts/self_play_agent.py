import chess
import os
import sys
import logging
import random
import uuid
import torch

current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_script_dir, "../../.."))
sys.path.insert(0, project_root)

from mcts_engine import MCTSEngine
from model import ChessAIModel


class TalbotPlayer:
    """
    A chess player wrapper for an MCTS engine designed for a multiprocessing
    environment with a central batcher. This class manages the game state
    for a single game worker and communicates with the MCTS instance.
    """
    def __init__(self, logger: logging.Logger,  config):
        self.logger = logger
        self.cpuct = config['talbot']['cpuct']
        self.batch_size = config['talbot']['batchsize']
        self.dirichlet_alpha = config['self_play']['dirichlet_alpha']
        self.dirichlet_epsilon = config['self_play']['dirichlet_epsilon']


        # These are reset each game
        self.mcts = None
        self.last_move = None

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ChessAIModel(num_input_planes=config['talbot']['input_planes'], num_residual_blocks=config['talbot']['resblocks'], num_filters=config['talbot']['filters'])

        checkpoint = torch.load(config['talbot']['model_path'], map_location=self.device, weights_only=True)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.logger.debug(f"Model loaded successfully from {config['talbot']['model_path']}")

        self.model.to(self.device)
        self.model.eval()

    
    def get_policy_value(self, board_tensor: torch.Tensor):
        with torch.no_grad():
            policy_logits, value_output = self.model(board_tensor)
        return policy_logits, value_output
    

    def get_move(self, board: chess.Board, move_number: int, search_depth: int) -> chess.Move:
        """
        Runs MCTS simulations to determine the best move for the current board state.
        This method replaces the previous version that ran the model directly.
        """
        self.logger.info(f"\n{'='*60}\n{' '*20}--- MOVE {move_number} STARTED ---\n{'='*60}\n")
        
        if board.is_game_over():
            self.logger.info("Game is already over, no move to make.")
            return None

        if self.mcts is None:
            self.mcts = MCTSEngine(
                self.logger, 
                self, 
                self.cpuct, 
                self.batch_size,
                self.dirichlet_alpha,
                self.dirichlet_epsilon
            )
            self.mcts.set_new_root(board.copy(), None) 
        else:
            self.mcts.set_new_root(board.copy(), self.last_move)

        self.mcts.run_simulations(search_depth)

        best_move = None
        max_visits = -1

        # Select the best move from the root's children based on visit count
        if self.mcts.root is not None and self.mcts.root.children:
            for move, child_node in self.mcts.root.children.items():
                if child_node.visits > max_visits:
                    max_visits = child_node.visits
                    best_move = move
        
        # Fallback to a random legal move if no move was found
        if best_move is None:
            legal_moves = list(board.legal_moves)
            if legal_moves:
                best_move = random.choice(legal_moves)
        
        self.last_move = best_move

        self.logger.info(f"MCTS for move {move_number} picked move: {best_move.uci()} with {max_visits} visits.")
        return best_move
    
    def reset_for_new_game(self):
        """
        Resets the player's state for a new game - called at the start of each new game.
        """
        self.logger.info(f"Resetting state for a new game.")

        # Re-initialize the MCTS engine to discard the old tree
        self.mcts = MCTSEngine(
            self.logger, 
            self, 
            self.cpuct, 
            self.batch_size,
            self.dirichlet_alpha,
            self.dirichlet_epsilon
        )
        self.last_move = None
        self.move_number = 0