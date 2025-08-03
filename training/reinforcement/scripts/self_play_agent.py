import chess
import os
import sys
import logging
import random
import uuid
import torch
import numpy as np

current_script_dir = os.path.dirname(os.path.abspath(__file__))
rl_root = os.path.abspath(os.path.join(current_script_dir, ".."))
project_root = os.path.abspath(os.path.join(current_script_dir, "../../.."))

sys.path.insert(0, rl_root)
sys.path.insert(0, project_root)

from mcts.mcts_engine import MCTSEngine
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

        self.temperature_threshold_move = config['self_play']['temperature_threshold_move']
        self.temperature_high = config['self_play']['temperature_high']
        self.temperature_low = config['self_play']['temperature_low']

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
    

    def get_move(self, board: chess.Board, move_number: int, search_depth: int):
        """
        Runs MCTS simulations and selects a move based on a temperature schedule.
        High temperature is used in the early game for exploration, and low
        temperature is used in the late game for exploitation.
        """
        self.logger.debug(f"\n{'='*60}\n{' '*20}--- MOVE {move_number} STARTED ---\n{'='*60}\n")
        
        if board.is_game_over():
            self.logger.info("Game is already over, no move to make.")
            return None, None, None

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

        moves = list(self.mcts.root.children.keys())
        visits = np.array([self.mcts.root.children[move].visits for move in moves], dtype=np.float32)

        # Determine temperature based on move number
        if move_number <= self.temperature_threshold_move:
            temperature = self.temperature_high
        else:
            temperature = self.temperature_low

        best_move = None
        if temperature < 1e-6: # Check for a very low temperature (near zero)
            # T=0, select the move with the highest visit count (pure exploitation)
            best_move_index = np.argmax(visits)
            best_move = moves[best_move_index]
        else:
            # T > 0, calculate probabilities based on temperature and sample a move
            visits_exp = visits ** (1.0 / temperature)
            probabilities = visits_exp / np.sum(visits_exp)

            # Select a move based on the calculated probabilities
            best_move = np.random.choice(moves, p=probabilities)
        
        self.last_move = best_move
        policy_vector, root_value = self.mcts.get_target_vectors()

        self.logger.debug(f"MCTS for move {move_number} picked move: {best_move.uci()} with temperature {temperature}.")
        return best_move, policy_vector, root_value
    
    def reset_for_new_game(self):
        """
        Resets the player's state for a new game - called at the start of each new game.
        """
        self.logger.debug(f"Resetting state for a new game.")

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