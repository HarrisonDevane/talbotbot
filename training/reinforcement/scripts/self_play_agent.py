import chess
import os
import sys
import logging
import random
import uuid
import torch
import numpy as np
import time

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
    def __init__(self, logger, model_config, self_play_config):
        self.logger = logger
        self.model_config = model_config
        self.self_play_config = self_play_config

        # These are reset each game
        self.mcts = None
        self.last_move = None

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ChessAIModel(num_input_planes=self.model_config['input_planes'], 
                                  num_residual_blocks=model_config['resblocks'], 
                                  num_filters=model_config['filters'])

        checkpoint = torch.load(model_config['model_path'], map_location=self.device, weights_only=True)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.logger.debug(f"Model loaded successfully from {model_config['model_path']}")

        self.model.to(self.device)
        self.model.eval()

    
    def get_policy_value(self, board_tensor):
        with torch.no_grad():
            policy_logits, value_output = self.model(board_tensor)
        return policy_logits, value_output
    

    def get_move(self, board, move_number, search_depth):
        """
        Runs MCTS simulations and selects a move based on a temperature schedule.
        High temperature is used in the early game for exploration, and low
        temperature is used in the late game for exploitation.
        """
        self.logger.info(f"\n{'='*60}\n{' '*20}--- MOVE {move_number} STARTED ---\n{'='*60}\n")
        move_start_time = time.time()
        
        if board.is_game_over():
            self.logger.info("Game is already over, no move to make.")
            return None, None, None

        if self.mcts is None:
            self.mcts = MCTSEngine(
                logger=self.logger, 
                model_player=self, 
                cpuct=self.self_play_config['cpuct'], 
                batch_size=self.self_play_config['batch_size'],
                virtual_loss=self.self_play_config['virtual_loss'],
                dirichlet_alpha=self.self_play_config['dirichlet_alpha'],
                dirichlet_epsilon=self.self_play_config['dirichlet_epsilon'],
                selection_workers=self.self_play_config['selection_workers'],
                update_workers=self.self_play_config['update_workers']
            )
            self.mcts.set_new_root(board.copy(), None) 
        else:
            self.mcts.set_new_root(board.copy(), self.last_move)
        
        sim_start_time = time.time()
        self.mcts.run_simulations(search_depth, self.self_play_config['early_cutoff_simulations'], self.self_play_config['early_cutoff_threshold'])
        sim_end_time = time.time()

        moves = list(self.mcts.root.children.keys())
        visits = np.array([self.mcts.root.children[move].visits for move in moves], dtype=np.float32)

        # Determine temperature based on move number
        if move_number <= self.self_play_config['temperature_threshold_move']:
            temperature = self.self_play_config['temperature_high']
        else:
            temperature = self.self_play_config['temperature_low']

        best_move = None
        if temperature < 1e-6: # Check for a very low temperature (near zero)
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

        self.logger.info(f"MCTS for move {move_number} picked move: {best_move.uci()} with temperature {temperature}.")
        move_end_time = time.time()

        avg_move_time = move_end_time - move_start_time
        avg_sim_time = sim_end_time - sim_start_time

        self.logger.info(f"Total move time: {avg_move_time:.4f}, simulation time: {avg_sim_time:.4f}")

        return best_move, policy_vector, root_value
    
    def reset_for_new_game(self):
        """
        Resets the player's state for a new game - called at the start of each new game.
        """
        self.logger.debug(f"Resetting state for a new game.")

        # Re-initialize the MCTS engine to discard the old tree
        self.mcts = MCTSEngine(
            logger=self.logger, 
            model_player=self, 
            cpuct=self.self_play_config['cpuct'], 
            batch_size=self.self_play_config['batch_size'],
            virtual_loss=self.self_play_config['virtual_loss'],
            dirichlet_alpha=self.self_play_config['dirichlet_alpha'],
            dirichlet_epsilon=self.self_play_config['dirichlet_epsilon'],
            selection_workers=self.self_play_config['selection_workers'],
            update_workers=self.self_play_config['update_workers']
        )
        self.last_move = None
        self.move_number = 0