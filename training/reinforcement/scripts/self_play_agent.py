import chess
import os
import sys
import logging
import random
import uuid
import numpy as np
import time

current_script_dir = os.path.dirname(os.path.abspath(__file__))
rl_root = os.path.abspath(os.path.join(current_script_dir, ".."))
project_root = os.path.abspath(os.path.join(current_script_dir, "../../.."))

sys.path.insert(0, rl_root)
sys.path.insert(0, project_root)

from mcts.mcts_engine import MCTSEngine


class SelfPlayAgent:
    """
    A chess player wrapper for an MCTS engine designed for a multiprocessing
    environment with a central batcher. This class manages the game state
    for a single game worker and communicates with the MCTS instance.
    """
    def __init__(self, name, logger, self_play_config, worker_id, inference_queue, result_queue):
        self.name = name
        self.logger = logger
        self.self_play_config = self_play_config
        
        # The agent now has direct access to the queues for its MCTS engine
        self.worker_id = worker_id
        self.inference_queue = inference_queue
        self.result_queue = result_queue
        self.worker_batch_size = self.self_play_config['batch_size_per_worker']

        # These are reset each game
        self.mcts = None
        self.our_last_move = None
    
    def get_move(self, board, move_number, search_depth, last_move_played):
        """
        Runs MCTS simulations and selects a move based on a temperature schedule.
        """
        self.logger.info(f"\n{'='*60}\n{' '*20}--- MOVE {move_number} STARTED ---\n{'='*60}\n")
        move_start_time = time.time()
        
        if board.is_game_over():
            self.logger.info("Game is already over, no move to make.")
            return None, None, None

        if self.mcts is None:
            # We now pass the queues to the MCTS engine
            self.mcts = MCTSEngine(
                logger=self.logger, 
                worker_id=self.worker_id,
                worker_batch_size=self.worker_batch_size,
                inference_queue=self.inference_queue,
                result_queue=self.result_queue,
                cpuct=self.self_play_config['cpuct'], 
                k_rave=self.self_play_config['k_rave'],
                virtual_loss=self.self_play_config['virtual_loss'],
                dirichlet_alpha=self.self_play_config['dirichlet_alpha'],
                dirichlet_epsilon=self.self_play_config['dirichlet_epsilon'],
            )
            self.mcts.set_new_root(board.copy(), None, None) 
        else:
            self.mcts.set_new_root(board.copy(), self.our_last_move, last_move_played)
        
        self.logger.info(f"Current player: {self.name}")
        self.logger.info(f"Our last move: {self.our_last_move}. Last move played {last_move_played}")

        simulation_count = self.mcts.run_simulations(search_depth, self.self_play_config['early_cutoff_simulations'], self.self_play_config['early_cutoff_threshold'])

        moves = list(self.mcts.root.children.keys())
        visits = np.array([self.mcts.root.children[move].visits for move in moves], dtype=np.float32)

        # Determine temperature based on move number
        if move_number <= self.self_play_config['temperature_threshold_move']:
            temperature = self.self_play_config['temperature_high']
        else:
            temperature = self.self_play_config['temperature_low']

        best_move = None
        if temperature < 1e-6:
            best_move_index = np.argmax(visits)
            best_move = moves[best_move_index]
        else:
            visits_exp = visits ** (1.0 / temperature)
            probabilities = visits_exp / np.sum(visits_exp)
            best_move = np.random.choice(moves, p=probabilities)
        
        self.our_last_move = best_move
        policy_vector, root_value = self.mcts.get_target_vectors()

        move_end_time = time.time()
        total_move_time = move_end_time - move_start_time
        simulation_speed = (simulation_count / total_move_time) if total_move_time > 0 else 0

        self.logger.info(f"Total move time: {total_move_time:.4f}, with {simulation_speed:.4f} simulations per second")

        return best_move, policy_vector, root_value, simulation_count
    
    def reset_for_new_game(self):
        """
        Resets the player's state for a new game.
        """
        self.logger.debug(f"Resetting state for a new game.")
        
        self.mcts = MCTSEngine(
            logger=self.logger, 
            worker_id=self.worker_id,
            worker_batch_size=self.worker_batch_size,
            inference_queue=self.inference_queue,
            result_queue=self.result_queue,
            cpuct=self.self_play_config['cpuct'],
            k_rave=self.self_play_config['k_rave'],
            virtual_loss=self.self_play_config['virtual_loss'],
            dirichlet_alpha=self.self_play_config['dirichlet_alpha'],
            dirichlet_epsilon=self.self_play_config['dirichlet_epsilon'],
        )
        self.our_last_move = None