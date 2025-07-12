import chess
import cython_chess
import os
import torch
import torch.nn.functional as F
import sys
import math
import random
import time
import logging
from .talbot_engine.mcts_engine_single import MCTSEngineSingle
from .talbot_engine.mcts_engine_batched import MCTSEngineBatched

# Adjust path for internal modules
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, parent_dir)

import training.supervised.model as model

class TalbotPlayer:
    def __init__(self, logger: logging.Logger, model_path: str, name="Talbot", num_residual_blocks: int = 20, num_input_planes: int = 18, num_filters: int = 128, cpuct: float = 1.0, batch_size: int = 16):
        self.name = name
        self.logger = logger
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = model.ChessAIModel(num_input_planes=num_input_planes, num_residual_blocks=num_residual_blocks, num_filters=num_filters)

        checkpoint = torch.load(model_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.logger.debug(f"Model loaded successfully from {model_path}")

        self.model.to(self.device)
        self.model.eval()

        self.cpuct = cpuct
        self.batch_size = batch_size
        self.mcts = None 
        self.last_own_move = None
        self.move_number = 0


    def get_policy_value(self, board_tensor: torch.Tensor):
        with torch.no_grad():
            policy_logits, value_output = self.model(board_tensor)
        return policy_logits, value_output

    def get_move(self, board: chess.Board, time_per_move: float = None, depth_limit: int = None) -> chess.Move:
        self.move_number += 1
        self.logger.info(f"\n{'='*60}\n{' '*20}--- MOVE {self.move_number} STARTED ---\n{'='*60}\n")

        if board.is_game_over():
            self.logger.info("Game is already over, no move to make.")
            return None

        opponent_last_move = None
        if board.move_stack:
            opponent_last_move = board.move_stack[-1]

        if self.mcts is None:
            self.mcts = MCTSEngineBatched(self.logger, self, self.cpuct, self.batch_size)
            self.mcts.set_new_root(board.copy(), None, None) 
        else:
            self.mcts.set_new_root(board.copy(), opponent_last_move, self.last_own_move)

        self.mcts.run_simulations(time_per_move)

        best_move = None
        max_visits = -1
        
        for move, child_node in self.mcts.root.children.items():
            if child_node.visits > max_visits:
                max_visits = child_node.visits
                best_move = move

        if best_move is None:
            legal_moves = cython_chess.generate_legal_moves(board, chess.BB_ALL, chess.BB_ALL)
            if legal_moves:
                best_move = random.choice(list(legal_moves))
        
        self.last_own_move = best_move

        self.logger.info(f"MCTS for move {self.move_number} picked move: {best_move.uci()} with {max_visits} visits.")
        return best_move
    

    def is_human(self):
        return False

    def close(self):
        return