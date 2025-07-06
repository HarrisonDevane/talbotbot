import chess
import os
import sys
import logging
from .talbot_engine.talbot_mcts import TalbotMCTSEngine

logger = logging.getLogger(__name__)

class TalbotPlayer:
    def __init__(self, model_path: str, name: str = "Talbot", num_residual_blocks: int = 20, cpuct: float = 1.0, batch_size: int = 16):
        self.name = name

        logger.info(f"Initializing TalbotPlayer '{self.name}' with model: {model_path}")

        self.engine = TalbotMCTSEngine(
            model_path=model_path,
            num_residual_blocks=num_residual_blocks,
            cpuct=cpuct,
            batch_size=batch_size
        )
        logger.info("TalbotMCTSEngine instance created.")

    def get_move(self, board: chess.Board, time_per_move: int = None, depth_limit: int = None) -> chess.Move:
        best_move = self.engine.find_best_move(board, time_per_move, depth_limit)
        return best_move

    def is_human(self) -> bool:
        return False

    def close(self):
        pass