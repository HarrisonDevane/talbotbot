import os
import time
import logging
import numpy as np
from datetime import datetime

# Assuming these imports are in your project structure
from self_play_agent import TalbotPlayer
from self_play_game_worker import SelfPlayGameWorker

class DataGenerationTask:
    def __init__(self, output_dir, model_config, data_generation_config, best_iter):
        self.output_dir = output_dir
        self.model_config = model_config
        self.data_generation_config = data_generation_config
        self.best_iter = best_iter

        # Set up loggers
        self.log_dir = os.path.join(self.output_dir, "logs")
        os.makedirs(self.log_dir, exist_ok=True)

        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        
        self.main_logger = self._setup_logger(
            "SelfPlayManager", 
            self.data_generation_config['main_logging_level'],
            os.path.join(self.log_dir, f"data_generaration_{timestamp}.log")
        )
        
        self.worker_logger = self._setup_logger(
            "SelfPlayWorker",
            self.data_generation_config['worker_logging_level'],
            os.path.join(self.log_dir, f"data_generaration_games_{timestamp}.log")
        )

        # Instantiate core components
        self.mcts_player = TalbotPlayer(
           name=f'best_model_iter_{self.best_iter} (self-play)',
            logger=self.worker_logger,
            model_path=self.model_config['best_model_path'],
            model_config=self.model_config,
            self_play_config=self.data_generation_config
        )
        self.game_manager = SelfPlayGameWorker(
            logger=self.worker_logger,
            player_1=self.mcts_player,
            player_2=self.mcts_player,
            model_config=self.model_config,
            self_play_config=self.data_generation_config
        )

        self.game_number = 1

    def _setup_logger(self, name: str, level: str, log_file: str):
        logger = logging.getLogger(name)
        logger.setLevel(level)
        if logger.hasHandlers():
            logger.handlers.clear()
        
        formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s")
        file_handler = logging.FileHandler(log_file, mode='w')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        return logger


    def run_for_n_positions(self, n_positions: int):
        """
        Generates a specified number of positions and returns them as a list.
        This is the method the RLOrchestrator will call repeatedly.
        """
        self.main_logger.info(f"Generating a chunk of up to {n_positions} self-play positions...")

        chunk_data = []
        positions_in_chunk = 0
        games_in_chunk = 0
        
        while positions_in_chunk < n_positions:
            start_time_game = time.time()
            
            # The play_one_game method returns the data for a single game
            training_data = self.game_manager.run_training_loop(self.game_number)
            
            end_time_game = time.time()
            
            num_new_positions = len(training_data)
            chunk_data.extend(training_data)
            positions_in_chunk += num_new_positions
            
            self.main_logger.info(
                f"Game {self.game_number} completed in {end_time_game - start_time_game:.2f}s "
                f"({num_new_positions} positions). Current chunk total: {positions_in_chunk}. "
            )
            self.game_number += 1
            games_in_chunk += 1
            
        return chunk_data, games_in_chunk