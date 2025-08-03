import os
import time
import yaml
import logging
import numpy as np
from datetime import datetime
from tqdm import tqdm

# Assuming these imports are in your project structure
from self_play_agent import TalbotPlayer
from self_play_game_worker import SelfPlayGameWorker

class SelfPlayTask:
    def __init__(self, config_path: str, output_dir: str):
        """
        Initializes the self-play task with configuration and paths.
        """
        self.output_dir = output_dir
        self.config_path = config_path

        # Load configuration from the specified path
        with open(self.config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # Set up loggers
        self.log_dir = os.path.join(self.output_dir, "logs")
        os.makedirs(self.log_dir, exist_ok=True)

        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        
        self.main_logger = self._setup_logger(
            "SelfPlayManager", 
            self.config['logging']['selfplay_main_logging_level'],
            os.path.join(self.log_dir, f"self_play_task_{timestamp}.log")
        )
        
        self.worker_logger = self._setup_logger(
            "SelfPlayWorker",
            self.config['logging']['selfplay_worker_logging_level'],
            os.path.join(self.log_dir, f"self_play_games_{timestamp}.log")
        )

        # Instantiate core components
        self.mcts_player = TalbotPlayer(
            logger=self.worker_logger,
            config=self.config,
        )
        self.game_manager = SelfPlayGameWorker(
            logger=self.worker_logger,
            player=self.mcts_player,
            config=self.config
        )

        # Initialize state variables
        self.all_training_data = []
        self.positions_generated = 0
        self.game_number = 1

    def _setup_logger(self, name: str, level: str, log_file: str):
        """
        Sets up a logger with a specific name and log file.
        """
        logger = logging.getLogger(name)
        logger.setLevel(level)
        if logger.hasHandlers():
            logger.handlers.clear()
        
        formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s")
        file_handler = logging.FileHandler(log_file, mode='w')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        
        return logger

    def run(self):
        """
        The main loop to generate self-play data until the position count is met,
        with a real-time progress bar.
        """
        self.main_logger.info(f"Starting self-play task. Logs are in {self.log_dir}")
        
        num_positions_total = self.config['stored_data']['positions_per_cycle']
        
        # Initialize the progress bar with the total number of positions
        with tqdm(total=num_positions_total, desc="Positions Generated", unit="pos", dynamic_ncols=True) as pbar:
            while self.positions_generated < num_positions_total:
                start_time_game = time.time()
                training_data = self.game_manager.play_one_game(self.game_number)
                end_time_game = time.time()
                
                num_new_positions = len(training_data)
                self.all_training_data.extend(training_data)
                self.positions_generated += num_new_positions
                
                # Update the progress bar with the new positions from this game
                pbar.update(num_new_positions)
                
                self.main_logger.info(
                    f"Game {self.game_number} completed in {end_time_game - start_time_game:.2f}s "
                    f"({num_new_positions} positions). Total: {self.positions_generated}/{num_positions_total}"
                )
                self.game_number += 1
        
        self.main_logger.info("--- Self-play data generation finished. ---")
        self.main_logger.info(f"Final count of games completed: {self.game_number - 1}")
        self.main_logger.info(f"Final count of positions generated: {self.positions_generated}")
        
        # Return the collected data directly
        return self.all_training_data
