import os
import time
import yaml
import logging
import random
import sys
import torch
from pathlib import Path
from datetime import datetime

# Assuming these imports are in your project structure
from self_play_agent import TalbotPlayer
from self_play_game_worker import SelfPlayGameWorker

current_script_dir = os.path.dirname(os.path.abspath(__file__))

def setup_logger(name, level, log_file=None):
    """
    Sets up a logger with a specific name, logging level, and an optional log file handler.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Clear existing handlers to prevent duplicate logs
    if logger.hasHandlers():
        logger.handlers.clear()
    
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s")
    
    # Create a single file handler for all log output
    if log_file:
        file_handler = logging.FileHandler(log_file, mode='w')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger

def run_self_play():
    """
    The main function to orchestrate the self-play loop in a single process.
    """
    config_file_path = os.path.join(current_script_dir, "self_play_config.yaml")

    log_dir_base = os.path.abspath(os.path.join(current_script_dir, "../logs/self_play"))
    run_timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    run_log_dir = os.path.join(log_dir_base, run_timestamp)
    Path(run_log_dir).mkdir(parents=True, exist_ok=True)
    
    # The main process log file
    main_log_file = os.path.join(run_log_dir, "run.log")

    # Load configuration
    with open(config_file_path, 'r') as f:
        config = yaml.safe_load(f)

    # Add the base log directory to the config to pass to all processes
    config['logging']['run_log_dir'] = run_log_dir

    # --- Create separate loggers for the main process and the worker ---
    main_logger = setup_logger(
        "SelfPlayManager", 
        level=config['logging']['main_logging_level'],
        log_file=main_log_file
    )
    
    worker_log_file = os.path.join(run_log_dir, "worker.log")
    worker_logger = setup_logger(
        "SelfPlayWorker",
        level=config['logging']['worker_logging_level'],
        log_file=worker_log_file
    )
    
    main_logger.info(f"Starting self-play manager. Main logs are being saved to {main_log_file}")
    main_logger.info(f"Worker logs are being saved to {worker_log_file}")
    main_logger.info(f"Config loaded: {config}")

    num_games_total = config['self_play']['num_games_total']
    
    # Instantiate the TalbotPlayer with the specific worker logger
    mcts_player = TalbotPlayer(
        logger=worker_logger,
        config=config
    )

    # Instantiate the game manager with the specific worker logger
    game_manager = SelfPlayGameWorker(
        logger=worker_logger,
        player=mcts_player,
        config=config
    )
    
    games_completed = []
    for pgn in game_manager.start_game_series():
        games_completed.append(pgn)
        main_logger.info(f"Game completed ({len(games_completed)}/{num_games_total}). PGN: \n{pgn}")

    main_logger.info("--- All games finished. ---")
    
if __name__ == '__main__':
    run_self_play()