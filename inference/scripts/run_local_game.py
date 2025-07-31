import tkinter as tk
import os
import logging
import sys
from datetime import datetime
import yaml

current_script_dir = os.path.dirname(os.path.abspath(__file__))
inference_root = os.path.abspath(os.path.join(current_script_dir, ".."))
project_root = os.path.abspath(os.path.join(current_script_dir, "../.."))

sys.path.insert(0, project_root)
sys.path.insert(0, inference_root)
sys.path.insert(0, project_root)


from agents.human import HumanPlayer
from agents.stockfish import StockfishPlayer
from agents.leela import LeelaPlayer
from agents.talbot import TalbotPlayer
from chess_gui import ChessGUI
from game_controller import GameController

def main():
    # Construct the full path to the local_config.yaml file
    full_config_path = os.path.join(inference_root, "config/local_config.yaml")
    
    # Load the configuration from the YAML file
    with open(full_config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Player settings
    game_config = config['game']
    talbot_config = config['talbot']
    logging_config = config['logging']


        # Set up logging dir + lichess logging
    log_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../logs/local_inference", datetime.now().strftime("%Y-%m-%d_%H-%M-%S")))
    os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger("local_main")
    logger.setLevel(logging_config['local_logging_level'])

    if not logger.handlers:
        handler = logging.FileHandler(os.path.join(log_dir, "main.log"), mode='w')
        formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    # Initialize white player based on type
    match game_config['white_player']:
        case "human":
            white_player=HumanPlayer()
        case "stockfish":
            white_player=StockfishPlayer()
        case "leela":
            white_player=LeelaPlayer()
        case "talbot":
            white_player = TalbotPlayer(
                model_path=talbot_config['model_path'],
                logger=logger,
                num_input_planes=talbot_config['input_planes'],
                num_residual_blocks=talbot_config['resblocks'],
                cpuct=talbot_config['cpuct'],
                batch_size=talbot_config['batchsize']
            )
        case _:
            logger.error(f"Unknown white player type: {game_config['white_player']}")
            sys.exit(1)

    # Initialize black player based on type
    match game_config['black_player']:
        case "human":
            black_player = HumanPlayer()
        case "stockfish":
            black_player = StockfishPlayer()
        case "leela":
            black_player = LeelaPlayer()
        case "talbot":
            black_player = TalbotPlayer(
                model_path=talbot_config['model_path'],
                logger=logger,
                num_input_planes=talbot_config['input_planes'],
                num_residual_blocks=talbot_config['resblocks'],
                cpuct=talbot_config['cpuct'],
                batch_size=talbot_config['batchsize']
            )
        case _:
            logger.error(f"Unknown black player type: {game_config['black_player']}")
            sys.exit(1)

    # Create root window and GUI
    root = tk.Tk()
    gui = ChessGUI(root, logger,)

    # Create game controller
    controller = GameController(
        white_player=white_player,
        black_player=black_player,
        logger=logger,
        white_player_time=game_config['white_time_per_move'],
        black_player_time=game_config['black_time_per_move'],
        num_games=game_config['total_games'],
        gui=gui,
        initial_fen=game_config['initial_fen'],
    )

    gui.set_controller(controller)

    # Start the game(s)
    controller.start_game()

    # Run the Tkinter event loop if GUI is used
    root.mainloop()

if __name__ == "__main__":
    main()
