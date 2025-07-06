import tkinter as tk
import os
import logging
import sys
from datetime import datetime
import yaml

# Determine absolute paths for src and config directories
current_script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_script_dir, ".."))

src_dir = os.path.join(project_root, "src")
config_dir = os.path.join(project_root, "config")

# Add src and config directories to Python's system path
sys.path.insert(0, src_dir)
sys.path.insert(0, config_dir)

# Import necessary classes from your project
from chess_gui import ChessGUI
from game_controller import GameController
from players import HumanPlayer, StockfishPlayer, LeelaPlayer, TalbotPlayer

# Setup logging
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../logs/local_inference"))
log_file_path = os.path.join(log_dir, f"talbot_inference_{timestamp}.log")

# For detailed step-by-step logs, change level to logging.DEBUG
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename=log_file_path,
    filemode='w'
)

# Logger for this module
logger = logging.getLogger(__name__)

def main():
    # Construct the full path to the local_config.yaml file
    full_config_path = os.path.join(config_dir, "local_config.yaml")
    
    # Load the configuration from the YAML file
    with open(full_config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Player settings
    player_settings = config.get('player_settings', {})
    white_player_type = player_settings.get('white_player_type')
    black_player_type = player_settings.get('black_player_type')

    # Game settings
    game_settings = config.get('game_settings', {})
    use_gui = game_settings.get('use_gui')
    num_games = game_settings.get('num_games')
    initial_fen = game_settings.get('initial_fen')

    # Talbot AI model settings
    talbot_config = config.get('talbot', {})
    talbot_model_path = os.path.abspath(talbot_config.get('model_path'))
    talbot_resblocks = talbot_config.get('resblocks')
    talbot_cpuct = talbot_config.get('cpuct')
    talbot_batchsize = talbot_config.get('batchsize')
    talbot_time_per_move = talbot_config.get('time_per_move')

    # Time settings for other engines
    engine_times = config.get('engine_times', {})
    stockfish_time_per_move = engine_times.get('stockfish_time_per_move')
    leela_time_per_move = engine_times.get('leela_time_per_move')

    # --- Player Initialization ---
    white_player_time = None
    black_player_time = None

    # Initialize white player based on type
    match white_player_type:
        case "human":
            white_player = HumanPlayer()
        case "stockfish":
            white_player = StockfishPlayer()
            white_player_time = stockfish_time_per_move
        case "leela":
            white_player = LeelaPlayer()
            white_player_time = leela_time_per_move
        case "talbot":
            white_player = TalbotPlayer(
                model_path=talbot_model_path,
                num_residual_blocks=talbot_resblocks,
                cpuct=talbot_cpuct,
                batch_size=talbot_batchsize,
            )
            white_player_time = talbot_time_per_move
        case _:
            logger.error(f"Unknown white player type: {white_player_type}")
            sys.exit(1)

    # Initialize black player based on type
    match black_player_type:
        case "human":
            black_player = HumanPlayer()
        case "stockfish":
            black_player = StockfishPlayer()
            black_player_time = stockfish_time_per_move
        case "leela":
            black_player = LeelaPlayer()
            black_player_time = leela_time_per_move
        case "talbot":
            black_player = TalbotPlayer(
                model_path=talbot_model_path,
                num_residual_blocks=talbot_resblocks,
                cpuct=talbot_cpuct,
                batch_size=talbot_batchsize,
            )
            black_player_time = talbot_time_per_move
        case _:
            logger.error(f"Unknown black player type: {black_player_type}")
            sys.exit(1)

    # Create root window and GUI if enabled
    if use_gui:
        root = tk.Tk()
        gui = ChessGUI(root)
    else:
        root = None 
        gui = None 

    # Create game controller
    controller = GameController(
        white_player=white_player,
        black_player=black_player,
        white_player_time=white_player_time,
        black_player_time=black_player_time,
        num_games=num_games,
        gui=gui,
        initial_fen=initial_fen,
    )

    # Set controller for GUI if it exists
    if use_gui:
        gui.set_controller(controller)

    # Start the game(s)
    controller.start_game()

    # Run the Tkinter event loop if GUI is used
    if use_gui:
        root.mainloop()

if __name__ == "__main__":
    main()
