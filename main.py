import tkinter as tk
import os
from chess_gui import ChessGUI
from game_controller import GameController
from players import HumanPlayer, StockfishPlayer, LeelaPlayer, TalbotPlayer

# Import configuration settings from config.py
import config
import logging

log_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "players/talbot_engine/logs"))
log_file_path = os.path.join(log_dir, "talbot_inference.log")
if os.path.exists(log_file_path): os.remove(log_file_path)

# For detailed step-by-step logs, change level to logging.DEBUG
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename=log_file_path,
    filemode='a'
)

def main():
    # Load settings from config.py
    white_player_type = config.WHITE_PLAYER_TYPE
    black_player_type = config.BLACK_PLAYER_TYPE
    use_gui = config.USE_GUI
    num_games = config.NUM_GAMES
    initial_fen = config.INITIAL_FEN

    talbot_model_path = os.path.abspath(config.TALBOT_MODEL_PATH)
    talbot_resblocks = config.TALBOT_RESBLOCKS
    talbot_cpuct = config.TALBOT_CPUCT
    talbot_batchsize = config.TALBOT_BATCHSIZE

    time_talbot = config.TALBOT_TIME_PER_MOVE
    time_stockfish = config.STOCKFISH_TIME_PER_MOVE
    time_leela = config.LEELA_TIME_PER_MOVE

    white_player_time = None
    black_player_time = None

    # Initialize players based on config settings
    match white_player_type:
        case "human":
            white_player = HumanPlayer()
        case "stockfish":
            white_player = StockfishPlayer()
            white_player_time = time_stockfish
        case "leela":
            white_player = LeelaPlayer()
            white_player_time = time_leela
        case "talbot":
            white_player = TalbotPlayer(
                model_path=talbot_model_path,
                num_residual_blocks=talbot_resblocks,
                cpuct=talbot_cpuct,
                batch_size=talbot_batchsize
            )
            white_player_time = time_talbot

    # Initialize black player using match/case
    match black_player_type:
        case "human":
            black_player = HumanPlayer()
        case "stockfish":
            black_player = StockfishPlayer()
            black_player_time = time_stockfish
        case "leela":
            black_player = LeelaPlayer()
            black_player_time = time_leela
        case "talbot":
            black_player = TalbotPlayer(
                model_path=talbot_model_path,
                num_residual_blocks=talbot_resblocks,
                cpuct=talbot_cpuct,
                batch_size=talbot_batchsize
            )
            black_player_time = time_talbot

    # Create root window and GUI
    if use_gui:
        root = tk.Tk()
        gui = ChessGUI(root)
    else:
        root = None # Ensure root is None if no GUI

    # Create game controller with optional eval engine (can be None)
    controller = GameController(
        white_player=white_player,
        black_player=black_player,
        white_player_time=white_player_time,
        black_player_time=black_player_time,
        num_games=num_games,
        gui=gui,
        initial_fen=initial_fen
    )

    if use_gui:
        gui.set_controller(controller)

    controller.start_game()

    if use_gui:
        root.mainloop()
    
    # Close players if they have a close method (important for engine players)
    if hasattr(white_player, 'close'):
        white_player.close()
    if hasattr(black_player, 'close'):
        black_player.close()

if __name__ == "__main__":
    main()
