import tkinter as tk
import os
# import argparse # No longer needed for these args
from chess_gui import ChessGUI
from game_controller import GameController
from players import HumanPlayer, StockfishPlayer, LeelaPlayer, TalbotPlayer, TalbotPlayerMCTS

# Import configuration settings from config.py
import config


def main():
    # Load settings from config.py
    white_player_type = config.WHITE_PLAYER_TYPE
    black_player_type = config.BLACK_PLAYER_TYPE
    use_gui = config.USE_GUI
    num_games = config.NUM_GAMES
    initial_fen = config.INITIAL_FEN

    talbot_model_path = os.path.abspath(config.TALBOT_MODEL_PATH)
    talbot_resblocks = config.TALBOT_RESBLOCKS
    talbot_timelimit = config.TALBOT_TIMELIMIT
    talbot_cpuct = config.TALBOT_CPUCT
    talbot_batchsize = config.TALBOT_BATCHSIZE

    # Initialize players based on config settings
    match white_player_type:
        case "human":
            white_player = HumanPlayer()
        case "stockfish":
            white_player = StockfishPlayer()
        case "leela":
            white_player = LeelaPlayer()
        case "talbot":
            white_player = TalbotPlayer(model_path=talbot_model_path)
        case "talbotMCTS":
            white_player = TalbotPlayerMCTS(
                model_path=talbot_model_path,
                num_residual_blocks=talbot_resblocks,
                time_per_move=talbot_timelimit,
                cpuct=talbot_cpuct,
                batch_size=talbot_batchsize
            )

    # Initialize black player using match/case
    match black_player_type:
        case "human":
            black_player = HumanPlayer()
        case "stockfish":
            black_player = StockfishPlayer()
        case "leela":
            black_player = LeelaPlayer()
        case "talbot":
            black_player = TalbotPlayer(model_path=talbot_model_path)
        case "talbotMCTS":
            black_player = TalbotPlayerMCTS(
                model_path=talbot_model_path,
                num_residual_blocks=talbot_resblocks,
                time_per_move=talbot_timelimit,
                cpuct=talbot_cpuct,
                batch_size=talbot_batchsize
            )

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
