import tkinter as tk
import os
import argparse
from chess_gui import ChessGUI
from game_controller import GameController
from players import HumanPlayer, StockfishPlayer, LeelaPlayer, TalbotPlayer, TalbotPlayerMCTS

# Current talbot dir
TALBOT_PATH = "training/supervised/v2_pol_mvplayed_val_sfeval/model/best_chess_ai_model.pth"

def parse_args():
    parser = argparse.ArgumentParser(description="Talbot Chess Engine")

    parser.add_argument('--white', choices=['human', 'stockfish', 'leela', 'talbot'], default='human',
                        help="Type of player for white")
    parser.add_argument('--black', choices=['human', 'stockfish', 'leela', 'talbot'], default='stockfish',
                        help="Type of player for black")
    parser.add_argument('--gui', action='store_true',
                        help="Flag for a GUI to be displayed")
    
    parser.add_argument('--num_games', type=int, default = 5,
                    help="Number of games played")
    
    parser.add_argument('--fen', type=str, default=None,
                    help="Initial board state in FEN format. If not provided, starts from standard initial position.")
    
    return parser.parse_args()

def main():
    args = parse_args()

    # Validate FEN if provided
    initial_fen = None

    talbot_path = os.path.abspath(os.path.join(os.path.dirname(__file__), TALBOT_PATH))

    if args.white == "human":
        white_player = HumanPlayer()
    elif args.white == "stockfish":
        white_player = StockfishPlayer()
    elif args.white == "leela":
        white_player = LeelaPlayer()
    elif args.white == "talbot":
        white_player = TalbotPlayer(model_path=talbot_path)

    if args.black == "human":
        black_player = HumanPlayer()
    elif args.black == "stockfish":
        black_player = StockfishPlayer()
    elif args.black == "leela":
        black_player = LeelaPlayer()
    elif args.black == "talbot":
        black_player = TalbotPlayerMCTS(model_path=talbot_path)

    # Create root window and GUI
    if args.gui:
        root = tk.Tk()
        gui = ChessGUI(root)
    else:
        gui = None

    if args.fen:
        initial_fen = args.fen

    # Create game controller with optional eval engine (can be None)
    controller = GameController(
        white_player=white_player,
        black_player=black_player,
        num_games=args.num_games,
        gui=gui,
        initial_fen=initial_fen
    )

    if args.gui:
        gui.set_controller(controller)

    controller.start_game()

    if args.gui:
        root.mainloop()

if __name__ == "__main__":
    main()