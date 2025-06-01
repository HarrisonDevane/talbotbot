#!/usr/bin/env py

import tkinter as tk
import argparse
from gui import ChessGUI
from game_controller import GameController
from players import HumanPlayer, StockfishPlayer, LeelaPlayer



def parse_args():
    parser = argparse.ArgumentParser(description="Talbot Chess Engine")

    parser.add_argument('--white', choices=['human', 'stockfish', 'leela'], default='human',
                        help="Type of player for white")
    parser.add_argument('--black', choices=['human', 'stockfish', 'leela'], default='stockfish',
                        help="Type of player for black")
    parser.add_argument('--gui', action='store_true',
                        help="Flag for a GUI to be displayed")
    
    parser.add_argument('--num_games', type=int, default = 5,
                    help="Number of games played")
    
    return parser.parse_args()

def main():
    args = parse_args()

    if args.white == "human":
        white_player = HumanPlayer()
    elif args.white == "stockfish":
        white_player = StockfishPlayer()
    elif args.white == "leela":
        white_player = LeelaPlayer()

    if args.black == "human":
        black_player = HumanPlayer()
    elif args.black == "stockfish":
        black_player = StockfishPlayer()
    elif args.black == "leela":
        black_player = LeelaPlayer()

    # Create root window and GUI

    if args.gui:
        root = tk.Tk()
        gui = ChessGUI(root)
    else:
        gui = None

    # Create game controller with optional eval engine (can be None)
    controller = GameController(
        white_player=white_player,
        black_player=black_player,
        num_games=args.num_games,
        gui=gui,
        eval_engine=None
    )

    if args.gui:
        gui.set_controller(controller)

    controller.start_game()

    if args.gui:
        root.mainloop()

if __name__ == "__main__":
    main()