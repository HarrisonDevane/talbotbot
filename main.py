#!/usr/bin/env py

import tkinter as tk
import os
import chess.engine
from gui import ChessGUI
from game_controller import GameController
from players import HumanPlayer, StockfishPlayer, LeelaPlayer  # assuming you have these player classes

def main():
    # Create players (could be HumanPlayer, StockfishPlayer, or your CNNPlayer)
    white_player = StockfishPlayer()          # human plays white
    black_player = LeelaPlayer(time_limit=2)  # engine plays black

    # Create root window and GUI
    root = tk.Tk()
    gui = ChessGUI(root)

    # Create game controller with optional eval engine (can be None)
    controller = GameController(
        white_player=white_player,
        black_player=black_player,
        eval_engine=None,  # or some EvalEngine object
        gui=gui
    )

    # Optionally bind GUI input to controller so human moves get pushed properly:
    gui.set_controller(controller)
    controller.start_game()
    # Run the GUI main loop, but game controller handles the moves
    root.mainloop()

    # Cleanup engine after GUI closes
    # stockfish_engine.quit()

if __name__ == "__main__":
    main()