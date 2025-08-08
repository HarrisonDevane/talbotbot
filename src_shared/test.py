import chess
import chess.pgn
import utils
import numpy as np

# Create a board and a game
board = chess.Board()
game = chess.pgn.Game()
node = game

# Knight repetition dance: Ng1–f3, Nf3–g1
moves = [
    chess.Move.from_uci("g1f3"),
    chess.Move.from_uci("g8f6"),
    chess.Move.from_uci("f3g1"),
    chess.Move.from_uci("f6g8"),
    chess.Move.from_uci("g1f3"),
    chess.Move.from_uci("g8f6"),
    chess.Move.from_uci("f3g1"),
    chess.Move.from_uci("f6g8"),
    chess.Move.from_uci("g1f3"),
    chess.Move.from_uci("g8f6"),
]

# Push moves and build the PGN
for move in moves:
    board.push(move)
    node = node.add_variation(move)

# Export PGN
print("--- PGN ---")
print(game)

# Check repetition planes
from pprint import pprint
planes = utils.board_to_tensor_68(board)
print("\nRepetition Planes (66 and 67):")
print("Plane 66 (2-fold repetition):", np.unique(planes[66]))
print("Plane 67 (3-fold repetition):", np.unique(planes[67]))