import chess.engine
import os

class StockfishPlayer:
    def __init__(self, name="Stockfish", time_limit=0.0001):
        self.name = name
        self.time_limit = time_limit
        self.engine = chess.engine.SimpleEngine.popen_uci(
            os.path.join(os.path.dirname(__file__), '..', 'data/engine', 'stockfish', 'stockfish-windows-x86-64-avx2.exe'))

    def get_move(self, board):
        # Ask Stockfish to find the best move
        result = self.engine.play(board, chess.engine.Limit(time=self.time_limit))
        return result.move

    def is_human(self):
        return False

    def close(self):
        self.engine.quit()