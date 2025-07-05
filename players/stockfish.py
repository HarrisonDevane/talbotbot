import chess.engine
import os

class StockfishPlayer:
    def __init__(self, name="Stockfish"):
        self.name = name
        self.engine = chess.engine.SimpleEngine.popen_uci(
            os.path.join(os.path.dirname(__file__), '..', 'data/engine', 'stockfish', 'stockfish-windows-x86-64-avx2.exe'))

    def get_move(self, board: chess.Board, time_per_move: int = None, depth_limit: int = None) -> chess.Move:
        result = self.engine.play(board, chess.engine.Limit(time=time_per_move))
        return result.move

    def is_human(self):
        return False

    def close(self):
        self.engine.quit()