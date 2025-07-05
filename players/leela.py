import chess.engine
import os

class LeelaPlayer:
    def __init__(self, name="Leela"):
        self.name = name
        self.engine = chess.engine.SimpleEngine.popen_uci(
            os.path.join(os.path.dirname(__file__), '..', 'data/engine', 'leela', 'lc0.exe'))

    def get_move(self, board: chess.Board, time_per_move: int = None, depth_limit: int = None) -> chess.Move:
        result = self.engine.play(board, chess.engine.Limit(time=time_per_move))
        return result.move

    def is_human(self):
        return False

    def close(self):
        self.engine.quit()