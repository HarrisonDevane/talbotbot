import chess.engine
import os

class LeelaPlayer:
    def __init__(self, name="Leela", time_limit=0.1):
        self.name = name
        self.time_limit = time_limit
        self.engine = chess.engine.SimpleEngine.popen_uci(
            os.path.join(os.path.dirname(__file__), '..', 'data/engine', 'leela', 'lc0.exe'))
        print(self.engine)

    def get_move(self, board):
        # Ask Stockfish to find the best move
        result = self.engine.play(board, chess.engine.Limit(time=self.time_limit))
        return result.move

    def is_human(self):
        return False

    def close(self):
        self.engine.quit()