import chess

class MCTSNode:
    """
    Represents a single state (board position) in our MCTS tree.
    """
    def __init__(self, board: chess.Board = None, parent=None, move: chess.Move = None):
        self._board = board
        self.parent = parent
        self.move = move
        self.children = {}
        self.visits = 0
        self.value_sum = 0.0
        self.prior_probabilities = None
        self.prior_probability_from_parent = 0.0
        self.is_expanded = False
        self.is_queued_for_inference = False

    @property
    def board(self) -> chess.Board:
        # Lazily create board to save memory
        if self._board is None and self.parent is not None:
            self._board = self.parent.board.copy()
            self._board.push(self.move)
        return self._board

    def is_leaf(self) -> bool:
        return not self.children and not self.is_expanded

    def is_root(self) -> bool:
        return self.parent is None

    def uct_score(self, cpuct: float, prior_probability_for_this_move: float, sqrt_parent_visits_term: float) -> float:
        if self.visits == 0:
            return float('inf')

        Q = -self.value_sum / self.visits
        U = cpuct * prior_probability_for_this_move * sqrt_parent_visits_term / (1 + self.visits)

        return Q + U