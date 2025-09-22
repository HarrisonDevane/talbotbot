import chess
import math

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
        self.uid = None
        self.prior_probabilities = None
        self.prior_probability_from_parent = 0.0
        self.expanded = False
        self.selected = False

        # Random action value estimation (RAVE) params
        self.rave_visits = 0
        self.rave_value_sum = 0.0

        # Terminal params
        self.forced_outcome = None
        self.distance_to_mate = None


    @property
    def board(self) -> chess.Board:
        # Lazily create board to save memory
        if self._board is None and self.parent is not None:
            self._board = self.parent.board.copy()
            self._board.push(self.move)
        return self._board


    def is_leaf(self) -> bool:
        return not self.children and not self.expanded


    def is_root(self) -> bool:
        return self.parent is None


    def uct_score(self, cpuct: float, k_rave: float, prior_probability_for_this_move: float, sqrt_parent_visits_term: float) -> float:
        """
        Calculates the UCT score using the specified RAVE weighting formula.
        
        Args:
            cpuct: A constant controlling the exploration vs exploitation trade-off.
            prior_probability_for_this_move: The policy-based prior probability for this move.
            sqrt_parent_visits_term: The square root of the parent's total visits.
            k_rave: A constant 'k' used in the RAVE beta weighting formula.
        """
        # If the node has not been visited, it has the highest priority for exploration.
        if self.visits == 0:
            return float('inf')

        Q = -self.value_sum / self.visits

        if self.rave_visits > 0:
            Q_rave = -self.rave_value_sum / self.rave_visits
            beta = math.sqrt(k_rave / (3 * self.visits + k_rave))
            Q = Q_rave * beta + Q * (1 - beta)

        U = cpuct * prior_probability_for_this_move * sqrt_parent_visits_term / (1 + self.visits)

        return Q + U