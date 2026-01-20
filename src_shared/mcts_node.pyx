# distutils: language = c
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: nonecheck=False

import math
import torch
import chess

# The MCTSNode is defined as a cdef class (Extension Type)
cdef class MCTSNode:
    """
    Cython-optimized MCTS Node.
    """

    def __init__(self, board: chess.Board = None, parent: 'MCTSNode' = None, move: chess.Move = None):
        # Assign Python objects directly
        self._board = board
        self.parent = parent
        self.move = move
        self.children = {}
        self.forced_outcome = None
        
        # Initialize C-typed numeric values
        self.visits = 0
        self.value_sum = 0.0
        self.prior_probability_from_parent = 0.0

        # Gumbel vars        
        self.gumbel_noise = 0.0
        self.gumbel_score = 0.0
        
        # Initialize Python object variables to None
        self.distance_to_mate = None
        
        # Initialize C-typed booleans
        self.expanded = False
        self.selected = False


    @property
    def board(self) -> chess.Board:
        # Lazily create board to save memory
        cdef object current_board
        
        if self._board is None and self.parent is not None:
            current_board = self.parent.board.copy()
            current_board.push(self.move)
            self._board = current_board
            
        return self._board


    cpdef double uct_score(self, double cpuct, double prior_probability_for_this_move, double sqrt_parent_visits_term):
        """
        Cython-optimized UCT score calculation (still fast).
        """
        # C-level local variables for the calculation
        cdef double Q, U
        cdef int visits_plus_one = self.visits + 1
        
        if self.visits == 0:
            return float('inf')

        Q = -self.value_sum / self.visits 
        U = cpuct * prior_probability_for_this_move * sqrt_parent_visits_term / visits_plus_one

        return Q + U


    cpdef double calculate_gumbel_score(self, double gumbel_c_base, double gumbel_c_scale, double max_visits):
        """
        Calculates and updates the gumbel_score for this node.
        """
        cdef double q_val
        cdef double q_norm
        cdef double logit
        cdef double sigma

        if self.visits > 0:
            q_val = -self.value_sum / self.visits
        else:
            q_val = self.parent.value_sum / self.parent.visits
            
        # 2. Normalize
        q_norm = (q_val + 1.0) / 2.0
        
        # 3. Logit
        logit = math.log(max(self.prior_probability_from_parent, 1e-8))

        # 4. Sigma (Corrected Formula)
        # DeepMind: (c_visit + N_max) * c_scale
        sigma = (gumbel_c_base + max_visits) * gumbel_c_scale

        # 5. Score
        self.gumbel_score = logit + self.gumbel_noise + (sigma * q_norm)
        
        return self.gumbel_score