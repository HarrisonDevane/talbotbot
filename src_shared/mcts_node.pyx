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
        self.prior_probabilities = None
        self.forced_outcome = None
        
        # Initialize C-typed numeric values
        self.visits = 0
        self.value_sum = 0.0
        self.prior_probability_from_parent = 0.0
        
        # Initialize Python object variables to None
        self.uid = None
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