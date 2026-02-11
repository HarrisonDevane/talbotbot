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
        self.raw_value = 0
        self.q_val = 0.0
        self.q_norm = 0.0

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


    cpdef double calculate_gumbel_score(self, double gumbel_c_base, double gumbel_c_scale, double max_visits, double min_q, double max_q,double gumbel_min_scale, double v_mix):
        """
        Calculates and updates the gumbel_score for this node.
        """
        cdef double q_val
        cdef double q_norm
        cdef double logit
        cdef double sigma

        if self.visits > 0:
            self.q_val = -self.value_sum / self.visits
        else:
            self.q_val = v_mix
            
        # 2. Normalize    
        scale = max_q - min_q
        if scale < gumbel_min_scale:
            scale = gumbel_min_scale
            
        self.q_norm = (self.q_val - min_q) / scale

        if self.q_norm < 0.0: self.q_norm = 0.0
        if self.q_norm > 1.0: self.q_norm = 1.0

        # 3. Logit
        logit = math.log(max(self.prior_probability_from_parent, 1e-8))

        # 4. Sigma (Corrected Formula)
        sigma = (gumbel_c_base + max_visits) * gumbel_c_scale

        # 5. Score
        self.gumbel_score = logit + self.gumbel_noise + (sigma * self.q_norm)
        
        return self.gumbel_score

    cpdef double calculate_v_mix(self):
        cdef double sum_visits = 0.0
        cdef double sum_visited_prob = 0.0
        cdef double sum_visited_q_weighted = 0.0
        cdef double child_q_parent_perspective

        # 1. Gather stats from VISITED children only
        for child in self.children.values():
            if child.visits > 0:
                sum_visits += child.visits
                sum_visited_prob += child.prior_probability_from_parent
                
                # Invert child value for parent perspective
                child_q_parent_perspective = -child.value_sum / child.visits
                sum_visited_q_weighted += (child.prior_probability_from_parent * child_q_parent_perspective)

        if sum_visits == 0:
            return self.raw_value

        # 2. Eq 33 Implementation
        cdef double scaling = 0.0
        if sum_visited_prob > 1e-8:
            scaling = sum_visits / sum_visited_prob
            
        cdef double term_2 = scaling * sum_visited_q_weighted
        return (1.0 / (1.0 + sum_visits)) * (self.raw_value + term_2)