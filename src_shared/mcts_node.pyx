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

    def __init__(self, parent: 'MCTSNode' = None, move: chess.Move = None):
        self.parent = parent
        self.move = move
        self.children = {}
        self.forced_outcome = None
        self.pending_logits = None
        
        # Initialize C-typed numeric values
        self.visits = 0
        self.value_sum = 0.0
        self.raw_logit = 0.0
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

    cpdef double calculate_gumbel_score(self, double gumbel_c_visit, double gumbel_c_scale, double max_visits, double v_mix):
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
        self.q_norm = (self.q_val + 1) / 2

        # 3. Sigma (Corrected Formula)
        sigma = (gumbel_c_visit + max_visits) * gumbel_c_scale

        # 4. Score
        self.gumbel_score = self.raw_logit + self.gumbel_noise + (sigma * self.q_norm)
        
        return self.gumbel_score

    cpdef double calculate_v_mix(self):
        cdef double sum_visits = 0.0
        cdef double sum_q_weighted = 0.0
        
        for child in self.children.values():
            if child.visits > 0:
                sum_visits += child.visits
                sum_q_weighted += (child.visits * (-child.value_sum / child.visits))

        return (self.raw_value + sum_q_weighted) / (1.0 + sum_visits)