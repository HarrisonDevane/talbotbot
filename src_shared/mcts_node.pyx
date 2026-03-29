# distutils: language = c
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: nonecheck=False

import math
import torch
import chess

# Value Sign Convention:
# value_sum accumulates from this node's own perspective (positive = good for the player to move here).
# When ranking children, q_val negates value_sum to convert to the parent's perspective.
# forced_outcome uses this node's perspective: -1 = forced win for the player here, 1 = forced loss, 0 = draw.

# ==============================================================================
# NEGAMAX VALUE CONVENTION
# ==============================================================================
# 1. NODE PERSPECTIVE (POV): 
#    'value_sum' is stored from the node's OWN perspective. 
#    A positive value means "The player whose turn it is AT THIS NODE is winning."
#    This is maintained by flipping the sign at every ply during backpropagation
#    (e.g., value = -value) in mcts_engine.pyx.
#
# 2. PARENT SELECTION (RANKING):
#    When a parent evaluates its children (moves), it must see them from ITS POV.
#    Since a "good" move for the child is "bad" for the parent, we negate:
#    q_val = -value_sum / visits.
#    This converts the child's "Good for me" into the parent's "Good for parent."
#
# 3. UNVISITED NODES (V_MIX):
#    v_mix is calculated at the parent level as a weighted average of the 
#    already-negated (parent-relative) q_vals of visited children.
#    Therefore, v_mix is already in the PARENT'S perspective.
#    To keep unvisited nodes on the same scale, we assign: q_val = v_mix.
#
# 4. TERMINAL SOLVER (FORCED OUTCOME):
#    Also follows the Node's Own POV:
#    -1 = FORCED WIN for the player moving at this node.
#     1 = FORCED LOSS for the player moving at this node.
#     0 = DRAW.

# The MCTSNode is defined as a cdef class (Extension Type)
cdef class MCTSNode:
    """
    Cython-optimized MCTS Node.
    """

    def __init__(self, parent: 'MCTSNode' = None, move: chess.Move = None):
        self.parent = parent
        self.move = move
        self.child_list = []
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
        cdef int i
        cdef int num_children = len(self.child_list)
        cdef MCTSNode child

        for i in range(num_children):
            child = <MCTSNode>self.child_list[i]
            if child.visits > 0:
                sum_visits += child.visits
                sum_q_weighted += (child.visits * (-child.value_sum / child.visits))

        return (self.raw_value + sum_q_weighted) / (1.0 + sum_visits)