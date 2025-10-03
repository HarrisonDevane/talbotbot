cdef class MCTSNode:
    # C-Typed Pointers for node traversal (must be declared)
    cdef public MCTSNode parent # <-- Crucial for cdef MCTSNode declarations
    
    # FIX: Declared 'move' as a generic Python object (object)
    cdef public object move
    cdef public object _board
    
    # --- C-Typed Attributes accessed by MCTSEngine (Remain C-types) ---
    cdef public int visits
    cdef public double value_sum
    cdef public double prior_probability_from_parent
    cdef public bint expanded
    cdef public bint selected
    
    # --- FIXES: Attributes that need to be None (CHANGE FROM int TO object) ---
    cdef public object forced_outcome
    cdef public object distance_to_mate
    
    cdef public object prior_probabilities
    cdef public object children
    
    # --- C-Typed Method signature accessed by MCTSEngine ---
    # This must match the signature in mcts_node.pyx
    cpdef double uct_score(self, double cpuct, double prior_probability_for_this_move, double sqrt_parent_visits_term)