cdef class MCTSNode:
    # C-Typed Pointers for node traversal (must be declared)
    cdef public MCTSNode parent # <-- Crucial for cdef MCTSNode declarations
    
    cdef public object move
    cdef public object pending_logits
    cdef public int visits
    cdef public int num_unselected_children
    cdef public double value_sum
    cdef public double raw_logit
    cdef public double raw_value
    cdef public double gumbel_noise
    cdef public double gumbel_score
    cdef public double q_val
    cdef public double q_norm
    cdef public bint expanded
    cdef public bint selected
    
    # --- FIXES: Attributes that need to be None (CHANGE FROM int TO object) ---
    cdef public object forced_outcome
    cdef public object distance_to_mate
    cdef public object children
    
    # --- C-Typed Method signature accessed by MCTSEngine ---
    cpdef double calculate_gumbel_score(self, double gumbel_c_visit, double gumbel_c_scale, double max_visits, double v_mix)
    cpdef double calculate_v_mix(self)