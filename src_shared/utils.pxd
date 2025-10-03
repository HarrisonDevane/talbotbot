# src_shared/utils.pxd
# Cython header file for utility functions (cimport definitions)

# C-level constants (must match internal names in .pyx)
cdef int _BOARD_DIM
cdef int _INPUT_CHANNELS
cdef int _TOTAL_INPUT_SIZE
cdef int _POLICY_CHANNELS
cdef int _TOTAL_POLICY_MOVES

# C-level function declarations

# 1. 'cdef' function (per your exception)
cdef tuple convert_coords(int rank, int file)

# 2. 'cpdef' functions (declared as cdef for fast cimport/C-level calling)
cdef int policy_components_to_flat_index(int from_row, int from_col, int channel)
cdef tuple policy_flat_index_to_components(int flat_index)