# COPYRIGHT 2021 D-WAVE SYSTEMS INC. ("D-WAVE") All Rights Reserved.
# This software is D-Wave confidential and proprietary information. 
# distutils: language = c++
# cython: language_level=3

from libcpp.vector cimport vector
from libcpp.utility cimport pair
cimport dimod
cimport numpy as np
from libcpp cimport bool


cdef extern from "swap_heuristic.h":
    vector[pair[vector[int], double]] solve(vector[vector[double]],
                                            vector[vector[int]],
                                            double, double, double, int)


cpdef solve_tsp(vector[vector[double]] distances, states,
              double beta,
              double max_beta, double scale, int n = 12):
    cdef vector[vector[int]] states_c
    if states is None:
        import numpy as np
        m = len(distances)
        r = list(range(m))
        states = []
        for _ in range(n):
            np.random.shuffle(r)
            states.append(r)
        states_c = states
    else:
        states_c = states
    return solve(distances, states_c, beta, max_beta, scale, n)
