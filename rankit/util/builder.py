import numpy as np
from fast_converter import fast_consistancy_matrix_build

def ConsistancyMatrix(D):
    """
    For input D, each element indicates how much superior i is compared to j.
    c_ij = #{ k | d_ik < d_jk } + # how many contests j is superior to i
           #{ k | d_ki > d_kj }   # how many other items think that j is more hard to compete than i
    C_ij is the potential that j is superior to i.
    """
    C = np.zeros(D.shape, dtype=np.int32)
    D = np.require(D, dtype=np.float32)
    E = D.T;
    fast_consistancy_matrix_build(D, E, C)
    return C
