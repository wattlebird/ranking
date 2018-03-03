from __future__ import division
import numpy as np
cimport numpy as np
cimport cython


@cython.boundscheck(False)
@cython.wraparound(False)
def fast_colley_build(np.ndarray[np.int32_t, ndim=2] pair,
                      np.ndarray[np.float64_t, ndim=2] rate,
                      np.int32_t dim, np.float64_t tiethr):
    cdef unsigned int i1, i2, i;
    cdef np.ndarray[np.float64_t, ndim=1] s = np.zeros(dim, dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=2] C = np.zeros((dim, dim), dtype=np.float64)
    cdef np.ndarray[np.float64_t, ndim=1] b = np.zeros(dim, dtype=np.float64)
    for i in range(pair.shape[0]):
        i1 = <unsigned int>(pair[i, 0])
        i2 = <unsigned int>(pair[i, 1])
        C[i1, i2]-=rate[i, 2]
        C[i2, i1]-=rate[i, 2]
        if abs(rate[i, 0]-rate[i, 1]) <= tiethr:
            continue; # tie wouldn't affect b
        elif rate[i, 0]>rate[i, 1]:
            b[i1]+=rate[i, 2]
            b[i2]-=rate[i, 2]
        elif rate[i, 0]<rate[i, 1]:
            b[i1]-=rate[i, 2]
            b[i2]+=rate[i, 2]
    s = -np.sum(C, axis=0)
    for i in range(dim):
        C[i,i] = s[i]+2
    b/=2
    b+=1
    return (C, b)
