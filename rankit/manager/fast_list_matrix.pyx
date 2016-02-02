from __future__ import division
import numpy as np
cimport numpy as np
cimport cython


@cython.boundscheck(False)
@cython.wraparound(False)
def fast_generate_rank_difference_matrix(np.ndarray[np.int32_t, ndim=2]lst,
                                         np.ndarray[np.float32_t, ndim=2] D):
    cdef unsigned int i, j, k;
    for i in xrange(D.shape[0]):
        for j in xrange(i, D.shape[0]):
            if i==j: continue;
            for k in xrange(lst.shape[1]):
                if lst[i, k]>lst[j, k]:
                    D[i, j]+=lst[i, k]-lst[j, k]
                else:
                    D[j, i]+=lst[j, k]-lst[i, k]


@cython.boundscheck(False)
@cython.wraparound(False)
def fast_generate_list_difference_matrix(np.ndarray[np.int32_t, ndim=2]lst,
                                         np.ndarray[np.float32_t, ndim=2] D):
    cdef unsigned int i, j, k;
    for i in xrange(D.shape[0]):
        for j in xrange(i, D.shape[0]):
            if i==j: continue;
            for k in xrange(lst.shape[1]):
                if lst[i, k]>lst[j, k]:
                    D[i, j]+=1
                else:
                    D[j, i]+=1
