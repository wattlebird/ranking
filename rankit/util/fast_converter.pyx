from __future__ import division
import numpy as np
cimport numpy as np
cimport cython


@cython.boundscheck(False)
@cython.wraparound(False)
def fast_colley_matrix_build(np.ndarray[np.int32_t, ndim=2] pair,
                             np.ndarray[np.float32_t, ndim=2] rate,
                             np.ndarray[np.float32_t, ndim=2] C):
    cdef unsigned int i1, i2, i;
    cdef np.ndarray[np.float32_t, ndim=1] s = np.empty(C.shape[0], dtype=np.float32)
    for i in xrange(pair.shape[0]):
        i1 = <unsigned int>(pair[i, 0])
        i2 = <unsigned int>(pair[i, 1])
        C[i1, i2]-=rate[i, 2]
        C[i2, i1]-=rate[i, 2]
    s = -np.sum(C, axis=0)
    for i in xrange(C.shape[0]):
        C[i,i] = s[i]+2

@cython.boundscheck(False)
@cython.wraparound(False)
def fast_colley_vector_build(np.ndarray[np.int32_t, ndim=2] pair,
                             np.ndarray[np.float32_t, ndim=2] rate,
                             np.ndarray[np.float32_t, ndim=1] b):
    cdef unsigned int i1, i2, i;
    for i in xrange(pair.shape[0]):
        i1 = <unsigned int>(pair[i, 0])
        i2 = <unsigned int>(pair[i, 1])
        if rate[i, 0]>rate[i, 1]:
            b[i1]+=rate[i, 2]
            b[i2]-=rate[i, 2]
        elif rate[i, 0]<rate[i, 1]:
            b[i1]-=rate[i, 2]
            b[i2]+=rate[i, 2]
        # tie wouldn't affect b
    b/=2
    b+=1

@cython.boundscheck(False)
@cython.wraparound(False)
def fast_rate_diff_vote_matrix_build(np.ndarray[np.int32_t, ndim=2] pair,
                             np.ndarray[np.float32_t, ndim=2] rate,
                             np.ndarray[np.float32_t, ndim=2] D):
    cdef unsigned int i1, i2, i;
    for i in xrange(pair.shape[0]):
        i1 = <unsigned int>(pair[i, 0])
        i2 = <unsigned int>(pair[i, 1])
        if rate[i, 0]>rate[i, 1]:
            D[i2, i1] += (rate[i, 0]-rate[i, 1])*rate[i, 2]
        else:
            D[i1, i2] += (rate[i, 1]-rate[i, 0])*rate[i, 2]

@cython.boundscheck(False)
@cython.wraparound(False)
def fast_simple_diff_vote_matrix_build(np.ndarray[np.int32_t, ndim=2] pair,
                             np.ndarray[np.float32_t, ndim=2] rate,
                             np.ndarray[np.float32_t, ndim=2] D):
    cdef unsigned int i1, i2, i;
    for i in xrange(pair.shape[0]):
        i1 = <unsigned int>(pair[i, 0])
        i2 = <unsigned int>(pair[i, 1])
        if rate[i, 0]>rate[i, 1]:
            D[i2, i1] += rate[i, 2]
        else:
            D[i1, i2] += rate[i, 2]

@cython.boundscheck(False)
@cython.wraparound(False)
def fast_rate_vote_matrix_build(np.ndarray[np.int32_t, ndim=2] pair,
                             np.ndarray[np.float32_t, ndim=2] rate,
                             np.ndarray[np.float32_t, ndim=2] D):
    cdef unsigned int i1, i2, i;
    for i in xrange(pair.shape[0]):
        i1 = <unsigned int>(pair[i, 0])
        i2 = <unsigned int>(pair[i, 1])
        D[i2, i1] += rate[i, 0]*rate[i, 2]
        D[i1, i2] += rate[i, 1]*rate[i, 2]

@cython.boundscheck(False)
@cython.wraparound(False)
def fast_consistancy_matrix_build(np.ndarray[np.float32_t, ndim=2] D,
                                  np.ndarray[np.float32_t, ndim=2] E,
                                  np.ndarray[np.int32_t, ndim=2] C):
    cdef unsigned int i, j, k;
    cdef int cnt, inv;
    for i in xrange(D.shape[0]):
        for j in xrange(i, D.shape[0]):
            if i==j: continue;
            cnt=0;inv=0;
            for k in xrange(D.shape[0]):
                if D[i, k]<D[j, k]:cnt+=1;
                elif D[i, k]>D[j, k]:inv+=1;
                if E[i, k]>E[j, k]:cnt+=1;
                elif E[i, k]<E[j, k]:inv+=1;
            C[i, j]=cnt;
            C[j, i]=inv;

@cython.boundscheck(False)
@cython.wraparound(False)
def fast_contest_count_matrix_build(np.ndarray[np.int32_t, ndim=2] pair,
                                    np.ndarray[np.float32_t, ndim=2] C):
    cdef unsigned int i, ii, ij;
    for i in xrange(pair.shape[0]):
        ii = <unsigned int>(pair[i, 0])
        ij = <unsigned int>(pair[i, 1])
        C[ii, ij]+=1
