from __future__ import division
import numpy as np
cimport numpy as np
cimport cython


@cython.boundscheck(False)
@cython.wraparound(False)
def fast_record_to_pairwise(np.ndarray[np.int32_t, ndim=1] item,
                             np.ndarray[np.float64_t, ndim=1] score):
    cdef unsigned int i1, i2, i, cnt;
    cdef double s1, s2;

    cdef np.ndarray[np.int32_t, ndim=2] pairidx = np.zeros((item.shape[0]*(item.shape[0]-1)//2, 2), dtype=np.int32)
    cdef np.ndarray[np.float64_t, ndim=2] pairscore = np.zeros((score.shape[0]*(score.shape[0]-1)//2, 2), dtype=np.float64)
    cnt = 0;
    for i in range(item.shape[0]):
        for j in range(i+1, item.shape[0]):
            i1 = <unsigned int>(item[i])
            i2 = <unsigned int>(item[j])
            s1 = <double>(score[i])
            s2 = <double>(score[j])
            pairidx[cnt, 0] = i1
            pairidx[cnt, 1] = i2
            pairscore[cnt, 0] = s1
            pairscore[cnt, 1] = s2
            cnt += 1
    return (pairidx, pairscore)

