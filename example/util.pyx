from __future__ import division
import numpy as np
cimport numpy as np
ctypedef np.int32_t INDEXTYPE_t
ctypedef np.float32_t RATETYPE_t
from libc.math cimport log
cimport cython

@cython.boundscheck(False)
@cython.wraparound(False)
def fast_convert(np.ndarray[INDEXTYPE_t, ndim=1] idxlst,
                 np.ndarray[RATETYPE_t, ndim=1] ratelst,
                 np.ndarray[RATETYPE_t, ndim=2] matx,
                 np.ndarray[RATETYPE_t, ndim=2] count,
                 int operation):
    cdef unsigned int i, j;
    cdef unsigned int ii, ij;
    cdef unsigned int len = idxlst.shape[0];
    for i in xrange(len):
        for j in xrange(i+1, len):
            ii = <unsigned int>(idxlst[i]);
            ij = <unsigned int>(idxlst[j]);
            count[ii, ij]+=1
            if operation==1:
                matx[ii, ij]+=ratelst[i]
                matx[ij, ii]+=ratelst[j]
            elif operation==2:
                matx[ii, ij]+=log(ratelst[i])
                matx[ij, ii]+=log(ratelst[j])
            else:
                matx[ii, ij]+=1 if ratelst[i]>ratelst[j] else 0
                matx[ij, ii]+=1 if ratelst[i]<ratelst[j] else 0
