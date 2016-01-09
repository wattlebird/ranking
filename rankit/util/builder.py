import numpy as np

def ConsistancyMatrix(D):
    C = np.zeros(D.shape, dtype=np.int32)
    E = D.T;
    for i in xrange(D.shape[0]-1):
        for j in xrange(1, D.shape[1]):
            cnt=0;inv=0;
            for k in xrange(D.shape[0]):
                if D[i, k]<D[j, k]:cnt+=1;
                elif D[i, k]>D[j, k]:inv+=1;
                if E[i, k]>E[j, k]:cnt+=1;
                elif E[i, k]<E[j, k]:inv+=1;
            C[i, j]=cnt;
            C[j, i]=inv;
    return C
