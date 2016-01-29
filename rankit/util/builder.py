import numpy as np

def ConsistancyMatrix(D):
    """
    For input D, each element indicates how much superior i is compared to j.
    c_ij = #{ k | d_ik < d_jk } + # how many contests j is superior to i
           #{ k | d_ki > d_kj }   # how many other items think that j is more hard to compete than i
    C_ij is the potential that j is superior to i.
    """
    C = np.zeros(D.shape, dtype=np.int32)
    E = D.T;
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
    return C
