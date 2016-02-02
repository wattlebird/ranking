import numpy as np
from BaseRank import BaseRank
from numpy.linalg import norm


class MarkovRank(BaseRank):
    """docstring for MarkovRank
    Please be noted that there are three options to force the forming of a
    irreducible matrix.
    equalvote will make a node with no out-link vote equally to all the nodes in
    a graph.
    epsilon: common practice in PageRank, S = epsilon*S+(1-epsilon)*E/n
    bounce: a node without out-link will bounce to every node that links to it.
    """
    def __init__(self, threshold=1e-4, equalvote=None, epsilon=None,
                 bounce=None, *args, **kwargs):
        super(MarkovRank, self).__init__(*args, **kwargs)
        self.equalvote = equalvote
        self.epsilon = epsilon
        self.bounce = bounce
        self.threshold = threshold

    def rate(self, S):
        l = S.shape[0]
        s = S.sum(axis=1)
        idx = np.where(s==0)[0]
        S = S/np.expand_dims(s, axis=1)
        S[idx,:]=0
        equalvote = self.equalvote
        epsilon = self.epsilon
        bounce = self.bounce
        if equalvote is not None:
            S[idx,:]=1./l
        elif epsilon is not None:
            S = epsilon*S+(1-epsilon)*np.ones((l, l), dtype = S.dtype)/float(l)
        elif bounce is not None:
            mask = np.require((S[:, idx]!=0).T, dtype=np.float)
            for i in xrange(idx.shape[0]):
                S[idx[i],:] = mask[i]
        else:
            if idx.shape[0]!=0:
                raise RuntimeError("Did not specify a method to make matrix irreducible.")

        threshold = self.threshold
        assert(np.any(np.isnan(S))==False)
        S = np.require(S.T, dtype=np.float32)
        x=np.ones(l,dtype=np.float32)
        prev = x.copy()
        while True:
            x/=x.sum()
            if norm(prev-x)<threshold:
                break;
            prev=x.copy()
            x=np.dot(S, x)
        return x

class ODRank(BaseRank):
    """Three options are given for output rank.
    'offence'
    'defence'
    'summary'
    """
    def __init__(self, threshold = 1e-4, output='summary',
                 epsilon=None, iteration=10000, *args, **kwargs):
        super(ODRank, self).__init__(*args, **kwargs)
        if output!='summary' and output!='defence' and output!='offence':
            raise ValueError("Wrong output rating.")
        self.output = output
        self.epsilon = epsilon
        self.threshold = threshold
        self.iteration = iteration

    def rate(self, A):
        output = self.output
        epsilon = self.epsilon
        threshold = self.threshold
        iteration = self.iteration
        if epsilon is not None:
            E = np.ones(A.shape, A.dtype)*epsilon
            E-=np.diag(np.diag(E))
            A+=E
        d = np.ones(A.shape[0], dtype=np.float32)
        d_prev = d.copy()
        cnt=0
        while True:
            d = np.dot(A, 1./np.dot(A.T, 1./d))
            if norm(d_prev-d)<threshold:
                break;
            else:
                cnt+=1;
                if cnt>iteration: break;
            d_prev = d.copy()

        o = np.dot(A.T, 1./d)
        if output=='summary':
            return o/d
        elif output=='offence':
            return o
        elif output=='defence':
            return d

class KeenerRank(BaseRank):
    """docstring for KeenerRank"""
    def __init__(self, regularization=True, func=None, epsilon=None, threshold=1e-4,
                 *args, **kwargs):
        super(KeenerRank, self).__init__(*args, **kwargs)
        self.regularization = regularization
        self.func = func
        self.epsilon = epsilon
        self.threshold = threshold

    def rate(self, A, C):
        regularization = self.regularization
        func = self.func
        epsilon = self.epsilon
        threshold = self.threshold
        B = self._KeenerMatrix(A, C, regularization, func, epsilon)
        prev_r = np.ones(B.shape[0])
        r = prev_r.copy()
        while True:
            r = r/np.sum(r)
            if norm(r-prev_r)<threshold:
                break;
            else:
                prev_r = r.copy()
            r = np.dot(B, r)
        return r

    def _KeenerMatrix(self, A, C, regularization, func, epsilon):
        """func is a regularization function imposed on every element of matrix.
        """
        # Apply Laplace Law
        B = A+A.T+2;
        A = A+1
        A = A/B
        # Regularization
        if func is not None:
            h = np.frompyfunc(func, 1, 1)
            A = np.require(h(A), dtype=np.float32)
        # divide by contest number
        C = C+C.T
        c = np.sum(C, axis=1)
        if regularization:
            A = A/np.expand_dims(c, axis=1)
        A[C==0]=0
        if epsilon is not None:
            A += epsilon*np.ones(A.shape, A.dtype)
        return A
