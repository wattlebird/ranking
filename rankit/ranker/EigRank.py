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
    def __init__(self, itemlist, threshold=1e-4, equalvote=None, epsilon=None, bounce=None):
        super(MarkovRank, self).__init__(itemlist)
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
    def __init__(self, itemlist, threshold = 1e-4, output='summary', epsilon=None):
        super(ODRank, self).__init__(itemlist)
        if output!='summary' and output!='defence' and output!='offence':
            raise ValueError("Wrong output rating.")
        self.output = output
        self.epsilon = epsilon
        self.threshold = threshold

    def rate(self, A):
        # we start from offence rate because each element in matrix A is the
        # score that how much j is superior than i
        output = self.output
        epsilon = self.epsilon
        threshold = self.threshold
        o = np.ones(A.shape[1], dtype=np.float32)
        o_prev = o.copy()
        while True:
            o = np.dot(A.T, 1./np.dot(A, 1./o))
            if norm(o_prev-o)<threshold:
                break;
            o_prev = o.copy()
        d = np.dot(A, 1./o)
        if output=='summary':
            return o/d
        elif output=='offence':
            return o
        elif output=='defence':
            return d
