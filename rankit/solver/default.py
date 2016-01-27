from numpy.linalg import solve, matrix_rank
import numpy as np

class DefaultSolver(object):
    """docstring for DefaultSolver"""
    def __init__(self):
        pass

    def solve(self, A, b):
        return solve(A, b)

class InsufficientRankSolver(DefaultSolver):
    """This Solver provides several ways to handle the solving of a sigular
    linear equation.
    Notice: rankcheck uses SVD, which can be very time consuming. Take care.
    """
    def __init__(self, rankcheck=False, epsilon=None, force=None):
        self.epsilon = epsilon
        self.force = force
        self.rankcheck = rankcheck

    def solve(self, A, b):
        epsilon = self.epsilon
        force = self.force
        rankcheck = self.rankcheck
        while True:
            if rankcheck:
                rank = matrix_rank(A)
                if rank==A.shape[0]:
                    rtn = super(InsufficientRankSolver, self).solve(A, b)
                    break;
            else:
                rtn = super(InsufficientRankSolver, self).solve(A, b)
                break;
            if epsilon is not None:
                diagval = np.sum(np.diagonal(A))
                E = np.ones(A.shape, A.dtype)*epsilon
                if diagval==0:
                    E-=np.diag(np.diag(E))
                A+=E
            elif force is not None:
                A[-1,:]=np.ones(A.shape[0], A.dtype)
                b[-1]=0
            else:
                raise RuntimeError("Did not provide a way to resolve sigular matrix")
        return rtn;
