from numpy.linalg import solve, matrix_rank
import numpy as np
from numpy.linalg import LinAlgError

class DefaultSolver(object):
    """docstring for DefaultSolver"""
    def __init__(self):
        pass

    def solve(self, A, b):
        return solve(A, b)

class InsufficientRankSolver(DefaultSolver):
    """This Solver provides several ways to handle the solving of a sigular
    linear equation.
    There are two options to solve a singular linear equation:
    epsilon:    by adding matrix A with an all-one matrix (will neglect diagonal
                values whenever A has all-zeros in main diagonal.)
    force:      by forcing the last row of A to all one and last value of b to 0.
    """
    def __init__(self, epsilon=None, force=None):
        self.epsilon = epsilon
        self.force = force

    def solve(self, A, b):
        epsilon = self.epsilon
        force = self.force
        while True:
            try:
                rtn = super(InsufficientRankSolver, self).solve(A, b)
                break;
            except LinAlgError as e:
                if epsilon is None and force is None:
                    raise RuntimeError("Did not provide a way to resolve sigular matrix")

            if epsilon is not None:
                diagval = np.sum(np.diagonal(A))
                E = np.ones(A.shape, A.dtype)*epsilon
                if diagval==0:
                    E-=np.diag(np.diag(E))
                A+=E
                epsilon = None
            else:
                A[-1,:] = np.ones(A.shape[0], A.dtype)
                b[-1] = 0
                force = None

        return rtn;
