from numpy.testing import assert_array_equal, assert_array_almost_equal
from numpy.testing import assert_almost_equal
from nose.tools import assert_raises, assert_true, assert_equal, assert_false
from rankit.solver import DefaultSolver, InsufficientRankSolver
import numpy as np


# toy test case
A = np.array([
    [1, 1],
    [1, 1]
], dtype=np.float32)
b = np.array([1, 1], dtype=np.float32)

def singular_matrix_test():
    solver = InsufficientRankSolver()
    assert_raises(RuntimeError, solver.solve, A, b)

def normal_matrix_test():
    solver = InsufficientRankSolver()
    x = solver.solve(np.eye(2), b)
    assert_array_equal(x, np.ones(2))


def singular_matrix_failure1_test():
    solver = InsufficientRankSolver(epsilon=1e-4)
    assert_raises(RuntimeError, solver.solve, A, b)

def singular_matrix_failure2_test():
    solver = InsufficientRankSolver(force=True)
    assert_raises(RuntimeError, solver.solve, A, b)
