from numpy.testing import assert_array_equal, assert_array_almost_equal
from numpy.testing import assert_almost_equal
from nose.tools import assert_raises, assert_true, assert_equal, assert_false
from rankit.util import ConsistancyMatrix
import numpy as np


# toy test case
D = np.array([
        [0,1,0],
        [0,0,1],
        [2,0,0]
    ])
C = np.array([
        [0,2,2],
        [2,0,2],
        [2,2,0]
    ], np.int32)

def consistancy_matrix_builder_test():
    assert_array_equal(ConsistancyMatrix(D), C)
