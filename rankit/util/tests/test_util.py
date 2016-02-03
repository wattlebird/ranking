from numpy.testing import assert_array_equal, assert_array_almost_equal
from numpy.testing import assert_almost_equal
from nose.tools import assert_raises, assert_true, assert_equal, assert_false
from rankit.util import ConsistancyMatrix, Converter
import numpy as np
import pandas as pd


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

testtable = pd.DataFrame({'primary':['A','B','C','A'],
                          'secondary':['B','C','B','C'],
                          'rate1':[1,2,1,2],
                          'rate2':[2,3,3,1],
                          'weight':[2,1,1,1]})

def consistancy_matrix_builder_test():
    assert_array_equal(ConsistancyMatrix(D), C)

def converter_item_list_test():
    cvt = Converter(table=testtable)
    assert_equal(cvt.ItemList().shape[0],3)
    assert_equal(cvt.itemlist['A'], 0)
    assert_equal(cvt.itemlist['B'], 1)
    assert_equal(cvt.itemlist['C'], 2)

def colley_test():
    cvt = Converter(table=testtable)
    assert_array_equal(cvt.ColleyMatrix(),
    np.array([[5,-2,-1],[-2,6,-2],[-1,-2,5]],dtype=np.float32))
    assert_array_equal(cvt.ColleyVector(),
    np.array([0.5, 2, 0.5], dtype=np.float32))

def massey_test():
    cvt = Converter(table=testtable)
    assert_array_equal(cvt.MasseyMatrix(),
    np.array([[3,-2,-1],[-2,4,-2],[-1,-2,3]],dtype=np.float32))
    assert_array_equal(cvt.MasseyVector(),
    np.array([-1, 3, -2], dtype=np.float32))

def count_test():
    cvt = Converter(table=testtable)
    assert_array_equal(cvt.CountMatrix(),
    np.array([[0,1,1],[0,0,1],[0,1,0]],dtype=np.float32))

def symmetric_difference_test():
    cvt = Converter(table=testtable)
    assert_array_equal(cvt.SymmetricDifferenceMatrix(),
    np.array([[0,-2,1],[2,0,0.5],[-1,-0.5,0]], dtype=np.float32))

def vote_matrix_test():
    cvt = Converter(table=testtable)
    assert_array_equal(cvt.SimpleDifferenceVoteMatrix(),
    np.array([[0,2,0],[0,0,1],[1,1,0]], dtype=np.float32))
    assert_array_equal(cvt.RateDifferenceVoteMatrix(),
    np.array([[0,2,0],[0,0,1],[1,2,0]], dtype=np.float32))
    assert_array_equal(cvt.RateVoteMatrix(),
    np.array([[0,4,1],[2,0,4],[2,5,0]], dtype=np.float32))
