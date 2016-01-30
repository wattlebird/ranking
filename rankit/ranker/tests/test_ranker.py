from numpy.testing import assert_array_equal, assert_array_almost_equal
from numpy.testing import assert_almost_equal
from nose.tools import assert_raises, assert_true, assert_equal, assert_false
from rankit.ranker import BaseRank
import numpy as np
import pandas as pd


# toy testcase
sample_item_list = pd.DataFrame({'itemid':['A', 'B', 'C'],
                              'index':[2,1,0]},
                              columns=['itemid', 'index'])
sample_rate = np.array([1,1,2])

def same_rate_test():
    ranker = BaseRank(sample_item_list, ascending=True)
    ranktable = ranker.rank(sample_rate)
    ranktable = ranktable[['title', 'rate', 'rank']]
    assert_array_equal(ranktable.iloc[:, 2].values,
                       np.array([1,1,3], dtype = np.int32))
