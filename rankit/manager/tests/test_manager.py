from numpy.testing import assert_array_equal, assert_array_almost_equal
from numpy.testing import assert_almost_equal
from nose.tools import assert_raises, assert_true, assert_equal, assert_false
from rankit.manager import RankMerger, RankManager, RankComparer
import numpy as np
import pandas as pd


r1 = pd.DataFrame({'title': pd.Series(['A', 'B', 'C']),
                   'rate': pd.Series([1, 2, 4]),
                   'rank': pd.Series([1, 2, 3], dtype='int32')})
r2 = pd.DataFrame({'title': pd.Series(['B', 'C', 'A']),
                   'rate': pd.Series([5, 4, 3]),
                   'rank': pd.Series([1, 2, 3], dtype='int32')})


def test_rank_manager():
    mgr = RankManager()
    assert_true(mgr.cnt==0)
    mgr.update('method1', r1)
    mgr.update('method2', r2)
    assert_true(mgr.cnt==2)
    rst = mgr.get('method1')
    assert_array_equal(rst.sort_values(by='method1').loc[:, 'title'].values,
                       np.array(['A', 'B', 'C']))
    rst = mgr.get('method2')
    assert_array_equal(rst.sort_values(by='method2').loc[:, 'title'].values,
                       np.array(['B', 'C', 'A']))
    assert_true(mgr.cnt==2)
    mgr.delete('method1')
    assert_true(mgr.cnt==1)
    assert_raises(KeyError, mgr.get, 'method1')

def test_complete_rank_comparer():
    rankset = dict({'method1': r1, 'method2': r2})
    cp = RankComparer(rankset)
    assert_equal(cp.KendallMeasure('method1', 'method2'), -1.0/3.0)
    assert_equal(cp.SpearmanMeasure('method1', 'method2'), 3.5)

def test_borda_count_rank_merger():
    rankset = dict({'method1': r1, 'method2': r2})
    mgr = RankMerger(rankset)
    rst = mgr.BordaCountMerge()
    assert_array_equal(rst.loc[:, 'title'].values, np.array(['B', 'A', 'C']))

def test_avarage_rank_merger():
    rankset = dict({'method1': r1, 'method2': r2})
    mgr = RankMerger(rankset)
    rst = mgr.AverageRankMerge()
    assert_array_equal(rst.loc[:, 'title'].values, np.array(['B', 'A', 'C']))
