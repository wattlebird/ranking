from numpy.testing import assert_array_equal, assert_array_almost_equal
from numpy.testing import assert_almost_equal, assert_raises
from nose.tools import assert_raises, assert_true, assert_equal, assert_false
from rankit.Ranker import *
from rankit.Table import Table
import numpy as np
import pandas as pd


# toy testcase
sample_paired = pd.DataFrame({
    "primary": ["Duke", "Duke", "Duke", "Duke", "Miami", "Miami", "Miami", "UNC", "UNC", "UVA"], 
    "secondary": ["Miami", "UNC", "UVA", "VT", "UNC", "UVA", "VT", "UVA", "VT", "VT"],
    "rate1": [7, 21, 7, 0, 34, 25, 27, 7, 3, 14],
    "rate2": [52, 24, 38, 45, 16, 17, 7, 5, 30, 52]
}, columns=["primary", "secondary", "rate1", "rate2"])

sample_with_time_1 = pd.DataFrame({
    'primary': [1,1,4,2,3,4,1,2],
    'secondary': [2,3,1,4,2,3,4,3],
    'rate1': [7,6,8,4,3,5,7,0],
    'rate2': [5,5,4,4,4,1,7,1],
    'date': [1,1,2,2,3,3,4,4]
}, columns=['primary', 'secondary', 'rate1', 'rate2', 'date'])

sample_with_time_2 = pd.DataFrame({
    'primary': [2,4,2,4],
    'secondary': [1,1,3,3],
    'rate1': [6,6,3,2],
    'rate2': [5,4,3,1],
    'date': [5,6,7,8]
}, columns=['primary', 'secondary', 'rate1', 'rate2', 'date'])

def colley_rank_test():
    data = Table(sample_paired, col=[0,1,2,3])
    r = ColleyRanker()
    rst = r.rank(data)
    assert_array_almost_equal(rst.loc[:, 'rating'].values, 
                              np.array([0.79, 0.65, 0.50, 0.36, 0.21]),decimal=2)

def massey_rank_test():
    data = Table(sample_paired, col=[0,1,2,3])
    r = MasseyRanker()
    rst = r.rank(data)
    assert_array_almost_equal(rst.loc[:, 'rating'].values, 
                              np.array([18.2, 18.0, -3.4, -8.0, -24.8]),decimal=2)

def keener_rank_test():
    data = Table(sample_paired, col=[0,1,2,3])
    r = KeenerRanker()
    rst = r.rank(data)

def markov_rank_test():
    data = Table(sample_paired, col=[0,1,2,3])
    r = MarkovRanker()
    rst = r.rank(data)

def od_rank_test():
    data = Table(sample_paired, col=[0,1,2,3])
    r1 = ODRanker(method='summary')
    rst = r1.rank(data)
    r2 = ODRanker(method='defence')
    rst = r2.rank(data)

def difference_rank_test():
    data = Table(sample_paired, col=[0,1,2,3])
    r = DifferenceRanker()
    rst = r.rank(data)
