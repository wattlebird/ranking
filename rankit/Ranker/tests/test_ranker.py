from numpy.testing import assert_array_equal, assert_array_almost_equal
from numpy.testing import assert_almost_equal
from nose.tools import assert_raises, assert_true, assert_equal, assert_false
from rankit.Ranker import ColleyRanker, MasseyRanker, KeenerRanker, MarkovRanker, ODRanker, DifferenceRanker
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

def colley_rank_test():
    data = Table(sample_paired, col=[0,1,2,3])
    r = ColleyRanker(table = data)
    rst = r.rank(ascending=False)
    assert_array_almost_equal(rst.loc[:, 'rating'].values, 
                              np.array([0.79, 0.65, 0.50, 0.36, 0.21]),decimal=2)

def massey_rank_test():
    data = Table(sample_paired, col=[0,1,2,3])
    r = MasseyRanker(table = data)
    rst = r.rank(ascending=False)
    assert_array_almost_equal(rst.loc[:, 'rating'].values, 
                              np.array([18.2, 18.0, -3.4, -8.0, -24.8]),decimal=2)

def keener_rank_test():
    data = Table(sample_paired, col=[0,1,2,3])
    r = KeenerRanker(table = data)
    rst = r.rank(ascending=False)
    print('Keener rank:')
    print(rst)

def markov_rank_test():
    data = Table(sample_paired, col=[0,1,2,3])
    r = MarkovRanker(table = data)
    rst = r.rank(ascending=False)
    print('Markov rank:')
    print(rst)

def od_rank_test():
    data = Table(sample_paired, col=[0,1,2,3])
    r = ODRanker(table = data)
    rst = r.rank(output='summary', ascending=False)
    print('OD rank: overall rank:')
    print(rst)
    rst = r.rank(output='offence', ascending=False)
    print('OD rank: offence rank:')
    print(rst)
    rst = r.rank(output='defence', ascending=True)
    print('OD rank: defence rank:')
    print(rst)

def difference_rank_test():
    data = Table(sample_paired, col=[0,1,2,3])
    r = DifferenceRanker(table = data)
    rst = r.rank(ascending=False)
    print('Difference rank:')
    print(rst)
