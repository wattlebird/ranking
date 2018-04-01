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

def elo_rank_test():
    data = Table(sample_with_time_1, col=['primary', 'secondary', 'rate1', 'rate2'], timecol='date')
    ranker = EloRanker(data)
    ranker.rank(ascending=False)

def eld_rank_update_test():
    data1 = Table(sample_with_time_1, col=['primary', 'secondary', 'rate1', 'rate2'], timecol='date')
    data2 = Table(sample_with_time_2, col=['primary', 'secondary', 'rate1', 'rate2'], timecol='date')
    ranker = EloRanker(data1)
    r0 = ranker.rank(ascending=False)
    r1 = ranker.update(data2)

    data3 = Table(pd.concat([sample_with_time_1, sample_with_time_2]), col=['primary', 'secondary', 'rate1', 'rate2'], timecol='date')
    ranker = EloRanker(data3)
    r2 = ranker.rank(ascending=False)

    assert_almost_equal(r1.rating.values, r2.rating.values)
    assert_raises(AssertionError, assert_array_equal, r0.rating.values, r2.rating.values)

def massey_rank_score_difference_test():
    table = Table(sample_paired, col=[0,1,2,3])
    ranker = MasseyRanker(table)
    rank = ranker.rank()
    rank = rank.set_index('name')
    score_diff = ranker.score_diff(sample_paired.primary.values, sample_paired.secondary.values)
    t = sample_paired.merge(rank, left_on='primary', right_index=True).\
        merge(rank, left_on='secondary', right_index=True).\
        sort_index()
    score_diff_2 = t.rating_x - t.rating_y
    assert_almost_equal(score_diff, score_diff_2)

def keener_rank_score_difference_test():
    table = Table(sample_paired, col=[0,1,2,3])
    ranker = KeenerRanker(table)
    rank = ranker.rank()
    rank = rank.set_index('name')
    score_diff = ranker.score_diff(sample_paired.primary.values, sample_paired.secondary.values)
    t = sample_paired.merge(rank, left_on='primary', right_index=True).\
        merge(rank, left_on='secondary', right_index=True).\
        sort_index()
    score_diff_2 = t.rating_x - t.rating_y
    assert_almost_equal(score_diff, score_diff_2)

def difference_rank_score_difference_test():
    table = Table(sample_paired, col=[0,1,2,3])
    ranker = DifferenceRanker(table)
    rank = ranker.rank()
    rank = rank.set_index('name')
    score_diff = ranker.score_diff(sample_paired.primary.values, sample_paired.secondary.values)
    t = sample_paired.merge(rank, left_on='primary', right_index=True).\
        merge(rank, left_on='secondary', right_index=True).\
        sort_index()
    score_diff_2 = t.rating_x - t.rating_y
    assert_almost_equal(score_diff, score_diff_2)