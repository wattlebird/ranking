from rankit.Ranker import EloRanker, TrueSkillRanker, GlickoRanker
from rankit.Table import Table
import pandas as pd
from numpy.testing import assert_array_equal, assert_array_almost_equal
from nose.tools import assert_raises, assert_true, assert_equal, assert_false

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

def elo_leaderboard_test():
    table = Table(sample_with_time_1, ['primary', 'secondary', 'rate1', 'rate2'], timecol='date')
    eloRanker = EloRanker()
    eloRanker.update(table)
    lb = eloRanker.leaderboard()
    assert_equal(lb.rating.values.sum(), 6000)
    return lb

def elo_update_test():
    table = Table(sample_with_time_1.iloc[:-1, :], ['primary', 'secondary', 'rate1', 'rate2'], timecol='date')
    eloRanker = EloRanker()
    eloRanker.update(table)
    eloRanker.update_single(2, 3, 0, 1)
    lb = eloRanker.leaderboard()
    assert_array_almost_equal(lb.rating.values, elo_leaderboard_test().rating.values)

def elo_prob_win_test():
    table = Table(sample_with_time_1, ['primary', 'secondary', 'rate1', 'rate2'], timecol='date')
    eloRanker = EloRanker()
    eloRanker.update(table)
    assert_true(eloRanker.prob_win(4,1) > 0.5)

def trueskill_leaderboard_test():
    table = Table(sample_with_time_1, ['primary', 'secondary', 'rate1', 'rate2'], timecol='date')
    tsRanker = TrueSkillRanker()
    tsRanker.update(table)
    lb = tsRanker.leaderboard()
    return lb

def trueskill_update_test():
    table = Table(sample_with_time_1.iloc[:-1, :], ['primary', 'secondary', 'rate1', 'rate2'], timecol='date')
    tsRanker = TrueSkillRanker()
    tsRanker.update(table)
    tsRanker.update_single(2, 3, 0, 1)
    lb = tsRanker.leaderboard()
    assert_array_almost_equal(lb.rating.values, trueskill_leaderboard_test().rating.values)

def trueskill_prob_win_test():
    table = Table(sample_with_time_1, ['primary', 'secondary', 'rate1', 'rate2'], timecol='date')
    tsRanker = TrueSkillRanker()
    tsRanker.update(table)
    assert_true(tsRanker.prob_win(4,1) > 0.5)

def glicko_leaderboard_test():
    table = Table(sample_with_time_1, ['primary', 'secondary', 'rate1', 'rate2'], timecol='date')
    gRanker = GlickoRanker()
    gRanker.update(table)
    lb = gRanker.leaderboard()
    return lb

def glicko_update_test():
    table1 = Table(sample_with_time_1, ['primary', 'secondary', 'rate1', 'rate2'], timecol='date')
    table2 = Table(sample_with_time_2, ['primary', 'secondary', 'rate1', 'rate2'], timecol='date')
    gRanker = GlickoRanker()
    gRanker.update(table1)
    gRanker.update(table2)

def glicko_prob_win_test():
    table = Table(sample_with_time_1, ['primary', 'secondary', 'rate1', 'rate2'], timecol='date')
    gRanker = GlickoRanker()
    gRanker.update(table)
    assert_true(gRanker.prob_win(4, 1) > 0.5)