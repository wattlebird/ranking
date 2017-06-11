from numpy.testing import assert_array_equal, assert_array_almost_equal
from numpy.testing import assert_almost_equal
from nose.tools import assert_raises, assert_true, assert_equal, assert_false
from rankit.Ranker import ColleyRanker, MasseyRanker
import numpy as np
import pandas as pd


# toy testcase
sample_paired = pd.DataFrame({
        "primary": ["Duke", "Duke", "Duke", "Duke", "Miami", "Miami", "UNC", "UNC", "UVA"], 
        "secondary": ["Miami", "UNC", "UVA", "VT", "UNC", "UVA", "VT", "UVA", "VT", "VT"],
        "rate1": [7, 21, 7, 0, 34, 25, 27, 7, 3, 14],
        "rate2": [52, 24, 38, 45, 16, 17, 7, 5, 30, 52]
    }, columns=["primary", "secondary", "rate1", "rate2", "weight"])

def colley_rank_test():
    data = Table(sample_paired, datatype="paired", col="1,2,3,4")
    r = ColleyRanker(table = data)
    rst = r.rank(ascending=False)
    assert_array_almost_equal(r.loc[:, 'rating'].values, 
                              np.array([0.79, 0.65, 0.50, 0.36, 0.21]),decimal=2)

def massey_rank_test():
    data = Table(sample_paired, datatype="paired", col="1,2,3,4")
    r = MasseyRanker(table = data)
    rst = r.rank(ascending=False)
    assert_array_almost_equal(r.loc[:, 'rating'].values, 
                              np.array([18.2, 18.0, -3.4, -8.0, -24.8]),decimal=2)

