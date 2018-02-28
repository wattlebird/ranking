import numpy as np
import pandas as pd
from rankit.Table import *
from nose.tools import assert_raises, assert_true, assert_equal, assert_false


sample_paired = pd.DataFrame({
        "primary": ["Duke", "Duke", "Duke", "Duke", "Miami", "Miami","Miami", "UNC", "UNC", "UVA"], 
        "secondary": ["Miami", "UNC", "UVA", "VT", "UNC", "UVA", "VT", "UVA", "VT", "VT"],
        "rate1": [7, 21, 7, 0, 34, 25, 27, 7, 3, 14],
        "rate2": [52, 24, 38, 45, 16, 17, 7, 5, 30, 52]
    }, columns=["primary", "secondary", "rate1", "rate2"])

def df_read_test():
    data = Table(sample_paired, col=[0, 1, 2, 3])
    data = Table(sample_paired, ['primary', 'secondary', 'rate1', 'rate2'])
