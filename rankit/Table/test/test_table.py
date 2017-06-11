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

def record_read_test():
    data = Table("Data\\record.test.tsv", datatype="record", col="1,2,6", hasheader=False)

def paired_read_test():
    data = Table("Data\\TourneyCompactResults.csv", col="3,5,4,6", delimiter=',', hasheader=True)

def paired_read_df_test():
    data = Table(sample_paired, datatype="paired", col="1,2,3,4")

def record_to_paired_test():
    data = Table("Data\\record.test.tsv", datatype="record", col="1,2,6", hasheader=False)
    data._gettable(datatype="paired")
