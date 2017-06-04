import numpy as np
import pandas as pd
from rankit.Table import *
from nose.tools import assert_raises, assert_true, assert_equal, assert_false


def record_read_test():
    data = Table("Data\\record.test.tsv", datatype="record", col="1,2,6", hasheader=False)

def paired_read_test():
    data = Table("Data\\TourneyCompactResults.csv", col="3,5,4,6", delimiter=',', hasheader=True)

def record_to_paired_test():
    data = Table("Data\\record.test.tsv", datatype="record", col="1,2,6", hasheader=False)
    data._gettable(datatype="paired")
