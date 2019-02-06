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

def table_init_test():
    t = Table()
    assert_true(hasattr(t, 'table'))
    assert_true(hasattr(t, 'indexlut'))
    assert_true(hasattr(t, 'itemnum'))
    assert_true(hasattr(t, 'itemlut'))

def table_load_test():
    data = Table(sample_paired, col=[0, 1, 2, 3])
    data = Table(sample_paired, ['primary', 'secondary', 'rate1', 'rate2'])
    assert_equal(5, data.itemnum)

def table_iteritem_test():
    data = Table(sample_paired, ['primary', 'secondary', 'rate1', 'rate2'])
    for rec in data.iteritem():
        assert_true(hasattr(rec, 'host'))
        assert_true(hasattr(rec, 'visit'))
        assert_true(hasattr(rec, 'hscore'))
        assert_true(hasattr(rec, 'vscore'))
        assert_true(hasattr(rec, 'indexHost'))
        assert_true(hasattr(rec, 'indexVisit'))

def table_update_test():
    df = pd.DataFrame([['sdf', 'wef', 2, 2]], columns=['host', 'visit', 'hscore', 'vscore'])
    t1 = Table()
    t2 = Table(df, ['host', 'visit', 'hscore', 'vscore'])
    t1.update(t2)
    assert_true(t1.itemnum == 2)

def table_update_raw_test():
    df = pd.DataFrame([['sdf', 'wef', 2, 2]], columns=['host', 'visit', 'hscore', 'vscore'])
    t1 = Table()
    t1.update_raw(df)
    assert_true(t1.itemnum == 2)

def table_setup_test():
    itemlut = {
        'sdf': 0,
        'wef': 1,
        'xdf': 2
    }
    indexlut = ['sdf', 'wef', 'xdf']
    t1 = Table()
    t1.setup(itemlut, indexlut, 3)

    df = pd.DataFrame([['sdf', 'wef', 2, 2]], columns=['host', 'visit', 'hscore', 'vscore'])
    t2 = Table(df, ['host', 'visit', 'hscore', 'vscore'])
    t1.update(t2)
    assert_true(t1.itemnum == 3)