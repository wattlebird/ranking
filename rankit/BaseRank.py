import pandas as pd

class BaseRank(object):
    """docstring for BaseRank"""
    def __init__(self, arg):
        self.arg = arg

    @classmethod
    def rank(cls, namelist, rate):
        assert(len(namelist)==len(rate))
        table = pd.DataFrame({
            'title':pd.series(namelist),
            'rate':pd.series(rate)
        })
        ranked = table.sort_values(by='rate')
        ranked['rank'] = pd.series(range(1,len(rate)+1),dtype = 'int32')
        return ranked
