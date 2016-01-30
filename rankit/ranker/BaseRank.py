import pandas as pd
import numpy as np


class BaseRank(object):
    """docstring for BaseRank"""

    def __init__(self, itemlist, ascending=False):
        """itemlist is a dataframe that contains two columns:
        `itemid` and `index`.
        """
        if itemlist is None:
            raise KeyError("There should be a solid item list for a rank task.")
        # The proper behaviour is to check if index is ascending.
        self.itemlist = itemlist
        self.ascending = ascending

    def rank(self, rate):
        itemlist = self.itemlist
        ascending = self.ascending
        assert (itemlist.shape[0] == len(rate))
        table = pd.DataFrame({
            'title': itemlist['itemid'],
            'rate': pd.Series(rate)
        }, columns = [['title', 'rate']])
        ranked = pd.DataFrame(table.sort_values(by='rate', ascending=ascending).\
                 values, columns = table.columns)
        # buggy
        #ranked['rank'] = pd.Series(range(1, len(rate) + 1), dtype='int32')
        ranked['rank'] = pd.Series(self._get_true_rank(ranked), dtype='int32')
        return ranked

    def _get_true_rank(self, rankedtable):
        # rankedtable: ['title', 'rate']
        r = np.zeros(rankedtable.shape[0], dtype = np.int32)
        r[0]=1;
        for i in xrange(1, rankedtable.shape[0]):
            if rankedtable.iloc[i, 1]!=rankedtable.iloc[i-1, 1]:
                r[i]=i+1
            else:
                r[i]=r[i-1]
        return r
