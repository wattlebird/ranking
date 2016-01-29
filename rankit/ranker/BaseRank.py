import pandas as pd


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
        })
        ranked = pd.DataFrame(table.sort_values(by='rate', ascending=ascending).\
                 values, columns = table.columns)
        ranked['rank'] = pd.Series(range(1, len(rate) + 1), dtype='int32')
        return ranked
