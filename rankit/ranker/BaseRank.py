import pandas as pd


class BaseRank(object):
    """docstring for BaseRank"""

    def __init__(self, itemlist):
        """itemlist is a dataframe that contains two columns:
        `itemid` and `index`.
        """
        if itemlist is None:
            raise KeyError("There should be a solid item list for a rank task.")
        self.itemlist = itemlist

    def rank(self, rate):
        itemlist = self.itemlist
        assert (itemlist.shape[0] == len(rate))
        table = pd.DataFrame({
            'title': itemlist['itemid'],
            'rate': pd.Series(rate)
        })
        ranked = pd.DataFrame(table.sort_values(by='rate', ascending=False).\
                 values, columns = table.columns)
        ranked['rank'] = pd.Series(range(1, len(rate) + 1), dtype='int32')
        return ranked
