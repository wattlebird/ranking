#!/home/ike/.anaconda/bin/python2.7
import numpy as np
from solver import DefaultSolver
from BaseRank import BaseRank
import pandas as pd


class ColleyRank(BaseRank):
    """docstring for ColleyRank"""
    def __init__(self, arg):
        super(ColleyRank, self).__init__()
        self.arg = arg

    def rate(self, C, b):
        return DefaultSolver.solve(C,b)

    @classmethod
    def convert(cls, filename, filetype="user_item_rate"):
        table = pd.read_hdf(filename, filetype)
        if filetype == 'user_item_rate':
            iid = table.loc[:, 'itemid'].unique()
            item_id = pd.DataFrame({'itemid': iid,
                                    'index': pd.Series(range(iid.shape[0]), dtype='int32')})
            watchlist = pd.merge(table, item_id, on='itemid')
            userlist = table.loc[:, 'username'].unique()
            icnt = iid.shape[0]
            # allocate space for computing
            C = np.zeros((icnt, icnt), dtype=np.float32)
            w = np.zeros((icnt, 2), dtype=np.float32)

            for user in userlist:
                userwatchlist = watchlist[(watchlist.username==user)]
                for i in xrange(userwatchlist.shape[0]):
                    for j in xrange(i, userwatchlist.shape[0]):
                        i1 = userwatchlist.iloc[i, 3]
                        i2 = userwatchlist.iloc[j, 3]
                        C[i1, i2] -= 1
                        C[i2, i1] -= 1
                        if userwatchlist.iloc[i, 2] > userwatchlist.iloc[j, 2]:
                            w[i1, 0] += 1
                            w[i2, 1] += 1
                        elif userwatchlist.iloc[i, 2] < userwatchlist.iloc[j, 2]:
                            w[i1, 1] += 1
                            w[i2, 0] += 1
                        else:
                            w[i1, :] += 0.5
                            w[i2, :] += 0.5

            b = np.ravel(1+0.5*(w[:, 0]-w[:, 1]))
            for i in xrange(icnt):
                C[i, i] = 2+w[i, 0]+w[i, 1]
            return (C, b)

