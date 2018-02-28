import numpy as np
import pandas as pd
import scipy as sp
from rankit.Table import Table
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import lsqr
from .matrix_build import fast_colley_build

class UnsupervisedRanker(object):
    """Base class for all unsupervised ranking algorithms."""
    def __init__(self, table, *args, **kwargs):
        self.data = table

    def rank(self, *args, **kwargs):
        raise NotImplementedError("UnsupervisedRanker is a abstract class.")

    def _showcase(self, ascending=True):
        # one need to translate item index to item name.
        indexlut = self.data.indexlut
        rating = self.rating # iitm, rating
        itemname = []
        for row in rating.itertuples(index=False, name=None):
            itemname.append(indexlut[row[0]])
        rst = pd.DataFrame({
            "name": itemname,
            "rating": rating["rating"]})
        rst = rst.sort_values(by='rating', ascending=ascending).reset_index(drop=True)
        rst['rank'] = pd.Series(self._get_true_rank(rst), dtype='int32')
        return rst

    def _get_true_rank(self, rankedtable):
        # rankedtable: ['title', 'rate']
        r = np.zeros(rankedtable.shape[0], dtype = np.int32)
        r[0]=1;
        for i in range(1, rankedtable.shape[0]):
            if rankedtable.iloc[i, 1]!=rankedtable.iloc[i-1, 1]:
                r[i]=i+1
            else:
                r[i]=r[i-1]
        return r

class MasseyRanker(UnsupervisedRanker):
    def __init__(self, *args, **kwargs):
        return super().__init__(*args, **kwargs)
        

    def rank(self, tiethreshold = 0.0, ascending=True):
        table = self.data.table[['hidx', 'vidx', 'hscore', 'vscore', 'weight']]

        m = table.shape[0]
        n = self.data.itemnum
        y = np.zeros(m)
        dat = np.zeros(m*2, dtype=np.float)
        col = np.zeros(m*2, dtype=np.int)
        row = np.zeros(m*2, dtype=np.int)
        for i, itm in enumerate(table.itertuples(index=False, name=None)):
            row[i*2]=i; col[i*2]=itm[0]; dat[i*2]=itm[4];
            row[i*2+1]=i; col[i*2+1]=itm[1]; dat[i*2+1]=-itm[4];
            if np.abs(itm[2]-itm[3])<=tiethreshold:
                y[i]=0.0
            else:
                y[i] = itm[4]*(itm[2]-itm[3])
    
        X = coo_matrix((dat, (row, col)), shape=(m, n))
        X = X.tocsr()

        rst = lsqr(X, y)
        rating = rst[0]
        if hasattr(self, "rating"):
            self.rating["rating"] = rating
        else:
            self.rating = pd.DataFrame({
                "iidx": np.arange(self.data.itemnum, dtype=np.int),
                "rating": rating})

        return self._showcase(ascending)


class ColleyRanker(UnsupervisedRanker):
    def __init__(self, *args, **kwargs):
        return super().__init__(*args, **kwargs)

    def rank(self, tiethreshold = 0.0, ascending=True):
        table = self.data.table[['hidx', 'vidx', 'hscore', 'vscore', 'weight']]

        idx = table.iloc[:, :2]
        score = table.iloc[:, 2:]
        C, b = fast_colley_build(np.require(idx, dtype=np.int32), np.require(score, dtype=np.float64), 
                                 self.data.itemnum, tiethreshold)

        rating = sp.linalg.solve(C, b)
        if hasattr(self, "rating"):
            self.rating["rating"] = rating
        else:
            self.rating = pd.DataFrame({
                "iidx": np.arange(self.data.itemnum, dtype=np.int),
                "rating": rating})

        return self._showcase(ascending)