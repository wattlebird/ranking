import pandas as pd
import numpy as np


class Converter(object):
    """Converter accepts dataframe hdf5 files and converts it to different matrix
    suitable for the needs of ranker. Users can call different matrix generator
    to generate their matrix wanted.
    """

    def __init__(self, filename):
        # Detect table format
        table = pd.read_hdf(filename, "item_pair_rate")
        self.table = table[['primary','secondary','rate1','rate2','weight']]
        # itemid to index table
        idx = self._extract_list(self.table, self.filetype)
        self.itemlist = idx
        temptable = table.iloc[:,:2].values
        pair = np.fromfunction(np.vectorize(lambda i, j: idx[temptable[i,j]]),
                        temptable.shape, dtype=np.int32)
        self.pair = pair

    def _extract_list(self, table, filetype):
        iid = np.hstack([table.loc[:,'primary'].values,
              table.loc[:,'secondary'].values])
        iid = np.unique(iid)
        # index to itemid
        self.iid = iid
        item_id = dict(zip(iid, range(iid.shape[0])))
        return item_id

    def ColleyMatrix(self):
        idx = self.itemlist
        table = self.table

        icnt = len(idx)
        # allocate space for computing
        C = np.zeros((icnt, icnt), dtype=np.float32)
        pair = self.pair

        fast_colley_matrix_build(pair, table.iloc[:,2:].values, C)

        return C

    def ColleyVector(self):
        idx = self.itemlist
        table = self.table

        icnt = len(idx)
        # allocate space for computing
        b = np.zeros(icnt, dtype=np.float32)
        pair = self.pair

        fast_colley_vector_build(pair, table.iloc[:,2:].values, b)

        return b


    def PointDifferenceVoteMatrix(self):
        """This function outputs only Point Difference Matrix.
        """
        idx = self.itemlist
        table = self.table

        icnt = len(idx)
        # allocate space for computing
        D = np.zeros((icnt, icnt), dtype=np.float32)
        pair = self.pair

        fast_point_diff_vote_matrix_build(pair, table.iloc[:,2:].values, D)

        return D

    def MasseyMatrix(self):
        """This function produces X'WX
        """
        pass

    def MasseyVector(self):
        """This function produces X'Wy
        """
        pass
