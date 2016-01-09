import pandas as pd
import numpy as np


class Converter(object):
    """Converter accepts dataframe hdf5 files and converts it to different matrix
    suitable for the needs of ranker. Users can call different matrix generator
    to generate their matrix wanted.
    """

    def __init__(self, filename, filetype="autodetect"):
        # Detect table format
        if filetype == 'autodetect':
            try:
                table = pd.read_hdf(filename, "user_item_rate")
            except KeyError:
                try:
                    table = pd.read_hdf(filename, "item_pair_rate")
                except KeyError:
                    raise KeyError("Failed to detect filetype.")
                else:
                    self.table = table
                    self.filetype = "item_pair_rate"
            else:
                self.table = table
                self.filetype = "user_item_rate"
        elif filetype == 'user_item_rate':
            try:
                table = pd.read_hdf(filename, "user_item_rate")
            except:
                raise
            else:
                self.table = table
                self.filetype = "user_item_rate"
        else:
            try:
                table = pd.read_hdf(filename, "item_pair_rate")
            except:
                raise
            else:
                self.table = table
                self.filetype = "item_pair_rate"

        self.itemlist = self._extract_list(self.table, self.filetype)

    def _extract_list(self, table, filetype):
        if filetype == 'user_item_rate':
            iid = table.loc[:, 'itemid'].unique()
            item_id = pd.DataFrame({'itemid': iid,
                                    'index': pd.Series(range(iid.shape[0]), dtype='int32')})
            return item_id

    def colleymatrix(self):
        item_id = self.itemlist
        table = self.table
        if self.filetype == "user_item_rate":
            watchlist = pd.merge(table, item_id, on='itemid')
            # It is only in this way we can ensure unified behavior!
            watchlist = watchlist[['username', 'itemid', 'rate', 'index']]
            userlist = table.loc[:, 'username'].unique()
            icnt = item_id.shape[0]
            # allocate space for computing
            C = np.zeros((icnt, icnt), dtype=np.float32)
            w = np.zeros((icnt, 2), dtype=np.float32)

            for user in userlist:
                userwatchlist = watchlist[(watchlist.username == user)]
                for i in xrange(userwatchlist.shape[0]-1):
                    for j in xrange(i+1, userwatchlist.shape[0]):
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

            b = np.ravel(1 + 0.5 * (w[:, 0] - w[:, 1]))
            for i in xrange(icnt):
                C[i, i] = 2 + w[i, 0] + w[i, 1]
            return (C, b)
