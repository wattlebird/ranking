import csv
import pandas as pd
import numpy as np


class Table(object):
    """ A Table object in rankit is equivalent to data. 
    It provides an interface to all ranking solutions in rankit.

    Table accepts <item1, item2, score1, score2> formatted input in pandas.dataframe/tsv/csv...
    """

    def __init__(self, data, col=[1,2,3,4], encoding="utf_8", delimiter='\t', hasheader=False):
        if len(col)!=4:
            raise ValueError("Parameter col must have four values, indicating columns for host, visit, host score and visit score.")
        col = [itm-1 for itm in col]

        if isinstance(data, str):
            val = []
            with open(data, newline='', encoding=encoding) as fr:
                iterreader = csv.reader(fr, delimiter=delimiter)
                if hasheader:
                    next(iterreader)
                for row in iterreader:
                    val.append([row[i] for i in col])
            host, visit, hscore, vscore = zip(*val)
            raw_table = pd.DataFrame({
                "host": host,
                "visit": visit,
                "hscore": np.require(hscore, dtype=np.float),
                "vscore": np.require(vscore, dtype=np.float)
                }, columns=["host", "visit", "hscore", "vscore"])
        elif isinstance(data, pd.DataFrame):
            raw_table = data.ix[:, col].copy()
            raw_table.columns = ["host", "visit", "hscore", "vscore"]
            raw_table.loc[:, ["hscore", "vscore"]] = raw_table.loc[:, ["hscore", "vscore"]].apply(pd.to_numeric)

        self.table = raw_table.dropna()

        itemlut = dict()
        indexlut = []
        idx = 0
        for row in self.table.itertuples(index=False, name=None):
            if not row[0] in itemlut:
                itemlut[row[0]] = idx
                indexlut.append(row[0])
                idx+=1
            if not row[1] in itemlut:
                itemlut[row[1]] = idx
                indexlut.append(row[1])
                idx+=1

        self.itemlut = itemlut
        self.indexlut = indexlut
        self.itemnum = idx

        # raw table need to be converted to standard indexed table.
        hidx = np.require(list(map(lambda x: itemlut[x], raw_table["host"].tolist())), dtype=np.int)
        vidx = np.require(list(map(lambda x: itemlut[x], raw_table["visit"].tolist())), dtype=np.int)
        table = pd.DataFrame({
            "host": raw_table["host"],
            "visit": raw_table["visit"],
            "hidx": hidx,
            "vidx": vidx,
            "hscore": raw_table["hscore"],
            "vscore": raw_table["vscore"],
        }, columns=["host", "visit", "hidx", "vidx", "hscore", "vscore"])

        self._table = table
    
    def gettable(self):
        return self.table.copy()

    def getitemlist(self):
        return self.table.loc[:, ["host", "visit"]].copy()

    def _gettable(self):
        return self._table.loc[:, ["hidx", "vidx", "hscore", "vscore"]]
    
    def __repr__(self):
        return "Table with provided data:\n"+self.gettable().__repr__()