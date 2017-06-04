import csv
import pandas as pd
import numpy as np
from .convert import fast_record_to_pairwise


class Table(object):
    """ A Table object in rankit is equivalent to data. 
    It provides an interface to all ranking solutions in rankit.

    1. Table accepts <item1, item2, score1, score2> formatted input in pandas.dataframe/tsv/csv...
    2. Table accepts <user, item, score> formatted input in pandas.dataframe/tsv/csv...
    """

    def __init__(self, data, datatype="paired", col="1,2,3,4", encoding="utf_8", delimiter='\t', hasheader=False):
        if not (datatype == "paired" or datatype == "record"):
            raise ValueError("Invalid datatype provided.")

        col = [int(i) for i in col.split(',')]
        if not((datatype == "paired" and len(col) == 4) or 
               (datatype == "record" and len(col) == 3)):
            raise ValueError("Given column list does not match datatype.")

        if isinstance(data, str):
            val = []
            with open(data, newline='', encoding=encoding) as fr:
                iterreader = csv.reader(fr, delimiter=delimiter)
                if hasheader:
                    next(iterreader)
                for row in iterreader:
                    val.append([row[i-1] for i in col])
            if datatype == "paired":
                host, visit, hscore, vscore = zip(*val)
                raw_table = pd.DataFrame({
                    "host": host,
                    "visit": visit,
                    "hscore": np.require(hscore, dtype=np.float),
                    "vscore": np.require(vscore, dtype=np.float)
                    }, columns=["host", "visit", "hscore", "vscore"])
            elif datatype == "record":
                user, item, score = zip(*val)
                score = [None if itm=="" else float(itm) for itm in score]
                raw_table = pd.DataFrame({
                    "user": user,
                    "item": item,
                    "score": np.require(score, dtype=np.float)
                    }, columns=["user", "item", "score"])
        elif isinstance(data, pd.DataFrame):
            raw_table = data.ix[:, map(lambda x: x-1, col)]
            # but how to ensure np.float64?
            if datatype=="paired":
                raw_table.columns = ["host", "visit", "hscore", "vscore"]
                raw_table[["hscore", "vscore"]] = raw_table[["hscore", "vscore"]].apply(pd.to_numeric)
            else:
                raw_table.columns = ["user", "item", "score"]
                raw_table[["score"]] = raw_table[["score"]].apply(pd.to_numeric)

        self.raw_table = raw_table
        self.datatype = datatype

        itemlut = dict()
        indexlut = []
        idx = 0
        if datatype=="paired":
            for row in raw_table.itertuples(index=False, name=None):
                if not row[0] in itemlut:
                    itemlut[row[0]] = idx
                    indexlut.append(row[0])
                    idx+=1
                if not row[1] in itemlut:
                    itemlut[row[1]] = idx
                    indexlut.append(row[1])
                    idx+=1
        else:
            for row in raw_table.itertuples(index=False, name=None):
                if not row[1] in itemlut:
                    itemlut[row[1]] = idx
                    indexlut.append(row[1])
                    idx+=1

        self.itemlut = itemlut
        self.indexlut = indexlut
        self.itemnum = idx

        # raw table need to be converted to standard indexed table.
        if datatype=="paired":
            hidx = np.require(list(map(lambda x: itemlut[x], raw_table["host"].tolist())), dtype=np.int)
            vidx = np.require(list(map(lambda x: itemlut[x], raw_table["visit"].tolist())), dtype=np.int)
            table = pd.DataFrame({
                "hidx": hidx,
                "vidx": vidx,
                "hscore": raw_table["hscore"],
                "vscore": raw_table["vscore"],
                "weight": np.ones(len(hidx), dtype=np.float)
            }, columns=["hidx", "vidx", "hscore", "vscore", "weight"])
            table.sort_values(["hidx", "vidx"], ascending=True, inplace=True)
        else:
            idx = np.require(list(map(lambda x: itemlut[x], raw_table["item"].tolist())), dtype=np.int)
            table = pd.DataFrame({
                "user": raw_table["user"],
                "iidx": idx,
                "score": raw_table["score"]
            }, columns=["user", "iidx", "score"])
            table.sort_values(["user", "iidx"], ascending=True, inplace=True)

        self.table = table.dropna()


    def _gettable(self, datatype="paired"):
        if self.datatype==datatype:
            return self.table
        else:
            if datatype!="paired" and datatype!="record":
                raise ValueError("Invalid datatype provided.")
            if datatype=="record":
                raise NotImplementedError("Sorry, it seems that no scenario exists for converting paired data to record data. Please raise an issue to describe your scenario so I would implement that.")
            else:
                if hasattr(self, 'converted_table'):
                    return self.converted_table
                self.converted_table = self._record_to_paired()
                return self.converted_table

    def gettabletype(self):
        return self.datatype
    
    def getrawtable(self):
        return self.raw_table
                
    def _record_to_paired(self):
        gptable = self.table.groupby("user")
        userlist = self.table["user"].unique()
        pairitemlist = []
        pairscorelist = []
        for u in userlist:
            pairitem, pairscore = fast_record_to_pairwise(
                np.require(gptable.get_group(u)["iidx"].values, dtype=np.int32), 
                np.require(gptable.get_group(u)["score"].values, dtype=np.float64))
            pairitemlist.append(pairitem)
            pairscorelist.append(pairscore)
        p = np.vstack(pairitemlist)
        s = np.vstack(pairscorelist)
        return pd.DataFrame({
            "hidx": p[:, 0],
            "vidx": p[:, 1],
            "hscore": s[:, 0],
            "vscore": s[:, 1],
            "weight": np.ones(p.shape[0], dtype=np.float)
        }, columns=["hidx", "vidx", "hscore", "vscore", "weight"])

    
