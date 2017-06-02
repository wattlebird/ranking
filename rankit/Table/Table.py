import pandas as pd
import numpy as np
import csv


class Table(object):
    """A Table object in rankit is equivalent to data. 
    It provides an interface to all ranking solutions in rankit.

    1. Table accepts <item1, item2, score1, score2> formatted input in pandas.dataframe/tsv/csv...
    2. Table accepts <user, item, score> formatted input in pandas.dataframe/tsv/csv...
    """
    def __init__(self, data, datatype="paired", col="1,2,3,4", delimiter='\t', hasheader=true):
        if not(datatype=="paired" or datatype=="record"):
            raise ValueError("Invalid datatype provided.")

        col = [int(i) for i in col.split(',')]
        if not((datatype=="paired" and len(col)==4) or (datatype=="record" and len(col)==3)):
            raise ValueError("Given column list does not match datatype.");

        if type(data) is str:
            val = []
            with open(data, newline='') as fr:
                iterreader = csv.reader(fr, delimiter=delimiter)
                if hasheader:
                    next(iterreader)
                for row in iterreader:
                    val.extend([row[i-1] for i in col])
                
            if datatype=="paired":
                host, visit, hscore, vscore = zip(*val)
                raw_table = pd.DataFrame({
                    "host": host,
                    "visit": visit,
                    "hscore": np.require(hscore, dtype=np.float),
                    "vscore": np.require(vscore, dtype=np.float)
                    }, columns=["host", "visit", "hscore", "vscore"])
            elif datatype=="record":
                user, item, score = zip(*val)
                raw_table = pd.DataFrame({
                    "user": user,
                    "item": item,
                    "score": np.require(score, dtype=np.float)
                    }, columns=["user", "item", "score"])
        elif type(data)==pd.DataFrame:
            raw_table = data.ix[:, map(lambda x:x-1, col)]
            if datatype=="paired":
                raw_table.columns = ["host", "visit", "hscore", "vscore"]
                raw_table[["hscore", "vscore"]] = raw_table[["hscore", "vscore"]].apply(pd.to_numeric, downcast='float')
            else:
                raw_table.columns = ["user", "item", "score"]
                raw_table[["score"]] = raw_table[["score"]].apply(pd.to_numeric, downcast='float')

        self.raw_table = raw_table
        self.datatype = datatype


    

