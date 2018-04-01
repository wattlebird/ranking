import csv
import pandas as pd
import numpy as np


class Table(object):
    """ A Table object in rankit is equivalent to data. 
    It provides an interface to all ranking solutions in rankit.

    Table accepts <item1, item2, score1, score2> formatted input in pandas.dataframe.

    Parameters
    ----------
    data: pandas.DataFrame
        Game result of paired players.
    col: list of index or column names
        Index or column names should indicating ['player1', 'player2', 'score1', 'score2']
    weightcol: index or name of column indicating weight
    timecol: index or name of column indicating time. Compulsory column for Elo Rank.
    hostavantagecol: index or name of column indicating host advantage

    Returns
    -------
    Table object to be fed to Rankers.
    """

    def __init__(self, data, col, weightcol=None, timecol=None, hostavantagecol=None):
        if len(col)!=4:
            raise ValueError("Parameter col must have four values, indicating columns for host, visit, host score and visit score.")
        if (not all(isinstance(i, str) for i in col)) and (not all(isinstance(i, int) for i in col)):
            raise ValueError("The type of col elements should be string or int.")
        if not isinstance(data, pd.DataFrame):
            raise ValueError("data should be pandas dataframe.")

        raw_table = data.iloc[:, col].copy() if all(isinstance(i, int) for i in col) else data[col].copy()
        raw_table.columns = ["host", "visit", "hscore", "vscore"]
        raw_table.loc[:, ["hscore", "vscore"]] = raw_table.loc[:, ["hscore", "vscore"]].apply(pd.to_numeric)

        if weightcol is not None:
            raw_table['weight'] = data.iloc[:, weightcol].copy() if isinstance(weightcol, int) else data.loc[:, weightcol].copy()
        else:
            raw_table['weight'] = 1.0
        
        if timecol is not None:
            raw_table['time'] = data.iloc[:, timecol].copy() if isinstance(timecol, int) else data.loc[:, timecol].copy()
        
        if hostavantagecol is not None:
            raw_table['hostavantage'] = data.iloc[:, hostavantagecol].copy() if isinstance(hostavantagecol, int) else data.loc[:, hostavantagecol].copy()
        else:
            raw_table['hostavantage'] = 0.0

        itemlut = dict()
        indexlut = []
        idx = 0
        for row in raw_table.itertuples(index=False, name=None):
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
        raw_table['hidx'] = np.require(list(map(lambda x: itemlut[x], raw_table["host"].tolist())), dtype=np.int)
        raw_table['vidx'] = np.require(list(map(lambda x: itemlut[x], raw_table["visit"].tolist())), dtype=np.int)

        self.table = raw_table

    def update(self, table):
        # update itemlut, indexlut, itemnum
        p = len(self.indexlut)
        for k,v in table.itemlut.items():
            if self.itemlut.get(k) is None:
                self.itemlut[k] = p
                self.indexlut.append(k)
                p += 1
        self.itemnum = p

        # update self.table
        table.table.hidx = table.table.host.apply(lambda x: self.itemlut[x])
        table.table.vidx = table.table.visit.apply(lambda x: self.itemlut[x])
        if table.table.columns.contains('time') and self.table.columns.contains('time') and \
            table.table.time.min()<self.table.time.max():
            raise ValueError('Given record\'s time should be no earlier than existing record.')
        self.table = pd.concat([self.table, table.table], ignore_index=True)


    def getitemlist(self):
        return self.table[["host", "visit"]].copy()

    def _gettable(self):
        return self.table[["hidx", "vidx", "hscore", "vscore"]]
    
    def __repr__(self):
        return "Table with provided data:\n"+self.table[['host', 'visit', 'hscore', 'vscore']].__repr__()