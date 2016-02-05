import pandas as pd
import numpy as np
from util import fast_convert
import sys


def user_rate_converter(filename, destfilename=None,
                        converttype='arithmetic_mean'):
    """This function is designed to convert a user_item_rate dataframe to
    item_pair_rate dataframe.
    There are altogether four convert types:
    'arithmetic_mean': one contest is the arithmetic mean of all watch records.
    'log_mean': one record is the log geometric mean of all watch records.
    'probability': compare by statistical frequency.
    """

    table = pd.read_hdf(filename, 'user_item_rate')
    table = table[['username', 'itemid', 'rate']]
    table.sort_values(by=['username', 'itemid'], inplace=True)
    iid = table.loc[:, 'itemid'].unique()
    idx = dict(zip(iid, range(iid.shape[0])))
    maxitem = len(idx)

    matx = np.zeros((maxitem, maxitem), dtype=np.float32)
    count = np.zeros((maxitem, maxitem), dtype=np.float32)

    b = 0
    for e in xrange(1, table.shape[0]+1):
        if e==table.shape[0] or table.iloc[e, 0]!=table.iloc[b, 0]: # new block start
            if e-b==1:
                b=e;
                continue;

            temptable = table.iloc[b:e]
            idxlst = np.zeros(e-b, dtype=np.int32)
            for i in xrange(e-b):
                idxlst[i] = idx[temptable.iloc[i, 1]]
            ratelst = np.require(temptable.iloc[:, 2].values, dtype=np.float32)
            if converttype=='arithmetic_mean':
                operation=1
            elif converttype=='log_mean':
                operation=2
            else:
                operation=3
            fast_convert(idxlst, ratelst, matx, count, operation)
            b=e
    count = np.triu(count+count.T, k=1)

    if destfilename is not None:
        # matrix is constructed. Now build the item pair table.
        primary = []
        secondary = []
        rate1 = []
        rate2 = []
        for i in xrange(maxitem):
            for j in xrange(i+1, maxitem):
                if count[i, j]!=0:
                    primary.append(iid[i])
                    secondary.append(iid[j])
                    rate1.append(matx[i, j]/count[i, j])
                    rate2.append(matx[j, i]/count[i, j])
        newtable = pd.DataFrame({'primary': primary,
                                 'secondary': secondary,
                                 'rate1': rate1,
                                 'rate2': rate2,
                                 'weight': np.ones(np.count_nonzero(count),
                                                   dtype=np.float32)},
                                 columns = ['primary', 'secondary', 'rate1',
                                            'rate2', 'weight'])
        newtable.to_hdf(destfilename, 'item_pair_rate')
    return (matx, count)

def run():
    while True:
        src = raw_input("Please input the source filename: ")
        if src: break
    dest = raw_input("Please input the dest filename (left black is the same as source file): ")
    if not dest: dest=src
    method = raw_input("Please input the matrix type: ")
    while True:
        if method !='arithmetic_mean' and method!='log_mean' and method!='probability':
            method = raw_input("Invalid matrix type. Please input the matrix type: ")
        else: break;
    user_rate_converter(src, dest, method)

if __name__ == '__main__':
    run()
