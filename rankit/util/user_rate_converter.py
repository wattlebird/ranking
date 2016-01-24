import pandas as pd
import numpy as np
from joblib import Parallel, delayed


def user_rate_converter(filename, filetype='user_item_rate', converttype='one_by_one'):
    """This function is designed to convert a user_item_rate dataframe to
    item_pair_rate dataframe.
    There are altogether four convert types:
    'one_by_one': stores the original watch record. We add weight to save memory.
    'arithmetic_mean': one contest is the arithmetic mean of all watch records.
    'log_mean': one record is the log geometric mean of all watch records.
    'probability': compare by statistical frequency.
    """

    table = pd.read_hdf(filename, filetype)
    userlist = table.loc[:,'username'].unique()

    if n_jobs==1:
        rst = []
        for user in userlist:
            rst.append(expand_watchlist(table, user))
    else:
        rst = Parallel(n_jobs=n_jobs)(
                delayed(expand_watchlist)(table, user)
                for user in userlist)
    return pd.concat(rst, ignore_index=True)

def expand_watchlist(table, user):
    watchlist = table[table.loc[:,'username']==user].\
                sort_values(by='itemid')
    watchlist = watchlist[['username', 'itemid', 'rate']]
    length = watchlist.shape[0]*(watchlist.shape[0]-1)/2
    itemleft = pd.Series(np.empty(length, dtype=watchlist['itemid'].dtype))
    itemright = pd.Series(np.empty(length, dtype=watchlist['itemid'].dtype))
    rateleft = pd.Series(np.empty(length, dtype=np.float32))
    rateright = pd.Series(np.empty(length, dtype=np.float32))
    weight = pd.Series(np.empty(length, dtype=np.float32))
    k=0
    for i in xrange(watchlist.shape[0]-1):
        for j in xrange(i+1, watchlist.shape[0]):
            itemleft.iloc[k] = watchlist.iloc[i,1]
            itemright.iloc[k] = watchlist.iloc[j,1]
            rateleft.iloc[k] = watchlist.iloc[i,2]
            rateright.iloc[k] = watchlist.iloc[i,2]
            weight.iloc[k] = 1.0
            k+=1
    partialpd = pd.DataFrame({'primary':itemleft, 'secondary':itemright,
                              'rate1':rateleft, 'rate2':rateright,
                              'weight':weight},
                              columns=['primary', 'secondary', 'rate1',
                              'rate2', 'weight'])
    return partialpd
