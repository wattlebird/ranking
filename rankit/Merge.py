import numpy as np
import pandas as pd


def borda_count_merge(rankings):
    if not isinstance(rankings, list):
        raise ValueError('rankings should be a list of ranker result.')
    if not all([isinstance(x, pd.DataFrame) for x in rankings]):
        raise ValueError('all items in rankings list should be pandas dataframe.')
    
    t = pd.concat([itm.set_index('name').sort_index()['rank'] for itm in rankings], axis=1)
    t = (t.shape[0]*2-t.sum(axis=1)).rename('BordaCount').to_frame()
    t['rank']=t.BordaCount.rank(method='min', ascending=False).astype(np.int32)
    return t.sort_values(by='BordaCount').reset_index()

def average_ranking_merge(rankings):
    if not isinstance(rankings, list):
        raise ValueError('rankings should be a list of ranker result.')
    if not all([isinstance(x, pd.DataFrame) for x in rankings]):
        raise ValueError('all items in rankings list should be pandas dataframe.')
    
    t = pd.concat([itm.set_index('name').sort_index()['rank'] for itm in rankings], axis=1)
    t = t.mean(axis=1).rename('AverageRank').to_frame()
    t['rank']=t.BordaCount.rank(method='min', ascending=False).astype(np.int32)
    return t.sort_values(by='AverageRank').reset_index()

def simulation_aggreation_merge(rankings, baseline, method):
    if not isinstance(rankings, list):
        raise ValueError('rankings should be a list of ranker result.')
    if not all([isinstance(x, pd.DataFrame) for x in rankings]):
        raise ValueError('all items in rankings list should be pandas dataframe.')
    
    