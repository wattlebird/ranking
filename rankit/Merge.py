import numpy as np
import pandas as pd
from .Ranker import *
from .Table import Table


def borda_count_merge(rankings):
    if not isinstance(rankings, list):
        raise ValueError('rankings should be a list of ranker result.')
    if not all([isinstance(x, pd.DataFrame) for x in rankings]):
        raise ValueError('all items in rankings list should be pandas dataframe.')
    
    t = pd.concat([itm.set_index('name').sort_index()['rank'] for itm in rankings], axis=1)
    t = (t.shape[0]*len(rankings)-t.sum(axis=1)).rename('BordaCount').to_frame()
    t['rank']=t.BordaCount.rank(method='min', ascending=False).astype(np.int32)
    return t.sort_values(by='BordaCount', ascending=False).reset_index()

def average_ranking_merge(rankings):
    if not isinstance(rankings, list):
        raise ValueError('rankings should be a list of ranker result.')
    if not all([isinstance(x, pd.DataFrame) for x in rankings]):
        raise ValueError('all items in rankings list should be pandas dataframe.')
    
    t = pd.concat([itm.set_index('name').sort_index()['rank'] for itm in rankings], axis=1)
    t = t.mean(axis=1).rename('AverageRank').to_frame()
    t['rank']=t.AverageRank.rank(method='min').astype(np.int32)
    return t.sort_values(by='AverageRank').reset_index()

def simulation_aggreation_merge(rankings, baseline, method='od'):
    if not isinstance(rankings, list):
        raise ValueError('rankings should be a list of ranker result.')
    if not all([isinstance(x, pd.DataFrame) for x in rankings]):
        raise ValueError('all items in rankings list should be pandas dataframe.')
    
    vhost = []
    vvisit = []
    vhscore = []
    vvscore = []
    for it in rankings:
        for i in range(it.shape[0]):
            for j in range(i+1, it.shape[0]):
                host = it.loc[i, 'name']
                visit = it.loc[j, 'name']
                delta = it.loc[j, 'rank'] - it.loc[i, 'rank'] # host wins delta score over visit
                hscore = baseline if delta<0 else baseline+delta
                vscore = baseline-delta if delta<0 else baseline
                vhost.append(host)
                vvisit.append(visit)
                vhscore.append(hscore)
                vvscore.append(vscore)
    sim = pd.DataFrame(data={
        'host': vhost,
        'visit': vvisit,
        'hscore': vhscore,
        'vscore': vvscore
    }, columns=['host', 'visit', 'hscore', 'vscore'])
    data = Table(data=sim, col = [0, 1, 2, 3])

    if method=='massey':
        ranker = MasseyRanker(table=data)
        return ranker.rank(ascending=False)
    elif method=='colley':
        ranker = ColleyRanker(table=data)
        return ranker.rank(ascending=False)
    elif method=='keener':
        ranker = KeenerRanker(table=data)
        return ranker.rank(ascending=False)
    elif method=='markov':
        ranker = MarkovRanker(table=data)
        return ranker.rank(ascending=False)
    elif method=='od':
        ranker = ODRanker(table=data)
        return ranker.rank(output='summary', ascending=False)
    elif method=='difference':
        ranker = DifferenceRanker(table=data)
        return ranker.rank(ascending=False)
    else:
        raise ValueError('method not available. Available methods are: massey, colley, keener, markov, od and difference.')

__all__ = ['borda_count_merge', 'average_ranking_merge', 'simulation_aggreation_merge']