from __future__ import division
import numpy as np
import pandas as pd
import scipy as sp
from rankit.Table import Table
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import lsqr
from .matrix_build import fast_colley_build
from numpy.linalg import norm

class UnsupervisedRanker(object):
    """Base class for all unsupervised ranking algorithms."""
    def __init__(self, table, method):
        self.data = table
        self.method = method

    def rank(self, *args, **kwargs):
        raise NotImplementedError("UnsupervisedRanker is a abstract class.")

    def _showcase(self, ascending=False):
        # one need to translate item index to item name.
        indexlut = self.data.indexlut
        rating = self.rating # iitm, rating
        itemname = []
        for row in rating.itertuples(index=False, name=None):
            itemname.append(indexlut[row[0]])
        rst = pd.DataFrame({
            "name": itemname,
            "rating": rating["rating"]})
        rst['rank'] = rst.rating.rank(method=self.method, ascending=ascending).astype(np.int32)
        return rst.sort_values(by=['rating', 'name'], ascending=ascending).reset_index(drop=True)

class MasseyRanker(UnsupervisedRanker):
    def __init__(self, table, method='min'):
        return super().__init__(table, method)
        

    def rank(self, tiethreshold = 0.0, ascending=True):
        table = self.data.table[['hidx', 'vidx', 'hscore', 'vscore', 'weight']]

        m = table.shape[0]
        n = self.data.itemnum
        y = np.zeros(m)
        dat = np.zeros(m*2, dtype=np.float)
        col = np.zeros(m*2, dtype=np.int)
        row = np.zeros(m*2, dtype=np.int)
        for i, itm in enumerate(table.itertuples(index=False, name=None)):
            row[i*2]=i; col[i*2]=itm[0]; dat[i*2]=itm[4];
            row[i*2+1]=i; col[i*2+1]=itm[1]; dat[i*2+1]=-itm[4];
            if np.abs(itm[2]-itm[3])<=tiethreshold:
                y[i]=0.0
            else:
                y[i] = itm[4]*(itm[2]-itm[3])
    
        X = coo_matrix((dat, (row, col)), shape=(m, n))
        X = X.tocsr()

        rst = lsqr(X, y)
        rating = rst[0]
        if hasattr(self, "rating"):
            self.rating["rating"] = rating
        else:
            self.rating = pd.DataFrame({
                "iidx": np.arange(self.data.itemnum, dtype=np.int),
                "rating": rating})

        return self._showcase(ascending)


class ColleyRanker(UnsupervisedRanker):
    def __init__(self, table, method='min'):
        return super().__init__(table, method)

    def rank(self, tiethreshold = 0.0, ascending=True):
        table = self.data.table[['hidx', 'vidx', 'hscore', 'vscore', 'weight']]

        idx = table.iloc[:, :2]
        score = table.iloc[:, 2:]
        C, b = fast_colley_build(np.require(idx, dtype=np.int32), np.require(score, dtype=np.float64), 
                                 self.data.itemnum, tiethreshold)

        rating = sp.linalg.solve(C, b)
        if hasattr(self, "rating"):
            self.rating["rating"] = rating
        else:
            self.rating = pd.DataFrame({
                "iidx": np.arange(self.data.itemnum, dtype=np.int),
                "rating": rating})

        return self._showcase(ascending)

class KeenerRanker(UnsupervisedRanker):
    def __init__(self, table, method='min'):
        return super().__init__(table, method)
    
    def rank(self, func=None, epsilon=1e-4, threshold=1e-4, ascending=True):
        mtx = pd.DataFrame(data={
            'hidx': pd.concat([self.data.table.hidx, self.data.table.vidx]),
            'vidx': pd.concat([self.data.table.vidx, self.data.table.hidx]),
            'hscore': pd.concat([self.data.table.hscore, self.data.table.vscore]),
            'vscore': pd.concat([self.data.table.vscore, self.data.table.hscore]),
            'weight': pd.concat([self.data.table.weight, self.data.table.weight])
        }, columns = ['hidx', 'vidx', 'hscore', 'vscore', 'weight']).reset_index(drop=True)
        mtx['score'] = mtx.hscore+mtx.vscore
        mtx['hscore'] = (mtx['hscore']+1)/(mtx['score']+2)
        mtx['vscore'] = (mtx['vscore']+1)/(mtx['score']+2)
        if func is not None:
            mtx['hscore'] = mtx.hscore.apply(func)
            mtx['vscore'] = mtx.vscore.apply(func)
        mtx['hscore'] = mtx['hscore']*mtx['weight']
        mtx['vscore'] = mtx['vscore']*mtx['weight']
        mtx = mtx.groupby(['hidx', 'vidx'])[['hscore', 'vscore']].mean()
        mtx.reset_index(inplace=True)

        D = coo_matrix((mtx.hscore.values, (mtx.hidx.values, mtx.vidx.values)), shape=(self.data.itemnum, self.data.itemnum)).tocsr()

        r = np.ones(self.data.itemnum)/self.data.itemnum
        pr = np.ones(self.data.itemnum)
        while norm(pr-r)>threshold:
            pr = r
            rho = np.sum(r)*epsilon
            r = D.dot(r)+rho*np.ones(self.data.itemnum)
            r /= np.sum(r)

        if hasattr(self, "rating"):
            self.rating["rating"] = r
        else:
            self.rating = pd.DataFrame({
                "iidx": np.arange(self.data.itemnum, dtype=np.int),
                "rating": r})
        return self._showcase(ascending)

class MarkovRanker(UnsupervisedRanker):
    def __init__(self, table, method='min'):
        return super().__init__(table, method)
    
    def rank(self, restart=0.3, threshold=1e-4, ascending=True):
        if restart>1 or restart<0:
            raise ValueError("restart rate should be between 0 and 1.")
        mtx = pd.DataFrame(data={
            'hidx': pd.concat([self.data.table.hidx, self.data.table.vidx]),
            'vidx': pd.concat([self.data.table.vidx, self.data.table.hidx]),
            'hscore': pd.concat([self.data.table.hscore, self.data.table.vscore]),
            'vscore': pd.concat([self.data.table.vscore, self.data.table.hscore]),
            'weight': pd.concat([self.data.table.weight, self.data.table.weight])
        }, columns = ['hidx', 'vidx', 'hscore', 'vscore', 'weight']).reset_index(drop=True)
        mtx['hscore'] = mtx['hscore']*mtx['weight']
        mtx['vscore'] = mtx['vscore']*mtx['weight']
        mtx = mtx.groupby(['hidx', 'vidx'])[['hscore', 'vscore']].mean()
        mtx = pd.concat([mtx.reset_index().set_index('hidx'), mtx.groupby('hidx').vscore.sum().rename('htotalvote')], axis=1).reset_index()
        mtx['prob'] = mtx['vscore']/mtx['htotalvote']

        D = coo_matrix((mtx.prob.values, (mtx.hidx.values, mtx.vidx.values)), shape=(self.data.itemnum, self.data.itemnum)).transpose().tocsr()
        r = np.ones(self.data.itemnum)/self.data.itemnum
        pr = np.ones(self.data.itemnum)
        while norm(pr-r)>threshold:
            pr = r
            vrestart = restart*np.ones(self.data.itemnum)/self.data.itemnum
            r = (1-restart)*D.dot(r)+vrestart
            r /= np.sum(r)
        
        if hasattr(self, "rating"):
            self.rating["rating"] = r
        else:
            self.rating = pd.DataFrame({
                "iidx": np.arange(self.data.itemnum, dtype=np.int),
                "rating": r})
        return self._showcase(ascending)

class ODRanker(UnsupervisedRanker):
    def __init__(self, table, method='min'):
        return super().__init__(table, method)
    
    def rank(self, output='summary', epsilon=1e-4, threshold=1e-4, ascending=True):
        mtx = pd.DataFrame(data={
            'hidx': pd.concat([self.data.table.hidx, self.data.table.vidx]),
            'vidx': pd.concat([self.data.table.vidx, self.data.table.hidx]),
            'hscore': pd.concat([self.data.table.hscore, self.data.table.vscore]),
            'vscore': pd.concat([self.data.table.vscore, self.data.table.hscore]),
            'weight': pd.concat([self.data.table.weight, self.data.table.weight])
        }, columns = ['hidx', 'vidx', 'hscore', 'vscore', 'weight']).reset_index(drop=True)
        mtx['hscore'] = mtx['hscore']*mtx['weight']
        mtx['vscore'] = mtx['vscore']*mtx['weight']
        mtx = mtx.groupby(['hidx', 'vidx'])[['hscore', 'vscore']].mean()
        mtx.reset_index(inplace=True)

        D = coo_matrix((mtx.vscore.values, (mtx.hidx.values, mtx.vidx.values)), shape=(self.data.itemnum, self.data.itemnum)).tocsr()
        Dt = D.transpose()

        prevd = np.ones(self.data.itemnum)/self.data.itemnum
        d = np.ones(self.data.itemnum)
        while norm(prevd-d)>threshold:
            prevd = d
            o = Dt.dot(1/d)+epsilon/d
            d = D.dot(1/o)+epsilon/o
        o = Dt.dot(1/d)

        if output=='summary':
            r = o/d
        elif output=='offence':
            r = o
        elif output=='defence':
            r = d
        else:
            raise ValueError('ouput should be one of summary, offence or defence.')
        if hasattr(self, "rating"):
            self.rating["rating"] = r
        else:
            self.rating = pd.DataFrame({
                "iidx": np.arange(self.data.itemnum, dtype=np.int),
                "rating": r})
        return self._showcase(ascending)

class DifferenceRanker(UnsupervisedRanker):
    def __init__(self, table, method='min'):
        return super().__init__(table, method)

    def rank(self, ascending=True):
        mtx = pd.DataFrame(data={
            'hidx': pd.concat([self.data.table.hidx, self.data.table.vidx]),
            'vidx': pd.concat([self.data.table.vidx, self.data.table.hidx]),
            'hscore': pd.concat([self.data.table.hscore, self.data.table.vscore]),
            'vscore': pd.concat([self.data.table.vscore, self.data.table.hscore]),
            'weight': pd.concat([self.data.table.weight, self.data.table.weight])
        }, columns = ['hidx', 'vidx', 'hscore', 'vscore', 'weight']).reset_index(drop=True)
        mtx['score'] = mtx['hscore']-mtx['vscore']
        mtx['score'] = mtx['score']*mtx['weight']
        mtx = mtx.groupby(['hidx', 'vidx']).score.mean().reset_index()
        r = mtx.groupby('hidx').score.sum()/self.data.itemnum
        r = r.sort_index()

        if hasattr(self, "rating"):
            self.rating["rating"] = r.values
        else:
            self.rating = pd.DataFrame({
                "iidx": np.arange(self.data.itemnum, dtype=np.int),
                "rating": r})
        return self._showcase(ascending)

class EloRanker(UnsupervisedRanker):
    def __init__(self, table, method='min'):
        if not table.table.columns.contains('time'):
            raise ValueError('The passed in table has no time information provided.')
        return super().__init__(table, method)

    def rank(self, K = 10, baseline = 0, xi=400, ascending=True):
        self.xi = xi
        self.K = K
        self.baseline = baseline
        self.ascending = ascending
        rating = baseline*np.ones(self.data.itemnum)
        t = self.data.table.sort_values(by='time', ascending=True)
        # refer to https://fivethirtyeight.com/features/how-we-calculate-nba-elo-ratings/ for margin involvement
        for itm in t.itertuples():
            s = 0.5 if itm.hscore == itm.vscore else (1 if itm.hscore > itm.vscore else 0)
            ha = 0 if itm.hostavantage<0 else itm.hostavantage
            va = 0 if itm.hostavantage>0 else -itm.hostavantage
            phwin = 1/(1+10**((rating[itm.vidx] + va - rating[itm.hidx] - ha)/xi))
            hmargin = (abs(itm.hscore-itm.vscore)+3)**0.8/(7.5+0.0006*(rating[itm.hidx] + ha - rating[itm.vidx] - va))
            delta = K*itm.weight*hmargin*(s-phwin)
            rating[itm.hidx] += delta
            rating[itm.vidx] -= delta
        
        if hasattr(self, "rating"):
            self.rating["rating"] = rating
        else:
            self.rating = pd.DataFrame({
                "iidx": np.arange(self.data.itemnum, dtype=np.int),
                "rating": rating})
        return self._showcase(ascending)

    def prob_win(self, host, visit):
        if not hasattr(self, 'rating'):
            return RuntimeError('No rating information calculated. Please involke rank first.')
        
        r = self.rating.rating.values
        if isinstance(host, np.ndarray) and host.ndim==1:
            host = host.tolist()
        if isinstance(visit, np.ndarray) and visit.ndim==1:
            visit = visit.tolist()
        if isinstance(host, list) and isinstance(visit, list):
            rst = []
            for rawh, rawv in zip(host, visit):
                h = self.data.itemlut[rawh]
                v = self.data.itemlut[rawv]
                rst.append(1/(1+10**((r[v]-r[h])/self.xi)))
            return rst
        else:
            h = self.data.itemlut[host]
            v = self.data.itemlut[visit]
            return 1/(1+10**((r[v]-r[h])/self.xi))
    
    def update(self, newtable):
        xi, K, baseline, ascending = self.xi, self.K, self.baseline, self.ascending
        self.data.update(newtable)
        rating = baseline*np.ones(self.data.itemnum)
        for itm in self.rating.itertuples():
            rating[itm.iidx] = itm.rating
        t = newtable.table.sort_values(by='time', ascending=True)
        for itm in t.itertuples():
            s = 0.5 if itm.hscore == itm.vscore else (1 if itm.hscore > itm.vscore else 0)
            ha = 0 if itm.hostavantage<0 else itm.hostavantage
            va = 0 if itm.hostavantage>0 else -itm.hostavantage
            phwin = 1/(1+10**((rating[itm.vidx] + va - rating[itm.hidx] - ha)/xi))
            hmargin = (abs(itm.hscore-itm.vscore)+3)**0.8/(7.5+0.0006*(rating[itm.hidx] + ha - rating[itm.vidx] - va))
            delta = K*itm.weight*hmargin*(s-phwin)
            rating[itm.hidx] += delta
            rating[itm.vidx] -= delta

        self.rating = pd.DataFrame({
            "iidx": np.arange(self.data.itemnum, dtype=np.int),
            "rating": rating})
        return self._showcase(ascending)
