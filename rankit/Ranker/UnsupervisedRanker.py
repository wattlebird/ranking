from __future__ import division
from __future__ import absolute_import
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
    def rank(self, table, **kargs):
        raise NotImplementedError("UnsupervisedRanker is a abstract class.")

    def _showcase(self, table, ascending=False):
        # one need to translate item index to item name.
        indexlut = table.indexlut
        rating = self.rating # iitm, rating
        itemname = []
        for row in rating.itertuples(index=False, name=None):
            itemname.append(indexlut[row[0]])
        rst = pd.DataFrame({
            "name": itemname,
            "rating": rating["rating"]})
        rst['rank'] = rst.rating.rank(method='min', ascending=ascending).astype(np.int32)
        return rst.sort_values(by=['rating', 'name'], ascending=ascending).reset_index(drop=True)

class MasseyRanker(UnsupervisedRanker):
    """Massey ranking system proposed by Kenneth Massey: Statistical models applied to the rating of sports teams. 
    Bachelor's thesis, Bluefield College, 1997.
    Core idea: The competition score difference is the rating difference, so one can solve a linear equation by minimize least square error.

    Parameters
    ----------
    drawMargin: [0, +Inf), default 0.
        When absolute difference between two teams are smaller than drawMargin, this competition is considered as a tie.
    """
    def __init__(self, drawMargin = 0.0):
        self.drawMargin = drawMargin
        

    def rank(self, table):
        """Calculate the rank and rating with specified parameters.

        Parameters
        ----------
        table: Table
            The record table to be ranked, should be a Table object.
        
        Returns
        -------
        pandas.DataFrame, with column ['name', 'rating', 'rank']
        """
        drawMargin = self.drawMargin
        data = table.table[['hidx', 'vidx', 'hscore', 'vscore', 'weight']]

        m = data.shape[0]
        n = table.itemnum
        y = np.zeros(m)
        dat = np.zeros(m*2, dtype=np.float)
        col = np.zeros(m*2, dtype=np.int)
        row = np.zeros(m*2, dtype=np.int)
        for i, itm in enumerate(data.itertuples(index=False, name=None)):
            row[i*2]=i; col[i*2]=itm[0]; dat[i*2]=itm[4];
            row[i*2+1]=i; col[i*2+1]=itm[1]; dat[i*2+1]=-itm[4];
            if np.abs(itm[2]-itm[3])<=drawMargin:
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
                "iidx": np.arange(n, dtype=np.int),
                "rating": rating})

        return self._showcase(table, False)

class ColleyRanker(UnsupervisedRanker):
    """Colley ranking system proposed by Wesley Colley: 
    Colley's bias free college football ranking method: The colley matrix explained, 2002.
    http://www.colleyrankings.com
    Core idea: All team's rating starts from 0.5, and with evolvement of games, the rating of each player deviates from 0.5
    according to probability of win. However, the average rating of all teams remains 0.5.

    Parameters
    ----------
    drawMargin: [0, +Inf), default 0.
        When absolute difference between two teams are smaller than drawMargin, this competition is considered as a tie.
    """
    def __init__(self, drawMargin = 0.0):
        self.drawMargin = drawMargin

    def rank(self, table):
        """Calculate the rank and rating with specified parameters.

        Parameters
        ----------
        table: Table
            The record table to be ranked, should be a Table object.
        
        Returns
        -------
        pandas.DataFrame, with column ['name', 'rating', 'rank']
        """
        drawMargin = self.drawMargin
        data = table.table[['hidx', 'vidx', 'hscore', 'vscore', 'weight']]

        idx = data.iloc[:, :2]
        score = data.iloc[:, 2:]
        C, b = fast_colley_build(np.require(idx, dtype=np.int32), np.require(score, dtype=np.float64), 
                                 table.itemnum, drawMargin)

        rating = sp.linalg.solve(C, b)
        if hasattr(self, "rating"):
            self.rating["rating"] = rating
        else:
            self.rating = pd.DataFrame({
                "iidx": np.arange(table.itemnum, dtype=np.int),
                "rating": rating})

        return self._showcase(table, False)

class KeenerRanker(UnsupervisedRanker):
    """Keener ranking system proposed by James Keener:
    The Perron-Frobenius theorem and the ranking of football teams, SIAM Review, 35(1):80-93, 1993
    The core idea are: 1. rating is proportional to real strength; 2. real strength is measured relatively by competitors' strength.

    Parameters
    ----------
    func: default None.
        If set, the score difference should be transformed by the function first then used for rating calculation.
    epsilon: [0, +Inf) default 1e-4
        The small value that applies an interference to game result that force each team had at least one game with each other.
    threshold: (0, +Inf), default 1e-4
        The threshold that controls when the algorithm will converge.
    """
    def __init__(self, func=None, epsilon=1e-4, threshold=1e-4):
        self.func = func
        self.epsilon = epsilon
        self.threshold = threshold
    
    def rank(self, table):
        """Calculate the rank and rating with specified parameters.

        Parameters
        ----------
        table: Table
            The record table to be ranked, should be a Table object.
        
        Returns
        -------
        pandas.DataFrame, with column ['name', 'rating', 'rank']
        """
        func, epsilon, threshold = self.func, self.epsilon, self.threshold
        mtx = pd.DataFrame(data={
            'hidx': pd.concat([table.table.hidx, table.table.vidx]),
            'vidx': pd.concat([table.table.vidx, table.table.hidx]),
            'hscore': pd.concat([table.table.hscore, table.table.vscore]),
            'vscore': pd.concat([table.table.vscore, table.table.hscore]),
            'weight': pd.concat([table.table.weight, table.table.weight])
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

        D = coo_matrix((mtx.hscore.values, (mtx.hidx.values, mtx.vidx.values)), shape=(table.itemnum, table.itemnum)).tocsr()

        r = np.ones(table.itemnum)/table.itemnum
        pr = np.ones(table.itemnum)
        while norm(pr-r)>threshold:
            pr = r
            rho = np.sum(r)*epsilon
            r = D.dot(r)+rho*np.ones(table.itemnum)
            r /= np.sum(r)

        if hasattr(self, "rating"):
            self.rating["rating"] = r
        else:
            self.rating = pd.DataFrame({
                "iidx": np.arange(table.itemnum, dtype=np.int),
                "rating": r})
        return self._showcase(table, False)

class MarkovRanker(UnsupervisedRanker):
    """Markov ranking is actually PageRank.
    The core idea is voting: in each game, each team will vote to each other by the number of scores they lost.
    If there are multiple games for a certain pair of player, their scores will be grouped and averaged.

    Parameters
    ----------
    restart: [0, 1], default 0.3.
        Random walk with restart: in order to avoid black hole in random walk graph.
    threshold: (0, +Inf), default 1e-4
        The threshold that controls when the algorithm will converge.
    """
    def __init__(self, restart=0.3, threshold=1e-4):
        self.restart = restart
        self.threshold = threshold
    
    def rank(self, table):
        """Calculate the rank and rating with specified parameters.

        Parameters
        ----------
        table: Table
            The record table to be ranked, should be a Table object.
        
        Returns
        -------
        pandas.DataFrame, with column ['name', 'rating', 'rank']
        """
        restart, threshold = self.restart, self.threshold
        if restart>1 or restart<0:
            raise ValueError("restart rate should be between 0 and 1.")
        mtx = pd.DataFrame(data={
            'hidx': pd.concat([table.table.hidx, table.table.vidx]),
            'vidx': pd.concat([table.table.vidx, table.table.hidx]),
            'hscore': pd.concat([table.table.hscore, table.table.vscore]),
            'vscore': pd.concat([table.table.vscore, table.table.hscore]),
            'weight': pd.concat([table.table.weight, table.table.weight])
        }, columns = ['hidx', 'vidx', 'hscore', 'vscore', 'weight']).reset_index(drop=True)
        mtx['hscore'] = mtx['hscore']*mtx['weight']
        mtx['vscore'] = mtx['vscore']*mtx['weight']

        mtx_ = mtx.groupby('hidx').vscore.sum().rename('htotalvote')
        
        mtx = mtx.groupby(['hidx', 'vidx'])[['hscore', 'vscore']].mean()
        mtx = pd.concat([mtx.reset_index().set_index('hidx'), mtx_], axis=1).reset_index()
        mtx['prob'] = mtx['vscore']/mtx['htotalvote']

        D = coo_matrix((mtx.prob.values, (mtx.hidx.values, mtx.vidx.values)), shape=(table.itemnum, table.itemnum)).transpose().tocsr()
        r = np.ones(table.itemnum)/table.itemnum
        pr = np.ones(table.itemnum)
        while norm(pr-r)>threshold:
            pr = r
            vrestart = restart*np.ones(table.itemnum)/table.itemnum
            r = (1-restart)*D.dot(r)+vrestart
            r /= np.sum(r)
        
        if hasattr(self, "rating"):
            self.rating["rating"] = r
        else:
            self.rating = pd.DataFrame({
                "iidx": np.arange(table.itemnum, dtype=np.int),
                "rating": r})
        return self._showcase(table, False)

class ODRanker(UnsupervisedRanker):
    """The Offence-defence rank tries to assign an offence rating and a defence rating to each team.
    By saying "offence rating", we assume that a team has a high offence rating when it gained a lot of points 
    from a team good in defence. Vise versa. The offence rating of a team is associated with defence rating of each
    competitor in a non-linear way. The defence rating of a team is also non-linearly related to each competitors' 
    offence rating.

    Parameters
    ----------
    method: {'summary', 'offence', 'defence'}, default 'summary'.
        The rating to be returned. 'summary' is offence/defence.
    epsilon: [0, +Inf) default 1e-4
        The small value that forces a convergence.
    threshold: (0, +Inf), default 1e-4
        The threshold that controls when the algorithm will converge.
    """
    def __init__(self, method='summary', epsilon=1e-4, threshold=1e-4):
        self.method = method
        self.epsilon = epsilon
        self.threshold = threshold
    
    def rank(self, table):
        """Calculate the rank and rating with specified parameters.

        Parameters
        ----------
        table: Table
            The record table to be ranked, should be a Table object.
        
        Returns
        -------
        pandas.DataFrame, with column ['name', 'rating', 'rank']
        """
        method, epsilon, threshold = self.method, self.epsilon, self.threshold
        mtx = pd.DataFrame(data={
            'hidx': pd.concat([table.table.hidx, table.table.vidx]),
            'vidx': pd.concat([table.table.vidx, table.table.hidx]),
            'hscore': pd.concat([table.table.hscore, table.table.vscore]),
            'vscore': pd.concat([table.table.vscore, table.table.hscore]),
            'weight': pd.concat([table.table.weight, table.table.weight])
        }, columns = ['hidx', 'vidx', 'hscore', 'vscore', 'weight']).reset_index(drop=True)
        mtx['hscore'] = mtx['hscore']*mtx['weight']
        mtx['vscore'] = mtx['vscore']*mtx['weight']
        mtx = mtx.groupby(['hidx', 'vidx'])[['hscore', 'vscore']].mean()
        mtx.reset_index(inplace=True)

        D = coo_matrix((mtx.vscore.values, (mtx.hidx.values, mtx.vidx.values)), shape=(table.itemnum, table.itemnum)).tocsr()
        Dt = D.transpose()

        prevd = np.ones(table.itemnum)/table.itemnum
        d = np.ones(table.itemnum)
        while norm(prevd-d)>threshold:
            prevd = d
            o = Dt.dot(1/d)+np.ones(d.shape[0])*epsilon*(np.sum(1/d))
            d = D.dot(1/o)+np.ones(o.shape[0])*epsilon*(np.sum(1/o))
        o = Dt.dot(1/d)

        if method=='summary':
            r = o/d
        elif method=='offence':
            r = o
        elif method=='defence':
            r = d
        else:
            raise ValueError('output should be one of summary, offence or defence.')
        if hasattr(self, "rating"):
            self.rating["rating"] = r
        else:
            self.rating = pd.DataFrame({
                "iidx": np.arange(table.itemnum, dtype=np.int),
                "rating": r})
        return self._showcase(table, True if method=='defence' else False)

class DifferenceRanker(UnsupervisedRanker):
    """This ranker targets at predicting score difference of games directly.
    The difference of ratings are proportional to the difference of score.
    """
    def rank(self, table):
        """Calculate the rank and rating with specified parameters.

        Parameters
        ----------
        table: Table
            The record table to be ranked, should be a Table object.
        
        Returns
        -------
        pandas.DataFrame, with column ['name', 'rating', 'rank']
        """
        mtx = pd.DataFrame(data={
            'hidx': pd.concat([table.table.hidx, table.table.vidx]),
            'vidx': pd.concat([table.table.vidx, table.table.hidx]),
            'hscore': pd.concat([table.table.hscore, table.table.vscore]),
            'vscore': pd.concat([table.table.vscore, table.table.hscore]),
            'weight': pd.concat([table.table.weight, table.table.weight])
        }, columns = ['hidx', 'vidx', 'hscore', 'vscore', 'weight']).reset_index(drop=True)
        mtx['score'] = mtx['hscore']-mtx['vscore']
        mtx['score'] = mtx['score']*mtx['weight']
        mtx = mtx.groupby(['hidx', 'vidx']).score.mean().reset_index()
        r = mtx.groupby('hidx').score.sum()/table.itemnum
        r = r.sort_index()

        if hasattr(self, "rating"):
            self.rating["rating"] = r.values
        else:
            self.rating = pd.DataFrame({
                "iidx": np.arange(table.itemnum, dtype=np.int),
                "rating": r})
        return self._showcase(table, False)

