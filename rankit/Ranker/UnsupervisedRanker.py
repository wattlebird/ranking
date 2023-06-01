from __future__ import division
from __future__ import absolute_import
import numpy as np
import pandas as pd
import scipy as sp
from rankit.Table import Table
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import lsqr
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
        rst['rank'] = rst.rating.rank(method='min', ascending=ascending).astype(int)
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
        dat = np.zeros(m*2, dtype=np.float32)
        col = np.zeros(m*2, dtype=int)
        row = np.zeros(m*2, dtype=int)
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
                "iidx": np.arange(n, dtype=int),
                "rating": rating})

        return self._showcase(table, False)

def colleyAgg(df, drawMargin, isInverted):
    rtn = np.sign(df.hscore - df.vscore) * df.weight
    if isInverted:
        rtn = -rtn
    rtn[np.abs(df.hscore - df.vscore) < drawMargin] = 0
    return np.sum(rtn)

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
        C = np.zeros((table.itemnum, table.itemnum))
        b = np.zeros(table.itemnum)
        for (i, j), c in table.table.groupby(['hidx', 'vidx'])['weight'].agg('count').items():
            C[i][j] -= c
        for inx, val in table.table.groupby('hidx').apply(colleyAgg, drawMargin, False).items():
            b[inx] += val
        for inx, val in table.table.groupby('vidx').apply(colleyAgg, drawMargin, True).items():
            b[inx] += val
        C = C + C.T
        np.fill_diagonal(C, -np.sum(C, axis=0)+2)
        b = b/2 + 1

        rating = sp.linalg.solve(C, b)
        if hasattr(self, "rating"):
            self.rating["rating"] = rating
        else:
            self.rating = pd.DataFrame({
                "iidx": np.arange(table.itemnum, dtype=int),
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
    def __init__(self, func=lambda x: x, epsilon=1e-4, threshold=1e-4):
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
        normalizedh = (table.table['hscore'] + 1) / (table.table['vscore'] + table.table['hscore'] + 2) * table.table['weight']
        normalizedv = table.table['weight'] - normalizedh
        gp = pd.DataFrame(data={
            'hidx': table.table['hidx'],
            'vidx': table.table['vidx'],
            'hscore': normalizedh,
            'vscore': normalizedv
        }).groupby(['hidx', 'vidx']).agg('mean').reset_index(drop=False)

        D = coo_matrix((
            func(np.concatenate((gp.hscore.values, gp.vscore.values))),
            (np.concatenate((gp.hidx.values, gp.vidx.values)),
            np.concatenate((gp.vidx.values, gp.hidx.values)))),
            shape=(table.itemnum, table.itemnum)
        ).tocsr()

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
                "iidx": np.arange(table.itemnum, dtype=int),
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
        gp = pd.DataFrame(data={
            'hidx': table.table['hidx'],
            'vidx': table.table['vidx'],
            'hscore': np.multiply(table.table.hscore, table.table.weight),
            'vscore': np.multiply(table.table.vscore, table.table.weight)
        }).groupby(['hidx', 'vidx']).agg('mean').reset_index(drop=False)

        D = coo_matrix((
            np.concatenate((gp.hscore.values, gp.vscore.values)),
            (np.concatenate((gp.hidx.values, gp.vidx.values)),
            np.concatenate((gp.vidx.values, gp.hidx.values)))),
            shape=(table.itemnum, table.itemnum)
        ).tocsr()
        r = np.ones(table.itemnum)/table.itemnum
        pr = np.ones(table.itemnum)
        while norm(pr-r)>threshold:
            pr = r
            vrestart = restart*np.ones(table.itemnum)/table.itemnum
            t = D.dot(r)
            t /= np.sum(t)
            r = (1-restart)*t+vrestart
        
        if hasattr(self, "rating"):
            self.rating["rating"] = r
        else:
            self.rating = pd.DataFrame({
                "iidx": np.arange(table.itemnum, dtype=int),
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
        gp = pd.DataFrame(data={
            'hidx': table.table['hidx'],
            'vidx': table.table['vidx'],
            'hscore': np.multiply(table.table.hscore, table.table.weight),
            'vscore': np.multiply(table.table.vscore, table.table.weight)
        }).groupby(['hidx', 'vidx']).agg('mean').reset_index(drop=False)

        D = coo_matrix((
            np.concatenate((gp.vscore.values, gp.hscore.values)),
            (np.concatenate((gp.hidx.values, gp.vidx.values)),
            np.concatenate((gp.vidx.values, gp.hidx.values)))),
            shape=(table.itemnum, table.itemnum)
        ).tocsr()
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
                "iidx": np.arange(table.itemnum, dtype=int),
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
        gp = pd.DataFrame(data={
            'hidx': table.table.hidx,
            'vidx': table.table.vidx,
            'score': (table.table.hscore - table.table.vscore) * table.table.weight
        }, columns = ['hidx', 'vidx', 'score']).groupby(['hidx', 'vidx']).agg('mean').reset_index(drop=False)

        D = coo_matrix((
            np.concatenate((gp.score.values, -gp.score.values)),
            (np.concatenate((gp.hidx.values, gp.vidx.values)),
            np.concatenate((gp.vidx.values, gp.hidx.values)))),
            shape=(table.itemnum, table.itemnum)
        ).tocsr()
        s = D.sum(axis=1).A1/table.itemnum

        if hasattr(self, "rating"):
            self.rating["rating"] = s
        else:
            self.rating = pd.DataFrame({
                "iidx": np.arange(table.itemnum, dtype=int),
                "rating": s})
        return self._showcase(table, False)

