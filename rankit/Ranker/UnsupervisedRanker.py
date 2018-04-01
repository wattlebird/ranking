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
    """Massey ranking system proposed by Kenneth Massey: Statistical models applied to the rating of sports teams. 
    Bachelor's thesis, Bluefield College, 1997.
    Core idea: The competition score difference is the rating difference, so one can solve a linear equation by minimize least square error.

    Parameters
    ----------
    table:
        Table object containing pairs of teams' playing scores, with optional weights.
    method: {'average', 'min', 'max', 'first', 'dense'}, default 'min'
        The method to calculate rank when draws happens in calculated rating.
        Same parameter described in pandas.Series.rank: https://pandas.pydata.org/pandas-docs/stable/generated/pandas.Series.rank.html
    """
    def __init__(self, table, method='min'):
        return super(MasseyRanker, self).__init__(table, method)
        

    def rank(self, tiethreshold = 0.0, ascending=False):
        """Calculate the rank and rating with specified parameters.

        Parameters
        ----------
        tiethreshold: [0, +Inf), default 0.
            When absolute difference between two teams are smaller than tiethreshold, this competition is considered as a tie.
        ascending: bool. default False.
            If set to True, the smaller the rating, the smaller the rank. For Massey rank it should be False.
        
        Returns
        -------
        pandas.DataFrame, with column ['name', 'rating', 'rank']
        """
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

    def score_diff(self, host, visit):
        """Calculate the score difference given host player and visit player.

        Parameters
        ----------
        host:
            It can be a single object or a list of object indicating host(s).
        visit: 
            Same as host, should have same length as host.

        Returns
        -------
        float or list of float
        """
        
        if not hasattr(self, 'rating'):
            return RuntimeError('No rating information calculated. Please invoke rank first.')
        
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
                rst.append(r[h]-r[v])
            return rst
        else:
            h = self.data.itemlut[host]
            v = self.data.itemlut[visit]
            return (r[h]-r[v])

class ColleyRanker(UnsupervisedRanker):
    """Colley ranking system proposed by Wesley Colley: 
    Colley's bias free college football ranking method: The colley matrix explained, 2002.
    http://www.colleyrankings.com
    Core idea: All team's rating starts from 0.5, and with evolvement of games, the rating of each player deviates from 0.5
    according to probability of win. However, the average rating of all teams remains 0.5.

    Parameters
    ----------
    table:
        Table object containing pairs of teams' playing scores, with optional weights.
    method: {'average', 'min', 'max', 'first', 'dense'}, default 'min'
        The method to calculate rank when draws happens in calculated rating.
        Same parameter described in pandas.Series.rank: https://pandas.pydata.org/pandas-docs/stable/generated/pandas.Series.rank.html
    """
    def __init__(self, table, method='min'):
        return super(ColleyRanker, self).__init__(table, method)

    def rank(self, tiethreshold = 0.0, ascending=False):
        """Calculate the rank and rating with specified parameters.

        Parameters
        ----------
        tiethreshold: [0, +Inf), default 0.
            When absolute difference between two teams are smaller than tiethreshold, this competition is considered as a tie.
        ascending: bool. default False.
            If set to True, the smaller the rating, the smaller the rank. For Colley rank it should be False.
        
        Returns
        -------
        pandas.DataFrame, with column ['name', 'rating', 'rank']
        """
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
    """Keener ranking system proposed by James Keener:
    The Perron-Frobenius theorem and the ranking of football teams, SIAM Review, 35(1):80-93, 1993
    The core idea are: 1. rating is proportional to real strength; 2. real strength is measured relatively by competitors' strength.

    Parameters
    ----------
    table:
        Table object containing pairs of teams' playing scores, with optional weights.
    method: {'average', 'min', 'max', 'first', 'dense'}, default 'min'
        The method to calculate rank when draws happens in calculated rating.
        Same parameter described in pandas.Series.rank: https://pandas.pydata.org/pandas-docs/stable/generated/pandas.Series.rank.html
    """
    def __init__(self, table, method='min'):
        return super(KeenerRanker, self).__init__(table, method)
    
    def rank(self, func=None, epsilon=1e-4, threshold=1e-4, ascending=False):
        """Calculate the rank and rating with specified parameters.

        Parameters
        ----------
        func: default None.
            If set, the score difference should be transformed by the function first then used for rating calculation.
        epsilon: [0, +Inf) default 1e-4
            The small value that applies an interference to game result that force each team had at least one game with each other.
        threshold: (0, +Inf), default 1e-4
            The threshold that controls when the algorithm will converge.
        ascending: bool. default False.
            If set to True, the smaller the rating, the smaller the rank. For Keener rank it should be False.
        
        Returns
        -------
        pandas.DataFrame, with column ['name', 'rating', 'rank']
        """
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

    def score_diff(self, host, visit):
        """Calculate the score difference given host player and visit player.

        Parameters
        ----------
        host:
            It can be a single object or a list of object indicating host(s).
        visit: 
            Same as host, should have same length as host.

        Returns
        -------
        float or list of float
        """
        
        if not hasattr(self, 'rating'):
            return RuntimeError('No rating information calculated. Please invoke rank first.')
        
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
                rst.append(r[h]-r[v])
            return rst
        else:
            h = self.data.itemlut[host]
            v = self.data.itemlut[visit]
            return (r[h]-r[v])

class MarkovRanker(UnsupervisedRanker):
    """Markov ranking is actually PageRank.
    The core idea is voting: in each game, each team will vote to each other by the number of scores they lost.
    If there are multiple games for a certain pair of player, their scores will be grouped and averaged.

    Parameters
    ----------
    table:
        Table object containing pairs of teams' playing scores, with optional weights.
    method: {'average', 'min', 'max', 'first', 'dense'}, default 'min'
        The method to calculate rank when draws happens in calculated rating.
        Same parameter described in pandas.Series.rank: https://pandas.pydata.org/pandas-docs/stable/generated/pandas.Series.rank.html
    """
    def __init__(self, table, method='min'):
        return super(MarkovRanker, self).__init__(table, method)
    
    def rank(self, restart=0.3, threshold=1e-4, ascending=False):
        """Calculate the rank and rating with specified parameters.

        Parameters
        ----------
        restart: [0, 1], default 0.3.
            Random walk with restart: in order to avoid black hole in random walk graph.
        threshold: (0, +Inf), default 1e-4
            The threshold that controls when the algorithm will converge.
        ascending: bool. default False.
            If set to True, the smaller the rating, the smaller the rank. For Markov rank it should be False.
        
        Returns
        -------
        pandas.DataFrame, with column ['name', 'rating', 'rank']
        """
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
    """The Offence-defence rank tries to assign an offence rating and a defence rating to each team.
    By saying "offence rating", we assume that a team has a high offence rating when it gained a lot of points 
    from a team good in defence. Vise versa. The offence rating of a team is associated with defence rating of each
    competitor in a non-linear way. The defence rating of a team is also non-linearly related to each competitors' 
    offence rating.

    Parameters
    ----------
    table:
        Table object containing pairs of teams' playing scores, with optional weights.
    method: {'average', 'min', 'max', 'first', 'dense'}, default 'min'
        The method to calculate rank when draws happens in calculated rating.
        Same parameter described in pandas.Series.rank: https://pandas.pydata.org/pandas-docs/stable/generated/pandas.Series.rank.html
    """
    def __init__(self, table, method='min'):
        return super(ODRanker, self).__init__(table, method)
    
    def rank(self, output='summary', epsilon=1e-4, threshold=1e-4, ascending=False):
        """Calculate the rank and rating with specified parameters.

        Parameters
        ----------
        output: {'summary', 'offence', 'defence'}, default 'summary'.
            The rating to be returned. 'summary' is offence/defence.
        epsilon: [0, +Inf) default 1e-4
            The small value that forces a convergence.
        threshold: (0, +Inf), default 1e-4
            The threshold that controls when the algorithm will converge.
        ascending: bool. default False.
            If set to True, the smaller the rating, the smaller the rank. For rating 'summary' and 'offence' it should be set to False.
            For rating 'defence', it should be set to True.
        
        Returns
        -------
        pandas.DataFrame, with column ['name', 'rating', 'rank']
        """
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
            raise ValueError('output should be one of summary, offence or defence.')
        if hasattr(self, "rating"):
            self.rating["rating"] = r
        else:
            self.rating = pd.DataFrame({
                "iidx": np.arange(self.data.itemnum, dtype=np.int),
                "rating": r})
        return self._showcase(ascending)

class DifferenceRanker(UnsupervisedRanker):
    """This ranker targets at predicting score difference of games directly.
    The difference of ratings are proportional to the difference of score.

    Parameters
    ----------
    table:
        Table object containing pairs of teams' playing scores, with optional weights.
    method: {'average', 'min', 'max', 'first', 'dense'}, default 'min'
        The method to calculate rank when draws happens in calculated rating.
        Same parameter described in pandas.Series.rank: https://pandas.pydata.org/pandas-docs/stable/generated/pandas.Series.rank.html
    """
    def __init__(self, table, method='min'):
        return super(DifferenceRanker, self).__init__(table, method)

    def rank(self, ascending=False):
        """Calculate the rank and rating with specified parameters.

        Parameters
        ----------
        ascending: bool. default False.
            If set to True, the smaller the rating, the smaller the rank. For difference rank it is set to False by default.
        
        Returns
        -------
        pandas.DataFrame, with column ['name', 'rating', 'rank']
        """
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
    
    def score_diff(self, host, visit):
        """Calculate the score difference given host player and visit player.

        Parameters
        ----------
        host:
            It can be a single object or a list of object indicating host(s).
        visit: 
            Same as host, should have same length as host.

        Returns
        -------
        float or list of float
        """
        
        if not hasattr(self, 'rating'):
            return RuntimeError('No rating information calculated. Please invoke rank first.')
        
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
                rst.append(r[h]-r[v])
            return rst
        else:
            h = self.data.itemlut[host]
            v = self.data.itemlut[visit]
            return (r[h]-r[v])

class EloRanker(UnsupervisedRanker):
    """Elo rank is proposed by Arpad Elo to rank international chess players. It has been gone through many
    adaptations for various competitions.

    Parameters
    ----------
    table:
        Table object containing pairs of teams' playing scores, with compulsory time column and optional hostavantage.
    method: {'average', 'min', 'max', 'first', 'dense'}, default 'min'
        The method to calculate rank when draws happens in calculated rating.
        Same parameter described in pandas.Series.rank: https://pandas.pydata.org/pandas-docs/stable/generated/pandas.Series.rank.html
    """
    def __init__(self, table, method='min'):
        if not table.table.columns.contains('time'):
            raise ValueError('The passed in table has no time information provided.')
        return super(EloRanker, self).__init__(table, method)

    def rank(self, K = 10, baseline = 0, xi=400, ascending=False):
        """Calculate the rank and rating with specified parameters.

        Parameters
        ----------
        K: (0, +Inf), default 10.
            K-factor determines how quickly the rating reacts to new game results.
        baseline: (-Inf, +Inf), default 0.
            The average performance of the match is baseline. As time passes by, the teams' performance deviates from baseline.
        xi: (0, +Inf), default 400.
            If one player has xi more rating points than its competitor, it is expected that this player a chance of winning 10 times of 
            the chance it will lose.
        ascending: bool. default False.
            If set to True, the smaller the rating, the smaller the rank. For Elo rank it is set to False by default.
        
        Returns
        -------
        pandas.DataFrame, with column ['name', 'rating', 'rank']
        """
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
        """Predict the probability of winning based on current elo rating.
        The ranker should have ranked a table first before invoking this function.

        Parameters
        ----------
        host:
            It can be a single object or a list of object indicating host(s).
        visit: 
            Same as host, should have same length as host.

        Returns
        -------
        probability of winning (scalar or list)
        """
        if not hasattr(self, 'rating'):
            return RuntimeError('No rating information calculated. Please invoke rank first.')
        
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
        """Update elo ranking given new table object. The new table should have time column no smaller than existing table.
        This method is equivalent to calling rank on a concatenated Table of original table and new table, without the need to
        construct a new Table object and a new Ranker.
        Calling this function will also update table object contained in the ranker.

        Parameters
        ----------
        newtable:
            Table object, with time no smaller than existing table.

        Returns
        ------
        pandas.DataFrame, with column ['name', 'rating', 'rank']
        """
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
