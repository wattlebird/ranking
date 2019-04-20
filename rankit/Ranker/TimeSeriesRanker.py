from __future__ import division
from __future__ import absolute_import
import numpy as np
import pandas as pd
from rankit.Table import Table
from scipy.stats import norm
import math
import warnings

pdf = norm.pdf
cdf = norm.cdf
ppf = norm.ppf
PI = math.pi

class TimeSeriesRanker(object):
    """Base class for all time series ranking algorithms."""

    def setup(self, *args, **kwargs):
        raise NotImplementedError("TimeSeriesRanker is a abstract class.")
    
    def update_single(self, *args, **kwargs):
        raise NotImplementedError("TimeSeriesRanker is a abstract class.")

    def update(self, table):
        for rec in table.iteritem():
            self.update_single(**rec)
        
    def prob_win(self, host, visit):
        raise NotImplementedError("TimeSeriesRanker is a abstract class.")
    
    def leaderboard(self):
        raise NotImplementedError("TimeSeriesRanker is a abstract class.")

class EloRanker(TimeSeriesRanker):
    """
    Elo Ranker is a traditional ranking algorithm adjusting player's rating by a series of gaming results.
    All players starts from 1500 first, and after each paired contest, two player's ranking will be updated in such a way that the sum of their ranking does not change.
    
    Parameters
    ----------
    K: amount of weight to be applied to each update.
    xi: somewhat related to "performance variance", the larger value assumes a more violent game performance and the ranking change will be more conservative.
    baseline: the initial ranking of each player.
    drawMargin: if the score difference is smaller or equal than drawMargin, this turn of game will be considered as draw. A draw will also effect player's rating.
    """
    def __init__(self, K=10, xi=400, baseline=1500, drawMargin=0):
        self.K = K
        self.xi = xi
        self.baseline = baseline
        self.drawMargin = drawMargin
        self.data = Table()
        self.indexScoreLut = []
        
    def setup(self, itemRatingLut = dict()):
        """
        Setup the initial state of EloRanker with existing Rating.
        This function is used in where prior information is given about competitiors, and one wants to continue ranking on that.
        
        Parameters
        ----------
        itemRatingLut: a dictionary working as player's name to rating look up table.
        """
        if len(itemRatingLut) != 0:
            # derive itemlut, indexlut and itemnum from itmScoreLut to setup self.data
            itemnum = len(itemRatingLut)
            itemlut = dict() # from item to index
            indexlut = [] # from index to item
            for i, itm in enumerate(itemRatingLut.items()):
                k, v = itm
                itemlut[i] = k
                indexlut.append(k)
                self.indexScoreLut.append(v)
            self.data.setup(itemlut, indexlut, itemnum)
            # TODO: should one infer baseline from itemscoreLut here?
        elif self.data.itemnum == 0:
            self.indexScoreLut = []
        
    def update_single(self, host, visit, hscore, vscore, time=""):
        """
        Update rating based on a single record.
        
        Parameters
        ----------
        host: name of host player
        visit: name of visit player
        hscore: score of host player
        vscore: score of visit player
        time: timestamp of the game. Should be numerical value. Default set to empty string.

        Return
        ------
        Tuple of (newHostRating, newVisitRating)
        """
        self.data.update_single(host, visit, hscore, vscore, time=time)
        ih = self.data.itemlut[host]
        iv = self.data.itemlut[visit]
        if ih >= len(self.indexScoreLut):
            self.indexScoreLut.append(self.baseline)
        if iv >= len(self.indexScoreLut):
            self.indexScoreLut.append(self.baseline)
        rh = self.indexScoreLut[ih]
        rv = self.indexScoreLut[iv]

        xi, K = self.xi, self.K
        s = 0.5 if abs(hscore - vscore) <= self.drawMargin else (1 if hscore > vscore else 0)
        phwin = 1/(1+10**((rv - rh)/xi))
        alpha = (abs(hscore-vscore)+3)**0.8/(7.5+0.0006*(rh - rv))
        delta = K*alpha*(s-phwin)
        self.indexScoreLut[ih] += delta
        self.indexScoreLut[iv] -= delta
        return (self.indexScoreLut[ih], self.indexScoreLut[iv])

    def update(self, table):
        """
        Update rating based on a table of record.
        
        Parameters
        ----------
        table: a Table object, consisting of new records that has never been previously feed to the ranker.
        """
        self.data.update(table)
        self.indexScoreLut = self.indexScoreLut[:] + [self.baseline] * (self.data.itemnum - len(self.indexScoreLut))

        xi, K = self.xi, self.K
        for rec in table.iteritem():
            ih, iv, hscore, vscore = rec.indexHost, rec.indexVisit, rec.hscore, rec.vscore
            rh = self.indexScoreLut[ih]
            rv = self.indexScoreLut[iv]
            s = 0.5 if abs(hscore - vscore) <= self.drawMargin else (1 if hscore > vscore else 0)
            phwin = 1/(1+10**((rv - rh)/xi))
            alpha = (abs(hscore-vscore)+3)**0.8/(7.5+0.0006*(rh - rv))
            delta = K*alpha*(s-phwin)
            self.indexScoreLut[ih] += delta
            self.indexScoreLut[iv] -= delta

    def prob_win(self, host, visit):
        """
        Probability of host player wins over visit player.

        Parameters
        ----------
        host: name of host player
        visit: name of visit player

        Return
        ------
        float: probability of winning.
        """
        ih = self.data.itemlut.get(host, None)
        iv = self.data.itemlut.get(visit, None)
        rh = self.indexScoreLut[ih] if ih is not None else self.baseline
        rv = self.indexScoreLut[iv] if iv is not None else self.baseline
        return 1/(1+10**((rv-rh)/self.xi))

    def leaderboard(self, method="min"):
        """
        Presenting current leaderboard.

        Parameters
        ----------
        method: method to process ranking value when rating is the same. See https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.rank.html for more.

        Return
        ------
        pd.DataFrame: with column "name", "rating" and "rank".
        """
        rtn = pd.DataFrame({
            "name": self.data.indexlut,
            "rating": self.indexScoreLut
        }, columns=["name", "rating"])
        rtn['rank'] = rtn.rating.rank(method=method, ascending=False).astype(np.int32)
        return rtn.sort_values(by=['rating', 'name'], ascending=False).reset_index(drop=True)

class TrueSkillRanker(TimeSeriesRanker):
    """
    Pairwise TrueSkill Ranker is subset of real TrueSkill ranker. See more: https://www.microsoft.com/en-us/research/project/trueskill-ranking-system/
    Unlike original TrueSkill ranker, this ranker only process pairwise gaming records.

    Parameters
    ----------
    baseline: the initial ranking value of new players. Default set to 1500.
    rd: rating deviation, the possible deviation of a player. Default set to 500.
    performanceRd: the possible deviation of each game. Default set to 250.
    drawProbability: the probability of draw. Default set to 0.1 and cannot be set to 0.
    darwMargin: if the score difference is smaller or equal than drawMargin, this turn of game will be considered as draw.
                Even if drawMargin is set to 0, drawProbability should never be set to 0.
    """
    def __init__(self, baseline=1500, rd=500, performanceRd=250, drawProbability=0.1, drawMargin=0):
        self.miu = baseline / rd
        self.sigma = 1
        self.baseline = baseline
        self.rd = rd
        self.performanceRd = performanceRd
        self.drawMargin = drawMargin
        self.drawProbability = drawProbability
        self.data = Table()
        self.indexMiuLut = []
        self.indexSigmaSqrLut = []

    @property
    def drawProbability(self):
        return self._draw_probability

    @drawProbability.setter
    def drawProbability(self, p):
        if p <= 0:
            warnings.warn("Probability of draw must be set above 0. Set to default value 0.1.")
            self._draw_probability = 0.1
        else:
            self._draw_probability = p

    @property
    def realDrawMargin(self):
        return math.sqrt(2) * self.beta * ppf((1+self.drawProbability)/2)

    @property
    def beta(self):
        return self.performanceRd/self.rd

    @staticmethod
    def v(t, a):
        return pdf(t-a)/cdf(t-a)

    @staticmethod
    def w(t, a):
        return TrueSkillRanker.v(t, a) * (TrueSkillRanker.v(t, a) + t - a)

    @staticmethod
    def vt(t, a):
        return (pdf(- a - t) - pdf(a - t))/(cdf(a - t) - cdf(- a - t))

    @staticmethod
    def wt(t, a):
        return TrueSkillRanker.vt(t, a) ** 2 + ((a-t)*pdf(a-t)+(a+t)*pdf(a+t))/(cdf(a - t) - cdf(- a - t))

    def setup(self, itemRatingLut = dict(), itemRDLut = dict()):
        """
        Setup the initial state of TrueSkill with existing rating and player deviation.
        This function is used in where prior information is given about competitiors, and one wants to continue ranking on that.
        
        Parameters
        ----------
        itemRatingLut: a dictionary working as player's name to rating look up table.
        itemRDLut: a dictionary working as player's name to deviation look up table.
        """
        player = set(itemRatingLut.keys())
        player.update(itemRDLut.keys())
        if len(player) != 0:
            # derive itemlut, indexlut and itemnum from itmScoreLut to setup self.data
            itemnum = len(player)
            itemlut = dict() # from item to index
            indexlut = [] # from index to item
            miu = []
            sigmasqr = []
            for i, itm in enumerate(player):
                itemlut[itm] = i
                indexlut.append(itm)
                miu.append((itemRatingLut.get(itm, self.baseline)) / self.rd)
                sigmasqr.append((itemRDLut.get(itm, self.rd) / self.rd)**2)
            self.data.setup(itemlut, indexlut, itemnum)

            self.indexMiuLut = miu
            self.indexSigmaSqrLut = sigmasqr
        elif self.data.itemnum == 0:
            self.indexMiuLut = []
            self.indexSigmaSqrLut = []
    
    def update_single(self, host, visit, hscore, vscore, time=""):
        """
        Update rating based on a single record.
        
        Parameters
        ----------
        host: name of host player
        visit: name of visit player
        hscore: score of host player
        vscore: score of visit player
        time: timestamp of the game. Should be numerical value. Default set to empty string.

        Return
        ------
        Tuple of (newHostRating, newVisitRating)
        """
        v, w, vt, wt, beta, drawMargin, realDrawMargin = TrueSkillRanker.v, TrueSkillRanker.w, TrueSkillRanker.vt, TrueSkillRanker.wt, self.beta, self.drawMargin, self.realDrawMargin
        self.data.update_single(host, visit, hscore, vscore, time=time)
        ih = self.data.itemlut[host]
        iv = self.data.itemlut[visit]
        if ih >= len(self.indexMiuLut) or iv >= len(self.indexMiuLut) :
            self.indexMiuLut = self.indexMiuLut[:] + [self.miu] * (self.data.itemnum - len(self.indexMiuLut))
            self.indexSigmaSqrLut = self.indexSigmaSqrLut[:] + [1] * (self.data.itemnum - len(self.indexSigmaSqrLut))
        mh = self.indexMiuLut[ih]
        mv = self.indexMiuLut[iv]
        sh = self.indexSigmaSqrLut[ih]
        sv = self.indexSigmaSqrLut[iv]
        cs = sh + sv + 2*beta*beta
        if abs(hscore - vscore) <= drawMargin:
            self.indexMiuLut[ih] += sh*vt(0, realDrawMargin/cs**(1/2))/cs**(1/2)
            self.indexMiuLut[iv] += sv*vt(0, realDrawMargin/cs**(1/2))/cs**(1/2)
            self.indexSigmaSqrLut[ih] *= (1 - sh/cs*wt(0, realDrawMargin/cs**(1/2)))
            self.indexSigmaSqrLut[iv] *= (1 - sv/cs*wt(0, realDrawMargin/cs**(1/2)))
        else:
            b = 1 if hscore > vscore + drawMargin else -1
            self.indexMiuLut[ih] += b * sh * v(b*(mh - mv)/cs**(1/2), realDrawMargin/cs**(1/2)) /cs**(1/2)
            self.indexMiuLut[iv] -= b * sv * v(b*(mv - mh)/cs**(1/2), realDrawMargin/cs**(1/2)) /cs**(1/2)
            self.indexSigmaSqrLut[ih] *= (1 - sh * w(b*(mh - mv)/cs**(1/2), realDrawMargin/cs**(1/2)) /cs)
            self.indexSigmaSqrLut[iv] *= (1 - sv * w(b*(mv - mh)/cs**(1/2), realDrawMargin/cs**(1/2)) /cs)

        return (self.rd * (self.indexMiuLut[ih] - 1.96 * self.indexSigmaSqrLut[ih]**(1/2)) + (self.baseline - 3 * self.rd), self.rd * (self.indexMiuLut[iv] - 1.96 * self.indexSigmaSqrLut[iv]**(1/2)) + (self.baseline - 3 * self.rd) )
        
    def update(self, table):
        """
        Update rating based on a table of record.
        
        Parameters
        ----------
        table: a Table object, consisting of new records that has never been previously feed to the ranker.
        """
        v, w, vt, wt, beta, drawMargin, realDrawMargin = TrueSkillRanker.v, TrueSkillRanker.w, TrueSkillRanker.vt, TrueSkillRanker.wt, self.beta, self.drawMargin, self.realDrawMargin
        self.data.update(table)
        if self.data.itemnum > len(self.indexMiuLut) :
            self.indexMiuLut = self.indexMiuLut[:] + [self.miu] * (self.data.itemnum - len(self.indexMiuLut))
            self.indexSigmaSqrLut = self.indexSigmaSqrLut[:] + [1] * (self.data.itemnum - len(self.indexSigmaSqrLut))
        
        for rec in table.iteritem():
            ih, iv, hscore, vscore = rec.indexHost, rec.indexVisit, rec.hscore, rec.vscore
            mh = self.indexMiuLut[ih]
            mv = self.indexMiuLut[iv]
            sh = self.indexSigmaSqrLut[ih]
            sv = self.indexSigmaSqrLut[iv]
            cs = sh + sv + 2*beta*beta
            if abs(hscore - vscore) <= drawMargin:
                self.indexMiuLut[ih] += sh*vt(0, realDrawMargin/cs**(1/2))/cs**(1/2)
                self.indexMiuLut[iv] += sv*vt(0, realDrawMargin/cs**(1/2))/cs**(1/2)
                self.indexSigmaSqrLut[ih] *= (1 - sh/cs*wt(0, realDrawMargin/cs**(1/2)))
                self.indexSigmaSqrLut[iv] *= (1 - sv/cs*wt(0, realDrawMargin/cs**(1/2)))
            else:
                b = 1 if hscore > vscore + drawMargin else -1
                self.indexMiuLut[ih] += b * sh * v(b*(mh - mv)/cs**(1/2), realDrawMargin/cs**(1/2)) /cs**(1/2)
                self.indexMiuLut[iv] -= b * sv * v(b*(mv - mh)/cs**(1/2), realDrawMargin/cs**(1/2)) /cs**(1/2)
                self.indexSigmaSqrLut[ih] *= (1 - sh * w(b*(mh - mv)/cs**(1/2), realDrawMargin/cs**(1/2)) /cs)
                self.indexSigmaSqrLut[iv] *= (1 - sv * w(b*(mv - mh)/cs**(1/2), realDrawMargin/cs**(1/2)) /cs)
    
    def prob_win(self, host, visit):
        """
        Probability of host player wins over visit player.

        Parameters
        ----------
        host: name of host player
        visit: name of visit player

        Return
        ------
        float: probability of winning.
        """
        beta = self.beta
        ih = self.data.itemlut.get(host, None)
        iv = self.data.itemlut.get(visit, None)
        mh = self.indexMiuLut[ih] if ih is not None else self.miu
        mv = self.indexMiuLut[iv] if iv is not None else self.miu
        sh = self.indexSigmaSqrLut[ih] if ih is not None else 1
        sv = self.indexSigmaSqrLut[iv] if iv is not None else 1
        cs = sh + sv + 2*beta*beta
        return cdf((mh-mv)/cs**(1/2))

    def leaderboard(self, method="min"):
        """
        Presenting current leaderboard.

        Parameters
        ----------
        method: method to process ranking value when rating is the same. See https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.rank.html for more.

        Return
        ------
        pd.DataFrame: with column "name", "rating" and "rank".
        """
        rtn = pd.DataFrame({
            "name": self.data.indexlut,
            "rating": [self.rd * (i - 1.96*j**(1/2) - 3) + self.baseline for (i,j) in zip(self.indexMiuLut, self.indexSigmaSqrLut)]
        }, columns=["name", "rating"])
        rtn['rank'] = rtn.rating.rank(method=method, ascending=False).astype(np.int32)
        return rtn.sort_values(by=['rating', 'name'], ascending=False).reset_index(drop=True)

class GlickoRanker(TimeSeriesRanker):
    """
    Glicko 2 ranker. See more: http://www.glicko.net/glicko.html
    Notice: different from previous rankers, Glicko algorithm involves a concept called "rating period". The update procedure is based on each rating period.
    In order to specify rating period, one have to state clearly the timestamp in record. Records in the same timestamp will be updated as a batch.
    If no timestamp is specified, the update algorithm will update the whole records in one batch.

    Parameters
    ----------
    baseline: the initial ranking value of new players. Default set to 1500.
    rd: rating deviation, the possible deviation of a player. Default set to 350.
    votality: this parameter is to measure the degree of expected fluctuation in a player's rating. Default set to 0.06.
    tau: constrains the change of votality over time. The more enormous changes involved in your game, the lower tau should be. Default set to 0.5.
    epsilon: parameter to control iteration. Default set to 1e-6.
    darwMargin: if the score difference is smaller or equal than drawMargin, this turn of game will be considered as draw. Default set to 0.
    """
    def __init__(self, baseline = 1500, rd = 350, votality = 0.06, tau = 0.5, epsilon = 0.000001, drawMargin = 0):
        self.baseline = baseline
        self.rd = rd
        self.votality = votality
        self.tau = tau
        self.epsilon = epsilon
        self.drawMargin = drawMargin
        self.data = Table()
        self.miu = [] # normalized rating
        self.phi = [] # normalized rd
        self.sigma = [] # votality
        
        self.factor = 400 / math.log(10)
        
    @staticmethod
    def g(phi):
        return 1/(1+3*phi**2/PI**2)**(1/2)
    
    @staticmethod
    def E(miu, miut, phit):
        g = GlickoRanker.g
        return 1/(1+math.exp(g(phit)*(miut-miu)))

    def volatility_iter(self, deltaSqr, phiSqr, sigma, votality):
        epsilon, tau = self.epsilon, self.tau
        f = lambda x: math.exp(x) * (deltaSqr - phiSqr - votality - math.exp(x)) / 2 / (phiSqr + votality + math.exp(x))**2 - (x - math.log(sigma**2)) / tau**2
        A = 2 * math.log(sigma)
        if deltaSqr > phiSqr + votality:
            B = math.log(deltaSqr - phiSqr - votality)
        else:
            k = 1
            while f(2 * math.log(sigma) - k * tau) < 0:
                k += 1
            B = 2 * math.log(sigma) - k * tau
        fA = f(A)
        fB = f(B)
        while abs(B-A) > epsilon:
            C = A + (A-B)*fA/(fB-fA)
            fC = f(C)
            if fC*fB<0:
                A = B
                fA = fB
            else:
                fA = fA/2
            B = C
            fB = fC
        return math.exp(A/2)

    def setup(self, itemRatingLut = dict(), itemRDLut = dict(), itemVolatilityLut = dict()):
        """
        Setup the initial state of Glick 2 with existing rating and player deviation.
        This function is used in where prior information is given about competitiors, and one wants to continue ranking on that.
        Notice: one does not have to provide full look up table on every player: if a player exist in one look up table but doesn't in another one, corresponding parameter will be set to default value.
        
        Parameters
        ----------
        itemRatingLut: a dictionary working as player's name to rating look up table.
        itemRDLut: a dictionary working as player's name to deviation look up table.
        itemVolatilityLut: a dictionary working as player's name to volatility look up table.
        """
        # TODO: By using this function, users have to know exactly the existing user rating, RD and volatility, which implies that source of information comes from saved data.
        # So there must be a model saving function.

        # Description: we assume that the user list is obtained by the union of key of all luts given.
        # That implies the provided Rating/RD/Volatility don't have to be of same length: the unfilled items will be given default value.
        player = set(itemRatingLut.keys())
        player.update(itemRDLut.keys())
        player.update(itemVolatilityLut.keys())
        if len(player) != 0:
            # derive itemlut, indexlut and itemnum from itmScoreLut to setup self.data
            itemnum = len(player)
            itemlut = dict() # from item to index
            indexlut = [] # from index to item
            miu = []
            phi = []
            sigma = []
            for i, itm in enumerate(player):
                itemlut[itm] = i
                indexlut.append(itm)
                miu.append((itemRatingLut.get(itm, self.baseline) - self.baseline) / self.factor)
                phi.append(itemRDLut.get(itm, self.rd) / self.factor)
                sigma.append(itemVolatilityLut.get(itm, self.votality))
            self.data.setup(itemlut, indexlut, itemnum)
            # Then setup self.miu, phi and sigma
            self.miu = miu
            self.phi = phi
            self.sigma = sigma
        elif self.data.itemnum == 0:
            self.miu = []
            self.phi = []
            self.sigma = []

    def update_single_batch(self, dataFrame):
        # check time info
        g, E, volatility_iter, drawMargin = GlickoRanker.g, GlickoRanker.E, self.volatility_iter, self.drawMargin
        self.data.update_raw(dataFrame, weightcol='weight', timecol='time')
        if self.data.itemnum > len(self.miu) :
            self.miu = self.miu[:] + [0] * (self.data.itemnum - len(self.miu))
            self.phi = self.phi[:] + [self.rd / self.factor] * (self.data.itemnum - len(self.phi))
            self.sigma = self.sigma[:] + [self.votality] * (self.data.itemnum - len(self.sigma))
        
        mtx = pd.DataFrame(data={
            'host': pd.concat([dataFrame.host, dataFrame.visit]),
            'visit': pd.concat([dataFrame.visit, dataFrame.host]),
            'hscore': pd.concat([dataFrame.hscore, dataFrame.vscore]),
            'vscore': pd.concat([dataFrame.vscore, dataFrame.hscore]),
            'time': pd.concat([dataFrame.time, dataFrame.time])
        }, columns = ['host', 'visit', 'hscore', 'vscore', 'time']).reset_index(drop=True).sort_values(by=['time', 'host'])

        gx = dict()
        players = set(mtx.host)
        for host in players:
            hidx = self.data.itemlut[host]
            gx[hidx] = g(self.phi[hidx])
        for recs in mtx.groupby(by='host'):
            host, results = recs
            hidx = self.data.itemlut[host]
            vt = []
            dt = []
            for rec in results.itertuples(index=False):
                vidx = self.data.itemlut[rec.visit]

                s = 0.5 if abs(rec.hscore - rec.vscore) <= drawMargin else (0 if rec.hscore < rec.vscore else 1)
                ex = E(self.miu[hidx], self.miu[vidx], self.phi[vidx])
                vt.append(gx[vidx]**2 * ex * (1 - ex))
                dt.append(gx[vidx] * (s - ex))
            v = 1/sum(vt)
            delta = v*sum(dt)
            sigma_ = volatility_iter(delta**2, self.phi[hidx]**2, self.sigma[hidx], self.votality)
            phiSqr_ = (self.phi[hidx]**2 + sigma_**2)
            self.phi[hidx] = 1/(1/phiSqr_ + 1/v)**(1/2)
            self.miu[hidx] += self.phi[hidx]**2 * sum(dt)
        
        # for players not attended, their RD will increase.
        for kv in self.data.itemlut.items():
            player, idx = kv
            if player not in players:
                self.phi[idx] = math.sqrt(self.phi[idx]**2 + self.sigma[idx]**2)
        return

    def update(self, table):
        """
        Update rating based on a table of record.
        
        Parameters
        ----------
        table: a Table object, consisting of new records that has never been previously feed to the ranker.
        """
        # Check time info
        if self.data.table.shape[0] > 0 and self.data.table.time.iloc[-1] is not None:
            if table.table.time.iloc[0] is None:
                warnings.warn("The table to be updated misses time information. In that case, the whole table record will be updated in a single rating period.")
            elif table.table.time.iloc[0] <= self.data.table.time.iloc[-1]:
                warnings.warn("The table to be updated is recorded in a time before or equal to existing period. We could not update players' info in a delayed manner. The time sequence will be discarded.")

        for grp in table.table.groupby(by='time'):
            cur_time, gp = grp
            self.update_single_batch(gp)

    def update_single(self, *args, **kwargs):
        raise Exception("Update single is not allowed in Glicko 2 ranking algorithm. You must specify the rating period explicitly in time column wrapped in Table.")

    def prob_win(self, host, visit):
        """
        Probability of host player wins over visit player.

        Parameters
        ----------
        host: name of host player
        visit: name of visit player

        Return
        ------
        float: probability of winning.
        """
        E = GlickoRanker.E
        hidx = self.data.itemlut.get(host, None)
        vidx = self.data.itemlut.get(visit, None)
        hmiu = self.miu[hidx] if hidx is not None else 0
        vmiu = self.miu[vidx] if vidx is not None else 0
        hphi = self.phi[hidx] if hidx is not None else self.rd / self.factor
        vphi = self.phi[vidx] if vidx is not None else self.rd / self.factor
        return E(
            hmiu, 
            vmiu, 
            math.sqrt(hphi**2 + vphi**2)
        )

    def leaderboard(self, method="min"):
        """
        Presenting current leaderboard.

        Parameters
        ----------
        method: method to process ranking value when rating is the same. See https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.Series.rank.html for more.

        Return
        ------
        pd.DataFrame: with column "name", "rating" and "rank".
        """
        rtn = pd.DataFrame({
            "name": self.data.indexlut,
            "rating": [self.factor * (i - 1.96*j) + self.baseline for (i,j) in zip(self.miu, self.phi)]
        }, columns=["name", "rating"])
        rtn['rank'] = rtn.rating.rank(method=method, ascending=False).astype(np.int32)
        return rtn.sort_values(by=['rating', 'name'], ascending=False).reset_index(drop=True)
