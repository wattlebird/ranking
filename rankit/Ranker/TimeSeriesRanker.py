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
    def __init__(self, K=10, xi=400, baseline=1500, drawMargin=0):
        self.K = K
        self.xi = xi
        self.baseline = baseline
        self.drawMargin = drawMargin
        self.data = Table()
        self.indexScoreLut = []
        
    def setup(self, itemRatingLut = dict()):
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
        
    def update_single(self, host, visit, hscore, vscore, time="", hostavantage=0.0):
        self.data.update_single(host, visit, hscore, vscore, time=time, hostavantage=hostavantage)
        ih = self.data.itemlut[host]
        iv = self.data.itemlut[visit]
        if ih >= len(self.indexScoreLut):
            self.indexScoreLut.append(self.baseline)
        if iv >= len(self.indexScoreLut):
            self.indexScoreLut.append(self.baseline)
        ha = 0 if hostavantage<0 else hostavantage
        va = 0 if hostavantage>0 else -hostavantage
        rh = self.indexScoreLut[ih] + ha
        rv = self.indexScoreLut[iv] + va

        xi, K = self.xi, self.K
        s = 0.5 if abs(hscore - vscore) <= self.drawMargin else (1 if hscore > vscore else 0)
        phwin = 1/(1+10**((rv - rh)/xi))
        alpha = (abs(hscore-vscore)+3)**0.8/(7.5+0.0006*(rh - rv))
        delta = K*alpha*(s-phwin)
        self.indexScoreLut[ih] += delta
        self.indexScoreLut[iv] -= delta
        return (self.indexScoreLut[ih], self.indexScoreLut[iv])

    def update(self, table):
        self.data.update(table)
        self.indexScoreLut = self.indexScoreLut[:] + [self.baseline] * (self.data.itemnum - len(self.indexScoreLut))

        xi, K = self.xi, self.K
        for rec in table.iteritem():
            ih, iv, hscore, vscore, hostavantage = rec.indexHost, rec.indexVisit, rec.hscore, rec.vscore, rec.hostavantage
            ha = 0 if hostavantage<0 else hostavantage
            va = 0 if hostavantage>0 else -hostavantage
            rh = self.indexScoreLut[ih] + ha
            rv = self.indexScoreLut[iv] + va
            s = 0.5 if abs(hscore - vscore) <= self.drawMargin else (1 if hscore > vscore else 0)
            phwin = 1/(1+10**((rv - rh)/xi))
            alpha = (abs(hscore-vscore)+3)**0.8/(7.5+0.0006*(rh - rv))
            delta = K*alpha*(s-phwin)
            self.indexScoreLut[ih] += delta
            self.indexScoreLut[iv] -= delta

    def prob_win(self, host, visit):
        ih = self.data.itemlut[host]
        iv = self.data.itemlut[visit]
        rh = self.indexScoreLut[ih]
        rv = self.indexScoreLut[iv]
        return 1/(1+10**((rv-rh)/self.xi))

    def leaderboard(self, method="min"):
        rtn = pd.DataFrame({
            "name": self.data.indexlut,
            "rating": self.indexScoreLut
        }, columns=["name", "rating"])
        rtn['rank'] = rtn.rating.rank(method=method, ascending=False).astype(np.int32)
        return rtn.sort_values(by=['rating', 'name'], ascending=False).reset_index(drop=True)

class TrueSkillRanker(TimeSeriesRanker):
    def __init__(self, baseline=1500, rd=500, drawProbability=0.1, drawMargin=0, beta=1/2):
        self.miu = baseline / rd
        self.sigma = 1
        self.baseline = baseline
        self.rd = rd
        self.beta = beta
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
    
    def update_single(self, host, visit, hscore, vscore, time="", hostavantage=0.0):
        v, w, vt, wt, beta, drawMargin, realDrawMargin = TrueSkillRanker.v, TrueSkillRanker.w, TrueSkillRanker.vt, TrueSkillRanker.wt, self.beta, self.drawMargin, self.realDrawMargin
        self.data.update_single(host, visit, hscore, vscore, time=time)
        ih = self.data.itemlut[host]
        iv = self.data.itemlut[visit]
        if ih >= len(self.indexMiuLut) or iv >= len(self.indexMiuLut) :
            self.indexMiuLut = self.indexMiuLut[:] + [self.miu] * (self.data.itemnum - len(self.indexMiuLut))
            self.indexSigmaSqrLut = self.indexSigmaSqrLut[:] + [1] * (self.data.itemnum - len(self.indexSigmaSqrLut))
        ha = 0 if hostavantage<0 else hostavantage
        va = 0 if hostavantage>0 else -hostavantage
        mh = self.indexMiuLut[ih] + ha
        mv = self.indexMiuLut[iv] + va
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
        v, w, vt, wt, beta, drawMargin, realDrawMargin = TrueSkillRanker.v, TrueSkillRanker.w, TrueSkillRanker.vt, TrueSkillRanker.wt, self.beta, self.drawMargin, self.realDrawMargin
        self.data.update(table)
        if self.data.itemnum > len(self.indexMiuLut) :
            self.indexMiuLut = self.indexMiuLut[:] + [self.miu] * (self.data.itemnum - len(self.indexMiuLut))
            self.indexSigmaSqrLut = self.indexSigmaSqrLut[:] + [1] * (self.data.itemnum - len(self.indexSigmaSqrLut))
        
        for rec in table.iteritem():
            ih, iv, hscore, vscore, hostavantage = rec.indexHost, rec.indexVisit, rec.hscore, rec.vscore, rec.hostavantage
            ha = 0 if hostavantage<0 else hostavantage
            va = 0 if hostavantage>0 else -hostavantage
            mh = self.indexMiuLut[ih] + ha
            mv = self.indexMiuLut[iv] + va
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
        beta = self.beta
        ih = self.data.itemlut[host]
        iv = self.data.itemlut[visit]
        mh = self.indexMiuLut[ih]
        mv = self.indexMiuLut[iv]
        sh = self.indexSigmaSqrLut[ih]
        sv = self.indexSigmaSqrLut[iv]
        cs = sh + sv + 2*beta*beta
        return cdf((mh-mv)/cs**(1/2))

    def leaderboard(self, method="min"):
        rtn = pd.DataFrame({
            "name": self.data.indexlut,
            "rating": [self.rd * (i - 1.96*j**(1/2) - 3) + self.baseline for (i,j) in zip(self.indexMiuLut, self.indexSigmaSqrLut)]
        }, columns=["name", "rating"])
        rtn['rank'] = rtn.rating.rank(method=method, ascending=False).astype(np.int32)
        return rtn.sort_values(by=['rating', 'name'], ascending=False).reset_index(drop=True)

class GlickoRanker(TimeSeriesRanker):
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
        self.data.update_raw(dataFrame, weightcol='weight', timecol='time', hostavantagecol='hostavantage')
        if self.data.itemnum > len(self.miu) :
            self.miu = self.miu[:] + [0] * (self.data.itemnum - len(self.miu))
            self.phi = self.phi[:] + [self.rd / self.factor] * (self.data.itemnum - len(self.phi))
            self.sigma = self.sigma[:] + [self.votality] * (self.data.itemnum - len(self.sigma))
        
        mtx = pd.DataFrame(data={
            'host': pd.concat([dataFrame.host, dataFrame.visit]),
            'visit': pd.concat([dataFrame.visit, dataFrame.host]),
            'hscore': pd.concat([dataFrame.hscore, dataFrame.vscore]),
            'vscore': pd.concat([dataFrame.vscore, dataFrame.hscore]),
            'time': pd.concat([dataFrame.time, dataFrame.time]),
            'hostavantage': pd.concat([dataFrame.hostavantage, -dataFrame.hostavantage])
        }, columns = ['host', 'visit', 'hscore', 'vscore', 'time', 'hostavantage']).reset_index(drop=True).sort_values(by=['time', 'host'])

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
                ha = 0 if rec.hostavantage<0 else rec.hostavantage
                va = 0 if rec.hostavantage>0 else -rec.hostavantage
                vidx = self.data.itemlut[rec.visit]

                s = 0.5 if abs(rec.hscore - rec.vscore) <= drawMargin else (0 if rec.hscore < rec.vscore else 1)
                ex = E(self.miu[hidx] + ha, self.miu[vidx] + va, self.phi[vidx])
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

    def update(self, table):
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
        E = GlickoRanker.E
        hidx = self.data.itemlut[host]
        vidx = self.data.itemlut[visit]
        return E(self.miu[hidx], self.miu[vidx], math.sqrt(self.phi[hidx]**2 + self.phi[vidx]**2))

    def leaderboard(self, method="min"):
        rtn = pd.DataFrame({
            "name": self.data.indexlut,
            "rating": [self.factor * (i - 1.96*j) + self.baseline for (i,j) in zip(self.miu, self.phi)]
        }, columns=["name", "rating"])
        rtn['rank'] = rtn.rating.rank(method=method, ascending=False).astype(np.int32)
        return rtn.sort_values(by=['rating', 'name'], ascending=False).reset_index(drop=True)
