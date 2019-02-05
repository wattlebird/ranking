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
    def __init__(self, K=10, xi=400, baseline=1500):
        self.K = K
        self.xi = xi
        self.baseline = baseline
        self.data = Table()
        self.indexScoreLut = []
        
    def setup(self, itemScoreLut = dict()):
        if len(itmScoreLut) != 0:
            # derive itemlut, indexlut and itemnum from itmScoreLut to setup self.data
            itemnum = len(itemScoreLut)
            itemlut = dict() # from item to index
            indexlut = [] # from index to item
            for i, itm in enumerate(itemScoreLut.items()):
                k, v = itm
                itemlut[i] = k
                indexlut.append(k)
                self.indexScoreLut.append(v)
            self.data.setup(itemlut, indexlut, itemnum)
            # TODO: should one infer baseline from itemscoreLut here?
        
    def update_single(self, host, visit, hscore, vscore, time=None, weight=1.0, hostavantage=0.0):
        self.data.update_single(host, visit, hscore, vscore, time=time, weight=weight, hostavantage=hostavantage)
        ih = self.data.itemlut[host]
        iv = self.data.itemlut[visit]
        if ih >= len(self.indexScoreLut):
            self.indexScoreLut.append(self.baseline)
        if iv >= len(self.indexScoreLut):
            self.indexScoreLut.append(self.baseline)
        rh = self.indexScoreLut[ih]
        rv = self.indexScoreLut[iv]

        xi, K = self.xi, self.K
        s = 0.5 if hscore == vscore else (1 if hscore > vscore else 0)
        phwin = 1/(1+10**((rv - rh)/xi))
        alpha = (abs(hscore-vscore)+3)**0.8/(7.5+0.0006*(rh - rv))
        delta = K*weight*alpha*(s-phwin)
        self.indexScoreLut[ih] += delta
        self.indexScoreLut[iv] -= delta
        return (self.indexScoreLut[ih], self.indexScoreLut[iv])

    def update(self, table):
        self.data.update(table)
        self.indexScoreLut = self.indexScoreLut[:] + [self.baseline] * (self.data.itemnum - len(self.indexScoreLut))

        xi, K = self.xi, self.K
        for rec in table.iteritem():
            ih, iv, hscore, vscore, weight = rec.indexHost, rec.indexVisit, rec.hscore, rec.vscore, rec.weight
            rh = self.indexScoreLut[ih]
            rv = self.indexScoreLut[iv]
            s = 0.5 if hscore == vscore else (1 if hscore > vscore else 0)
            phwin = 1/(1+10**((rv - rh)/xi))
            alpha = (abs(hscore-vscore)+3)**0.8/(7.5+0.0006*(rh - rv))
            delta = K*weight*alpha*(s-phwin)
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
    def __init__(self, range=50, drawMargin=0.01, confidenceInterval=3, beta=1/2):
        self.miu = confidenceInterval
        self.sigma = 1
        self.beta = 1
        self.drawMargin = drawMargin if drawMargin != 0 else 0.01
        if drawMargin == 0:
            warnings.warn("drawMargin shouldn't be set to 0. 0.01 is set alternatively.")
        self.range = range
        self.data = Table()
        self.indexMiuLut = []
        self.indexSigmaSqrLut = []

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
        return TrueSkillRanker.vt(t, a) ** 2 + ((a-t)*pdf(a-t)-(a+t)*pdf(a+t))/(cdf(a - t) - cdf(- a - t))

    def setup(self, itemMiuLut = dict(), itemSigmaSqrLut = dict()):
        if len(itemMiuLut) != len(itemSigmaSqrLut):
            raise Exception("Number of items in miu lut and sigmaSqr lut are not equal.")
        if len(itemMiuLut) != 0:
            # derive itemlut, indexlut and itemnum from itmScoreLut to setup self.data
            itemnum = len(itemMiuLut)
            itemlut = dict() # from item to index
            indexlut = [] # from index to item
            for i, itm in enumerate(itemMiuLut.items()):
                k, v = itm
                itemlut[i] = k
                indexlut.append(k)
            self.data.setup(itemlut, indexlut, itemnum)

            self.indexMiuLut = [itm for itm in itemMiuLut.values()]
            self.indexSigmaSqrLut = [itm for itm in itemSigmaSqrLut.values()]
    
    def update_single(self, host, visit, hscore, vscore, time=None, weight=1.0, hostavantage=0.0):
        v, w, vt, wt, beta, drawMargin = TrueSkillRanker.v, TrueSkillRanker.w, TrueSkillRanker.vt, TrueSkillRanker.wt, self.beta, self.drawMargin
        self.data.update_single(host, visit, hscore, vscore, time, weight)
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
            self.indexMiuLut[ih] = mh + sh*vt(0, drawMargin/cs**(1/2))/cs**(1/2)
            self.indexMiuLut[iv] = mv + sv*vt(0, drawMargin/cs**(1/2))/cs**(1/2)
            self.indexSigmaSqrLut[ih] = sh*(1 - sh/cs*wt(0, drawMargin/cs**(1/2)))
            self.indexSigmaSqrLut[iv] = sh*(1 - sv/cs*wt(0, drawMargin/cs**(1/2)))
        else:
            b = 1 if hscore > vscore + drawMargin else -1
            self.indexMiuLut[ih] += b * sh * v(b*(mh - mv)/cs**(1/2), drawMargin/cs**(1/2)) /cs**(1/2)
            self.indexMiuLut[iv] += b * sv * v(b*(mv - mh)/cs**(1/2), drawMargin/cs**(1/2)) /cs**(1/2)
            self.indexSigmaSqrLut[ih] *= (1 - sh * w(b*(mh - mv)/cs**(1/2), drawMargin/cs**(1/2)))
            self.indexSigmaSqrLut[iv] *= (1 - sv * w(b*(mv - mh)/cs**(1/2), drawMargin/cs**(1/2)))

        r = self.range / self.miu / 2
        return r * (self.indexMiuLut[ih] - 3 * self.indexSigmaSqrLut[ih]**(1/2)), self.indexMiuLut[iv] - 3 * self.indexSigmaSqrLut[iv]**(1/2)
        
    def update(self, table):
        v, w, vt, wt, beta, drawMargin = TrueSkillRanker.v, TrueSkillRanker.w, TrueSkillRanker.vt, TrueSkillRanker.wt, self.beta, self.drawMargin
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
                self.indexMiuLut[ih] = mh + sh*vt(0, drawMargin/cs**(1/2))/cs**(1/2)
                self.indexMiuLut[iv] = mv + sv*vt(0, drawMargin/cs**(1/2))/cs**(1/2)
                self.indexSigmaSqrLut[ih] = sh*(1 - sh/cs*wt(0, drawMargin/cs**(1/2)))
                self.indexSigmaSqrLut[iv] = sh*(1 - sv/cs*wt(0, drawMargin/cs**(1/2)))
            else:
                b = 1 if hscore > vscore + drawMargin else -1
                self.indexMiuLut[ih] += b * sh * v(b*(mh - mv)/cs**(1/2), drawMargin/cs**(1/2)) /cs**(1/2)
                self.indexMiuLut[iv] += b * sv * v(b*(mv - mh)/cs**(1/2), drawMargin/cs**(1/2)) /cs**(1/2)
                self.indexSigmaSqrLut[ih] *= (1 - sh * w(b*(mh - mv)/cs**(1/2), drawMargin/cs**(1/2)))
                self.indexSigmaSqrLut[iv] *= (1 - sv * w(b*(mv - mh)/cs**(1/2), drawMargin/cs**(1/2)))
    
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
        r = self.range / self.miu / 2
        rtn = pd.DataFrame({
            "name": self.data.indexlut,
            "rating": [r * (i - 3*j**(1/2)) for (i,j) in zip(self.indexMiuLut, self.indexSigmaSqrLut)]
        }, columns=["name", "rating"])
        rtn['rank'] = rtn.rating.rank(method=method, ascending=False).astype(np.int32)
        return rtn.sort_values(by=['rating', 'name'], ascending=False).reset_index(drop=True)

class GlickoRanker(TimeSeriesRanker):
    def __init__(self, *, baseline = 1500, rd = 350, votality = 0.06, tau = 0.5, epsilon = 0.000001):
        self.baseline = rating
        self.rd = rd
        self.votality = votality
        self.tau = tau
        self.epsilon = epsilon
        self.data = Table()
        self.miu = [] # normalized rating
        self.phi = [] # normalized rd
        self.sigma = [] # votality
        # stateful variables, valid in each rating period
        self.v = dict()
        self.delta = dict()
        
    @staticmethod
    def g(phi):
        return 1/(1+3*phi**2/PI**2)**(1/2)
    
    @staticmethod
    def E(miu, miut, phit):
        return 1/(1+math.exp(g(phit)*(miut-miu)))

    @staticmethod
    def volatility_iter(deltaSqr, phiSqr, sigma, votality):
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
            C = A + (A-B)*fA(fB-fA)
            fC = f(C)
            if fC*fB<0:
                A = B
                fA = fB
            else:
                fA = fA/2
            B = C
            fB = fC
        return math.exp(A/2)

    def setup(self, itemRatingLut = dict(), itemPhiLut = dict(), itemSigmaLut = dict()):
        # TODO: By using this function, users have to know exactly the existing user rating, RD and volatility, which implies that source of information comes from saved data.
        # So there must be a model saving function.
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
        # Then setup self.miu, phi and sigma
        self.miu = []*itemnum
        self.phi = []*itemnum
        self.sigma = []*itemnum
        for itm in itemRatingLut.items():
            k, rating = itm
            i = itemlut[k]
            self.miu[i] = rating
            self.phi[i] = itemPhiLut[k]
            self.sigma[i] = itemSigmaLut[k]

    def update(self, table):
        # check time info
        if table.table.time.iloc[0] is None:
            warnings.warn("The table to be updated misses time information. In that case, the whole table record will be updated in a single rating period.")
        else if self.data.table.shape[0] > 0 and table.table.time.iloc[0] <= self.data.table.time.iloc[-1]:
            warnings.warn("The table to be updated is recorded in a time before or equal to existing period. We could not update players' info in a delayed manner. The time sequence will be discarded.")
        g, E, volatility_iter, drawMargin = GlickoRanker.g, GlickoRanker.E, GlickoRanker.volatility_iter, self.drawMargin
        self.data.update(table)
        if self.data.itemnum > len(self.miu) :
            self.miu = self.miu[:] + [(self.baseline - 1500) / 173.7178] * (self.data.itemnum - len(self.miu))
            self.phi = self.phi[:] + [self.rd / 173.7178] * (self.data.itemnum - len(self.phi))
            self.sigma = self.sigma[:] + [self.tau] * (self.data.itemnum - len(self.sigma))
        
        mtx = pd.DataFrame(data={
            'hidx': pd.concat([table.table.hidx, table.table.vidx]),
            'vidx': pd.concat([table.table.vidx, table.table.hidx]),
            'hscore': pd.concat([table.table.hscore, table.table.vscore]),
            'vscore': pd.concat([table.table.vscore, table.table.hscore]),
            'weight': pd.concat([table.table.weight, table.table.weight]),
            'time': pd.concat([table.table.time, table.table.time])
        }, columns = ['hidx', 'vidx', 'hscore', 'vscore', 'weight', 'time']).reset_index(drop=True).sort_values(by=['time', 'hidx'])
        
        for grp in mtx.groupby(by='time'):
            cur_time, gp = grp
            gx = dict()
            ex = dict()
            vx = dict()
            for i in gp.hidx.unique():
                gx[i] = g(self.phi[i])
            for recs in gp.groupby(by='hidx'):
                player, results = recs
                vt = []
                dt = []
                for rec in results.itertuples(index=False):
                    s = 0.5 if abs(rec.hscore - rec.vscore) <= drawMargin else (0 if rec.hscore < rec.vscore else 1)
                    ex[(rec.hidx, rec.vidx)] = E(self.miu[rec.hidx], self.miu[rec.vidx], self.phi[rec.vidx])
                    vt.append(gx[rec.vidx]**2 * ex[(rec.hidx, rec.vidx)] * (1 - ex[(rec.hidx, rec.vidx)]))
                    dt.append(gx[rec.vidx] * (s - ex[(rec.hidx, rec.vidx)]))
                v = 1/vt.sum()
                delta = v*dt.sum()
                sigma_ = volatility_iter(delta**2, self.phi[player]**2, self.sigma[player], self.votality)
                phiSqr_ = (self.phi[player]**2 + sigma_**2)
                self.phi[player] = 1/(1/phiSqr_ + 1/v)**(1/2)
                self.miu[player] += self.phi[player]**2 * dt.sum()

                