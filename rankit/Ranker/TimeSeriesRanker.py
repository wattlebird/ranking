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
    def __init__(self, K=10, xi=400, baseline=1500, drawMargin=0):
        self.K = K
        self.xi = xi
        self.baseline = baseline
        self.drawMargin = drawMargin
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
        s = 0.5 if abs(hscore - vscore) <= self.drawMargin else (1 if hscore > vscore else 0)
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
            s = 0.5 if abs(hscore - vscore) <= self.drawMargin else (1 if hscore > vscore else 0)
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
            "rating": [r * (i - 1.96*j**(1/2)) for (i,j) in zip(self.indexMiuLut, self.indexSigmaSqrLut)]
        }, columns=["name", "rating"])
        rtn['rank'] = rtn.rating.rank(method=method, ascending=False).astype(np.int32)
        return rtn.sort_values(by=['rating', 'name'], ascending=False).reset_index(drop=True)

class GlickoRanker(TimeSeriesRanker):
    def __init__(self, *, baseline = 1500, rd = 350, votality = 0.06, tau = 0.5, epsilon = 0.000001, drawMargin = 0):
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

    def update_single_batch(self, dataFrame):
        # check time info
        g, E, volatility_iter, drawMargin = GlickoRanker.g, GlickoRanker.E, self.volatility_iter, self.drawMargin
        self.data.update_raw(dataFrame, weightcol='weight', timecol='time', hostavantagecol='hostavantage')
        if self.data.itemnum > len(self.miu) :
            self.miu = self.miu[:] + [0] * (self.data.itemnum - len(self.miu))
            self.phi = self.phi[:] + [self.rd / self.factor] * (self.data.itemnum - len(self.phi))
            self.sigma = self.sigma[:] + [self.tau] * (self.data.itemnum - len(self.sigma))
        
        mtx = pd.DataFrame(data={
            'host': pd.concat([dataFrame.host, dataFrame.visit]),
            'visit': pd.concat([dataFrame.visit, dataFrame.host]),
            'hscore': pd.concat([dataFrame.hscore, dataFrame.vscore]),
            'vscore': pd.concat([dataFrame.vscore, dataFrame.hscore]),
            'weight': pd.concat([dataFrame.weight, dataFrame.weight]),
            'time': pd.concat([dataFrame.time, dataFrame.time])
        }, columns = ['host', 'visit', 'hscore', 'vscore', 'weight', 'time']).reset_index(drop=True).sort_values(by=['time', 'host'])

        gx = dict()
        ex = dict()
        vx = dict()
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
                ex[(hidx, vidx)] = E(self.miu[hidx], self.miu[vidx], self.phi[vidx])
                vt.append(gx[vidx]**2 * ex[(hidx, vidx)] * (1 - ex[(hidx, vidx)]))
                dt.append(gx[vidx] * (s - ex[(hidx, vidx)]))
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
