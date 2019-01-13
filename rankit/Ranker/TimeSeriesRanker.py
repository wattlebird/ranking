from __future__ import division
from __future__ import absolute_import
import numpy as np
import pandas as pd
from rankit.Table import Table
from scipy.stats.norm import pdf, cdf

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
        
    def update_single(self, host, visit, hscore, vscore, time, weight):
        self.data.update_single(host, visit, hscore, vscore, time, weight)
        ih = self.data.itemlut[host]
        iv = self.data.itemlut[visit]
        if ih >= len(self.indexScoreLut):
            self.indexScoreLut.append(self.baseline)
        if iv >= len(self.indexScoreLut):
            self.indexScoreLut.append(self.baseline)
        rh = self.indexScoreLut[ih]
        rv = self.indexScoreLut[rv]

        xi, K = self.xi, self.K
        s = 0.5 if hscore == vscore else (1 if hscore > vscore else 0)
        phwin = 1/(1+10**((rv - rh)/xi))
        alpha = (abs(hscore-vscore)+3)**0.8/(7.5+0.0006*(rh - rv))
        delta = K*weight*alpha*(s-phwin)
        self.indexScoreLut[ih] += delta
        self.indexScoreLut[iv] -= delta
        return (self.indexScoreLut[ih], self.indexScoreLut[iv])

    def update(self, table):
        self.data.update_single(host, visit, hscore, vscore, time, weight)
        self.indexScoreLut = self.indexScoreLut[:] + [self.baseline] * (len(self.data.itemnum - self.indexScoreLut))

        xi, K = self.xi, self.K
        for rec in self.iteritem():
            ih, iv, hscore, vscore = rec.indexHost, rec.indexVisit, rec.hscore, rec.vscore
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
            "name": self.data.indexlut
            "rating": self.indexScoreLut
        }, columns=["name", "rating"])
        rtn['rank'] = rtn.rating.rank(method=method, ascending=False).astype(np.int32)
        return rtn.sort_values(by=['rating', 'name'], ascending=False).reset_index(drop=True)

class TrueSkillRanker(TimeSeriesRanker):
    def __init__(self, range=50, drawMargin=0, confidenceInterval=3, beta=1/2):
        self.miu = confidenceInterval
        self.sigma = 1
        self.beta = 1
        self.drawMargin = drawMargin
        self.range = range
        self.data = Table()
        self.indexMiuLut = []
        self.indexSigmaSqrLut = []
        self.v = lambda t, a : pdf(t-a)/cdf(t-a)
        self.w = lambda t, a: self.v(t, a) * (self.v(t, a) + t - a)
        self.vt = lambda t, a: (pdf(- a - t) - pdf(a - t))/(cdf(a - t) - cdf(- a - t))
        self.wt = lambda t, a: self.vt(t, a) ** 2 + ((a-t)*pdf(a-t)-(a+t)*pdf(a+t))/(cdf(a - t) - cdf(- a - t))

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
    
    def update_single(self, host, visit, hscore, vscore, time, weight):
        v, w, vt, wt, beta, drawMargin = self.v, self.w, self.vt, self.wt, self.beta, self.drawMargin
        self.data.update_single(host, visit, hscore, vscore, time, weight)
        ih = self.data.itemlut[host]
        iv = self.data.itemlut[visit]
        if ih >= len(self.indexMiuLut) || iv >= len(self.indexMiuLut) :
            self.indexMiuLut = self.indexMiuLut[:] + [self.confidenceInterval] * (self.data.itemnum - len(self.indexMiuLut))
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

        return self.indexMiuLut[ih] - self.indexSigmaSqrLut[ih]**(1/2), self.indexMiuLut[iv] - self.indexSigmaSqrLut[iv]**(1/2)
        