import pandas as pd
import warnings
from rankit.ranker import BaseRank, MarkovRank, LeastViolatedRank
import numpy as np
from fast_list_matrix import fast_generate_rank_difference_matrix,\
fast_generate_list_difference_matrix


class RankManager(object):
    """
    Ranker Merger should manage different ranking results produced by ranker.
    """
    def __init__(self, availableranks=dict()):
        """
        availableranks: a dictionary with the method of ranking as key and
                        ranking result as value. Can be left blank.
        """
        super(RankManager, self).__init__()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            assert(type(availableranks)==dict)
        cnt = len(availableranks)
        self.cnt=0

        if cnt!=0:
            for k, v in availableranks.iteritems():
                assert(type(v)==pd.core.frame.DataFrame)
                v = v[['title', 'rate', 'rank']]
                if self.cnt==0:
                    ranktable = pd.DataFrame({'title': v.iloc[:, 0].values,
                                              k: v.iloc[:, 2].values},
                                             columns=['title', k])
                    self.cnt+=1
                else:
                    ranktable = ranktable.merge(v, how='outer', on='title')
                    ranktable.rename(columns={'rank':k}, inplace=True)
                    ranktable.drop('rate', axis=1, inplace=True, errors='ignore')
                    self.cnt+=1
            self.ranktable=ranktable
        else:
            self.ranktable=None

    def update(self, rankmethod, newrank):
        """
        rankmethod: the ranktable's ranking method. Should be unique against all
                    the available ranking methods.
        newrank:    the new ranktable.
        return:     None
        """
        newrank = newrank[['title', 'rate', 'rank']]
        ranktable = self.ranktable
        if self.cnt!=0:
            if rankmethod in ranktable.columns:
                raise KeyError("Duplicated rankmethod!")
            ranktable = ranktable.merge(newrank, how='outer', on='title')
            ranktable.rename(columns={'rank': rankmethod}, inplace=True)
            ranktable.drop('rate', axis=1, inplace=True, errors='ignore')
            self.ranktable = ranktable
            self.cnt+=1
        else:
            ranktable = pd.DataFrame({'title': newrank.iloc[:, 0].values,
                                      rankmethod: newrank.iloc[:, 2].values},
                                     columns=['title', rankmethod])
            self.ranktable = ranktable
            self.cnt+=1

    def delete(self, rankmethod, ignore=False):
        """
        rankmethod: the rankmethod's rank to be deleted.
        return:     None
        """
        ranktable = self.ranktable
        if not ignore:
            ranktable.drop(rankmethod, axis=1, inplace=True, errors='raise')
        else:
            ranktable.drop(rankmethod, axis=1, inplace=True, errors='ignore')
        self.ranktable = ranktable
        self.cnt-=1

    def get(self, rankmethod, default=None):
        """
        rankmethod: the rankmethod's rank to be retrived.
        default:    if the rank couldn't be retrived, the return value should be
                    default. If default is None, it will raise an exception!
        return:     dataframe
        """
        ranktable = self.ranktable
        try:
            rtn = ranktable[['title', rankmethod]]
            return rtn
        except KeyError as e:
            if default is None:
                raise e
            else:
                return default


class RankMerger(RankManager):
    def __init__(self, *args, **kwargs):
        super(RankMerger, self).__init__(*args, **kwargs)

    def BordaCountMerge(self):
        """
        return: a new DataFrame ranktable, with column=['title', 'rank']
        """
        if self.cnt==0:
            return pd.DataFrame(columns=['title', 'rate', 'rank'])
        ranktable = self.ranktable
        # ignore all the items that have nan as rank
        ranktable = ranktable.dropna()
        methods = ranktable.columns.drop('title')
        candidate_score = np.zeros((ranktable.shape[0], ranktable.shape[1]-1),
                                   np.int32)

        for c in xrange(methods.shape[0]):
            method = methods[c]
            tmprank = ranktable[['title', method]].sort_values(by=method)
            tmprank['_borda_score'] = pd.Series(
                self._get_borda_score(tmprank.loc[:,method].values),
                index=tmprank.index)
            tmprank.sort_index(inplace=True)
            candidate_score[:, c]=tmprank.loc[:,'_borda_score']
        borda_score = candidate_score.sum(axis=1)

        tmpitemlst = pd.DataFrame({'itemid': ranktable.loc[:,'title'],
                                   'index': range(ranktable.shape[0])},
                                  columns=['itemid', 'index'])
        ranker = BaseRank(tmpitemlst)
        rtn = ranker.rank(borda_score)
        return rtn

    def AverageRankMerge(self):
        if self.cnt==0:
            return pd.DataFrame(columns=['title', 'rate', 'rank'])
        ranktable = self.ranktable
        ranktable = ranktable.dropna()
        methods = ranktable.columns.drop('title')
        average_score = ranktable[methods].sum(axis=1).values

        tmpitemlst = pd.DataFrame({'itemid': ranktable.loc[:,'title'],
                                   'index': range(ranktable.shape[0])},
                                  columns=['itemid', 'index'])
        ranker = BaseRank(tmpitemlst, ascending=True)
        rtn = ranker.rank(average_score)
        return rtn

    def RankListVoteMerge(self, matrix_type="rankdifference", epsilon=0.75):
        """
        matrix_type: 'rankdifference' or 'listdifference'
        epsilon:     PR specific value
        """
        if self.cnt==0:
            return pd.DataFrame(columns=['title', 'rate', 'rank'])
        ranktable = self.ranktable
        ranktable = ranktable.dropna()
        methods = ranktable.columns.drop('title')
        D = np.zeros((ranktable.shape[0], ranktable.shape[0]), dtype=np.float32)
        if matrix_type=="rankdifference":
            fast_generate_rank_difference_matrix(ranktable[methods].values, D)
        elif matrix_type=="listdifference":
            fast_generate_list_difference_matrix(ranktable[methods].values, D)
        else:
            raise ValueError("Invalid matrix type.")

        tmpitemlst = pd.DataFrame({'itemid': ranktable.loc[:,'title'],
                                   'index': range(ranktable.shape[0])},
                                  columns=['itemid', 'index'])
        ranker = MarkovRank(itemlist=tmpitemlst, epsilon=epsilon)
        rate = ranker.rate(D)
        rtn = ranker.rank(rate)
        return rtn

    def LeastViolatedMerge(self, verbose=0):
        if self.cnt==0:
            return pd.DataFrame(columns=['title', 'rate', 'rank'])
        ranktable = self.ranktable
        ranktable = ranktable.dropna()
        methods = ranktable.columns.drop('title')
        C = np.zeros((ranktable.shape[0], ranktable.shape[0]), dtype=np.float32)
        fast_generate_list_difference_matrix(ranktable[methods].values, C)
        # originally in this funciton, C[i, j] is the list count that ranked i lower than j
        # but here, C[i, j] should be the number of lists that i is ranked higher than j minus
        # the number of lists that i is ranked lower than j.
        C = C.T-C

        tmpitemlst = pd.DataFrame({'itemid': ranktable.loc[:,'title'],
                                   'index': range(ranktable.shape[0])},
                                  columns=['itemid', 'index'])
        ranker = LeastViolatedRank(itemlist=tmpitemlst, minimize=False,
                                   verbose=verbose, ascending=True)
        rate = ranker.rate(C)
        rtn = ranker.rank(rate)
        return rtn


    def _get_borda_score(self, rate):
        score = np.zeros(rate.shape)
        # one thing that could be made sure is that rate is sorted.
        b=0
        for e in xrange(1, rate.shape[0]+1):
            if e==rate.shape[0] or rate[b]!=rate[e]:
                for i in xrange(b, e):
                    score[i] = rate.shape[0]-e
                b=e
        return score

class RankComparer(RankManager):
    def __init__(self, *args, **kwargs):
        super(RankComparer, self).__init__(*args, **kwargs)

    def KendallMeasure(self, method1, method2):
        """Kandall Measure ranges from -1 to 1
        The closer to 1, the more similar two ranks are.
        The closer to -1, the less similar two ranks are.
        Currently, this only applies to complete comparision.
        """
        ranktable = self.ranktable
        ranktable = ranktable.dropna() # Force Complete Kendall meaure
        methods = ranktable.columns.drop('title')
        a = ranktable[[method1, method2]].sort_values(by=method1).values
        nd=0.0;
        for i in xrange(a.shape[0]):
            for b in xrange(a.shape[0]-1, i, -1):
                if a[b-1, 1]>a[b, 1]:
                    a[b, 1], a[b-1, 1] = a[b-1, 1], a[b, 1]
                    nd+=1;
        n = a.shape[0]*(a.shape[0]-1)/2.0
        return (n-nd*2)/n

    def SpearmanMeasure(self, method1, method2):
        """SpearmanMeaure ranges from 0 to infinite.
        The smaller Spearman Measure, the more similar to ranks are.
        """
        ranktable = self.ranktable
        ranktable = ranktable.dropna() # Force Complete Kendall meaure
        methods = ranktable.columns.drop('title')
        a = ranktable[[method1, method2]].values
        phi = 0.0
        for k in xrange(a.shape[0]):
            phi+=abs(a[k, 0] - a[k, 1])/float(min(a[k, 0], a[k, 1]))
        return phi
