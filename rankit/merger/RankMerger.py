import pandas as pd
import warnings
from rankit.ranker import BaseRank


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
                assert(type(v), pd.core.frame.DataFrame)
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
            ranktable = pd.DataFrame({'title': ranktable.iloc[:, 0].values,
                                      rankmethod: ranktable.iloc[:, 2].values},
                                     columns=['title', rankmethod])
            self.ranktable = ranktable
            self.cnt+=1

    def delete(self, rankmethod):
        """
        rankmethod: the rankmethod's rank to be deleted.
        return:     None
        """
        ranktable = self.ranktable
        ranktable.drop(rankmethod, axis=1, inplace=True, errors='raise')
        self.ranktable = ranktable
        self.cnt-=1

    def get(self, rankmethod):
        """
        rankmethod: the rankmethod's rank to be retrived.
        return:     dataframe
        """
        ranktable = self.ranktable
        rtn = ranktable[['title', rankmethod]]
        return rtn

class RankMerger(RankManager):
    def __init__(self, *args, **kwargs):
        super(RankMerger, self).__init__(*args, **kwargs)

    def BordaCountMerge(self):
        """
        return: a new DataFrame ranktable, with column=['title', 'rank']
        """
        if self.cnt==0:
            return pd.DataFrame(columns=['title', 'rank'])
        ranktable = self.ranktable
        # ignore all the items that have nan as rank
        ranktable = ranktable.dropna()
        methods = ranktable.columns.drop('title')
        tmprate = ranktable[methods].sum(axis=1)
        tmpitemlst = pd.DataFrame({'itemid': ranktable.loc[:,'title'],
                                   'index': range(ranktable.shape[0])},
                                  columns=['itemid', 'index'])
        ranker = BaseRank(tmpitemlst, ascending=True)
        rtn = ranker.rank(tmprate)
        return rtn.drop('rate', axis=1)
