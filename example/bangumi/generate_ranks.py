from rankit.util import Converter
from rankit.ranker import *
from rankit.manager import *
import cPickle


def main():
    ranks = dict()
    cvt = Converter(filename='average.bin')

    C = cvt.ColleyMatrix()
    b = cvt.ColleyVector()
    ranker = ColleyRank(itemlist=cvt.ItemList())
    x = ranker.rate(C, b)
    r = ranker.rank(x)
    ranks['ariavg_colley'] = r

    M=cvt.MasseyMatrix()
    b=cvt.MasseyVector()
    ranker=MasseyRank(itemlist=cvt.ItemList())
    x=ranker.rate(M,b)
    r=ranker.rank(x)
    ranks['ariavg_massey']=r

    M=cvt.SymmetricDifferenceMatrix()
    ranker=DifferenceRank(itemlist=cvt.ItemList())
    x=ranker.rate(M)
    r=ranker.rank(x)
    ranks['ariavg_differ']=r

    M=cvt.RateVoteMatrix()
    ranker=MarkovRank(itemlist=cvt.ItemList(), epsilon=0.6)
    x = ranker.rate(M)
    r = ranker.rank(x)
    ranks['ariavg_markov_rv']=r

    M=cvt.RateDifferenceVoteMatrix()
    ranker=MarkovRank(itemlist=cvt.ItemList(), epsilon=0.6)
    x = ranker.rate(M)
    r = ranker.rank(x)
    ranks['ariavg_markov_rdv']=r

    M=cvt.SimpleDifferenceVoteMatrix()
    ranker=MarkovRank(itemlist=cvt.ItemList(), epsilon=0.6)
    x = ranker.rate(M)
    r = ranker.rank(x)
    ranks['ariavg_markov_sdv']=r

    M=cvt.RateVoteMatrix()
    ranker=ODRank(itemlist=cvt.ItemList(), iteration=1000000)
    x=ranker.rate(M)
    r=ranker.rank(x)
    ranks['ariavg_od']=r

    C = cvt.CountMatrix()
    ranker=KeenerRank(itemlist=cvt.ItemList(), epsilon=1e-4)
    x = ranker.rate(M.T, C)
    r = ranker.rank(x)
    ranks['ariavg_keener']=r

    cvt = Converter(filename='prob.bin')

    C=cvt.ColleyMatrix()
    b=cvt.ColleyVector()
    ranker=ColleyRank(itemlist=cvt.ItemList())
    x=ranker.rate(C,b)
    r=ranker.rank(x)
    ranks['prob_colley']=r

    M=cvt.MasseyMatrix()
    b=cvt.MasseyVector()
    ranker=MasseyRank(itemlist=cvt.ItemList())
    x=ranker.rate(M,b)
    r=ranker.rank(x)
    ranks['prob_massey']=r

    M=cvt.SymmetricDifferenceMatrix()
    ranker=DifferenceRank(itemlist=cvt.ItemList())
    x=ranker.rate(M)
    r=ranker.rank(x)
    ranks['prob_differ']=r

    M=cvt.RateVoteMatrix()
    ranker=MarkovRank(itemlist=cvt.ItemList(), epsilon=0.6)
    x = ranker.rate(M)
    r = ranker.rank(x)
    ranks['prob_markov_rv']=r

    M=cvt.RateDifferenceVoteMatrix()
    ranker=MarkovRank(itemlist=cvt.ItemList(), epsilon=0.6)
    x = ranker.rate(M)
    r = ranker.rank(x)
    ranks['prob_markov_rdv']=r

    M=cvt.SimpleDifferenceVoteMatrix()
    ranker=MarkovRank(itemlist=cvt.ItemList(), epsilon=0.6)
    x = ranker.rate(M)
    r = ranker.rank(x)
    ranks['prob_markov_sdv']=r

    M=cvt.RateVoteMatrix()
    ranker=ODRank(itemlist=cvt.ItemList(), iteration=1000000)
    x=ranker.rate(M)
    r=ranker.rank(x)
    ranks['prob_od']=r

    C = cvt.CountMatrix()
    ranker=KeenerRank(itemlist=cvt.ItemList(), epsilon=1e-4)
    x = ranker.rate(M.T, C)
    r = ranker.rank(x)
    ranks['prob_keener']=r

    mgr = RankMerger(availableranks=ranks)
    finalrank = mgr.BordaCountMerge()

    with open('result.pkl', 'wb') as fw:
        cPickle.dump(mgr, fw)
        cPickle.dump(finalrank, fw)

if __name__=="__main__":
    main()
