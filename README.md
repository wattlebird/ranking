
# Rankit

Game Start!


## What is rankit?

Rankit is created for the purpose of a more "scientific" ranking of rankable objects. 

We rank objects by giving objects a score, we call that score rating. Traditionally, people would generate ratings by calculating average score, median or some other statistical meaningful numbers. However, eventhough this method is widely accepted, it can have bias in some extreme cases. Average score would easily be manipulated if the number of scores are unrestricted. One cope to this cheat is weighting, but this can only leverage the problem but not solving it.

Here in rankit, we solve the problem by borrowing wisdoms from linear algebra. Every rankable objects are ranked based on their performance when they compare with each other. We compare each pair of rankable objects, and we generate a matrix. Then we get the ranking from that matrix, using different ranking methods. Most of the ranking methods are based on a same idea: ranking is a state that should be extracted from observations which are encoded in matrix.

To further compete with ranking cheating, rankit also included ranking merging methods and provided measures to measure distance between different ranking results.

All the algorithms implemented in rankit have been described in [Who's \#1? The Science of Rating and Ranking](http://www.amazon.com/Whos-1-Science-Rating-Ranking/dp/0691154228/ref=sr_1_1?s=books&ie=UTF8&qid=1454383363&sr=1-1&keywords=who%27s+1+the+science+of+rating+and+ranking). In fact, rankit is a sheer implementation of this book.

## Quick start

Suppose we want to generate the rank of five teams from NCAA American football competition by using their scores in season 2005 (this is also the example used more than once in the book I mentioned above:)


```python
from rankit.util import Converter
cvt = Converter(filename="Data/test_small.bin")
cvt.table
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>primary</th>
      <th>secondary</th>
      <th>rate1</th>
      <th>rate2</th>
      <th>weight</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Duke</td>
      <td>Miami</td>
      <td>7</td>
      <td>52</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Duke</td>
      <td>UNC</td>
      <td>21</td>
      <td>24</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Duke</td>
      <td>UVA</td>
      <td>7</td>
      <td>38</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Duke</td>
      <td>VT</td>
      <td>0</td>
      <td>45</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Miami</td>
      <td>UNC</td>
      <td>34</td>
      <td>16</td>
      <td>1</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Miami</td>
      <td>UVA</td>
      <td>25</td>
      <td>17</td>
      <td>1</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Miami</td>
      <td>VT</td>
      <td>27</td>
      <td>7</td>
      <td>1</td>
    </tr>
    <tr>
      <th>7</th>
      <td>UNC</td>
      <td>UVA</td>
      <td>7</td>
      <td>5</td>
      <td>1</td>
    </tr>
    <tr>
      <th>8</th>
      <td>UNC</td>
      <td>VT</td>
      <td>3</td>
      <td>30</td>
      <td>1</td>
    </tr>
    <tr>
      <th>9</th>
      <td>UVA</td>
      <td>VT</td>
      <td>14</td>
      <td>52</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
</div>



In the code above, I preloaded the contest data that has been stored in "Data/test_small.bin". The Pandas DataFrame has recorded all the contest details. The primary and secondary columns are the name for each rankable objects, and the rate1, rate2 columns are their corresponding scores in this match. The last column, weight, is to indicate the significance of this match, which could be counted in the following algorithms.

Now, have the necessary contest data, it is time to calculate the ranking of these five football teams. To make it easier for programmers who does not know too much about the ranking techiques mentioned in that book, we start from a most obvious case. You may have heard of [PageRank](https://en.wikipedia.org/wiki/PageRank), that famous algorithm that proposed by the founders of Google in the end of last centary. The main idea of PR is to identify in-links and out-links between different webpages on the Internet, and under [certain conditions](https://en.wikipedia.org/wiki/Markov_chain), we can obtain the final state vector which indicates the long-term probability that we visit every pages.

The good news is that this idea could also be used in the rating of different teams. Here we do not have the concept of in or out links, but teams would vote to other teams by their performance. For example, in the first contest record, Miami beated Duke by 52:7. This result could be interpreted in this way: Duke voted 45 scores to Miami. By checking every contest, we could get the following voting matrix:


```python
D = cvt.RateDifferenceVoteMatrix()
D
```




    array([[  0.,  45.,   3.,  31.,  45.],
           [  0.,   0.,   0.,   0.,   0.],
           [  0.,  18.,   0.,   0.,  27.],
           [  0.,   8.,   2.,   0.,  38.],
           [  0.,  20.,   0.,   0.,   0.]], dtype=float32)



Here we have encoded the teams' voting information in a matrix D. Now we could calculate the ranking! But before we start, I have to let you know that in order for the result to coverage, it is necessary to do some modifications to make the matrix satisify coverage conditions. If you know PageRank, it does not use the probability matrix directly, instead they used

$$
\mathbf{A} = \epsilon \mathbf{A} + (1-\epsilon) \mathbf{e} \mathbf{e}^T
$$

to enforce the reducibility. In the ranker, we can also use the same technique:


```python
from rankit.ranker import MarkovRank
ranker = MarkovRank(itemlist=cvt.ItemList(), epsilon=0.8)
r = ranker.rate(D)
ranker.rank(r)
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>title</th>
      <th>rate</th>
      <th>rank</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Miami</td>
      <td>0.509513</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>VT</td>
      <td>0.25781</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>UVA</td>
      <td>0.0903284</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>UNC</td>
      <td>0.0748185</td>
      <td>4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Duke</td>
      <td>0.0675298</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>



In the code above, we pass the ranker MarkovRank with `itemlist=cvt.ItemList()` to tell ranker the rankable objects' names. It should be a pandas DataFrame with column "itemid" and "index". "index" must starts from 0.


```python
cvt.ItemList()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>itemid</th>
      <th>index</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Duke</td>
      <td>0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Miami</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>UNC</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>UVA</td>
      <td>3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>VT</td>
      <td>4</td>
    </tr>
  </tbody>
</table>
</div>



Now we have calculated the ranking of five teams in one method! But things shouldn't have stopped here. We have provided a lot more methods for you to try. If you do not want to know the details of these algorithms, you can arrange all your contest information in a pandas DataFrame as shown in the first table and let our Converter to generate suitable matrix for you. If you know how to arrange matrix, it would be unnecessary to rely on Converter to get itemlists and matrix. You could roll your own matrix!

Now, what if I have calculated two or more rankings and I want to merge them into one ranking? There are many ways to do merging, and we take Borda Count as an example:


```python
from rankit.manager import RankMerger
from rankit.util import Converter
from rankit.ranker import ColleyRank, MasseyRank, ODRank, MarkovRank, KeenerRank
cvt = Converter("Data/test_small.bin")

C=cvt.ColleyMatrix()
b=cvt.ColleyVector()
ranker=ColleyRank(itemlist=cvt.ItemList())
x=ranker.rate(C,b)
r1=ranker.rank(x)

M=cvt.MasseyMatrix()
b=cvt.MasseyVector()
ranker=MasseyRank(itemlist=cvt.ItemList())
x=ranker.rate(M,b)
r2=ranker.rank(x)

M=cvt.RateVoteMatrix()
ranker=ODRank(itemlist=cvt.ItemList())
x=ranker.rate(M)
r3=ranker.rank(x)

ranker=MarkovRank(itemlist=cvt.ItemList(), epsilon=0.8)
x = ranker.rate(M)
r4 = ranker.rank(x)

ranker=KeenerRank(itemlist=cvt.ItemList())
C = cvt.CountMatrix()
x = ranker.rate(M.T, C)
r5 = ranker.rank(x)

ranktable = dict()
ranktable['colley']=r1
ranktable['massey']=r2
ranktable['od']=r3
ranktable['markov']=r4
ranktable['keener']=r5

mgr = RankMerger(availableranks=ranktable)
mgr.BordaCountMerge()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>title</th>
      <th>rate</th>
      <th>rank</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Miami</td>
      <td>19</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>VT</td>
      <td>16</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>UVA</td>
      <td>9</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>UNC</td>
      <td>6</td>
      <td>4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Duke</td>
      <td>0</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>



Merger is created to make final rank more reliable and more difficult for cheaters to manipulate the ranking. In rankit, we also provided a set of interesting merging algorithms.

So that's rankit! I hope that with rankit, there will be less dispute on the cheating of ranking and common people who does not know about the science of ranking will benefit from it.

## Reference

Adaptable matrix and ranking algorithms:

| Algorithms in rankit.ranker | Corresponding Matrix provided by rankit.util.Converter |
|---|---|
| ColleyRank | ColleyMatrix and ColleyVector |
| MasseyRank | MasseyMatrix and MasseyVector |
| DifferenceRank | SymmetricDifferenceMatrix |
| MarkovRank | RateDifferenceVoteMatrix or SimpleDifferenceVoteMatrix or RateVoteMatrix |
| ODRank | RateDifferenceVoteMatrix or SimpleDifferenceVoteMatrix or RateVoteMatrix |
| KeenerRank | Transpotation of RateVoteMatrix and CountMatrix |
| LeastViolatedRank\* | Transpotation of RateDifferenceVoteMatrix -> ConsistancyMatrix |

Rank merging algorithms:

| Merging methods in rankit.manager.Merger | Notes |
|---|---|
| BordaCountMerge | Merge by Borda count |
| AverageRankMerge | Merge by Averaging the rank |
| RankListVoteMerge | Merge by graph algorithms |
| LeastViolatedMerge\* | Merge by optimize the violation loss |

\* Not suitable for large ranking (# ranked objects>100). Requires [Google or-tools](https://developers.google.com/optimization/) with python interface installed. And we did not provide interface to CPlex.

## License

MIT Licensed.


