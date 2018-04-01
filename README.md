
# Rankit

[![Build Status](https://travis-ci.org/wattlebird/ranking.svg?branch=master)](https://travis-ci.org/wattlebird/ranking) [![PyPI version](https://badge.fury.io/py/rankit.svg)](https://badge.fury.io/py/rankit)

## What is rankit?

Rankit is created for the purpose of a more "scientific" ranking of rankable objects.

We rank objects by giving objects a score, we call that score rating. Traditionally, people would generate ratings by calculating average score, median or some other statistical meaningful numbers. However, eventhough this method is widely accepted, it can have bias in some extreme cases. Average score would easily be manipulated if the number of scores are unrestricted. One cope to this cheat is weighting, but this can only leverage the problem but not solving it.

Here in Rankit, we provide a variety of ranking solutions other than simple average. These methods includes famous sports ranking solutions like Massey ranking system, Colley ranking system, Keener ranking system, Elo ranking system... Some of the methods borrow the wisdom from PageRank and HITS, and a ranking system aims to predict score difference also exists.

To further compete with ranking cheating, rankit also included ranking merging methods and provided measures to measure distance between different ranking results.

All the algorithms implemented in rankit have been described in [Who's \#1? The Science of Rating and Ranking](http://www.amazon.com/Whos-1-Science-Rating-Ranking/dp/0691154228/ref=sr_1_1?s=books&ie=UTF8&qid=1454383363&sr=1-1&keywords=who%27s+1+the+science+of+rating+and+ranking). In fact, rankit is a sheer implementation of this book.

## Quick start

Suppose we want to generate the Massey rank of five teams from NCAA American football competition by using their scores in season 2005 (this is also the example used more than once in the book I mentioned above:)


```python
import pandas as pd

data = pd.DataFrame({
    "primary": ["Duke", "Duke", "Duke", "Duke", "Miami", "Miami", "Miami", "UNC", "UNC", "UVA"], 
    "secondary": ["Miami", "UNC", "UVA", "VT", "UNC", "UVA", "VT", "UVA", "VT", "VT"],
    "rate1": [7, 21, 7, 0, 34, 25, 27, 7, 3, 14],
    "rate2": [52, 24, 38, 45, 16, 17, 7, 5, 30, 52]
}, columns=["primary", "secondary", "rate1", "rate2"])
data
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
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Duke</td>
      <td>Miami</td>
      <td>7</td>
      <td>52</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Duke</td>
      <td>UNC</td>
      <td>21</td>
      <td>24</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Duke</td>
      <td>UVA</td>
      <td>7</td>
      <td>38</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Duke</td>
      <td>VT</td>
      <td>0</td>
      <td>45</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Miami</td>
      <td>UNC</td>
      <td>34</td>
      <td>16</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Miami</td>
      <td>UVA</td>
      <td>25</td>
      <td>17</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Miami</td>
      <td>VT</td>
      <td>27</td>
      <td>7</td>
    </tr>
    <tr>
      <th>7</th>
      <td>UNC</td>
      <td>UVA</td>
      <td>7</td>
      <td>5</td>
    </tr>
    <tr>
      <th>8</th>
      <td>UNC</td>
      <td>VT</td>
      <td>3</td>
      <td>30</td>
    </tr>
    <tr>
      <th>9</th>
      <td>UVA</td>
      <td>VT</td>
      <td>14</td>
      <td>52</td>
    </tr>
  </tbody>
</table>
</div>




```python
from rankit.Table import Table
from rankit.Ranker import MasseyRanker

data = Table(data, ['primary', 'secondary', 'rate1', 'rate2'])
ranker = MasseyRanker(data)
ranker.rank()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>rating</th>
      <th>rank</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Miami</td>
      <td>18.2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>VT</td>
      <td>18.0</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>UVA</td>
      <td>-3.4</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>UNC</td>
      <td>-8.0</td>
      <td>4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Duke</td>
      <td>-24.8</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>



That's it! All the things you have to do is preparing the games data in the form of pandas DataFrame, specifying the players' columns and score columns, pick a ranker and rank!

There are a variety of ranking methods for you to choose, but what if one wants to merge several ranking results?


```python
from rankit.Ranker import MasseyRanker, ColleyRanker, KeenerRanker, MarkovRanker
from rankit.Merge import borda_count_merge

mergedrank = borda_count_merge([
    MasseyRanker(data).rank(), KeenerRanker(data).rank(), MarkovRanker(data).rank()])
mergedrank
```



<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>name</th>
      <th>BordaCount</th>
      <th>rank</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Miami</td>
      <td>12</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>VT</td>
      <td>9</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>UVA</td>
      <td>6</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>UNC</td>
      <td>3</td>
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



So that's rankit! I hope that with rankit, there will be less dispute on the cheating of ranking and common people who does not know about the science of ranking will benefit from it.

## License

MIT Licensed.
