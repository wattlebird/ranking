
# Rankit

[![Build Status](https://travis-ci.org/wattlebird/ranking.svg?branch=master)](https://travis-ci.org/wattlebird/ranking) [![PyPI version](https://badge.fury.io/py/rankit.svg)](https://badge.fury.io/py/rankit)

## What is Rankit?

_Rankit_ is a project facilitating ranking process through pairwise comparision.

Suppose there's a soccer/football/baseball competition consisting of a series of matches, one needs to come up with the authentic power of each team based on completed match results. Each match consists of two teams and their corresponding final scores. This could not be done through simply averaging each team's score, since not all teams plays the same number of matches and, besides that, the team that defeated early may not necessarily be the weakest team since they may got paired with a powerhouse at early stage.

_Rankit_ provides two kinds of solutions to solve this problem:

1. All records are presented at once, and then come up with a rating. We provide the following ranking solutions in this way:
   - Massey Rank ([See more](https://www.masseyratings.com/))
   - Colley Rank ([See more](https://www.colleyrankings.com/))
   - Keener Rank ([See more](http://public.gettysburg.edu/~cwessell/RankingPage/keener.pdf))
   - Markov Rank (Page rank)
   - OD Rank (a.k.a. Offence-Defence Rating, [see more](http://www.matterofstats.com/mafl-wagers-and-tips/2012/5/22/the-offense-defense-team-rating-system.html))
   - Difference Rank
2. Matches have time sequence infomation, and player's ratings are got updated match by match. We provide the corresponding ranking solutions:
   - Elo Rank ([See more](https://en.wikipedia.org/wiki/Elo_rating_system))
   - Glicko 2 Rank ([See more](www.glicko.net/glicko.html))
   - TrueSkill Rank (pairwise only)([See more](https://www.microsoft.com/en-us/research/project/trueskill-ranking-system/))

## Quickstart

In Data folder of the code repository, we provide a sample data file illustration the usage of Rankit. The data file records the 2018 NCAA Division I Men's Basketball Tournament result.


```python
import pandas as pd

record = pd.read_csv("Data/sample_data.csv")
```


```python
record.head()
```




<div>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>WTeamName</th>
      <th>LTeamName</th>
      <th>WScore</th>
      <th>LScore</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Radford</td>
      <td>Long Island</td>
      <td>71</td>
      <td>61</td>
    </tr>
    <tr>
      <th>1</th>
      <td>St Bonaventure</td>
      <td>UCLA</td>
      <td>65</td>
      <td>58</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Syracuse</td>
      <td>Arizona St</td>
      <td>60</td>
      <td>56</td>
    </tr>
    <tr>
      <th>3</th>
      <td>TX Southern</td>
      <td>NC Central</td>
      <td>64</td>
      <td>46</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Alabama</td>
      <td>Virginia Tech</td>
      <td>86</td>
      <td>83</td>
    </tr>
  </tbody>
</table>
</div>



Our next step is to, say, provide a ranking from the competition result. The ranking should reflect the authentic strength of each team, rather than bare result concluded from the table above. By using Rankit, one can obtain that ranking all at one stop!


```python
from rankit.Table import Table
from rankit.Ranker import KeenerRanker

data = Table(record, col = ['WTeamName', 'LTeamName', 'WScore', 'LScore'])
ranker = KeenerRanker()
keenerRank = ranker.rank(data)
keenerRank.head()
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
      <td>Villanova</td>
      <td>0.097704</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Michigan</td>
      <td>0.085547</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Kansas</td>
      <td>0.055587</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Loyola-Chicago</td>
      <td>0.048188</td>
      <td>4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Texas Tech</td>
      <td>0.042649</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>



Here we are using Keener ranker. The core idea of Keener ranker is that the absolute strength of a team is measured by all opponents' absolute strength they have encountered and their interaction observation from given data. Besides, the final ranking should be in proportional to a team's absolute strength. The Keener ranking algorithm works best when there are abundant historical data that can have each pair of team's competition record, which in most situation does not stand. In our implementation, we set an epsilon = 0.0001 to each unobserved paired team's competition result.

Besides, since our record table is recorded according to time evolvement, one can also use ranking algorithms that have time series information considered. Maybe you have heard of Elo ranking, and Rankit also provides that:


```python
from rankit.Ranker import EloRanker

eloRanker = EloRanker()
eloRanker.update(data)
eloRank = eloRanker.leaderboard()
eloRank.head()
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
      <td>Villanova</td>
      <td>1544.128053</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Michigan</td>
      <td>1520.358536</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Duke</td>
      <td>1518.033430</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Clemson</td>
      <td>1513.402300</td>
      <td>4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Florida St</td>
      <td>1513.204426</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>



Invoking time series ranking algorithm is a little different from unsupervised ranking algorithm, since one biggest difference is that time series ranker allows you to update leaderboard after a pause, while unsupervised ranking algorithm provides final rank after having observed all records.

Rankit also provides ranking merge functions that allows you to merge different ranks, let's try Borda Count:


```python
from rankit.Merge import borda_count_merge

mergedRank = borda_count_merge([eloRank, keenerRank])
mergedRank.head()
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
      <td>Villanova</td>
      <td>134</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Michigan</td>
      <td>132</td>
      <td>2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Kansas</td>
      <td>126</td>
      <td>3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Florida St</td>
      <td>125</td>
      <td>4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>West Virginia</td>
      <td>123</td>
      <td>5</td>
    </tr>
  </tbody>
</table>
</div>



## API

### rankit.Table

```python
rankit.Table.Table(data=pd.DataFrame(columns=['host', 'visit', 'hscore', 'vscore', 'weight', 'time']), col=['host', 'visit', 'hscore', 'vscore'], weightcol=None, timecol=None)
```

A Table object in rankit is equivalent to data.  
It provides an interface to all ranking solutions in rankit.  

Table accepts <item1, item2, score1, score2> formatted input in pandas.dataframe.  

#### Parameters
**data: pandas.DataFrame** Game result of paired players.  
**col: list of index or column names** Index or column names should indicating ['player1', 'player2', 'score1', 'score2']  
**weightcol:** index or name of column indicating weight. Weight does not take effect in TimeSeries Ranker.  
**timecol:** index or name of column indicating time.  

##### Returns
Table object to be fed to Rankers.

##### Method

| Method | Definition |
|---|---:|
| ```iteritem()``` | Returns an iterator containing each item in table. |
| ```update(table)``` | Update table instance given new table instance, which is going to append existing one. |

### rankit.Ranker

##### MasseyRanker

```python
rankit.Ranker.MasseyRanker(drawMargin = 0.0)
```

Massey ranking system proposed by *Kenneth Massey: Statistical models applied to the rating of sports teams. 
    Bachelor's thesis, Bluefield College, 1997.*
    Core idea: The competition score difference is the rating difference, so one can solve a linear equation by minimize least square error.

#### Parameters
**drawMargin:** [0, +Inf), default 0.  
When absolute difference between two teams are smaller than drawMargin, this competition is considered as a tie.

##### Method

| Method | Definition |
|---|---|
| ```rank(table)``` | Calculate the rank and rating with specified parameters. |

#### ColleyRanker

```python
rankit.Ranker.ColleyRanker(drawMargin = 0.0)
```

Colley ranking system proposed by Wesley Colley: 
    *Colley's bias free college football ranking method: The colley matrix explained, 2002.*
    Core idea: All team's rating starts from 0.5, and with evolvement of games, the rating of each player deviates from 0.5 according to probability of win. However, the average rating of all teams remains 0.5.

##### Parameters
**drawMargin:** [0, +Inf), default 0.  
When absolute difference between two teams are smaller than drawMargin, this competition is considered as a tie.

##### Method

| Method | Definition |
|---|---|
| ```rank(table)``` | Calculate the rank and rating with specified parameters. |

#### KeenerRanker

```python
rankit.Ranker.KeenerRanker(func=None, epsilon=1e-4, threshold=1e-4)
```

Keener ranking system proposed by James Keener:
    *The Perron-Frobenius theorem and the ranking of football teams, SIAM Review, 35(1):80-93, 1993*
    The core idea are: 1. rating is proportional to real strength; 2. real strength is measured relatively by competitors' strength.

##### Parameters
**func:** default None. If set, the score difference should be transformed by the function first then used for rating calculation.  
**epsilon:** [0, +Inf) default 1e-4. The small value that applies an interference to game result that force each team had at least one game with each other.  
**threshold:** (0, +Inf), default 1e-4. The threshold that controls when the algorithm will converge.  

##### Method

| Method | Definition |
|---|---|
| ```rank(table)``` | Calculate the rank and rating with specified parameters. |

#### MarkovRanker

```python
rankit.Ranker.MarkovRanker(restart=0.3, threshold=1e-4)
```

Markov ranking is actually PageRank.
    The core idea is voting: in each game, each team will vote to each other by the number of scores they lost.
    If there are multiple games for a certain pair of player, their scores will be grouped and averaged.

##### Parameters
**restart:** [0, 1], default 0.3. Random walk with restart: in order to avoid black hole in random walk graph.  
**threshold:** (0, +Inf), default 1e-4. The threshold that controls when the algorithm will converge.  "

##### Method

| Method | Definition |
|---|---|
| ```rank(table)``` | Calculate the rank and rating with specified parameters. |

#### ODRanker

```python
rankit.Ranker.ODRanker(method='summary', epsilon=1e-4, threshold=1e-4)
```

The Offence-defence rank tries to assign an offence rating and a defence rating to each team.
    By saying "offence rating", we assume that a team has a high offence rating when it gained a lot of points 
    from a team good in defence. Vise versa. The offence rating of a team is associated with defence rating of each
    competitor in a non-linear way. The defence rating of a team is also non-linearly related to each competitors' 
    offence rating.

##### Parameters
**method:** {'summary', 'offence', 'defence'}, default 'summary'. The rating to be returned. 'summary' is offence/defence.  
**epsilon:** [0, +Inf) default 1e-4. The small value that forces a convergence.  
**threshold:** (0, +Inf), default 1e-4. The threshold that controls when the algorithm will converge.

##### Method

| Method | Definition |
|---|---|
| ```rank(table)``` | Calculate the rank and rating with specified parameters. |

#### DifferenceRanker

```python
rankit.Ranker.DifferenceRanker()
```

This ranker targets at predicting score difference of games directly.
    The difference of ratings are proportional to the difference of score.

##### Method

| Method | Definition |
|---|---|
| ```rank(table)``` | Calculate the rank and rating with specified parameters. |

#### EloRanker

```python
rankit.Ranker.EloRanker(K=10, xi=400, baseline=1500, drawMargin=0)
```

Elo Ranker is a traditional ranking algorithm adjusting player's rating by a series of gaming results. All players starts from 1500 first, and after each paired contest, two player's ranking will be updated in such a way that the sum of their ranking does not change.

##### Parameters
**K:** amount of weight to be applied to each update.  
**xi:** somewhat related to "performance variance", the larger value assumes a more violent game performance and the ranking change will be more conservative.  
**baseline:** the initial ranking of each player.  
**drawMargin:** if the score difference is smaller or equal than drawMargin, this turn of game will be considered as draw. A draw will also effect player's rating.  

##### Method

| Method | Definition |
|---|---|
| ```update(table)``` | Update rating based on a table of record. |
| ```update_single(host, visit, hscore, vscore, time="")``` | Update rating based on a single record. |
| ```prob_win(host, visit)``` | Probability of host player wins over visit player. |
| ```leaderboard(method='min')``` | Presenting current leaderboard. |

#### TrueSkillRanker

```python
rankit.Ranker.TrueSkillRanker(baseline=1500, rd=500, performanceRd=250, drawProbability=0.1, drawMargin=0)
```

Pairwise TrueSkill Ranker is subset of real TrueSkill ranker. See more: https://www.microsoft.com/en-us/research/project/trueskill-ranking-system/ Unlike original TrueSkill ranker, this ranker only process pairwise gaming records.

##### Parameters
**baseline:** the initial ranking value of new players. Default set to 1500.  
**rd:** rating deviation, the possible deviation of a player. Default set to 500.  
**performanceRd:** the possible deviation of each game. Default set to 250.  
**drawProbability:** the probability of draw. Default set to 0.1 and cannot be set to 0.  
**darwMargin:** if the score difference is smaller or equal than drawMargin, this turn of game will be considered as draw. Even if drawMargin is set to 0, drawProbability should never be set to 0.

##### Method

| Method | Definition |
|---|---|
| ```update(table)``` | Update rating based on a table of record. |
| ```update_single(host, visit, hscore, vscore, time="")``` | Update rating based on a single record. |
| ```prob_win(host, visit)``` | Probability of host player wins over visit player. |
| ```leaderboard(method='min')``` | Presenting current leaderboard. |

#### GlickoRanker

```python
rankit.Ranker.GlickoRanker(baseline = 1500, rd = 350, votality = 0.06, tau = 0.5, epsilon = 0.000001, drawMargin = 0)
```

Glicko 2 ranker. See more: http://www.glicko.net/glicko.html  
**Notice:** different from previous rankers, Glicko algorithm involves a concept called "rating period". The update procedure is based on each rating period. In order to specify rating period, one have to state clearly the timestamp in record. Records in the same timestamp will be updated as a batch. If no timestamp is specified, the update algorithm will update the whole records in one batch.

##### Parameters
**baseline:** the initial ranking value of new players. Default set to 1500.  
**rd:** rating deviation, the possible deviation of a player. Default set to 350.  
**votality:** this parameter is to measure the degree of expected fluctuation in a player's rating. Default set to 0.06.  
**tau:** constrains the change of votality over time. The more enormous changes involved in your game, the lower tau should be. Default set to 0.5.  
**epsilon:** parameter to control iteration. Default set to 1e-6.  
**darwMargin:** if the score difference is smaller or equal than drawMargin, this turn of game will be considered as draw. Default set to 0.

##### Method

| Method | Definition |
|:---|---:|
| ```update(table)``` | Update rating based on a table of record. |
| ```update_single(host, visit, hscore, vscore, time="")``` | Update rating based on a single record. |
| ```prob_win(host, visit)``` | Probability of host player wins over visit player. |
| ```leaderboard(method='min')``` | Presenting current leaderboard. |

### rankit.Merge

| Method | Definition |
|---|---|
| ```borda_count_merge(rankings)``` | Merge rankings by using Borda count. |
| ```average_ranking_merge(rankings)``` | Merge rankings by using average of rankings. |
| ```simulation_aggreation_merge(rankings, baseline, method='od')``` | Merge rankings by running simulation of existing rankings. This would first extract relative position of different ranking results, and relative position are considered as simulated games. The game results are sent to another ranker that gives merged ranking result. |

## License

MIT Licensed.
