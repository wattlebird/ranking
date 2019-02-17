
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

## Quick start



## License

MIT Licensed.
