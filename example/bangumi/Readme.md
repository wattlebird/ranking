# Example: Bangumi anime rank

[Bangumi](http://chii.in) is a website that allows users to record, rate and discuss about their favorite Animation, Comic and Galgame works. In this example, we try to use the users' anime rate records to generate a merged anime rank. Rankit provides sevaral methods to calculate the rank by using data from competitions, but the concept of "competition" does not exist in anime rating. First we transform existing user-item-rating information to competitions between animes, then we apply our rankit toolkit.

## Use user\_rate\_converter.py to generate anime competitions

There are three ways to generate competitions between animes.

'arithmetic_mean': Now we select two animes, and keep the set of people who has watched them both. By calculating the arithmetic mean of the ratings rated by this set of people, we can get the scores of the "competition" between these two animes.

'log\_mean': Same as above, but the scores are calculated using $\sum_i \log(rate_i)$.

'probability': Does not calculate scores based on rating, just count how many people rated one anime higher than another, and divide by the total number of this set of people.

The original input is a pandas DataFrame recording user-item-rate. the user\_rate\_converter.py converts this format to the competition format. Once running a script, one has to specify input filename, outputfilename and the conversion algorithm you want to use.

## Calcluate the merged rank in generate\_rank.py

In this example, I have generated all three competition data using all these three algorithms. generate\_rank.py will take all these pandas dataframes and calcluate the rank by using 8 methods:

1. Colley Rank
2. Massey Rank
3. Difference Rank
4. Markov Rank, using rate vote matrix as input
5. Markov Rank, using rate difference vote matrix as input
6. Markov Rank, using simple difference vote matrix as input
7. Offence-defence Rank
8. Keener Rank (no bias)

These 8 methods are applied in all three kind of competition data, thus generating 24 ranks. Then we merge all these ranks by Borda Count, and stores the result to 'result.pkl'.
