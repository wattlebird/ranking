from rankit.Ranker import TrueSkillRanker
from rankit.Table import Table
import pandas as pd

sample_with_time_1 = pd.DataFrame({
    'primary': [1,1,4,2,3,4,1,2],
    'secondary': [2,3,1,4,2,3,4,3],
    'rate1': [7,6,8,4,3,5,7,0],
    'rate2': [5,5,4,4,4,1,7,1],
    'date': [1,1,2,2,3,3,4,4]
}, columns=['primary', 'secondary', 'rate1', 'rate2', 'date'])

table = Table(sample_with_time_1, ['primary', 'secondary', 'rate1', 'rate2'], timecol='date')
tsRanker = TrueSkillRanker()
tsRanker.update(table)
tsRanker.leaderboard()
