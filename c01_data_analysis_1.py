# Analyse data to have feel of the data

import numpy as np
import pandas as pd
import matplotlib as plt
import seaborn as sbs

# import data
df1 = pd.read_csv('data/body_fat.csv')

# 5 Summary Statistics
# Five-Number Summary or pandas df.describe()
# https://statisticsbyjim.com/basics/five-number-summary/

df1.describe()
'''
         pct_fat
count  92.00 : 
mean   28.56 : 
std     6.98 : 
min    16.80 : 
25%    23.15 : < Q2. 25% are less than 23.15
50%    27.35 : chekc mean vs 50%. 
  IQR is from 23.15(Q1-Q3) to 33.07 (50% of data is in this range).
  Conversely, 50% is outside of IQR
75%    33.07 : > Q3. 25% are greather than 33.07
max    46.80 : 
'''

# mean         0.060770
# std          0.089202
# min          0.000000
# 25%          0.000000 < Q2
# 50%          0.029466   IQR is 50% of data. The range between 25%(Q2) to 75%(Q3
# 75%          0.114044 > Q3
# max          1.000000

#Box Plot to see

