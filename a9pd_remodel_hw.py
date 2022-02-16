# remodel using new data

import numpy as np
import pandas as pd

# import the two datasets
loan_data_2007_2014 = pd.read_csv('data\loan_data_2007_2014.csv', index_col=0, header=0, low_memory=False)
# loan_data_2007_2014.shape  # (466285, 74)
# loan_data_2007_2014[loan_data_2007_2014['id'].value_counts()>1]
# loan_data_2007_2014['id'].duplicated().sum()

loan_data_2015 = pd.read_csv('data\loan_data_2015.csv', index_col=None, header=0, low_memory=False)
# loan_data_2015.shape # (421094, 74)
# loan_data_2015['id'].duplicated().sum()

loan_data = pd.concat([loan_data_2007_2014, loan_data_2015], axis=0, ignore_index=True)

# t = 466285 + 421094 # 887379
# loan_data.shape # (887379, 74)
# loan_data['id'].duplicated().sum()
# loan_data['member_id'].duplicated().sum()

# explore data columns and dtypes
loan_data_info = pd.DataFrame(loan_data.dtypes)
loan_data_info = loan_data_info.sort_index()



