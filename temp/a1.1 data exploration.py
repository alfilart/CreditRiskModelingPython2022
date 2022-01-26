# import libraries
import numpy as np
import pandas as pd
#set option to display all columns
pd.set_option('display.max_columns', None)
# to go back to default value  pd.reset_option(“max_columns”)

# Import Data
# add r to convert from normal string into raw string
file_path = r'C:\Users\alfil\iCloudDrive\Documents\02.2 Learning Python\DataSets\LendingClubLoanData'
# file_path = file_path.replace('\\','\') # replaces backslashes with forward slashes
file_name_import = r'\loan_data_2007_2014.csv'
loan_data = pd.read_csv(file_path + file_name_import, low_memory=False)


# -----------------------------
# Explore data - get a feel
# HIDT - head, info, describe, tail
loan_data.head()
loan_data.tail()
loan_data.columns.values  # Displays all column names
loan_data.info()  #descriptive stat, but only for numeric (not dates or categorical)
loan_data['emp_length'].unique()
loan_data['emp_length'].head(5)
type(loan_data['term_int'][0]) #check data type
#[466285 rows x 75 columns]

#Data Overview
#1) HITD (head,info,tail,describe)
loan_data.columns
loan_data.columns.values  # Displays all column names
#2) value_counts(). with an s, similar to group by:
loan_data['good_bad'].value_counts() or loan_data['grade'].value_counts()
#3) unique:
loan_data['good_bad'].unique()  or loan_data['grade'].unique()
#4) group by:
# get counts for each Grade. groupby grade, summarize count.
loan_data[['grade', 'dti', 'good_bad']].groupby(['grade']).count()
loan_data[['grade', 'dti', 'good_bad']].groupby(['grade']).agg(['mean', 'count'])
#   size() vs count(). count() excludes NaN values while size() does not




# frequently used display options.
# * see https://pandas.pydata.org/pandas-docs/stable/user_guide/options.html

#display.max_rows and display.max_columns sets the maximum number of rows and columns -
#displayed when a frame is pretty-printed. Truncated lines are replaced by an ellipsis.
# reset_option() - reset one or more options to their default value.
# old ver: pd.options.display.max_rows = None  or pd.options.display.max_rows = 100

#display.expand_frame_repr allows for the representation of dataframes to stretch across pages, wrapped over the full column vs row-wise.
pd.set_option("expand_frame_repr", True)
loan_data

pd.set_option("max_rows", 4)
pd.set_option("max_columns", 50)
loan_data

#rest to default
pd.reset_option("max_rows")
pd.reset_option("max_columns")
pd.set_option("expand_frame_repr", False)
