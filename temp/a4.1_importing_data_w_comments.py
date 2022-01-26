#section 4.15 Importing data
#

import numpy as np
import pandas as pd

# add r to convert from normal string into raw string
file_path = r'C:\Users\alfil\iCloudDrive\Documents\02.2 Learning\Python\DataSets\LendingClubLoanData'
file_name = r'loan_data_2007_2014.csv'
loan_data_backup = pd.read_csv(file_path)
loan_data = loan_data_backup.copy()

#Explore Data.
# hidt. head, info, describe, tail
#pd.options.display.max_columns = None
#pd.options.display.max_rows = None
# Sets the pandas dataframe options to display all columns/ rows.
"""
loan_data.head()  #.tail()
loan_data.info()
# Displays all column names.
loan_data.columns.values
# Displays column names, complete (non-missing) cases per column, and datatype per column.
loan_data.info()
"""

#*************************************************************
# General Preprocessing
# Data Cleansing
# 1)Convert text into numeric
# 2)Convert text date to datetime. * if needed, convert to numeric (i.e. months) to a reference
# date.
#*************************************************************

#------------------------------------
# Preprocessing few CONTINUOUS variables
# NOTE: mostly time related data
#------------------------------------

#_____________________
# PREPROCESS COLUMN: loan_data['emp_length_int']
# We store the preprocessed ‘employment length’ variable in a new variable called ‘employment length int’. Cleanse string ‘+ years’, ''< 1 year', 'n/a', and so on.

#Display unique values of a column.
# loan_data['emp_length'].unique()

# 1)Convert text into numerical
# we use \ escape option in string if there's a non-character like +
loan_data['emp_length_int'] = loan_data['emp_length'].str.replace('\+ years', '')
loan_data['emp_length_int'] = loan_data['emp_length_int'].str.replace('< 1 year', str(0))
loan_data['emp_length_int'] = loan_data['emp_length_int'].str.replace('n/a',  str(0))
loan_data['emp_length_int'] = loan_data['emp_length_int'].str.replace(' years', '')
loan_data['emp_length_int'] = loan_data['emp_length_int'].str.replace(' year', '')
# Check new column's unique values
# loan_data['emp_length_int'].unique()

# Checks the datatype of a single element of a column.
# type(loan_data['emp_length_int'][0])

# Transforms the values to numeric.
# loan_data['emp_length_int'] = pd.to_numeric(loan_data['emp_length_int'])

# Checks the datatype of a single element of a column.
#type(loan_data['emp_length_int'][0])

#------------------------
# PREPROCESS COLUMN: loan_data['term']  #HomeWork
# loan_data['term'].unique()
# loan_data['term_int'] = loan_data['term'].str.replace('months','')
# loan_data['term_int'] = pd.to_numeric(loan_data['term_int'])
#shortcut by joining two lines above
loan_data['term_int'] = pd.to_numeric(loan_data['term'].str.replace('months',''))
#check
# loan_data['term_int'].unique()

#------------------------  vid. sec.4, #16,  min 4:47
# PREPROCESS COLUMN: loan_data['earliest_cr_line']
# Displays a column.
# loan_data['earliest_cr_line'].unique()

#data = 2007 to 2014
# tmp['earliest_cr_line_date'] = loan_data[pd.to_numeric(loan_data['earliest_cr_line'].str[-2:]) >=1] # right function equivalent
# tmp = loan_data[loan_data['earliest_cr_line_date'] > '2014-12-31' ]

tmp = loan_data['earliest_cr_line']==loan_data[pd.to_numeric(loan_data['earliest_cr_line'].str[-2:]) >=1] # right function equivalent



# tmp = pd.to_stringloan_data['earliest_cr_line']
# tmp2 = tmp[pd.to_numeric(tmp['earliest_cr_line'].str[-2:]) >=1] # right function equivalent
# tmp = loan_data[loan_data['earliest_cr_line_date'] > '2014-12-31' ]

# Extracts the date and the time from a string variable that is in a given format.
# see parameters here: https://docs.python.org/3/library/datetime.html#strftime-and-strptime-behavior
loan_data['earliest_cr_line_date'] = pd.to_datetime(loan_data['earliest_cr_line'], format = '%b-%y')

# Checks the datatype of a single element of a column.
# type(loan_data['earliest_cr_line_date'][0])
# loan_data['earliest_cr_line_date'].unique()


# Calculates the difference between two dates and times.
# pd.to_datetime('2017-12-01') - loan_data['earliest_cr_line_date']

# Assume we are now in December 2017
# We calculate the difference between two dates in months, turn it to numeric datatype and round it.
# We save the result in a new variable.
loan_data['mths_since_earliest_cr_line'] = round(pd.to_numeric((pd.to_datetime('2017-12-01') - loan_data['earliest_cr_line_date']) / np.timedelta64(1, 'M')))

# Shows some descriptive statisics for the values of a column.
# Dates from 1969 and before are not being converted well, i.e., they have become 2069 and similar,
# and negative differences are being calculated.
# loan_data['mths_since_earliest_cr_line'].describe()

# We take three columns from the dataframe. Then, we display them only for the rows where a variable has negative value.
# There are 2303 strange negative values.
# loan_data.loc[: , ['earliest_cr_line', 'earliest_cr_line_date', 'mths_since_earliest_cr_line']][loan_data['mths_since_earliest_cr_line'] < 0]
#SQL!!!!! .loc is the SQL equivalent. since we select all rows then : then we specify only columns and the where condition
# SELECT [r,c]->[:,[c1,c2,c3]] where [loan_data['mths_since_earliest_cr_line'] < 0]]

# We set the rows that had negative differences to the maximum value.
loan_data['mths_since_earliest_cr_line'][loan_data['mths_since_earliest_cr_line'] < 0] = loan_data['mths_since_earliest_cr_line'].max()

# SQL challenge
# loan_data['mths_since_earliest_cr_line'][(loan_data['mths_since_earliest_cr_line'] < 0) and (loan_data['emp_length_int'] > 0)]


# Calculates and shows the minimum value of a column.
# min(loan_data['mths_since_earliest_cr_line'])
# loan_data['mths_since_earliest_cr_line'].unique()
# loan_data['mths_since_earliest_cr_line'].describe()


#HW term and issue_date
# TERM
loan_data['term_int'] = pd.to_numeric(loan_data['term'].str.replace(' months',''))

# ISSUE DATE
# loan_data['issue_d'].unique()
# loan_data['issue_d'].describe()
# Assume we are now in December 2017
# We calculate the difference between two dates in months, turn it to numeric datatype and round it.
# We save the result in a new variable.
loan_data['issue_date'] = pd.to_datetime(loan_data['issue_d'], format='%b-%y')
loan_data['mths_since_issue_date'] = round(pd.to_numeric((pd.to_datetime('2017-12-01') - loan_data['issue_date']) / np.timedelta64(1, 'M')))

# loan_data['mths_since_issue_date'].describe()

# ***********************************************************
#------------------------------------
### Preprocessing few DISCRETE variables
#Sec. 4, video # 18.
#------------------------------------
# Displays column names, complete (non-missing) cases per column, and datatype per column.
# loan_data.info()

"""
We are going to preprocess the following discrete variables: 
grade, sub_grade, home_ownership, verification_status, loan_status, purpose, addr_state, initial_list_status. 
Most likely, we are not going to use sub_grade, as it overlaps with grade.
"""

# Create dummy variables from a variable.
# DUMMY VARIABLES are binary indicators. 1 if an observation belongs to a category, ), and 0 if its not.
# We only need K-1 dummy variables to represent information about categories

# pd.get_dummies(loan_data['grade'])
# loan_data['grade'].unique()

# Create dummy variables from a variable. Add columns prefexis
# pd.get_dummies(loan_data['grade'], prefix = 'grade', prefix_sep = ':')

#We store the dummy variables as a list, separate by comma
loan_data_dummies = [pd.get_dummies(loan_data['grade'], prefix = 'grade', prefix_sep = ':'),
                     pd.get_dummies(loan_data['sub_grade'], prefix = 'sub_grade', prefix_sep = ':'),
                     pd.get_dummies(loan_data['home_ownership'], prefix = 'home_ownership', prefix_sep = ':'),
                     pd.get_dummies(loan_data['verification_status'], prefix = 'verification_status', prefix_sep = ':'),
                     pd.get_dummies(loan_data['loan_status'], prefix = 'loan_status', prefix_sep = ':'),
                     pd.get_dummies(loan_data['purpose'], prefix = 'purpose', prefix_sep = ':'),
                     pd.get_dummies(loan_data['addr_state'], prefix = 'addr_state', prefix_sep = ':'),
                     pd.get_dummies(loan_data['initial_list_status'], prefix = 'initial_list_status', prefix_sep = ':')]
# We create dummy variables from all 8 original independent variables, and save them into a list.
# Note that we are using a particular naming convention for all variables: original variable name, colon, category name.


# turn the dummy variables into a dataframe using pd.CONCAT()
loan_data_dummies = pd.concat(loan_data_dummies, axis = 1)

# Returns the type of the variable.
# type(loan_data_dummies)

# Concatenates two dataframes.
# Here we concatenate the dataframe with original data with the dataframe with dummy variables,
# we concatenate by columns by specifiying axis=0, roefers to rows(observations) or vertical
# axis=1, refers to columns (variables) or horizantal concat.
loan_data = pd.concat([loan_data, loan_data_dummies], axis = 1)

# Displays all column names.
loan_data.columns.values

#------------------------------------
### Check for missing values and clean
#------------------------------------

# It returns 'False' if a value is not missing and 'True' if a value is missing, for each value in a dataframe.
loan_data.isnull()

# Sets the pandas dataframe options to display all columns/ rows.
pd.options.display.max_rows = None
loan_data.isnull().sum()

# Sets the pandas dataframe options to display 100 columns/ rows.
pd.options.display.max_rows = 100

# 'Total revolving high credit/ credit limit', so it makes sense that the missing values are equal to funded_amnt.
loan_data['total_rev_hi_lim'].fillna(loan_data['funded_amnt'], inplace=True)
# We fill the missing values with the values of another variable, ex. 'funded_amnt' and we want
# missing values to be replaced in the same variable. We set the inplace to true
# loan_data['total_rev_hi_lim'].isnull().sum()

#HW
#fill the missing values with mean of column
loan_data['annual_inc'].fillna(loan_data['annual_inc'].mean(), inplace=True)
#fill the missing values with zeroes.
loan_data['mths_since_earliest_cr_line'].fillna(0, inplace=True)
loan_data['acc_now_delinq'].fillna(0, inplace=True)
loan_data['total_acc'].fillna(0, inplace=True)
loan_data['pub_rec'].fillna(0, inplace=True)
loan_data['open_acc'].fillna(0, inplace=True)
loan_data['inq_last_6mths'].fillna(0, inplace=True)
loan_data['delinq_2yrs'].fillna(0, inplace=True)
loan_data['emp_length_int'].fillna(0, inplace=True)

""" 
#SQL to Panda excercise
loan_data['mths_since_earliest_cr_line'].bool(loan_data['mths_since_earliest_cr_line'] > 0 and loan_data['emp_length_int'] == 0)

loan_data['mths_since_earliest_cr_line']

loan_data[(loan_data['mths_since_earliest_cr_line'] > 0) & (loan_data['emp_length_int']==0)]
"""

#export cleaned loand_data

file_name2 = r'\loan_data_2007_2014_clean.csv'
loan_data.to_csv(file_path + file_name2, index=False, header=True)

