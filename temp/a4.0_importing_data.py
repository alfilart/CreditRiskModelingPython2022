#section 4.15 Importing data  / Data Preparation

"""*************************************************************
General Daa Preprocessing. Most import step in modelling.
Data Cleansing
1)Convert text into numeric
2)Convert text date to datetime. * if needed, convert to numeric (i.e. months) to a reference date.
*************************************************************"""
# import libraries
import numpy as np
import pandas as pd

#Import Data
# add r to convert from normal string into raw string
file_path = r'C:\Users\alfil\iCloudDrive\Documents\02.2 Learning Python\DataSets\LendingClubLoanData'
# file_path = file_path.replace('\\','\') # replaces backslashes with forward slashes
file_name_import = r'\loan_data_2007_2014.csv'
loan_data_backup = pd.read_csv(file_path + file_name_import, low_memory=False)  #
loan_data = loan_data_backup.copy()

#View HIDT
loan_data   # [466285 rows x 75 columns]
loan_data.head()
loan_data.info()  #Shows data types
  # also can use  loan_data.columns
loan_data.describe() #
loan_data.tail()
loan_data.values

#to display everyting
pd.options.display.max_columns = None  #To see all columns
pd.options.display.max_columns = False

#-----------------------------
#Explore data - get a feel
# see 1.1 data_exploration.py

# ****************************************
# Preprocessing CONTINUOUS variables
# ****************************************
#-----------------------------
#EMPLOYMENT LENGTH
# Convert text into numerical
loan_data['emp_length'].unique() #show unique values, like group by.

loan_data['emp_length_int'] = loan_data['emp_length'].str.replace('\+ years', '')
loan_data['emp_length_int'] = loan_data['emp_length_int'].str.replace('< 1 year', str(0))
loan_data['emp_length_int'] = loan_data['emp_length_int'].str.replace('n/a',  str(0))
loan_data['emp_length_int'] = loan_data['emp_length_int'].str.replace(' years', '')
loan_data['emp_length_int'] = loan_data['emp_length_int'].str.replace(' year', '')
loan_data['emp_length_int'].unique() #check again that all are int

#Check datatype. check first element(which is at the moment is string)
type(loan_data['emp_length_int'][0])

# Transforms the values to numeric.
loan_data['emp_length_int'] = pd.to_numeric(loan_data['emp_length_int'])

#run the check above line again
type(loan_data['emp_length_int'][0])
#re-check values
loan_data['emp_length_int'].unique()

# ? how to convert to integer int64 , it's currently float

#-----------------------------
# EARLIEST CREDIT LINE DATE
# Extracts the date and the time from a string variable that is in a given format.
# see parameters here: https://docs.python.org/3/library/datetime.html#strftime-and-strptime-behavior
loan_data['earliest_cr_line_date'] = pd.to_datetime(loan_data['earliest_cr_line'], format = '%b-%y')

#Checks the datatype of a single element of a column.
# type(loan_data['earliest_cr_line_date'][0])
# loan_data['earliest_cr_line_date'].unique()

# Assume we are now in December 2017
# Calculate the difference between two dates in months, turn it to numeric datatype and round it. Save the result in a new variable.
loan_data['mths_since_earliest_cr_line'] = round(pd.to_numeric((pd.to_datetime('2017-12-01') - loan_data['earliest_cr_line_date']) / np.timedelta64(1, 'M')))

#Shows some descriptive statisics for the values of a column.
# loan_data['mths_since_earliest_cr_line'].describe()

# We set the rows that had negative differences to the maximum value.
loan_data['mths_since_earliest_cr_line'][loan_data['mths_since_earliest_cr_line'] < 0] = loan_data['mths_since_earliest_cr_line'].max()

#-----------------------------
# TERM
loan_data['term_int'] = pd.to_numeric(loan_data['term'].str.replace(' months',''))

#-----------------------------
# ISSUE DATE
# Assume we are now in December 2017
# Calculate the difference between two dates in months, turn it to numeric datatype and round it. Save the result in a new variable.
loan_data['issue_date'] = pd.to_datetime(loan_data['issue_d'], format='%b-%y')
loan_data['mths_since_issue_date'] = round(pd.to_numeric((pd.to_datetime('2017-12-01') - loan_data['issue_date']) / np.timedelta64(1, 'M')))
# loan_data['mths_since_issue_date'].describe()

# ****************************************
# Preprocessing DISCRETE variables
# grade, sub_grade, home_ownership, verification_status, loan_status, purpose, addr_state, initial_list_status.
# ****************************************
#-----------------------------

# Create dummy variables from a variable.
# DUMMY VARIABLES are binary indicators. 1 if an observation belongs to a category, ), and 0 if its not.
# We only need K-1 dummy variables to represent information about categories

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

# Concatenates the two dataframes.
# Concatenate original dataframe with with dummy variables dataframe
# By specifying axis=0, refers to rows(observations) or Vertical concat.
# By specifying axis=1, refers to columns (variables) or Horizontal concat.
loan_data = pd.concat([loan_data, loan_data_dummies], axis = 1)

# Displays all column names.
loan_data.columns.values

# ****************************************
# Check for missing values and clean
# ****************************************

# Sets the pandas dataframe options to display all columns/ rows.
# pd.options.display.max_rows = None
# loan_data.isnull().sum()
#pd.options.display.max_rows = 100

# fill missing values with another column. 'Total revolving high credit/ credit limit', with funded amt.
loan_data['total_rev_hi_lim'].fillna(loan_data['funded_amnt'], inplace=True)

#fill missing values with MEAN of column
loan_data['annual_inc'].fillna(loan_data['annual_inc'].mean(), inplace=True)

#fill missing values with ZEROES.
loan_data['mths_since_earliest_cr_line'].fillna(0, inplace=True)
loan_data['acc_now_delinq'].fillna(0, inplace=True)
loan_data['total_acc'].fillna(0, inplace=True)
loan_data['pub_rec'].fillna(0, inplace=True)
loan_data['open_acc'].fillna(0, inplace=True)
loan_data['inq_last_6mths'].fillna(0, inplace=True)
loan_data['delinq_2yrs'].fillna(0, inplace=True)
loan_data['emp_length_int'].fillna(0, inplace=True)

# ****************************************
# Export cleaned loand_data
# ****************************************
file_name_export = r'\loan_data_2007_2014_clean.csv'
loan_data.to_csv(file_path + file_name_export, index=False, header=True)

""" 
# ****************************************
# TEMPLATE to load cleaned data
# ****************************************

import numpy as np
import pandas as pd

file_path = r'C:\Users\alfil\iCloudDrive\Documents\02.2 Learning Python\DataSets\LendingClubLoanData'
file_name_import = r'\loan_data_2007_2014_clean.csv'
loan_data = pd.read_csv(file_path + file_name_import, low_memory=False)

"""
#=============================
""" 
#SQL to Panda excercise
loan_data['mths_since_earliest_cr_line'].bool(loan_data['mths_since_earliest_cr_line'] > 0 and loan_data['emp_length_int'] == 0)

loan_data['mths_since_earliest_cr_line']

loan_data[(loan_data['mths_since_earliest_cr_line'] > 0) & (loan_data['emp_length_int']==0)]
# SQL challenge
# loan_data['mths_since_earliest_cr_line'][(loan_data['mths_since_earliest_cr_line'] < 0) and (loan_data['emp_length_int'] > 0)]

**
loan_data.loc[: , ['earliest_cr_line', 'earliest_cr_line_date', 'mths_since_earliest_cr_line']][loan_data['mths_since_earliest_cr_line'] < 0]
# We take three columns from the dataframe. Then, we display them only for the rows where a variable has negative value.
# There are 2303 strange negative values.
"""


