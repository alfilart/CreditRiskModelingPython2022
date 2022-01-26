#--------------------------------
#SECTION 4: General Preprocessing  Lesson 15-20
#--------------------------------

## Import Libraries

import numpy as np
import pandas as pd

# ## Import Data
file_path = r'C:\Users\alfil\iCloudDrive\Documents\02.2 Learning Python\DataSets\LendingClubLoanData'
file_name_import = r'\loan_data_2007_2014.csv'  # raw data
loan_data_backup = pd.read_csv(file_path + file_name_import, low_memory=False)
loan_data = loan_data_backup.copy()

#****************************************
# Explore Data

# loan_data
# pd.options.display.max_columns = None
# loan_data
# loan_data.head()
# loan_data.tail()
# loan_data.columns.values
# loan_data.info()

#***************************************************************
### General Preprocessing
#***************************************************************

#****************************************
## Preprocessing few continuous variables

# loan_data['emp_length'].unique()

loan_data['emp_length_int'] = loan_data['emp_length'].str.replace('\+ years', '')
loan_data['emp_length_int'] = loan_data['emp_length_int'].str.replace('< 1 year', str(0))
loan_data['emp_length_int'] = loan_data['emp_length_int'].str.replace('n/a',  str(0))
loan_data['emp_length_int'] = loan_data['emp_length_int'].str.replace(' years', '')
loan_data['emp_length_int'] = loan_data['emp_length_int'].str.replace(' year', '')

# type(loan_data['emp_length_int'][0])

loan_data['emp_length_int'] = pd.to_numeric(loan_data['emp_length_int'])

# type(loan_data['emp_length_int'][0])

# loan_data['earliest_cr_line']

loan_data['earliest_cr_line_date'] = pd.to_datetime(loan_data['earliest_cr_line'], format = '%b-%y')

# type(loan_data['earliest_cr_line_date'][0])
# pd.to_datetime('2017-12-01') - loan_data['earliest_cr_line_date']

loan_data['mths_since_earliest_cr_line'] = round(pd.to_numeric((pd.to_datetime('2017-12-01') - loan_data['earliest_cr_line_date']) / np.timedelta64(1, 'M')))

# loan_data['mths_since_earliest_cr_line'].describe()
# loan_data.loc[: , ['earliest_cr_line', 'earliest_cr_line_date', 'mths_since_earliest_cr_line']][loan_data['mths_since_earliest_cr_line'] < 0]
# loan_data['mths_since_earliest_cr_line'][loan_data['mths_since_earliest_cr_line'] < 0] = loan_data['mths_since_earliest_cr_line'].max()

# min(loan_data['mths_since_earliest_cr_line'])


#****************************************
## Homework

# loan_data['term']
# loan_data['term'].describe()
loan_data['term_int'] = loan_data['term'].str.replace(' months', '')
# loan_data['term_int']
# type(loan_data['term_int'][25])
loan_data['term_int'] = pd.to_numeric(loan_data['term'].str.replace(' months', ''))
# loan_data['term_int']
# type(loan_data['term_int'][0])
# loan_data['issue_d']

loan_data['issue_d_date'] = pd.to_datetime(loan_data['issue_d'], format = '%b-%y')
loan_data['mths_since_issue_d'] = round(pd.to_numeric((pd.to_datetime('2017-12-01') - loan_data['issue_d_date']) / np.timedelta64(1, 'M')))
# loan_data['mths_since_issue_d'].describe()


#****************************************
## Preprocessing Discrete variables

# loan_data.info()
# pd.get_dummies(loan_data['grade'])
# grade have 7 categories (A to G). Func will create 7 columns for each categories, all 0 with one 1 to correspond the categ.
# the column names will be the same as the cat. names. We need to be more explicit by adding the column ex. grade:A

# pd.get_dummies = pandas built-in function to create dummy variables
# output is a dataframe
pd.get_dummies(loan_data['grade'], prefix = 'grade', prefix_sep = ':')

# We create a "List" of dataframes. 1 df for each of the variables below.
loan_data_dummies = [pd.get_dummies(loan_data['grade'], prefix = 'grade', prefix_sep = ':'),
                     pd.get_dummies(loan_data['sub_grade'], prefix = 'sub_grade', prefix_sep = ':'),
                     pd.get_dummies(loan_data['home_ownership'], prefix = 'home_ownership', prefix_sep = ':'),
                     pd.get_dummies(loan_data['verification_status'], prefix = 'verification_status', prefix_sep = ':'),
                     pd.get_dummies(loan_data['loan_status'], prefix = 'loan_status', prefix_sep = ':'),
                     pd.get_dummies(loan_data['purpose'], prefix = 'purpose', prefix_sep = ':'),
                     pd.get_dummies(loan_data['addr_state'], prefix = 'addr_state', prefix_sep = ':'),
                     pd.get_dummies(loan_data['initial_list_status'], prefix = 'initial_list_status', prefix_sep = ':')]

# we concatenate the list of dataframes (consisting of dummy columns) as column (Axis 1)
# into a single dataframe of dummies
loan_data_dummies = pd.concat(loan_data_dummies, axis = 1)

# type(loan_data_dummies)

# we add this dataframe(of dummies) into the main loan_data datafrome as columns (Axis=1)
loan_data = pd.concat([loan_data, loan_data_dummies], axis = 1)

# initially, we had 81 columns (last being 'mths_since_issue_d')
# 126 new columns with datatype 'uint8' will be added, total of 207 columns with the last col = 'initial_list_status:w'
# loan_data.columns.values
# loan_data.info()

#****************************************
## Check for missing values and clean

# loan_data.isnull()
# pd.options.display.max_rows = None
# loan_data.isnull().sum()
# pd.options.display.max_rows = 100

loan_data['total_rev_hi_lim'].fillna(loan_data['funded_amnt'], inplace = True)

# loan_data['total_rev_hi_lim'].isnull().sum()

#****************************************
## Homework

loan_data['annual_inc'].fillna(loan_data['annual_inc'].mean(), inplace=True)

loan_data['mths_since_earliest_cr_line'].fillna(0, inplace=True)
loan_data['acc_now_delinq'].fillna(0, inplace=True)
loan_data['total_acc'].fillna(0, inplace=True)
loan_data['pub_rec'].fillna(0, inplace=True)
loan_data['open_acc'].fillna(0, inplace=True)
loan_data['inq_last_6mths'].fillna(0, inplace=True)
loan_data['delinq_2yrs'].fillna(0, inplace=True)
loan_data['emp_length_int'].fillna(0, inplace=True)

#----- END --------
