## Section 9: chp 58 HW : PD Model monitoring.
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


# **************************************************************
# ## General Preprocessing  from
# change datatypes of strings as integers and dates
# change dates to int using reference date
# **************************************************************

# ***************************************
# # Preprocessing few continuous variables=4
# earliest_cr_line ->mths_since_earliest_cr_line
# emp_length ->emp_length_int
# issue_d ->mths_since_issue_d
# term ->term_int
# ---------------------------------------

# loan_data['emp_length'].unique()
loan_data['emp_length_int'] = loan_data['emp_length'].str.replace('\+ years', '', regex=True)  # \s escape-sequence is a regex, same as search paterns *,?,etc.
loan_data['emp_length_int'] = loan_data['emp_length_int'].str.replace('< 1 year', str(0))
loan_data['emp_length_int'] = loan_data['emp_length_int'].str.replace('n/a',  str(0))
loan_data['emp_length_int'] = loan_data['emp_length_int'].str.replace(' years', '')
loan_data['emp_length_int'] = loan_data['emp_length_int'].str.replace(' year', '')

# emp_length_int
loan_data['emp_length_int'] = pd.to_numeric(loan_data['emp_length_int'])

# earliest_cr_line_date
loan_data['earliest_cr_line_date'] = pd.to_datetime(loan_data['earliest_cr_line'], format='%b-%y')

# mths_since_earliest_cr_line
reference_date = pd.to_datetime('2017-12-01')
loan_data['mths_since_earliest_cr_line'] = round(pd.to_numeric((reference_date - loan_data['earliest_cr_line_date']) / np.timedelta64(1, 'M')))


## Homework -----------------------------
# term
loan_data['term_int'] = loan_data['term'].str.replace(' months', '')  # redundant?
# term_int
loan_data['term_int'] = pd.to_numeric(loan_data['term'].str.replace(' months', ''))

# issue_d: Time since the loan was funded
# # Assume we are now in December 2017. Calculate monhts difference to this point
loan_data['issue_d_date'] = pd.to_datetime(loan_data['issue_d'], format='%b-%y')
loan_data['mths_since_issue_d'] = round(pd.to_numeric((reference_date - loan_data['issue_d_date']) / np.timedelta64(1, 'M')))
del reference_date

# ***************************************
# # Preprocessing Discrete variables
# ***************************************
# addr_state
# grade
# home_ownership
# initial_list_status
# loan_status
# purpose
# sub_grade
# verification_status
# ---------------------------------------

# loan_data.info()
# pd.get_dummies(loan_data['grade'])
# grade have 7 categories (A to G). Func will create 7 columns for each categories, all 0 with one 1 to correspond the categ.
# the column names will be the same as the cat. names. We need to be more explicit by adding the column ex. grade:A

# pd.get_dummies=pandas built-in function to create dummy variables
# output is a dataframe
# pd.get_dummies(loan_data['grade'], prefix='grade', prefix_sep=':')

# We create a "List" of dataframes. 1 df for each of the variables below.
loan_data_dummies = [pd.get_dummies(loan_data['addr_state'], prefix='addr_state', prefix_sep=':'),
                     pd.get_dummies(loan_data['grade'], prefix='grade', prefix_sep=':'),
                     pd.get_dummies(loan_data['home_ownership'], prefix='home_ownership', prefix_sep=':'),
                     pd.get_dummies(loan_data['initial_list_status'], prefix='initial_list_status', prefix_sep=':'),
                     pd.get_dummies(loan_data['loan_status'], prefix='loan_status', prefix_sep=':'),
                     pd.get_dummies(loan_data['purpose'], prefix='purpose', prefix_sep=':'),
                     pd.get_dummies(loan_data['sub_grade'], prefix='sub_grade', prefix_sep=':'),
                     pd.get_dummies(loan_data['verification_status'], prefix='verification_status', prefix_sep=':')]

# we concatenate the list of dataframes (consisting of dummy columns) as column (Axis 1)
# into a single dataframe of dummies
loan_data_dummies = pd.concat(loan_data_dummies, axis=1)

# type(loan_data_dummies)

# we add this dataframe(of dummies) into the main loan_data datafrome as columns (Axis=1)
loan_data = pd.concat([loan_data, loan_data_dummies], axis=1)

# initially, we had 81 columns (last being 'mths_since_issue_d')
# 126 new columns with datatype 'uint8' will be added, total of 207 columns with the last col='initial_list_status:w'
# loan_data.columns.values
# loan_data.info()

# ***************************************
# # Check for missing values and clean
# ***************************************
# total_rev_hi_lim
# annual_inc
# mths_since_earliest_cr_line *
# acc_now_delinq
# total_acc
# pub_rec
# open_acc
# inq_last_6mths
# delinq_2yrs
# emp_length_int *
# ---------------------------------------

# loan_data.isnull()
# pd.options.display.max_rows = None
# loan_data.isnull().sum()
# dft = pd.DataFrame(loan_data.isnull().sum(), columns=['sum'])
# # dft[dft['sum'] > 0]  # equivalent to where clause
# list(dft.columns)
# pd.options.display.max_rows = 100

loan_data['total_rev_hi_lim'].fillna(loan_data['funded_amnt'], inplace=True)

# loan_data['total_rev_hi_lim'].isnull().sum()

## Homework --------------
loan_data['annual_inc'].fillna(loan_data['annual_inc'].mean(), inplace=True)
loan_data['mths_since_earliest_cr_line'].fillna(0, inplace=True)
loan_data['acc_now_delinq'].fillna(0, inplace=True)
loan_data['total_acc'].fillna(0, inplace=True)
loan_data['pub_rec'].fillna(0, inplace=True)
loan_data['open_acc'].fillna(0, inplace=True)
loan_data['inq_last_6mths'].fillna(0, inplace=True)
loan_data['delinq_2yrs'].fillna(0, inplace=True)
loan_data['emp_length_int'].fillna(0, inplace=True)

# ***************************************
# Prepare Y Targets
# PD model: Data preparation: Good/ Bad (DV for the PD model)
# loan_data['loan_status'].unique()
# loan_data['loan_status'].value_counts()
# loan_data['loan_status'].value_counts() / loan_data['loan_status'].count()

# Create Target column: Good=1/ Bad Definition=0
loan_data['good_bad'] = np.where(loan_data['loan_status'].isin(['Charged Off', 'Default',
                                                       'Does not meet the credit policy. Status:Charged Off',
                                                       'Late (31-120 days)']), 0, 1)
# **** END PRE=PRECOSSING ******************************************************************
#****************************************************************************
# Section 5: Data Preparation for PD model. Chp 25 and up
# a5_pd_model_data_preparation.py


# Y dependent var preprocessing. Y= a + bX; equation  Yi = f(X, beta) + Ei (error term)
# Create new column (numeric/boolean) based on loan_status. Non-default/good=1 Default/bad=0.
loan_data['good_bad'] = np.where(loan_data['loan_status'].isin(['Charged Off', 'Default',
                                                               'Does not meet the credit policy. Status:Charged Off',
                                                               'Late (31-120 days)']), 0, 1)
# Split Data into Train and Test sets using sklearn
# set test_size= 0.2, therefore 80/20 split train/test. Dfaults is 75/25
# set random_state = 42 (to have stable results,). Each run of train_test_split, sklearn shuffles the deck, therefore give different results.

from sklearn.model_selection import train_test_split

loan_data_inputs_train, loan_data_inputs_test, loan_data_targets_train, loan_data_targets_test = train_test_split(loan_data.drop('good_bad', axis=1), loan_data['good_bad'], test_size= 0.2, random_state = 42)
print('loan_data_inputs_train.shape = ' + str(loan_data_inputs_train.shape))
print('loan_data_inputs_test.shape = ' + str(loan_data_inputs_test.shape))


#-----------------------------------------
# Data Preparation: load Training data: loan_data_inputs_train, loan_data_targets_train
# create df for preprocessing.  calculate WoE and IV   / # REF: Section 5, video 26

# step 1) run using: set a) train set and last part and save to csv

# a) train set
df_inputs_prepr = loan_data_inputs_train
df_targets_prep = loan_data_targets_train

# b) test set
# df_inputs_prepr = loan_data_inputs_test
# df_targets_prep = loan_data_targets_test
#-----------------------------------------

#---------------
# Class test

import importlib
import b02_pd_model_class as pddf
# importlib.reload(pddf)

d = pddf.PdDataframe(loan_data)
# d.columns
# d.shape
# d.info()
# d['dti']
d.sum_column('dti')

# d.groupby_sum('addr_state','annual_inc','mean')
# del d

#---------------
# Class test woe_discrete

# ---- df1_grade : no combining needed ---
df_temp = d.woe_discrete('grade',df_targets_prep)
# plot_by_woe(df_temp)

