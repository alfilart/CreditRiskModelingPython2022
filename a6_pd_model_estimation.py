#  a6_pd_model_estimation
'''
TO DO:
create a loop on df inputs_train and check for column sum <>1

https://www.kaggle.com/getting-started/34336
Error: DataConversionWarning: A column-vector y was passed when a 1d array was expected. Please change the shape of y to (n_samples, ), for example using ravel().
y = column_or_1d(y, warn=True)

How to Adjust CPU Cores Number
https://www.jetbrains.com/help/objc/how-to-adjust-cpu-cores-number.html

Multi-Core example: Multi-Core Machine Learning in Python With Scikit-Learn
https://machinelearningmastery.com/multi-core-machine-learning-in-python/

Increase the memory heap of the IDE
https://www.jetbrains.com/help/pycharm/increasing-memory-heap.html

https://stackoverflow.com/questions/48296019/python-pycharm-memory-and-cpu-allocation-for-faster-runtime

cuML rapids
https://docs.rapids.ai/api/cuml/stable/
https://www.youtube.com/watch?v=ML3vCTOl690
https://developer.nvidia.com/blog/run-rapids-on-microsoft-windows-10-using-wsl-2-the-windows-subsystem-for-linux/
https://docs.rapids.ai/api/cuml/stable/




'''


# Import Libraries
import numpy as np
import pandas as pd

# Loading the Data and Selecting the Features

### Import Data  ************************************************************

loan_data_inputs_train = pd.read_csv('data/loan_data_inputs_train.csv', index_col = 0)
loan_data_targets_train = pd.read_csv('data/loan_data_targets_train.csv', index_col = 0, header = None)
loan_data_inputs_test = pd.read_csv('data/loan_data_inputs_test.csv', index_col = 0)
loan_data_targets_test = pd.read_csv('data/loan_data_targets_test.csv', index_col = 0, header = None)

#Import data (cleaned) from .FEATHER
# loan_data_inputs_train = pd.read_feather('data/loan_data_inputs_train.feather').set_index('index')
# loan_data_targets_train = pd.read_feather('data/loan_data_targets_train.feather').set_index('index')
# loan_data_inputs_test = pd.read_feather('data/loan_data_inputs_test.feather').set_index('index')
# loan_data_targets_test = pd.read_feather('data/loan_data_targets_test.feather').set_index('index')

### Explore Data  ************************************************************
# train
loan_data_inputs_train.head()
loan_data_targets_train.head()

loan_data_inputs_train.shape
loan_data_targets_train.shape

loan_data_inputs_train.index
loan_data_targets_train.index

#target
loan_data_inputs_test.shape
loan_data_targets_test.shape

loan_data_inputs_test.iloc[:,200:].columns

### 1)
### Selecting the Features  ************************************************************
# Here we select a limited set of input variables in a new dataframe.

inputs_train_with_ref_cat = loan_data_inputs_train.loc[: , ['grade:A',
'grade:B',
'grade:C',
'grade:D',
'grade:E',
'grade:F',
'grade:G',
'home_ownership:RENT_OTHER_NONE_ANY',
'home_ownership:OWN',
'home_ownership:MORTGAGE',
'addr_state:ND_NE_IA_NV_FL_HI_AL',
'addr_state:NM_VA',
'addr_state:NY',
'addr_state:OK_TN_MO_LA_MD_NC',
'addr_state:CA',
'addr_state:UT_KY_AZ_NJ',
'addr_state:AR_MI_PA_OH_MN',
'addr_state:RI_MA_DE_SD_IN',
'addr_state:GA_WA_OR',
'addr_state:WI_MT',
'addr_state:TX',
'addr_state:IL_CT',
'addr_state:KS_SC_CO_VT_AK_MS',
'addr_state:WV_NH_WY_DC_ME_ID',
'verification_status:Not Verified',
'verification_status:Source Verified',
'verification_status:Verified',
'purpose:educ__sm_b__wedd__ren_en__mov__house',
'purpose:credit_card',
'purpose:debt_consolidation',
'purpose:oth__med__vacation',
'purpose:major_purch__car__home_impr',
'initial_list_status:f',
'initial_list_status:w',
'term:36',
'term:60',
'emp_length:0',
'emp_length:1',
'emp_length:2-4',
'emp_length:5-6',
'emp_length:7-9',
'emp_length:10',
'mths_since_issue_d:<38',
'mths_since_issue_d:38-39',
'mths_since_issue_d:40-41',
'mths_since_issue_d:42-48',
'mths_since_issue_d:49-52',
'mths_since_issue_d:53-64',
'mths_since_issue_d:65-84',
'mths_since_issue_d:>84',
'int_rate:<9.548',
'int_rate:9.548-12.025',
'int_rate:12.025-15.74',
'int_rate:15.74-20.281',
'int_rate:>20.281',
'mths_since_earliest_cr_line:<140',
'mths_since_earliest_cr_line:141-164',
'mths_since_earliest_cr_line:165-247',
'mths_since_earliest_cr_line:248-270',
'mths_since_earliest_cr_line:271-352',
'mths_since_earliest_cr_line:>352',
'delinq_2yrs:0',
'delinq_2yrs:1-3',
'delinq_2yrs:>=4',
'inq_last_6mths:0',
'inq_last_6mths:1-2',
'inq_last_6mths:3-6',
'inq_last_6mths:>6',
'open_acc:0',
'open_acc:1-3',
'open_acc:4-12',
'open_acc:13-17',
'open_acc:18-22',
'open_acc:23-25',
'open_acc:26-30',
'open_acc:>=31',
'pub_rec:0-2',
'pub_rec:3-4',
'pub_rec:>=5',
'total_acc:<=27',
'total_acc:28-51',
'total_acc:>=52',
'acc_now_delinq:0',
'acc_now_delinq:>=1',
'total_rev_hi_lim:<=5K',
'total_rev_hi_lim:5K-10K',
'total_rev_hi_lim:10K-20K',
'total_rev_hi_lim:20K-30K',
'total_rev_hi_lim:30K-40K',
'total_rev_hi_lim:40K-55K',
'total_rev_hi_lim:55K-95K',
'total_rev_hi_lim:>95K',
'annual_inc:<20K',
'annual_inc:20K-30K',
'annual_inc:30K-40K',
'annual_inc:40K-50K',
'annual_inc:50K-60K',
'annual_inc:60K-70K',
'annual_inc:70K-80K',
'annual_inc:80K-90K',
'annual_inc:90K-100K',
'annual_inc:100K-120K',
'annual_inc:120K-140K',
'annual_inc:>140K',
'dti:<=1.4',
'dti:1.4-3.5',
'dti:3.5-7.7',
'dti:7.7-10.5',
'dti:10.5-16.1',
'dti:16.1-20.3',
'dti:20.3-21.7',
'dti:21.7-22.4',
'dti:22.4-35',
'dti:>35',
'mths_since_last_delinq:Missing',
'mths_since_last_delinq:0-3',
'mths_since_last_delinq:4-30',
'mths_since_last_delinq:31-56',
'mths_since_last_delinq:>=57',
'mths_since_last_record:Missing',
'mths_since_last_record:0-2',
'mths_since_last_record:3-20',
'mths_since_last_record:21-31',
'mths_since_last_record:32-80',
'mths_since_last_record:81-86',
'mths_since_last_record:>86',
]]

# Here we store the names of the reference category dummy variables in a list.
ref_categories = ['grade:G',
'home_ownership:RENT_OTHER_NONE_ANY',
'addr_state:ND_NE_IA_NV_FL_HI_AL',
'verification_status:Verified',
'purpose:educ__sm_b__wedd__ren_en__mov__house',
'initial_list_status:f',
'term:60',
'emp_length:0',
'mths_since_issue_d:>84',
'int_rate:>20.281',
'mths_since_earliest_cr_line:<140',
'delinq_2yrs:>=4',
'inq_last_6mths:>6',
'open_acc:0',
'pub_rec:0-2',
'total_acc:<=27',
'acc_now_delinq:0',
'total_rev_hi_lim:<=5K',
'annual_inc:<20K',
'dti:>35',
'mths_since_last_delinq:0-3',
'mths_since_last_record:0-2']

# create intermediate inputs and targets train df's
# From the dataframe with input variables, we drop the variables with variable names in the list with reference categories.
inputs_train = inputs_train_with_ref_cat.drop(ref_categories, axis=1)

inputs_train.head()

# export to csv
# inputs_train.to_csv('data/inputs_train.csv')
# targets_train.to_csv('data/targets_train.csv')

# import shortcuts ****************************
# Import Libraries

import numpy as np
import pandas as pd
import time # time the run
from datetime import timedelta

# Train dataset
inputs_train = pd.read_csv('data/inputs_train.csv', index_col = 0)
loan_data_targets_train = pd.read_csv('data/loan_data_targets_train.csv', index_col = 0, header = None)

targets_train = loan_data_targets_train.to_numpy()
targets_train = targets_train.ravel()

# test dataset
inputs_test = pd.read_csv('data/inputs_test.csv', index_col = 0)
loan_data_targets_test = pd.read_csv('data/loan_data_targets_test.csv', index_col = 0, header = None)

targets_test = loan_data_targets_test.to_numpy()
targets_test = targets_test.ravel()


# PD Model Estimation

## Logistic Regression  ************************************************************

from sklearn.linear_model import LogisticRegression
from sklearn import metrics

reg = LogisticRegression(max_iter=1000, solver='lbfgs') # , n_jobs=-1)
# reg = LogisticRegression(max_iter=1000, solver='lbfgs', n_jobs=-1)
# expected intercept is -1.89

# reg = LogisticRegression()
# reg = LogisticRegression(max_iter=120, solver='liblinear')
# reg = LogisticRegression(max_iter=120)

# reg = LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=1.0, fit_intercept=True,
#                          intercept_scaling=1, class_weight=None, random_state=None, solver='lbfgs',
#                          max_iter=150, multi_class='auto', verbose=0, warm_start=False, n_jobs=None, l1_ratio=None)
#
# LogisticRegression(C=1.0, class_weight=None, dual=False, fit_intercept=True,
#           intercept_scaling=1, max_iter=100, multi_class='ovr', n_jobs=1,
#           penalty='l2', random_state=None, solver='liblinear', tol=0.0001,
#           verbose=0, warm_start=False)

# default is max_iter=100
# We create an instance of an object from the 'LogisticRegression' class.

pd.options.display.max_rows = None
# Sets the pandas dataframe options to display all columns/ rows.


# reg.fit(inputs_train, loan_data_targets_train.ravel())
# Estimates the coefficients of the object from the 'LogisticRegression' class
# with inputs (independent variables) contained in the first dataframe
# and targets (dependent variables) contained in the second dataframe.
start_time = time.time()

reg.fit(inputs_train, targets_train)

elapsed_time_secs = time.time() - start_time
msg = "Execution took: %s secs (Wall clock time)" % timedelta(seconds=round(elapsed_time_secs))
print(msg)
# ------------------------------------------------------------------

# reg.intercept_
# Displays the intercept contain in the estimated ("fitted") object from the 'LogisticRegression' class.

# reg.coef_
# Displays the coefficients contained in the estimated ("fitted") object from the 'LogisticRegression' class.

feature_name = inputs_train.columns.values
# Stores the names of the columns of a dataframe in a variable.

summary_table = pd.DataFrame(columns=['Feature name'], data=feature_name)
# deleted redundant

summary_table['Coefficients'] = np.transpose(reg.coef_)
# Creates a new column in the dataframe, called 'Coefficients',
# with row values the transposed coefficients from the 'LogisticRegression' object.
summary_table.index = summary_table.index + 1
# Increases the index of every row of the dataframe with 1.
summary_table.loc[0] = ['Intercept', reg.intercept_[0]]
# Assigns values of the row with index 0 of the dataframe.
summary_table = summary_table.sort_index()
# Sorts the dataframe by index.
print('summary_table')
summary_table

### 2) *****************************************************************************************************************
## Build a Logistic Regression Model with P-Values
# P values for sklearn logistic regression.

# Class to display p-values for logistic regression in sklearn.

from sklearn import linear_model
import scipy.stats as stat


class LogisticRegression_with_p_values:

    def __init__(self, *args, **kwargs):  # ,**kwargs):
        self.model = linear_model.LogisticRegression(*args, **kwargs)  # ,**args)

    def fit(self, X, y):
        self.model.fit(X, y)

        #### Get p-values for the fitted model ####
        denom = (2.0 * (1.0 + np.cosh(self.model.decision_function(X))))
        denom = np.tile(denom, (X.shape[1], 1)).T
        F_ij = np.dot((X / denom).T, X)  ## Fisher Information Matrix
        Cramer_Rao = np.linalg.inv(F_ij)  ## Inverse Information Matrix
        sigma_estimates = np.sqrt(np.diagonal(Cramer_Rao))
        z_scores = self.model.coef_[0] / sigma_estimates  # z-score for eaach model coefficient
        p_values = [stat.norm.sf(abs(x)) * 2 for x in z_scores]  ### two tailed test for p-values

        self.coef_ = self.model.coef_
        self.intercept_ = self.model.intercept_
        self.p_values = p_values

reg = LogisticRegression_with_p_values(max_iter=1000, solver='lbfgs')
# We create an instance of an object from the newly created 'LogisticRegression_with_p_values()' class.

start_time = time.time()
# Estimates the coefficients of the object from the 'LogisticRegression' class
# with inputs (independent variables) contained in the first dataframe
# and targets (dependent variables) contained in the second dataframe.
# reg.fit(inputs_train, loan_data_targets_train)
reg.fit(inputs_train, targets_train)

elapsed_time_secs = time.time() - start_time
msg = "Execution took: %s secs (Wall clock time)" % timedelta(seconds=round(elapsed_time_secs))
print(msg)

# Same as above.
summary_table = pd.DataFrame(columns=['Feature name'], data=feature_name)
summary_table['Coefficients'] = np.transpose(reg.coef_)
summary_table.index = summary_table.index + 1
summary_table.loc[0] = ['Intercept', reg.intercept_[0]]
summary_table = summary_table.sort_index()
summary_table

# This is a list.
p_values = reg.p_values
# We take the result of the newly added method 'p_values' and store it in a variable 'p_values'.

# Add the intercept for completeness.
p_values = np.append(np.nan, np.array(p_values))
# We add the value 'NaN' in the beginning of the variable with p-values.

summary_table['p_values'] = p_values
# In the 'summary_table' dataframe, we add a new column, called 'p_values', containing the values from the 'p_values' variable.

summary_table

# export to csv
# summary_table.to_csv('data/summary_table.csv')

### 3) ************************************************************************************************************

# We are going to remove some features, the coefficients for all or almost all of the dummy variables for which, are not tatistically significant.
# Conventionally, if a p-value is lower than 0.05 ( p-values < 0.05), we conclude that the coefficient of a vriable is statistically significant.

# We do that by specifying another list of dummy variables as reference categories, and a list of variables to remove.
# Then, we are going to drop the two datasets from the original list of dummy variables.

# New set of Variables
# removed Cat related to ref categories
# 'delinq_2yrs:>=4', 'open_acc:0','pub_rec:0-2', 'total_acc:<=27', 'total_rev_hi_lim:<=5K',
inputs_train_with_ref_cat2 = loan_data_inputs_train.loc[:, ['grade:A',
                                                           'grade:B',
                                                           'grade:C',
                                                           'grade:D',
                                                           'grade:E',
                                                           'grade:F',
                                                           'grade:G',
                                                           'home_ownership:RENT_OTHER_NONE_ANY',
                                                           'home_ownership:OWN',
                                                           'home_ownership:MORTGAGE',
                                                           'addr_state:ND_NE_IA_NV_FL_HI_AL',
                                                           'addr_state:NM_VA',
                                                           'addr_state:NY',
                                                           'addr_state:OK_TN_MO_LA_MD_NC',
                                                           'addr_state:CA',
                                                           'addr_state:UT_KY_AZ_NJ',
                                                           'addr_state:AR_MI_PA_OH_MN',
                                                           'addr_state:RI_MA_DE_SD_IN',
                                                           'addr_state:GA_WA_OR',
                                                           'addr_state:WI_MT',
                                                           'addr_state:TX',
                                                           'addr_state:IL_CT',
                                                           'addr_state:KS_SC_CO_VT_AK_MS',
                                                           'addr_state:WV_NH_WY_DC_ME_ID',
                                                           'verification_status:Not Verified',
                                                           'verification_status:Source Verified',
                                                           'verification_status:Verified',
                                                           'purpose:educ__sm_b__wedd__ren_en__mov__house',
                                                           'purpose:credit_card',
                                                           'purpose:debt_consolidation',
                                                           'purpose:oth__med__vacation',
                                                           'purpose:major_purch__car__home_impr',
                                                           'initial_list_status:f',
                                                           'initial_list_status:w',
                                                           'term:36',
                                                           'term:60',
                                                           'emp_length:0',
                                                           'emp_length:1',
                                                           'emp_length:2-4',
                                                           'emp_length:5-6',
                                                           'emp_length:7-9',
                                                           'emp_length:10',
                                                           'mths_since_issue_d:<38',
                                                           'mths_since_issue_d:38-39',
                                                           'mths_since_issue_d:40-41',
                                                           'mths_since_issue_d:42-48',
                                                           'mths_since_issue_d:49-52',
                                                           'mths_since_issue_d:53-64',
                                                           'mths_since_issue_d:65-84',
                                                           'mths_since_issue_d:>84',
                                                           'int_rate:<9.548',
                                                           'int_rate:9.548-12.025',
                                                           'int_rate:12.025-15.74',
                                                           'int_rate:15.74-20.281',
                                                           'int_rate:>20.281',
                                                           'mths_since_earliest_cr_line:<140',
                                                           'mths_since_earliest_cr_line:141-164',
                                                           'mths_since_earliest_cr_line:165-247',
                                                           'mths_since_earliest_cr_line:248-270',
                                                           'mths_since_earliest_cr_line:271-352',
                                                           'mths_since_earliest_cr_line:>352',
                                                           'inq_last_6mths:0',
                                                           'inq_last_6mths:1-2',
                                                           'inq_last_6mths:3-6',
                                                           'inq_last_6mths:>6',
                                                           'acc_now_delinq:0',
                                                           'acc_now_delinq:>=1',
                                                           'annual_inc:<20K',
                                                           'annual_inc:20K-30K',
                                                           'annual_inc:30K-40K',
                                                           'annual_inc:40K-50K',
                                                           'annual_inc:50K-60K',
                                                           'annual_inc:60K-70K',
                                                           'annual_inc:70K-80K',
                                                           'annual_inc:80K-90K',
                                                           'annual_inc:90K-100K',
                                                           'annual_inc:100K-120K',
                                                           'annual_inc:120K-140K',
                                                           'annual_inc:>140K',
                                                           'dti:<=1.4',
                                                           'dti:1.4-3.5',
                                                           'dti:3.5-7.7',
                                                           'dti:7.7-10.5',
                                                           'dti:10.5-16.1',
                                                           'dti:16.1-20.3',
                                                           'dti:20.3-21.7',
                                                           'dti:21.7-22.4',
                                                           'dti:22.4-35',
                                                           'dti:>35',
                                                           'mths_since_last_delinq:Missing',
                                                           'mths_since_last_delinq:0-3',
                                                           'mths_since_last_delinq:4-30',
                                                           'mths_since_last_delinq:31-56',
                                                           'mths_since_last_delinq:>=57',
                                                           'mths_since_last_record:Missing',
                                                           'mths_since_last_record:0-2',
                                                           'mths_since_last_record:3-20',
                                                           'mths_since_last_record:21-31',
                                                           'mths_since_last_record:32-80',
                                                           'mths_since_last_record:81-86',
                                                           'mths_since_last_record:>86',
                                                           ]]
# removed ref 'delinq_2yrs:>=4', 'open_acc:0','pub_rec:0-2', 'total_acc:<=27', 'total_rev_hi_lim:<=5K',
ref_categories2 = ['grade:G',
                  'home_ownership:RENT_OTHER_NONE_ANY',
                  'addr_state:ND_NE_IA_NV_FL_HI_AL',
                  'verification_status:Verified',
                  'purpose:educ__sm_b__wedd__ren_en__mov__house',
                  'initial_list_status:f',
                  'term:60',
                  'emp_length:0',
                  'mths_since_issue_d:>84',
                  'int_rate:>20.281',
                  'mths_since_earliest_cr_line:<140',
                  'inq_last_6mths:>6',
                  'acc_now_delinq:0',
                  'annual_inc:<20K',
                  'dti:>35',
                  'mths_since_last_delinq:0-3',
                  'mths_since_last_record:0-2']

# remove dropped categories
# inputs_train2 = inputs_train_with_ref_cat2.drop(ref_categories2, axis=1)
# inputs_train2.head()
# inputs_train2.shape
#
# # export to csv
# inputs_train2.to_csv('data/inputs_train2.csv')


# Part 3) start ************************************************

import numpy as np
import pandas as pd
import time # time the run
from datetime import timedelta

# Import Train2 dataset (with removed columns
inputs_train2 = pd.read_csv('data/inputs_train2.csv', index_col = 0)
loan_data_targets_train = pd.read_csv('data/loan_data_targets_train.csv', index_col = 0, header = None)

targets_train2 = loan_data_targets_train.to_numpy()
targets_train2 = targets_train2.ravel()

# Here we run a new model.
reg2 = LogisticRegression_with_p_values(max_iter=1000, solver='lbfgs') # , n_jobs=-1)

start_time = time.time()
reg2.fit(inputs_train2, targets_train2)
elapsed_time_secs = time.time() - start_time
msg = "reg2.fit - Execution took: %s secs (Wall clock time)" % timedelta(seconds=round(elapsed_time_secs))
print(msg)

# get column names as array
feature_name = inputs_train2.columns.values

# Same as above.
summary_table = pd.DataFrame(columns=['Feature name'], data=feature_name)
summary_table['Coefficients'] = np.transpose(reg2.coef_)
summary_table.index = summary_table.index + 1
summary_table.loc[0] = ['Intercept', reg2.intercept_[0]]
summary_table = summary_table.sort_index()
summary_table

# We add the 'p_values' here, just as we did before.
p_values = reg2.p_values
p_values = np.append(np.nan, np.array(p_values)) # add Nan in p-values to line up to the intercept
summary_table['p_values'] = p_values
summary_table

# export to csv
summary_table.to_csv('data/summary_table2.csv')

# Here we get the results for our final PD model.
import pickle

# Here we export our model to a 'SAV' file with file name 'pd_model.sav'.
pickle.dump(reg2, open('data/pd_model.sav', 'wb'))

# Pickle is used for serializing and de-serializing Python object structures,
# also called marshalling or flattening. Serialization refers to the process of converting an object in memory
# to a byte stream that can be stored on disk or sent over a network
# The pickled python object can be converted back to a Python object using the pickle.load() method
# 'wb' means 'write binary' and is used for the file handle: open('save. p', 'wb' ) which writes
# the pickeled data into a file.

