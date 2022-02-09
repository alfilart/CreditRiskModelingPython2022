

# inputs_test_with_ref_cat.to_csv('data/inputs_test_with_ref_cat.csv')
# summary_table.to_csv('data/summary_table.csv')
# np.savetxt('data/categories_with_ref_list.csv', categories_with_ref_list, delimiter =", ", fmt ='%s') # data format as string '%s'
# np.savetxt('data/ref_categories_list.csv', ref_categories_list, delimiter =", ", fmt ='%s') # data format as string '%s'
# np.savetxt('data/y_hat_test_proba.csv', y_hat_test_proba, delimiter =", ", fmt ='%s') # data format as string '%s'

## Import Libraries
import numpy as np
import pandas as pd
from numpy import genfromtxt

inputs_test_with_ref_cat = pd.read_csv('data/inputs_test_with_ref_cat.csv', index_col=0) # header=None)
summary_table = pd.read_csv('data/summary_table.csv', index_col=0)
y_hat_test_proba = genfromtxt('data/y_hat_test_proba.csv', delimiter=',')
df_actual_predicted_probs = pd.read_csv('data/df_actual_predicted_probs.csv', index_col=0)

ref_categories_list = ['grade:G',
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


#***************************************************
# Sec 8: Applying the PD Model  chp 48 - 54
#***************************************************


##Calculating PD of individual accounts
pd.options.display.max_columns = None
# Sets the pandas dataframe options to display all columns/ rows.

inputs_test_with_ref_cat.head()

summary_table

ref_categories_list

y_hat_test_proba


##Creating a Scorecard
summary_table

ref_categories_list

df_ref_categories = pd.DataFrame(ref_categories_list, columns = ['Feature name'])
# We create a new dataframe with one column. Its values are the values from the 'reference_categories' list.
# We name it 'Feature name'.
df_ref_categories['Coefficients'] = 0
# We create a second column, called 'Coefficients', which contains only 0 values.
df_ref_categories['p_values'] = np.nan
# We create a third column, called 'p_values', with contains only NaN values.
df_ref_categories

df_scorecard = pd.concat([summary_table, df_ref_categories])
# Concatenates two dataframes.

df_scorecard['Original feature name'] = df_scorecard['Feature name'].str.split(':').str[0]
# We create a new column, called 'Original feature name', which contains the value of the 'Feature name' column,
# up to the column symbol.

# df_scorecard = df_scorecard.sort_values(['Original feature name','Feature name'])
# sort by orig feat name
df_scorecard = df_scorecard.reset_index()
# We reset the index of a dataframe.

min_score = 300
max_score = 850


# Groups the data by the values of the 'Original feature name' column.
# Aggregates the data in the 'Coefficients' column, calculating their minimum.
min_sum_coef = df_scorecard.groupby('Original feature name')['Coefficients'].min().sum()
# Up to the 'min()' method everything is the same as in te line above.
# Then, we aggregate further and sum all the minimum values.

# Groups the data by the values of the 'Original feature name' column.
# Aggregates the data in the 'Coefficients' column, calculating their maximum.

max_sum_coef = df_scorecard.groupby('Original feature name')['Coefficients'].max().sum()
# Up to the 'min()' method everything is the same as in te line above.
# Then, we aggregate further and sum all the maximum values.


df_scorecard['Score - Calculation'] = df_scorecard['Coefficients'] * (max_score - min_score) / (max_sum_coef - min_sum_coef)
# We multiply the value of the 'Coefficients' column by the ration of the differences between
# maximum score and minimum score and maximum sum of coefficients and minimum sum of cefficients.

# Fix the intercept value
df_scorecard['Score - Calculation'][0] = ((df_scorecard['Coefficients'][0] - min_sum_coef) / (max_sum_coef - min_sum_coef)) * (max_score - min_score) + min_score
# We divide the difference of the value of the 'Coefficients' column and the minimum sum of coefficients by
# the difference of the maximum sum of coefficients and the minimum sum of coefficients.
# Then, we multiply that by the difference between the maximum score and the minimum score.
# Then, we add minimum score.

df_scorecard['Score - Preliminary'] = df_scorecard['Score - Calculation'].round()
# We round the values of the 'Score - Calculation' column.

min_sum_score_prel = df_scorecard.groupby('Original feature name')['Score - Preliminary'].min().sum()
# Groups the data by the values of the 'Original feature name' column.
# Aggregates the data in the 'Coefficients' column, calculating their minimum.
# Sums all minimum values.

max_sum_score_prel = df_scorecard.groupby('Original feature name')['Score - Preliminary'].max().sum()
# Groups the data by the values of the 'Original feature name' column.
# Aggregates the data in the 'Coefficients' column, calculating their maximum.
# Sums all maximum values.

# One has to be subtracted from the maximum score for one original variable. Which one? We'll evaluate based on differences.

df_scorecard['Difference'] = df_scorecard['Score - Preliminary'] - df_scorecard['Score - Calculation']

# df_scorecard.sort_values('Difference',ascending=False)
df_scorecard['Score - Final'] = df_scorecard['Score - Preliminary']

# fix end points by rounding errors at the end to make i 300 (get the min diff and round up) 850 (get the max diff and round down)
# df_scorecard.groupby('Original feature name')['Score - Preliminary'].min().sort_values(ascending=True)
# df_scorecard.groupby('Original feature name')['Score - Preliminary'].max().sort_values(ascending=True)
# df_scorecard.groupby('Original feature name')['Difference'].max().sort_values(ascending=True)
df_scorecard['Score - Preliminary'][55] = -5
df_scorecard['Score - Preliminary'][68] = 29

min_sum_score_prel = df_scorecard.groupby('Original feature name')['Score - Final'].min().sum()
# Groups the data by the values of the 'Original feature name' column.
# Aggregates the data in the 'Coefficients' column, calculating their minimum.
# Sums all minimum values.

max_sum_score_prel = df_scorecard.groupby('Original feature name')['Score - Final'].max().sum()
# Groups the data by the values of the 'Original feature name' column.
# Aggregates the data in the 'Coefficients' column, calculating their maximum.
# Sums all maximum values.

## ----------------------------------------------------------------------
## s8. chp 50: Caclulating Credit Score
## ----------------------------------------------------------------------
inputs_test_with_ref_cat_w_intercept = inputs_test_with_ref_cat

inputs_test_with_ref_cat_w_intercept.insert(0, 'Intercept', 1)
# Insert the intercept in the main df
# We insert a column in the dataframe, with an index of 0, that is, in the beginning of the dataframe.
# The name of that column is 'Intercept', and its values are 1s.

inputs_test_with_ref_cat_w_intercept = inputs_test_with_ref_cat_w_intercept[df_scorecard['Feature name'].values]
# Here, from the 'inputs_test_with_ref_cat_w_intercept' dataframe, we keep only the columns with column names,
# exactly equal to the row values of the 'Feature name' column from the 'df_scorecard' dataframe.
# The values attribute ouputs a list

scorecard_scores = df_scorecard['Score - Final']

# before multiplying df with dummies by scores, we need they have compatible dimension
#inputs_test_with_ref_cat_w_intercept.shape # (93257, 102)
# scorecard_scores.shape # (102,)
# there is no second dimiension and this can be an issue. Functions like multiplication  requires matching dimensions
scorecard_scores = scorecard_scores.values.reshape(102, 1)
### study pandas df dot. Why  df1 col y (93257, 102) is multiplied by df2 rows (102,1)
scorecard_scores.shape # (102, 1)

y_scores = inputs_test_with_ref_cat_w_intercept.dot(scorecard_scores)
# Here we multiply the values of each row of the test data df (dummy variables 0 & 1) by the column with their respective scores,
# then sum it up for each row. Which is an argument of the 'dot' method. It's essentially the "sum of the products"

y_scores.head()
y_scores.tail()


##From Credit Score to PD
sum_coef_from_score = ((y_scores - min_score) / (max_score - min_score)) * (max_sum_coef - min_sum_coef) + min_sum_coef
# We divide the difference between the scores and the minimum score by
# the difference between the maximum score and the minimum score.
# Then, we multiply that by the difference between the maximum sum of coefficients and the minimum sum of coefficients.
# Then, we add the minimum sum of coefficients.

y_hat_proba_from_score = np.exp(sum_coef_from_score) / (np.exp(sum_coef_from_score) + 1)
# The exponent raised to sum of coefficients from score divided by
# the exponent raised to sum of coefficients from score plus one.
y_hat_proba_from_score.head()

y_hat_test_proba[0: 5]

df_actual_predicted_probs['y_hat_test_proba'].head()

## ----------------------------------------------------------------------
## s8 chp 53: Setting Cut-offs
## ----------------------------------------------------------------------
# We need the confusion matrix again.
# np.where(np.squeeze(np.array(loan_data_targets_test)) == np.where(y_hat_test_proba >= tr, 1, 0), 1, 0).sum() / loan_data_targets_test.shape[0]
threshold = 0.9
df_actual_predicted_probs['y_hat_test'] = np.where(df_actual_predicted_probs['y_hat_test_proba'] > threshold, 1, 0)
#df_actual_predicted_probs['loan_data_targets_test'] == np.where(df_actual_predicted_probs['y_hat_test_proba'] >= threshold, 1, 0)

# Creates a cross-table where the actual values are displayed by rows and the predicted values by columns.
# This table is known as a Confusion Matrix.
pd.crosstab(df_actual_predicted_probs['loan_data_targets_test'], df_actual_predicted_probs['y_hat_test'], rownames = ['Actual'], colnames = ['Predicted'])

# Here we divide each value of the table by the total number of observations,
# thus getting percentages, or, rates.
pd.crosstab(df_actual_predicted_probs['loan_data_targets_test'], df_actual_predicted_probs['y_hat_test'], rownames = ['Actual'], colnames = ['Predicted']) / df_actual_predicted_probs.shape[0]

# Here we calculate Accuracy of the model, which is the sum of the diagonal rates.
(pd.crosstab(df_actual_predicted_probs['loan_data_targets_test'], df_actual_predicted_probs['y_hat_test'], rownames = ['Actual'], colnames = ['Predicted']) / df_actual_predicted_probs.shape[0]).iloc[0, 0] + (pd.crosstab(df_actual_predicted_probs['loan_data_targets_test'], df_actual_predicted_probs['y_hat_test'], rownames = ['Actual'], colnames = ['Predicted']) / df_actual_predicted_probs.shape[0]).iloc[1, 1]

from sklearn.metrics import roc_curve, roc_auc_score

roc_curve(df_actual_predicted_probs['loan_data_targets_test'], df_actual_predicted_probs['y_hat_test_proba'])

fpr, tpr, thresholds = roc_curve(df_actual_predicted_probs['loan_data_targets_test'], df_actual_predicted_probs['y_hat_test_proba'])

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

plt.plot(fpr, tpr)
plt.plot(fpr, fpr, linestyle = '--', color = 'k')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')

thresholds

thresholds.shape

# We concatenate 3 dataframes along the columns.
df_cutoffs = pd.concat([pd.DataFrame(thresholds), pd.DataFrame(fpr), pd.DataFrame(tpr)], axis = 1)

# We rename the columns of the dataframe 'thresholds', 'fpr', and 'tpr'.
df_cutoffs.columns = ['thresholds', 'fpr', 'tpr']


df_cutoffs.head()

df_cutoffs['thresholds'][0] = 1 - 1 / np.power(10, 16)

# Let the first threshold (the value of the thresholds column with index 0) be equal to a number, very close to 1
# but smaller than 1, say 1 - 1 / 10 ^ 16.

df_cutoffs['Score'] = ((np.log(df_cutoffs['thresholds'] / (1 - df_cutoffs['thresholds'])) - min_sum_coef) * ((max_score - min_score) / (max_sum_coef - min_sum_coef)) + min_score).round()
# The score corresponsing to each threshold equals:
# The the difference between the natural logarithm of the ratio of the threshold and 1 minus the threshold and
# the minimum sum of coefficients multiplied by
# the sum of the minimum score and the ratio of the difference between the maximum score and minimum score and
# the difference between the maximum sum of coefficients and the minimum sum of coefficients.

df_cutoffs.head()

df_cutoffs['Score'][0] = max_score

df_cutoffs.head()

df_cutoffs.tail()

# We define a function called 'n_approved' which assigns a value of 1 if a predicted probability
# is greater than the parameter p, which is a threshold, and a value of 0, if it is not.
# Then it sums the column.
# Thus, if given any percentage values, the function will return
# the number of rows(or peeople) wih estimated probabilites greater than the threshold. approved people
def n_approved(p):
    return np.where(df_actual_predicted_probs['y_hat_test_proba'] >= p, 1, 0).sum()

df_cutoffs['N Approved'] = df_cutoffs['thresholds'].apply(n_approved)
# this is a double loop. 1st loop i is going thru the rows of the thresholds,
# for each treshaolds, it loops j at the population of 93,257 is sorted if they are above cut-off, and this iterated number is summed.

# Assuming that all credit applications above a given probability of being 'good' will be approved,
# when we apply the 'n_approved' function to a threshold, it will return the number of approved applications.
# Thus, here we calculate the number of approved appliations for al thresholds.

# Then, we calculate the number of rejected applications for each threshold.
# It is the difference between the total number of applications and the approved applications for that threshold.
df_cutoffs['N Rejected'] = df_actual_predicted_probs['y_hat_test_proba'].shape[0] - df_cutoffs['N Approved']

# Approval rate equalts the ratio of the approved applications and all applications.
df_cutoffs['Approval Rate'] = df_cutoffs['N Approved'] / df_actual_predicted_probs['y_hat_test_proba'].shape[0]

# Rejection rate equals one minus approval rate.
df_cutoffs['Rejection Rate'] = 1 - df_cutoffs['Approval Rate']


df_cutoffs.head()

df_cutoffs.tail()
df_cutoffs[(df_cutoffs['thresholds'] >= 0.88560) and (df_cutoffs['thresholds'] <= 0.88579)]

df_cutoffs[(df_cutoffs['thresholds'] > 0.885675) & (df_cutoffs['thresholds'] < 0.88678)]
df[(df[col] > 0.5) & (df[col] < 0.7)]

# threshold = 0.8857


df_cutoffs.iloc[5000: 6200, ]
# Here we display the dataframe with cutoffs form line with index 5000 to line with index 6200.

df_cutoffs.iloc[1000: 2000, ]
# Here we display the dataframe with cutoffs form line with index 1000 to line with index 2000.

df_cutoffs.to_csv('data/df_cutoffs.csv')

df_scorecard.to_csv('data/df_scorecard.csv')

