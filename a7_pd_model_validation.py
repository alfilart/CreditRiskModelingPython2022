## SECTION 7. PD Model Validation

## Import Libraries
import numpy as np
import pandas as pd
import time  # time the run
from datetime import timedelta
import pickle

### 1) Load the Data and Select the Features/Variables ********************************

### Import Data
loan_data_inputs_train = pd.read_csv('data/loan_data_inputs_train.csv', index_col=0)
loan_data_targets_train = pd.read_csv('data/loan_data_targets_train.csv', index_col=0, header=None)
loan_data_inputs_test = pd.read_csv('data/loan_data_inputs_test.csv', index_col=0)
loan_data_targets_test = pd.read_csv('data/loan_data_targets_test.csv', index_col=0, header=None)

### Explore Data
# loan_data_inputs_train.head()
# loan_data_targets_train.head()
#
# loan_data_inputs_train.shape
# loan_data_targets_train.shape
#
# loan_data_inputs_test.shape
# loan_data_targets_test.shape

### Selecting the Features
# Here we select a limited set of input variables in a new dataframe.
# We are going to remove some features, the coefficients for all or almost all of the dummy variables for which, are not tatistically significant.

# We do that by specifying another list of dummy variables as reference categories, and a list of variables to remove.
# Then, we are going to drop the two datasets from the original list of dummy variables.

# Variables List
categories_with_ref_list = ['grade:A',
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
                            'mths_since_last_record:>86',]

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

# Variables INPUT


# Pre-process TRAINING inputs and targets
inputs_train_with_ref_cat = loan_data_inputs_train.loc[:, categories_with_ref_list]
inputs_train = inputs_train_with_ref_cat.drop(ref_categories_list, axis=1) # drop reference categories

targets_train = loan_data_targets_train.to_numpy()
targets_train = targets_train.ravel()

# Pre-process TEST inputs and targets
inputs_test_with_ref_cat = loan_data_inputs_test.loc[:, categories_with_ref_list]
inputs_test = inputs_test_with_ref_cat.drop(ref_categories_list, axis=1) # drop reference categories

targets_test = loan_data_targets_test.to_numpy()
targets_test = targets_test.ravel()

# drop unused variables (list and df's)
del categories_with_ref_list
del ref_categories_list
del inputs_train_with_ref_cat
del inputs_test_with_ref_cat
del loan_data_inputs_train
del loan_data_targets_train
del loan_data_inputs_test
del loan_data_targets_test

### 2) Build a Logistic Regression Model with P-Values ********************************
# P values for sklearn logistic regression.

# 2.A) Class to display p-values for logistic regression in sklearn.
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
        # self.z_scores = z_scores
        self.p_values = p_values
        # self.sigma_estimates = sigma_estimates
        # self.F_ij = F_ij


# We create an instance of an object from the newly created 'LogisticRegression_with_p_values()' class.
# reg = LogisticRegression_with_p_values()
reg = LogisticRegression_with_p_values(max_iter=1000, solver='lbfgs') # , n_jobs=-1)

# Estimates the coefficients of the object from the 'LogisticRegression' class
# with inputs (independent variables) contained in the first dataframe and targets (dependent variables) contained in the second dataframe.

start_time = time.time()
reg.fit(inputs_train, targets_train)
elapsed_time_secs = time.time() - start_time
msg = "reg.fit - Execution took: %s secs (Wall clock time)" % timedelta(seconds=round(elapsed_time_secs))
print(msg)


# Create summary for coefficients of the log reg model

feature_name = inputs_train.columns.values # Stores the names of the columns of df (the variables)
summary_table = pd.DataFrame(columns=['Feature name'], data=feature_name)
summary_table['Coefficients'] = np.transpose(reg.coef_) # create coefficient column w reg obj coef transposed
summary_table.index = summary_table.index + 1 # add row for intercept
summary_table.loc[0] = ['Intercept', reg.intercept_[0]]
summary_table = summary_table.sort_index()
print(summary_table)

## Add p-values to the summary table
# This is a list. # We take the result of the newly added method 'p_values' and store it in a variable 'p_values'.
p_values = reg.p_values

# Add the intercept for completeness. # We add the value 'NaN' in the beginning of the variable with p-values.
p_values = np.append(np.nan, np.array(p_values))

# In the 'summary_table' dataframe, we add a new column, called 'p_values', containing the values from the 'p_values' variable.
summary_table['p_values'] = p_values
print(summary_table)

# clean up unused variables
del feature_name
del p_values


# Here, we get the results for our final PD model. we export our model to a 'SAV' file with file name 'pd_model.sav'. wb = write binary
pickle.dump(reg, open('data/pd_model.sav', 'wb'))

## ********************************************************************************
## 3 ) PD Model Validation (Test)
## ********************************************************************************

# NOTE: if starting from this point
#  re-run Part 1) Load the Data and Select the Features/Variables
#  re-run # 2.A) Class to display p-values for logistic regression in sklearn.

## ----------------------------------------------------------------------
## Part 3.A) Load Model and predict y (y_hat). Get actual predicted class and probabilities (of 1=good)
## ----------------------------------------------------------------------

# Load model from disk. rb = read binary
reg = pickle.load(open('data/pd_model.sav', 'rb'))

### Out-of-sample validation (test)
# Here, from the dataframe with inputs for testing, we keep the same variables that we used in our final PD model.
# inputs_test.head()

# Calculates the predicted values for the dependent variable (targets)
# based on the values of the independent variables (inputs) supplied as an argument.
y_hat_test = reg.model.predict(inputs_test)

# This is an array of predicted discrete classess (in this case, 0s and 1s).
print(y_hat_test)

# Calculates the predicted probability values for the dependent variable (targets)
# based on the values of the independent variables (inputs) supplied as an argument.
y_hat_test_proba = reg.model.predict_proba(inputs_test)

# This is an array of arrays of predicted class probabilities for all classes.
# In this case, the first value of every sub-array is the probability for the observation to belong to the first class, i.e. 0=default=bad
# and the second value is the probability for the observation to belong to the first class, i.e. 1=not default=good
print(y_hat_test_proba)

# Here we take all the arrays in the array, and from each array, we take all rows, and only the element with index 1, that is, the second element.
# In other words, we take only the probabilities for being 1=not default=good
# We store these probabilities in a variable.
y_hat_test_proba = y_hat_test_proba[:][:, 1]

# This variable contains an array of probabilities of being 1.
print(y_hat_test_proba)

# delete this part ?
# loan_data_targets_test = pd.read_csv('data/loan_data_targets_test.csv', index_col=0, header=None)
# loan_data_targets_test_temp = loan_data_targets_test
#
# We reset the index of a dataframe to match up with y_hat_test_proba
# loan_data_targets_test_temp.reset_index(drop=True, inplace=True)

# Concatenates two dataframes.
df_actual_predicted_probs = pd.concat([pd.DataFrame(targets_test), pd.DataFrame(y_hat_test_proba)], axis=1)

df_actual_predicted_probs.shape  # (93257, 2)

df_actual_predicted_probs.columns = ['loan_data_targets_test', 'y_hat_test_proba']

# Makes the index of one dataframe equal to the index of another dataframe.
df_actual_predicted_probs.index = inputs_test.index

df_actual_predicted_probs.head()

# delete unused variables
del y_hat_test
del y_hat_test_proba
## ----------------------------------------------------------------------
## Part 3.B) Get Accuracy and Area under the Curve. Visualize via graph
## ----------------------------------------------------------------------
# Set threshold; cut-off for the category to be classified as 1=Good

# We create a new column with an indicator, where every observation that has predicted probability
# greater than the threshold has a value of 1, and every observation that has predicted probability
# lower than the threshold has a value of 0.
threshold = 0.8857 #optima threshold
df_actual_predicted_probs['y_hat_test'] = np.where(df_actual_predicted_probs['y_hat_test_proba'] > threshold, 1, 0)

# Creates a cross-table where the actual values are displayed by rows and the predicted values by columns.
# This table is known as a Confusion Matrix.
conf_mtrx_obs = pd.crosstab(df_actual_predicted_probs['loan_data_targets_test'], df_actual_predicted_probs['y_hat_test'],
            rownames=['Actual'], colnames=['Predicted'])

# Here we divide each value of the table by the total number of observations,
# thus getting percentages, or, rates.
pd.crosstab(df_actual_predicted_probs['loan_data_targets_test'], df_actual_predicted_probs['y_hat_test'],
            rownames=['Actual'], colnames=['Predicted']) / df_actual_predicted_probs.shape[0]

# Here we calculate Accuracy of the model, which is the sum of the diagonal rates.
# (pd.crosstab(df_actual_predicted_probs['loan_data_targets_test'], df_actual_predicted_probs['y_hat_test'],
#              rownames=['Actual'], colnames=['Predicted']) / df_actual_predicted_probs.shape[0]).iloc[0, 0] + (
#             pd.crosstab(df_actual_predicted_probs['loan_data_targets_test'], df_actual_predicted_probs['y_hat_test'],
#                         rownames=['Actual'], colnames=['Predicted']) / df_actual_predicted_probs.shape[0]).iloc[1, 1]

conf_mtrx_pct = pd.crosstab(df_actual_predicted_probs['loan_data_targets_test'], df_actual_predicted_probs['y_hat_test'],
            rownames=['Actual'], colnames=['Predicted']) / df_actual_predicted_probs.shape[0]

# ? Model Accuracy= True Neg + True Pos

# --------------------------------------
# another way of generating Conf. Matrix via sklearn
# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.ConfusionMatrixDisplay.html
# --------------------------------------
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix

# Create the CM
# cm = confusion_matrix(y_true, y_pred)
cm = confusion_matrix(df_actual_predicted_probs['loan_data_targets_test'], df_actual_predicted_probs['y_hat_test'])
#    0	 1
# 0	TN	FN
# 1	FP	TP

# a) original
cm_disp = ConfusionMatrixDisplay(cm, display_labels=['Negative', 'Positive'])
# cm_disp = ConfusionMatrixDisplay(cm)  # display as 0 and 1
plt.xlabel('Actual')
plt.ylabel('Predicted')
cm_disp.plot()

# b) To see first cell as TP. Flip it using Numpy and feed it to the display function
cm2 =  np.flip(cm, (0, 1))
cm_disp2 = ConfusionMatrixDisplay(cm2, display_labels=['Positive', 'Negative'])
plt.xlabel('Actual')
plt.ylabel('Predicted')
cm_disp2.plot()

# Set variables to be used to calculate Evaluation Metrics
tp = cm[1,1]
tn = cm[0,0]
fp = cm[0,1]
fn = cm[1,0]

# Evaluation Metrics
# https://classeval.wordpress.com/introduction/basic-evaluation-measures/#:~:text=False%20positive%20rate%20(FPR)%20is,be%20calculated%20as%201%20%E2%80%93%20specificity.
# 1) Accuracy
m_accuracy = (tp + tn)/(tp + fp + fn + tn)
# 2) Sensitivity or Recall or TPR (True positive rate)
m_recall = tp/(tp + fn)
# 3) Specificity or TNR (True Negative Rate)
m_specificity = tn/(fp + tn)
# 4) Precision  (Positive predictive value)
m_precision = tp/(tp + fp)
# 5) FPR or False positive rate
m_fpr = fp/(tn + fp)
# 5) F1 - Score
m_F1_score = 2*((m_precision * m_accuracy)/(m_precision + m_accuracy))
# Other notes: FP (type I error), FN (type II error)
print('Threshold: {}'.format(round(threshold,4)))
print('Model Accuracy: {}'.format(round(m_accuracy, 4)))
print('Recall or TPR: {}; Specificity or TNR: {}'.format(round(m_recall, 4), round(m_specificity, 4)))
print('Model Precision: {}, F1_score: {}'.format(round(m_precision, 4), round(m_F1_score, 4)))
print('Confusion Matrix: {}'.format(cm))

## ----------------------------------------------------------------------
# the Receiver Operating Characteristic (ROC) Curve S7.Chp 46
## ----------------------------------------------------------------------

from sklearn.metrics import roc_curve, roc_auc_score

# Returns the Receiver Operating Characteristic (ROC) Curve from a set of actual values and their predicted probabilities.
# As a result, we get three arrays: the false positive rates, the true positive rates, and the thresholds.
roc_curve(df_actual_predicted_probs['loan_data_targets_test'], df_actual_predicted_probs['y_hat_test_proba'])

# Here we store each of the three arrays in a separate variable.
fpr, tpr, thresholds = roc_curve(df_actual_predicted_probs['loan_data_targets_test'],
                                 df_actual_predicted_probs['y_hat_test_proba'])

# to convert the arrays into df, for later use like plotting
df_roc_curve = pd.DataFrame({'FPR':fpr, 'TPR':tpr, 'Threshold':thresholds})

# find row where threshold is
df_roc_curve[df_roc_curve['Threshold'].round(4) == threshold]

df_roc_curve[(df_roc_curve['TPR'] >= round(m_recall-0.0001,6)) & (df_roc_curve['TPR'] <= round(m_recall+0.0001,6))]

#index FPR TPR Threshold (orig param = 0.8857
# 6643  0.348773  0.645214   0.885705

# Plot ROC Curve
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

plt.plot(fpr, tpr, linestyle='-', color='b')
# We plot the false positive rate along the x-axis and the true positive rate along the y-axis,
# thus plotting the ROC curve.
plt.plot(fpr, fpr, linestyle='--', color='k')
# We plot a seconary diagonal line, with dashed line style and black color.
# The diagnol straight line can be created by providng the same data for both x and y (ex. fpr, fpr)
x = 0.348773
y = 0.645214
# Mark single point. https://matplotlib.org/stable/api/markers_api.html
plt.plot(0.348773, 0.645214,  marker="o", markersize=10, markeredgecolor="red", markerfacecolor="green")
plt.annotate("Optimal Threshold = 0.885705", (x + .05, y -.01))
plt.xlabel('False positive rate')
# We name the x-axis "False positive rate".
plt.ylabel('True positive rate')
# We name the x-axis "True positive rate".
plt.title('ROC curve')
# We name the graph "ROC curve".
plt.show()

# --------------------------------------------
# AUROC
# --------------------------------------------
AUROC = roc_auc_score(df_actual_predicted_probs['loan_data_targets_test'],
                      df_actual_predicted_probs['y_hat_test_proba'])
# Calculates the Area Under the Receiver Operating Characteristic Curve (AUROC)
# from a set of actual values and their predicted probabilities.
AUROC


# --------------------------------------------
# Get the optimal threshold
# G-mean method
# --------------------------------------------

# Calculate the G-mean and store as array
gmean = np.sqrt(tpr * (1 - fpr))

# Find the optimal threshold. Where the gmean is maximum
# argmax= Returns the indices of the maximum values along an axis.
index = np.argmax(gmean)
thresholdOpt = round(thresholds[index], ndigits = 4)  # 0.8857
gmeanOpt = round(gmean[index], ndigits = 4)
fprOpt = round(fpr[index], ndigits = 4)
tprOpt = round(tpr[index], ndigits = 4)
print('Best Threshold: {} with G-Mean: {}'.format(thresholdOpt, gmeanOpt))
print('FPR: {}, TPR: {}'.format(fprOpt, tprOpt))

# to convert arrays into DF for further analysis
# df_auroc_values = pd.DataFrame({'fpr': fpr,'tpr':  tpr, 'thresholds': thresholds})


## ----------------------------------------------------------------------
# Gini and Kolmogorov-Smirnov S7.Chp 47
## ----------------------------------------------------------------------

### Gini and Kolmogorov-Smirnov
# To measure, the prerequisite is to sort the predicted probabilities (y_hat_test_probab) in ascending order (small to big).
# To calculate the cumulative probabilities, we need to reset index to zero. The lowest probability will have an index of 0 (zero)
# To make plots of our model performance criteria, we need
#   1) The cumulative percentage % of the total population
#   2) The cumulative percentage % of good borrowers
#   3) The cumulative percentage % of bad borrowers.

df_actual_predicted_probs = df_actual_predicted_probs.sort_values('y_hat_test_proba')
# Sorts a dataframe by the values of a specific column.

# df_actual_predicted_probs.head()
# df_actual_predicted_probs.tail()

df_actual_predicted_probs = df_actual_predicted_probs.reset_index()
# We reset the index of a dataframe and overwrite it.

# df_actual_predicted_probs.head()

df_actual_predicted_probs['Cumulative N Population'] = df_actual_predicted_probs.index + 1
# We calculate the cumulative number of all observations.
# We use the new index for that. Since indexing in ython starts from 0, we add 1 to each index.
df_actual_predicted_probs['Cumulative N Good'] = df_actual_predicted_probs['loan_data_targets_test'].cumsum()
# We calculate cumulative number of 'good', which is the cumulative sum of the column with actual observations.
df_actual_predicted_probs['Cumulative N Bad'] = df_actual_predicted_probs['Cumulative N Population'] - \
                                                df_actual_predicted_probs['loan_data_targets_test'].cumsum()
# We calculate cumulative number of 'bad', which is the difference between the cumulative number of all observations
# and cumulative number of 'good' for each row.

df_actual_predicted_probs.head()

df_actual_predicted_probs['Cumulative Perc Population'] = df_actual_predicted_probs['Cumulative N Population'] / (
df_actual_predicted_probs.shape[0])
# We calculate the cumulative percentage of all observations.
df_actual_predicted_probs['Cumulative Perc Good'] = df_actual_predicted_probs['Cumulative N Good'] / \
                                                    df_actual_predicted_probs['loan_data_targets_test'].sum()
# We calculate cumulative percentage of 'good'.
df_actual_predicted_probs['Cumulative Perc Bad'] = df_actual_predicted_probs['Cumulative N Bad'] / (
            df_actual_predicted_probs.shape[0] - df_actual_predicted_probs['loan_data_targets_test'].sum())
# We calculate the cumulative percentage of 'bad'.

# df_actual_predicted_probs.head()
# df_actual_predicted_probs.tail()

# Plot Gini
plt.plot(df_actual_predicted_probs['Cumulative Perc Population'], df_actual_predicted_probs['Cumulative Perc Bad'])
# We plot the cumulative percentage of all along the x-axis and the cumulative percentage 'good' along the y-axis,
# thus plotting the Gini curve.
plt.plot(df_actual_predicted_probs['Cumulative Perc Population'],
         df_actual_predicted_probs['Cumulative Perc Population'], linestyle='--', color='k')
# We plot a seconary diagonal line, with dashed line style and black color.
plt.xlabel('Cumulative % Population')
# We name the x-axis "Cumulative % Population".
plt.ylabel('Cumulative % Bad')
# We name the y-axis "Cumulative % Bad".
plt.title('Gini')
# We name the graph "Gini".

Gini = AUROC * 2 - 1
# Here we calculate Gini from AUROC.
Gini

# Plot KS
plt.plot(df_actual_predicted_probs['y_hat_test_proba'], df_actual_predicted_probs['Cumulative Perc Bad'], color='r')
# We plot the predicted (estimated) probabilities along the x-axis and the cumulative percentage 'bad' along the y-axis,
# colored in red.
plt.plot(df_actual_predicted_probs['y_hat_test_proba'], df_actual_predicted_probs['Cumulative Perc Good'], color='b')
# We plot the predicted (estimated) probabilities along the x-axis and the cumulative percentage 'good' along the y-axis,
# colored in red.
plt.xlabel('Estimated Probability for being Good')
# We name the x-axis "Estimated Probability for being Good".
plt.ylabel('Cumulative %')
# We name the y-axis "Cumulative %".
plt.title('Kolmogorov-Smirnov')
# We name the graph "Kolmogorov-Smirnov".

KS = max(df_actual_predicted_probs['Cumulative Perc Bad'] - df_actual_predicted_probs['Cumulative Perc Good'])
# We calculate KS from the data. It is the maximum of the difference between the cumulative percentage of 'bad'
# and the cumulative percentage of 'good'.
KS


