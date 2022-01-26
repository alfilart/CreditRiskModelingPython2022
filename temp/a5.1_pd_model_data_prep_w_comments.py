
import numpy as np
import pandas as pd
#set option to display all columns
pd.set_option('display.max_columns', None)
# to go back to default value  pd.reset_option(“max_columns”)

file_path = r'C:\Users\alfil\iCloudDrive\Documents\02.2 Learning Python\DataSets\LendingClubLoanData'
file_name_import = r'\loan_data_2007_2014_clean.csv'
loan_data = pd.read_csv(file_path + file_name_import, low_memory=False)

# Data Exploration
# See script a1.1 data exploration.py

#****************************************
# Data Preparation for PD model  . Section 5, video 25 and up
#****************************************

#Dependent variable preprocessing. Non-default/good=1 Default/bad=0
#create new column (numeric/boolean) based on loan_status
loan_data['good_bad'] = np.where(loan_data['loan_status'].isin(['Charged Off', 'Default',
                                                               'Does not meet the credit policy. Status:Charged Off',
                                                               'Late (31-120 days)']), 0, 1)

""" 
For continuous variable to convert into discrete/categorical, we use:
    Fine Classsing = split data into bucket ranges, equally spaced interval
    Coarse Classing = further split it according to knowledge of data (ex. WoE = Weight of Evidence, 
"""

#-----------------------------------------
# Splitting Data into Train and Test sets using sklearn
# inputs_train, targets_train / inputs_test, targets_test
# REF: Section 5, video 26 : code sample 5-5
#-----------------------------------------

from sklearn.model_selection import train_test_split

#method train_test_split(parmeters df inputs, df targets)
#store output into 4 df's. inputs train and test, targets train and test.
#this defaults to 75% train and 25% test. but we want 80% train and 20% test
# set test_size= 0.2 (therefore train is 0.8), random_state = 42 *each time we run train_test_split, sklearn shuffles the deck and therefore give different results. to have stable results, set shuffle to a constant, ex random_state = 42
loan_data_inputs_train, loan_data_inputs_test, loan_data_targets_train, loan_data_targets_test \
    = train_test_split(loan_data.drop('good_bad', axis=1), loan_data['good_bad'], test_size= 0.2, random_state = 42)

#check shape of the df's
# loan_data_inputs_train.shape   # (373028, 207) #Row, Columns
# loan_data_targets_train.shape  #(373028,)
# loan_data_inputs_test.shape    #(93257, 207)
# loan_data_targets_test.shape   #(93257,)
#-----------------------------------------
# Data Preparation:
# for Training data: loan_data_inputs_train, loan_data_targets_train
# REF: Section 5, video 26 : code sample 5-6
#-----------------------------------------

# create a working df for preprocessing.
# we will check WoE for each variable and see it's explanatory power
df_inputs_train_prep = loan_data_inputs_train
df_targets_train_prep = loan_data_targets_train

df_inputs_train_prep['grade'].unique()
#Put side by side, Grade column from inputs df and good_bad from Targets df
# merge the two dataframes pd.concat(). axis = 1 means we take 2nd column (base 0) of the right df.
df1 = pd.concat([df_inputs_train_prep['grade'], df_targets_train_prep], axis=1)
df1.head()
df1.describe()
df1[['grade','good_bad']].groupby(['grade']).agg(['mean', 'count'])
#data preview mneum = hitd

# 332250/373028 = 0.890683

# groupby indexes results, we don't need it as it's limiting
# get counts for each Grade. groupby grade, summarize count.
# note: df1.columns.values[0] = first field 'grade' string & df1.columns.values[1] = 2nd field
#df1.groupby(df1.columns.values[0],as_index=False).count() #shortcut
df1.groupby(df1.columns.values[0],as_index=False)[df1.columns.values[1]].count() #explicit. If there are several columns
# and you want particular colummn

#get proportion of good/bad in each group
# get mean for each Grade. will only show avg of good borrowers as borrowers in default is 0
# df1.groupby(df1.columns.values[0],as_index=False).mean() #shortcut
df1.groupby(df1.columns.values[0],as_index=False)[df1.columns.values[1]].mean()

# merge the two results into one df
df1 = pd.concat([df1.groupby(df1.columns.values[0], as_index=False)[df1.columns.values[1]].count(), df1.groupby(df1.columns.values[0], as_index=False)[df1.columns.values[1]].mean()], axis=1)
df1 = df1.iloc[:, [0, 1, 3]] # drop 2nd field, redundant field.

#rename columns
# column(0)=as is; column(1)=number of oservations; (2) proportion of good and bad
df1.columns = [df1.columns.values[0], 'n_obs', 'prop_good']
# df1['n_obs']  ;  df1['n_obs'].sum()

#proportion of the number of good borrowers per group
#? no need to calculate, not used?!
df1['prop_n_obs'] = df1['n_obs'] / df1['n_obs'].sum()

#1) get the number of good and bad borrowers by grade group
df1['n_good'] = df1['prop_good'] * df1['n_obs']
df1['n_bad'] = (1 - df1['prop_good']) * df1['n_obs']

#2)get the proportion of good/bad borrowers for each grade
df1['prop_n_good'] = df1['n_good'] / df1['n_good'].sum()
df1['prop_n_bad'] = df1['n_bad'] / df1['n_bad'].sum()

#above, we have all we need to calculate WoE for the variable grade
df1['WoE'] = np.log(df1['prop_n_good'] / df1['prop_n_bad'])

# sort categories with the highest default rate first
df1 = df1.sort_values(['WoE'])
df1 = df1.reset_index(drop=True)

#Optional
#calculate subsequent categories and the difference of WoE between the two subsequent cat.
#method diff(), the current row is substracted from the previous row. result stored in current row
df1['diff_prop_good'] = df1['prop_good'].diff().abs()
df1['diff_WoE'] = df1['WoE'].diff().abs()

# finally, calculate the information value
df1['inf_val'] = (df1['prop_n_good'] - df1['prop_n_bad']) * df1['WoE']
#summation in the equation
df1['inf_val'] = df1['inf_val'].sum()
# inf_val is the same for all rows, because this metric refers to the IV of the
# variable "Grade" overall.

