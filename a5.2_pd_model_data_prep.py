#Calculate WoE and Information Value for the independent variables (X)

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

# For continuous variable to convert into discrete/categorical, we use:
#    Fine Classsing = split data into bucket ranges, equally spaced interval
#    Coarse Classing = further split it according to knowledge of data (ex. WoE = Weight of Evidence,

#-----------------------------------------
# Splitting Data into Train and Test sets using sklearn
# inputs_train, targets_train / inputs_test, targets_test

from sklearn.model_selection import train_test_split

#method train_test_split(parmeters df inputs, df targets)
#this defaults to 75% train and 25% test. but we want 80% train and 20% test
# set test_size= 0.2 (therefore train is 0.8), random_state = 42 *each time we run train_test_split, sklearn shuffles the deck and therefore give different results. to have stable results, set shuffle to a constant, ex random_state = 42
loan_data_inputs_train, loan_data_inputs_test, loan_data_targets_train, loan_data_targets_test \
    = train_test_split(loan_data.drop('good_bad', axis=1), loan_data['good_bad'], test_size= 0.2, random_state = 42)

#check shape of the df's
# loan_data_inputs_train.shape   # (373028, 207) #Row, Columns
# loan_data_targets_train.shape  #(373028,)
# loan_data_inputs_test.shape    #(93257, 207) /  # loan_data_targets_test.shape   #(93257,)
#-----------------------------------------
# Data Preparation: load Training data: loan_data_inputs_train, loan_data_targets_train
# Prep df to calculate WoE and IV   / # REF: Section 5, video 26 : code sample 5-6
#-----------------------------------------
# create a working df for preprocessing.
df_inputs_train_prep = loan_data_inputs_train
df_targets_train_prep = loan_data_targets_train
# df_inputs_train_prep['grade'].unique()

# name of excel file
file_name = 'temp/df_inputs_train_prep.csv'
df_inputs_train_prep[['id','grade','home_ownership','addr_state']].head()
df_inputs_train_prep.iloc[:10].to_csv(file_name)
df_inputs_train_prep[:20,['id','grade','home_ownership','addr_state']]
df_inputs_train_prep[['id','grade','home_ownership','addr_state']].to_csv(file_name)



#***************************************************************************#
# Ref: S5.27  Data preparation. Preprocessing discrete variables: automating calculations

def woe_discrete(df, discrete_variable_name, df_good_bad_variable):
    # get column from inputs df and good_bad column from Targets df. merge the two df.
    df = pd.concat([df[discrete_variable_name], df_good_bad_variable], axis=1)
    # merge the two results into one df
    df = pd.concat([df.groupby(df.columns.values[0], as_index=False)[df.columns.values[1]].count(),
                     df.groupby(df.columns.values[0], as_index=False)[df.columns.values[1]].mean()], axis=1)
    df = df.iloc[:, [0, 1, 3]]  # drop 2nd field, redundant field.
    # rename columns. Col(0)=as is; col(1)=number of oservations; Ccol(2) proportion of good and bad
    df.columns = [df.columns.values[0], 'n_obs', 'prop_good']
    # 1) get the number of good and bad borrowers by grade group
    df['n_good'] = df['prop_good'] * df['n_obs']
    df['n_bad'] = (1 - df['prop_good']) * df['n_obs']
    # 2)get the proportion of good/bad borrowers for each grade
    df['prop_n_good'] = df['n_good'] / df['n_good'].sum()
    df['prop_n_bad'] = df['n_bad'] / df['n_bad'].sum()
    # calculate WoE for the variable grade. W
    df['WoE'] = np.log(df['prop_n_good'] / df['prop_n_bad'])
    # sort categories with the highest default rate first, then reset index
    df = df.sort_values(['WoE'])
    df = df.reset_index(drop=True)
    df['WoE'].replace(np.inf, np.nan, inplace = True) #replace infinite to NaN
    # calculate Information Value
    df['IV'] = (df['prop_n_good']-df['prop_n_bad']) * df['WoE']
    df['IV'] = df['IV'].sum()
    return df

#***************************************************************************#
#sec.5 L28: Visualizing results
import matplotlib.pyplot as plt
import seaborn as sns
sns.set() #overwrite the default matplotlib look. Think of it like a skin plt

#matplotlib works well with np.array but not df and strings. as well as scipy
def plot_by_WoE(df_WoE, roation_of_axis_labels = 0, width=15, height=7 ):
    x = np.array(df_WoE.iloc[:,0].apply(str)) #Convert values into srting, then convert to an array.
    #x = df_WoE.iloc[:, 0].apply(str)  # Convert values into srting, then convert to an array.
    y  = df_WoE['WoE'] #its a numeric variable so no need to do anything about it.
    plt.figure(figsize=(width,height)) #specify dimension of the chart.  (figsize = (Width(X), Height(Y)
    # now plot the data. Mark o for each point, tied by a dotted line, with color black(k)
    plt.plot(x,y, marker = 'o', linestyle='--', color='k')
    plt.xlabel( df_WoE.columns[0])
    plt.ylabel('Weight of Evidence')
    plt.title(str("Weight of Evidence by " + (df_WoE.columns[0])))
    plt.xticks(rotation = roation_of_axis_labels)
    #plt.show()


#CALL the plot function
# plot_by_WoE(df1_grade)

#***************************************************************************#
# FUNCTION CALL:woe_discrete(dfs, discrete_variable_name, df_good_bad_variable):
# variables (df input data, independent variable(X), dependent variable(Target/Y)

df1_grade = woe_discrete(df_inputs_train_prep,'grade',df_targets_train_prep)

#   df2 = woe_discrete(df_inputs_train_prep,'addr_state',df_targets_train_prep)
#   df3 = woe_discrete(df_inputs_train_prep,'purpose',df_targets_train_prep)

#***************************************************************************#
#summarize indep variables and IF in a table. Create a List and convert to a df
#note: never grow a df! First, accumulate data in a list then create the df
#see: https://www.geeksforgeeks.org/different-ways-to-create-pandas-dataframe/
#   c0 = ['grade','addr_state','purpose']
#   c1 = [df1_grade['IV'].iloc[0],df2['IV'].iloc[0],df3['IV'].iloc[0]]
#   l = list(zip(c0,c1))
#   df_IF = pd.DataFrame(data=l, columns=['ind_var','IF'])
#   df_IF.dtypes
#***************************************************************************#
# name of excel file
#   file_name = 'DataFrameExport.xlsx'
#   # saving tp excel. Note: requires openpyxl package installed
#   df1_grade.to_excel(file_name)
#   print('DataFrame is written to Excel File successfully.')
#
#   file_name2 = 'DataFrameExport.csv'
#   df1_grade.to_csv(file_name2)
#   print('DataFrame is written to CSV File successfully.')


#***************************************************************************#
#sec.5 L29: Data Prep: Preprocessing Discrete Variables. creating dummies part I
# home ownsership independent variable (x)
# code: CreditRiskModelingPreparation5.9.py

df2_home_own = woe_discrete(df_inputs_train_prep,'home_ownership',df_targets_train_prep)
# plot_by_WoE(df2_home_own)

# combine such underrepresented categories that are similar
# Combine, OTHER, NONE, RENT, and ANY (WoE as very low). OWN and MORTGAGE will be in a separate dummy var
df_inputs_train_prep['home_ownership:RENT_OTHER_NONE_ANY'] = sum([df_inputs_train_prep['home_ownership:RENT'], df_inputs_train_prep['home_ownership:OTHER'],
                                                            df_inputs_train_prep['home_ownership:NONE'], df_inputs_train_prep['home_ownership:ANY']])


#sec.5 L30: Data Prep: Preprocessing Discrete Variables. creating dummies part I
# address state - independent variable (x)

# addr_state = df_inputs_train_prep['addr_state'].unique()
# len(addr_state)
# df_addr_sorted = df_inputs_train_prep['addr_state'].sort_values().unique()
# df_addr_sorted = df_inputs_train_prep['addr_state'].unique().sort_values() # does not work. sort first, then unique

df3_addr_st = woe_discrete(df_inputs_train_prep,'addr_state',df_targets_train_prep)

# if column 'addr_state:ND' exist, leave it, else create a new column and set it to 0.
if ['addr_state:ND'] in df_inputs_train_prep.columns.values:
    pass
else:
    df_inputs_train_prep['addr_state:ND'] = 0

# COARSE CLASSING
# visualize data and group together
# plot_by_WoE(df3_addr_st)
# #remove the first and last two elements
# plot_by_WoE(df3_addr_st.iloc[2:-2, :])
# #check the remaining states
# plot_by_WoE(df3_addr_st.iloc[6:-6, :])

#***************************************************************************#
#sec.5 L30: Data Prep: Preprocessing Discrete Variables. creating dummies part II
# code: CreditRiskModelingPreparation5.10.py

# Check shape
# df_inputs_train_prep.shape #(373028, 208)

#creates new column and field with boolean 1 or 0
df_inputs_train_prep['addr_state:ND_NE_IA_NV_FL_HI_AL'] = sum([df_inputs_train_prep['addr_state:ND'], df_inputs_train_prep['addr_state:NE'],
                                                         df_inputs_train_prep['addr_state:IA'], df_inputs_train_prep['addr_state:NV'],
                                                         df_inputs_train_prep['addr_state:FL'], df_inputs_train_prep['addr_state:HI'],
                                                         df_inputs_train_prep['addr_state:AL']])

df_inputs_train_prep['addr_state:NM_VA'] = sum([df_inputs_train_prep['addr_state:NM'], df_inputs_train_prep['addr_state:VA']])

df_inputs_train_prep['addr_state:OK_TN_MO_LA_MD_NC'] = sum([df_inputs_train_prep['addr_state:OK'], df_inputs_train_prep['addr_state:TN'],
                                              df_inputs_train_prep['addr_state:MO'], df_inputs_train_prep['addr_state:LA'],
                                              df_inputs_train_prep['addr_state:MD'], df_inputs_train_prep['addr_state:NC']])

df_inputs_train_prep['addr_state:UT_KY_AZ_NJ'] = sum([df_inputs_train_prep['addr_state:UT'], df_inputs_train_prep['addr_state:KY'],
                                              df_inputs_train_prep['addr_state:AZ'], df_inputs_train_prep['addr_state:NJ']])

df_inputs_train_prep['addr_state:AR_MI_PA_OH_MN'] = sum([df_inputs_train_prep['addr_state:AR'], df_inputs_train_prep['addr_state:MI'],
                                              df_inputs_train_prep['addr_state:PA'], df_inputs_train_prep['addr_state:OH'],
                                              df_inputs_train_prep['addr_state:MN']])

df_inputs_train_prep['addr_state:RI_MA_DE_SD_IN'] = sum([df_inputs_train_prep['addr_state:RI'], df_inputs_train_prep['addr_state:MA'],
                                              df_inputs_train_prep['addr_state:DE'], df_inputs_train_prep['addr_state:SD'],
                                              df_inputs_train_prep['addr_state:IN']])

df_inputs_train_prep['addr_state:GA_WA_OR'] = sum([df_inputs_train_prep['addr_state:GA'], df_inputs_train_prep['addr_state:WA'],
                                              df_inputs_train_prep['addr_state:OR']])

df_inputs_train_prep['addr_state:WI_MT'] = sum([df_inputs_train_prep['addr_state:WI'], df_inputs_train_prep['addr_state:MT']])

df_inputs_train_prep['addr_state:IL_CT'] = sum([df_inputs_train_prep['addr_state:IL'], df_inputs_train_prep['addr_state:CT']])

df_inputs_train_prep['addr_state:KS_SC_CO_VT_AK_MS'] = sum([df_inputs_train_prep['addr_state:KS'], df_inputs_train_prep['addr_state:SC'],
                                              df_inputs_train_prep['addr_state:CO'], df_inputs_train_prep['addr_state:VT'],
                                              df_inputs_train_prep['addr_state:AK'], df_inputs_train_prep['addr_state:MS']])

df_inputs_train_prep['addr_state:WV_NH_WY_DC_ME_ID'] = sum([df_inputs_train_prep['addr_state:WV'], df_inputs_train_prep['addr_state:NH'],
                                              df_inputs_train_prep['addr_state:WY'], df_inputs_train_prep['addr_state:DC'],
                                              df_inputs_train_prep['addr_state:ME'], df_inputs_train_prep['addr_state:ID']])


#***************************************************************************#
# TEMP DELETE THIS
# name of excel file
file_name = 'df_inputs_train_prep.xlsx'
# sheet_name='df_inputs_train_prep'
# saving tp excel. Note: requires openpyxl package installed
# df_inputs_train_prep[['id','grade','home_ownership','addr_state','addr_state:ND','addr_state:NE','addr_state:IA','addr_state:NV','addr_state:FL','addr_state:HI','addr_state:AL','addr_state:ND_NE_IA_NV_FL_HI_AL']].to_excel(file_name)
df_inputs_train_prep[['id','grade','home_ownership','addr_state']]
df_inputs_train_prep.to_excel(file_name)
# df_inputs_train_prep.to_excel(file_name, sheet_name=sheet_name)
print('DataFrame is written to Excel File successfully.')

import os
cwd = os.getcwd()
print(cwd)
