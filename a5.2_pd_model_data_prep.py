# Calculate WoE and Information Value for the independent variables (X)
#
# # For continuous variable to convert into discrete/categorical, we use:
# #    Fine Classsing = split data into bucket ranges, equally spaced interval
# #    Coarse Classing = further split it according to knowledge of data (ex. WoE = Weight of Evidence,

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
sns.set() #overwrite the default matplotlib look. Think of it like a skin plt

#Import data (cleaned) from .FEATHER
file_name_import = 'data/loan_data_2007_2014_clean.feather'
loan_data = pd.read_feather(file_name_import)

print('loan_data.shape = ' + str(loan_data.shape))

#****************************************************************************
# Section 5: Data Preparation for PD model. Chp 25 and up

# Y dependent var preprocessing. Y= a + bX; equation  Yi = f(X, beta) + Ei (error term)
# Create new column (numeric/boolean) based on loan_status. Non-default/good=1 Default/bad=0.
loan_data['good_bad'] = np.where(loan_data['loan_status'].isin(['Charged Off', 'Default',
                                                               'Does not meet the credit policy. Status:Charged Off',
                                                               'Late (31-120 days)']), 0, 1)
# Split Data into Train and Test sets using sklearn
# set test_size= 0.2, therefore 80/20 split train/test. Dfaults is 75/25
# set random_state = 42 (to have stable results,). Each run of train_test_split, sklearn shuffles the deck, therefore give different results.

from sklearn.model_selection import train_test_split

loan_data_inputs_train, loan_data_inputs_test, loan_data_targets_train, loan_data_targets_test \
    = train_test_split(loan_data.drop('good_bad', axis=1), loan_data['good_bad'], test_size= 0.2, random_state = 42)
print('loan_data_inputs_train.shape = ' + str(loan_data_inputs_train.shape))
print('loan_data_inputs_test.shape = ' + str(loan_data_inputs_test.shape))

#-----------------------------------------
# Data Preparation: load Training data: loan_data_inputs_train, loan_data_targets_train
# create df for preprocessing.  calculate WoE and IV   / # REF: Section 5, video 26
df_inputs_train_prep = loan_data_inputs_train
df_targets_train_prep = loan_data_targets_train

#***************************************************************************#
#  Preprocessing DISCRETE variables: automating calculations of WoE for discrete vars. Ref: S5.27
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
# Visualizing results. sec.5 L28:
#matplotlib works well with np.array but not df and strings. Same goes for scipy
def plot_by_WoE(df_WoE, roation_of_axis_labels = 0, width=15, height=7 ):
    x = np.array(df_WoE.iloc[:,0].apply(str)) # Convert values into srting, then convert to an array.
    y  = df_WoE['WoE'] # its a numeric variable so no need to do anything about it.
    plt.figure(figsize=(width,height)) # specify dimension of the chart.  (figsize = (Width(X), Height(Y)

    # now plot the data. Mark o for each point, tied by a dotted line, with color black(k)
    plt.plot(x,y, marker = 'o', linestyle='--', color='k')
    plt.xlabel( df_WoE.columns[0])
    plt.ylabel('Weight of Evidence')
    plt.title(str("Weight of Evidence by " + (df_WoE.columns[0])))
    plt.xticks(rotation = roation_of_axis_labels)

# Calling the plot function example:
# plot_by_WoE(df1_grade)

#***************************************************************************#
# Calculate WoE for the x vars. Call function: woe_discrete(df, discrete_var_name(x), df_target_var(Y))

df1_grade = woe_discrete(df_inputs_train_prep,'grade',df_targets_train_prep)
df2_home_own = woe_discrete(df_inputs_train_prep,'home_ownership',df_targets_train_prep)
df3_addr_st = woe_discrete(df_inputs_train_prep,'addr_state',df_targets_train_prep)

#***************************************************************************#
# Creating dummies part I; Sec.5 L29: Data Prep: Preprocessing Discrete Variables.

# COARSE CLASSING
# visualize data and group/combine together categories that are similar or underrepresented (by the counts)
# ex. plot_by_WoE(df3_addr_st)

# df1_grade : no combining needed
# df2_home_own : Combine, OTHER, NONE, RENT, and ANY (WoE as very low). OWN and MORTGAGE will be in a separate dummy var
df_inputs_train_prep['home_ownership:RENT_OTHER_NONE_ANY'] = sum([df_inputs_train_prep['home_ownership:RENT'], df_inputs_train_prep['home_ownership:OTHER'],
                                                            df_inputs_train_prep['home_ownership:NONE'], df_inputs_train_prep['home_ownership:ANY']])

# df3_addr_st :
# if column 'addr_state:ND' exist, leave it, else create a new column and set it to 0.
if ['addr_state:ND'] in df_inputs_train_prep.columns.values:
    pass
else:
    df_inputs_train_prep['addr_state:ND'] = 0

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
## TEMP, DELETE THIS

# Visualize DataFrame to CSV
dataframe_name = 'df_inputs_train_prep'
file_name = 'temp/df_inputs_train_prep.csv'

# Try Except: https://www.w3schools.com/python/python_try_except.asp
try:
    # start from column HA = good_bad
    df_inputs_train_prep[:51].to_csv(file_name)
except Exception as e:
    print('exec loan_data.to_csv(file_name); error = ' + e)
else:
    print(f'DataFrame {dataframe_name} is written to CSV File = {file_name} successfully.')


# Visualize x vars and information values (IV)
#summarize indep variables and IV in a table. Create a List and convert to a df
#note: never grow a df! First, accumulate data in a list then create the df
#see: https://www.geeksforgeeks.org/different-ways-to-create-pandas-dataframe/
#   c0 = ['grade','addr_state','purpose']
#   c1 = [df1_grade['IV'].iloc[0],df2['IV'].iloc[0],df3['IV'].iloc[0]]
#   l = list(zip(c0,c1))
#   df_IV = pd.DataFrame(data=l, columns=['ind_var','IV'])
#   df_IV.dtypes
#***************************************************************************#

df_inputs_train_prep.iloc[:51,(0:2, 206:)].to_csv(file_name)
df_inputs_train_prep.iloc[:51, 205:]

df_tmp = pd.concat([df_inputs_train_prep.iloc[:51,1:4], df_inputs_train_prep.iloc[:52,206:]], axis=1)
