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
    # merge the two results into one df. note: as_index=False means group by the column names, and not by the index, =False,is same as SQL group by
    # aggregate mean() is the mean of true=1=good values. therefore, the mean is the percentage of Good values ex. 88/100 = 88%
    df = pd.concat([df.groupby(df.columns.values[0], as_index=False)[df.columns.values[1]].count(),
                     df.groupby(df.columns.values[0], as_index=False)[df.columns.values[1]].mean()], axis=1)
    df = df.iloc[:, [0, 1, 3]]  # drop 2nd field, redundant field.
    # rename columns. Col(0)=as is; col(1)=number of oservations; Ccol(2) proportion of good and bad
    df.columns = [df.columns.values[0], 'n_obs', 'prop_good']
    # get percentage of observations n over total N. Insert after n_obs
    df.insert(2, 'prop_n_obs', df['n_obs'] / df['n_obs'].sum())
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
    df = df.reset_index(drop=True) # here, there's no groupby so we reset the index afer df sort.
    df['WoE'].replace(np.inf, np.nan, inplace = True) #replace infinite to NaN
    # calculate Information Value
    df['IV'] = (df['prop_n_good']-df['prop_n_bad']) * df['WoE']
    df['IV'] = df['IV'].sum()
    return df

#***************************************************************************#
#  Preprocessing CONTINOUS variables: automating calculations of WoE for discrete vars. Ref: S5.27
def woe_ordered_continuous(df, discrete_variable_name, df_good_bad_variable):
    # get column from inputs df and good_bad column from Targets df. merge the two df.
    df = pd.concat([df[discrete_variable_name], df_good_bad_variable], axis=1)
    # merge the two results into one df. note: as_index=False means group by the column names, and not by the index, =False,is same as SQL group by
    # aggregate mean() is the mean of true=1=good values. therefore, the mean is the percentage of Good values ex. 88/100 = 88%
    df = pd.concat([df.groupby(df.columns.values[0], as_index=False)[df.columns.values[1]].count(),
                     df.groupby(df.columns.values[0], as_index=False)[df.columns.values[1]].mean()], axis=1)
    df = df.iloc[:, [0, 1, 3]]  # drop 2nd field, redundant field.
    # rename columns. Col(0)=as is; col(1)=number of oservations; Ccol(2) proportion of good and bad
    df.columns = [df.columns.values[0], 'n_obs', 'prop_good']
    # get percentage of observations n over total N. Insert after n_obs
    df.insert(1, 'prop_n_obs', df['n_obs'] / df['n_obs'].sum())
    # 1) get the number of good and bad borrowers by grade group
    df['n_good'] = df['prop_good'] * df['n_obs']
    df['n_bad'] = (1 - df['prop_good']) * df['n_obs']
    # 2)get the proportion of good/bad borrowers for each grade
    df['prop_n_good'] = df['n_good'] / df['n_good'].sum()
    df['prop_n_bad'] = df['n_bad'] / df['n_bad'].sum()
    # calculate WoE for the variable grade. W
    df['WoE'] = np.log(df['prop_n_good'] / df['prop_n_bad'])
    # In continous, we remove sort by WoE and keep the natural sort of the input df
    df['WoE'].replace(np.inf, np.nan, inplace = True) #replace infinite to NaN
    # calculate Information Value
    df['IV'] = (df['prop_n_good']-df['prop_n_bad']) * df['WoE']
    df['IV'] = df['IV'].sum()
    return df


def append_IV_list(df, lst):
    # if lst is None:
    #     lst = []
    val = [df_temp.columns[0], df_temp.iloc[0, 9]]
    lst.append(val)
    return lst

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
# COARSE CLASSING (DISCRETE VARIABLES)
# Creating dummies part I; Sec.5 L29: Data Prep: Preprocessing Discrete Variables.
# 1. Calculate WoE for the x vars. Call function: woe_discrete(df, discrete_var_name(x), df_target_var(Y))
# 2. Visualize data with func: plot_by_WoE(df_w_WoE) ;
# 3. Coarse class by grouping similar WoE vars and creating new dummy columns. As each column will only have one value wit 1, this can be grouped by Summing
# 4  Decide the reference catgegory and include names of dummy vars and reference dummy vars in a list in excel
# In general, you have to put the categories with similar weight of evidence in one and the same category (dummy variable). However, sometimes, other considerations may play a role, such as how large the initial categories are, or their meaning, etc.


# ---- df1_grade : no combining needed ---
df_temp = woe_discrete(df_inputs_train_prep,'grade',df_targets_train_prep)
# plot_by_WoE(df_temp)

# Collect variable names and IV values to outputed to excel in the end for analysis of IV
lst_IV = [] # initialize list_IV
lst_IV = append_IV_list(df_temp, lst_IV)

# --- df2_home_own : Combine, OTHER, NONE, RENT, and ANY (WoE as very low). OWN and MORTGAGE will be in a separate dummy var
df_temp = woe_discrete(df_inputs_train_prep,'home_ownership',df_targets_train_prep)
# plot_by_WoE(df2_home_own)
lst_IV = append_IV_list(df_temp, lst_IV)

df_inputs_train_prep['home_ownership:RENT_OTHER_NONE_ANY'] = sum([df_inputs_train_prep['home_ownership:RENT'], df_inputs_train_prep['home_ownership:OTHER'],
                                                            df_inputs_train_prep['home_ownership:NONE'], df_inputs_train_prep['home_ownership:ANY']])

# ---- df3_addr_st : if column 'addr_state:ND' exist, leave it, else create a new column and set it to 0.
df_temp = woe_discrete(df_inputs_train_prep,'addr_state',df_targets_train_prep)
# plot_by_WoE(df3_addr_st)
lst_IV = append_IV_list(df_temp, lst_IV)

if ['addr_state:ND'] in df_inputs_train_prep.columns.values:  # handle missing values
    pass
else:
    df_inputs_train_prep['addr_state:ND'] = 0

# Note: we can use Sum() as all will have 0 values, except 1 (it's exclusive) therefore, it will always sum to only 1
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

# Sec 5. chp 31 HW
# df4_verification_status :
df_temp = woe_discrete(df_inputs_train_prep, 'verification_status', df_targets_train_prep)
# plot_by_WoE(df4_verification_status)
lst_IV = append_IV_list(df_temp, lst_IV)

# df5_purpose : combine
df_temp = woe_discrete(df_inputs_train_prep, 'purpose', df_targets_train_prep)
# plot_by_WoE(df5_purpose)
df_inputs_train_prep['purpose:educ__sm_b__wedd__ren_en__mov__house'] = sum([df_inputs_train_prep['purpose:educational'], df_inputs_train_prep['purpose:small_business'],
                                                                 df_inputs_train_prep['purpose:wedding'], df_inputs_train_prep['purpose:renewable_energy'],
                                                                 df_inputs_train_prep['purpose:moving'], df_inputs_train_prep['purpose:house']])
df_inputs_train_prep['purpose:oth__med__vacation'] = sum([df_inputs_train_prep['purpose:other'], df_inputs_train_prep['purpose:medical'],
                                             df_inputs_train_prep['purpose:vacation']])
df_inputs_train_prep['purpose:major_purch__car__home_impr'] = sum([df_inputs_train_prep['purpose:major_purchase'], df_inputs_train_prep['purpose:car'],
                                                        df_inputs_train_prep['purpose:home_improvement']])


# df5_initial_list_status : no combining needed
df_temp = woe_discrete(df_inputs_train_prep, 'initial_list_status', df_targets_train_prep)
# plot_by_WoE(df6_initial_list_status)
lst_IV = append_IV_list(df_temp, lst_IV)

# **************************************************************************
# Preprocessing CONTINUOUS Variables: Creating Dummy Variables, Part 1
# Same as discrete with differnce that we don't need to fine class, also dummies are combined using pd.isin([list]) unlike aggregate groupby in Discrete

# term_int
df_temp = woe_ordered_continuous(df_inputs_train_prep, 'term_int', df_targets_train_prep)
# plot_by_woe(df_temp)
lst_IV = append_IV_list(df_temp, lst_IV)

df_inputs_train_prep['term:36'] = np.where((df_inputs_train_prep['term_int'] == 36), 1, 0)
df_inputs_train_prep['term:60'] = np.where((df_inputs_train_prep['term_int'] == 60), 1, 0)

# emp_length_int
df_temp = woe_ordered_continuous(df_inputs_train_prep, 'emp_length_int', df_targets_train_prep)
# plot_by_woe(df_temp)
lst_IV = append_IV_list(df_temp, lst_IV)

df_inputs_train_prep['emp_length:0'] = np.where(df_inputs_train_prep['emp_length_int'].isin([0]), 1, 0)
df_inputs_train_prep['emp_length:1'] = np.where(df_inputs_train_prep['emp_length_int'].isin([1]), 1, 0)
df_inputs_train_prep['emp_length:2-4'] = np.where(df_inputs_train_prep['emp_length_int'].isin(range(2, 5)), 1, 0)
df_inputs_train_prep['emp_length:5-6'] = np.where(df_inputs_train_prep['emp_length_int'].isin(range(5, 7)), 1, 0)
df_inputs_train_prep['emp_length:7-9'] = np.where(df_inputs_train_prep['emp_length_int'].isin(range(7, 10)), 1, 0)
# df_inputs_train_prep['emp_length:10'] = np.where(df_inputs_train_prep['emp_length_int'].isin([10]), 1, 0)
df_inputs_train_prep['emp_length:10'] = np.where(df_inputs_train_prep['emp_length_int'] > 9, 1, 0)

## Preprocessing Continuous Variables: Creating Dummy Variables, Part 2
# mths_since_issue_d
# fine class to 50 buckets
df_inputs_train_prep['mths_since_issue_d_factor'] = pd.cut(df_inputs_train_prep['mths_since_issue_d'], 50)
df_temp = woe_ordered_continuous(df_inputs_train_prep, 'mths_since_issue_d_factor', df_targets_train_prep)
# plot_by_woe(df_temp)
# plot_by_woe(df_temp, 90) # We plot the weight of evidence values, rotating the labels 90 degrees.
lst_IV = append_IV_list(df_temp, lst_IV)

# We create the following categories:
# < 38, 38 - 39, 40 - 41, 42 - 48, 49 - 52, 53 - 64, 65 - 84, > 84.
df_inputs_train_prep['mths_since_issue_d:<38'] = np.where(df_inputs_train_prep['mths_since_issue_d'] < 38, 1, 0)
df_inputs_train_prep['mths_since_issue_d:38-39'] = np.where(df_inputs_train_prep['mths_since_issue_d'].isin(range(38, 40)), 1, 0)
df_inputs_train_prep['mths_since_issue_d:40-41'] = np.where(df_inputs_train_prep['mths_since_issue_d'].isin(range(40, 42)), 1, 0)
df_inputs_train_prep['mths_since_issue_d:42-48'] = np.where(df_inputs_train_prep['mths_since_issue_d'].isin(range(42, 49)), 1, 0)
df_inputs_train_prep['mths_since_issue_d:49-52'] = np.where(df_inputs_train_prep['mths_since_issue_d'].isin(range(49, 53)), 1, 0)
df_inputs_train_prep['mths_since_issue_d:53-64'] = np.where(df_inputs_train_prep['mths_since_issue_d'].isin(range(53, 65)), 1, 0)
df_inputs_train_prep['mths_since_issue_d:65-84'] = np.where(df_inputs_train_prep['mths_since_issue_d'].isin(range(65, 85)), 1, 0)
# df_inputs_train_prep['mths_since_issue_d:>84'] = np.where(df_inputs_train_prep['mths_since_issue_d'].isin(range(85, int(df_inputs_train_prep['mths_since_issue_d'].max()))), 1, 0)
df_inputs_train_prep['mths_since_issue_d:>84'] = np.where(df_inputs_train_prep['mths_since_issue_d'] > 84, 1, 0)

# int_rate
# Here we do fine-classing: using the 'cut' method, we split the variable into 50 categories by its values.
df_inputs_train_prep['int_rate_factor'] = pd.cut(df_inputs_train_prep['int_rate'], 50)
df_temp = woe_ordered_continuous(df_inputs_train_prep, 'int_rate_factor', df_targets_train_prep)
# plot_by_woe(df_temp, 90) # rotating the labels 90 degrees.
lst_IV = append_IV_list(df_temp, lst_IV)

# '< 9.548', '9.548 - 12.025', '12.025 - 15.74', '15.74 - 20.281', '> 20.281'
df_inputs_train_prep['int_rate:<9.548'] = np.where((df_inputs_train_prep['int_rate'] <= 9.548), 1, 0)
df_inputs_train_prep['int_rate:9.548-12.025'] = np.where((df_inputs_train_prep['int_rate'] > 9.548) & (df_inputs_train_prep['int_rate'] <= 12.025), 1, 0)
df_inputs_train_prep['int_rate:12.025-15.74'] = np.where((df_inputs_train_prep['int_rate'] > 12.025) & (df_inputs_train_prep['int_rate'] <= 15.74), 1, 0)
df_inputs_train_prep['int_rate:15.74-20.281'] = np.where((df_inputs_train_prep['int_rate'] > 15.74) & (df_inputs_train_prep['int_rate'] <= 20.281), 1, 0)
df_inputs_train_prep['int_rate:>20.281'] = np.where((df_inputs_train_prep['int_rate'] > 20.281), 1, 0)

# funded_amnt
df_inputs_train_prep['funded_amnt_factor'] = pd.cut(df_inputs_train_prep['funded_amnt'], 50)
# Here we do fine-classing: using the 'cut' method, we split the variable into 50 categories by its values.
df_temp = woe_ordered_continuous(df_inputs_train_prep, 'funded_amnt_factor', df_targets_train_prep)
# plot_by_woe(df_temp, 90) # rotating the labels 90 degrees.
lst_IV = append_IV_list(df_temp, lst_IV)



#*****************************
# List of variables and IV, print to CSV for analysis

# convert list into df
df_IV = pd.DataFrame(data=lst_IV, columns=['Variable','IV'])
dataframe_name = 'df_IV'
file_name = 'temp/df_IV.csv'

try:
    # start from column HA = good_bad
    df_IV.to_csv(file_name)
except Exception as e:
    print('exec loan_data.to_csv(file_name); error = ' + e)
else:
    print(f'DataFrame {dataframe_name} is written to CSV File = {file_name} successfully.')



#***************************************************************************#
## TEMP, DELETE THIS
'''
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
'''
#***************************************************************************#
'''
df_inputs_train_prep.iloc[:51,(0:2, 206:)].to_csv(file_name)
df_inputs_train_prep.iloc[:51, 205:]

df_tmp = pd.concat([df_inputs_train_prep.iloc[:51,1:4], df_inputs_train_prep.iloc[:52,206:]], axis=1)


# -------------
df_inputs_train_prep['addr_state:]


# df[df['A'].str.contains("hello")
df_inputs_train_prep[df_inputs_train_prep.columns.str.contains('addr_state')]

df.loc[:,df.columns.str.startswith('al')]  # col name start with

df_tmp = df_inputs_train_prep.loc[:10, df_inputs_train_prep.columns.str.startswith('addr_st')]

# -------------------------

#***************************************************************************#
#summarize indep variables and IF in a table. Create a List and convert to a df
#note: never grow a df! First, accumulate data in a list then create the df
#see: https://www.geeksforgeeks.org/different-ways-to-create-pandas-dataframe/
  c0 = ['grade','addr_state','purpose']
  c1 = [df1_grade['IV'].iloc[0],df2['IV'].iloc[0],df3['IV'].iloc[0]]
  l = list(zip(c0,c1))
  df_IF = pd.DataFrame(data=l, columns=['ind_var','IF'])
  df_IF.dtypes


df5_purpose = df5_purpose.sort_values('n_obs')

'''

