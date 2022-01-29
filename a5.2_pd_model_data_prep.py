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
def plot_by_woe(df_WoE, roation_of_axis_labels = 0, width=15, height=7 ):
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
# plot_by_woe(df1_grade)

#***************************************************************************#
# PRE-PROCESSING DISCRETE VARIABLES
# Creating dummies part I; Sec.5 L29: Data Prep: Preprocessing Discrete Variables.
# 1. Calculate WoE for the x vars. Call function: woe_discrete(df, discrete_var_name(x), df_target_var(Y))
# 2. Visualize data with func: plot_by_woe(df_w_WoE) ;
# 3. COARSE class by grouping similar WoE vars and creating new dummy columns. As each column will only have one value wit 1, this can be grouped by Summing
# 4  Decide the reference catgegory and include names of dummy vars and reference dummy vars in a list in excel
# In general, you have to put the categories with similar weight of evidence in one and the same category (dummy variable). However, sometimes, other considerations may play a role, such as how large the initial categories are, or their meaning, etc.


# ---- df1_grade : no combining needed ---
df_temp = woe_discrete(df_inputs_train_prep,'grade',df_targets_train_prep)
# plot_by_woe(df_temp)

# Collect variable names and IV values to outputed to excel in the end for analysis of IV
lst_IV = [] # initialize list_IV
lst_IV = append_IV_list(df_temp, lst_IV) # append variable and IV values to list IV # append variable and IV values to list IV

# --- df2_home_own : Combine, OTHER, NONE, RENT, and ANY (WoE as very low). OWN and MORTGAGE will be in a separate dummy var
df_temp = woe_discrete(df_inputs_train_prep,'home_ownership',df_targets_train_prep)
# plot_by_woe(df2_home_own)
lst_IV = append_IV_list(df_temp, lst_IV) # append variable and IV values to list IV

df_inputs_train_prep['home_ownership:RENT_OTHER_NONE_ANY'] = sum([df_inputs_train_prep['home_ownership:RENT'], df_inputs_train_prep['home_ownership:OTHER'],
                                                            df_inputs_train_prep['home_ownership:NONE'], df_inputs_train_prep['home_ownership:ANY']])

# ---- df3_addr_st : if column 'addr_state:ND' exist, leave it, else create a new column and set it to 0.
df_temp = woe_discrete(df_inputs_train_prep,'addr_state',df_targets_train_prep)
# plot_by_woe(df3_addr_st)
lst_IV = append_IV_list(df_temp, lst_IV) # append variable and IV values to list IV

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
# plot_by_woe(df4_verification_status)
lst_IV = append_IV_list(df_temp, lst_IV) # append variable and IV values to list IV

# df5_purpose : combine
df_temp = woe_discrete(df_inputs_train_prep, 'purpose', df_targets_train_prep)
# plot_by_woe(df5_purpose)
df_inputs_train_prep['purpose:educ__sm_b__wedd__ren_en__mov__house'] = sum([df_inputs_train_prep['purpose:educational'], df_inputs_train_prep['purpose:small_business'],
                                                                 df_inputs_train_prep['purpose:wedding'], df_inputs_train_prep['purpose:renewable_energy'],
                                                                 df_inputs_train_prep['purpose:moving'], df_inputs_train_prep['purpose:house']])
df_inputs_train_prep['purpose:oth__med__vacation'] = sum([df_inputs_train_prep['purpose:other'], df_inputs_train_prep['purpose:medical'],
                                             df_inputs_train_prep['purpose:vacation']])
df_inputs_train_prep['purpose:major_purch__car__home_impr'] = sum([df_inputs_train_prep['purpose:major_purchase'], df_inputs_train_prep['purpose:car'],
                                                        df_inputs_train_prep['purpose:home_improvement']])


# df5_initial_list_status : no combining needed
df_temp = woe_discrete(df_inputs_train_prep, 'initial_list_status', df_targets_train_prep)
# plot_by_woe(df6_initial_list_status)
lst_IV = append_IV_list(df_temp, lst_IV) # append variable and IV values to list IV

#***************************************************************************#
# PRE-PROCESSING CONTINUOUS VARIABLES
# Creating dummies part I; Sec.5 L29: Data Prep: Preprocessing Discrete Variables.
# 1. Calculate WoE for the x vars. Call function: woe_discrete(df, discrete_var_name(x), df_target_var(Y))
# 2. Visualize data with func: plot_by_woe(df_w_WoE) ;
# 3. COARSE class by grouping similar WoE vars and creating new dummy columns. As each column will only have one value wit 1, this can be grouped by Summing
# 4  Decide the reference catgegory and include names of dummy vars and reference dummy vars in a list in excel
# In general, you have to put the categories with similar weight of evidence in one and the same category (dummy variable). However, sometimes, other considerations may play a role, such as how large the initial categories are, or their meaning, etc.

# Preprocessing CONTINUOUS Variables: Creating Dummy Variables, Part 1
# Same as discrete with differnce that FINE classing is done by pd.cut method,
# also dummies are combined using np.where method and filtered by isin([list]) unlike aggregate groupby or filtered by >= operators
# also, no need to sort by WoE but use the default sort of the inputs

# term_int
df_temp = woe_ordered_continuous(df_inputs_train_prep, 'term_int', df_targets_train_prep)
# plot_by_woe(df_temp) ---------------------------
lst_IV = append_IV_list(df_temp, lst_IV) # append variable and IV values to list IV

df_inputs_train_prep['term:36'] = np.where((df_inputs_train_prep['term_int'] == 36), 1, 0)
df_inputs_train_prep['term:60'] = np.where((df_inputs_train_prep['term_int'] == 60), 1, 0)

# emp_length_int ---------------------------
df_temp = woe_ordered_continuous(df_inputs_train_prep, 'emp_length_int', df_targets_train_prep)
# plot_by_woe(df_temp)
lst_IV = append_IV_list(df_temp, lst_IV) # append variable and IV values to list IV

df_inputs_train_prep['emp_length:0'] = np.where(df_inputs_train_prep['emp_length_int'].isin([0]), 1, 0)
df_inputs_train_prep['emp_length:1'] = np.where(df_inputs_train_prep['emp_length_int'].isin([1]), 1, 0)
df_inputs_train_prep['emp_length:2-4'] = np.where(df_inputs_train_prep['emp_length_int'].isin(range(2, 5)), 1, 0)
df_inputs_train_prep['emp_length:5-6'] = np.where(df_inputs_train_prep['emp_length_int'].isin(range(5, 7)), 1, 0)
df_inputs_train_prep['emp_length:7-9'] = np.where(df_inputs_train_prep['emp_length_int'].isin(range(7, 10)), 1, 0)
# df_inputs_train_prep['emp_length:10'] = np.where(df_inputs_train_prep['emp_length_int'].isin([10]), 1, 0)
df_inputs_train_prep['emp_length:10'] = np.where(df_inputs_train_prep['emp_length_int'] > 9, 1, 0)

## Preprocessing Continuous Variables: Creating Dummy Variables, Part 2
# mths_since_issue_d  ---------------------------
# FINE Class to 50 buckets
df_inputs_train_prep['mths_since_issue_d_factor'] = pd.cut(df_inputs_train_prep['mths_since_issue_d'], 50)
df_temp = woe_ordered_continuous(df_inputs_train_prep, 'mths_since_issue_d_factor', df_targets_train_prep)
# plot_by_woe(df_temp)
# plot_by_woe(df_temp, 90) # We plot the weight of evidence values, rotating the labels 90 degrees.
lst_IV = append_IV_list(df_temp, lst_IV) # append variable and IV values to list IV

# COURSE class(new cats based on subj criteria). We create the following categories:
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

# int_rate ---------------------------
# FINE Class to 50 buckets
df_inputs_train_prep['int_rate_factor'] = pd.cut(df_inputs_train_prep['int_rate'], 50)
df_temp = woe_ordered_continuous(df_inputs_train_prep, 'int_rate_factor', df_targets_train_prep)
# plot_by_woe(df_temp, 90) # rotating the labels 90 degrees.
lst_IV = append_IV_list(df_temp, lst_IV) # append variable and IV values to list IV

# # COURSE class(new cats based on subj criteria). We create the following categories:
# '< 9.548', '9.548 - 12.025', '12.025 - 15.74', '15.74 - 20.281', '> 20.281'
df_inputs_train_prep['int_rate:<9.548'] = np.where((df_inputs_train_prep['int_rate'] < 9.548), 1, 0)
df_inputs_train_prep['int_rate:9.548-12.025'] = np.where((df_inputs_train_prep['int_rate'] >= 9.548) & (df_inputs_train_prep['int_rate'] < 12.025), 1, 0)
df_inputs_train_prep['int_rate:12.025-15.74'] = np.where((df_inputs_train_prep['int_rate'] >= 12.025) & (df_inputs_train_prep['int_rate'] < 15.74), 1, 0)
df_inputs_train_prep['int_rate:15.74-20.281'] = np.where((df_inputs_train_prep['int_rate'] >= 15.74) & (df_inputs_train_prep['int_rate'] < 20.281), 1, 0)
df_inputs_train_prep['int_rate:>20.281'] = np.where((df_inputs_train_prep['int_rate'] >= 20.281), 1, 0)

# funded_amnt ---------------------------
df_inputs_train_prep['funded_amnt_factor'] = pd.cut(df_inputs_train_prep['funded_amnt'], 50)
# Here we do fine-classing: using the 'cut' method, we split the variable into 50 categories by its values.
df_temp = woe_ordered_continuous(df_inputs_train_prep, 'funded_amnt_factor', df_targets_train_prep)
# plot_by_woe(df_temp, 90) # rotating the labels 90 degrees.
lst_IV = append_IV_list(df_temp, lst_IV) # append variable and IV values to list IV

# **  mths_since_earliest_cr_line ---------------------------
# df_inputs_train_prep['mths_since_earliest_cr_line'].value_counts()
# FINE Class to 50 buckets
df_inputs_train_prep['mths_since_earliest_cr_line_factor'] = pd.cut(df_inputs_train_prep['mths_since_earliest_cr_line'], 50)
df_temp = woe_ordered_continuous(df_inputs_train_prep, 'mths_since_earliest_cr_line_factor', df_targets_train_prep)
# plot_by_woe(df_temp, 45) # rotating the labels 90 degrees.
# plot_by_woe(df_temp.iloc[20: , : ], 45)
lst_IV = append_IV_list(df_temp, lst_IV) # append variable and IV values to list IV

# # COURSE class(new cats based on subj criteria). We create the following categories:
# < 140, # 141 - 164, # 165 - 247, # 248 - 270, # 271 - 352, # > 352
df_inputs_train_prep['mths_since_earliest_cr_line:<140'] = np.where(df_inputs_train_prep['mths_since_earliest_cr_line'].isin(range(140)), 1, 0)
df_inputs_train_prep['mths_since_earliest_cr_line:141-164'] = np.where(df_inputs_train_prep['mths_since_earliest_cr_line'].isin(range(140, 165)), 1, 0)
df_inputs_train_prep['mths_since_earliest_cr_line:165-247'] = np.where(df_inputs_train_prep['mths_since_earliest_cr_line'].isin(range(165, 248)), 1, 0)
df_inputs_train_prep['mths_since_earliest_cr_line:248-270'] = np.where(df_inputs_train_prep['mths_since_earliest_cr_line'].isin(range(248, 271)), 1, 0)
df_inputs_train_prep['mths_since_earliest_cr_line:271-352'] = np.where(df_inputs_train_prep['mths_since_earliest_cr_line'].isin(range(271, 353)), 1, 0)
df_inputs_train_prep['mths_since_earliest_cr_line:>352'] = np.where(df_inputs_train_prep['mths_since_earliest_cr_line'].isin(range(353, int(df_inputs_train_prep['mths_since_earliest_cr_line'].max()))), 1, 0)

# line 719

# delinq_2yrs ---------------------------
# df_inputs_train_prep['delinq_2yrs'].value_counts()
# no FINE Class required, only few cats
df_temp = woe_ordered_continuous(df_inputs_train_prep, 'delinq_2yrs', df_targets_train_prep)
# plot_by_woe(df_temp, 45) # rotating the labels 90 degrees.
lst_IV = append_IV_list(df_temp, lst_IV) # append variable and IV values to list IV

# COURSE class(new cats based on subj criteria). We create the following categories
# Categories: 0, 1-3, >=4
df_inputs_train_prep['delinq_2yrs:0'] = np.where((df_inputs_train_prep['delinq_2yrs'] == 0), 1, 0)
df_inputs_train_prep['delinq_2yrs:1-3'] = np.where((df_inputs_train_prep['delinq_2yrs'] >= 1) & (df_inputs_train_prep['delinq_2yrs'] <= 3), 1, 0)
df_inputs_train_prep['delinq_2yrs:>=4'] = np.where((df_inputs_train_prep['delinq_2yrs'] >= 9), 1, 0)

# inq_last_6mths ---------------------------
# df_inputs_train_prep['inq_last_6mths'].value_counts()
# no FINE Class required, only few cats
df_temp = woe_ordered_continuous(df_inputs_train_prep, 'inq_last_6mths', df_targets_train_prep)
# plot_by_woe(df_temp, 45) # rotating the labels 90 degrees.
lst_IV = append_IV_list(df_temp, lst_IV) # append variable and IV values to list IV

# COURSE class(new cats based on subj criteria). We create the following categories
# Categories: 0, 1 - 2, 3 - 6, > 6
df_inputs_train_prep['inq_last_6mths:0'] = np.where((df_inputs_train_prep['inq_last_6mths'] == 0), 1, 0)
df_inputs_train_prep['inq_last_6mths:1-2'] = np.where((df_inputs_train_prep['inq_last_6mths'] >= 1) & (df_inputs_train_prep['inq_last_6mths'] <= 2), 1, 0)
df_inputs_train_prep['inq_last_6mths:3-6'] = np.where((df_inputs_train_prep['inq_last_6mths'] >= 3) & (df_inputs_train_prep['inq_last_6mths'] <= 6), 1, 0)
df_inputs_train_prep['inq_last_6mths:>6'] = np.where((df_inputs_train_prep['inq_last_6mths'] > 6), 1, 0)

# open_acc ---------------------------
# df_inputs_train_prep['open_acc'].value_counts()
# no FINE Class required, only few cats
df_temp = woe_ordered_continuous(df_inputs_train_prep, 'open_acc', df_targets_train_prep)
# plot_by_woe(df_temp, 45) # rotating the labels 90 degrees.
lst_IV = append_IV_list(df_temp, lst_IV) # append variable and IV values to list IV

# COURSE class(new cats based on subj criteria). We create the following categories
# Categories: '0', '1-3', '4-12', '13-17', '18-22', '23-25', '26-30', '>30'
df_inputs_train_prep['open_acc:0'] = np.where((df_inputs_train_prep['open_acc'] == 0), 1, 0)
df_inputs_train_prep['open_acc:1-3'] = np.where((df_inputs_train_prep['open_acc'] >= 1) & (df_inputs_train_prep['open_acc'] <= 3), 1, 0)
df_inputs_train_prep['open_acc:4-12'] = np.where((df_inputs_train_prep['open_acc'] >= 4) & (df_inputs_train_prep['open_acc'] <= 12), 1, 0)
df_inputs_train_prep['open_acc:13-17'] = np.where((df_inputs_train_prep['open_acc'] >= 13) & (df_inputs_train_prep['open_acc'] <= 17), 1, 0)
df_inputs_train_prep['open_acc:18-22'] = np.where((df_inputs_train_prep['open_acc'] >= 18) & (df_inputs_train_prep['open_acc'] <= 22), 1, 0)
df_inputs_train_prep['open_acc:23-25'] = np.where((df_inputs_train_prep['open_acc'] >= 23) & (df_inputs_train_prep['open_acc'] <= 25), 1, 0)
df_inputs_train_prep['open_acc:26-30'] = np.where((df_inputs_train_prep['open_acc'] >= 26) & (df_inputs_train_prep['open_acc'] <= 30), 1, 0)
df_inputs_train_prep['open_acc:>=31'] = np.where((df_inputs_train_prep['open_acc'] >= 31), 1, 0)

# pub_rec ---------------------------
# df_inputs_train_prep['pub_rec'].value_counts()
# no FINE Class required, only few cats
df_temp = woe_ordered_continuous(df_inputs_train_prep, 'pub_rec', df_targets_train_prep)
# plot_by_woe(df_temp, 45) # rotating the labels 90 degrees.
lst_IV = append_IV_list(df_temp, lst_IV) # append variable and IV values to list IV

# COURSE class(new cats based on subj criteria). We create the following categories
# Categories '0-2', '3-4', '>=5'
df_inputs_train_prep['pub_rec:0-2'] = np.where((df_inputs_train_prep['pub_rec'] >= 0) & (df_inputs_train_prep['pub_rec'] <= 2), 1, 0)
df_inputs_train_prep['pub_rec:3-4'] = np.where((df_inputs_train_prep['pub_rec'] >= 3) & (df_inputs_train_prep['pub_rec'] <= 4), 1, 0)
df_inputs_train_prep['pub_rec:>=5'] = np.where((df_inputs_train_prep['pub_rec'] >= 5), 1, 0)

# total_acc ---------------------------
# df_inputs_train_prep['total_acc'].value_counts()
# FINE Class to 50 buckets
df_inputs_train_prep['total_acc_factor'] = pd.cut(df_inputs_train_prep['total_acc'], 50)
df_temp = woe_ordered_continuous(df_inputs_train_prep, 'total_acc_factor', df_targets_train_prep)
# plot_by_woe(df_temp, 45) # rotating the labels 90 degrees.
lst_IV = append_IV_list(df_temp, lst_IV) # append variable and IV values to list IV

# COURSE class(new cats based on subj criteria). We create the following categories
# Categories: '<=27', '28-51', '>51'
df_inputs_train_prep['total_acc:<=27'] = np.where((df_inputs_train_prep['total_acc'] <= 27), 1, 0)
df_inputs_train_prep['total_acc:28-51'] = np.where((df_inputs_train_prep['total_acc'] >= 28) & (df_inputs_train_prep['total_acc'] <= 51), 1, 0)
df_inputs_train_prep['total_acc:>=52'] = np.where((df_inputs_train_prep['total_acc'] >= 52), 1, 0)

# acc_now_delinq ---------------------------
# df_inputs_train_prep['acc_now_delinq'].value_counts()
# no FINE Class required, only few cats
df_temp = woe_ordered_continuous(df_inputs_train_prep, 'acc_now_delinq', df_targets_train_prep)
# plot_by_woe(df_temp, 45) # rotating the labels 90 degrees.
lst_IV = append_IV_list(df_temp, lst_IV) # append variable and IV values to list IV

# COURSE class(new cats based on subj criteria). We create the following categories
# Categories: '0', '>=1'
df_inputs_train_prep['acc_now_delinq:0'] = np.where((df_inputs_train_prep['acc_now_delinq'] == 0), 1, 0)
df_inputs_train_prep['acc_now_delinq:>=1'] = np.where((df_inputs_train_prep['acc_now_delinq'] >= 1), 1, 0)

# total_rev_hi_lim ---------------------------
# df_inputs_train_prep['total_rev_hi_lim'].value_counts()
# FINE Class to 50 buckets
df_inputs_train_prep['total_rev_hi_lim_factor'] = pd.cut(df_inputs_train_prep['total_rev_hi_lim'], 2000)
df_temp = woe_ordered_continuous(df_inputs_train_prep, 'total_rev_hi_lim_factor', df_targets_train_prep)
# plot_by_woe(df_temp, 45) # rotating the labels 90 degrees.
lst_IV = append_IV_list(df_temp, lst_IV) # append variable and IV values to list IV

# COURSE class(new cats based on subj criteria). We create the following categories
# '<=5K', '5K-10K', '10K-20K', '20K-30K', '30K-40K', '40K-55K', '55K-95K', '>95K'
df_inputs_train_prep['total_rev_hi_lim:<=5K'] = np.where((df_inputs_train_prep['total_rev_hi_lim'] <= 5000), 1, 0)
df_inputs_train_prep['total_rev_hi_lim:5K-10K'] = np.where((df_inputs_train_prep['total_rev_hi_lim'] > 5000) & (df_inputs_train_prep['total_rev_hi_lim'] <= 10000), 1, 0)
df_inputs_train_prep['total_rev_hi_lim:10K-20K'] = np.where((df_inputs_train_prep['total_rev_hi_lim'] > 10000) & (df_inputs_train_prep['total_rev_hi_lim'] <= 20000), 1, 0)
df_inputs_train_prep['total_rev_hi_lim:20K-30K'] = np.where((df_inputs_train_prep['total_rev_hi_lim'] > 20000) & (df_inputs_train_prep['total_rev_hi_lim'] <= 30000), 1, 0)
df_inputs_train_prep['total_rev_hi_lim:30K-40K'] = np.where((df_inputs_train_prep['total_rev_hi_lim'] > 30000) & (df_inputs_train_prep['total_rev_hi_lim'] <= 40000), 1, 0)
df_inputs_train_prep['total_rev_hi_lim:40K-55K'] = np.where((df_inputs_train_prep['total_rev_hi_lim'] > 40000) & (df_inputs_train_prep['total_rev_hi_lim'] <= 55000), 1, 0)
df_inputs_train_prep['total_rev_hi_lim:55K-95K'] = np.where((df_inputs_train_prep['total_rev_hi_lim'] > 55000) & (df_inputs_train_prep['total_rev_hi_lim'] <= 95000), 1, 0)
df_inputs_train_prep['total_rev_hi_lim:>95K'] = np.where((df_inputs_train_prep['total_rev_hi_lim'] > 95000), 1, 0)

# installment ---------------------------
# df_inputs_train_prep['installment'].value_counts()
# FINE Class to 50 buckets
df_inputs_train_prep['installment_factor'] = pd.cut(df_inputs_train_prep['installment'], 50)
df_temp = woe_ordered_continuous(df_inputs_train_prep, 'installment_factor', df_targets_train_prep)
# plot_by_woe(df_temp, 45) # rotating the labels 90 degrees.
lst_IV = append_IV_list(df_temp, lst_IV) # append variable and IV values to list IV

# COURSE class(new cats based on subj criteria). We create the following categories
# ?

### Preprocessing Continuous Variables: Creating Dummy Variables, Part 3
# annual_inc ---------------------------
# df_inputs_train_prep['annual_inc'].value_counts()
# FINE Class to 50 buckets
df_inputs_train_prep['annual_inc_factor'] = pd.cut(df_inputs_train_prep['annual_inc'], 100)

df_temp = woe_ordered_continuous(df_inputs_train_prep, 'annual_inc_factor', df_targets_train_prep)
# plot_by_woe(df_temp, 45) # rotating the labels 90 degrees.
lst_IV = append_IV_list(df_temp, lst_IV) # append variable and IV values to list IV

## ***
# Initial examination shows that there are too few individuals with large income and too many with small income.
# Hence, we are going to have one category for more than 150K, and we are going to apply our approach to determine
# the categories of everyone with 140k or less.
df_inputs_train_prep_temp = df_inputs_train_prep.loc[df_inputs_train_prep['annual_inc'] <= 140000, : ]
#loan_data_temp = loan_data_temp.reset_index(drop = True)
#df_inputs_train_prep_temp

df_inputs_train_prep_temp["annual_inc_factor"] = pd.cut(df_inputs_train_prep_temp['annual_inc'], 50)
# Here we do fine-classing: using the 'cut' method, we split the variable into 50 categories by its values.
# note: the target needs to be in line with the inputs index. We use the .index attribute to only get targets with the same index as the inputs
df_temp = woe_ordered_continuous(df_inputs_train_prep_temp, 'annual_inc_factor', df_targets_train_prep[df_inputs_train_prep_temp.index])
# plot_by_woe(df_temp, 45) # rotating the labels 90 degrees.
lst_IV = append_IV_list(df_temp, lst_IV) # append variable and IV values to list IV

# COURSE class(new cats based on subj criteria). We create the following categories
# WoE is monotonically decreasing with income, so we split income in 10 equal categories, each with width of 15k.
df_inputs_train_prep['annual_inc:<20K'] = np.where((df_inputs_train_prep['annual_inc'] <= 20000), 1, 0)
df_inputs_train_prep['annual_inc:20K-30K'] = np.where((df_inputs_train_prep['annual_inc'] > 20000) & (df_inputs_train_prep['annual_inc'] <= 30000), 1, 0)
df_inputs_train_prep['annual_inc:30K-40K'] = np.where((df_inputs_train_prep['annual_inc'] > 30000) & (df_inputs_train_prep['annual_inc'] <= 40000), 1, 0)
df_inputs_train_prep['annual_inc:40K-50K'] = np.where((df_inputs_train_prep['annual_inc'] > 40000) & (df_inputs_train_prep['annual_inc'] <= 50000), 1, 0)
df_inputs_train_prep['annual_inc:50K-60K'] = np.where((df_inputs_train_prep['annual_inc'] > 50000) & (df_inputs_train_prep['annual_inc'] <= 60000), 1, 0)
df_inputs_train_prep['annual_inc:60K-70K'] = np.where((df_inputs_train_prep['annual_inc'] > 60000) & (df_inputs_train_prep['annual_inc'] <= 70000), 1, 0)
df_inputs_train_prep['annual_inc:70K-80K'] = np.where((df_inputs_train_prep['annual_inc'] > 70000) & (df_inputs_train_prep['annual_inc'] <= 80000), 1, 0)
df_inputs_train_prep['annual_inc:80K-90K'] = np.where((df_inputs_train_prep['annual_inc'] > 80000) & (df_inputs_train_prep['annual_inc'] <= 90000), 1, 0)
df_inputs_train_prep['annual_inc:90K-100K'] = np.where((df_inputs_train_prep['annual_inc'] > 90000) & (df_inputs_train_prep['annual_inc'] <= 100000), 1, 0)
df_inputs_train_prep['annual_inc:100K-120K'] = np.where((df_inputs_train_prep['annual_inc'] > 100000) & (df_inputs_train_prep['annual_inc'] <= 120000), 1, 0)
df_inputs_train_prep['annual_inc:120K-140K'] = np.where((df_inputs_train_prep['annual_inc'] > 120000) & (df_inputs_train_prep['annual_inc'] <= 140000), 1, 0)
df_inputs_train_prep['annual_inc:>140K'] = np.where((df_inputs_train_prep['annual_inc'] > 140000), 1, 0)

# mths_since_last_delinq ---------------------------
# df_inputs_train_prep['mths_since_last_delinq'].value_counts()

# We have to create one category for missing values and do fine and coarse classing for the rest.
# We use pd.notnull () function
df_inputs_train_prep_temp = df_inputs_train_prep[pd.notnull(df_inputs_train_prep['mths_since_last_delinq'])]
df_inputs_train_prep_temp['mths_since_last_delinq_factor'] = pd.cut(df_inputs_train_prep_temp['mths_since_last_delinq'], 50)
# we usd .index method to match targets and inputs based on index
df_temp = woe_ordered_continuous(df_inputs_train_prep_temp, 'mths_since_last_delinq_factor', df_targets_train_prep[df_inputs_train_prep_temp.index])

# plot_by_woe(df_temp, 45) # rotating the labels 90 degrees.
lst_IV = append_IV_list(df_temp, lst_IV) # append variable and IV values to list IV

# COURSE class(new cats based on subj criteria). We create the following categories
# Categories: Missing, 0-3, 4-30, 31-56, >=57
df_inputs_train_prep['mths_since_last_delinq:Missing'] = np.where((df_inputs_train_prep['mths_since_last_delinq'].isnull()), 1, 0)
df_inputs_train_prep['mths_since_last_delinq:0-3'] = np.where((df_inputs_train_prep['mths_since_last_delinq'] >= 0) & (df_inputs_train_prep['mths_since_last_delinq'] <= 3), 1, 0)
df_inputs_train_prep['mths_since_last_delinq:4-30'] = np.where((df_inputs_train_prep['mths_since_last_delinq'] >= 4) & (df_inputs_train_prep['mths_since_last_delinq'] <= 30), 1, 0)
df_inputs_train_prep['mths_since_last_delinq:31-56'] = np.where((df_inputs_train_prep['mths_since_last_delinq'] >= 31) & (df_inputs_train_prep['mths_since_last_delinq'] <= 56), 1, 0)
df_inputs_train_prep['mths_since_last_delinq:>=57'] = np.where((df_inputs_train_prep['mths_since_last_delinq'] >= 57), 1, 0)

### Sec5.chp 37  Preprocessing Continuous Variables: Creating Dummy Variables, Part 3: Homework

# dti ---------------------------
# df_inputs_train_prep['dti'].value_counts()
# FINE Class to 100 buckets
df_inputs_train_prep['dti_factor'] = pd.cut(df_inputs_train_prep['dti'], 100)
df_temp = woe_ordered_continuous(df_inputs_train_prep, 'dti_factor', df_targets_train_prep)

# Similarly to income, initial examination shows that most values are lower than 200.
# Hence, we are going to have one category for more than 35, and we are going to apply our approach to determine
# the categories of everyone with 150k or less.
df_inputs_train_prep_temp = df_inputs_train_prep.loc[df_inputs_train_prep['dti'] <= 35, : ]

df_inputs_train_prep_temp['dti_factor'] = pd.cut(df_inputs_train_prep_temp['dti'], 50)
# Here we do fine-classing: using the 'cut' method, we split the variable into 50 categories by its values.
df_temp = woe_ordered_continuous(df_inputs_train_prep_temp, 'dti_factor', df_targets_train_prep[df_inputs_train_prep_temp.index])

# plot_by_woe(df_temp, 45) # rotating the labels 90 degrees.
lst_IV = append_IV_list(df_temp, lst_IV) # append variable and IV values to list IV

# COURSE class(new cats based on subj criteria). We create the following categories
df_inputs_train_prep['dti:<=1.4'] = np.where((df_inputs_train_prep['dti'] <= 1.4), 1, 0)
df_inputs_train_prep['dti:1.4-3.5'] = np.where((df_inputs_train_prep['dti'] > 1.4) & (df_inputs_train_prep['dti'] <= 3.5), 1, 0)
df_inputs_train_prep['dti:3.5-7.7'] = np.where((df_inputs_train_prep['dti'] > 3.5) & (df_inputs_train_prep['dti'] <= 7.7), 1, 0)
df_inputs_train_prep['dti:7.7-10.5'] = np.where((df_inputs_train_prep['dti'] > 7.7) & (df_inputs_train_prep['dti'] <= 10.5), 1, 0)
df_inputs_train_prep['dti:10.5-16.1'] = np.where((df_inputs_train_prep['dti'] > 10.5) & (df_inputs_train_prep['dti'] <= 16.1), 1, 0)
df_inputs_train_prep['dti:16.1-20.3'] = np.where((df_inputs_train_prep['dti'] > 16.1) & (df_inputs_train_prep['dti'] <= 20.3), 1, 0)
df_inputs_train_prep['dti:20.3-21.7'] = np.where((df_inputs_train_prep['dti'] > 20.3) & (df_inputs_train_prep['dti'] <= 21.7), 1, 0)
df_inputs_train_prep['dti:21.7-22.4'] = np.where((df_inputs_train_prep['dti'] > 21.7) & (df_inputs_train_prep['dti'] <= 22.4), 1, 0)
df_inputs_train_prep['dti:22.4-35'] = np.where((df_inputs_train_prep['dti'] > 22.4) & (df_inputs_train_prep['dti'] <= 35), 1, 0)
df_inputs_train_prep['dti:>35'] = np.where((df_inputs_train_prep['dti'] > 35), 1, 0)

# mths_since_last_record ---------------------------
# df_inputs_train_prep['var'].value_counts()

# We have to create one category for missing values and do fine and coarse classing for the rest.
df_inputs_train_prep_temp = df_inputs_train_prep[pd.notnull(df_inputs_train_prep['mths_since_last_record'])]
#sum(loan_data_temp['mths_since_last_record'].isnull())

# FINE Class to 50 buckets
df_inputs_train_prep_temp['mths_since_last_record_factor'] = pd.cut(df_inputs_train_prep_temp['mths_since_last_record'], 50)

df_temp = woe_ordered_continuous(df_inputs_train_prep_temp, 'mths_since_last_record_factor', df_targets_train_prep[df_inputs_train_prep_temp.index])
# plot_by_woe(df_temp, 45) # rotating the labels 90 degrees.
lst_IV = append_IV_list(df_temp, lst_IV) # append variable and IV values to list IV

# COURSE class(new cats based on subj criteria). We create the following categories
# Categories: 'Missing', '0-2', '3-20', '21-31', '32-80', '81-86', '>86'
df_inputs_train_prep['mths_since_last_record:Missing'] = np.where((df_inputs_train_prep['mths_since_last_record'].isnull()), 1, 0)
df_inputs_train_prep['mths_since_last_record:0-2'] = np.where((df_inputs_train_prep['mths_since_last_record'] >= 0) & (df_inputs_train_prep['mths_since_last_record'] <= 2), 1, 0)
df_inputs_train_prep['mths_since_last_record:3-20'] = np.where((df_inputs_train_prep['mths_since_last_record'] >= 3) & (df_inputs_train_prep['mths_since_last_record'] <= 20), 1, 0)
df_inputs_train_prep['mths_since_last_record:21-31'] = np.where((df_inputs_train_prep['mths_since_last_record'] >= 21) & (df_inputs_train_prep['mths_since_last_record'] <= 31), 1, 0)
df_inputs_train_prep['mths_since_last_record:32-80'] = np.where((df_inputs_train_prep['mths_since_last_record'] >= 32) & (df_inputs_train_prep['mths_since_last_record'] <= 80), 1, 0)
df_inputs_train_prep['mths_since_last_record:81-86'] = np.where((df_inputs_train_prep['mths_since_last_record'] >= 81) & (df_inputs_train_prep['mths_since_last_record'] <= 86), 1, 0)
df_inputs_train_prep['mths_since_last_record:>86'] = np.where((df_inputs_train_prep['mths_since_last_record'] > 86), 1, 0)


# ********************************
### Sec5 chp 38, Preprocessing the Test Dataset

#-----------------------------------------
# remeber: delete this part

# Data Preparation: load Training data: loan_data_inputs_train, loan_data_targets_train
# create df for preprocessing.  calculate WoE and IV   / # REF: Section 5, video 26
df_inputs_train_prep = loan_data_inputs_train
df_targets_train_prep = loan_data_targets_train
#-----------------------------------------

# loan_data_inputs_train = df_inputs_train_prep
loan_data_inputs_test = df_inputs_train_prep

loan_data_inputs_train.to_csv('loan_data_inputs_train.csv')
loan_data_targets_train.to_csv('loan_data_targets_train.csv')
loan_data_inputs_test.to_csv('loan_data_inputs_test.csv')
loan_data_targets_test.to_csv('loan_data_targets_test.csv')



#*********************************************************************
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

