
# ****************************************************
# PART 2: START HERE
# ****************************************************
#-----------------------------------------
# Data Preparation: load Training data: loan_data_inputs_train, loan_data_targets_train
# create df for preprocessing.  calculate WoE and IV   / # REF: Section 5, video 26

import numpy as np
import pandas as pd

### THIS IS A SHORT CUT. TO SKIP PART I DATA PREP:
# Import Data from csv
loan_data_inputs_train = pd.read_csv('data/loan_data_inputs_train.csv', index_col=0)
loan_data_targets_train = pd.read_csv('data/loan_data_targets_train.csv', index_col=0)
# loan_data_inputs_test = pd.read_csv('data/loan_data_inputs_test.csv', index_col=0)
# loan_data_targets_test = pd.read_csv('data/loan_data_targets_test.csv', index_col=0, header=None)

# a) train set
df_inputs_prepr = loan_data_inputs_train
df_targets_prep = loan_data_targets_train

# b) test set
# df_inputs_prepr = loan_data_inputs_test
# df_targets_prep = loan_data_targets_test
#-----------------------------------------

#***************************************************************************#
# PRE-PROCESSING DISCRETE VARIABLES
# Creating dummies part I; Sec.5 L29: Data Prep: Preprocessing Discrete Variables.
# 1. Calculate WoE for the x vars. Call function: woe_discrete(df, discrete_var_name(x), df_target_var(Y))
# 2. Visualize data with func: plot_by_woe(df_w_WoE) ;
# 3. COARSE class by grouping similar WoE vars and creating new dummy columns. As each column will only have one value wit 1, this can be grouped by Summing
# 4  Decide the reference catgegory and include names of dummy vars and reference dummy vars in a list in excel
# In general, you have to put the categories with similar weight of evidence in one and the same category (dummy variable). However, sometimes, other considerations may play a role, such as how large the initial categories are, or their meaning, etc.

import importlib
import b10_pd_model_class as pddf
# importlib.reload(pddf)

# insantiate dataframe for pd model inputs
pd_inputs = pddf.PdDataframe(df_inputs_prepr)

# ------- grade : no combining needed ------------------------
df_temp = pd_inputs.woe_discrete('grade',df_targets_prep)

# insantiate dataframe for WoE plot inputs. Then call plot function
pd_plot_WoE = pddf.PdDataframe(df_temp)
pd_plot_WoE.plot_by_woe()

# ----- home_ownership : Combine, OTHER, NONE, RENT, and ANY (WoE as very low). OWN and MORTGAGE will be in a separate dummy var
df_temp = pd_inputs.woe_discrete('home_ownership',df_targets_prep)

# insantiate dataframe for WoE plot inputs. Then call plot function
pd_plot_WoE = pddf.PdDataframe(df_temp)
pd_plot_WoE.plot_by_woe()

df_inputs_prepr['home_ownership:RENT_OTHER_NONE_ANY'] = sum([df_inputs_prepr['home_ownership:RENT'], df_inputs_prepr['home_ownership:OTHER'],
                                                            df_inputs_prepr['home_ownership:NONE'], df_inputs_prepr['home_ownership:ANY']])

# ------ addr_state : if column 'addr_state:ND' exist, leave it, else create a new column and set it to 0.
df_temp = pd_inputs.woe_discrete('addr_state',df_targets_prep)

# insantiate dataframe for WoE plot inputs. Then call plot function
pd_plot_WoE = pddf.PdDataframe(df_temp)
pd_plot_WoE.plot_by_woe()


if ['addr_state:ND'] in df_inputs_prepr.columns.values:  # handle missing values
    pass
else:
    df_inputs_prepr['addr_state:ND'] = 0

# Note: we can use Sum() as all will have 0 values, except 1 (it's exclusive) therefore, it will always sum to only 1
df_inputs_prepr['addr_state:ND_NE_IA_NV_FL_HI_AL'] = sum([df_inputs_prepr['addr_state:ND'], df_inputs_prepr['addr_state:NE'],
                                                         df_inputs_prepr['addr_state:IA'], df_inputs_prepr['addr_state:NV'],
                                                         df_inputs_prepr['addr_state:FL'], df_inputs_prepr['addr_state:HI'],
                                                         df_inputs_prepr['addr_state:AL']])

df_inputs_prepr['addr_state:NM_VA'] = sum([df_inputs_prepr['addr_state:NM'], df_inputs_prepr['addr_state:VA']])

df_inputs_prepr['addr_state:OK_TN_MO_LA_MD_NC'] = sum([df_inputs_prepr['addr_state:OK'], df_inputs_prepr['addr_state:TN'],
                                              df_inputs_prepr['addr_state:MO'], df_inputs_prepr['addr_state:LA'],
                                              df_inputs_prepr['addr_state:MD'], df_inputs_prepr['addr_state:NC']])

df_inputs_prepr['addr_state:UT_KY_AZ_NJ'] = sum([df_inputs_prepr['addr_state:UT'], df_inputs_prepr['addr_state:KY'],
                                              df_inputs_prepr['addr_state:AZ'], df_inputs_prepr['addr_state:NJ']])

df_inputs_prepr['addr_state:AR_MI_PA_OH_MN'] = sum([df_inputs_prepr['addr_state:AR'], df_inputs_prepr['addr_state:MI'],
                                              df_inputs_prepr['addr_state:PA'], df_inputs_prepr['addr_state:OH'],
                                              df_inputs_prepr['addr_state:MN']])

df_inputs_prepr['addr_state:RI_MA_DE_SD_IN'] = sum([df_inputs_prepr['addr_state:RI'], df_inputs_prepr['addr_state:MA'],
                                              df_inputs_prepr['addr_state:DE'], df_inputs_prepr['addr_state:SD'],
                                              df_inputs_prepr['addr_state:IN']])

df_inputs_prepr['addr_state:GA_WA_OR'] = sum([df_inputs_prepr['addr_state:GA'], df_inputs_prepr['addr_state:WA'],
                                              df_inputs_prepr['addr_state:OR']])

df_inputs_prepr['addr_state:WI_MT'] = sum([df_inputs_prepr['addr_state:WI'], df_inputs_prepr['addr_state:MT']])

df_inputs_prepr['addr_state:IL_CT'] = sum([df_inputs_prepr['addr_state:IL'], df_inputs_prepr['addr_state:CT']])

df_inputs_prepr['addr_state:KS_SC_CO_VT_AK_MS'] = sum([df_inputs_prepr['addr_state:KS'], df_inputs_prepr['addr_state:SC'],
                                              df_inputs_prepr['addr_state:CO'], df_inputs_prepr['addr_state:VT'],
                                              df_inputs_prepr['addr_state:AK'], df_inputs_prepr['addr_state:MS']])

df_inputs_prepr['addr_state:WV_NH_WY_DC_ME_ID'] = sum([df_inputs_prepr['addr_state:WV'], df_inputs_prepr['addr_state:NH'],
                                              df_inputs_prepr['addr_state:WY'], df_inputs_prepr['addr_state:DC'],
                                              df_inputs_prepr['addr_state:ME'], df_inputs_prepr['addr_state:ID']])

# Sec 5. chp 31 HW
# ------  verification_status : no combining needed ---------------------
df_temp = pd_inputs.woe_discrete('verification_status',df_targets_prep)

# insantiate dataframe for WoE plot inputs. Then call plot function
pd_plot_WoE = pddf.PdDataframe(df_temp)
pd_plot_WoE.plot_by_woe()

# ------  purpose ---------------------
df_temp = pd_inputs.woe_discrete('purpose',df_targets_prep)

# insantiate dataframe for WoE plot inputs. Then call plot function
pd_plot_WoE = pddf.PdDataframe(df_temp)
pd_plot_WoE.plot_by_woe()

df_inputs_prepr['purpose:educ__sm_b__wedd__ren_en__mov__house'] = sum([df_inputs_prepr['purpose:educational'], df_inputs_prepr['purpose:small_business'],
                                                                 df_inputs_prepr['purpose:wedding'], df_inputs_prepr['purpose:renewable_energy'],
                                                                 df_inputs_prepr['purpose:moving'], df_inputs_prepr['purpose:house']])
df_inputs_prepr['purpose:oth__med__vacation'] = sum([df_inputs_prepr['purpose:other'], df_inputs_prepr['purpose:medical'],
                                             df_inputs_prepr['purpose:vacation']])
df_inputs_prepr['purpose:major_purch__car__home_impr'] = sum([df_inputs_prepr['purpose:major_purchase'], df_inputs_prepr['purpose:car'],
                                                        df_inputs_prepr['purpose:home_improvement']])


# ------  initial_list_status : no combining needed
df_temp = pd_inputs.woe_discrete('initial_list_status',df_targets_prep)

# insantiate dataframe for WoE plot inputs. Then call plot function
pd_plot_WoE = pddf.PdDataframe(df_temp)
pd_plot_WoE.plot_by_woe()

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

# ------  term_int --------------------------------
df_temp = pd_inputs.woe_discrete('term_int',df_targets_prep)

# insantiate dataframe for WoE plot inputs. Then call plot function
pd_plot_WoE = pddf.PdDataframe(df_temp)
pd_plot_WoE.plot_by_woe()

df_inputs_prepr['term:36'] = np.where((df_inputs_prepr['term_int'] == 36), 1, 0)
df_inputs_prepr['term:60'] = np.where((df_inputs_prepr['term_int'] == 60), 1, 0)

# ------  emp_length_int ---------------------------
df_temp = pd_inputs.woe_discrete('emp_length_int',df_targets_prep)

# insantiate dataframe for WoE plot inputs. Then call plot function
pd_plot_WoE = pddf.PdDataframe(df_temp)
pd_plot_WoE.plot_by_woe()

df_inputs_prepr['emp_length:0'] = np.where(df_inputs_prepr['emp_length_int'].isin([0]), 1, 0)
df_inputs_prepr['emp_length:1'] = np.where(df_inputs_prepr['emp_length_int'].isin([1]), 1, 0)
df_inputs_prepr['emp_length:2-4'] = np.where(df_inputs_prepr['emp_length_int'].isin(range(2, 5)), 1, 0)
df_inputs_prepr['emp_length:5-6'] = np.where(df_inputs_prepr['emp_length_int'].isin(range(5, 7)), 1, 0)
df_inputs_prepr['emp_length:7-9'] = np.where(df_inputs_prepr['emp_length_int'].isin(range(7, 10)), 1, 0)
# df_inputs_prepr['emp_length:10'] = np.where(df_inputs_prepr['emp_length_int'].isin([10]), 1, 0)
df_inputs_prepr['emp_length:10'] = np.where(df_inputs_prepr['emp_length_int'] > 9, 1, 0)

# ************************************************************************
## Preprocessing Continuous Variables: Creating Dummy Variables, Part 2
# ------   mths_since_issue_d  ---------------------------
# FINE Class to 50 buckets
df_inputs_prepr['mths_since_issue_d_factor'] = pd.cut(df_inputs_prepr['mths_since_issue_d'], 50)
df_temp = woe_ordered_continuous(df_inputs_prepr, 'mths_since_issue_d_factor', df_targets_prep)
# plot_by_woe(df_temp)
# plot_by_woe(df_temp, 90) # We plot the weight of evidence values, rotating the labels 90 degrees.
lst_IV = append_IV_list(df_temp, lst_IV) # append variable and IV values to list IV

# COURSE class(new cats based on subj criteria). We create the following categories:
# < 38, 38 - 39, 40 - 41, 42 - 48, 49 - 52, 53 - 64, 65 - 84, > 84.
df_inputs_prepr['mths_since_issue_d:<38'] = np.where(df_inputs_prepr['mths_since_issue_d'] < 38, 1, 0)
df_inputs_prepr['mths_since_issue_d:38-39'] = np.where(df_inputs_prepr['mths_since_issue_d'].isin(range(38, 40)), 1, 0)
df_inputs_prepr['mths_since_issue_d:40-41'] = np.where(df_inputs_prepr['mths_since_issue_d'].isin(range(40, 42)), 1, 0)
df_inputs_prepr['mths_since_issue_d:42-48'] = np.where(df_inputs_prepr['mths_since_issue_d'].isin(range(42, 49)), 1, 0)
df_inputs_prepr['mths_since_issue_d:49-52'] = np.where(df_inputs_prepr['mths_since_issue_d'].isin(range(49, 53)), 1, 0)
df_inputs_prepr['mths_since_issue_d:53-64'] = np.where(df_inputs_prepr['mths_since_issue_d'].isin(range(53, 65)), 1, 0)
df_inputs_prepr['mths_since_issue_d:65-84'] = np.where(df_inputs_prepr['mths_since_issue_d'].isin(range(65, 85)), 1, 0)
# df_inputs_prepr['mths_since_issue_d:>84'] = np.where(df_inputs_prepr['mths_since_issue_d'].isin(range(85, int(df_inputs_prepr['mths_since_issue_d'].max()))), 1, 0)
df_inputs_prepr['mths_since_issue_d:>84'] = np.where(df_inputs_prepr['mths_since_issue_d'] > 84, 1, 0)

# ------  int_rate ---------------------------
# FINE Class to 50 buckets
df_inputs_prepr['int_rate_factor'] = pd.cut(df_inputs_prepr['int_rate'], 50)
df_temp = woe_ordered_continuous(df_inputs_prepr, 'int_rate_factor', df_targets_prep)
# plot_by_woe(df_temp, 90) # rotating the labels 90 degrees.
lst_IV = append_IV_list(df_temp, lst_IV) # append variable and IV values to list IV

# # COURSE class(new cats based on subj criteria). We create the following categories:
# '< 9.548', '9.548 - 12.025', '12.025 - 15.74', '15.74 - 20.281', '> 20.281'
df_inputs_prepr['int_rate:<9.548'] = np.where((df_inputs_prepr['int_rate'] < 9.548), 1, 0)
df_inputs_prepr['int_rate:9.548-12.025'] = np.where((df_inputs_prepr['int_rate'] >= 9.548) & (df_inputs_prepr['int_rate'] < 12.025), 1, 0)
df_inputs_prepr['int_rate:12.025-15.74'] = np.where((df_inputs_prepr['int_rate'] >= 12.025) & (df_inputs_prepr['int_rate'] < 15.74), 1, 0)
df_inputs_prepr['int_rate:15.74-20.281'] = np.where((df_inputs_prepr['int_rate'] >= 15.74) & (df_inputs_prepr['int_rate'] < 20.281), 1, 0)
df_inputs_prepr['int_rate:>20.281'] = np.where((df_inputs_prepr['int_rate'] >= 20.281), 1, 0)

# ------  funded_amnt ---------------------------
df_inputs_prepr['funded_amnt_factor'] = pd.cut(df_inputs_prepr['funded_amnt'], 50)
# Here we do fine-classing: using the 'cut' method, we split the variable into 50 categories by its values.
df_temp = woe_ordered_continuous(df_inputs_prepr, 'funded_amnt_factor', df_targets_prep)
# plot_by_woe(df_temp, 90) # rotating the labels 90 degrees.
lst_IV = append_IV_list(df_temp, lst_IV) # append variable and IV values to list IV

# ------  **  mths_since_earliest_cr_line ---------------------------
# df_inputs_prepr['mths_since_earliest_cr_line'].value_counts()
# FINE Class to 50 buckets
df_inputs_prepr['mths_since_earliest_cr_line_factor'] = pd.cut(df_inputs_prepr['mths_since_earliest_cr_line'], 50)
df_temp = woe_ordered_continuous(df_inputs_prepr, 'mths_since_earliest_cr_line_factor', df_targets_prep)
# plot_by_woe(df_temp, 45) # rotating the labels 90 degrees.
# plot_by_woe(df_temp.iloc[20: , : ], 45)
lst_IV = append_IV_list(df_temp, lst_IV) # append variable and IV values to list IV

# # COURSE class(new cats based on subj criteria). We create the following categories:
# < 140, # 141 - 164, # 165 - 247, # 248 - 270, # 271 - 352, # > 352
df_inputs_prepr['mths_since_earliest_cr_line:<140'] = np.where(df_inputs_prepr['mths_since_earliest_cr_line'].isin(range(140)), 1, 0)
df_inputs_prepr['mths_since_earliest_cr_line:141-164'] = np.where(df_inputs_prepr['mths_since_earliest_cr_line'].isin(range(140, 165)), 1, 0)
df_inputs_prepr['mths_since_earliest_cr_line:165-247'] = np.where(df_inputs_prepr['mths_since_earliest_cr_line'].isin(range(165, 248)), 1, 0)
df_inputs_prepr['mths_since_earliest_cr_line:248-270'] = np.where(df_inputs_prepr['mths_since_earliest_cr_line'].isin(range(248, 271)), 1, 0)
df_inputs_prepr['mths_since_earliest_cr_line:271-352'] = np.where(df_inputs_prepr['mths_since_earliest_cr_line'].isin(range(271, 353)), 1, 0)
df_inputs_prepr['mths_since_earliest_cr_line:>352'] = np.where(df_inputs_prepr['mths_since_earliest_cr_line'].isin(range(353, int(df_inputs_prepr['mths_since_earliest_cr_line'].max()))), 1, 0)

# line 719

# ------   delinq_2yrs ---------------------------
# df_inputs_prepr['delinq_2yrs'].value_counts()
# no FINE Class required, only few cats
df_temp = woe_ordered_continuous(df_inputs_prepr, 'delinq_2yrs', df_targets_prep)
# plot_by_woe(df_temp, 45) # rotating the labels 90 degrees.
lst_IV = append_IV_list(df_temp, lst_IV) # append variable and IV values to list IV

# COURSE class(new cats based on subj criteria). We create the following categories
# Categories: 0, 1-3, >=4
df_inputs_prepr['delinq_2yrs:0'] = np.where((df_inputs_prepr['delinq_2yrs'] == 0), 1, 0)
df_inputs_prepr['delinq_2yrs:1-3'] = np.where((df_inputs_prepr['delinq_2yrs'] >= 1) & (df_inputs_prepr['delinq_2yrs'] <= 3), 1, 0)
df_inputs_prepr['delinq_2yrs:>=4'] = np.where((df_inputs_prepr['delinq_2yrs'] >= 9), 1, 0)

# ------   inq_last_6mths ---------------------------
# df_inputs_prepr['inq_last_6mths'].value_counts()
# no FINE Class required, only few cats
df_temp = woe_ordered_continuous(df_inputs_prepr, 'inq_last_6mths', df_targets_prep)
# plot_by_woe(df_temp, 45) # rotating the labels 90 degrees.
lst_IV = append_IV_list(df_temp, lst_IV) # append variable and IV values to list IV

# COURSE class(new cats based on subj criteria). We create the following categories
# Categories: 0, 1 - 2, 3 - 6, > 6
df_inputs_prepr['inq_last_6mths:0'] = np.where((df_inputs_prepr['inq_last_6mths'] == 0), 1, 0)
df_inputs_prepr['inq_last_6mths:1-2'] = np.where((df_inputs_prepr['inq_last_6mths'] >= 1) & (df_inputs_prepr['inq_last_6mths'] <= 2), 1, 0)
df_inputs_prepr['inq_last_6mths:3-6'] = np.where((df_inputs_prepr['inq_last_6mths'] >= 3) & (df_inputs_prepr['inq_last_6mths'] <= 6), 1, 0)
df_inputs_prepr['inq_last_6mths:>6'] = np.where((df_inputs_prepr['inq_last_6mths'] > 6), 1, 0)

# ------   open_acc ---------------------------
# df_inputs_prepr['open_acc'].value_counts()
# no FINE Class required, only few cats
df_temp = woe_ordered_continuous(df_inputs_prepr, 'open_acc', df_targets_prep)
# plot_by_woe(df_temp, 45) # rotating the labels 90 degrees.
lst_IV = append_IV_list(df_temp, lst_IV) # append variable and IV values to list IV

# COURSE class(new cats based on subj criteria). We create the following categories
# Categories: '0', '1-3', '4-12', '13-17', '18-22', '23-25', '26-30', '>30'
df_inputs_prepr['open_acc:0'] = np.where((df_inputs_prepr['open_acc'] == 0), 1, 0)
df_inputs_prepr['open_acc:1-3'] = np.where((df_inputs_prepr['open_acc'] >= 1) & (df_inputs_prepr['open_acc'] <= 3), 1, 0)
df_inputs_prepr['open_acc:4-12'] = np.where((df_inputs_prepr['open_acc'] >= 4) & (df_inputs_prepr['open_acc'] <= 12), 1, 0)
df_inputs_prepr['open_acc:13-17'] = np.where((df_inputs_prepr['open_acc'] >= 13) & (df_inputs_prepr['open_acc'] <= 17), 1, 0)
df_inputs_prepr['open_acc:18-22'] = np.where((df_inputs_prepr['open_acc'] >= 18) & (df_inputs_prepr['open_acc'] <= 22), 1, 0)
df_inputs_prepr['open_acc:23-25'] = np.where((df_inputs_prepr['open_acc'] >= 23) & (df_inputs_prepr['open_acc'] <= 25), 1, 0)
df_inputs_prepr['open_acc:26-30'] = np.where((df_inputs_prepr['open_acc'] >= 26) & (df_inputs_prepr['open_acc'] <= 30), 1, 0)
df_inputs_prepr['open_acc:>=31'] = np.where((df_inputs_prepr['open_acc'] >= 31), 1, 0)

# ------   pub_rec ---------------------------
# df_inputs_prepr['pub_rec'].value_counts()
# no FINE Class required, only few cats
df_temp = woe_ordered_continuous(df_inputs_prepr, 'pub_rec', df_targets_prep)
# plot_by_woe(df_temp, 45) # rotating the labels 90 degrees.
lst_IV = append_IV_list(df_temp, lst_IV) # append variable and IV values to list IV

# COURSE class(new cats based on subj criteria). We create the following categories
# Categories '0-2', '3-4', '>=5'
df_inputs_prepr['pub_rec:0-2'] = np.where((df_inputs_prepr['pub_rec'] >= 0) & (df_inputs_prepr['pub_rec'] <= 2), 1, 0)
df_inputs_prepr['pub_rec:3-4'] = np.where((df_inputs_prepr['pub_rec'] >= 3) & (df_inputs_prepr['pub_rec'] <= 4), 1, 0)
df_inputs_prepr['pub_rec:>=5'] = np.where((df_inputs_prepr['pub_rec'] >= 5), 1, 0)

# ------   total_acc ---------------------------
# df_inputs_prepr['total_acc'].value_counts()
# FINE Class to 50 buckets
df_inputs_prepr['total_acc_factor'] = pd.cut(df_inputs_prepr['total_acc'], 50)
df_temp = woe_ordered_continuous(df_inputs_prepr, 'total_acc_factor', df_targets_prep)
# plot_by_woe(df_temp, 45) # rotating the labels 90 degrees.
lst_IV = append_IV_list(df_temp, lst_IV) # append variable and IV values to list IV

# COURSE class(new cats based on subj criteria). We create the following categories
# Categories: '<=27', '28-51', '>51'
df_inputs_prepr['total_acc:<=27'] = np.where((df_inputs_prepr['total_acc'] <= 27), 1, 0)
df_inputs_prepr['total_acc:28-51'] = np.where((df_inputs_prepr['total_acc'] >= 28) & (df_inputs_prepr['total_acc'] <= 51), 1, 0)
df_inputs_prepr['total_acc:>=52'] = np.where((df_inputs_prepr['total_acc'] >= 52), 1, 0)

# ------   acc_now_delinq ---------------------------
# df_inputs_prepr['acc_now_delinq'].value_counts()
# no FINE Class required, only few cats
df_temp = woe_ordered_continuous(df_inputs_prepr, 'acc_now_delinq', df_targets_prep)
# plot_by_woe(df_temp, 45) # rotating the labels 90 degrees.
lst_IV = append_IV_list(df_temp, lst_IV) # append variable and IV values to list IV

# COURSE class(new cats based on subj criteria). We create the following categories
# Categories: '0', '>=1'
df_inputs_prepr['acc_now_delinq:0'] = np.where((df_inputs_prepr['acc_now_delinq'] == 0), 1, 0)
df_inputs_prepr['acc_now_delinq:>=1'] = np.where((df_inputs_prepr['acc_now_delinq'] >= 1), 1, 0)

# ------   total_rev_hi_lim ---------------------------
# df_inputs_prepr['total_rev_hi_lim'].value_counts()
# FINE Class to 50 buckets
df_inputs_prepr['total_rev_hi_lim_factor'] = pd.cut(df_inputs_prepr['total_rev_hi_lim'], 2000)
df_temp = woe_ordered_continuous(df_inputs_prepr, 'total_rev_hi_lim_factor', df_targets_prep)
# plot_by_woe(df_temp, 45) # rotating the labels 90 degrees.
lst_IV = append_IV_list(df_temp, lst_IV) # append variable and IV values to list IV

# COURSE class(new cats based on subj criteria). We create the following categories
# '<=5K', '5K-10K', '10K-20K', '20K-30K', '30K-40K', '40K-55K', '55K-95K', '>95K'
df_inputs_prepr['total_rev_hi_lim:<=5K'] = np.where((df_inputs_prepr['total_rev_hi_lim'] <= 5000), 1, 0)
df_inputs_prepr['total_rev_hi_lim:5K-10K'] = np.where((df_inputs_prepr['total_rev_hi_lim'] > 5000) & (df_inputs_prepr['total_rev_hi_lim'] <= 10000), 1, 0)
df_inputs_prepr['total_rev_hi_lim:10K-20K'] = np.where((df_inputs_prepr['total_rev_hi_lim'] > 10000) & (df_inputs_prepr['total_rev_hi_lim'] <= 20000), 1, 0)
df_inputs_prepr['total_rev_hi_lim:20K-30K'] = np.where((df_inputs_prepr['total_rev_hi_lim'] > 20000) & (df_inputs_prepr['total_rev_hi_lim'] <= 30000), 1, 0)
df_inputs_prepr['total_rev_hi_lim:30K-40K'] = np.where((df_inputs_prepr['total_rev_hi_lim'] > 30000) & (df_inputs_prepr['total_rev_hi_lim'] <= 40000), 1, 0)
df_inputs_prepr['total_rev_hi_lim:40K-55K'] = np.where((df_inputs_prepr['total_rev_hi_lim'] > 40000) & (df_inputs_prepr['total_rev_hi_lim'] <= 55000), 1, 0)
df_inputs_prepr['total_rev_hi_lim:55K-95K'] = np.where((df_inputs_prepr['total_rev_hi_lim'] > 55000) & (df_inputs_prepr['total_rev_hi_lim'] <= 95000), 1, 0)
df_inputs_prepr['total_rev_hi_lim:>95K'] = np.where((df_inputs_prepr['total_rev_hi_lim'] > 95000), 1, 0)

# ------  installment ---------------------------
# df_inputs_prepr['installment'].value_counts()
# FINE Class to 50 buckets
df_inputs_prepr['installment_factor'] = pd.cut(df_inputs_prepr['installment'], 50)
df_temp = woe_ordered_continuous(df_inputs_prepr, 'installment_factor', df_targets_prep)
# plot_by_woe(df_temp, 45) # rotating the labels 90 degrees.
lst_IV = append_IV_list(df_temp, lst_IV) # append variable and IV values to list IV

# COURSE class(new cats based on subj criteria). We create the following categories
# ?

### Preprocessing Continuous Variables: Creating Dummy Variables, Part 3
# ------   annual_inc ---------------------------
# df_inputs_prepr['annual_inc'].value_counts()
# FINE Class to 50 buckets
df_inputs_prepr['annual_inc_factor'] = pd.cut(df_inputs_prepr['annual_inc'], 100)

df_temp = woe_ordered_continuous(df_inputs_prepr, 'annual_inc_factor', df_targets_prep)
# plot_by_woe(df_temp, 45) # rotating the labels 90 degrees.
lst_IV = append_IV_list(df_temp, lst_IV) # append variable and IV values to list IV

## ***
# Initial examination shows that there are too few individuals with large income and too many with small income.
# Hence, we are going to have one category for more than 150K, and we are going to apply our approach to determine
# the categories of everyone with 140k or less.
df_inputs_prepr_temp = df_inputs_prepr.loc[df_inputs_prepr['annual_inc'] <= 140000, : ]
#loan_data_temp = loan_data_temp.reset_index(drop = True)
#df_inputs_prepr_temp

df_inputs_prepr_temp["annual_inc_factor"] = pd.cut(df_inputs_prepr_temp['annual_inc'], 50)
# Here we do fine-classing: using the 'cut' method, we split the variable into 50 categories by its values.
# note: the target needs to be in line with the inputs index. We use the .index attribute to only get targets with the same index as the inputs
df_temp = woe_ordered_continuous(df_inputs_prepr_temp, 'annual_inc_factor', df_targets_prep[df_inputs_prepr_temp.index])
# plot_by_woe(df_temp, 45) # rotating the labels 90 degrees.
lst_IV = append_IV_list(df_temp, lst_IV) # append variable and IV values to list IV

# COURSE class(new cats based on subj criteria). We create the following categories
# WoE is monotonically decreasing with income, so we split income in 10 equal categories, each with width of 15k.
df_inputs_prepr['annual_inc:<20K'] = np.where((df_inputs_prepr['annual_inc'] <= 20000), 1, 0)
df_inputs_prepr['annual_inc:20K-30K'] = np.where((df_inputs_prepr['annual_inc'] > 20000) & (df_inputs_prepr['annual_inc'] <= 30000), 1, 0)
df_inputs_prepr['annual_inc:30K-40K'] = np.where((df_inputs_prepr['annual_inc'] > 30000) & (df_inputs_prepr['annual_inc'] <= 40000), 1, 0)
df_inputs_prepr['annual_inc:40K-50K'] = np.where((df_inputs_prepr['annual_inc'] > 40000) & (df_inputs_prepr['annual_inc'] <= 50000), 1, 0)
df_inputs_prepr['annual_inc:50K-60K'] = np.where((df_inputs_prepr['annual_inc'] > 50000) & (df_inputs_prepr['annual_inc'] <= 60000), 1, 0)
df_inputs_prepr['annual_inc:60K-70K'] = np.where((df_inputs_prepr['annual_inc'] > 60000) & (df_inputs_prepr['annual_inc'] <= 70000), 1, 0)
df_inputs_prepr['annual_inc:70K-80K'] = np.where((df_inputs_prepr['annual_inc'] > 70000) & (df_inputs_prepr['annual_inc'] <= 80000), 1, 0)
df_inputs_prepr['annual_inc:80K-90K'] = np.where((df_inputs_prepr['annual_inc'] > 80000) & (df_inputs_prepr['annual_inc'] <= 90000), 1, 0)
df_inputs_prepr['annual_inc:90K-100K'] = np.where((df_inputs_prepr['annual_inc'] > 90000) & (df_inputs_prepr['annual_inc'] <= 100000), 1, 0)
df_inputs_prepr['annual_inc:100K-120K'] = np.where((df_inputs_prepr['annual_inc'] > 100000) & (df_inputs_prepr['annual_inc'] <= 120000), 1, 0)
df_inputs_prepr['annual_inc:120K-140K'] = np.where((df_inputs_prepr['annual_inc'] > 120000) & (df_inputs_prepr['annual_inc'] <= 140000), 1, 0)
df_inputs_prepr['annual_inc:>140K'] = np.where((df_inputs_prepr['annual_inc'] > 140000), 1, 0)

# ------   mths_since_last_delinq ---------------------------
# df_inputs_prepr['mths_since_last_delinq'].value_counts()

# We have to create one category for missing values and do fine and coarse classing for the rest.
# We use pd.notnull () function
df_inputs_prepr_temp = df_inputs_prepr[pd.notnull(df_inputs_prepr['mths_since_last_delinq'])]
df_inputs_prepr_temp['mths_since_last_delinq_factor'] = pd.cut(df_inputs_prepr_temp['mths_since_last_delinq'], 50)
# df_inputs_prepr_temp.iloc[:,-1] = pd.cut(df_inputs_prepr_temp['mths_since_last_delinq'], 50) # get last column -1

# we usd .index method to match targets and inputs based on index
df_temp = woe_ordered_continuous(df_inputs_prepr_temp, 'mths_since_last_delinq_factor', df_targets_prep[df_inputs_prepr_temp.index])

# plot_by_woe(df_temp, 45) # rotating the labels 90 degrees.
lst_IV = append_IV_list(df_temp, lst_IV) # append variable and IV values to list IV

# COURSE class(new cats based on subj criteria). We create the following categories
# Categories: Missing, 0-3, 4-30, 31-56, >=57
df_inputs_prepr['mths_since_last_delinq:Missing'] = np.where((df_inputs_prepr['mths_since_last_delinq'].isnull()), 1, 0)
df_inputs_prepr['mths_since_last_delinq:0-3'] = np.where((df_inputs_prepr['mths_since_last_delinq'] >= 0) & (df_inputs_prepr['mths_since_last_delinq'] <= 3), 1, 0)
df_inputs_prepr['mths_since_last_delinq:4-30'] = np.where((df_inputs_prepr['mths_since_last_delinq'] >= 4) & (df_inputs_prepr['mths_since_last_delinq'] <= 30), 1, 0)
df_inputs_prepr['mths_since_last_delinq:31-56'] = np.where((df_inputs_prepr['mths_since_last_delinq'] >= 31) & (df_inputs_prepr['mths_since_last_delinq'] <= 56), 1, 0)
df_inputs_prepr['mths_since_last_delinq:>=57'] = np.where((df_inputs_prepr['mths_since_last_delinq'] >= 57), 1, 0)

### Sec5.chp 37  Preprocessing Continuous Variables: Creating Dummy Variables, Part 3: Homework

# ------  dti ---------------------------
# df_inputs_prepr['dti'].value_counts()
# FINE Class to 100 buckets
df_inputs_prepr['dti_factor'] = pd.cut(df_inputs_prepr['dti'], 100)
df_temp = woe_ordered_continuous(df_inputs_prepr, 'dti_factor', df_targets_prep)

# Similarly to income, initial examination shows that most values are lower than 200.
# Hence, we are going to have one category for more than 35, and we are going to apply our approach to determine
# the categories of everyone with 150k or less.
df_inputs_prepr_temp = df_inputs_prepr.loc[df_inputs_prepr['dti'] <= 35, : ]

df_inputs_prepr_temp['dti_factor'] = pd.cut(df_inputs_prepr_temp['dti'], 50)
# Here we do fine-classing: using the 'cut' method, we split the variable into 50 categories by its values.
df_temp = woe_ordered_continuous(df_inputs_prepr_temp, 'dti_factor', df_targets_prep[df_inputs_prepr_temp.index])

# plot_by_woe(df_temp, 45) # rotating the labels 90 degrees.
lst_IV = append_IV_list(df_temp, lst_IV) # append variable and IV values to list IV

# COURSE class(new cats based on subj criteria). We create the following categories
df_inputs_prepr['dti:<=1.4'] = np.where((df_inputs_prepr['dti'] <= 1.4), 1, 0)
df_inputs_prepr['dti:1.4-3.5'] = np.where((df_inputs_prepr['dti'] > 1.4) & (df_inputs_prepr['dti'] <= 3.5), 1, 0)
df_inputs_prepr['dti:3.5-7.7'] = np.where((df_inputs_prepr['dti'] > 3.5) & (df_inputs_prepr['dti'] <= 7.7), 1, 0)
df_inputs_prepr['dti:7.7-10.5'] = np.where((df_inputs_prepr['dti'] > 7.7) & (df_inputs_prepr['dti'] <= 10.5), 1, 0)
df_inputs_prepr['dti:10.5-16.1'] = np.where((df_inputs_prepr['dti'] > 10.5) & (df_inputs_prepr['dti'] <= 16.1), 1, 0)
df_inputs_prepr['dti:16.1-20.3'] = np.where((df_inputs_prepr['dti'] > 16.1) & (df_inputs_prepr['dti'] <= 20.3), 1, 0)
df_inputs_prepr['dti:20.3-21.7'] = np.where((df_inputs_prepr['dti'] > 20.3) & (df_inputs_prepr['dti'] <= 21.7), 1, 0)
df_inputs_prepr['dti:21.7-22.4'] = np.where((df_inputs_prepr['dti'] > 21.7) & (df_inputs_prepr['dti'] <= 22.4), 1, 0)
df_inputs_prepr['dti:22.4-35'] = np.where((df_inputs_prepr['dti'] > 22.4) & (df_inputs_prepr['dti'] <= 35), 1, 0)
df_inputs_prepr['dti:>35'] = np.where((df_inputs_prepr['dti'] > 35), 1, 0)

# ------  mths_since_last_record ---------------------------
# df_inputs_prepr['var'].value_counts()

# We have to create one category for missing values and do fine and coarse classing for the rest.
df_inputs_prepr_temp = df_inputs_prepr[pd.notnull(df_inputs_prepr['mths_since_last_record'])]
#sum(loan_data_temp['mths_since_last_record'].isnull())

# FINE Class to 50 buckets
df_inputs_prepr_temp['mths_since_last_record_factor'] = pd.cut(df_inputs_prepr_temp['mths_since_last_record'], 50)

df_temp = woe_ordered_continuous(df_inputs_prepr_temp, 'mths_since_last_record_factor', df_targets_prep[df_inputs_prepr_temp.index])
# plot_by_woe(df_temp, 45) # rotating the labels 90 degrees.
lst_IV = append_IV_list(df_temp, lst_IV) # append variable and IV values to list IV

# COURSE class(new cats based on subj criteria). We create the following categories
# Categories: 'Missing', '0-2', '3-20', '21-31', '32-80', '81-86', '>86'
df_inputs_prepr['mths_since_last_record:Missing'] = np.where((df_inputs_prepr['mths_since_last_record'].isnull()), 1, 0)
df_inputs_prepr['mths_since_last_record:0-2'] = np.where((df_inputs_prepr['mths_since_last_record'] >= 0) & (df_inputs_prepr['mths_since_last_record'] <= 2), 1, 0)
df_inputs_prepr['mths_since_last_record:3-20'] = np.where((df_inputs_prepr['mths_since_last_record'] >= 3) & (df_inputs_prepr['mths_since_last_record'] <= 20), 1, 0)
df_inputs_prepr['mths_since_last_record:21-31'] = np.where((df_inputs_prepr['mths_since_last_record'] >= 21) & (df_inputs_prepr['mths_since_last_record'] <= 31), 1, 0)
df_inputs_prepr['mths_since_last_record:32-80'] = np.where((df_inputs_prepr['mths_since_last_record'] >= 32) & (df_inputs_prepr['mths_since_last_record'] <= 80), 1, 0)
df_inputs_prepr['mths_since_last_record:81-86'] = np.where((df_inputs_prepr['mths_since_last_record'] >= 81) & (df_inputs_prepr['mths_since_last_record'] <= 86), 1, 0)
df_inputs_prepr['mths_since_last_record:>86'] = np.where((df_inputs_prepr['mths_since_last_record'] > 86), 1, 0)


# ********************************
### Sec5 chp 38, Preprocessing the Test Dataset

#-----------------------------------------
# remeber: delete this part

# Data Preparation: load Training data: loan_data_inputs_train, loan_data_targets_train
# create df for preprocessing.  calculate WoE and IV   / # REF: Section 5, video 26
# df_inputs_prepr = loan_data_inputs_train
# df_targets_prep = loan_data_targets_train
#-----------------------------------------

'''
shape of df's
loan_data_inputs_train.shape  # (373028, 324)
loan_data_targets_train.shape # (373028,)
loan_data_inputs_test.shape #  (93257, 324)
loan_data_targets_test.shape # (93257,)
'''

# use temp df (df_inputs_prepr) to set main df's
'''
# a) train set
loan_data_inputs_train = df_inputs_prepr
# b) test set
loan_data_inputs_test = df_inputs_prepr
'''

# Save pe-processed data as CSV for modelling
loan_data_inputs_train.to_csv('data/loan_data_inputs_train.csv')
loan_data_targets_train.to_csv('data/loan_data_targets_train.csv')
loan_data_inputs_test.to_csv('data/loan_data_inputs_test.csv')
loan_data_targets_test.to_csv('data/loan_data_targets_test.csv')

# Save pe-processed data as .FEATHER for modelling
loan_data_inputs_train.reset_index().to_feather('data/loan_data_inputs_train.feather')
loan_data_targets_train.reset_index().to_feather('data/loan_data_targets_train.feather')
loan_data_inputs_test.reset_index().to_feather('data/loan_data_inputs_test.feather')
loan_data_targets_test.reset_index().to_feather('data/loan_data_targets_test.feather')



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