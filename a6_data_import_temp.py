


import numpy as np
import pandas as pd

loan_data_inputs_train = pd.read_csv('data/loan_data_inputs_train.csv', index_col = 0)
loan_data_targets_train = pd.read_csv('data/loan_data_targets_train.csv', index_col = 0) # , header = None)
loan_data_inputs_test = pd.read_csv('data/loan_data_inputs_test.csv', index_col = 0)
loan_data_targets_test = pd.read_csv('data/loan_data_targets_test.csv', index_col = 0) # , header = None)


loan_data_targets_train.reshape(373029,)


loan_data_targets_train = pd.read_csv('data/loan_data_targets_train.csv')

loan_data_inputs_train.head()
loan_data_inputs_train.index

loan_data_targets_train.head()
loan_data_targets_train.index
loan_data_targets_train.shape

loan_data_targets_train.dropna(how ='any')


loan_data_inputs_test.head()
loan_data_inputs_test.index

loan_data_targets_test.head()
loan_data_targets_test.index


# Save pe-processed data as .FEATHER for modelling
loan_data_inputs_train.to_feather('data/loan_data_inputs_train.feather')
loan_data_targets_train.to_feather('data/loan_data_targets_train.feather')
loan_data_inputs_test.to_feather('data/loan_data_inputs_test.feather')
loan_data_targets_test.to_feather('data/loan_data_targets_test.feather')


# Save pe-processed data as .FEATHER for modelling
loan_data_inputs_train.reset_index().to_feather('data/loan_data_inputs_train.feather')
loan_data_targets_train.reset_index().to_feather('data/loan_data_targets_train.feather')
loan_data_inputs_test.reset_index().to_feather('data/loan_data_inputs_test.feather')
loan_data_targets_test.reset_index().to_feather('data/loan_data_targets_test.feather')

#*************************************

loan_data_inputs_train = pd.read_feather('data/loan_data_inputs_train.feather').set_index('index')
loan_data_targets_train = pd.read_feather('data/loan_data_targets_train.feather').set_index('index')
loan_data_inputs_test = pd.read_feather('data/loan_data_inputs_test.feather').set_index('index')
loan_data_targets_test = pd.read_feather('data/loan_data_targets_test.feather').set_index('index')





