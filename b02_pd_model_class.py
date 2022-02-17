
# Class that inherits from pandas.DataFrame then customizes it with additonal methods.

import numpy as np
import pandas as pd
from pandas import DataFrame

class PdDataframe(DataFrame):
    # constuctor
    def __init__(self, *args, **kwargs):
      super().__init__(*args, **kwargs)
      print('Object created, change 0')

    # @property
    # def _constructor(self):
    #     return MyDataframe

    def sum_column(self, column_name):
        return self[column_name].sum()

    def groupby_sum(self, groupby_col, aggregate_col, aggregate_method):

        if aggregate_method == 'sum':
            df = self.groupby(groupby_col)[aggregate_col].sum()
        elif aggregate_method == 'mean':
            df = self.groupby(groupby_col)[aggregate_col].mean()
        else:
            df = self.groupby(groupby_col)[aggregate_col].count()
        return df

    #  Preprocessing DISCRETE variables: automating calculations of WoE for discrete vars. Ref: S5.27
    def woe_discrete(self, discrete_variable_name, df_good_bad_variable):
        # get column from inputs df and good_bad column from Targets df. merge the two df.
        df = pd.concat([self[discrete_variable_name], df_good_bad_variable], axis=1)
        # merge the two results into one df. note: as_index=False means group by the column names, and not by the
        # index, =False,is same as SQL group by
        # aggregate mean() is the mean of true=1=good values. therefore, the mean is the percentage of Good values
        # ex. 88/100 = 88%
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
        # calculate WoE for the variable grade.
        df['WoE'] = np.log(df['prop_n_good'] / df['prop_n_bad'])
        # sort categories with the highest default rate first, then reset index
        df = df.sort_values(['WoE'])
        df = df.reset_index(drop=True)  # here, there's no groupby so we reset the index afer df sort.
        df['WoE'].replace([np.inf, -np.inf], np.nan, inplace=True)  # replace infinite to NaN
        # calculate Information Value
        df['IV'] = (df['prop_n_good'] - df['prop_n_bad']) * df['WoE']
        df['IV'] = df['IV'].sum()
        return df

    # destructor
    def __del__(self):
        print('Object destroyed')

