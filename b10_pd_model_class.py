
# Class that inherits from pandas.DataFrame then customizes it with additonal methods.

import numpy as np
import pandas as pd
from pandas import DataFrame
import matplotlib.pyplot as plt
import seaborn as sns
sns.set() #overwrite the default matplotlib look. Think of it like a skin plt

class PdDataframe(DataFrame):
    # constuctor
    def __init__(self, *args, **kwargs):
      super().__init__(*args, **kwargs)
      print('Object created, change 0')

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

    #  Preprocessing CONTINOUS variables: automating calculations of WoE for discrete vars. Ref: S5.27
    def woe_ordered_continuous(self, discrete_variable_name, df_good_bad_variable):
        # get column from inputs df and good_bad column from Targets df. merge the two df.
        df = pd.concat([self[discrete_variable_name], df_good_bad_variable], axis=1)
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
        # calculate WoE for the variable grade. Handle divide by zero by replacing 0 with NaN
        # df['WoE'] = np.log(df['prop_n_good'] / df['prop_n_bad'].replace(0, np.nan, inplace = True))
        df['WoE'] = np.log(df['prop_n_good'] / df['prop_n_bad'])
        # In continous, we remove sort by WoE and keep the natural sort of the input df
        df['WoE'].replace([np.inf, -np.inf], np.nan, inplace=True)  # replace infinite to NaN
        # calculate Information Value
        df['IV'] = (df['prop_n_good'] - df['prop_n_bad']) * df['WoE']
        df['IV'] = df['IV'].sum()

        return df

    # Visualizing results. sec.5 L28:
    # matplotlib works well with np.array but not df and strings. Same goes for scipy
    def plot_by_woe(df, roation_of_axis_labels=0, width=15, height=7):
        x = np.array(df.iloc[:, 0].apply(str))  # Convert values into srting, then convert to an array.
        y = df['WoE']  # its a numeric variable so no need to do anything about it.
        plt.figure(figsize=(width, height))  # specify dimension of the chart.  (figsize = (Width(X), Height(Y)

        # now plot the data. Mark o for each point, tied by a dotted line, with color black(k)
        plt.plot(x, y, marker='o', linestyle='--', color='k')
        plt.xlabel(df.columns[0])
        plt.ylabel('Weight of Evidence')
        plt.title(str("Weight of Evidence by " + (df.columns[0])))
        plt.xticks(rotation=roation_of_axis_labels)
        plt.show()

    # Calling the plot function example:
    # plot_by_woe(df1_grade)

    def append_IV_list(self, lst):
        # if lst is None:
        #     lst = []
        val = [self.columns[0], self.iloc[0, 9]]
        lst.append(val)

        return lst

    # destructor
    def __del__(self):
        print('Object destroyed')

'''
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
'''