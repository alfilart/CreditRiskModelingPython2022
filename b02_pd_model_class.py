
# Class that inherits from pandas.DataFrame then customizes it with additonal methods.

from pandas import DataFrame

class PdDataframe(DataFrame):
    # constuctor
    def __init__(self, *args, **kwargs):
      super().__init__(*args, **kwargs)

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

    # destructor
    def __del__(self):
        print('Object destroyed')

