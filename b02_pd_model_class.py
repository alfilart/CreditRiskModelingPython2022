# Class DataFrame inputs
# https://stackoverflow.com/questions/29459186/building-class-that-inherits-pandas-dataframe/48507776


# https://stackoverflow.com/questions/70125762/python-creating-classes-calculation-methods-with-dataframe-inputs


from pandas import DataFrame

class MyDataframe(DataFrame):
    #Class that inherits from pandas.DataFrame then customizes it with additonal methods.
    def __init__(self, *args, **kwargs):
      super().__init__(*args, **kwargs)

    @property
    def _constructor(self):
        return MyDataframe

    def sum_column(self, column_name):
        return self[column_name].sum()

'''
# v2
#import libraries
from pandas import DataFrame

class PrepPandas(DataFrame):
    #Class that inherits from pandas.DataFrame then customizes it with additonal methods.
    def __init__(self, *args, **kwargs):
      super().__init__(*args, **kwargs)

    @property
    def _constructor(self):
        return PrepPandas

'''
