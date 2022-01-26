# Aurélien Géron. Hands-on Machine Learning with Scikit-Learn, Keras, and TensorFlow, 2nd Edition
# Chp. 4, Topic 6 - Logistic Regression p. 142-50

# Build a classifier to detect the Iris-Virginica type based only on the petal width feature.

# data set:
# https://scikit-learn.org/stable/auto_examples/datasets/plot_iris_dataset.html
# https://en.wikipedia.org/wiki/Iris_flower_data_set
# input: rows=samples  ; columns=Sepal Length, Sepal Width, Petal Length and Petal Width
# stored in a 150x4 numpy.ndarray
# Instances: 150 (50 in each of three classes)
# Class or target: ['setosa', 'versicolor', 'virginica'] == 0,1,2
# Attributes/Features: 4 numeric 'sepal length (cm)','sepal width (cm)','petal length (cm)','petal width (cm)'
# Iris is a Bunch object, a Container object exposing keys as attributes.
# Bunch objects are sometimes used as an output for functions and methods. They extend dictionaries by enabling
# values to be accessed by "key", bunch["value_key"], or by an "attribute", bunch.value_key.

#-------------------------
# import libraries
#------------------------
import numpy as np
import pandas as pd   # https://pandas.pydata.org/pandas-docs/stable/user_guide/index.html
import matplotlib.pyplot as plt
import seaborn as sns

#------------------------
# Load the Iris dataset. Default input will be cast to float64
#------------------------
from sklearn import datasets
data = datasets.load_iris() # data is of datatype Bunch

#-------------------------
# Review/Inspect data set
#------------------------

# ?data  # Show information about the bunch dataset
print(data.keys())   # out: ['data', 'target', 'target_names', 'DESCR', 'feature_names', 'filename']

# data and target keys are just numpy arrays
print(type(data['data']), data['data'].shape) # same: data.data , data.data.shape
print(type(data['target']), data['target'].shape) # same: data.target, data.target.shape
print(data['target_names']) # same: data.target_names
print(data['feature_names']) # same: data.feature_names

# bunch type doesn't have hits-dc. To use hits-dc, must be converted into a pandas df
# sklearn.utils.bunch convert to pandas dataframe
df_data = pd.DataFrame(data.data, columns=data.feature_names)
df_data.head()
df_data.describe()

#-------------------------
#  Data Preparation / Pre-processing
#------------------------
X = data['data']
y = data['target']

# Merge data(X) and target(y)
values = np.c_[X, y].astype(np.float32)

# stuff it into a dataframe
df = pd.DataFrame(values)
df.info(); df.head()


##  label the data
#  create list of column names
cols = data['feature_names'] + ['flower_codes']
# rename columns with cols list
df.columns = cols
df.info(); df.head()

## un-encode the flower_names column
# verbose, but also generic
d = dict(zip(range(len(data['target_names'])), data['target_names']))  # zip(key, value) pairs. zip pairs the 2 sets in prep for dict.

# help(dict)  -> new dictionary initialized from a mapping object's (key, value) pairs or an iterable
# help(zip)  -> The zip object yields n-length tuples, where n is the number of iterables passed as positional arguments to zip()

# pair the df with column==flower_names against the dict  using map() method
df['flower_names'] = df['flower_codes'].map(d)  # map = mapping correspondence
df.info(); df.head()

# RESOURCES:
#*** https://napsterinblue.github.io/notes/machine_learning/datasets/iris/
#** https://refactored.ai/microcourse/notebook?path=content%2F02-Python_for_Data_Scientists%2F09-Data_Analysis_with_Pandas%2FLab-Dataframes.ipynb
#https://pandas.pydata.org/pandas-docs/stable/user_guide/merging.html


#-------------------------
# Visualize data set
#------------------------
# Create scatterplots of all pairwise combination of the 4 variables in the dataset

sns.pairplot(df)
sns.pairplot(df, hue='flower_names')

#*_todo use of args: hue_order, palette
sns.pairplot(df, hue='flower_names',palette=['blue','orange','green'])
sns.pairplot(df, hue='flower_names',palette='Set2')
sns.pairplot(df, hue='flower_names',palette='Dark2')
# palette favourites'Accent','Paired','Paired', 'Pastel2

custom_palette = sns.color_palette("Set2")
sns.palplot(custom_palette)

#RESOURCES:
# https://seaborn.pydata.org/generated/seaborn.pairplot.html
## SEABORN(SNS)COLOR PALETES
# https://seaborn.pydata.org/tutorial/color_palettes.html
# https://www.geeksforgeeks.org/seaborn-color-palette/
# https://medium.com/@morganjonesartist/color-guide-to-seaborn-palettes-da849406d44f
# https://colorbrewer2.org/#type=sequential&scheme=BuGn&n=3


#------------------------
# Load selected features into X and targets into Y
# Attributes/Features: 4 numeric '0=sepal length (cm)',1='sepal width (cm)',2='petal length (cm)',3='petal width (cm)'
#------------------------
# petal width, we only take feature 3
X = data["data"][:, 3:]  # ,3:] creates a shape {tuple:2}(150,1) single column

# target: ['setosa', 'versicolor', 'virginica'] == 0,1,2
# 1(True) if Iris-Virginica, else 0(False)
y = (data["target"] == 2).astype(int)  #shape (150,) vector
# np.int is depreciated see: https://numpy.org/devdocs/release/1.20.0-notes.html#deprecations

#-------------------------
# train a Logistic Regression model == log_reg
#------------------------
from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression()
log_reg.fit(X, y)

#-------------------------
# View/Visualize model’s estimated probabilities for flowers with petal widths varying from 0 to 3 cm
# there is a decision boundary at around 1.6 cm where both probabilities are equal to 50%
#------------------------
X_new = np.linspace(0, 3, 1000).reshape(-1, 1)
y_proba = log_reg.predict_proba(X_new) # log_reg will only have 2 class (Binomial)
plt.plot(X_new, y_proba[:, 1], "g-", label="Iris-Virginica")
plt.plot(X_new, y_proba[:, 0], "b--", label="Not Iris-Virginica")
plt.legend()
plt.xlabel("Petal width (cm)")
plt.ylabel("Probability")
plt.show()


plt.plot(X_new, y_proba[:, 1], linestyle='-', color='green')
plt.plot(X_new, y_proba[:, 0], linestyle='--', color='blue')
plt.xlabel("Iris-Virginica")
plt.ylabel("Not Iris-Virginica")
plt.legend()
plt.show()
# + more Matplotlib code to make the image look pretty

#-------------------------
# Predict using
#------------------------

log_reg.predict([[1.7], [1.5]])
# out: array([1, 0])

#-------------------------
# Clear Console
#------------------------

import os
clear = lambda: os.system('cls')
clear()