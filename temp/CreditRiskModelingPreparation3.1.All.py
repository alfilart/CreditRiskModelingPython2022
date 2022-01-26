#!/usr/bin/env python
# coding: utf-8

# Data Preparation

## Import Libraries
import numpy as np
import pandas as pd

## Import Data
loan_data_backup = pd.read_csv('loan_data_2007_2014.csv')
loan_data = loan_data_backup.copy()

## Explore Data
loan_data

pd.options.display.max_columns = None

loan_data

loan_data.head()
loan_data.tail()
loan_data.columns.values
loan_data.info()

