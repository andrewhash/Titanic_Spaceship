
"""
Created on Sun Apr 28 16:16:07 2024

@author: andrewhashoush
"""

import pandas as pd
import sklearn.model_selection
import matplotlib.pyplot as plt

df = pd.read_csv('/Users/andrewhashoush/Downloads/spaceship-titanic/train.csv')
df_test = pd.read_csv('/Users/andrewhashoush/Downloads/spaceship-titanic/test.csv')

print(df.head())

print(df.info())