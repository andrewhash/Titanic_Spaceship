
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


# Most of the passengers were between 17-32 years old
plt.figure()
plt.hist(df['Age'])
plt.title('Age of Passengers')
plt.xlabel('Age')
plt.ylabel('Count')
plt.grid(True)
plt.show()

print(df.columns)

#Split Cabin 
df[['Deck', 'Cabin_num', 'Side']] = df['Cabin'].str.split("/", expand=True)
df_test[['Deck', 'Cabin_num', 'Side']] = df_test['Cabin'].str.split("/", expand=True)

#Fill the missing values with the mode
def fill_mode(column):
    mode = df[column].mode()[0]
    df[column] = df[column].fillna(mode)
    df_test[column] = df_test[column].fillna(mode)
fill_mode('Deck')
fill_mode('Cabin_num')
fill_mode('Side')

df = df.drop('Cabin', axis=1)
df_test = df_test.drop('Cabin', axis=1)

deck_counts = df['Deck'].value_counts()
plt.figure()
deck_counts.plot()
plt.title('Frequency of Passengers by Deck')
plt.xlabel('Deck')
plt.ylabel('Number of Passengers')
plt.show()
