
"""
@author: andrewhashoush
"""
import pandas as pd
import numpy as np
import sklearn.model_selection
import matplotlib.pyplot as plt
import sklearn as sk
import sklearn.tree
import sklearn.ensemble

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

#changing missing values to Mode
categorical_columns = ['HomePlanet', 'CryoSleep','Destination', 'VIP']
for column in categorical_columns:
    mode_value = df[column].mode()[0]
    df[column] = df[column].fillna(mode_value)
    df_test[column] = df_test[column].fillna(mode_value)
    #Checking if It worked
    print(f"Missing values in {column}: {df[column].isnull().sum()}")
    
#changing missing values to 0 since it means they didn't spend any extra money
spending_columns = ['RoomService', 'FoodCourt', 'ShoppingMall', 'Spa', 'VRDeck']
for column in spending_columns:
    df[column] = df[column].fillna(0)
    df_test[column] = df_test[column].fillna(0)

#Changing missing value to Median for Age
median_age = df['Age'].median()
df['Age'] = df['Age'].fillna(median_age)
df_test['Age']= df_test['Age'].fillna(median_age)
print(f"Age Values Missing: {df['Age'].isnull().sum()}")

#Changing Dummy Variables to 0 or 1
df = pd.get_dummies(df, columns=['HomePlanet', 'CryoSleep', 'Destination', 'VIP', 'Deck', 'Side'])
df_test = pd.get_dummies(df_test, columns=['HomePlanet', 'CryoSleep','Destination', 'VIP', 'Deck', 'Side'])

#Split Data
df_train, df_val = sklearn.model_selection.train_test_split(df, train_size =.8, random_state= 123)
X_train, y_train = df_train.drop('Transported', axis=1), df_train['Transported']
X_val, y_val = df_val.drop('Transported', axis=1), df_val['Transported']

#I'm Dropping Name as this varibale does not affect the data
X_train = X_train.drop('Name', axis=1)
X_val = X_val.drop('Name', axis=1)

#Decision Tree Classifier
dt_model = sk.tree.DecisionTreeClassifier(max_depth = 5,random_state=123)
dt_model.fit(X_train, y_train)
dt_val_pred = dt_model.predict(X_val).astype('float')
dt_Score = np.mean(dt_val_pred  == y_val)
print(f"Decision Tree : {dt_Score}")

# Random Forest Classifier
rf_model = sk.ensemble.RandomForestClassifier(n_estimators=300,random_state=123)
rf_model.fit(X_train, y_train)
rf_val_pred = rf_model.predict(X_val)
rf_Score = np.mean(rf_val_pred  == y_val)
print(f"Random Forest: {rf_Score}")

# Gradient Boosting Classifier
gb_model = sk.ensemble.GradientBoostingClassifier(n_estimators=150, learning_rate=0.1,random_state=123)
gb_model.fit(X_train, y_train)
gb_val_pred = gb_model.predict(X_val)
gb_Score = np.mean(gb_val_pred  == y_val)
print(f"Gradient Boosting Score: {gb_Score}")

#The best classifer seems to be Random Forest with 0.819, however Gradient Boosting gives a little higher score om Kaggle
final_pred = gb_model.predict(df_test.drop('Name', axis=1)) 
submission_df = pd.DataFrame({ 'PassengerId': df_test['PassengerId'], 'Transported': final_pred})
submission_df.to_csv('/Users/andrewhashoush/Downloads/spaceship-titanic/sample_submission.csv', index=False)
