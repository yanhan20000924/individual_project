# Multiple Linear Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('all_data.csv')
X = dataset.iloc[:, [0, 1, 4, 12, 13]]
y = dataset.iloc[:, 15]
print(X)
print(y)

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Fitting Multiple Linear Regression to the Training set
from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(X_train, y_train)

# Predicting the Test set results
y_pred = regressor.predict(X_test)
print(X_test)
print(y_pred)
# use NO.32 as the test set
x_test = X.iloc[30:31, :]
print(x_test)
Y_predict = regressor.predict(x_test)
print(Y_predict)
from sklearn.metrics import r2_score

score = r2_score(y_test, y_pred)
print(score)
