# Polynomial Regression

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset_name = 'radius of curvature(cs4-cs9).csv'
y_label = 'radius of curvature(mm-1)'
title_name = 'CS4-CS9 Radius of Curvature'

# Importing the dataset
dataset = pd.read_csv(dataset_name)
X = dataset.iloc[:, 0:1].values
y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
"""from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)"""

# Feature Scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)"""

# Fitting Linear Regression to the dataset
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X, y)

# Fitting Polynomial Regression to the dataset
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 5)
X_poly = poly_reg.fit_transform(X)
poly_reg.fit(X_poly, y)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(X_poly, y)

# Visualising the Linear Regression results

# plt.scatter(X, y, color = 'red')
# plt.plot(X, lin_reg.predict(X), color = 'blue')
# plt.title(dataset_name)
# plt.xlabel('Age')
# plt.ylabel(y_label)
# plt.show()

# Visualising the Polynomial Regression results
# plt.scatter(X, y, color = 'red')
# plt.plot(X, lin_reg_2.predict(poly_reg.fit_transform(X)), color = 'blue')
# plt.show()

# Visualising the Polynomial Regression results (for higher resolution and smoother curve)

# linear regression model as comparison

plt.scatter(X, y, color = 'red')
plt.plot(X, lin_reg.predict(X), color = 'blue')
plt.title(dataset_name)

# polynomial regression

X_grid = np.arange(min(X), max(X), 0.1)
X_grid = X_grid.reshape((len(X_grid), 1))
plt.plot(X_grid, lin_reg_2.predict(poly_reg.fit_transform(X_grid)), color = 'black')
plt.title(title_name)
plt.xlabel('Age(year)')
plt.ylabel(y_label)
plt.show()

# Predicting a new result with Linear Regression
# lin_reg.predict(6)

# Predicting a new result with Polynomial Regression
# lin_reg_2.predict(poly_reg.fit_transform(6.5))