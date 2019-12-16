# -*- coding: utf-8 -*-
"""
Created on Wed Aug  1 22:24:46 2018

@author: tunji
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Salary_Data.csv')



#Next we create Matrix of Features//.values below converts DF into arrays
x = dataset.iloc[:, :-1].values#selecting all records and columns except last column as it is our Classifier
y = dataset.iloc[:, 1].values#Selecting last column only, our Classifier 

#Splitting Dataset into Training and Test sets
from sklearn.model_selection import train_test_split
X_Train, X_Test, Y_Train , Y_Test= train_test_split(x, y, test_size = 1/3, random_state = 0)

#Fitting Simple Linear Regression to the Training Set
from sklearn.linear_model import LinearRegression
regressor_var = LinearRegression()
regressor_var.fit(X_Train,Y_Train)

#Predicting the Test Set results
y_pred = regressor_var.predict(X_Test)

#Visualization comparing the linear regression line to Training Set
plt.scatter(X_Train, Y_Train, color = 'red')
plt.plot(X_Train,regressor_var.predict(X_Train))
plt.title('Salary vs Experience (Training Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()

#visualizing the liner regression line to the Test Set

plt.scatter(X_Test, Y_Test, color = 'red')
plt.plot(X_Train,regressor_var.predict(X_Train),color = 'blue')
plt.title('Salary vs Experience (Test Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()


plt.scatter(X_Test,Y_Test, color='green')
plt.scatter(X_Test,y_pred, color='red')
plt.title('Actual Salaries vs Predicted Salaries')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()



























