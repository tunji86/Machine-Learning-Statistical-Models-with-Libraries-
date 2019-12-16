# -*- coding: utf-8 -*-
"""
Created on Mon Aug  6 03:07:27 2018

@author: tunji
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Position_Salaries.csv')

#****************PREPROCESSING****************************************
#Next we create Matrix of Features//.values below converts DF into arrays
x = dataset.iloc[:, 1:2].values#selecting all records and columns except last column as it is our Classifier
y = dataset.iloc[:, 2].values#Selecting last column only, our Classifier 

#Because we have only 10 records here it is advisable we train the model on the entire set
'''from sklearn.model_selection import train_test_split
X_Train, X_Test, Y_Train , Y_Test= train_test_split(x, y, test_size = 0.2, random_state = 0)'''



#fit Polynomial Regression to training set
from sklearn.preprocessing import PolynomialFeatures
polyreg = PolynomialFeatures(degree=4)
x_poly = polyreg.fit_transform(x)#create new x (polynomial version of it)
polyreg.fit(x_poly,y)#training new x (x_poly) and y

#now we fit a Linear regression to x_poly against y
from sklearn.linear_model import LinearRegression
linreg = LinearRegression()
linreg.fit(x_poly,y)

plt.scatter(x,y, color='red')
plt.plot(x,linreg.predict(polyreg.fit_transform(x)), color='blue')
plt.title("Polynomial Regression")
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

#linear prediction
linreg.predict(6.5)

#Polynomial prediction
linreg2.predict(polyreg.fit_transform(6.5))





#below put x values into lower granularity of 0.1 increments. Wit that we can see more detiled predictions of x,
#we just have to replace x with it
x_grid = np.arange(min(x), max(x), 0.1)
x_grid = x_grid.reshape((len(x_grid), 1))
plt.scatter(x,y, color='red')
plt.plot(x_grid,linreg2.predict(polyreg.fit_transform(x_grid)), color='blue')
plt.title("Polynomial Regression")
plt.xlabel('Position Level')
plt.ylabel('Salary')
plt.show()

