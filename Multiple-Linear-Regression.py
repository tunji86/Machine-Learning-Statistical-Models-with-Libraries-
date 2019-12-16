# -*- coding: utf-8 -*-
"""
Created on Thu Aug  2 05:29:38 2018

@author: tunji
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('50_Startups.csv')

#Next we create Matrix of Features//.values below converts DF into arrays
x = dataset.iloc[:, :-1].values#selecting all records and columns except last column as it is our Classifier
y = dataset.iloc[:, 4].values#Selecting last column only, our Classifier 

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
x[:, 3] = labelencoder_X.fit_transform(x[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [3])
x = onehotencoder.fit_transform(x).toarray()

#Avoiding Dummyvariable trap
x=x[:, 1:]#This is taken care of by python but not all libraries do so automatically

#Splitting Dataset into Training and Test sets
from sklearn.model_selection import train_test_split
X_Train, X_Test, Y_Train , Y_Test= train_test_split(x, y, test_size = 0.2, random_state = 0)

#***************PREPROCESSING DONE***************

#Fitting Multiple Linear Regression to Training Set
from sklearn.linear_model import LinearRegression
linearRegression_var = LinearRegression()
linearRegression_var.fit(X_Train,Y_Train)

#Predicting our Test Set
y_pred = linearRegression_var.predict(X_Test)

#BACKWARD ELIMINATION
import statsmodels.formula.api as sm#this library does not implicitly include the b0 constant in the MLR, so
#we have to fix this by adding a new column to our dataframe
x=np.append(arr=np.ones((50,1)),values=x.astype(int),axis=1)
#x=np.delete(x,0,1)

#Fit full(not just training) model with all possible predictors
x_opt=x[:, [0,1,2,3,4,5]]
regresor_OLS = sm.OLS(endog=y, exog=x_opt).fit()
#Find independent variable(feature or predictor) with best p-value
regresor_OLS.summary()
#if highest predictor p-value>SL, remove it
x_opt=x[:, [0,1,3,4,5]]
regresor_OLS = sm.OLS(endog=y, exog=x_opt).fit()
regresor_OLS.summary()
#iterate again
x_opt=x[:, [0,3,4,5]]
regresor_OLS = sm.OLS(endog=y, exog=x_opt).fit()
regresor_OLS.summary()
#iterate again
x_opt=x[:, [0,3,5]]
regresor_OLS = sm.OLS(endog=y, exog=x_opt).fit()
regresor_OLS.summary()
#iterate again
x_opt=x[:, [0,3]]
regresor_OLS = sm.OLS(endog=y, exog=x_opt).fit()
regresor_OLS.summary()
#Last two are both <SL and equal. But one is a Constant so R&D is our best predictor

#Now use R&D tomplot linear regression against y
RD = dataset["R&D Spend"].values

RD = RD.reshape(-1,1)

from sklearn.model_selection import train_test_split
X_Train, X_Test, Y_Train , Y_Test= train_test_split(RD, y, test_size = 0.2, random_state = 0)


linreg = LinearRegression()
linreg.fit(X_Train,Y_Train)

linreg.predict(RD)

plt.scatter(X_Test,Y_Test, color='red')
plt.plot(X_Test,linreg.predict(X_Test), color='blue')
plt.title("Regression")
plt.xlabel('R&D Spend')
plt.ylabel('Profit')
plt.show()
