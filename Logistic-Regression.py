# -*- coding: utf-8 -*-
"""
Created on Sun Aug 12 22:29:42 2018

@author: tunji
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Social_Network_Ads.csv')

#Next we create Matrix of Features//.values below converts DF into arrays
x = dataset.iloc[:, 2:4].values#selecting all records and columns except last column as it is our Classifier
y = dataset.iloc[:, 4:5].values#Selecting last column only, our Classifier 

#Splitting Dataset into Training and Test sets
from sklearn.model_selection import train_test_split
X_Train, X_Test, Y_Train , Y_Test= train_test_split(x, y, test_size = 0.25, random_state = 0)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_Train = scaler.fit_transform(X_Train)
X_Test = scaler.transform(X_Test)

#Fit Regressor Algorithm to Training Data
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0)
classifier.fit(X_Train,Y_Train)

y_pred = classifier.predict(X_Test)

#Evaluating Classifier with Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_Test,y_pred)

from matplotlib.colors import ListedColormap
X_set, y_set = X_Test, Y_Test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('Logistic Regression (Test set)')
plt.xlabel('Age')
plt.ylabel('Estimated Salary')
plt.legend()
plt.show()