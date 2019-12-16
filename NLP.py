# -*- coding: utf-8 -*-
"""
Created on Mon Sep 17 17:43:07 2018

@author: tunji
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter='\t', quoting=3)

#text clean up
import re
import nltk
nltk.download('stopwords')#package of words deemed unhelpful in opinion minning
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus =[]
for i in range(0,1000):
    review = re.sub('[^a-zA-Z]',' ',dataset['Review'][i])#remove everything that's not a letter
    review = review.lower()#turn everything to lower case
    review = review.split()#splits the words into individual columns
    ps = PorterStemmer()#stemming reduces spacity, by reducing number of words. e.g loving, loved, love, lovely become love
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]#loops through to remove irrelevant words
    review = ' '.join(review)
    corpus.append(review)

#create Bag Of Words model
from sklearn.feature_extraction.text import CountVectorizer    
cv = CountVectorizer(max_features=1500)#max_features puts limit on number of words to train model with
x = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values

from sklearn.model_selection import train_test_split
X_Train, X_Test, Y_Train , Y_Test= train_test_split(x, y, test_size = 0.20, random_state = 0)

'''#Fit K-NN Algorithm to Training Data
from sklearn.neighbors import KNeighborsClassifier
classifier = KNeighborsClassifier(n_neighbors=5,metric='minkowski',p=2)
classifier.fit(X_Train,Y_Train)'''

#Fit Regressor Algorithm to Training Data
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0)
classifier.fit(X_Train,Y_Train)





'''
it Classifier Algorithm to Training Data
from sklearn.tree import I
classifier = DecisionTreeClassifier(criterion='entropy',random_state=0)
classifier.fit(X_Train,Y_Train)'''

'''
#Fit Classifier Algorithm to Training Data
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=10,criterion='entropy',random_state=0)
classifier.fit(X_Train,Y_Train)'''


'''
#Fit Classifier Algorithm to Training Data
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion='entropy',random_state=0)
classifier.fit(X_Train,Y_Train)'''
'''
#Fit Classifier Algorithm to Training Data
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_Train,Y_Train)'''

#Predict a y_value
y_pred = classifier.predict(X_Test)
proba = classifier.predict_proba(X_Test)

#Evaluating Classifier with Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_Test,y_pred)


#**********************************************************************************
from sklearn import tree
import graphviz
classifier1 = tree.DecisionTreeClassifier(random_state=0)
classifier1 = classifier1.fit(X_Test,Y_Test)

graph_var1 = tree.export_graphviz(classifier1, out_file=None)
graph_var = graphviz.Source(graph_var1)

graph_var.render('X_Test')















