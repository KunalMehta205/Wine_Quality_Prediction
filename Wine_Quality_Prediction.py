# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 23:49:27 2020

@author: hp
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.model_selection import train_test_split
get_ipython().run_line_magic('matplotlib', 'inline')

wine = pd.read_csv('C:\\Users\\hp\\Desktop\\winequality-red.csv')

wine.head()

wine.info()  #get information about data columns

wine['quality'].unique()

sns.pairplot(wine)  #very less predictability through this

fig=plt.figure(figsize=(10,6))   #fixed acidity does not help us much to classify the quality
sns.barplot(x='quality',y='fixed acidity', data=wine)

fig=plt.figure(figsize=(10,6))   #as the quality increases volatile acidity decreases 
sns.barplot(x='quality',y='volatile acidity', data=wine)


fig=plt.figure(figsize=(10,6))   #as quality increases the amount of citic acid also increases
sns.barplot(x='quality',y='citric acid', data=wine)


fig=plt.figure(figsize=(10,6))   #not contributing much in quality
sns.barplot(x='quality',y='residual sugar', data=wine)

fig=plt.figure(figsize=(10,6))   #with increase in quality, amount of chlorides decreases
sns.barplot(x='quality',y='chlorides', data=wine)

fig=plt.figure(figsize=(10,6))   
sns.barplot(x='quality',y='free sulfur dioxide', data=wine)

fig=plt.figure(figsize=(10,6))   
sns.barplot(x='quality',y='total sulfur dioxide', data=wine)

fig=plt.figure(figsize=(10,6))      #density is almost same and constant with increase in quality 
sns.barplot(x='quality',y='density', data=wine)

fig=plt.figure(figsize=(10,6))   
sns.barplot(x='quality',y='pH', data=wine)


fig=plt.figure(figsize=(10,6))   #sulphur level increases with quality
sns.barplot(x='quality',y='sulphates', data=wine)


fig=plt.figure(figsize=(10,6))   #alcohol level also increases with quality
sns.barplot(x='quality',y='alcohol', data=wine)


wine.describe()

'''
Now to identify the quality of wine we will be creating a new column as "Wine Review".
We will classfiy the wine in 3 categories:
a. 1 - Bad
b. 2 - Average 
c. 3 - Good

Splitting the quality:
1,2,3 - Bad(1) 
4,5,6,7 - Average(2) 
8,9,10 - Good(3)
'''

wine_review = []
for i in wine['quality']:
    if i>=1 and i<=3:
        wine_review.append('1')
    elif i>=4 and i<=7:
        wine_review.append('2')
    elif i>=8 and i<=10:
        wine_review.append('3')
wine['Wine Review'] = wine_review

wine.columns   #final data
wine['Wine Review'].unique()

from collections import Counter
Counter(wine['Wine Review'])

#Splitting the data to features and labels 
features = wine.iloc[:,:11].values
label = wine.iloc[:,-1].values

#Feature Scaling - due to a large range(different values) in data and more than one feature
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
features = sc.fit_transform(features)

#Applying PCA - Principal Component Analysis 
# -> Speed up the fitting/learning algorithm
# -> Changing dimensions (Dimensions Reduction)

from sklearn.decomposition import PCA
pca = PCA()
features_pca = pca.fit_transform(features)

#plotting the graph to find the principal components
plt.figure(figsize=(10,10))
plt.plot(np.cumsum(pca.explained_variance_ratio_), 'ro-')   #cumsum = cumulative sum
plt.grid()

#AS per the graph, we can see that 8 principal components attribute for 90% of variation in the data. 
#we shall pick the first 8 components for our prediction.

pca_new = PCA(n_components=8)
features_new = pca_new.fit_transform(features)

#splitting dataset into training and testing data
from sklearn.model_selection import train_test_split
features_train, features_test, label_train, label_test = train_test_split(features_new, label, test_size = 0.25,random_state=0)

print(features_train.shape)
print(label_train.shape)
print(features_test.shape)
print(label_test.shape)

#Applying Machine Learning Models
#Logistic Regression
#Decision Trees
#Naive Bayes
#Random Forests
#SVM


# ### 1. Logistic Regression
from sklearn.linear_model import LogisticRegression
lr=LogisticRegression()
lr.fit(features_train,label_train)

label_pred = lr.predict(features_test)

from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(label_test,label_pred)
ac = accuracy_score(label_test,label_pred)
print(cm)
print(ac*100)

# 98% accuracy with Logistic Regression

#2. Decision Tree 
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
dt.fit(features_train,label_train)
dt_predict = dt.predict(features_test)

dt_cm = confusion_matrix(label_test, dt_predict)
dt_ac = accuracy_score(label_test, dt_predict)
print(dt_cm)
print(dt_ac*100)

# 3.Naive Bayes
from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(features_train,label_train)
nb_predict=nb.predict(features_test)

nb_cm = confusion_matrix(label_test, nb_predict)
nb_ac = accuracy_score(label_test, nb_predict)
print(nb_cm)
print(nb_ac*100)

# 4.SVM Classifier - Kernel
from sklearn.svm import SVC
svc =SVC(kernel='rbf',random_state=0)
svc.fit(features_train,label_train)
svc_predict =svc.predict(features_test)

svc_cm = confusion_matrix(label_test, svc_predict)
svc_ac = accuracy_score(label_test, svc_predict)
print(svc_cm)
print(svc_ac*100)


# Linear SVM
from sklearn.svm import SVC
lin_svc =SVC(kernel='linear',random_state=0)
lin_svc.fit(features_train,label_train)
lin_svc_predict =lin_svc.predict(features_test)

lin_svc_cm = confusion_matrix(label_test, lin_svc_predict)
lin_svc_ac = accuracy_score(label_test, lin_svc_predict)
print(lin_svc_cm)
print(lin_svc_ac*100)
