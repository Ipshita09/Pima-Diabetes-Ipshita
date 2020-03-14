# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 18:21:06 2020

@author: Ipshita Chowdhury
"""
#Name: Ipshita Chowdhury
#Comparison of Machine Learning Classification Algorithms
#Dataset used: Pima Diabetes
#Source: Kaggle
#About the Dataset: 
#The dataset used here has been originally obtained from the National Institute of Diabetes and 
#Kidney Diseases. The objective of the dataset is to diagnostically predict whether a patient is 
#diabetic based on certain diagnostic measurements included in the dataset. Several constraints were 
#placed on the selection of these instances from a larger database. All patients here are females at 
#least 21 years old of Pima Indian Heritage.
#The dataset consists of several medical predictor variables and one Target Variable ‘Outcome’, 
#i.e., whether the patient has diabetes or not. Predictor Variables include the Number of Pregnancies
#the patient has had, their BMI, Insulin levels, Age, Glucose levels, Blood Pressure, Skin Thickness 
#and Diabetes Pedigree Function. 

#Algorithms: 
#1. Naive Bayes
#2. Random Forest
#3.Logistic Regression

#We will now use these algorithms to classify our dataset and use it to predict whether a person 
#with certain diagnostic measurements will be diabteic or not. 
#We further aim at comparing the efficiencies of these classification algorithms thereby
#determining the best among them. 

#Code:

#Importing the Packages:  
from scipy import optimize
import pandas as pd
import numpy as np
from pandas import DataFrame
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt

#Importing the Dataset: 
filename = 'diabetes.csv'
data = pd.read_csv(filename)
data1=data

#Bird-eye view of the dataset: 
data.shape                      #768 observations across 9 features.
data.head(5)                    #Head view of how the dataset looks like.

#Data Preprocessing: 
    #1. Null and Zero Values: 
data.isnull().values.any()      #No missing Values in the dataset. 

    #2. Correlated Columns: 
data.corr()
    #Now, translating the correlation table as a heat map: 
import seaborn as sns
corr=data[data.columns].corr()
sns.heatmap(corr,cmap="YlGnBu",annot=True)
#A relatively moderate positive correlation of found to be between: 
#a. Age and Pregnancies
#b. Outcome and Glucose
#c. Insulin and Skin Thickness
#d. Insulin and Skin Thickness
#e. BMI and Skin Thickness

#A weak negative correlation was found between: 
#a. Skin Thickness and Pregnancies
#b. Insulin and Pregnancies
#c. Diabetes Pedigree Function and Pregnancies

    #3. True and False Ratio: 
#To ensure good results, we need to ensure that there are adequate amount if both favourable and 
#unfavourable outcomes. Thus, we find the true and false ratio of the dataset, i.e, it 
#determines the ratio of the patients with diabetes and those who are free of diabetes. 
total_obsv = len(data)
true_total = len(data.loc[data['Outcome'] == 1])
false_total = len(data.loc[data['Outcome'] == 0])
print("Number of True cases:  {0} ({1:2.2f}%)".format(true_total, (true_total/total_obsv) * 100))
print("Number of False cases: {0} ({1:2.2f}%)".format(false_total, (false_total/total_obsv) * 100))
#It is observed that there are a 34.9% cases off diabetes as against a 65.10% cases otherwise.
#This results fairly amounts to a good distribution of both true and false cases. 


#Implementing Algorithms: 

#Splitting the Dataset: 
feature= ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']
predict= ['Outcome']
X = data[feature].values 
y = data[predict].values 
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.3,random_state=0)
#Algorithms: 
#Algorithm 1. Naive Bayes: 
from sklearn.naive_bayes import GaussianNB 
NB = GaussianNB()
NB.fit(X_train, y_train.ravel())
#Predicting Values for training set: 
NB_predict_train = NB.predict(X_train)
#Predicting Values for test set: 
NB_predict_test=NB.predict(X_test)

from sklearn.datasets import make_classification      #to review
NB_samples = len(data)
A,B = make_classification(n_samples=NB_samples, n_features=9, n_informative=9, n_redundant=0)
A,B

#Validation: 
#1. Accuracy check for training set: 
print("Accuracy: {0:.4f}".format(metrics.accuracy_score(y_train,NB_predict_train)))
#76.72% accurate.

#Accuracy check for test set:
print("Accuracy:{0:.4f}".format(metrics.accuracy_score(y_test,NB_predict_test)))
#76.19% accurate.

#2. Classification Report: 
print(format(metrics.classification_report(y_test,NB_predict_test)))
#Precision is at 0.67. 
#Recall is at 0.51.

#3. Confusion Matrix: 
print("Confusion Matrix")
print(metrics.confusion_matrix(y_test, NB_predict_test) )
#Wrongly classified observations: 55 v/s
#Correctly classified observations: 176. 


#Algorithm 2: Random Forest: 
from sklearn.ensemble import RandomForestClassifier
RF= RandomForestClassifier(random_state=0) 
RF.fit(X_train,y_train.ravel())
#Predicting Values for training set: 
RF_predict_train = RF.predict(X_train)
#Predicting Values for test set: 
RF_predict_test=RF.predict(X_test)

#Validation: 
#1.Checking Accuracy: 
print("Accuracy: {0:.4f}".format(metrics.accuracy_score(y_train,RF_predict_train)))
print()
#Random forest classification gives an accuracy of about 97.77%
RF_predict_test = RF.predict(X_test)
print("Accuracy:{0:.4f}".format(metrics.accuracy_score(y_test,RF_predict_test)))        #(review-probably overfitting-underfitting problem)
print()
#Test set gives an accuracy of about 76.62%

#2. Classification Report
print(format(metrics.classification_report(y_test,RF_predict_test)))
#Precision is at 0.67. 
#Recall is at 0.54.

#3. Confusion Matrix: 
print("Confusion Matrix")
print(metrics.confusion_matrix(y_test, RF_predict_test) )
#Wrongly classified observations: 54 v/s
#Correctly classified observations: 177

#Algorithm 3: Logistic Regression: 
from sklearn.linear_model import LogisticRegression
LR=LogisticRegression(C=0.7,random_state=0)
LR.fit(X_train,y_train.ravel())
LR_predict_test = LR.predict(X_test)
#Predicting Values for training set: 
LR_predict_train = LR.predict(X_train)
#Predicting Values for test set: 
LR_predict_test=LR.predict(X_test)

#Validation: 
#1.Checking Accuracy: 
print("Accuracy: {0:.4f}".format(metrics.accuracy_score(y_train,LR_predict_train)))
print()
#Random forest classification gives an accuracy of about 76.72%
LR_predict_test = LR.predict(X_test)
print("Accuracy:{0:.4f}".format(metrics.accuracy_score(y_test,LR_predict_test)))        #(review-probably overfitting-underfitting problem)
print()
#Test set gives an accuracy of about 78.79%

#2. Classification Report
print(format(metrics.classification_report(y_test,LR_predict_test)))
#Precision is at 0.74. 
#Recall is at 0.53.

#3. Confusion Matrix: 
print("Confusion Matrix")
print(metrics.confusion_matrix(y_test, LR_predict_test) )
#Wrongly classified observations: 49 v/s
#Correctly classified observations: 182


#####SUMMARY: 
#We will determine the best value in terms of the recall value. 
#For: 1. Naive Bayes-Recall value is at: 0.51
#2. Random Forest- Recall value is at 0.54
#3. Logistic Regression- Recall Value is at 0.53

#Among these models, based in Recall value, Random Forest seems to be the best classifier.


