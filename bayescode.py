import pandas as pd 
import numpy as num 
data=pd.read_csv(r"C:\Users\DELL\Downloads\archive\IRIS.csv")
x=data.iloc[:,:-1].values
print(x)
y=data.iloc[:,:-1].values
print(y)

#ENCODING THE Category 
from sklearn.preprocessing import LabelEncoder
y=LabelEncoder().fit_transform(y)
print(y)

#split test and train data
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test =train_test_split(x,y,test_size=0.2,random_state=3)
print(x_train)
print(y_train)

#trian the model 
from sklearn.naive_bayes import GaussianNB
gnb=GaussianNB()
gnb.fit(x_train,y_train)

#Prediction using bayes
y_predict=gnb.predict(x_test)
print("Predicted Value by model :",y_predict)
print("Actual Value of model :",y_test)

#confusion matrix

from sklearn.metrics import confusion_matrix
confusion_mat=confusion_matrix(y_test,y_predict)
print("Confusion matrix :",confusion_mat)

#Classifier Accuracy

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_predict)*100
print("Accuracy of the model :",accuracy)


#Naivy bayes 

'''Naïve Bayes algorithm is a supervised learning algorithm, which is based on Bayes theorem and used for solving classification problems
Advantages of Naïve Bayes Classifier:
Naïve Bayes is one of the fast and easy ML algorithms to predict a class of datasets.
It can be used for Binary as well as Multi-class Classification

Disadvantages of Naïve Bayes Classifier:
Naive Bayes assumes that all features are independent or unrelated, so it cannot learn the relationship between features.

Applications of Naïve Bayes Classifier:
It is used for Credit Scoring.
It is used in medical data classification.
It can be used in real-time predictions 
sentiment analysis

Accuracy Score = (TP + TN)/ (TP + FN + TN + FP)
Accuracy score is the metrics that is defined as the ratio of true positives and true negatives to all positive and negative observations
 Precision Score = True Positives/ (False Positives + True Positives)
 Recall Score = True Positives / (False Negatives + True Positives)
 Recall score in machine learning represents the model’s ability to correctly predict the positives out of actual positives.
 F1 Score = 2* Precision Score * Recall Score/ (Precision Score + Recall Score)
 F1 score is harmonic mean of precision and recall score.
 Precision score in machine learning measures the proportion of positively predicted labels that are actually correct. Precision is also known as the positive predictive value. Precision score is used in conjunction with the recall score to trade-off false positives and false negatives
 A confusion matrix is a table used in machine learning and statistics to assess the performance of a classification model. 
 '''