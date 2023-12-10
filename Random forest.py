import pandas as pd
import numpy as np
iris=pd.read_csv("iris.csv")
print(iris.target_names) 
print(iris.feature_names)
# dividing the datasets into two parts i.e. training datasets and test datasets 
X, y = datasets.load_iris( return_X_y = True) 

# Splitting arrays or matrices into random train and test subsets 
from sklearn.model_selection import train_test_split 
# i.e. 70 % training dataset and 30 % test datasets 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30) 

# importing random forest classifier from assemble module 
from sklearn.ensemble import RandomForestClassifier 
import pandas as pd 
# creating dataframe of IRIS dataset 
data = pd.DataFrame( { ‘ sepallength ’: iris.data[:, 0], ’sepalwidth’: iris.data[:, 1], 
					’petallength’: iris.data[:, 2], ’petalwidth’: iris.data[:, 3], 
					’species’: iris.target}) 
# printing the top 5 datasets in iris dataset 
print(data.head()) 
# creating a RF classifier 
clf = RandomForestClassifier(n_estimators = 100) 

# Training the model on the training dataset 
# fit function is used to train the model using the training sets as parameters 
clf.fit(X_train, y_train) 

# performing predictions on the test dataset 
y_pred = clf.predict(X_test) 

# metrics are used to find accuracy or error 
from sklearn import metrics 
print() 

# using metrics module for accuracy calculation 
print("ACCURACY OF THE MODEL: ", metrics.accuracy_score(y_test, y_pred)) 
# predicting which type of flower it is. 
clf.predict([[3, 3, 2, 2]]) 
# importing random forest classifier from assemble module 
from sklearn.ensemble import RandomForestClassifier 
# Create a Random forest Classifier 
clf = RandomForestClassifier(n_estimators = 100) 

# Train the model using the training sets 
clf.fit(X_train, y_train)
# using the feature importance variable 
import pandas as pd 
feature_imp = pd.Series(clf.feature_importances_, index = iris.feature_names).sort_values(ascending = False) 
feature_imp















#Random Forest is a classifier that contains a number of decision trees on various subsets of the given dataset and takes the average to improve the predictive accuracy of that dataset.
#Applications of Random Forest . The greater number of trees in the forest leads to higher accuracy and prevents the problem of overfitting.
'''
There are mainly four sectors where Random forest mostly used:

Banking: Banking sector mostly uses this algorithm for the identification of loan risk.
Medicine: With the help of this algorithm, disease trends and risks of the disease can be identified.
Land Use: We can identify the areas of similar land use by this algorithm.
Marketing: Marketing trends can be identified using this algorithm.'''

#Advantages of Random Forest
'''Random Forest is capable of performing both Classification and Regression tasks.
It is capable of handling large datasets with high dimensionality.
It enhances the accuracy of the model and prevents the overfitting issue.
Disadvantages of Random Forest
Although random forest can be used for both classification and regression tasks, it is not more suitable for Regression tasks.''' 

