#Assignment based on simple linear regression on any dataset
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from  sklearn.linear_model import LinearRegression
data=pd.read_csv("houseprices.csv")
#print(data.columns)
f=['Price', ' Area']
data=data[f]
#print(data)
x=data.iloc[:,1:].values
print("independent variable area as x \n",x)
y=data.iloc[:,:1].values
print("Dependent variable price as y \n", y)
#split data into training and testing
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=3)
#print(x_train)
reg=LinearRegression()
reg.fit(x_train,y_train)
prediction=reg.predict(x_test)
print(prediction)
print(y_test)
#calculating mse,rmse and r-squre
from sklearn.metrics import mean_squared_error
mse=mean_squared_error(y_test,prediction)
print("mean squeare error: \n",mse)
r2score=reg.score(x_test,y_test)
print(r2score)
print("weight or slope M is : \n",reg.coef_)
print("intercept C is :\n",reg.intercept_)
plt.scatter(x_test,y_test)
plt.plot(x_test, prediction, color = "g")
plt.xlabel('x')
plt.ylabel('y')
plt.show()

















# R-squared method:

''' R-squared is a statistical method that determines the goodness of fit.
It measures the strength of the relationship between the dependent and independent variables on a scale of 0-100%.
The high value of R-square determines the less difference between the predicted values and actual values and hence represents a good model.
It is also called a coefficient of determination, or coefficient of multiple determination for multiple regression.
It can be calculated from the below formula: 
    R-square=explained variation / total variation 

Linear regression algorithm shows a linear relationship between a dependent (y) and one or more independent (y) '''

# Logistic regression :
''' predicts the output of a categorical dependent variable. Therefore the outcome must be a categorical or discrete value. It can be either Yes or No, 0 or 1, true or False, etc. but instead of giving the exact value as 0 and 1, it gives the probabilistic values which lie between 0 and 1.
Logistic Regression is much similar to the Linear Regression except that how they are used. Linear Regression is used for solving Regression problems, whereas Logistic regression is used for solving the classification problems. '''

'''Polynomial Regression is a regression algorithm that models the relationship between a dependent(y) and independent variable(x) as nth degree polynomial. The Polynomial Regression equation is given below:
y= b0+b1x1+ b2x12+ b2x13+...... bnx1n
It is also called the special case of Multiple Linear Regression in ML. Because we add some polynomial terms to the Multiple Linear regression equation to convert it into Polynomial Regression.
'''
#performancr matrix :
'''To evaluate the performance or quality of the model, different metrics are used, and these metrics are known as performance metrics or evaluation metrics. '''

#Mean Absolute Error (MAE)
'''Mean Absolute Error or MAE is one of the simplest metrics, which measures the absolute difference between actual and predicted values, where absolute means taking a number as Positive '''
#Mean Squared Error
'''Mean Squared error or MSE is one of the most suitable metrics for Regression evaluation. It measures the average of the Squared difference between predicted values and the actual value given by the model.

Since in MSE, errors are squared, therefore it only assumes non-negative values, and it is usually positive and non-zero. '''
#Adjusted R Squared
'''Adjusted R squared, as the name suggests, is the improved version of R squared error. R square has a limitation of improvement of a score on increasing the terms, even though the model is not improving, and it may mislead the data scientists. '''