# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import necessary libraries (e.g., pandas, numpy,matplotlib).
2. Load the dataset and then split the dataset into training and testing sets using sklearn library.
3. Create a Linear Regression model and train the model using the training data (study hours as input, marks scored as output).
4. Use the trained model to predict marks based on study hours in the test dataset.
5. Plot the regression line on a scatter plot to visualize the relationship between study hours and marks scored.

## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: SARAVANA KUMAR M
RegisterNumber:212222230133  
*/
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df=pd.read_csv('/content/score.csv')
df.head(10)
plt.scatter(df['X'],df['Y'])
plt.xlabel('X')
plt.ylabel('Y')
x=df.iloc[:,0:1]
y=df.iloc[:,-1]
from sklearn.model_selection import train_test_split as t
X_train,X_test,Y_train,Y_test=t(x,y,test_size=0.2,random_state=0)
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(X_train,Y_train)
lr.predict(X_test.iloc[0].values.reshape(1,1))
plt.scatter(df['X'],df['Y'])
plt.xlabel('X')
plt.ylabel('Y')
plt.plot(X_train,lr.predict(X_train),color='blue')
```

## Output:
### Head:
![Screenshot 2024-02-22 220640](https://github.com/Saravana-kumar369/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/117925254/1fd68968-94d0-43f0-9e4a-324c5ab8a093)

### Graph of training dataset:
![Screenshot 2024-02-22 221138](https://github.com/Saravana-kumar369/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/117925254/3cd73d4b-6ce1-4302-9215-8006b46b4e66)

### Array value of X and Y:
![image](https://github.com/Saravana-kumar369/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/117925254/3cb190c8-c472-4a3d-a208-66cfbcacf5fd)

### Test Set Graph:
![image](https://github.com/Saravana-kumar369/Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored/assets/117925254/77bebdba-44e1-4bf7-a77c-9a14198e6e05)



## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
