# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the standard Libraries.


2.Set variables for assigning dataset values.


3.Import linear regression from sklearn.


4.Assign the points for representing in the graph.


5.Predict the regression for marks by using the representation of the graph.


6.Compare the graphs and hence we obtained the linear regression for the given datas.


## Program:
```
/*
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: RITHIKA K
RegisterNumber:  212224230230


*/
```


```
/*
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df=pd.read_csv('student.csv')
df.head(10)
plt.scatter(df['X'],df['Y'])
plt.xlabel('X')
plt.ylabel('Y')
x=df.iloc[:,0:-1]
y=df.iloc[:,-1]
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(x,y,test_size=0.2,random_state=0)
from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(X_train,Y_train)
X_train
Y_train
lr.predict(x_test.iloc[0].values.reshape(1,1))
plt.scatter(df['X'],df['Y'])
plt.xlabel('X')
plt.ylabel('Y')
plt.plot(X_train,lr.predict(X_train),color='orange')
lr.coef_
lr.intercept_

*/
```

## Output:
1. HEAD:


![image](https://github.com/user-attachments/assets/df5863f0-df0a-488b-bb64-b2270bbf1401)

   
2.GRAPH OF PLOTTED DATA:


![image](https://github.com/user-attachments/assets/14d27aa6-a387-44d7-9b96-9d95056efc18)


3.TRAINED DATA:


![image](https://github.com/user-attachments/assets/f6fd8ee0-ddcb-4a4c-bab8-2229e270a4e8)


4.LINE OF REGRESSION:


![image](https://github.com/user-attachments/assets/6c84fdfe-2a1e-4e43-91f6-e53a04bf5e96)


5.COEFFICIENT AND INTERCEPT VALUES:


![image](https://github.com/user-attachments/assets/a6e1db72-cd22-4b12-b61e-6501be2dae42)
## Result:
Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
