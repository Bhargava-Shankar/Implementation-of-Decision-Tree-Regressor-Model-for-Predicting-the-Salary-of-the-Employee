# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Initialize a decision tree structure.
2.Split data recursively based on features to minimize variance or mean squared error.
3.Stop splitting when a predefined stopping criterion is met (e.g., maximum depth, minimum samples per leaf).
4.Predict the average target value of the samples in each leaf node.

## Program:
```python
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: Bhargava S
RegisterNumber: 212221040029
*/
import pandas as pd
data = pd.read_csv("/content/Salary_EX7.csv")
data.head()
data.info()
data.isnull().sum()
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data["Position"] = le.fit_transform(data["Position"])
data.head()
x = data[["Position","Level"]]
y = data["Salary"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=2)
from sklearn.tree import DecisionTreeRegressor
dt = DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred = dt.predict(x_test)
from sklearn import metrics
mse = metrics.mean_squared_error(y_test,y_pred)
r2 = metrics.r2_score(y_test,y_pred)
dt.predict([[5,6]])
r2
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

plt.figure(figsize=(20,8))
plot_tree(dt,feature_names=x.columns,filled=True)
plt.show()
```

## Output:
![image](https://github.com/Bhargava-Shankar/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/85554376/6ee59942-5000-4a34-a541-c074e4ff4a95)
![image](https://github.com/Bhargava-Shankar/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/85554376/99d25abb-f1a7-4952-8073-b388eccc0be7)




## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
