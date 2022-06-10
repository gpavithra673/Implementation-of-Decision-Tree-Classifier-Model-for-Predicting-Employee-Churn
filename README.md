# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Moodle-Code Runner

## Algorithm
1. Import the required libraries.
2.Upload and read the dataset.
3.Check for any null values using the isnull() function.
4.From sklearn.tree import DecisionTreeClassifier and use criterion as entropy.
5.Find the accuracy of the model and predict the required values by importing the required module from sklearn.

## Program:
```
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: G.Pavithra
RegisterNumber:  212221240036
```
```
import pandas as pd
data = pd.read_csv("Employee.csv")
data.head()
data.info()
data.isnull().sum()
data["left"].value_counts()
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data["salary"] = le.fit_transform(data["salary"])
data.head()
x = data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
x.head()
y = data["left"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=100)
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(criterion = "entropy")
dt.fit(x_train,y_train)
y_pred = dt.predict(x_test)
from sklearn import metrics
accuracy = metrics.accuracy_score(y_test,y_pred)
accuracy
dt.predict([[0.5,0.8,9,260,6,0,1,2]])
```
## Output:
![decision tree classifier model](sam.png)
### DATA HEAD:
![image](https://user-images.githubusercontent.com/93427264/172993330-4d120118-2fd7-4265-b7cd-615e3e950276.png)
### INFORMATION:
![image](https://user-images.githubusercontent.com/93427264/172993460-33b8af67-0cc6-481a-9f20-8763518e99eb.png)
### NULL DATASET:
![image](https://user-images.githubusercontent.com/93427264/172993547-82ba9727-c3fa-484a-ba3c-61966d396baa.png)
### VALUE COUNTS():
![image](https://user-images.githubusercontent.com/93427264/172993595-8417dab2-c599-43e2-82f5-7e7f4d84b1fc.png)
### DATA HEAD():
![image](https://user-images.githubusercontent.com/93427264/172993626-b02bad3b-fb10-4c59-9f43-4f52a69ff8f3.png)
### X.HEAD():
![image](https://user-images.githubusercontent.com/93427264/172993649-0532c308-ec9e-430a-aa62-ed1d720ee905.png)
### ACUURACY():
![image](https://user-images.githubusercontent.com/93427264/172993704-d0c99a69-3693-4d3f-9376-f98420460519.png)
### DATA PREDICTION:
![image](https://user-images.githubusercontent.com/93427264/172993729-7160e996-f79b-4cd5-90a5-b11399a5bd5e.png)
## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
