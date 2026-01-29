"""
Evaluation methods:
1. MAE: Mean Absolute Error : Measure the magnitude of error without considering their directions.
use case: suitable when all have equal importance

2. Mean square error (MSE): Measure the average differences between predicted values and actual values.
use case: Penalizes larger errors more. sensitive to outliers.

3. Root mean square error(RMSE): square root of mse proviting error in same unit as the target variable.

4. R-squared (R^2): Measures how well the model explains the variability of the target.

CLASSIFICATION METRICS:
1. Accuracy: percentage of correctly measured instances.
2. Precision: Fraction of true positive predictions among all positeve predictions. (keyword-predicted.)
3. Recall (sentivity):Fraction of true positive among all actual positives.(keyword-identified.)
4. F1 Score: harmonic mean of precision and recall.
5. ROC-AUC: measures the ability of the model to distinguish between classes.

"""

###EXERCISE###
"""
Objective: 1. Train a classification model, calculate confusion matrix and interpret precision, recall and f1 score.
2. Train a regression model and do it's evaluation.
"""

# import pandas as pd
# from sklearn.datasets import load_iris
# from sklearn.model_selection import train_test_split
# from sklearn.linear_model import LogisticRegression
# from sklearn.metrics import (
#     confusion_matrix,
#     classification_report,
#     ConfusionMatrixDisplay,
#     mean_squared_error,
#     r2_score,
# )
# import matplotlib.pyplot as plt

# # load dataset
# data = load_iris()

# x = data.data
# y = (data.target == 0).astype(int)

# # split dataset
# x_train, x_test, y_train, y_test = train_test_split(
#     x, y, test_size=0.2, random_state=42
# )

# # Train logistic regresion model
# model = LogisticRegression()
# model.fit(x_train, y_train)

# # predict values
# y_predict = model.predict(x_test)

# # Model Evaluation
# cm = confusion_matrix(y_test, y_predict)
# print(
#     "Classification report of Logistic regression: \n",
#     classification_report(y_test, y_predict),
# )
# print("Confusion matrix of logistic regression: \n", cm)

# disp = ConfusionMatrixDisplay(
#     confusion_matrix=cm, display_labels=["Not class 0", "class 0"]
# )
# disp.plot(cmap="Blues")
# plt.title("confusion")
# plt.show()

# # Model Evaluation
# print("Mean squared error: ", mean_squared_error(y_test, y_predict))
# print("Root mean squared error: ", r2_score(y_test, y_predict))


#####Tast -2 ######
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# load the dataset
data = fetch_california_housing()
x = data.data #loading features
y = data.target #loading target

#split the model
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

#training the model
model=LinearRegression()
model.fit(x_train,y_train)

#predict the values
y_predict=model.predict(x_test)

#Evaluate the model
mae=mean_absolute_error(y_test,y_predict)
mse=mean_squared_error(y_test,y_predict)
r_score=r2_score(y_test,y_predict)

print("mean absolute error: ", mae)
print("mean square error: ", mse)
print("root mean square error: ",r_score)

