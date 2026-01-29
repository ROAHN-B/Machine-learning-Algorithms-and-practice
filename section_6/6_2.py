"""
Normalization and scaling: It is used to normalize or scale the values of the dataset so that,
any one value dosen't dominate while making predictions.
They are preprocessing techniques used to transform numerical features

WHY SCALING AND NORMALIZATION?
1. Improves algorithm performance.
2. Ensures fair comparison.
3. Stabilizes Training.

Min-Max scaling: Scales the values between 0-1. Ensures all values are within same range.
limitation- sensitive to outliers.

Standardization (z-score scaling): centers the data around zero and have a standard deviation of 1

WHEN TO USE SCALING AND NORMALIZATION?
1. KNN, SVM , K-Mean clustering
2. Gradient based models (algorithms which includes optimized theta.) like ,
Linear regression and logistic regression and neural networks.
3. Algorithms less sensitive to scaling - decision tree, random forest , gradient boosting.

"""

###EXERCISE###
"""
objective : 1.Apply MIN_MAX scaling and standardizationa to dataset using skit-learn.
2.observe the effect of scaling on model performance by training a KNN classifier before and after.

"""

import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler

# load dataset
data = load_iris()
x = pd.DataFrame(data.data, columns=data.feature_names)
y = data.target

# info of data
print("dataset info: ", x.describe())
print("\n target classes: ", data.target_names)

# split the dataset
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

# train the model
model = KNeighborsClassifier(n_neighbors=5)
model.fit(x_train, y_train)

# prediction of the model
y_predict = model.predict(x_test)

# Evaluation
print("Accuracy score of the KNN classification: ", accuracy_score(y_test, y_predict))
print(
    "Classification report of knn classification: ",
    classification_report(y_test, y_predict),
)


# Applied min-max scaling
scaler = MinMaxScaler()
x_scaled = scaler.fit_transform(x)

# split scaled data
x_train_scaled, x_test_scaled, y_train_scaled, y_test_scaled = train_test_split(
    x_scaled, y, test_size=0.2, random_state=42
)

# model training after scaling
scaled_model = KNeighborsClassifier(n_neighbors=5)
scaled_model.fit(x_train_scaled, y_train_scaled)

# prediction on scaled data
scaled_y_predict = scaled_model.predict(x_test_scaled)

# Evaluation of model with scaled values

print(
    "Accuracy report with min_max scaling: \n",
    accuracy_score(y_test_scaled, scaled_y_predict),
)
print(
    "Classification report with min_max scaling: \n",
    classification_report(y_test_scaled, scaled_y_predict),
)


# Applying standardization
scaler = StandardScaler()
x_std = scaler.fit_transform(x)

# split scaled data
x_train_std, x_test_std, y_train_std, y_test_std = train_test_split(
    x_std, y, test_size=0.2, random_state=42
)

# model training after scaling
std_model = KNeighborsClassifier(n_neighbors=5)
std_model.fit(x_train_std, y_train_std)

# prediction on scaled data
std_y_predict = std_model.predict(x_test_std)

# Evaluation of model with scaled values

print(
    "Accuracy report with standardized scaling: \n",
    accuracy_score(y_test_std, std_y_predict),
)
print(
    "Classification report standardized scaling: \n",
    classification_report(y_test_std, std_y_predict),
)
