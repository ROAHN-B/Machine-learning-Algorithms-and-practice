"""
Ensemble Methods: Machine learning methods which uses multiple regression models to provide final output.
TYPES OF ENSEMBLE:
1. Bagging (Bootstrap Aggregation): Training model on different subsets of data created using bootstrapping.
Combines Predictions by averaging or majority voting(classification)
ex- Random Forest

2. Boosting: Training model sequentially, where each model focuses on creating the errors made by the previous ones.
combines predictions using weighted averaging or voting.
ex- AdaBoost, Gradient Boosting, XGBoost,LightGBM.
strengths: educes bais and varience by focusing on hard-to-predict instances.


3. STACKING
Combines predictions from multiple base models (of different types) using a meta-model to learn how to best combine outputs.
strengths: canutilize diverse model types to maximize performance.

"""

####EXERCISE####
"""
OBJECTIVE: build a basic ensemble model combining predictions from linear regression, decision tree, and K-nn classification.

"""

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score

# load dataset
data = load_iris()
x = data.data
y = data.target

# split the dataset
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

# scale features
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)

# NOW WE WILL TRAIN LOGISTIC REGRESSION , DECISION TREE, K-NN CLASSIFICATION
log_regg = LogisticRegression()
dec_tree_regg = DecisionTreeClassifier()
knn = KNeighborsClassifier()

log_regg.fit(x_train, y_train)
dec_tree_regg.fit(x_train, y_train)
knn.fit(x_train, y_train)

# creating voting classifier
ensemble_model = VotingClassifier(
    estimators=[("log_regg", log_regg), ("dec_tree_regg", dec_tree_regg), ("knn", knn)],
    voting="hard",  # majority voting
)

# Train the ensemble model
ensemble_model.fit(x_train, y_train)

# predictions
y_predict_ensemble = ensemble_model.predict(x_test)

# Evaluate accuracy
accuracy = accuracy_score(y_test, y_predict_ensemble)
print(f"Ensemble model accuracy is : {accuracy:.2f}")

# compare the model performance
y_pred_log = log_regg.predict(x_test)
y_pred_dt = dec_tree_regg.predict(x_test)
y_pred_knn = knn.predict(x_test)

print(f"Logistic regression score: {accuracy_score(y_test, y_pred_log)}")
print(f"Decision Tree score: {accuracy_score(y_test, y_pred_dt)}")
print(f"Knn-classification score: {accuracy_score(y_test, y_pred_knn)}")
