"""
XGBoost: Extreme Gradient boosting--> Advance implementation of Gradienf boosing algorithm designed for speed and performance.
It introduces various enhancements that make it faster, more efficient and capable handling complex datasets.

WHAT IT IMPROVES:
1. Speed
2. automatically handles missing data
3. regularization
4. custom loss functions
5. Tree pruning

HYPERPARAMETER IN XGBOOST
1. Learning rate
2. number of trees (n_estimators)
3. tree depth (max_depth)
4. Subsample
5. Cosample_bytree
6. Regularization parameters
"""

####Exercise####

"""
Objective: Train XGBoost model on a dataset , tune hyperparameter using cross-validation
and compare it's performance with a gradient Boosting model.
"""

import xgboost as xgb
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier
from sklearn.ensemble import GradientBoostingClassifier

# load dataset
data = load_breast_cancer()
x = data.data
y = data.target

print("features", data.feature_names)
print("Target", data.target_names)

# split data
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

# convert dataset to matrix
dtrain = xgb.DMatrix(x_train, label=y_train)  # creates matrix
dtest = xgb.DMatrix(x_test, label=y_test)

params = {
    "objective": "binary:logistic",
    "eval_metric": "logloss",
    "max_depth": 3,
    "eta": 0.1
}
xgb_model = xgb.train(params, dtrain, num_boost_round=100)

# predict
y_pred = (xgb_model.predict(dtest) > 0.5).astype(int)

# evaluate performance
accuracy = accuracy_score(y_test, y_pred)
print(f"XGBOOST accuracy: {accuracy}")
print(f"XGBOOST classification report: {classification_report(y_test, y_pred)}")


##########HYPERPARAMETER TUNING##########
params_grid = {
    "learning_rate": [0.01, 0.1, 0.2],
    "n_estimators": [50, 100, 200],
    "max_depth": [3, 5, 7],
    "subsample": [0.8, 1.0]
    
}

# initialize XGBoost classifier
xgb_clf = XGBClassifier(eval_metric="logloss", random_state=42)

# perform grid search
grid_search = GridSearchCV(
    estimator=xgb_clf, param_grid=params_grid, cv=5, scoring="accuracy", n_jobs=-1
)

grid_search.fit(x_train, y_train)

# display the best parameters
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best cross-validation accuracy: {grid_search.best_score_}")


####TRAIN THE GRADIENT BOOSTING MODEL#######
gb_model = GradientBoostingClassifier(random_state=42)
gb_model.fit(x_train, y_train)
y_pred_gb = gb_model.predict(x_test)


# Evaluate Gradient Boosting Performance
accuracy_gb = accuracy_score(y_test, y_pred_gb)
classification_report_gb = classification_report(y_test, y_pred_gb)

print("Accuracy score of gradient boosting: ", accuracy_gb)
print("Classification report of gradient boosting: ", classification_report_gb)
