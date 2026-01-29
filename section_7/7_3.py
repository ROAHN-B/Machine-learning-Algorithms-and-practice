"""
Gradient Boosting: Boosting algorithm that builds models sequentially by minimizing a loss function using
gradient descent.

"""

#####EXERCISE#####
"""
Objective: Train and evaluate a gradient Boosting model on dataset , tune key parameters and 
compare its performance with a raendom forest model.
"""

from sklearn.datasets import load_breast_cancer
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# load the dataset
data = load_breast_cancer()
x = data.data
y = data.target

# split the dataset
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)


# display dataset information
# print(f"Features : {data.feature_names}")
# print(f"Target: {data.target_names}")


# Train the gradient boosting model
grad_boost = GradientBoostingClassifier(random_state=42)
grad_boost.fit(x_train, y_train)

# prediction
grad_boost_predict = grad_boost.predict(x_test)

# Evaluation
print(f"Accuracy: {accuracy_score(y_test, grad_boost_predict):.2f}")
print(f"Classification report: ", classification_report(y_test, grad_boost_predict))


# Define Hyperparameter grid
param_grid = {
    "learning_rate": [0.01, 0.1, 0.2],
    "n_estimators": [50, 100, 200],
    "max_depth": [3, 5, 7],
}

# perform grid search
grid_search = GridSearchCV(
    estimator=GradientBoostingClassifier(random_state=42),
    param_grid=param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1,
)

grid_search.fit(x_train, y_train)

# display best parameters and score
print(f"Best parameters: {grid_search.best_params_}")
print(f"best cross-validation accuracy: {grid_search.best_score_}")

# Train the random forest classifier
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(x_train, y_train)

# predict
y_pred_rf = rf_model.predict(x_test)

# Evaluate the performance
print(f"Accuracy score of random forest: {accuracy_score(y_test, y_pred_rf)}")
print(f"Classification of random forest: {classification_report(y_test, y_pred_rf)}")
