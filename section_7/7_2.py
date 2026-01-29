"""
Bagging: Ensemble learning technique that trains on multiple models on different subsets of data,
created by random sampling with replacement.
Regression: Average the predictions of individuals
classification: Use majority voting to determine final class.

Random Forest: Ensemble learning method that builds multiple decision trees using bagging.
key features:
boosting sampling
feature randomness
prediction aggregation

Advantages:
1. Handles both regression and classification tasks effectively.
2. Works well with high-dimentional data.
3. Reduces overfitting

KEY PARAMETERS:
1. NUMBER OF TREES(n_estimators)
    The number of decision trees in the forest.
    Large values reduce variance but increases computational cost.
2. Maximum_depth(max_depth):
    Limits depth to avoid overfitting
    shallower trees generalizes better but may underfit.
3. Feature Selection(max_features):
    Number of features to consider when looking for the best split.
    options:
        sqrt|log2|None
4. Minimum_samples per leaf(min_samples_leaf):
    Minimum number of samples required in a leaf node.
    Prevents overly complex trees by ensuring each leaf contains enough samples.
"""

####EXERCISE####
"""
OBJECTIVE: Train a Random forest classifier on a dataset, tune it's parameters,
and evaluate its performance.
"""

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import GridSearchCV

# load the dataset
data = load_breast_cancer()
x = data.data
y = data.target

# split the dataset
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

# display the dataset info
print("Features: \n", data.feature_names)
print("Target: \n", data.target_names)

# Train the random forest classifier
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(x_train, y_train)

# prediction
y_predict = rf_model.predict(x_test)

# Evaluation
print("Accuracy score: \n", accuracy_score(y_test, y_predict))
print("Classification report: \n", classification_report(y_test, y_predict))


########DEFINE THE HYPERPARAMETER GRID###########
param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [None, 10, 20],
    "max_features": ["sqrt", "log2", "None"],
}

grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=5,
    scoring="accuracy",
    n_jobs=-1,
)

grid_search.fit(x_train, y_train)

print(f"Best Parameters: {grid_search.best_params_}")
print(f"Best Cross-validation Accuracy: {grid_search.best_score_}")
