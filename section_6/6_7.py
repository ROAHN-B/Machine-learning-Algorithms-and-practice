"""
Cross Validation: used to assess how well a model generalizes to a independent dataset.
k-fold,startified k-fold, Leave-one-out cross validation(LOOCV).

Hyper parameters: Hyperparameters are the paramaters that are not learned by the model but are set but before training,
tuning these hyper parameters is crucuial for optimizing the model performance.

techniques of hyperparameter:
1. Grid search: Exhaustively searches for a hyperparameter space.
2. random search: Randomly samples combinations of hyperparameters from the predefined space.

IMPORTANCE:
1. prevent overfitting and underfitting
2. Enhances model performance by optimizing critical settings.

"""

"""
OBJECTIVE: PERFORM END-TO-END FEATURE ENGINEERING , MODEL EVALUATION AND HYPERPARAMETER TUNING ON A DATASET.

TASK-1 : PERFORM FEATURE ENGINEERING
TASK-2 : TRAIN AND EVALUATE MODELS
TASK-3 : APPLY GRID SEARCH FOR HYPERPARAMETER TUNING 
"""
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

df = pd.read_csv("titanic.csv")
print(df.head())

# select relevent fields
# PassengerId,Survived,Pclass,Name,Sex,Age,SibSp,Parch,Ticket,Fare,Cabin,Embarked
df = df[["Pclass", "Sex", "Age", "Fare", "Embarked", "Survived"]]

# handle missing values
df.fillna({"Age": df["Age"].median()}, inplace=True)
df.fillna({"Embarked": df["Embarked"].mode()[0]}, inplace=True)

# set features and target
x = df.drop(columns=["Survived"])
y = df["Survived"]


# Apply feature Scaling and encoding
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), ["Age", "Fare"]),  # Numerical scaling
        ("cat", OneHotEncoder(), ["Pclass", "Sex", "Embarked"]),  # Categorical encoding
    ]
)

x_preprocessed = preprocessor.fit_transform(x)
# -----------------------------------------------------
# ----------------TASK-1 DONE---------------------------

####train and Evaluate logistic regression

log_model = LogisticRegression()
log_scores = cross_val_score(log_model, x_preprocessed, y, cv=5, scoring="accuracy")
print(f"Logistic regression Accuracy:  {log_scores.mean():.2f}")

# Train and evaluate random forest
rf_model = RandomForestClassifier(random_state=42)
rf_scores = cross_val_score(rf_model, x_preprocessed, y, cv=5, scoring="accuracy")
print(f"Rnadom forest accuracy: {rf_scores.mean():.2f}")
# -----------------------------------------------------
# ----------------TASK-2 DONE--------------------------


# Define Hyperparameter Grid
param_grid = {
    "n_estimators": [50, 100, 200],
    "max_depth": [None, 10, 20],
    "min_samples_split": [2, 5, 10],
}

# perform grid search
grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    scoring="accuracy",
    cv=5,
    n_jobs=-1,
)

grid_search.fit(x_preprocessed, y)

# display the best hyperparameter and score
print(f"Best Hyperparameter score is : {grid_search.best_params_}")
print(f"Best Accuracy: {grid_search.best_score_}")
# -----------------------------------------------------
# ----------------TASK-3 DONE--------------------------
