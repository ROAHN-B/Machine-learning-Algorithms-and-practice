# Model Evaluation Metrices for Regression and Classification
# MSE = measures average squared difference between predicted and actual values
# MAE = measures average absolute difference between predicted and actual values
# RMSE = provides errors in same unit as target

##########MODEL EVALUATION#########
# Accuracy = (TP+TN)/(TP+TN+FP+FN)
# precision = (TP)/(TP/FP) fraction of positive predictions that are correct
# Recall (sensitvity) = (TP)=(TP+FN) Actual positives that are correctly identified
# F2 Score = 2 * ((precision * recall)/(precision + recall) harmonic mean of precision and recall

#############CROSS-VALIDATION################3
"""Splitting dataset into multiple training and testing dataset

K-Fold Cross validation: dividing dataset in k equal parts
Trains model on k-1 folds and tests on remaining fold, repeating the process K times

Startified K-fold: Ensures each fold has equal calssification of the data

Leave-One-Out Cross-Validation(LOOCV): Trains model on n-1 samples and tests model on 1, repeated same on all samples
COPUTATIONALLY EXPENSIVE!!

Advantages: Reduces risk of overfitting
provides more generalize evluation of model performance
"""

""" CONFUSION MATRIX
                |   Predicted positives   |      Predicted Negative
                |                         |
Actual Positive |    True Positive(TP)    |       False Negative(FN)
                |                         |
Actual Negative |    False Positive(FP)   |       True Negative(FN)

"""


"""Objective:
        Use K-fold-Cross Validation to obtain more accurate estimate of model performance
"""

# from sklearn.datasets import load_iris
# from sklearn.model_selection import cross_val_score, KFold
# from sklearn.ensemble import RandomForestClassifier

# data = load_iris()
# x, y = data.data, data.target

# # initialize classifier
# model = RandomForestClassifier(random_state=42)

# # perform K-fold cross-validation
# kf = KFold(n_splits=5, shuffle=True, random_state=42)
# cv_scores = cross_val_score(model, x, y, cv=kf, scoring="accuracy")

# print("Cross validation scores: \n", cv_scores)
# print("mean accuracy: \n", cv_scores.mean())


"""Objective: 
    make one logistic regression model and perform confusion matrix on it."""

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report,
)
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt

# load dataset
data = load_iris()
x, y = data.data, data.target

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

# Train logistic regression
model = LogisticRegression(
    max_iter=200
)  # max iterations to get optimum gradient descent
model.fit(x_train, y_train)


# predict
y_pred = model.predict(x_test)

# Generate confusion matrix
cm = confusion_matrix(y_test, y_pred)

# display confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=data.target_names)

disp.plot(cmap="Blues")
plt.tight_layout()
plt.show()

print("Classification report: \n", classification_report(y_test, y_pred))
