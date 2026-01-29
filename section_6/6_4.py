"""
Introduction to feature engineering: Process of retaining and including most relevant features.
And removing the irrelevenr features.

WHY FEATURE ENGINEERING:
    1.Improves model's performance.
    2. Enhances interpretability.
    3. Reduces overfitting.
    4. Increases efficiency.

WHEN TO USE FEATURE SELECTION
    1. for high dimentional dataset
    2. when data is interrelated to eachother

TECHNIQUES FOR FEATURE ENGINEERING:
    Filter method:
        1. Evaluate the relevance of feature by analyzing their statistical properties in relation with the target variable.
        Ex - correlation | Mutual Information.
    Wrapper Methods:
        1. Iteratively selects features by tarining and evaluating a model.
        ex: forward selection | Backward selection.
    Embedded methods:
        1. Perform feature selection as a part of model training process.
        ex: lasso Regression | Tree-Based models.
"""

####EXERCISE####
"""
1. Use correlation and mutual information to select important features from dataset.
2. Apply tree based models to identify most important features.
"""
from sklearn.datasets import load_diabetes
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_selection import mutual_info_regression
from sklearn.ensemble import RandomForestRegressor
import numpy as np


# load the dataset
data = load_diabetes()
df = pd.DataFrame(data.data, columns=data.feature_names)
df["target"] = data.target

# display dataset information
# print(df.info())
# print(df.describe())
# print(df.head())

# Calculate correlation matrix
# correlation_matrix = df.corr()

# # plot the graph
# plt.figure(figsize=(10, 8))
# sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
# plt.title("Correlation matrix")
# plt.show()

# # select features with high correlation to the target
# correlated_features = correlation_matrix["target"].sort_values(ascending=False)
# print("Features most correlated with target are: ", correlated_features)


###Feature selection using mutual information###

# Seprate features and target
x = df.drop(columns=["target"])
y = df["target"]

# # calculate mutual information
# mutual_info = mutual_info_regression(x, y)

# # create a dataframe
# mi_df = pd.DataFrame({"Feature": x.columns, "Mutual Information": mutual_info})
# mi_df = mi_df.sort_values(by="Mutual Information", ascending=False)
# print("Mutual information scores: \n")
# print(mi_df)


#### feature selection using tree-based model ####
model = RandomForestRegressor(random_state=42)
model.fit(x, y)

# Get feature importances
feature_importance = model.feature_importances_
importance_df = pd.DataFrame({"Feature": x.columns, "Importance": feature_importance})

importance_df = importance_df.sort_values(by="Importance", ascending=False)
print("Feature importamce from random forest: \n")
print(importance_df)

plt.figure(figsize=(10, 8))
plt.bar(importance_df["Feature"], importance_df["Importance"])
plt.gca().invert_yaxis()
plt.title("Feature importance from random forest")
plt.show()
