# Feature Engineering: categorical, numerical and ordinal
# Scaling, Encoding, labeling

"""
Exercises: load a dataset and expolre it's features and identify categorical and numerical data.
Plan which feature will be most suitable for dataset
"using dataset- titanic.csv"
"""

import pandas as pd

# load dataset
df = pd.read_csv("titanic.csv")
print(df.info())
print(df.describe())
print(df.isnull().sum())


# seprate features
categorical_features = df.select_dtypes(include=["object"]).columns
print("categorical features: \n", categorical_features)
numerical_features = df.select_dtypes(include=["Int64"]).columns
print("numerical features: \n", numerical_features)


# display feature engineering
print("categorical engineering: \n")
for col in categorical_features:
    print(f"{col}: \n", df[col].value_counts(), "\n")

# diplay summary of numerical features
print("numerical features of summary: ", df[numerical_features].describe())

