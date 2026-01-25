"""
After feature selection we go for feature creation, where we create features
which will improve the performance of the model.

Examples of feature creation:
1. Date-Time Features
2. Interaction features: Capturing two or more features and creating a new feature.
3. Aggregations

Common Transformations:
    1. Logarithmic Transformation: reduces the skewness
    2. Square root transformations: moderately reduces skewness, often used for count for data
    3. Polynomial Transformation: Adds higher order terms (x^2,x^3) to capture non-linear terms.

Importance:
    Enhances model's ability to fit non-linear relationships.
"""

"""
Objective: 
1. Create a new feature from a date column.
2. Apply polynomial transformations to a dataset and compare model 
performance before and after that.
"""

import pandas as pd

df = pd.read_csv("bike_sharing_daily.csv")
print(df.info())
print(df.describe())
print(df.head())
