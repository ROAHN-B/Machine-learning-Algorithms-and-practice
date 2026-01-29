import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

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


df = pd.read_csv("bike_dataset.csv")
# print(df.info())
# print(df.describe())
# print(df.head())

# convert dteday into datetime
df["dteday"] = pd.to_datetime(df["dteday"])
print(df["dteday"].head())

# Create new feature
df["day_of_week"] = df["dteday"].dt.day_name()
df["month"] = df["dteday"].dt.month
df["year"] = df["dteday"].dt.year

# display the new features
print("\n New features derived from dte")
print(df[["dteday", "day_of_week", "month", "year"]].head())


###Applying Polynomial regression

##Select features and target
x = df[["temp"]]
y = df["cnt"]

# apply polynomial transformation
poly = PolynomialFeatures(degree=2, include_bias=False)
x_poly = poly.fit_transform(x)

# display the transformed feature
# print("\n Original and polynomial features: \n")
# print(pd.DataFrame(x_poly, columns=["temp", "temp^2"]).head())



#split the dataset 
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)

#polynomial tarinig tests for comparing performances.
x_poly_train,x_poly_test=train_test_split(x,test_size=0.2,random_state=42)

#Train and Evaluate model with original features
model_original=LinearRegression()
model_poly=LinearRegression()

#Original 
model_original.fit(x_train,y_train)
y_predict_original=model_original.predict(x_test)
mse_original=mean_squared_error(y_test,y_predict_original)
print("Mean squared error of original model with original features is: ", mse_original)

#Polynomial
model_poly.fit(x_poly_train,y_train)
y_predict_poly=model_poly.predict(x_poly_test)
mse_poly=mean_squared_error(y_test,y_predict_poly)
print("Mean squared error of polynomial model is: ", mse_poly)

