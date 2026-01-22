# # polynomial regression - Used for non-linear regression
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import Ridge, Lasso
from sklearn.model_selection import train_test_split

# # Generate synthetic data
# np.random.seed(42)
# x = np.random.rand(100, 1) * 10
# y = 3 * x**2 + 2 * x + np.random.randn(100, 1) * 5

# poly_features = PolynomialFeatures(degree=2, include_bias=False)
# x_poly = poly_features.fit_transform(x)

# # fit polynimial regression
# model = LinearRegression()
# model.fit(x_poly, y)
# y_pred = model.predict(x_poly)

# # plot the results
# plt.scatter(x, y, color="blue", label="Actual data")
# plt.scatter(x, y_pred, color="red", label="predicted data")
# plt.title("Plotnomail regression")
# plt.xlabel("x")
# plt.ylabel("y")
# plt.legend()
# plt.show()


# # evaluate model
# mse = mean_squared_error(y, y_pred)
# rscore = r2_score(y, y_pred)

# print("mse: \n", mse)
# print("r square: \n", rscore)


#####REGULARIZATION USED FOR REDUCING THE OVERFITTING OF THE MODEL###########

# np.random.seed(42)
# x = np.random.rand(100, 1) * 10
# y = 3 * x**2 + 2 * x + np.random.randn(100, 1) * 5

# poly_features = PolynomialFeatures(degree=2, include_bias=False)
# x_poly = poly_features.fit_transform(x)

# x_test, x_train, y_test, y_train = train_test_split(
#     x_poly, y, test_size=0.2, random_state=42
# )

# # Ridge Regression
# ridge_model = Ridge(alpha=1)
# ridge_model.fit(x_train, y_train)
# ridge_predictions = ridge_model.predict(x_test)

# # Lasso Regression
# lasso_model = Lasso(alpha=1)
# lasso_model.fit(x_train, y_train)
# lasso_predictions = lasso_model.predict(x_test)


# # Evaluate ridge
# ridge_mse = mean_squared_error(y_test, ridge_predictions)
# print("ridge regression mse: ", ridge_mse)

# # Evaluate lasso
# lasso_mse = mean_squared_error(y_test, lasso_predictions)
# print("lasso regression mse: ", lasso_mse)


#############HOUSE PREDICTION MODEL USING POLYNOMIAL REGRESSION#############3
# load the dataset
# data = fetch_california_housing(as_frame=True)
# df = data.frame


# # select feature (median income) and target (median house value)
# x = df[["MedInc"]]
# y = df[["MedHouseVal"]]

# # Transform features to polynomial features
# poly = PolynomialFeatures(degree=2, include_bias=False)
# x_poly = poly.fit_transform(x)

# # fit polynomial regression model
# model = LinearRegression()
# y_pred = model.fit(x_poly, y)

# y_pred = model.predict(x_poly)

# # plot the actual v/s predictions values
# plt.figure(figsize=(10, 6))
# plt.scatter(x, y, color="blue", label="Actual data", alpha=0.5)
# plt.scatter(x, y_pred, color="red", label="Predicted curve", alpha=0.5)
# plt.title("Polynomial Regression")
# plt.xlabel("Midian income in california")
# plt.ylabel("median house value in california")
# plt.legend()
# plt.show()


# # Evaluation of model

# mse = mean_squared_error(y, y_pred)
# rscore = r2_score(y, y_pred)

# print("mse : \n", mse)
# print("rscore: \n", rscore)


#############RIDGE AND LASSO REGRESSION###############
data = fetch_california_housing(as_frame=True)
df = data.frame

# select feature (median income) and target (median house value)
x = df[["MedInc"]]
y = df[["MedHouseVal"]]

# Transform features to polynomial features
poly = PolynomialFeatures(degree=2, include_bias=False)
x_poly = poly.fit_transform(x)

x_test, x_train, y_test, y_train = train_test_split(
    x_poly, y, test_size=0.2, random_state=42
)

ridge_model = Ridge(alpha=0.5)
ridge_model.fit(x_train, y_train)  # train model
ridge_predictions = ridge_model.predict(x_test)


lasso_model = Lasso(alpha=0.5)
lasso_model.fit(x_train, y_train)
lasso_predictions = lasso_model.predict(x_test)

# Evaluate ridge regression
ridge_mse = mean_squared_error(y_test, ridge_predictions)
print("mse :\n", ridge_mse)


lasso_mse = mean_squared_error(y_test, lasso_predictions)
print("mse :\n", ridge_mse)


plt.figure(figsize=(10, 6))
plt.scatter(x_test[:, 0], y_test, color="blue", label="Actual data", alpha=0.5)
plt.scatter(
    x_test[:, 0], ridge_predictions, color="green", label="ridgr predictions", alpha=0.5
)
plt.scatter(
    x_test[:, 0],
    lasso_predictions,
    color="orange",
    label="Lasso predictions",
    alpha=0.5,
)
plt.title("ridge regression v/s lasso regression")
plt.xlabel("median of income")
plt.ylabel("medain house value in california")
plt.legend()
plt.show()
