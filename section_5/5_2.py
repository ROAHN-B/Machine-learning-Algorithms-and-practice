import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt


# Generating synthesised data
np.random.seed(42)
x = np.random.rand(100, 1) * 100
y = 3 * x + np.random.randn(100, 1) * 2

print("x: ", x)
print("y: ", y)


# Split data
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

# Fit Linear Regression
model = LinearRegression()
model.fit(x_train, y_train)

# make predictions
y_pred = model.predict(x_test)
print("Y_predict: ", y_pred)

# print coefficient
print("Slope: ", model.coef_[0][0])
print("Intercept: ", model.intercept_[0])


plt.scatter(x_test, y_test, color="blue", label="Actual data")
plt.plot(x_test, y_pred, color="red", label="predict data")

plt.title("Linear regression model")
plt.xlabel("X")
plt.ylabel("Y_predict")
plt.legend()
plt.show()


# Evaluate performance
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mse: ", mse)
print("r squared: ", r2)
