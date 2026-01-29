import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


df = pd.read_csv("Iris.csv")
x = df[["SepalLengthCm"]].values
y = df[["SepalWidthCm"]].values

# Add bias to feature matrix
x_b = np.c_[np.ones((len(x), 1)), x]

# Initialize parameters
theta = np.random.randn(2, 1)
learning_rate = 0.01
iterations = 1000


def predict(X, theta):
    return np.dot(X, theta)


# gradient descent to optimize the model's parameters
def gradient_descent(X, y, theta, learning_rate, iterations):
    m = len(y)
    for _ in range(iterations):
        gradients = (2 / m) * np.dot(X.T, (np.dot(X, theta) - y))
        theta -= learning_rate * gradients
    return theta


# Calculate Evaluation metrices
def mean_squared_errors(y_true, y_pred):
    return np.mean((y_true - y_pred) ** 2)


def r_squared(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred) ** 2)
    ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
    return 1 - (ss_res / ss_tot)


# perform the gradient descent
theta_optimized = gradient_descent(x_b, y, theta, learning_rate, iterations)

# Predictions and evaluations
y_pred = predict(x_b, theta_optimized)
mse = mean_squared_errors(y, y_pred)
r2 = r_squared(y, y_pred)

print("Optimized parameters: \n", theta_optimized)
print("MSE: \n", mse)
print("R_square: \n", r2)
plt.figure(figsize=(10, 6))
plt.scatter(x, y, color="blue", alpha=0.5, label="Actual Iris data")
y_line_pred = theta_optimized[0] + theta_optimized[1] * x

plt.plot(
    x,
    y_line_pred,
    color="red",
    linewidth=2,
    label="Model prediction line",
)

plt.xlabel("Sepal length(cm)")
plt.ylabel("sepal width (cm)")
print("Linear regression: Sepal Length V/S Width ")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.7)
plt.show()
