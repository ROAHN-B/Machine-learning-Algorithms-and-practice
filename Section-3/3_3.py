# derivative - tell hiw a function changes with respect to it's input
# helps in optimizing loss functions
# library - sympy
import sympy as sp
import numpy as np

# x = sp.Symbol("x")
# f = x**2
# derivative = sp.diff(f, x)  # calculates derivatives
# print(derivative)

# # Gradient - Vector of all partial derivatives, indicating the direction of the steepest ascent
# x, y = sp.symbols("x y")
# f = x**2 + y**2
# grad_x = sp.diff(f, x)
# gread_y = sp.diff(f, y)
# print("Partial derivative: \n", grad_x, gread_y)

# Gradient descent - Iterative algorithm used to minimize a loss function and improve model's predectibility
# x = sp.symbols("x")
# f = x / 2
# derivative = sp.diff(f, x)
# print("derivative of the function: \n", derivative)

# x, y = sp.symbols("x y")
# f = x**2 + 3 * y**5 - 4 * x * y

# grad_x = sp.diff(f, x)  # Partial difference with respect to x
# grad_y = sp.diff(f, y)  # Partial difference with respect to y

# print("Gradient X :\n", grad_x)
# print("Gradient Y : \n", grad_y)


# Defin gradient descent function
def gradient_descent(X, y, theta, learning_rate, iterations):
    m = len(y)
    for _ in range(iterations):
        predictions = np.dot(X, theta)
        errors = predictions - y
        gradients = 1 / m * np.dot(X.T, errors)
        theta -= learning_rate * gradients
    return theta


# sample data
X = np.array([[1, 2], [4, 5], [3, 4]])
y = np.array([2, 2.5, 0.1])

theta = np.array([0.1, 0.1])

learning_rate = 0.1
iterations = 1000


optimized_theta = gradient_descent(X, y, theta, learning_rate, iterations)
print(optimized_theta)
