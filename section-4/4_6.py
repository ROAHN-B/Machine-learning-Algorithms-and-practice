import numpy as np
from scipy.stats import pearsonr, spearmanr
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
# x = np.array([1, 2, 3, 4, 5])
# y = np.array([2, 4, 6, 8, 10])

# # pearson correlation
# r, _ = pearsonr(x, y)
# print("Pearson correlation coefficient: ", r)


# # spearman correlation
# rh0, _ = spearmanr(x, y)
# print("spearman correlation coefficient", rh0)


# Linear Regerssion : method to model the relationship between
# dependent(y) and one or more independent variables (x)

# from sklearn.linear_model import LinearRegression
# import numpy as np

# x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9]).reshape(-1, 1)
# y = np.array([2, 4, 6, 8, 10, 12, 14, 16, 18])
# model = LinearRegression()
# model.fit(x, y)

# print("slope is: \n", model.coef_[0]) # indicates magnitide and relationship of the variables
# print("Intercept: ", model.intercept_) #starting point of regression
# print("r squared: ", model.score(x, y)) #closer to 1 indicate better fit


#############EXERCISE#############################

# df = pd.read_csv("Iris.csv")
# del df["Species"]
# correlation_matrix = df.corr()

# # plot a heat map
# sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
# plt.title("Featur correlation")
# plt.show()


##Fit a linear regression model to a random data
df = pd.read_csv("Iris.csv")
x = np.array(df["SepalLengthCm"]).reshape(-1, 1)
y = np.array(df["SepalWidthCm"]).reshape(-1, 1)
z = np.array(df["PetalLengthCm"]).reshape(-1, 1)
w = np.array(df["PetalWidthCm"]).reshape(-1, 1)

# implement polynomial regression
poly_transformer = PolynomialFeatures(degree=2)
x_poly = poly_transformer.fit_transform(x)


model = LinearRegression()
model.fit(x_poly, y)
x_smooth = np.linspace(x.min(), x.max(), 100).reshape(-1, 1)
x_smooth_poly = poly_transformer.transform(x_smooth)
y_pred = model.fit(x_smooth_poly)
plt.scatter(x, y, z, w, cmap="blue", alpha=0.6, label="data size")
plt.plot(x_smooth, y_pred, color="red", linewidth=2, label="Polynomial Fit (Deg 2)")
# print("slope of data: ", model.coef_[0])
# print("intercept of line: ", model.intercept_)
# print("r square: ", model.score(x, y))
