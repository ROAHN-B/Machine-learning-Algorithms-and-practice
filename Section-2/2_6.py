# Using matplot for plotting
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# basic plot
# x = [1, 2, 3, 4, 6, 8, 9]
# y = [10, 20, 60, 40, 45, 98, 26]
# plt.plot(x, y)
# plt.show()


# line plot
# plt.plot([1, 2, 3, 4, 6, 8, 9], [10, 20, 60, 40, 45, 98, 26], label="Trend")
# plt.title("Line plot")
# plt.xlabel("X-axis")
# plt.ylabel("Y-axis")
# plt.legend()
# plt.show()


# Bar chart
# categories = ["A", "B", "C"]
# values = [12, 43, 45]
# plt.bar(categories, values, color="blue")
# plt.title("Bar chart")
# plt.show()


# Histogram
# data = [2, 3, 2, 4, 6, 3]
# plt.hist(data, bins=4, color="green", edgecolor="black")
# plt.title("histogram")
# plt.show()


# Scatter Plot
# x = [1, 2, 3, 4, 5]
# y = [10, 20, 30, 40, 50]
# plt.scatter(x, y, color="red")
# plt.title("Scatter plot")
# plt.show()


# Customizing Plots
# x = [1, 2, 3, 4, 5]
# y = [10, 20, 30, 40, 50]
# plt.scatter(x, y, color="red", linestyle="--", marker="X")
# plt.title("Scatter plot")
# plt.xlabel("X-axis")
# plt.ylabel("Y-axis")
# plt.legend("database")
# plt.show()


#####################SEABORN FOR ADVANCED VISUALIZATION####################

# heatmap
# data = np.random.rand(5, 5)
# # sns.heatmap(data, annot=True, cmap="coolwarm")
# # plt.title("Heat Map")
# sns.pairplot(df) #shows pair wise relationships
# plt.show()


#####################EXECISE##########################

# line plot
# years = [2010, 2013, 2017, 2023]
# sales = [100, 200, 300, 400]

# plt.plot(years, sales, label="sales Trend", color="red", marker="o")
# plt.xlabel("X-axis")
# plt.ylabel("Years")
# plt.legend()
# plt.show()

# Bar chart
# categories = ["Electronics", "Clothing", "Groceries"]
# revenue = [230, 340, 456]
# plt.bar(categories, revenue, color="green")
# plt.title("Revenue of electronics items")
# plt.show()


# Scatter plot
# hours = [2, 3, 10, 5, 6]
# exams = [30, 20, 70, 40, 60]
# plt.scatter(hours, exams, color="red")
# plt.title("Hours V/S Exam scores")
# plt.xlabel("time in hours")
# plt.ylabel("Exams score")
# plt.show()


# Heat map using seaborn
# df = pd.read_csv("Iris.csv")
# del df["Species"]

# # calculate correlation matrix

# correlation_matrix = df.corr()

# sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
# plt.title("Correlation heatmap of Iris dataset")
# plt.show()
