# EDA (Exploretory Data Analysis) includes - datacleaning, data transformation , aggregation and filtering

# Visual Tools for Insights:
# 1. Line plots
# 2. Bar Charts
# 3. Scatter plots
# 4. Heatmaps

# key patterns looking for:
# 1. Relationships between variables
# 2. Distribution of variables (histogram, boxplots)
# 3. outliers or anomalies

# Hands-on project: EDA on sample dataset
# task1: perform data cleaning, Aggregation and filtering
# task2:

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("titanic.csv")
print(df.info())
print(df.describe())

# 1. Handling missing values
df["Age"] = df["Age"].fillna(df["Age"].median())
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode())

# 2. Remove duplicates
d = df.drop_duplicates()

# 3. Filter data: passengers in firstclass
first_class = df[df["Pclass"] == 1]
print("First class Passengers: \n", first_class.head())

# 4. Generate Visualizations
# Bar Graph
survival_by_class = df.groupby("Pclass")["Survived"].mean()
survival_by_class.plot(kind="bar", color="blue")
plt.title("Survival rate ny class")
plt.ylabel("Survival Rate")
plt.show()

# Histogram for Age distribution
sns.histplot(df["Age"], kde=True, bins=20, color="purple")
plt.title("Age distribution")
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.show()

# Scatter Plot: Age V/S Fare
plt.scatter(df["Age"], df["Fare"], alpha=0.5, color="Purple")
plt.title("Age V/S Fare")
plt.xlabel("Age")
plt.ylabel("Fare")
plt.show()
