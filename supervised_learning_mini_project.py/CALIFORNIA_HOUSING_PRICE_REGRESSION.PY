from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_california_housing

data = fetch_california_housing(as_frame=True)
df = data.frame

# define features and target
x = df[["MedInc", "HouseAge", "AveRooms"]]
y = df["MedHouseVal"]

# split training and testing data
x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=42, test_size=0.2
)

# Train model
model = LinearRegression()
model.fit(x_train, y_train)

# prediction
y_predict = model.predict(x_test)

# Evaluation
print("Mean squared Error: ", mean_squared_error(y_test, y_predict))
print("R squared Error: ", r2_score(y_test, y_predict))



