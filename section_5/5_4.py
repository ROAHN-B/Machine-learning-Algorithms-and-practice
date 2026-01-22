# Logistic regression- applies sigmoid function to linear equation
# import numpy as np
# import matplotlib.pyplot as plt


# # sigmoid function
# def sigmoid(z):
#     return 1 / (1 + np.exp(-z))


# # generate values
# z = np.linspace(-10, 10, 100)
# sigmoid_values = sigmoid(z)

# plt.plot(z, sigmoid_values)
# plt.title("Sigmoid function")
# plt.xlabel("z")
# plt.ylabel("sigmoid")
# plt.grid()
# plt.show()


####################IMPLEMENT LOGISTIC LOGISTIC REGRESSION############
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
)

import matplotlib.pyplot as plt


# generate symthetic data
np.random.seed(42)
n_samples = 200
x = np.random.rand(n_samples, 2) * 10
y = (x[:, 0] * 1.5 + x[:, 1] > 15).astype(int)

# create dataframe
df = pd.DataFrame(x, columns=["Age", "Salary"])
df["Purchase"] = y

# split data
x_train, x_test, y_train, y_test = train_test_split(
    df[["Age", "Salary"]], df[["Purchase"]], test_size=0.2, random_state=42
)

# Train logistic regression model
model = LogisticRegression()
model.fit(x_train, y_train)

y_pred = model.predict(x_test)

# Evaluation of model
print("Accuracy: \n", accuracy_score(y_test, y_pred))
print("Precision: \n", precision_score(y_test, y_pred))
print("Recall: \n", recall_score(y_test, y_pred))
print("f1 score: \n", f1_score(y_test, y_pred))
print("Classification report: \n", classification_report(y_test, y_pred))


# plot decision
x_min, x_max = x[:, 0].min() - 1, x["Age"].max() + 1

y_min, y_max = x[:, 0].min() - 1, x["Salary"].max() + 1

xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.1), np.arange(y_min, y_max, 0.1))

# predict ptobabilities on grid
z = model.predict(np.c_[xx.ravel(), yy.ravel()])


# plot
plt.contour(xx, yy, z, aplha=0.8, color="coolwarm")
plt.scatter(x_test["Age"], x_test["Salary"], c=y_test, edgecolors="k", cmap="coolwarm")
plt.title("Logistic regression")
plt.show()
