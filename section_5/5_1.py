import pandas as pd
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt

# load data
df = pd.read_csv("tips.csv")

# define features and target
features = df[["total_bill", "size"]]
target = df[
    "tip"
]  # I am predicting the target (tip) column using features i.e. total_bill column and size column

print("Features:\n", features.head())
print("Target :\n", target.head())

# Spliting data in training and testing data
x_train, x_test, y_train, y_test = train_test_split(  # X= Feature , Y=Target
    features, target, test_size=0.2, random_state=42
)  # test data is 20% of the tarining data
# That 20% will be picked randomly from data thats why we used
# "random_state=42"


print("training dataset: ", x_train.shape)
print("testing dataset: ", x_test.shape)

# output: training dataset:  (195, 2)
# testing dataset:  (49, 2)

sns.pairplot(
    df,
    x_vars=["total_bill", "size"],
    y_vars="tip",
    height=5,
    aspect=0.8,
    kind="scatter",
)
plt.title("Features v/s tarhet relationship")
plt.show()
