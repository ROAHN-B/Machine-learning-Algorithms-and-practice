"""
One-hot Encoding and Label Encoding : this is done on categorical data

One-hot Encoding: creates binary columns for each category in a categorical features.
divides each rows in 0 or 1 respective of the category

Application:
Categorical features with a small number of unique categories.

Label Encoding: ordinal features where order matters

DEALING WITH HIGH CARDINALITY CATEGORICAL FEATURES:
high cardinality means - categorical values which contain large number of unique categories.
ex- name of cities,etc.

solution:
1. frequency encoding: replace categories with their occurence frequency in the dataset.
2. target encoding: replace categories with the name of the target variable for each category.

WHEN TO USE WHICH TYPE OF ENCODING?
ONE-HOT ENCODING: Nominal features with a small numner if unique categories.
LABEL ENCODING: Ordinal features or when used with algorithms like tree-based models.
FREQUENCY ENCODING: High-cardinality features in both regression and classification tasks.
TARGET ENCODING: High-cardinality features in supervised learning tasks.
"""

###EXERCISE###
"""
Objective: 1.Apply one-hot encoding to a dataset
2. Experiment with different encoding techniques and observe their impact on model's performance.
"""

import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# load the titanic dataset
df = pd.read_csv("titanic.csv")
print(df.info())
print(df.describe())

print("dataset preview: \n", df.head())

# Apply one hot encoding
df_one_hot = pd.get_dummies(df, columns=["Sex", "Embarked"], drop_first=True)

# display the encoded dataset
print("\n one-hot encoded dataset: ", df_one_hot)


#####APPLY DIFFERENT TYPES OF ENCODING#####
# Applying label encoding
label_encoder = LabelEncoder()
df["Pclass_Encoded"] = label_encoder.fit_transform(df["Pclass"])

# display encoded dataset
print("\n label encoded dataset: ", df[["Pclass", "Pclass_Encoded"]].head())


# Apply frequency Encoding
df["Ticket_frequency"] = df["Ticket"].map(df["Ticket"].value_counts())

# display frequency encoded features
print("\n Freuquency encoded features: ", df[["Ticket", "Ticket_frequency"]].head())

df_one_hot = df_one_hot.fillna(df_one_hot.mode().iloc[0])
print(df_one_hot["Age"])
x = df_one_hot.drop(columns=["Survived", "Name", "Cabin", "Ticket"])
y = df["Survived"]

# split dataset
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

# train model
model = LogisticRegression(max_iter=200)
model.fit(x_train, y_train)

# predictictions
y_predict = model.predict(x_test)

# Evalution
print("Accuracy score after one-hot encoding: ", accuracy_score(y_test, y_predict))
