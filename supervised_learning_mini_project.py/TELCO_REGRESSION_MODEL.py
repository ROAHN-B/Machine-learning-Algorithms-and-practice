import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
import joblib

df = pd.read_csv("Telco_customer.csv")

yes_no_columns = [
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "PaperlessBilling",
    "Churn",
    "gender",
]

le = LabelEncoder()
for col in yes_no_columns:
    df[col] = le.fit_transform(df[col])

# le = LabelEncoder()
# df["Churn"] = le.fit_transform(df["Churn"])
# df["gender"] = le.fit_transform(df["gender"])
# df["Partner"] = le.fit_transform(df["Partner"])
# df["Dependents"] = le.fit_transform(df["Dependents"])
# df["PhoneService"] = le.fit_transform(df["PhoneService"])


# define features and target
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())
x = df.drop(
    columns=[
        "Churn",
        "customerID",
        "MultipleLines",
        "InternetService",
        "Contract",
        "PaymentMethod",
    ]
)


y = df["Churn"]

# scaling features
scaler = StandardScaler()
x = scaler.fit_transform(x)
# Scaling y is not needed as it already in binary (0,1) format

# split dataset
x_train, x_test, y_train, y_test = train_test_split(
    x, y, random_state=42, test_size=0.2
)

# training Logistic regression model
log_reg = LogisticRegression(max_iter=200)
log_reg.fit(x_train, y_train)

# Training KNN model
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(x_train, y_train)

# Evaluate models
log_pred = log_reg.predict(x_test)
knn_pred = knn_model.predict(x_test)

print(
    "Logistic regression classification report: \n",
    classification_report(y_test, log_pred),
)
print("Confusion Matrix of Logistic regression: \n", confusion_matrix(y_test, log_pred))

print("Knn classification report : \n", classification_report(y_test, knn_pred))
print("Confusion matrix of Knn classification: \n", confusion_matrix(y_test, knn_pred))


joblib.dump(knn_model, "models/knn_model.pkl")
joblib.dump(log_reg, "models/logistic_regression_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")
