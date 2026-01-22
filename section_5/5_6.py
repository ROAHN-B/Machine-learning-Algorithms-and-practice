"""K-Nearest Neighbors:
1. instance based learning
2. Distance metric
3. classification
4. regression

Application:
1. Image calssification
2. Recomendation system
3. Medical diagnostics
4. customer segmentation

steps:
1. Feature scaling
2. calculate distances (Euclidean distance)
3. Identify K nearest neighbors

To calculate optimal values of K:
1. usig cross validation
2. k=sqrt(n) ; n= number of training samples
"""

# Exercise
"""objective: Implement knn classification for different values of k.
    And compare the accuracy with logistic regression."""
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.linear_model import LogisticRegression


# Do KNN classification
data = load_iris()
x, y = data.data, data.target

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42
)

"""We scale features so that features with bigger values don't dominate!!"""
scaler = StandardScaler()  # this library uses Min-Max scaling formula
x_train = scaler.fit_transform(x_train)
x_test = scaler.fit_transform(x_test)

best_k = 5

knn = KNeighborsClassifier(n_neighbors=best_k)
knn.fit(x_train, y_train)

y_pred_knn = knn.predict(x_test)
accuracy_knn = accuracy_score(y_test, y_pred_knn)
print(f"Accuracy of knn: {accuracy_knn} ")
# Experiment with different values of K
# for k in range(1,11):
#     #initialize knn model
#     knn=KNeighborsClassifier(n_neighbors=k)
#     knn.fit(x_train,y_train)

#     #predict on test data
#     y_pred=knn.predict(x_test)

#     accuracy=accuracy_score(y_test,y_pred)
#     print(f"f: {k}, accuracy= {accuracy: .2f}")


# Train Logistic regression model
log_reg = LogisticRegression(max_iter=100)
log_reg.fit(x_train, y_train)


# predict using logistic regression
y_pred = log_reg.predict(x_test)

# Evaluation of logistic regression
accuracy_lr = accuracy_score(y_test, y_pred)
print(f"Logistic regression accuracy is {accuracy_lr} ")


# Detailed comparison
print("\nLogistic regression classification report")
print(classification_report(y_test, y_pred))

print("Knn classification report : ")
print(classification_report(y_test, y_pred))
