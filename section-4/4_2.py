# Gaussian disribution: used in algorithms like naive bayes
# binomial distribution : used in binary classification like logistic regression
# poisson distribution: applied in modeling count data
# Uniform distribution: Used in random sampling and initialization of parameters

# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.stats import norm, binom, poisson, uniform
# import seaborn as sns

# # Gaussian distribution
# x = np.linspace(-4, 4, 100)
# plt.plot(x, norm.pdf(x, loc=0, scale=1), label="Gaussian (u=0, s=1 )")

# # bonomial distribution
# n, p = 10, 0.5
# x = np.arange(0, n + 1)
# plt.bar(x, binom.pmf(x, n, p), alpha=0.7, label="Binomial (n=10,p=0.5)")

# # Poisson distribution
# lam = 3
# x = np.arange(0, 10)
# plt.bar(x, poisson.pmf(x, lam), alpha=0.7, label="poisson (l=3)")

# # Uniform distribution
# x = np.random.uniform(low=0, high=10, size=1000)
# sns.histplot(x, kde=True, label="Uniform", color="red")

# plt.title("Probability distributions")
# plt.legend()
# plt.show()


############Exercise##################3
from scipy.stats import skew, kurtosis
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv("Iris.csv")
# Analyze sepal length
feature = df["SepalLengthCm"]
print("skewness: ", skew(feature))
print("kurtosis: ", kurtosis(feature))


# Visualization distrabution
sns.histplot(feature, kde=True)
plt.title("Distribution of sepal length")
plt.show()
