import numpy as np
import pandas as pd
from scipy.stats import norm, t

# smaple data
data = pd.read_csv("Iris.csv")
print(data.head())
# calculate mean and standard deviation
mean = np.mean(data["SepalLengthCm"])
std = np.std(data["SepalLengthCm"], ddof=1)

# for 99% confidence level

n = len(data)
t_value = t.ppf(0.975, df=n - 1)
margin_of_error = t_value * (std / np.sqrt(n))
ci = (mean - margin_of_error, mean + margin_of_error)
print("Smaple mean: \n", mean)
print("95% of confidence interferance: ", ci)
