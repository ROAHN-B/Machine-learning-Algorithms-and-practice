# provide staristical and probability analysis of data

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import ttest_ind, chi2_contingency
from sklearn.linear_model import LinearRegression
import numpy as np
# load the dataset
# df = pd.read_csv("tips.csv")
# print(df.info())
# print(df.describe())

# del df["sex"]
# del df["smoker"]
# del df["day"]
# del df["time"]
# del df["Payer Name"]
# del df["Payment ID"]
# # visualize the data
# sns.histplot(df['total_bill'],kde=True)
# plt.title("Distribution of total bill")
# plt.show()

# # correlation heatmap

# sns.heatmap(df.corr(), cmap="coolwarm", annot=True)
# plt.title("Heatmap distribution")
# plt.show()


##############FINDIND CORRELATION BETWEEN DATA################
# df = pd.read_csv("tips.csv")
# print(df.info())
# print(df.describe())

# male_tips=df[df['sex']=='Male'['tip']]
# female_tips = df[df['sex']=='Female'['tip']]

# #Perform the t_test
# t_stat,p_values=ttest_ind(male_tips,female_tips)

# print("T_stats: ", t_stat)
# print("p_values: ", p_values)

# alpha=0.05
# if p_values<=alpha:
#     print("Rejecting the null hypothesis: significant differnce")
# else:
#     print("rejecting the null hypothesis: no significant difference")


#############APPLT LINEAR REGRESSION##################
# df = pd.read_csv("tips.csv")
# print(df.info())
# print(df.describe())

# x=df['total_bill'].values.reshape(-1,1)
# y=df['tip'].values.reshape(-1,1)

# model =LinearRegression()
# model.fit(x,y)

# print("Slope",model.coef_[0])
# print("Intercept", model.intercept_)
# print("r squared: ", model.score(x,y))


# sns.scatterplot(x=df['total_bill'],y=df['tip'],color='blue')
# plt.plot(df['total_bill'],model.predict(x), color='red', label="Regression Line")
# plt.title("Total bill vs Tips")
# plt.legend()
# plt.show()


###################APPLY EXPLORATORY DATA ANALYSIS################

df = pd.read_csv("tips.csv")
print(df.info())
print(df.describe())

del df["sex"]
del df["smoker"]
del df["day"]
del df["time"]
del df["Payer Name"]
del df["Payment ID"]

sns.histplot(df["total_bill"], kde=True)
plt.title("Distribution of total bill")
plt.show()

sns.heatmap(df.corr(), cmap="coolwarm", annot=True)
plt.title("Heatmap")
plt.show()


contingency_table = pd.crosstab(df['smoker'], df['time'])

t_stat, p_value = chi2_contingency(contingency_table)

print("chi2 stat: ", t_stat)
print("p value: ", p_value)

if p_value <= 0.05:
    print("reject the null hypothesis: variables are dependent")
else:
    print("failed to reject the null hypothesis: not significant difference")
