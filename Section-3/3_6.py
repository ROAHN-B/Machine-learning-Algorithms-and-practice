# from statistics import mode
# from scipy.stats import stats, ttest_ind

# data = [1, 2, 2, 2, 3, 3, 3, 6, 6, 5, 44]
# mean = sum(data) / len(data)
# print("Mean of the data: \n", mean)

# print("Print mode of data: \n", mode(data))

# variance = sum((x - mean) ** 2 for x in data) / len(data)
# print("Variance: \n", variance)
# std = variance**0.5
# print("standard deviation: \n", std)


# HYPOTHESIS TESTING
# confidence intervals - tells where most of the population lies

# data = [1, 2, 2, 2, 3, 3, 3, 6, 6, 5, 44]
# mean = sum(data) / len(data)
# variance = sum((x - mean) ** 2 for x in data) / len(data)
# print("Variance: \n", variance)
# std = variance**0.5
# sample_mean = mean
# z_score = 1.96

# ci = (
#     sample_mean - z_score * (std / len(data) ** 0.5),
#     sample_mean - z_score * (std / len(data) ** 0.5),
# )

# print("95% of confidence: \n", ci)


# Exercise  - perform T-test to perform sample data test

# group_1 = [2.1, 3.2, 5.0, 4.3, 6.1]
# group_2 = [0.6, 1.0, 2.3, 2.5, 4.0]

# t_stat, p_value = ttest_ind(group_1, group_2)
# print("T-statistics: \n", t_stat)
# print("P-Value: \n", p_value)


# Interpretation
# alpha = 0.05
# if p_value < alpha:
#     print("reject the null hypothesis: significant difference")
# else:
#     print("Failed to reject the null hypothesis: no significance difference")
