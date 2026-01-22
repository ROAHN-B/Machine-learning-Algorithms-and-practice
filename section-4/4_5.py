# Types if t-ests:
# 1. one-sample T-test: if mean of sample value differ form known vlaue and proportion
# 2. Two- sample T-Test(independent- T-Test): compares mean of two independent groups
# 3. Paired-sample t-test: Compares means of two related groups

# Chi- square test: Test for goodness-of-fit in categorical data
from scipy.stats import f_oneway, chi2_contingency

# data = [[50, 30], [20, 40]]

# # perform chi square test

# chi2, p_value,dof, expected = chi2_contingency(data)

# print("chi 2 values: ", chi2)
# print("P_value: ", p_value)
# print("expacted values: \n", expected)


# ANOVA (Analysis of variance)
# group1 = [12, 6, 32, 87, 62]
# group2 = [11, 33, 22, 44, 66]
# group3 = [85, 65, 12, 45, 32]

# # perform ANOVA
# f_stat, p_value = f_oneway(group1, group2, group3)

# print("F-Statistics: ", f_stat)
# print("p_values: ", p_value)


# Perform one-sample, Two sample and two sample

# 1. one sample t-test
from scipy.stats import ttest_1samp, ttest_ind, ttest_rel

data = [12, 32, 65, 87]
population_mean = 15

t_stat, p_value = ttest_1samp(data, population_mean)
print("one-sample t-test: ", t_stat, p_value)
# 2. Two sample t-test
group1 = [55, 66, 11, 4]
group2 = [88, 16, 4, 5]

t_stat, p_value = ttest_ind(group1, group2)
print("Two sample t-test: ", t_stat, p_value)

# 3. Paired sample t-test
pre_test = [12, 32, 65, 42, 30]
post_test = [15, 54, 98, 65, 20]

t_stat, p_value = ttest_rel(pre_test, post_test)

print("Paired sample test", t_stat, p_value)
