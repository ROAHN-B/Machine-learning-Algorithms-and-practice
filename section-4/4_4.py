## HYPOTHESIS - Statistical method to determine if there are enough evidence in a sample to infer a conclusion about the population
# key components - NUll hypothesis - no effect or no difference
# Alternative Hypothesis  - Indicates an effect or difference

# Hypothesis test
# import numpy as np
# from scipy.stats import ttest_1samp

# data = [12, 13, 14, 15, 16, 17, 18, 19]

# # Null hypothesis: mean = 15
# population_mean = 15

# # Perform T-Test
# t_stat, p_value = ttest_1samp(data, population_mean)
# print("T-statistic: ", t_stat)
# print("P-value: ", p_value)

# # results
# alpha = 0.05
# if p_value <= alpha:
#     print("Reject the null hypothesis: significant difference")
# else:
#     print("Failed to reject Null hypothesis: no significant difference")


##### COMPARE TEST SCORE OF TWO CLASSES##########
import numpy as np
from scipy.stats import ttest_ind

group1 = [10, 20, 30, 40, 50, 60]
group2 = [11, 22, 33, 44, 55, 66]

# perform t-test :
t_stat, p_val = ttest_ind(group2, group2)

print("T-statistic: ", t_stat)
print("p_values: ", p_val)

alpha = 0.05

if t_stat <= alpha:
    print("Reject the null hypothesis: significant difference")
else:
    print("Fail to reject the null hypothseis: not any significant changes")
