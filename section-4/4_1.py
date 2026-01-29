from scipy.stats import uniform
from itertools import product
import numpy as np
import matplotlib.pyplot as plt

# # Sample space of a dice roll
# sample_space = list(range(1, 7))

# # probablity of rolling an even number
# even_numbers = [2, 4, 6]
# probability_of_even = len(even_numbers) / len(sample_space)
# print("P(Even): \n", probability_of_even)

# Random Variables- maps out random experiment to numerical values
# probability mass function- probalility of a descrete random variable
# probability density function(pdf) - distribution of continuous random variable

# Expectation => E[X] =  weighted average of random variable's possible values (Mean)
# variance = measure of spread of random variable
# standard deviation = width or distribution of data around the center

# outcomes = np.array([2, 3, 5, 7, 9, 5])
# probabilities = np.array([1 / 6] * 6)

# # Expectations
# expectations = np.sum(outcomes * probabilities)
# print("Expectaions (Mean): \n", expectations)

# # variance
# variance = np.sum((outcomes - probabilities) ** 2 * probabilities)
# standard_deviation = np.sqrt(variance)
# print("Variance: \n", variance)
# print("Standard deviation: \n", standard_deviation)


###########Exercise################3
# np.random.seed(2)
# rolls = np.random.randint(1, 7, size=10000)

# # to get an even numbers
# P_even = np.sum(rolls % 2 == 0) / len(rolls)
# print("Probability of even numbers: \n", P_even)

# # to get greater than 4
# p_greater_than_4 = np.sum(rolls > 4) / len(rolls)
# print("Numbers greater than 4: \n", p_greater_than_4)


# random_variables = [2, 3, 5, 6, 7, 8]
# probabilities = [1 / 3] * 6
# plt.bar(random_variables, probabilities, color="purple", alpha=0.7)
# plt.title("PMF of dice roll")
# plt.xlabel("Random_variables")
# plt.ylabel("Probabilities")
# plt.show()


# Continuous random variable: uniform variables
# x = np.linspace(0, 1, 100)
# pdf = uniform.pdf(x, loc=0, scale=1)
# plt.plot(x, pdf, color="red")
# plt.title("PDF of uniform random bariables")
# plt.xlabel("X")
# plt.ylabel("f(x)")
# plt.show()

outcomes = np.random.randint(0, 1, size=10000)
P_head = np.sum((outcomes == 1) / len(outcomes))
P_tails = np.sum((outcomes == 0) / len(outcomes))

print("P[tails]: ", P_tails)
print("P[heads]:", P_head)
