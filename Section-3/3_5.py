import numpy as np
import matplotlib.pyplot as plt

# mu, sigma = 0, 1
# x = np.linspace(-5, 5, 100)
# y = (1 / (np.sqrt(2 * np.pi) * sigma**2)) * np.exp(-0.5 * ((x - mu) / sigma) ** 2)

# plt.plot(x, y)
# plt.title("Normal Distribution:")
# plt.show()


# p = 0.6
# plt.bar([0, 1], [1 - p, p])
# plt.title("Bernoulli Distribution: p=0.6")
# plt.xticks([0, 1], labels=["0 Failure", "1 Success"])
# plt.show()

from scipy.stats import binom, poisson

# n, p = 10, 0.5
# x = np.arange(0, n + 1)
# y = binom.pmf(x, n, p)
# plt.bar(x, y)
# plt.title("Binomial Distribution: n=10, p=0.5")
# plt.xlabel("Number of Successes")
# plt.ylabel("Probability")
# plt.show()

# lam = 3
# x = np.arange(0, 15)
# y = poisson.pmf(x, lam)
# plt.bar(x, y)
# plt.title("Poisson Distribution: λ=3")
# plt.xlabel("Number of Events")
# plt.ylabel("Probability")
# plt.show()


#######EXERCISE#############
def bayes_theorem(prior, sensitivity, specificity):
    evidence = (sensitivity * prior) + (1 - specificity) * (1 - prior)
    posterior = (sensitivity * prior) / evidence
    return posterior


prior = 0.01
sensitivity = 0.99
specificity = 0.95
posterior = bayes_theorem(prior, sensitivity, specificity)
print(f"Posterior Probability: {posterior:.4f}")
