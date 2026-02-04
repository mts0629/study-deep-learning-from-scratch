import os
import numpy as np
import matplotlib.pyplot as plt


path = os.path.join(os.path.dirname(__file__), "height.txt")
xs = np.loadtxt(path)
print(xs.shape)  # 25000

# Histogram of heights
plt.hist(xs, bins="auto", density=True)
plt.xlabel("Height (cm)")
plt.ylabel("Probability Density")
plt.savefig("./hist.png")


# Parameters for normal distribution model
mu = np.mean(xs)
sigma = np.std(xs)
print(f"mean = {mu}")
print(f"std dev = {sigma}")


def normal(x, mu=0, sigma=1):
    y = 1 / (np.sqrt(2 * np.pi) * sigma) * np.exp(-(x - mu)**2 / (2 * sigma**2))
    return y


# Fit a normal distribution model
x = np.linspace(150, 190, 1000)
y = normal(x, mu, sigma)
plt.plot(x, y)
plt.savefig("./fit.png")
