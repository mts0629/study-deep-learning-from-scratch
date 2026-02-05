import os
import numpy as np
import matplotlib.pyplot as plt


path = os.path.join(os.path.dirname(__file__), "height.txt")
xs = np.loadtxt(path)

mu = np.mean(xs)
sigma = np.std(xs)

# Generate new 10000 samples
samples = np.random.normal(mu, sigma, 10000)

# Original samples
plt.hist(xs, bins="auto", density=True, alpha=0.7, label="original")
# Generated samples
plt.hist(samples, bins="auto", density=True, alpha=0.7, label="generated")
plt.xlabel("Height (cm)")
plt.ylabel("Probaility Density")
plt.legend()
plt.savefig("generated_samples.png")
