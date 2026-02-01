import numpy as np

def normal(x, mu=0, sigma=1):
    y = 1 / (np.sqrt(2 * np.pi) * sigma) * np.exp(-(x - mu)**2 / 2 * sigma**2)
    return y

import matplotlib.pyplot as plt

x = np.linspace(-5, 5, 100)
y = normal(x)

plt.plot(x, y)
plt.xlabel("x")
plt.ylabel("y")
plt.savefig("./normal.png")
