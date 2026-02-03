import numpy as np
import matplotlib.pyplot as plt

NS = [1, 2, 4, 10]
x_means = {}

for N in NS:  # Sample sizes
    x_means[N] = []
    for _ in range(10000):  # Sampling iterations
        xs = []
        for n in range(N):
            x = np.random.rand()  # From uniform distribution
            xs.append(x)

        mean = np.mean(xs)
        x_means[N].append(mean)

# Draw histograms
for i, N in enumerate(NS):
    plt.subplot(2, 2, (i + 1))
    plt.hist(x_means[N], bins="auto", density=True)
    plt.title(f"N={N}")
    plt.xlabel("x")
    plt.ylabel("Probability Density")
    plt.ylim(-0.05, 1.05)
    plt.ylim(0, 5)

plt.tight_layout()
plt.savefig("./sample_avg.png")
