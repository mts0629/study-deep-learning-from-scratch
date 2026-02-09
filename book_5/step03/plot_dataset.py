import os
import numpy as np
import matplotlib.pyplot as plt


path = os.path.join(os.path.dirname(__file__), "height_weight.txt")
xs = np.loadtxt(path)
print(xs.shape)


# Scatter plot
small_xs = xs[:500]
plt.scatter(small_xs[:, 0], small_xs[:, 1])
plt.xlabel("Height (cm)")
plt.ylabel("Weight (kg)")
plt.savefig("./height_and_weight.png")

plt.clf()

# Maximum Likelihood Estimation (MLE)
# N = xs.shape[0]
# mu = np.sum(xs, axis=0)
# mu /= N
mu = np.mean(xs, axis=0)

# cov = 0
# for n in range(N):
#     x = xs[n]
#     z = x - mu
#     z = z[:, np.newaxis]
#     cov += z @ z.T
# cov /= N
cov = np.cov(xs, rowvar=False)

print(mu)
print(cov)


# Estimated distribution
def multivariate_normal(x, mu, cov):
    det = np.linalg.det(cov)
    inv = np.linalg.inv(cov)
    D = len(x)
    z = 1 / np.sqrt((2 * np.pi) ** D * det)
    y = z * np.exp((x - mu).T @ inv @ (x - mu) / -2.0)
    return y


# Data points
xs = np.arange(np.floor(mu[0] - cov[0][0]), np.ceil(mu[0] + cov[0][0]), 0.2)
ys = np.arange(np.floor(mu[1] - cov[1][1]), np.ceil(mu[1] + cov[1][1]), 0.2)
X, Y = np.meshgrid(xs, ys)
Z = np.zeros_like(X)
for i in range(X.shape[0]):
    for j in range(X.shape[1]):
        x = np.array([X[i, j], Y[i, j]])
        Z[i, j] = multivariate_normal(x, mu, cov)


# 3-d plot
fig = plt.figure()
ax1 = fig.add_subplot(1, 2, 1, projection="3d")
ax1.set_xlabel("x")
ax1.set_ylabel("y")
ax1.set_zlabel("z")
ax1.plot_surface(X, Y, Z, cmap="viridis")

# Contour & scatter plot
ax2 = fig.add_subplot(1, 2, 2)
ax2.set_xlabel("x")
ax2.set_ylabel("y")
ax2.contour(X, Y, Z)
ax2.scatter(small_xs[:, 0], small_xs[:, 1])

plt.savefig("./mle.png")
