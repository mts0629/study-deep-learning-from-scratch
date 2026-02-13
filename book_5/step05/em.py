import os
import numpy as np


path = os.path.join(os.path.dirname(__file__), "old_faithful.txt")
xs = np.loadtxt(path)
print(xs.shape)

# Initial parameters
phis = np.array([0.5, 0.5])
mus = np.array(
    [[0.0, 50.0],
     [0.0, 100.0]]
)
covs = np.array([np.eye(2), np.eye(2)])  # Identity matrix

K = len(phis)
N = len(xs)
MAX_ITERS = 100
THRESHOLD = 1e-4


def multivariate_normal(x, mu, cov):
    det = np.linalg.det(cov)
    inv = np.linalg.int(cov)
    d = len(x)
    z = 1 / np.sqrt((2 * np.pi) ** d * det)
    y = z * np.exp((x - mu).T @ inv @ (x - mu) / -2.0)
    return y


def gmm(x, phis, mus, covs):
    K = len(phis)
    y = 0
    for k in range(K):
        phi, mu, cov = phis[k], mus[k], covs[k]
        y += phi * multivariate_normal(x, mu, cov)
    return y


def likelihood(xs, phis, mus, covs):
    eps = 1e-6  # Avoid log(0)
    L = 0
    N = len(xs)
    for x in xs:
        y = gmm(x, phis, mus, covs)
        L += np.log(y + eps)
    return L / N
