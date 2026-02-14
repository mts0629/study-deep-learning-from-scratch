import os
import numpy as np
import matplotlib.pyplot as plt


path = os.path.join(os.path.dirname(__file__), "old_faithful.txt")
xs = np.loadtxt(path)
# print(xs.shape)

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
    inv = np.linalg.inv(cov)
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


current_likelihood = likelihood(xs, phis, mus, covs)

# EM (Expectation-Maximization) algorithm
for iter in range(MAX_ITERS):
    # E-step: estimate probability distribution
    qs = np.zeros((N, K))
    for n in range(N):
        x = xs[n]
        for k in range(K):
            phi, mu, cov = phis[k], mus[k], covs[k]
            qs[n, k] = phi * multivariate_normal(x, mu, cov)
        qs[n] /= gmm(x, phis, mus, covs)

    # M-step: maximize ELBO (Evidence Lower BOund)
    qs_sum = qs.sum(axis=0)
    for k in range(K):
        # Phis
        phis[k] = qs_sum[k] / N

        # Mus
        c = 0
        for n in range(N):
            c += qs[n, k]  * xs[n]
        mus[k] = c / qs_sum[k]

        # Covs
        c = 0
        for n in range(N):
            z = xs[n] - mus[k]
            z = z[:, np.newaxis]  # Align to column vector
            c += qs[n, k] * z @ z.T
        covs[k] = c / qs_sum[k]

    # Check termination condition
    print(f"likelihood={current_likelihood:.3f}")
    next_likelihood = likelihood(xs, phis, mus, covs)
    diff = np.abs(next_likelihood - current_likelihood)
    if diff < THRESHOLD:
        break
    current_likelihood = next_likelihood


def sample(phis, mus, covs):
    k = np.random.choice(2, p=phis)
    mu, cov = mus[k], covs[k]
    x = np.random.multivariate_normal(mu, cov)
    return x


# Plot
N = 500
new_xs = np.zeros((N, 2))
for n in range(N):
    new_xs[n] = sample(phis, mus, covs)

plt.scatter(xs[:,0], xs[:,1], label="Original")
plt.scatter(new_xs[:,0], new_xs[:,1], label="Generated", color="orange", alpha=0.7)
plt.xlabel("Eruptions(min)")
plt.ylabel("Waiting(min)")
plt.legend()
plt.savefig("em_plot.png")
