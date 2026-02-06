import numpy as np


# Multivariate normal distribution
def multivariate_normal(x, mu, cov):
    det = np.linalg.det(cov)
    inv = np.linalg.inv(cov)
    D = len(x)
    z = 1 / np.sqrt((2 * np.pi) ** D * det)
    y = z * np.exp((x - mu).T @ inv @ (x - mu) / -2.0)
    return y


x = np.array([[0], [0]])
mu = np.array([[1], [2]])
cov = np.array(
    [[1, 0],
     [0, 1]]
)
y = multivariate_normal(x, mu, cov)
print(y)
