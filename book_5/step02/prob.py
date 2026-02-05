import os
import numpy as np
from scipy.stats import norm

path = os.path.join(os.path.dirname(__file__), "height.txt")
xs = np.loadtxt(path)

mu = np.mean(xs)
sigma = np.std(xs)

# Calculate cumulative distribution function (CDF)
# F(x) = p(X <= x)
p1 = norm.cdf(160, mu, sigma)
print(f"p(x <= 160) = {p1}")

# p(X > x) = 1 - F(x)
p2 = norm.cdf(180, mu, sigma)
print(f"p(x > 180) = {1 - p2}")

# p(a < X <= b) = F(b) - F(a) (a <= b)
print(f"p(160 < x <= 180) = {p2 - p1}")
