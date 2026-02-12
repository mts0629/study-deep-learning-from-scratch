import numpy as np


# KL (Kullback-Leibler) divergence
def kl_divergence(ps, qs):
    d = 0
    for p, q in zip(ps, qs):
        d += p * np.log(p / q)
    return d


ps = [0.7, 0.3]
qs = [0.5, 0.5]
print(kl_divergence(ps, qs))

qs = [0.2, 0.8]
print(kl_divergence(ps, qs))

qs = [0.7, 0.3]
print(kl_divergence(ps, qs))
