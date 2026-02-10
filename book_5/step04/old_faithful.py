import os
import numpy as np
import matplotlib.pyplot as plt


path = os.path.join(os.path.dirname(__file__), "old_faithful.txt")
xs = np.loadtxt(path)

print(xs.shape)
print(xs[0])

plt.scatter(xs[:,0], xs[:,1])
plt.xlabel("Eruptions(min)")
plt.ylabel("Waiting(min)")
plt.savefig("./old_faithful.png")
