import numpy as np
import matplotlib.pyplot as plt


X = np.array(
    [[-2, -1, 0, 1, 2],
     [-2, -1, 0, 1, 2],
     [-2, -1, 0, 1, 2],
     [-2, -1, 0, 1, 2],
     [-2, -1, 0, 1, 2]]
)
Y = np.array(
    [[-2, -2, -2, -2, -2],
     [-1, -1, -1, -1, -1],
     [0, 0, 0, 0, 0],
     [1, 1, 1, 1, 1],
     [2, 2, 2, 2, 2]]
)
Z = X ** 2 + Y ** 2


ax = plt.axes(projection="3d")
ax.plot_surface(X, Y, Z, cmap="viridis")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
plt.savefig("plot_3d.png")

plt.cla()


xs = np.arange(-2, 2, 0.1)
ys = np.arange(-2, 2, 0.1)
X, Y = np.meshgrid(xs, ys)
Z = X ** 2 + Y ** 2

ax = plt.axes(projection="3d")
ax.plot_surface(X, Y, Z, cmap="viridis")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
plt.savefig("plot_3d_mesh.png")

plt.cla()


x = np.arange(-2, 2, 0.1)
y = np.arange(-2, 2, 0.1)

X, Y = np.meshgrid(x, y)
Z = X ** 2 + Y ** 2

ax = plt.axes()
ax.contour(X, Y, Z)
ax.set_xlabel("x")
ax.set_ylabel("y")
plt.savefig("contour.png")
