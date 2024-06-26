if "__file__" in globals():
    import os, sys

    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import matplotlib.pyplot as plt
from dezero import Variable
import dezero.functions as F


x = Variable(np.linspace(-7, 7, 200))  # x = [-7, 7] with 200 samples
y = F.sin(x)
y.backward(create_graph=True)

logs = [y.data.flatten()]

# Calculate second, third and forth derivatives
for i in range(3):
    logs.append(x.grad.data.flatten())
    gx = x.grad
    x.cleargrad()
    gx.backward(create_graph=True)

# Plot y and its derivatives
labels = ["y=sin(x)", "y'", "y''", "y'''"]
for i, v in enumerate(logs):
    plt.plot(x.data, logs[i], label=labels[i])
plt.legend(loc="lower right")
plt.show()
plt.savefig(os.path.join(os.path.dirname(__file__), "output", "sin_plot.png"))
