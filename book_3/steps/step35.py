if "__file__" in globals():
    import os, sys

    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
from dezero import Variable
import dezero.functions as F
from dezero.utils import plot_dot_graph


# To fifth derivative ... sixth, seventh and eighth derivatives are too large
for iters in range(5):
    x = Variable(np.array(1.0))
    y = F.tanh(x)
    x.name = "x"
    y.name = "y"
    y.backward(create_graph=True)

    # Calculate (iters + 1) derivative
    for i in range(iters):
        gx = x.grad
        x.cleargrad()
        gx.backward(create_graph=True)

    # Plot a graph
    gx = x.grad
    gx.name = "gx" + str(iters + 1)
    plot_dot_graph(
        gx,
        verbose=False,
        to_file=os.path.join(os.path.dirname(__file__), "output", f"{gx.name}.png"),
    )
