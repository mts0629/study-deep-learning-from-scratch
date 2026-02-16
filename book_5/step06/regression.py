import torch
import numpy as np
import matplotlib.pyplot as plt


# Toy dataset
torch.manual_seed(0)
x = torch.rand(100, 1)
y = 2 * x + 5 + torch.rand(100, 1)

# Esimate: y = W * x + b
W = torch.zeros((1, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)

def predict(x):
    y = x @ W + b
    return y

def mean_squared_error(x0, x1):
    diff = x0 - x1
    N = len(diff)
    return torch.sum(diff ** 2) / N

lr = 0.1  # Learning rate
iters = 100  # Iteration

# Regression: minimize MSE
for i in range(iters):
    y_hat = predict(x)
    loss = mean_squared_error(y, y_hat)

    loss.backward()

    W.data -= lr * W.grad.data
    b.data -= lr * b.grad.data

    W.grad.zero_()
    b.grad.zero_()

    if i % 10 == 0:
        print(loss.item())

print(f"loss={loss.item()}")
print("=====")
print(f"W = {W.item()}")
print(f"b = {b.item()}")


# Plot
plt.scatter(x, y)
xs = torch.reshape(torch.arange(0, 1.0, 0.1), (10, 1))
ys = predict(xs)
plt.plot(xs.detach().numpy().flatten(), ys.detach().numpy().flatten(), color="red")
plt.xlabel("x")
plt.ylabel("y")
plt.savefig("./regression.png")
