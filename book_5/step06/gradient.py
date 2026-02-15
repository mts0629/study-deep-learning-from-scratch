import torch


# Rosenblock function
def rosenblock(x0, x1):
    y = 100 * (x1 - x0 ** 2) ** 2 + (x0 - 1) ** 2
    return y


# Initial value
x0 = torch.tensor(0.0, requires_grad=True)
x1 = torch.tensor(2.0, requires_grad=True)

# Search the minimum of the function by the gradient method
lr = 0.001  # Learning rate
iters = 10000  # Iterations
for i in range(iters):
    if i % 1000 == 0:
        print(x0.item(), x1.item())

    y = rosenblock(x0, x1)

    y.backward()

    # Update values
    x0.data -= lr * x0.grad.data
    x1.data -= lr * x1.grad.data

    # Reset gradients
    x0.grad.zero_()
    x1.grad.zero_()

print(f"x0={x0.item()}, x1={x1.item()}")
