import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt


# Network model 
class Model(nn.Module):
    def __init__(self, input_size=1, hidden_size=10, output_size=1):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        y = self.linear1(x)
        y = F.sigmoid(y)
        y = self.linear2(y)
        return y


# Dataset
torch.manual_seed(0)
x = torch.rand(100, 1)
y = torch.sin(2 * torch.pi * x) + torch.rand(100, 1)


lr = 0.2
iters = 10000

model = Model()
optimizer = torch.optim.SGD(model.parameters(), lr=lr)  # SGD (Stochastic Gradient Descent)

# Backprop
for i in range(iters):
    y_pred = model(x)
    loss = nn.functional.mse_loss(y, y_pred)

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()

    if i % 1000 == 0:
        print(loss.item())

print(loss.item())


# Plot
plt.scatter(x, y)
xs = torch.reshape(torch.arange(0, 1.0, 0.01), (100, 1))
ys = model(xs)
plt.plot(xs.detach().numpy().flatten(), ys.detach().numpy().flatten(), color="red")
plt.xlabel("x")
plt.ylabel("y")
plt.savefig("./predict.png")
