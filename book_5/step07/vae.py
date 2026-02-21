import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms


# Encoder: convert input to latent space
class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, hidden_dim)
        self.linear_mu = nn.Linear(hidden_dim, latent_dim)
        self.linear_logvar = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        h = self.linear(x)
        h = F.relu(h)
        mu = self.linear_mu(h)
        logvar = self.linear_logvar(h)
        sigma = torch.exp(0.5 * logvar)
        return mu, sigma


# Decoder: reproduce input from latent space
class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim):
        super().__init__()
        self.linear1 = nn.Linear(latent_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, z):
        h = self.linear1(z)
        h = F.relu(h)
        h = self.linear2(h)
        x_hat = F.sigmoid(h)
        return x_hat


# Reparameterization trick: enable backpropagation
def reparameterize(mu, sigma):
    eps = torch.randn_like(sigma)
    z = mu + eps * sigma
    return z


# VAE (Variational Auto Encoder)
class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dim, input_dim)

    def get_loss(self, x):
        mu, sigma = self.encoder(x)
        z = reparameterize(mu, sigma)
        x_hat = self.decoder(z)

        # Loss function
        batch_size = len(x)
        L1 = F.mse_loss(x_hat, x, reduction="sum")
        L2 = - torch.sum(1 + torch.log(sigma ** 2) - mu ** 2 - sigma ** 2)
        return (L1 + L2) / batch_size


# Hyperparameters
input_dim = 784  # Input size (28x28)
hidden_dim = 200  # Dimension of hidden layer
latent_dim = 20  # Dimension of latent vector
epochs = 30
learning_rate = 3e-4
batch_size = 32


# MNIST dataset and dataloader
transform = transforms.Compose([
    transforms.ToTensor(),  # Normalize
    transforms.Lambda(torch.flatten)  # (28,28) -> (784)
])
dataset = datasets.MNIST(
    root="./data",
    train=True,
    download=True,
    transform=transform
)
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True
)

# VAE model
model = VAE(input_dim, hidden_dim, latent_dim)
# Optimizer: Adam
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Learning
losses = []
for epoch in range(epochs):
    loss_sum = 0.0
    cnt = 0

    for x, label in dataloader:
        optimizer.zero_grad()
        loss = model.get_loss(x)
        loss.backward()
        optimizer.step()

        loss_sum += loss.item()
        cnt += 1

    loss_avg = loss_sum / cnt
    losses.append(loss_avg)
    print(loss_avg)


# Plot loss
plt.plot([i for i in range(epochs)], losses)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.savefig("./loss.png")


# Generate new images
with torch.no_grad():
    # Generate 64 latent variables
    sample_size = 64
    z = torch.randn(sample_size, latent_dim)
    # Generate images from the latent variables
    x = model.decoder(z)
    generated_images = x.view(sample_size, 1, 28, 28)  # Reshape to (28,28)

# Sort images to 8x8 grid
grid_img = torchvision.utils.make_grid(
    generated_images,
    nrow=8,
    padding=2,
    normalize=True
)
# Draw generated images
plt.cla()
plt.imshow(grid_img.permute(1, 2, 0))
plt.axis("off")
plt.savefig("./generated_images.png")
