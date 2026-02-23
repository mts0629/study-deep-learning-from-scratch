import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms


# Encoder
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


# Decoder
class Decoder(nn.Module):
    def __init__(self, latent_dim, hidden_dim, output_dim, use_sigmoid=False):
        super().__init__()
        self.linear1 = nn.Linear(latent_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, output_dim)
        self.use_sigmoid = use_sigmoid

    def forward(self, z):
        h = self.linear1(z)
        h = F.relu(h)
        h = self.linear2(h)
        if self.use_sigmoid:
            h = F.sigmoid(h)
        return h


# Reparameterization trick: enable backpropagation
def reparameterize(mu, sigma):
    eps = torch.randn_like(sigma)
    z = mu + eps * sigma
    return z


# Hierarchical VAE (Variational Auto Encoder)
class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super().__init__()
        # 2 layers
        self.encoder1 = Encoder(input_dim, hidden_dim, latent_dim)
        self.encoder2 = Encoder(latent_dim, hidden_dim, latent_dim)
        self.decoder1 = Decoder(
            latent_dim, hidden_dim, input_dim, use_sigmoid=True
        )
        self.decoder2 = Decoder(latent_dim, hidden_dim, latent_dim)

    def get_loss(self, x):
        mu1, sigma1 = self.encoder1(x)
        z1 = reparameterize(mu1, sigma1)
        mu2, sigma2 = self.encoder2(z1)
        z2 = reparameterize(mu2, sigma2)

        z_hat = self.decoder2(z2)
        x_hat = self.decoder1(z1)

        # Loss function
        batch_size = len(x)
        L1 = F.mse_loss(x_hat, x, reduction="sum")
        L2 = - torch.sum(1 + torch.log(sigma2 ** 2) - mu2 ** 2 - sigma2 ** 2)
        L3 = - torch.sum(1 + torch.log(sigma1 ** 2) - (mu1 - z_hat) ** 2 - sigma1 ** 2)
        return (L1 + L2 + L3) / batch_size


# Hyperparameters
input_dim = 784  # Input size (28x28)
hidden_dim = 100
latent_dim = 20

# VAE model
model = VAE(input_dim, hidden_dim, latent_dim)

# Path to save the model
model_path = "./hvae.pth"

do_training = True
if do_training:
    # Hyperparameters
    epochs = 30
    learning_rate = 1e-4
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

    # Save a state_dict
    torch.save(model.state_dict(), model_path)


# Load the trained model
model.load_state_dict(torch.load(model_path))
model.eval()

# Generate new images
with torch.no_grad():
    # Generate 64 latent variables
    sample_size = 64
    z2 = torch.randn(sample_size, latent_dim)
    z1_hat = model.decoder2(z2)
    # Generate images from the latent variables
    z1 = reparameterize(z1_hat, torch.ones_like(z1_hat))
    x = model.decoder1(z1)
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
