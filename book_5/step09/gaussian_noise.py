import os
import matplotlib.pyplot as plt

import numpy as np
import torch
import torchvision.transforms as transforms


# Load an image
current_dir = os.path.dirname(os.path.abspath(__file__))
file_path = os.path.join(current_dir, "flower.png")
image = plt.imread(file_path)
print(image.shape)  # (64, 64, 3)

# Preprocess
preprocess = transforms.ToTensor()
x = preprocess(image)

# Reverse from torch.Tensor to image
def reverse_to_img(x):
    x = x * 255
    x = x.clamp(0, 255)
    x = x.to(torch.uint8)
    to_pil = transforms.ToPILImage()
    return to_pil(x)


# Hyperparemeters
T = 1000
beta_start = 0.0001
beta_end = 0.02
betas = torch.linspace(beta_start, beta_end, T)
imgs = []

for t in range(T):
    if t % 100 == 0:
        # Save an image per 100 steps
        img = reverse_to_img(x)
        imgs.append(img)

    # Add noise
    beta = betas[t]
    eps = torch.randn_like(x)
    x = torch.sqrt(1 - beta) * x + torch.sqrt(beta) * eps


# Plot (2x5)
plt.figure(figsize=(15, 6))
for i, img in enumerate(imgs[:10]):
    plt.subplot(2, 5, i + 1)
    plt.imshow(img)
    plt.title(f"Noise: {i * 100}")
    plt.axis("off")
plt.savefig("./gaussian_noise.png")

plt.clf()


# Add gaussian noise at time t to the image
def add_noise(x_0, t, betas):
    T = len(betas)
    assert t >= 1 and t <= T

    alphas = 1 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)  # Cumulative product
    t_idx = t - 1
    alpha_bar = alpha_bars[t_idx]

    eps = torch.randn_like(x_0)
    x_t = torch.sqrt(alpha_bar) * x_0 + torch.sqrt(1 - alpha_bar) * eps

    return x_t


image = plt.imread(file_path)
x = preprocess(image)

t = 100
x_t = add_noise(x, t, betas)

img = reverse_to_img(x_t)
plt.imshow(img)
plt.title(f"Noise: t={t}")
plt.axis("off")
plt.savefig("./noise_t100.png")
