import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


# Preprocess: transform to tensors
transform = transforms.ToTensor()

# Download MNIST dataset
dataset = torchvision.datasets.MNIST(
    root="./data",
    train=True,
    transform=transform,
    download=True,
)

x, label = dataset[0]
print(f"size: {len(dataset)}")  # 60000
print(f"type: {type(x)}")
print(f"shape: {x.shape}")
print(f"label: {label}")  # 5

# plt.imshow(x, cmap="gray")
# plt.savefig("./test.png")

# Data loader
dataloader = torch.utils.data.DataLoader(
    dataset, batch_size=32, shuffle=True
)

for x, label in dataloader:
    # See mini-batch
    print(f"x shape: {x.shape}")
    print(f"label shape: {label.shape}")
    break
