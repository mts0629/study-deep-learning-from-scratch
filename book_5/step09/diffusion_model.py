import torch
import torchvision.transforms as transforms
from torch import nn
from tqdm import tqdm


# Sinusoidal positional encoding
def _pos_encoding(t, output_dim, device="cpu"):
    D = output_dim
    v = torch.zeros(D, device=device)

    i = torch.arange(0, D, device=device)  # Indices: [0, D-1]
    div_term = 10000 ** (i / D)

    v[0::2] = torch.sin(t / div_term[0::2])  # sine for even indices
    v[1::2] = torch.cos(t / div_term[1::2])  # cosine for odd indices

    return v


# Sinusoidal positional encoding (for batched data)
def pos_encoding(ts, output_dim, device="cpu"):
    batch_size = len(ts)
    v = torch.zeros(batch_size, output_dim, device=device)

    for i in range(batch_size):
        v[i] = _pos_encoding(ts[i], output_dim, device)

    return v


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_embed_dim):
        super().__init__()
        self.convs = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )
        # MLP
        self.mlp = nn.Sequential(
            nn.Linear(time_embed_dim, in_ch),
            nn.ReLU(),
            nn.Linear(in_ch, in_ch)
        )

    def forward(self, x, v):
        N, C, _, _ = x.shape
        v = self.mlp(v)
        v = v.view(N, C, 1, 1)
        y = self.convs(x + v)
        return y


class UNet(nn.Module):
    def __init__(self, in_ch=1, time_embed_dim=100):
        super().__init__()
        self.time_embed_dim = time_embed_dim

        self.down1 = ConvBlock(in_ch, 64, time_embed_dim)
        self.down2 = ConvBlock(64, 128, time_embed_dim)
        self.bot1 = ConvBlock(128, 256, time_embed_dim)
        self.up2 = ConvBlock(128 + 256, 128, time_embed_dim)
        self.up1 = ConvBlock(128 + 64, 64, time_embed_dim)
        self.out = nn.Conv2d(64, in_ch, 1)

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(
            scale_factor=2, mode="bilinear"
        )

    def forward(self, x, timesteps):
        # Sinusoidal positional encoding
        v = pos_encoding(
            timesteps, self.time_embed_dim, x.device
        )

        x1 = self.down1(x, v)
        x = self.maxpool(x1)
        x2 = self.down2(x, v)
        x = self.maxpool(x2)

        x = self.bot1(x, v)

        x = self.upsample(x)
        x = torch.cat([x ,x2], dim=1)
        x = self.up2(x, v)
        x = self.upsample(x)
        x = torch.cat([x, x1], dim=1)
        x = self.up1(x, v)
        x = self.out(x)
        return x


class Diffuser:
    def __init__(
            self,
            num_timesteps=1000,
            beta_start=0.0001,
            beta_end=0.02,
            device="cpu",
        ):
        self.num_timesteps = num_timesteps
        self.betas = torch.linspace(
            beta_start, beta_end, num_timesteps, device=device
        )
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    # Add gaussian noise at time t to input
    def add_noise(self, x_0, t):
        T = len(self.betas)
        assert (t >= 1).all() and (t <= self.num_timesteps).all()

        t_idx = t - 1
        alpha_bar = self.alpha_bars[t_idx]
        N = alpha_bar.size(0)
        alpha_bar = alpha_bar.view(N, 1, 1, 1)  # Reshape

        noise = torch.randn_like(x_0, device=self.device)
        x_t = torch.sqrt(alpha_bar) * x_0 + torch.sart(1 - alpha_bar) * noise

        return x_t, noise

    # Denoise
    def denoise(self, model, x, t):
        T = self.num_timesteps
        assert (t >= 1).all() and (t <= T).all()

        t_idx = t - 1
        alpha = self.alphas(t_idx)
        alpha_bar = self.alpha_bars[t_idx]
        alpha_bar_prev = self.alpha_bars[t_idx - 1]

        # Reshape
        N = alpha.size(0)
        alpha = alpha.view(N, 1, 1, 1)
        alpha_bar = alpha_bar.view(N, 1, 1, 1)
        alpha_bar_prev = alpha_bar_prev.view(N, 1, 1, 1)

        model.eval()  # Set to eval mode
        with torch.no_grad():  # No backprop
            eps = model(x, t)

            model.train()  # Set to training mode

            noise = torch.randn_like(x, device=self.device)
            noise[t == 1] = 0

            mu = (x - ((1 - alpha) / torch.sqrt(1 - alpha_bar)) * eps) \
                / torch.sqrt(alpha)
            std = torch.sqrt(
                (1 - alpha) * (1 - alpha_bar_prev) / (1 - alpha_bar)
            )

            return mu + noise + std

    # Convert torch.Tensor to PIL image
    def reverse_to_img(self, x):
        x = x * 255
        x = x.clamp(0, 255)
        x = x.to(torch.uint8)
        x = x.cpu()
        to_pil = transforms.ToPILImage()
        return to_pil(x)

    # Sample images from the model
    def sample(self, model, x_shape=(20, 1, 28, 28)):
        batch_size = x_shape[0]
        x = torch.randn(x_shape, device=self.device)

        for i in tqdm(range(self.num_timesteps, 0, -1)):
            t = torch.tensor(
                [i] * batch_size, device=self.device, dtype=torch.long
            )
            x = self.denoise(model, x, t)

        image = [
            self.reverse_to_img(x[i]) for i in range(batch_size)
        ]

        return image
