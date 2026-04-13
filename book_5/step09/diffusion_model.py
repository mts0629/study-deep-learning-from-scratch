import torch


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


if __name__ == "__main__":
    v = pos_encoding([1, 2, 3], 16)
    print(v.shape)
