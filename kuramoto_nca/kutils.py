import torch
import numpy as np
from torch import Tensor
import math

# ------------------------------------------------------------------ reshape
def group(x: Tensor, n: int) -> Tensor:               # (B,C,H,W)→(B,G,n,H,W)
    B, C, H, W = x.shape
    assert C % n == 0, "channel dim not divisible by n"
    return x.contiguous().view(B, C // n, n, H, W)


def ungroup(g: Tensor) -> Tensor:                     # (B,G,n,H,W)→(B,C,H,W)
    B, G, n, H, W = g.shape
    return g.contiguous().view(B, G * n, H, W)

# ---------------------------------------------------------------- normalize
def l2_normalize(x: Tensor, n: int, eps: float = 1e-8) -> Tensor:
    g = group(x, n)
    mag = torch.linalg.norm(g, dim=2, keepdim=True).clamp_min(eps)
    g = g / mag
    return ungroup(g)

# --------------------------------------------------------------- gauss kern
def gaussian_kernel_2d(ksize: int, sigma: float) -> Tensor:
    ax = np.arange(ksize, dtype=np.float32) - (ksize - 1) / 2.0
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()
    return torch.from_numpy(kernel)                   # (ksize,ksize)
