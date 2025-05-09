import torch, math
from torch import nn, Tensor
from kutils import group, ungroup, l2_normalize, gaussian_kernel_2d


class OmegaLayer(nn.Module):
    def __init__(self, n: int, ch: int, init=0.05, global_omega=True, learn=True):
        super().__init__()
        blocks = 1 if global_omega else ch // n
        raw = torch.randn(blocks, n, n) * init / math.sqrt(2)
        self.omg = nn.Parameter(raw - raw.transpose(-2, -1), requires_grad=learn)
        self.register_buffer("max_val", torch.tensor(1.0))   # tighter clamp

    def forward(self, x: Tensor) -> Tensor:
        ω = torch.clamp(self.omg, -self.max_val, self.max_val)
        g = group(x, ω.shape[-1])
        if ω.shape[0] == 1:
            y = torch.einsum("nm,bgnhw->bgmhw", ω[0], g)
        else:
            y = torch.einsum("gnm,bgnhw->bgmhw", ω, g)
        return ungroup(y)


class KuramotoCell(nn.Module):
    def __init__(self, n=2, ch=16, gamma=0.01, ksize=7, sigma=2.0):
        super().__init__()
        self.n, self.gamma = n, gamma
        weight = gaussian_kernel_2d(ksize, sigma).view(1, 1, ksize, ksize)
        self.convJ = nn.Conv2d(ch, ch, ksize, 1, ksize // 2, groups=ch, bias=False)
        with torch.no_grad():
            self.convJ.weight.copy_(weight.repeat(ch, 1, 1, 1))
        self.convJ.weight.requires_grad_(False)
        self.omega = OmegaLayer(n, ch, learn=True)

    @staticmethod
    def _proj(y: Tensor, x: Tensor) -> Tensor:
        return y - (y * x).sum(2, keepdim=True) * x

    def energy(self, x: Tensor) -> Tensor:
        return -(x * self.convJ(x)).reshape(x.shape[0], -1).sum(-1)

    def forward(self, x: Tensor, stimulus: Tensor) -> Tensor:
        y_raw = self.omega(x) + stimulus + self.convJ(x)
        y = self._proj(group(y_raw, self.n), group(x, self.n))
        x = x + self.gamma * ungroup(y)
        x = l2_normalize(x, self.n)
        return torch.nan_to_num(x)                # final NaN guardian
