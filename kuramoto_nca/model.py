# kuramoto_nca/model.py
import torch
from torch import nn, Tensor
from cell import KuramotoCell


class KuramotoCA(nn.Module):
    """
    Kuramoto‑powered cellular automaton grid with an optional external
    stimulus field.  If no stimulus is supplied we use a detached copy of
    x (equivalent to the original self‑coupling).
    """

    def __init__(self, n: int = 2, ch: int = 16, gamma: float = 0.01,
                 fire_rate: float = 0.5, micro_T: int = 1):
        super().__init__()
        self.cell = KuramotoCell(n, ch, gamma)
        self.fire_rate, self.micro_T = fire_rate, micro_T

    # ------------------------ helpers -------------------------
    @staticmethod
    def _alive(x: Tensor, idx: int = 3) -> Tensor:
        return torch.nn.functional.max_pool2d(
            x[:, idx:idx + 1], 3, 1, 1) > 0.1

    # ------------------------ forward -------------------------
    def forward(self, x: Tensor, steps: int,
                stimulus: Tensor | None = None,
                return_energy: bool = False):
        energies = []
        for _ in range(steps):
            pre = self._alive(x)

            for _ in range(self.micro_T):
                mask = (torch.rand_like(x[:, :1]) < self.fire_rate).float()

                stim = stimulus if stimulus is not None else x.detach()
                dx = self.cell(x, stimulus=stim) - x
                x = x + dx * mask

            x = x * (pre & self._alive(x))

            if return_energy:
                energies.append(self.cell.energy(x))

            if torch.isnan(x).any():
                raise RuntimeError("NaNs detected – reduce γ or clamp Ω.")

        if return_energy:
            return x, torch.stack(energies, 1)          # (B,steps)
        return x
