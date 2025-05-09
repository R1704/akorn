# kuramoto_nca/train.py

import io, os, requests
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import Resize, ToTensor
from tqdm import tqdm

from model import KuramotoCA

# ─── hyper-parameters ─────────────────────────────────────────────
EMOJI      = "🦎"
SIZE       = 40
PAD        = 16
CHANNELS   = 16
POOL       = 1024
BATCH      = 8
STEPS_MIN, STEPS_MAX = 32, 48
ITERATIONS = 10000
LR         = 2e-3
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"

# ─── data utils ────────────────────────────────────────────────────
def load_emoji(ch: str, size: int) -> torch.Tensor:
    url = (
        f"https://github.com/googlefonts/noto-emoji/blob/"
        f"main/png/128/emoji_u{ord(ch):x}.png?raw=true"
    )
    r = requests.get(url, timeout=10)
    r.raise_for_status()
    img = Image.open(io.BytesIO(r.content)).convert("RGBA")
    img = Resize(size, interpolation=Image.LANCZOS)(img)
    t = ToTensor()(img)      # (4,H,W)
    t[:3] *= t[3:]           # premultiply RGB by A
    return t

# ─── prepare target ─────────────────────────────────────────────────
target = load_emoji(EMOJI, SIZE)                # (4,h,w)
target = F.pad(target, (PAD, PAD, PAD, PAD))    # (4,H,W)
_, H, W = target.shape
target = target.unsqueeze(0).to(DEVICE)         # (1,4,H,W)

# save a preview of what we’re aiming at
Image.fromarray((target[0].permute(1,2,0).cpu().numpy()*255).astype(np.uint8))\
     .save("emoji_preview.png")

# ─── build model & readout ──────────────────────────────────────────
ca      = KuramotoCA(ch=CHANNELS, gamma=0.01).to(DEVICE)
# allow J to train immediately:
ca.cell.convJ.weight.requires_grad_(True)

class ReadOutConv(nn.Module):
    def __init__(self, inch, outch, out_dim, kernel_size=1):
        super().__init__()
        self.invconv = nn.Conv2d(inch, outch*out_dim, kernel_size, bias=False)
        self.bias    = nn.Parameter(torch.zeros(outch))

    def forward(self, x):
        x = self.invconv(x)                             # (B,4*out_dim,H,W)
        x = x.unflatten(1, (self.bias.numel(), -1))     # (B,4,out_dim,H,W)
        x = torch.linalg.norm(x, dim=2)                 # (B,4,H,W)
        return x + self.bias.view(1,-1,1,1)

readout = ReadOutConv(CHANNELS, 4, out_dim=2).to(DEVICE)

opt = torch.optim.Adam([
    {"params": ca.cell.omega.parameters(), "lr": LR},
    {"params": ca.cell.convJ.parameters(),  "lr": LR*0.5},
    {"params": readout.parameters(),        "lr": LR},
])

# ─── initialize pool ────────────────────────────────────────────────
seed = torch.randn(1, CHANNELS, H, W, device=DEVICE)
seed = seed / (seed.pow(2).sum(1, keepdim=True).sqrt() + 1e-8)
seed[:,3:,H//2,W//2] = 1.0
seed[:,:3] *= 0.5
pool = seed.repeat(POOL,1,1,1)

# ─── training loop ──────────────────────────────────────────────────
with tqdm(range(ITERATIONS), desc="Training") as pbar:
    for it in pbar:
        idx   = torch.randint(0, POOL, (BATCH,), device=DEVICE)
        x0    = pool[idx]
        steps = torch.randint(STEPS_MIN, STEPS_MAX, (1,), device=DEVICE).item()

        # 1) physics
        x, E = ca(x0, steps, return_energy=True)   # (B,C,H,W)

        # 2) read-out + masked RGBA loss
        rgba     = readout(x)                      # (B,4,H,W)
        pred     = torch.sigmoid(rgba)             # [0,1]
        B        = pred.size(0)
        alpha_gt = target[:,3:4].expand(B,-1,-1,-1) # (B,1,H,W)
        rgb_gt   = target[:,:3].expand(B,-1,-1,-1)  # (B,3,H,W)

        # supervise alpha everywhere
        loss_a   = F.mse_loss(pred[:,3:4], alpha_gt)
        # supervise RGB only where alpha_gt>0
        mask     = (alpha_gt > 0).float()
        loss_rgb = F.mse_loss(pred[:,:3]*mask, rgb_gt*mask)
        loss     = loss_a + 3.0*loss_rgb

        # 3) backward & step
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(ca.parameters(), 0.5)
        opt.step()

        pool[idx] = x.detach()

        # log
        if it % 200 == 0:
            pbar.set_postfix(loss=f"{loss.item():.4e}",
                             α=f"{loss_a.item():.4e}",
                             rgb=f"{loss_rgb.item():.4e}",
                             energy=f"{E.mean().item():.1f}")

# ─── save checkpoint ────────────────────────────────────────────────
os.makedirs("checkpoints", exist_ok=True)
torch.save({"ca": ca.state_dict(),
            "readout": readout.state_dict()},
           "checkpoints/kuramoto_nca_gauss.pth")
print("Training finished.")
