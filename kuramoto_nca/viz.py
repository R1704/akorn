# kuramoto_nca/viz.py
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from imageio import mimsave
from model import KuramotoCA
import random
import os 
import argparse


# Function to set seeds for reproducibility
def set_seed(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)

class ReadOutConv(nn.Module):
    def __init__(
        self,
        inch,
        outch,
        out_dim,
        kernel_size=1,
        stride=1,
        padding=0,
    ):
        super().__init__()
        self.outch = outch
        self.out_dim = out_dim
        self.invconv = nn.Conv2d(
            inch,
            outch * out_dim,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.bias = nn.Parameter(torch.zeros(outch))

    def forward(self, x):
        x = self.invconv(x).unflatten(1, (self.outch, -1))
        x = torch.linalg.norm(x, dim=2) + self.bias[None, :, None, None]
        return x
# --- Main execution block ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize Kuramoto-NCA evolution.")
    parser.add_argument('--seed', type=int, default=None, help='Random seed for reproducibility.')
    args = parser.parse_args()

    if args.seed is None:
        actual_seed = random.randint(0, 2**32 - 1)
        print(f"No seed provided. Using randomly generated seed: {actual_seed}")
    else:
        actual_seed = args.seed
        print(f"Using provided seed: {actual_seed}")
    
    set_seed(actual_seed)

    CKPT   = "checkpoints/kuramoto_nca_gauss.pth"
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    CHANNELS = 16
    GRID = (72, 72)
    FRAMES, STEP = 500, 20

    def to_rgb(state, camera=None):
        if camera is None:
            rgb = torch.clamp(state[0, :3], 0, 1)
        else:
            rgb = torch.sigmoid(camera(state))[0, :3]
        return rgb.permute(1, 2, 0).detach().cpu().numpy()

    # ---------- load checkpoint dict ----------
    # Ensure checkpoints directory exists if CKPT path implies it.
    # os.makedirs(os.path.dirname(CKPT), exist_ok=True) # If checkpoint might not exist or dir needs creation
    if not os.path.exists(CKPT):
        print(f"Error: Checkpoint file not found at {CKPT}")
        exit(1)
    ckpt = torch.load(CKPT, map_location=DEVICE)
    ca_state  = ckpt["ca"] if "ca" in ckpt else ckpt
    ro_state  = ckpt.get("readout", None)

    # ---------- build KuramotoCA --------------
    # Setting seed before model instantiation is important if strict=False
    # allows for random initialization of any missing keys.
    ca = KuramotoCA(ch=CHANNELS, gamma=0.01).to(DEVICE)
    # The key remapping logic seems specific to your checkpoint version.
    ca.load_state_dict({k.replace("cell.cell.", "cell."): v for k, v in ca_state.items()},
                       strict=False)
    ca.eval()

    # ---------- choose proper read‑out --------
    camera = None
    if ro_state is not None:
        if "invconv.weight" in ro_state:              # ReadOutConv
            out_dim = ro_state["invconv.weight"].shape[0] // 4
            camera  = ReadOutConv(CHANNELS, 4, out_dim).to(DEVICE)
        else:                                         # plain Conv2d
            camera  = torch.nn.Conv2d(CHANNELS, 4, 1, bias=True).to(DEVICE)
        camera.load_state_dict(ro_state, strict=False)
        camera.eval()

    # ---------- seed & rollout ----------------
    H, W = GRID
    # This 'seed' is the initial state, not the random seed.
    initial_state = torch.zeros(1, CHANNELS, H, W, device=DEVICE)
    initial_state[:, 3:, H // 2, W // 2] = 1.0

    frames, x = [], initial_state.clone()
    # Consider adding tqdm here if FRAMES is large
    # from tqdm import tqdm
    # for t in tqdm(range(0, FRAMES * STEP, STEP), desc="Generating frames"):
    for t in range(0, FRAMES * STEP, STEP):
        with torch.no_grad(): # Important for inference
            x, _ = ca(x, STEP, stimulus=None, return_energy=True)
        frames.append(to_rgb(x, camera))

    # ---------- plot --------------------------
    output_filename_prefix = f"seed_{actual_seed}"
    strip_filename = f"kuramoto_strip_{output_filename_prefix}.png"
    gif_filename = f"evolution_{output_filename_prefix}.gif"

    # Ensure output directory exists (e.g., if you want to save to a subfolder)
    os.makedirs("visualizations", exist_ok=True) 
    strip_filename = os.path.join("visualizations", strip_filename)
    gif_filename = os.path.join("visualizations", gif_filename)


    fig, axes = plt.subplots(1, FRAMES, figsize=(2 * FRAMES, 2))
    for ax, img, t_val in zip(axes, frames, range(0, FRAMES * STEP, STEP)):
        ax.imshow(img); ax.set_title(f"t={t_val}"); ax.axis('off')
    plt.tight_layout(); plt.savefig(strip_filename); plt.show()

    mimsave(gif_filename, [(img * 255).astype(np.uint8) for img in frames], fps=5)
    print(f"Saved {strip_filename} & {gif_filename}")
