import argparse
import copy
import io
import json
import os
import time

import numpy as np
import PIL.Image
import requests
import torch
import torch.nn as nn
from kuramoto_nca.akorn.akorn_nca_model import \
    AKOrN_NCA  # Assuming akorn_nca_model.py is in the same directory
from torch.optim.lr_scheduler import MultiStepLR


def get_target_image(image_url='https://greydanus.github.io/files/lizard_1f98e.png', target_size=(40,40), padding=16):
    r = requests.get(image_url)
    img = PIL.Image.open(io.BytesIO(r.content))
    img.thumbnail(target_size, PIL.Image.Resampling.LANCZOS) # Updated for Pillow 9+
    img_np = np.float32(img)/255.0
    img_np = img_np[...,:4] # Ensure it's RGBA, discard other channels if any
    
    # Premultiply RGB by Alpha
    if img_np.shape[-1] == 4:
        img_np[..., :3] *= img_np[..., 3:4]
        
    img_np = img_np.transpose(2,0,1)[None,...] # [N, C, H, W]
    
    if padding > 0:
        img_np = np.pad(img_np, ((0,0),(0,0),(padding,padding),(padding,padding)))
    return torch.from_numpy(img_np)

def make_circle_masks(n, h, w, device='cpu'):
    x = torch.linspace(-1.0, 1.0, w, device=device)[None, None, :]
    y = torch.linspace(-1.0, 1.0, h, device=device)[None, :, None]
    center = torch.rand(2, n, 1, 1, device=device)-.5
    r = 0.3 * torch.rand(n, 1, 1, device=device) + 0.1
    x, y = (x-center[0])/r, (y-center[1])/r
    return (1-(x*x+y*y < 1.0).float()).unsqueeze(1) # mask is OFF in circle, add channel dim

def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class ObjectView(object):
    def __init__(self, d): self.__dict__ = d

def get_args():
    parser = argparse.ArgumentParser(description="Train AKOrN-NCA model.")
    parser.add_argument('--image_url', type=str, default='https://greydanus.github.io/files/lizard_1f98e.png', help='URL of the target image.')
    parser.add_argument('--target_size_h', type=int, default=40, help='Target height for image before padding.')
    parser.add_argument('--target_size_w', type=int, default=40, help='Target width for image before padding.')
    parser.add_argument('--padding', type=int, default=16, help='Padding around the target image.')
    
    parser.add_argument('--nca_state_dim', type=int, default=16, help='Number of channels in the NCA state (e.g., RGBA + latent).')
    parser.add_argument('--n_osc_components', type=int, default=2, help='Number of components per Kuramoto oscillator.')
    parser.add_argument('--num_klayers', type=int, default=1, help='Number of KLayers per NCA update step.')
    parser.add_argument('--k_layer_config', type=str, default='{"J": "conv", "ksize": 3, "use_omega": true}', help='JSON string for KLayer config (dict or list of dicts).')
    parser.add_argument('--k_layer_gamma', type=float, default=0.1, help='Step size gamma for KLayer internal dynamics.')
    parser.add_argument('--rgba_readout_out_dim', type=int, default=2, help='Output dimension for the RGBA ReadOutConv.')
    parser.add_argument('--use_conditioning_readout', type=lambda x: (str(x).lower() == 'true'), default=False, help='Whether to use ReadOutConv for KLayer conditioning.')
    parser.add_argument('--cond_readout_out_dim', type=int, default=1, help='Output dimension for the conditioning ReadOutConv.')

    # Training Args
    parser.add_argument('--num_nca_steps_min', type=int, default=64, help='Min number of NCA steps per training iteration.')
    parser.add_argument('--num_nca_steps_max', type=int, default=96, help='Max number of NCA steps per training iteration.')
    parser.add_argument('--k_layer_steps_per_nca_step', type=int, default=1, help='Number of KLayer internal steps (T) per NCA step.')
    parser.add_argument('--pool_size', type=int, default=1024, help='Size of the persistent CA pool.')
    parser.add_argument('--perturb_n', type=int, default=0, help='Number of CAs in batch to perturb.')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size for training.')
    parser.add_argument('--learning_rate', type=float, default=2e-3, help='Learning rate.')
    parser.add_argument('--milestones', type=str, default="[2000, 3000]", help='JSON list for LR scheduler milestones.')
    parser.add_argument('--gamma', type=float, default=0.2, help='LR scheduler gamma.')
    parser.add_argument('--total_steps', type=int, default=4000, help='Total training steps.')
    parser.add_argument('--checkpoint_path', type=str, default='checkpoints/akorn_nca.pth', help='Path to save checkpoints.')
    parser.add_argument('--save_every', type=int, default=500, help='Frequency of saving checkpoints.')
    parser.add_argument('--log_every', type=int, default=100, help='Frequency of logging training progress.')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use for training.')
    parser.add_argument('--seed', type=int, default=42, help='Random seed.')
    parser.add_argument('--diag_image_dir', type=str, default='diag_images', help='Directory to save diagnostic images.')
    
    args = parser.parse_args()
    
    # Post-process JSON arguments
    args.k_layer_config = json.loads(args.k_layer_config)
    args.milestones = json.loads(args.milestones)
    args.grid_size = (args.target_size_h + 2*args.padding, args.target_size_w + 2*args.padding)

    return args

def normalize_grads(model):
    for p in model.parameters():
        if p.grad is not None:
            p.grad.data.div_(p.grad.data.norm(2) + 1e-8)

def train(model, args, target_rgba):
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    scheduler = MultiStepLR(optimizer, milestones=args.milestones, gamma=args.gamma)

    target_rgba = target_rgba.to(args.device)
    
    # Create directory for diagnostic images
    os.makedirs(args.diag_image_dir, exist_ok=True)

    akorn_total_channels = args.nca_state_dim * args.n_osc_components
    init_state_akorn = torch.zeros(1, akorn_total_channels, *args.grid_size, device=args.device)
    # Seed: a single "active" point in the center for all oscillator components of all NCA channels
    center_h, center_w = args.grid_size[0] // 2, args.grid_size[1] // 2
    init_state_akorn[0, :, center_h, center_w] = 1.0 

    pool = init_state_akorn.repeat(args.pool_size, 1, 1, 1)

    results = {'loss':[], 'tprev': [time.time()]}
    print(f"Starting training on {args.device}...")

    for step in range(args.total_steps + 1):
        if args.pool_size > 0:
            pool_ixs = np.random.randint(args.pool_size, size=[args.batch_size])
            input_states_akorn = pool[pool_ixs].clone()
        else:
            input_states_akorn = init_state_akorn.repeat(args.batch_size, 1, 1, 1).clone()

        if args.perturb_n > 0 and args.batch_size >= args.perturb_n:
            perturb_masks = make_circle_masks(args.perturb_n, *args.grid_size, device=args.device)
            # Perturb all akorn channels for the selected batch items
            input_states_akorn[-args.perturb_n:] *= perturb_masks 

        num_nca_steps_for_iter = np.random.randint(args.num_nca_steps_min, args.num_nca_steps_max + 1)
        
        # Forward pass
        evolved_akorn_states_history, last_step_final_k_layer_energy_batch = model(input_states_akorn, 
                                                                                 num_nca_steps=num_nca_steps_for_iter,
                                                                                 k_layer_steps_per_nca_step=args.k_layer_steps_per_nca_step)
        
        final_akorn_state = evolved_akorn_states_history[-1] # Get the last state [B, AKORN_CH, H, W]
        output_rgba = model.get_rgba_from_akorn_state(final_akorn_state) # [B, 4, H, W]

        # Compute loss
        # Compare only RGBA channels (first 4)
        loss_per_pixel = (target_rgba[:, :4] - output_rgba[:, :4]).pow(2)
        
        # Weight loss by target alpha (more important to match where target is opaque)
        target_alpha = target_rgba[:, 3:4].clamp(0.01, 1.0) # Clamp to avoid zero division if used as weight
        loss = (loss_per_pixel * target_alpha).mean()


        optimizer.zero_grad()
        loss.backward()
        normalize_grads(model) # Optional: helps stabilize training
        optimizer.step()
        scheduler.step()

        if args.pool_size > 0:
            # Update pool: replace worst performing with seed, keep others
            batch_mses_per_item = loss_per_pixel.mean(dim=[1,2,3]) # Avg MSE for each item in batch
            worst_idx_in_batch = batch_mses_per_item.argmax()
            
            final_states_detached = evolved_akorn_states_history[-1].detach()
            final_states_detached[worst_idx_in_batch] = init_state_akorn[0].clone() # Replace worst with seed
            pool[pool_ixs] = final_states_detached

        results['loss'].append(loss.item())
        avg_energy_val = None # Define outside to be available for diag image name
        if last_step_final_k_layer_energy_batch is not None:
            avg_energy_val = last_step_final_k_layer_energy_batch.mean().item()

        if step % args.log_every == 0:
            log_message = f'Step {step}/{args.total_steps}, Time: {time.time()-results["tprev"][-1]:.2f}s, Loss: {loss.item():.4e}, LR: {optimizer.param_groups[0]["lr"]:.2e}'
            if avg_energy_val is not None:
                log_message += f', AvgKLayerEnergy: {avg_energy_val:.4e}'
            print(log_message)
            results['tprev'].append(time.time())

        if step % args.save_every == 0 or step == args.total_steps:
            os.makedirs(os.path.dirname(args.checkpoint_path), exist_ok=True)
            checkpoint = {
                'step': step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': loss.item(),
                'args': vars(args) # Save model architecture and training args
            }
            torch.save(checkpoint, args.checkpoint_path)
            print(f"Checkpoint saved to {args.checkpoint_path} at step {step}")

            # Save a diagnostic image from the current batch
            if output_rgba is not None and output_rgba.shape[0] > 0:
                # Take the first image from the batch
                img_tensor = output_rgba[0].detach().cpu() # [C, H, W]
                # Ensure it's 3 channels (RGB) for saving as common image format
                if img_tensor.shape[0] == 4: # RGBA
                    img_tensor = img_tensor[:3] # Take RGB
                
                # Permute to [H, W, C] for PIL
                img_np = img_tensor.permute(1, 2, 0).numpy()
                img_np = (np.clip(img_np, 0, 1) * 255).astype(np.uint8)
                pil_img = PIL.Image.fromarray(img_np)
                
                energy_str = f"_energy_{avg_energy_val:.2e}" if avg_energy_val is not None else "_energy_NA"
                diag_img_filename = os.path.join(args.diag_image_dir, f"step_{step:06d}_loss_{loss.item():.4e}{energy_str}.png")
                try:
                    pil_img.save(diag_img_filename)
                    print(f"Saved diagnostic image to {diag_img_filename}")
                except Exception as e:
                    print(f"Error saving diagnostic image: {e}")

    print("Training finished.")
    return results

def main():
    args = get_args()
    set_seed(args.seed)

    print("Effective Arguments:")
    for k, v in vars(args).items():
        print(f"  {k}: {v}")

    target_rgba_tensor = get_target_image(image_url=args.image_url, 
                                          target_size=(args.target_size_h, args.target_size_w),
                                          padding=args.padding)
    
    print(f"Target image shape (padded): {target_rgba_tensor.shape}")
    
    model = AKOrN_NCA(
        nca_state_dim=args.nca_state_dim,
        n_osc_components=args.n_osc_components,
        num_klayers=args.num_klayers,
        k_layer_config=args.k_layer_config,
        grid_size=args.grid_size,
        device=args.device,
        k_layer_gamma=args.k_layer_gamma, 
        rgba_readout_out_dim=args.rgba_readout_out_dim,
        use_conditioning_readout=args.use_conditioning_readout,
        cond_readout_out_dim=args.cond_readout_out_dim
    ).to(args.device)

    print(f"Model initialized with {sum(p.numel() for p in model.parameters() if p.requires_grad)} trainable parameters.")

    train(model, args, target_rgba_tensor)

if __name__ == '__main__':
    main()