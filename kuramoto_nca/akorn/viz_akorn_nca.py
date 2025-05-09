import argparse
import json
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from kuramoto_nca.akorn.akorn_nca_model import \
    AKOrN_NCA  # Assuming akorn_nca_model.py is in the same directory
from imageio import mimsave


# Function to set seeds for reproducibility
def set_seed(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)

def to_rgb(akorn_state_batch, model_for_rgba_conversion):
    # Input: akorn_state_batch [B, AKORN_CH, H, W]
    # Output: list of numpy arrays [H, W, C] for RGB
    rgba_batch = model_for_rgba_conversion.get_rgba_from_akorn_state(akorn_state_batch) # [B, 4, H, W]
    
    rgb_images = []
    for i in range(rgba_batch.shape[0]):
        # Extract RGBA for the i-th item
        rgba = rgba_batch[i] # [4, H, W]
        
        # Premultiplied alpha: rgb_display = rgb_raw * alpha + background * (1-alpha)
        # Assuming white background (1.0, 1.0, 1.0)
        rgb_raw = rgba[:3]  # [3, H, W]
        # alpha = rgba[3:4].clamp(0,1) # [1, H, W] # Alpha is handled by get_rgba_from_akorn_state if it uses sigmoid
        
        # display_rgb = rgb_raw # If your get_rgba_from_akorn_state already premultiplies or handles it.
        # The get_rgba_from_akorn_state applies sigmoid, so values are 0-1.
        # For visualization, we usually want to see the RGB part.
        # If alpha is the 4th channel and means transparency, viewers handle it.
        # If it means opacity over black, then rgb_raw * alpha.
        # For now, let's assume direct RGB output is what we want to see, and alpha is for compositing.
        display_rgb = rgb_raw 
        
        # Convert to numpy: [H, W, 3]
        display_rgb_np = display_rgb.permute(1, 2, 0).detach().cpu().numpy()
        rgb_images.append(np.clip(display_rgb_np, 0, 1)) # Clip to be safe
    return rgb_images


# --- Main execution block ---
def main():

    parser = argparse.ArgumentParser(description="Visualize AKOrN-NCA evolution.")
    parser.add_argument('--checkpoint_path', type=str, default='checkpoints/akorn_nca.pth', help='Path to the model checkpoint (.pth file).')
    parser.add_argument('--output_dir', type=str, default='visualizations/akorn_nca', help='Directory to save visualizations.')
    parser.add_argument('--num_frames', type=int, default=200, help='Number of NCA steps to visualize.')
    parser.add_argument('--steps_per_frame', type=int, default=1, help='Number of NCA steps between saved frames for GIF.')
    parser.add_argument('--k_layer_steps_per_nca_step', type=int, default=None, help='Override KLayer steps per NCA step from checkpoint args.')
    parser.add_argument('--seed', type=int, default=None, help='Random seed for initial state generation (if needed, though usually fixed for viz).')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use.')

    vis_args = parser.parse_args()

    if vis_args.seed is None:
        actual_seed = random.randint(0, 2**32 - 1)
        print(f"No visualization seed provided. Using randomly generated seed for output naming: {actual_seed}")
    else:
        actual_seed = vis_args.seed
        print(f"Using provided visualization seed for output naming: {actual_seed}")
    set_seed(actual_seed)

    # --- Load Checkpoint and Model Args ---
    if not os.path.exists(vis_args.checkpoint_path):
        print(f"Error: Checkpoint file not found at {vis_args.checkpoint_path}")
        return
    
    checkpoint = torch.load(vis_args.checkpoint_path, map_location=vis_args.device)
    train_args_dict = checkpoint.get('args')
    if train_args_dict is None:
        print("Error: Training arguments not found in checkpoint. Cannot infer model structure.")
        return
    
    # Convert train_args dict to an object for easier access if needed, or just use as dict
    # train_args = type('Args', (object,), train_args_dict) # If you prefer attribute access

    print("Loaded training arguments from checkpoint:")
    for k, v in train_args_dict.items():
        print(f"  {k}: {v}")

    # --- Build Model ---
    grid_h, grid_w = train_args_dict['grid_size'] 

    # Parse k_layer_config from JSON string if it's a string
    k_layer_config_from_args = train_args_dict['k_layer_config']
    k_layer_config_parsed = None
    if isinstance(k_layer_config_from_args, str):
        try:
            k_layer_config_parsed = json.loads(k_layer_config_from_args)
        except json.JSONDecodeError:
            print(f"Error: Could not parse k_layer_config JSON string from checkpoint: {k_layer_config_from_args}")
            return
    else:
        # If it's already a dict or list (e.g., if args were modified before saving)
        k_layer_config_parsed = k_layer_config_from_args

    model = AKOrN_NCA(
        nca_state_dim=train_args_dict['nca_state_dim'],
        n_osc_components=train_args_dict['n_osc_components'],
        num_klayers=train_args_dict['num_klayers'],
        k_layer_config=k_layer_config_parsed,
        grid_size=(grid_h, grid_w),
        device=vis_args.device,
        k_layer_gamma=train_args_dict.get('k_layer_gamma', 0.1), # Added with default
        rgba_readout_out_dim=train_args_dict.get('rgba_readout_out_dim', 2), # Added with default
        use_conditioning_readout=train_args_dict.get('use_conditioning_readout', False), # Added with default
        cond_readout_out_dim=train_args_dict.get('cond_readout_out_dim', 1) # Added with default
    ).to(vis_args.device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"Model loaded from {vis_args.checkpoint_path} and set to eval mode.")

    # --- Initial State (Seed) ---
    akorn_total_channels = train_args_dict['nca_state_dim'] * train_args_dict['n_osc_components']
    initial_akorn_state = torch.zeros(1, akorn_total_channels, grid_h, grid_w, device=vis_args.device)
    center_h, center_w = grid_h // 2, grid_w // 2
    # A more robust seed: set a small patch of channels to 1.0
    # For example, setting all components of the first NCA state channel to 1.0 at the center
    start_channel_idx = 0 # e.g. first NCA state channel
    num_components = train_args_dict['n_osc_components']
    initial_akorn_state[0, start_channel_idx*num_components : (start_channel_idx+1)*num_components, center_h, center_w] = 1.0


    # --- Rollout ---
    evolved_frames_akorn = []
    current_akorn_state = initial_akorn_state.clone()
    
    k_steps_override = vis_args.k_layer_steps_per_nca_step if vis_args.k_layer_steps_per_nca_step is not None else train_args_dict['k_layer_steps_per_nca_step']

    print(f"Starting rollout for {vis_args.num_frames} frames ({vis_args.steps_per_frame} NCA steps per frame)... KLayer T={k_steps_override}")
    for i in range(vis_args.num_frames):
        with torch.no_grad():
            # model.forward returns (stacked_akorn_states, last_nca_step_final_k_layer_energy)
            stacked_states_history, _ = model(current_akorn_state, 
                                              num_nca_steps=vis_args.steps_per_frame, 
                                              k_layer_steps_per_nca_step=k_steps_override)
            current_akorn_state = stacked_states_history[-1] # Get the state after 'steps_per_frame' NCA steps
        evolved_frames_akorn.append(current_akorn_state.clone()) # Store the state [1, CH, H, W]
        if (i+1) % 20 == 0 or i == 0 or i == vis_args.num_frames -1 : # Log first, last and every 20th
            print(f"  Generated frame {i+1}/{vis_args.num_frames}")

    # --- Convert to RGB and Save ---
    print("Converting Akorn states to RGB...")
    rgb_frames_for_gif = []
    for akorn_frame_batch in evolved_frames_akorn: # Each item is [1, CH, H, W]
        rgb_list = to_rgb(akorn_frame_batch, model) # Returns list with one image
        rgb_frames_for_gif.append(rgb_list[0])

    os.makedirs(vis_args.output_dir, exist_ok=True)
    
    # Save GIF
    # Use a unique filename based on checkpoint name and seed
    checkpoint_name_base = os.path.splitext(os.path.basename(vis_args.checkpoint_path))[0]
    gif_filename = os.path.join(vis_args.output_dir, f"{checkpoint_name_base}_viz_seed{actual_seed}.gif")
    mimsave(gif_filename, [(img * 255).astype(np.uint8) for img in rgb_frames_for_gif], fps=15)
    print(f"Saved GIF: {gif_filename}")

    # Save Strip Plot
    num_strip_frames = min(len(rgb_frames_for_gif), 10) # Show up to 10 frames in strip
    if num_strip_frames > 0:
        indices = np.linspace(0, len(rgb_frames_for_gif) - 1, num_strip_frames, dtype=int).tolist()
        
        fig, axes = plt.subplots(1, num_strip_frames, figsize=(2 * num_strip_frames, 2.2))
        if num_strip_frames == 1: axes = [axes] # Ensure axes is iterable for single frame
            
        for i, frame_idx in enumerate(indices):
            ax = axes[i]
            ax.imshow(rgb_frames_for_gif[frame_idx])
            ax.set_title(f"NCA Step: {frame_idx * vis_args.steps_per_frame}") # Frame index corresponds to this many NCA steps
            ax.axis('off')
        
        plt.tight_layout()
        strip_filename = os.path.join(vis_args.output_dir, f"{checkpoint_name_base}_strip_seed{actual_seed}.png")
        plt.savefig(strip_filename)
        print(f"Saved strip plot: {strip_filename}")
    else:
        print("No frames generated for strip plot.")
    # plt.show() # Optionally display plot

if __name__ == "__main__":
    main()