import torch
import torch.nn as nn
import torch.nn.functional as F

from source.layers.klayer import \
    KLayer  # Assuming KLayer is in source/layers/klayer.py
from source.layers.common_layers import ReadOutConv # Added import


class AKOrN_NCA(nn.Module):
    def __init__(self,
                 nca_state_dim=16,       # Number of channels in the NCA state (e.g., RGBA + latent)
                 n_osc_components=2,     # Number of components per Kuramoto oscillator (e.g., 2 for (phase, amplitude) or (x,y))
                 num_klayers=1,          # Number of KLayers to apply sequentially per NCA step
                 k_layer_config=None,    # Dict or list of dicts for KLayer configurations
                 grid_size=(72, 72),
                 device="cpu",
                 k_layer_gamma=0.1,      # Step size for KLayer internal dynamics
                 rgba_readout_out_dim=2, # out_dim for the RGBA ReadOutConv
                 use_conditioning_readout=False, # Whether to use ReadOutConv for KLayer conditioning
                 cond_readout_out_dim=1): # out_dim for the conditioning ReadOutConv
        super().__init__()
        self.nca_state_dim = nca_state_dim
        self.n_osc_components = n_osc_components
        self.akorn_total_channels = nca_state_dim * n_osc_components
        self.num_klayers = num_klayers
        self.device = device
        self.grid_size = grid_size
        self.k_layer_gamma = k_layer_gamma # Store gamma
        self.use_conditioning_readout = use_conditioning_readout # Store this flag
        self.rgba_readout_out_dim = rgba_readout_out_dim
        self.cond_readout_out_dim = cond_readout_out_dim

        # Validate n_osc_components early, as it's the default for KLayer's 'n'
        if self.n_osc_components != 2:
            raise ValueError(
                f"AKOrN_NCA initialization error: The underlying KLayer "
                f"(used by AKOrN_NCA) currently requires 'n_osc_components' to be exactly 2. "
                f"Received: {self.n_osc_components}"
            )

        if k_layer_config is None:
            k_layer_config = {} # Default config

        self.k_layers = nn.ModuleList()
        for i in range(num_klayers):
            config = {}
            if isinstance(k_layer_config, list):
                config = k_layer_config[i] if i < len(k_layer_config) else k_layer_config[-1]
            elif isinstance(k_layer_config, dict):
                config = k_layer_config

            klayer_n_value = config.get('n', self.n_osc_components)

            if klayer_n_value != 2:
                raise ValueError(
                    f"AKOrN_NCA KLayer configuration error (for KLayer instance {i}): "
                    f"The underlying KLayer currently requires 'n' (number of oscillator components) "
                    f"to be exactly 2. Configuration for this KLayer resolved to n={klayer_n_value}."
                )

            current_ch = config.get('ch', self.akorn_total_channels)
            current_n = klayer_n_value
            current_J = config.get('J', "conv")
            current_ksize = config.get('ksize', 3)
            current_gta = config.get('gta', False)
            current_hw = config.get('hw', self.grid_size if current_J == "attn" else None)
            current_use_omega = config.get('use_omega', True)
            current_c_norm = config.get('c_norm', 'none')
            current_apply_proj = config.get('apply_proj', True)
            current_init_omg = config.get('init_omg', 1.0)
            current_global_omg = config.get('global_omg', False)
            current_heads = config.get('heads', 8)
            current_learn_omg = config.get('learn_omg', True)
            
            self.k_layers.append(
                KLayer(ch=current_ch,
                       n=current_n,
                       J=current_J,
                       ksize=current_ksize,
                       gta=current_gta,
                       hw=current_hw,
                       use_omega=current_use_omega,
                       c_norm=current_c_norm,
                       apply_proj=current_apply_proj,
                       init_omg=current_init_omg,
                       global_omg=current_global_omg,
                       heads=current_heads,
                       learn_omg=current_learn_omg
                      ).to(device)
            )

        # RGBA Readout Module (for final image interpretation)
        self.rgba_readout = ReadOutConv(
            inch=self.akorn_total_channels,
            outch=4, # For RGBA
            out_dim=self.rgba_readout_out_dim
        ).to(device)

        # Conditioning Readout Module (for generating KLayer's 'c' signal)
        if self.use_conditioning_readout:
            self.conditioning_readout = ReadOutConv(
                inch=self.akorn_total_channels,
                outch=self.akorn_total_channels, # c must match KLayer's channel dimension
                out_dim=self.cond_readout_out_dim 
            ).to(device)
        else:
            self.conditioning_readout = None

        # Define which NCA channel is alpha (e.g., the 4th channel if RGBA is first)
        # This is still needed for the alive_mask logic, which operates on the raw state
        self.alpha_nca_channel_idx = 3 # Assuming RGBA where A is at index 3
        # Define which oscillator component of the alpha channel to use for alive check
        self.alpha_osc_component_idx = 0 # e.g., the first component

    def get_rgba_from_akorn_state(self, akorn_state):
        # akorn_state: [B, AKORN_TOTAL_CHANNELS, H, W]
        # Pass through the RGBA readout module
        rgba_output = self.rgba_readout(akorn_state)
        # Apply sigmoid to keep color values in a displayable range [0,1]
        return torch.sigmoid(rgba_output) # [B, 4, H, W]

    def alive_mask(self, akorn_state):
        # akorn_state: [B, AKORN_TOTAL_CHANNELS, H, W]
        b, _, h, w = akorn_state.shape
        if self.nca_state_dim <= self.alpha_nca_channel_idx:
            # If not enough channels for a designated alpha, assume all are alive
            return torch.ones(b, 1, h, w, device=akorn_state.device).bool()

        # Reshape to access NCA channels and their oscillator components for alive check
        # This is independent of the RGBA readout method
        reshaped_state = akorn_state.view(b, self.nca_state_dim, self.n_osc_components, h, w)
        
        # Get the specific component of the alpha oscillator
        alpha_values = reshaped_state[:, self.alpha_nca_channel_idx, self.alpha_osc_component_idx, :, :] # [B, H, W]
        
        # NCA's alive rule: max pooling of alpha > threshold
        return (F.max_pool2d(alpha_values.unsqueeze(1), kernel_size=3, stride=1, padding=1) > 0.1) # [B, 1, H, W]


    def forward(self, initial_akorn_state, num_nca_steps, k_layer_steps_per_nca_step=1):
        # initial_akorn_state: [B, AKORN_TOTAL_CHANNELS, H, W]
        x_akorn = initial_akorn_state.to(self.device)
        
        history_akorn_states = []
        last_nca_step_final_k_layer_energy = None # To store energy from the last KLayer of the last NCA step

        for nca_step_idx in range(num_nca_steps):
            x_akorn_old = x_akorn.clone() 

            pre_update_alive = self.alive_mask(x_akorn_old)
            stochastic_mask = (torch.rand(x_akorn.shape[0], 1, x_akorn.shape[2], x_akorn.shape[3],
                                          device=self.device) > 0.5).float()

            # Pass through KLayer(s)
            current_x = x_akorn_old
            
            # Generate initial conditioning signal 'c' for the KLayer sequence
            if self.conditioning_readout is not None:
                current_c = self.conditioning_readout(current_x)
            else:
                current_c = torch.zeros_like(current_x) 
            
            current_step_last_k_layer_energy = None # Energy for the current NCA step

            for k_idx, k_layer in enumerate(self.k_layers):
                # If this is not the first KLayer in the sequence (k_idx > 0)
                # AND conditioning_readout is enabled,
                # then update 'c' based on the output of the previous KLayer.
                # Otherwise, 'c' remains as it was (either from initial readout or zeros).
                if k_idx > 0 and self.conditioning_readout is not None:
                    # current_x here is the output from the (k_idx-1)-th KLayer
                    current_c = self.conditioning_readout(current_x)
                # If conditioning_readout is None, current_c remains zeros for all k_layers.
                # If conditioning_readout is not None, for k_idx=0, c is from readout(x_akorn_old).
                # For k_idx > 0, c is from readout(output of previous KLayer).
                
                k_layer_output_states, k_layer_energies_list = k_layer(current_x, 
                                                                       c=current_c,
                                                                       T=k_layer_steps_per_nca_step,
                                                                       gamma=self.k_layer_gamma)
                current_x = k_layer_output_states[-1] # Update current_x for the next KLayer or for final output
                
                # Store energy from the last internal step of this KLayer
                if k_layer_energies_list: # Check if KLayer produced energies (T > 0)
                    current_step_last_k_layer_energy = k_layer_energies_list[-1]


            # If this is the last NCA step, store the energy from its last KLayer
            if nca_step_idx == num_nca_steps - 1:
                last_nca_step_final_k_layer_energy = current_step_last_k_layer_energy
            
            x_akorn_after_klayers = current_x
            delta_x_akorn = x_akorn_after_klayers - x_akorn_old
            
            x_akorn = x_akorn_old + stochastic_mask * delta_x_akorn
            
            current_alive = self.alive_mask(x_akorn)
            alive_and_kicking = pre_update_alive * current_alive # Boolean, True for cells alive before AND after
            
            x_akorn = x_akorn * alive_and_kicking.float() # Zero out dead cells or cells that died
            
            history_akorn_states.append(x_akorn.clone())

        stacked_akorn_states = torch.stack(history_akorn_states) # [N_steps, B, CH, H, W]
        return stacked_akorn_states, last_nca_step_final_k_layer_energy