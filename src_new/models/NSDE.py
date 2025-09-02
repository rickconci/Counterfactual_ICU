import os
import sys

import matplotlib.pyplot as plt
import torch
import wandb
from lightning import LightningModule
from raindrop import Raindrop_v2
from torch import distributions, nn
import torchsde
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "utils"))
from utils_beta import (
    LinearScheduler,
    MLPSimple,
    get_last_valid_timestep_fast,
    select_tensor_by_index_list_advanced,
)

# <<< Global DEBUG flag for model_beta.py, to be set by instance >>>
# This is more of a placeholder if a module-level default is ever needed,
# but instance-level self.debug passed from main_beta.py is the primary control.
DEBUG = False


class NSDE(LightningModule):
    def __init__(
        self,
        use_encoder,
        # Encoder
        encoder_input_dim,
        encoder_hidden_dim,
        encoder_SDENN_dims,
        expert_latent_dims,
        encoder_num_layers,
        encoder_w_time,
        encoder_reverse_time,
        n_medications,
        encoder_context_len,
        # New static fusion params
        static_input_dim,
        static_hidden_dim,
        fusion_hidden_dim,
        # SDE params
        prior_tx_sigma,
        prior_tx_mu,
        include_time,
        # SDE model params
        num_samples,
        SDEnet_hidden_dim,
        SDEnet_depth,
        final_activation,
        use_batch_norm,
        integration_step_size,
        integration_method,
        rtol,
        atol,
        integration_adaptive,
        # decoder params
        decoder_output_dims,
        log_lik_output_scale,
        normalised_data,
        start_dec_at_treatment,
        # admin
        train_dir,
        learning_rate,
        log_wandb,
        adjoint,
        plot_every,
        dataset,
        KL_weighting_SDE,
        debug=False,  # <<< Add debug flag >>>
    ):
        super().__init__()
        self.debug = debug  # <<< Store debug flag >>>
        global DEBUG
        DEBUG = (
            self.debug
        )  # <<< Update module-level DEBUG if needed, primarily use self.debug >>>

        if self.debug:
            print(
                f"[DEBUG] ODE __init__: Initializing... adjoint={adjoint}, use_encoder={use_encoder}, normalise_for_ODENN={normalise_for_ODENN}"
            )

        ### ADMIN
        self.train_dir = train_dir
        self.learning_rate = learning_rate
        self.log_wandb = log_wandb
        self.plot_every = plot_every

        ### Bifurcation options
        self.use_encoder = use_encoder
        self.include_time = include_time

        ### Encoder model
        self.encoder_SDENN_dims = encoder_SDENN_dims
        self.encoder_output_dim = encoder_SDENN_dims + expert_latent_dims

        self.start_dec_at_treatment = start_dec_at_treatment

        self.noise_scale = prior_tx_sigma

        self.KL_weighting_SDE = KL_weighting_SDE

        temporal_embedding_dim = 0  # To store the output dim of the temporal encoder

        if use_encoder == "raindrop":
            # For Raindrop, d_model is its internal feature size.
            # Its output before projection will be d_model + d_pe if sensor_wise_mask is False
            # TODO not quite sure if this is still right
            d_ob = max(int(encoder_hidden_dim / encoder_input_dim), 2)
            temporal_embedding_dim = encoder_input_dim * d_ob + 16  # d_model + d_pe
            max_len_ctx = (
                120 if encoder_context_len is None else int(encoder_context_len)
            )
            self.temporal_encoder = Raindrop_v2(
                d_inp=encoder_input_dim,
                d_model=encoder_hidden_dim,
                output_dim=temporal_embedding_dim,  # Not used since we commented out final layer
                nhead=4,
                nhid=128,
                max_len=max_len_ctx,
                global_structure=torch.ones(
                    encoder_input_dim, encoder_input_dim
                ),  # pass a complete adj matrix
                nlayers=encoder_num_layers,
                static=False,
                debug=self.debug,
            )
        elif use_encoder != "none":
            temporal_embedding_dim = encoder_hidden_dim
            self.temporal_encoder = Encoder(
                input_dim=encoder_input_dim,
                hidden_dim=encoder_hidden_dim,
                latent_dim=temporal_embedding_dim,  # GRU output is hidden_dim
                expert_latent_dims=expert_latent_dims,
                variational=variational_encoder,
                encode_with_time_dim=encoder_w_time,
                encoder_num_layers=encoder_num_layers,
                reverse=encoder_reverse_time,
                debug=self.debug,
            )
        else:
            self.temporal_encoder = None

        # --- New Static Encoder and Fusion Heads ---
        if use_encoder != "none":
            self.static_encoder = MLPSimple(
                input_dim=static_input_dim,
                output_dim=static_hidden_dim,
                hidden_dim=static_hidden_dim,
                depth=2,
                activations=[nn.ReLU(), nn.ReLU()],
                debug=self.debug,
            )
            # This MLP fuses the temporal and static embeddings
            self.fusion_mlp = MLPSimple(
                input_dim=temporal_embedding_dim + static_hidden_dim,
                output_dim=fusion_hidden_dim,
                hidden_dim=(temporal_embedding_dim + static_hidden_dim) // 2,
                depth=2,
                activations=[nn.ReLU(), nn.ReLU()],
                debug=self.debug,
            )
            # Head 1: Predicts the initial state for all 14 expert ODE variables
            self.ode_latent_head = nn.Sequential(
                nn.Linear(fusion_hidden_dim, expert_latent_dims), nn.Sigmoid()
            )
            # Head 2: Predicts the separate embedding for the neural SDE component
            self.neural_embedding_head = nn.Linear(
                fusion_hidden_dim, encoder_SDENN_dims
            )
        else:
            self.static_encoder = None
            self.fusion_mlp = None
            self.ode_latent_head = None
            self.neural_embedding_head = None
        # --- End New ---

        ### LATENT MODEL

        self.num_samples = num_samples
        self.expert_latent_dims = expert_latent_dims

        self.SDEnet_hidden_dim = SDEnet_hidden_dim
        self.SDEnet_depth = SDEnet_depth
        self.SDEnet_out_dims = 2

        net_input_dims = self.encoder_output_dim
        net_input_dims = net_input_dims + 2 if include_time else net_input_dims

        if self.use_encoder != "none":
            self.ic_consistency_weight = 0.1
            # each medication has rate and last administration info
            net_input_dims = net_input_dims + n_medications * 2
        else:
            self.ic_consistency_weight = 0
            # each medication has rate and last administration info
            net_input_dims = self.expert_latent_dims + n_medications * 2
            net_input_dims = net_input_dims + 2 if include_time else net_input_dims

        activations = {"relu": nn.ReLU(), "tanh": nn.Tanh(), "none": None}
        final_activation_real = activations[final_activation.lower()]

        # TODO change net input dims to be 14 + number of meds if there is no encoder, else encoder dim + 14 + meds

        self.SDEnet = MLPSimple(
            input_dim=net_input_dims,
            output_dim=2,
            hidden_dim=SDEnet_hidden_dim,
            depth=SDEnet_depth,
            activations=[nn.Tanh() for _ in range(SDEnet_depth)],
            final_activation=final_activation_real,
            use_batch_norm=use_batch_norm,
            debug=self.debug,
        )

        # Initialization trick from Glow.
        # self.SDEnet.output_layer[0].weight.data.fill_(0.)
        # self.SDEnet.output_layer[0].bias.data.fill_(0.)

        ### DECODER
        self.decoder_output_dims = decoder_output_dims
        self.normalised_data = normalised_data
        self.dataset = dataset

        self.sdeint_fn = torchsde.sdeint_adjoint if adjoint else torchsde.sdeint
        self.integration_step_size = integration_step_size
        self.integration_method = integration_method
        self.rtol = rtol
        self.atol = atol
        self.integration_adaptive = integration_adaptive

        ### LOSS
        self.MSE_loss = nn.MSELoss(reduction="none")
        self.log_lik_output_scale = log_lik_output_scale
        self.kl_scheduler = LinearScheduler(
            start=70, iters=600, startval=1.0, endval=0.01
        )
        self.save_hyperparameters()
        if self.debug:
            print(
                f"[DEBUG] Hybrid_SDE __init__: Initialization complete. Encoder output dim: {self.encoder_output_dim}, SDEnet input dims: {net_input_dims}"
            )

        # check baroreflex sensitivity
        self.physio_ranges = {
            "p_a": (39, 180.0),
            "p_v": (0.0, 39.0),
            "s_reflex": (0, 1),
            "sv": (40.0, 120.0),
            "r_tpr_mod": (-1.0, 1.0),
            "f_hr_max": (2.0, 3.0),
            "f_hr_min": (0.9, 1.1),
            "r_tpr_max": (1.8, 2.4),
            "r_tpr_min": (0.45, 0.6),
            "ca": (2.0, 6.0),
            "cv": (90.0, 120.0),
            "k_width": (0.1, 0.3),
            "p_aset": (50.0, 90.0),
            "tau": (15, 25),
        }

        self.noise_type = "diagonal"  # required
        self.sde_type = "ito"

        # In __init__, add these parameters (you'll need to pass them as arguments):
        self.first_two_normalization_mu = torch.tensor(
            [78.937, 8.505], dtype=torch.float32
        )
        self.first_two_normalization_sigma = torch.tensor(
            [23.009, 7.948], dtype=torch.float32
        )

        # Pre-compute range tensors for efficiency
        self.register_buffer(
            "physio_min_vals",
            torch.tensor(
                [
                    self.physio_ranges[k][0]
                    for k in [
                        "p_a",
                        "p_v",
                        "s_reflex",
                        "sv",
                        "r_tpr_mod",
                        "f_hr_max",
                        "f_hr_min",
                        "r_tpr_max",
                        "r_tpr_min",
                        "ca",
                        "cv",
                        "k_width",
                        "p_aset",
                        "tau",
                    ]
                ]
            ),
        )

        self.register_buffer(
            "physio_max_vals",
            torch.tensor(
                [
                    self.physio_ranges[k][1]
                    for k in [
                        "p_a",
                        "p_v",
                        "s_reflex",
                        "sv",
                        "r_tpr_mod",
                        "f_hr_max",
                        "f_hr_min",
                        "r_tpr_max",
                        "r_tpr_min",
                        "ca",
                        "cv",
                        "k_width",
                        "p_aset",
                        "tau",
                    ]
                ]
            ),
        )

    def transform_sigmoid_to_physiological_ranges(self, sigmoid_values):
        # TODO check this again in encoder setting
        """Simplified version using pre-computed ranges"""
        # Check input for NaN/inf
        if self.debug:
            print("[DEBUG] Physiological transform input stats:")
            print(f"  Shape: {sigmoid_values.shape}")
            print(
                f"  Min/Max: {sigmoid_values.min().item()}/{sigmoid_values.max().item()}"
            )
            print(f"  Contains NaN: {torch.isnan(sigmoid_values).any()}")
            print(f"  Sample values: {sigmoid_values[0, 0, :5]}")
            try:
                p10 = torch.quantile(sigmoid_values, 0.10).item()
                p50 = torch.quantile(sigmoid_values, 0.50).item()
                p90 = torch.quantile(sigmoid_values, 0.90).item()
                print(f"  Quantiles p10/p50/p90: {p10:.3f}/{p50:.3f}/{p90:.3f}")
                pa_raw = sigmoid_values[..., 0]
                pv_raw = sigmoid_values[..., 1]
                print(
                    f"  Raw p_a sigmoid min/max: {pa_raw.min().item():.3f}/{pa_raw.max().item():.3f}"
                )
                print(
                    f"  Raw p_v sigmoid min/max: {pv_raw.min().item():.3f}/{pv_raw.max().item():.3f}"
                )
            except Exception:
                pass

        # Check that sigmoid values are actually in [0,1] range
        if sigmoid_values.min().item() < 0 or sigmoid_values.max().item() > 1:
            print("[WARNING] Sigmoid values outside [0,1] range!")

        transformed = self.physio_min_vals + sigmoid_values * (
                self.physio_max_vals - self.physio_min_vals
        )
        transformed = torch.clamp(
            transformed, min=self.physio_min_vals, max=self.physio_max_vals
        )

        if self.debug:
            print("[DEBUG] Physiological transform output stats:")
            print(f"  Min/Max: {transformed.min().item()}/{transformed.max().item()}")
            print(f"  Contains NaN: {torch.isnan(transformed).any()}")
            try:
                pa = transformed[..., 0]
                pv = transformed[..., 1]
                print(
                    f"  p_a (mmHg) range: {pa.min().item():.2f} - {pa.max().item():.2f}"
                )
                print(
                    f"  p_v (mmHg) range: {pv.min().item():.2f} - {pv.max().item():.2f}"
                )
                print(f"  Sample transformed (first 5 dims): {transformed[0, 0, :5]}")
            except Exception:
                print(f"  Sample transformed: {transformed[0, 0, :5]}")

        return transformed

    def normalize_static_zenker_vars(self, static_vars):
        """Normalize positions 2-13 of Zenker variables once at initialization"""
        # static_vars: [batch, 1, 12] - positions 2-13 of Zenker state

        remaining_min = self.physio_min_vals[2:].to(static_vars.device)  # Skip first 2
        remaining_max = self.physio_max_vals[2:].to(static_vars.device)  # Skip first 2
        midpoints = remaining_min + 0.5 * (remaining_max - remaining_min)
        ranges = remaining_max - remaining_min
        ranges = torch.clamp(ranges, min=1e-8)

        normalized_static = (static_vars - midpoints) / ranges
        return normalized_static

    def normalize_pressures_only(self, pressures):
        """Normalize only p_a, p_v using their specific mu/sigma"""
        first_two_mu = self.first_two_normalization_mu.to(pressures.device)
        first_two_sigma = self.first_two_normalization_sigma.to(pressures.device)
        normalized_pressures = (pressures - first_two_mu) / first_two_sigma
        return normalized_pressures

    def forward_enc(self, input_vals, time_in, static=None, lengths=None):
        if self.debug:
            print(
                f"[DEBUG] Hybrid_SDE forward_enc: input_vals_shape={input_vals.shape}, time_in_shape={time_in.shape}, use_encoder={self.use_encoder}"
            )

        if self.use_encoder == "raindrop":
            # Fix: should be self.temporal_encoder, not self.enc_model
            z1, _, _ = self.temporal_encoder(
                src=input_vals, static=static, times=time_in, lengths=lengths
            )
            return z1, None, 0

        elif self.use_encoder != "none":
            if self.start_dec_at_treatment:
                if self.variational_encoder:
                    # Fix: should be self.temporal_encoder
                    z1_mean, z1_logvar = self.temporal_encoder(input_vals, time_in)
                    z1 = z1_mean.unsqueeze(1).repeat(1, self.num_samples, 1)
                    logqp0 = 0
                else:
                    # Fix: should be self.temporal_encoder
                    z1 = self.temporal_encoder(input_vals, time_in)
                    if self.debug:
                        print(
                            f"[DEBUG] Hybrid_SDE forward_enc (non-variational): Encoder output z1_shape (before repeat): {z1.shape}"
                        )
                    # The following line seems incorrect as sigmoid_scale is not a method of this class. Assuming it's a typo from original code.
                    # z1 = torch.cat([self.sigmoid_scale(z1[:,:self.expert_latent_dims], self.use_2_5std_encoder_minmax), z1[:, self.expert_latent_dims:] ], dim =-1)
                    z1 = z1.unsqueeze(1).repeat(1, self.num_samples, 1)
                    logqp0 = 0
                    z1_logvar = None
            else:
                z1 = input_vals.unsqueeze(1).repeat(1, self.num_samples, 1)
                logqp0 = 0
                z1_logvar = None
        else:  # No encoder
            z1 = input_vals.unsqueeze(1).repeat(1, self.num_samples, 1)
            if self.debug:
                print(
                    f"[DEBUG] Hybrid_SDE forward_enc (no encoder, no variational sampling): z1_shape={z1.shape}"
                )
            logqp0 = 0
            z1_logvar = None

        if self.debug:
            print(
                f"[DEBUG] Hybrid_SDE forward_enc: Returning z1_shape={z1.shape}, z1_logvar_type={type(z1_logvar)}, logqp0_type={type(logqp0)}"
            )
        return z1, z1_logvar, logqp0

    def get_medication_context(self, t, expanded_batch_size):
        """Medication context at time t, using precomputed per-time tensors when available."""
        batch_size, _, n_meds = self.current_med_values.shape

        # Fast path: precomputed per-time context
        if hasattr(self, "med_context_by_time") and hasattr(self, "_dt_grid"):
            t_tensor = (
                t
                if torch.is_tensor(t)
                else torch.tensor(t, dtype=self._t0.dtype, device=self._t0.device)
            )
            idx = torch.round((t_tensor - self._t0) / self._dt_grid).to(torch.long)
            idx = torch.clamp(idx, 0, self._time_len - 1)
            ctx_bt = self.med_context_by_time[:, idx, :]  # [B, 2*M]
            samples_per_batch = expanded_batch_size // batch_size
            return ctx_bt.repeat_interleave(samples_per_batch, dim=0)

        # Fallback: on-the-fly computation (slower)
        time_steps = self.current_med_values.shape[1]
        t_tensor = (
            t
            if torch.is_tensor(t)
            else torch.tensor(
                t, dtype=self.current_med_time.dtype, device=self.current_med_time.device
            )
        )
        time_valid = self.current_med_time <= t_tensor  # (batch, time)
        med_present = self.current_med_mask > 0  # (batch, time, meds)
        valid_mask = time_valid.unsqueeze(-1) & med_present  # (batch, time, meds)

        time_indices = torch.arange(time_steps, device=self.device).float()
        time_indices = time_indices.view(1, time_steps, 1).expand(
            batch_size, -1, n_meds
        )
        masked_indices = torch.where(
            valid_mask, time_indices, torch.full_like(time_indices, -1e6)
        )
        last_valid_indices = masked_indices.argmax(dim=1)
        has_valid_data = masked_indices.max(dim=1)[0] > -1e5
        batch_idx = torch.arange(batch_size, device=self.device).unsqueeze(1)
        med_idx = torch.arange(n_meds, device=self.device).unsqueeze(0)
        last_rates = self.current_med_values[batch_idx, last_valid_indices, med_idx]
        last_times = self.current_med_time[batch_idx, last_valid_indices]
        time_since = t_tensor - last_times
        recency_weights = torch.clamp(time_since / 1200 - 1 / 1200, min=0)
        last_rates = torch.where(has_valid_data, last_rates, torch.zeros_like(last_rates))
        recency_weights = torch.where(
            has_valid_data, recency_weights, torch.ones_like(recency_weights)
        )
        result = torch.stack([last_rates, recency_weights], dim=-1).flatten(start_dim=1)
        samples_per_batch = expanded_batch_size // batch_size
        return result.repeat_interleave(samples_per_batch, dim=0)

    def f(self, t, y):
        """SDE drift: predict pressure derivatives using full Zenker context"""
        batch_size = y.shape[0]  # This is actually batch*samples

        # y is just [p_a, p_v] - the only evolving variables: [batch*samples, 2]
        current_pressures = y

        # Normalize only the evolving pressures
        normalized_pressures = self.normalize_pressures_only(current_pressures)

        # Combine with pre-normalized static Zenker context (already flattened)
        normalized_full_zenker = torch.cat([
            normalized_pressures,  # [batch*samples, 2]
            self.normalized_static_zenker  # [batch*samples, 12]
        ], dim=-1)  # [batch*samples, 14]

        # Build network input: normalized Zenker + neural embedding
        if self.static_neural_embedding is not None:
            nn_input = torch.cat([normalized_full_zenker, self.static_neural_embedding], dim=-1)
        else:
            nn_input = normalized_full_zenker

        # Add time encoding if enabled
        if self.include_time:
            time_encoding = torch.stack([torch.sin(t), torch.cos(t)]).repeat(batch_size, 1)
            nn_input = torch.cat([nn_input, time_encoding], dim=-1)

        # Add medication context (need to expand for samples)
        med_context = self.get_medication_context(t, batch_size)
        med_context = (med_context - 0.5) * 4.0
        nn_input = torch.cat([nn_input, med_context], dim=-1)

        # Get pressure derivatives from network
        pressure_derivatives = self.SDEnet(nn_input)  # [batch*samples, 2]

        return pressure_derivatives

    def g(self, t, y):
        """Diffusion function - adds stochasticity to pressures only"""
        batch_size = y.shape[0]  # This is actually batch*samples

        # Create diffusion matrix for 2D pressure state
        diffusion = torch.zeros(batch_size, 2, device=y.device)

        # Add noise to both pressure variables
        diffusion[:, 0] = self.noise_scale  # p_a noise
        diffusion[:, 1] = self.noise_scale  # p_v noise

        return diffusion

    def normalise_sde_inputs(self, expert_vars):
        """
        Normalize expert variables with different strategies:
        - First two variables: (x - mu) / sigma
        - Remaining variables: (x - midpoint) / (max - min)

        Args:
            expert_vars: Expert latent variables [batch, expert_latent_dims]

        Returns:
            normalized expert variables [batch, expert_latent_dims]
        """
        # Normalize first two expert variables with mu/sigma
        first_two = expert_vars[:, :2]  # [batch, 2]
        first_two_mu = self.first_two_normalization_mu.to(self.device)
        first_two_sigma = self.first_two_normalization_sigma.to(self.device)
        normalized_first_two = (first_two - first_two_mu) / first_two_sigma

        # Normalize remaining expert variables with midpoint normalization
        remaining_vars = expert_vars[:, 2:]  # [batch, remaining_dims]
        if remaining_vars.shape[1] > 0:
            # Calculate midpoints and ranges for remaining variables
            remaining_min = self.physio_min_vals[2:].to(self.device)  # Skip first 2
            remaining_max = self.physio_max_vals[2:].to(self.device)  # Skip first 2
            midpoints = remaining_min + 0.5 * (remaining_max - remaining_min)
            ranges = remaining_max - remaining_min

            # Avoid division by zero
            ranges = torch.clamp(ranges, min=1e-8)

            normalized_remaining = (remaining_vars - midpoints) / ranges

            # Combine normalized parts
            normalized_expert_vars = torch.cat(
                [normalized_first_two, normalized_remaining], dim=-1
            )
        else:
            normalized_expert_vars = normalized_first_two

        return normalized_expert_vars
    def _prepare_encoder_input(self, X, init_states):
        """Prepares the input for the `forward_enc` method based on whether an encoder is used."""
        if self.use_encoder != "none":
            # When using an encoder, provide only the observable variables.
            # The encoder will infer the full latent state.
            X_for_encoder = select_tensor_by_index_list_advanced(X, [0, 1, 2, 3])
        else:
            # When not using an encoder, we manually construct the initial latent state.
            # This state must match the dimensions expected by the SDE dynamics.
            # It consists of the control signal (i_ext, starts at 0) and the expert variables.
            batch_size = X.shape[0]
            zeros_for_i_ext = torch.zeros(
                batch_size, self.SDEnet_out_dims, device=self.device
            )
            expert_inits = init_states[:, : self.expert_latent_dims]
            X_for_encoder = torch.cat([zeros_for_i_ext, expert_inits], dim=1)

        if self.debug:
            print(
                f"[DEBUG] _prepare_encoder_input: use_encoder='{self.use_encoder}', output_shape={X_for_encoder.shape}"
            )

        return X_for_encoder
    def _prepare_no_encoder_initial_state(self, init_states, ic_mask):
        """
        Prepares safe initial conditions for the no-encoder case.
        - For first 5 IC values: use init_states if ic_mask=1, else sample from physio bounds
        - For remaining positions (6-14): always sample from physio bounds
        """
        batch_size = init_states.shape[0]
        num_ic_vars = init_states.shape[-1]  # Should be 5

        # Initialize tensor for all 14 expert variables
        safe_expert_states = torch.zeros(
            batch_size, self.expert_latent_dims, device=init_states.device
        )

        # Use midpoint of physiological ranges for all positions
        # Calculate midpoint between min and max for each variable
        first_two_means = self.first_two_normalization_mu.to(init_states.device)
        remaining_midpoints = self.physio_min_vals[2:] + 0.5 * (
            self.physio_max_vals[2:] - self.physio_min_vals[2:]
        )
        midpoint_states = torch.cat([first_two_means, remaining_midpoints])
        sampled_states = midpoint_states.unsqueeze(0).repeat(batch_size, 1)

        # Start with sampled values for all positions
        safe_expert_states = sampled_states

        # For the first num_ic_vars (should be 5), use actual values where ic_mask=1
        for i in range(min(num_ic_vars, self.expert_latent_dims)):
            safe_expert_states[:, i] = torch.where(
                ic_mask[:, i] == 1,
                init_states[:, i],  # Use actual measured value
                safe_expert_states[:, i],  # Keep sampled value
            )

        if self.debug:
            print("[DEBUG] _prepare_no_encoder_initial_state:")
            print(f"  batch_size: {batch_size}")
            print(f"  safe_expert_states shape: {safe_expert_states.shape}")
            print(
                f"  Used actual IC values: {ic_mask.sum().item()}/{ic_mask.numel()} positions"
            )
            print(f"  Sample values for patient 0: {safe_expert_states[0, :5]}")
            print(f"  Physio bounds - min: {self.physio_min_vals[:5]}")
            print(f"  Physio bounds - max: {self.physio_max_vals[:5]}")

        return safe_expert_states

    def forward_latent(
        self,
        init_latents,
        ts,
        Tx,
        time_to_tx,
        valid_lengths=None,
        med_traj_values=None,
        med_traj_mask=None,
        med_traj_time=None,
    ):
        batch_size, num_samples, state_dim = init_latents.shape

        # Store medication context
        self.current_med_values = med_traj_values
        self.current_med_mask = med_traj_mask
        self.current_med_time = med_traj_time

        if self.use_encoder != "none":
            # init_latents: [batch, samples, 14 expert + encoder_dims]
            zenker_state = init_latents[:, :, :self.expert_latent_dims]  # [batch, samples, 14]
            neural_embedding = init_latents[:, :, self.expert_latent_dims:]  # [batch, samples, encoder_dims]

            # Normalize static context and flatten for f() function access
            static_zenker_vars = zenker_state[:, :, 2:]  # [batch, samples, 12]
            self.normalized_static_zenker = self.normalize_static_zenker_vars(static_zenker_vars)
            self.normalized_static_zenker = self.normalized_static_zenker.reshape(-1, 12)  # [batch*samples, 12]

            self.static_neural_embedding = neural_embedding.reshape(-1,self.encoder_SDENN_dims)  # [batch*samples, encoder_dims]

            # Evolving state: just pressures
            y0 = zenker_state[:, :, :2]  # [batch, samples, 2]


        else:

            # No encoder case

            zenker_state = init_latents  # [batch, samples, 14]

            # Normalize static context and flatten for f() function access

            static_zenker_vars = zenker_state[:, :, 2:]  # [batch, samples, 12]

            self.normalized_static_zenker = self.normalize_static_zenker_vars(static_zenker_vars)

            self.normalized_static_zenker = self.normalized_static_zenker.reshape(-1, 12)  # [batch*samples, 12]

            self.static_neural_embedding = None

            # Evolving state: just pressures

            y0 = zenker_state[:, :, :2]

        y0_flattened = y0.reshape(-1, state_dim)  # [batch*samples, state_dim]

        if self.debug:
            print(f"[DEBUG] NSDE forward_latent: y0_flattened shape = {y0_flattened.shape}")
            print(f"[DEBUG] batch_size={batch_size}, num_samples={num_samples}")

        # Integrate SDE with all samples in parallel
        trajectory_flat = self.sdeint_fn(
            sde=self,
            y0=y0_flattened,
            ts=ts,
            method=self.integration_method,
            dt=self.integration_step_size,
            rtol=self.rtol,
            atol=self.atol
        )

        # Reshape back: [time, batch*samples, state] -> [time, batch, samples, state] -> [batch, samples, time, state]
        time_steps = trajectory_flat.shape[0]
        trajectory = trajectory_flat.view(time_steps, batch_size, num_samples, state_dim)
        trajectory = trajectory.permute(1, 2, 0, 3)  # [batch, samples, time, state]

        # Extract pressure trajectory (first 2 dimensions)
        pressure_traj = trajectory[:, :, :, :2]

        return pressure_traj, torch.zeros(batch_size), None

    def forward_dec(self, latent_out):
        """Apply physiological constraints to pressure outputs"""
        pa = torch.clamp(latent_out[..., 0], min=40.0, max=220.0)
        pv = torch.clamp(latent_out[..., 1], min=0.0, max=39.0)
        return torch.stack([pa, pv], dim=-1)

    def compute_ic_consistency_loss(
        self, predicted_ode_latents_sigmoid, init_states, ic_mask
    ):
        """
        Computes loss between sigmoid of real IC values and already-sigmoided predicted ODE latents
        where we have actual measurements (ic_mask == 1).
        """
        if self.debug:
            print(
                f"[DEBUG] compute_ic_consistency_loss: predicted_shape={predicted_ode_latents_sigmoid.shape}, init_states_shape={init_states.shape}"
            )

        # Get the number of IC variables we have
        num_ic_vars = init_states.shape[-1]

        # predicted_ode_latents are already sigmoided, so use them directly
        sigmoid_predicted = predicted_ode_latents_sigmoid[
            :, :, :num_ic_vars
        ]  # [batch, samples, ic_vars]

        # Apply sigmoid only to the real values
        sigmoid_real = torch.sigmoid(
            init_states.unsqueeze(1).repeat(1, self.num_samples, 1)
        )  # [batch, samples, ic_vars]

        # Expand ic_mask to match dimensions
        ic_mask_expanded = ic_mask.unsqueeze(1).repeat(
            1, self.num_samples, 1
        )  # [batch, samples, ic_vars]

        # Compute MSE loss only where we have real data
        mse_loss = ((sigmoid_predicted - sigmoid_real) ** 2) * ic_mask_expanded

        # Average over samples and sum over features, then average over batch
        # Normalize by number of valid measurements
        valid_count = ic_mask_expanded.sum()
        if valid_count > 0:
            ic_consistency_loss = mse_loss.sum() / valid_count
        else:
            ic_consistency_loss = torch.tensor(0.0, device=predicted_ode_latents.device)

        if self.debug:
            print(
                f"[DEBUG] IC consistency loss: {ic_consistency_loss.item()}, valid_measurements: {valid_count.item()}"
            )

        return ic_consistency_loss

    def compute_factual_loss(self, predicted_traj, true_traj, logqp, mask=None):
        true_traj_expanded = true_traj.unsqueeze(1)
        if self.debug:
            print(
                f"[DEBUG] Hybrid_SDE compute_factual_loss: Shapes: predicted_traj={predicted_traj.shape}, true_traj={true_traj.shape}, logqp_mean={logqp.mean().item() if logqp.numel() > 0 else 'N/A'}"
            )

            print(
                f"[DEBUG] Hybrid_SDE compute_factual_loss: Shapes: predicted_traj={predicted_traj.shape}, true_traj={true_traj.shape}"
            )

            # ADD THESE DEBUG CHECKS:
            print(f"[DEBUG] log_lik_output_scale: {self.log_lik_output_scale}")
            print(
                f"[DEBUG] log_lik_output_scale type: {type(self.log_lik_output_scale)}"
            )
            print(
                f"[DEBUG] predicted_traj contains inf: {torch.isinf(predicted_traj).any()}"
            )
            print(
                f"[DEBUG] predicted_traj contains nan: {torch.isnan(predicted_traj).any()}"
            )
            print(
                f"[DEBUG] predicted_traj min/max: {predicted_traj.min().item()}/{predicted_traj.max().item()}"
            )

            print(
                f"[DEBUG] Hybrid_SDE compute_factual_loss: Shapes: predicted_traj={predicted_traj.shape}, true_traj={true_traj.shape}"
            )
        # Compute log probability
        logpy = distributions.Normal(
            loc=predicted_traj, scale=self.log_lik_output_scale
        ).log_prob(true_traj_expanded)
        if self.debug:
            print(f"Logpy: {logpy[0]}")
            print(f"True traj expanded: {true_traj_expanded[0]}")
            print(f"Predicted traj: {predicted_traj[0]}")
            print(
                f"Mask shape: {mask.shape if mask is not None else None} (expected [B, T, C])"
            )

        # FIXED: Correct normalization
        if mask is not None:
            # Ensure mask dtype/device compatibility
            mask = mask.to(device=logpy.device, dtype=logpy.dtype)
            mask_expanded = mask.unsqueeze(1).expand(
                -1, predicted_traj.shape[1], -1, -1
            )
            logpy = logpy * mask_expanded

            # Sum over time and features
            logpy_sum = logpy.sum(dim=(2, 3))  # [batch, samples]

            # Count total valid elements per sample (time * features)
            valid_count = mask.sum(dim=(1, 2))  # [batch] - total valid elements
            # Avoid division by zero
            valid_count = torch.clamp(valid_count, min=1.0)

            # Normalize correctly
            logpy = logpy_sum / valid_count.unsqueeze(
                1
            )  # [batch, samples] / [batch, 1]
            logpy = logpy.mean(dim=1)  # Average over samples
        else:
            logpy = logpy.sum(dim=(2, 3)).mean(dim=1)

        current_kl_weight = self.kl_scheduler.val
        self.kl_scheduler.step()

        loss = -logpy.mean() + self.KL_weighting_SDE * current_kl_weight * logqp.mean()

        return loss, -logpy.mean(), logqp.mean()

    def _prepare_sde_initial_state(
        self, predicted_ode_latents, neural_embedding, init_states, ic_mask
    ):
        """
        Prepares the initial state for the SDE by combining the interpolated initial
        conditions with the encoder's two-headed output.
        """
        # Number of variables provided by the IC tensor
        num_ic_vars = init_states.shape[-1]

        # Part 1: Take the accurate interpolated values
        init_states_expanded = init_states.unsqueeze(1).repeat(1, self.num_samples, 1)
        if self.debug:
            print(
                f"Interpolated part dims: {init_states_expanded.shape}. Expect: [23 x 7 x 5]"
            )

        ic_mask_expanded = ic_mask.unsqueeze(1).repeat(1, self.num_samples, 1)

        # For the first num_ic_vars variables, use mask to choose between actual and inferred
        expert_part_1 = torch.where(
            ic_mask_expanded == 1,
            init_states_expanded,
            predicted_ode_latents[:, :, :num_ic_vars],
        )

        # For remaining ODE variables, always use inferred values
        expert_part_2 = predicted_ode_latents[:, :, num_ic_vars:]

        # Combine all expert variables
        expert_part = torch.cat([expert_part_1, expert_part_2], dim=-1)

        # By fiat, we replace the inferred ODE vals with the actual init states
        # Part 2: Take the inferred values for the remaining ODE variables from the specific head
        # inferred_part = predicted_ode_latents[:, :, num_ic_vars:]

        # Part 3: The separate neural embedding
        neural_part = neural_embedding

        # Concatenate to form the full initial state for the SDE.
        # Note: The neural part comes *after* the expert ODE part.
        # expert_part = torch.cat([expert_part, inferred_part], dim=-1)
        z1_for_sde = torch.cat([expert_part, neural_part], dim=-1)

        if self.debug:
            print(f"Z1 for SDE dim: {z1_for_sde.shape}. Expect: [23 x 7 x 18]")

        if self.debug:
            print(
                f"  final z1_for_sde shape: {z1_for_sde.shape}, snippet:\n{z1_for_sde[0, 0, :6]}"
            )

        if torch.isnan(z1_for_sde).any():
            print("[ERROR] NaN in final z1_for_sde!")
            nan_locs = torch.where(torch.isnan(z1_for_sde))
            print(f"NaN locations in z1_for_sde: {nan_locs}")

        return z1_for_sde

    def common_step(self, batch, batch_idx):
        if self.debug and batch_idx == 0:
            print(f"[DEBUG] Hybrid_SDE validation_step: batch_idx={batch_idx}")
        if self.dataset == "mimic":
            (
                rd_src,
                rd_times,
                rd_length,
                static_features,
                init_states,
                ic_mask,
                Y,
                Y_mask,
                t_Y,
                med_trajectory_values,
                med_trajectory_mask,
                med_trajectory_time,
                med_context,
            ) = batch
        else:
            raise NotImplementedError(
                "Synthetic path not supported in simplified pipeline"
            )

        batch_size = Y.shape[0]
        ts = t_Y[0, :]

        if self.use_encoder != "none":
            if self.use_encoder == "raindrop":
                src = rd_src.permute(1, 0, 2)
                times = rd_times.permute(1, 0)
                lengths = rd_length
                temporal_embedding, _, _ = self.temporal_encoder(
                    src=src, static=None, times=times, lengths=lengths
                )
                logqp0 = 0
            else:
                raise NotImplementedError(
                    "Only raindrop encoder supported in MIMIC pipeline"
                )

            if self.debug:
                print(f"Static features shape: {static_features.shape}")
            static_embedding = self.static_encoder(static_features)
            fused_embedding = torch.cat([temporal_embedding, static_embedding], dim=-1)
            fused_rep = self.fusion_mlp(fused_embedding)

            predicted_ode_latents_sigmoid = (
                self.ode_latent_head(fused_rep)
                .unsqueeze(1)
                .repeat(1, self.num_samples, 1)
            )
            predicted_ode_latents = self.transform_sigmoid_to_physiological_ranges(
                predicted_ode_latents_sigmoid
            )
            neural_embedding = (
                self.neural_embedding_head(fused_rep)
                .unsqueeze(1)
                .repeat(1, self.num_samples, 1)
            )
            z1_for_sde = self._prepare_sde_initial_state(
                predicted_ode_latents, neural_embedding, init_states, ic_mask
            )

            ic_consistency_loss = self.compute_ic_consistency_loss(
                predicted_ode_latents_sigmoid=predicted_ode_latents_sigmoid,
                init_states=init_states,
                ic_mask=ic_mask,
            )
        else:
            initial_condition = self._prepare_no_encoder_initial_state(
                init_states, ic_mask
            )
            z1_for_sde = initial_condition.unsqueeze(1).repeat(1, self.num_samples, 1)
            logqp0 = 0
            ic_consistency_loss = 0

        valid_lengths = (Y_mask.sum(dim=2) > 0).sum(dim=1)

        # Attach precomputed med_context for fast per-step indexing
        self.current_med_context = med_context if med_context is not None else None

        latent_traj, logqp_path, i_ext_path = self.forward_latent(
            init_latents=z1_for_sde,
            ts=ts,
            Tx=torch.ones(batch_size, device=self.device),
            time_to_tx=torch.zeros(batch_size, device=self.device),
            valid_lengths=valid_lengths,
            med_traj_values=med_trajectory_values,
            med_traj_mask=med_trajectory_mask,
            med_traj_time=med_trajectory_time,
        )

        decoded_traj = self.forward_dec(latent_traj)

        # Mask for loss computation (already respects valid lengths)
        combined_mask = Y_mask
        if self.debug:
            print(
                f"Y_mask stats: min={Y_mask.min()}, max={Y_mask.max()}, mean={Y_mask.mean()}"
            )
        if self.debug:
            print(f"Combined mask sum per sample: {combined_mask.sum(dim=(1, 2))}")
        if self.debug:
            print(
                f"Valid timesteps per sample: {combined_mask.sum(dim=(1, 2)) / Y.shape[-1]}"
            )

        total_logqp = logqp0 + logqp_path
        loss, nll, kl_div = self.compute_factual_loss(
            predicted_traj=decoded_traj,
            true_traj=Y,
            logqp=total_logqp,
            mask=combined_mask,
        )

        total_loss = loss + self.ic_consistency_weight * ic_consistency_loss

        # Optional control TV/L2 smoothness loss over mean path: mean ||u_t - u_{t-1}||^2
        if getattr(self, "use_control_tv_loss", False):
            try:
                u = i_ext_path
                if u is not None and u.shape[1] > 1:
                    u_mean = u.mean(1)  # [B,T,D]
                    du = u_mean[:, 1:, :] - u_mean[:, :-1, :]
                    tv_loss = (du ** 2).mean()
                    total_loss = total_loss + self.control_tv_weight * tv_loss
            except Exception:
                pass

        return {
            "loss": loss,
            "nll": nll,
            "kl_div": kl_div,
            "total_loss": total_loss,
            "ic_consistency_loss": ic_consistency_loss,
            "decoded_traj": decoded_traj,
            "Y": Y,
            "combined_mask": combined_mask,
            "i_ext_path": i_ext_path,
            "z1_for_sde": z1_for_sde,
        }

    def training_step(self, batch, batch_idx):
        if self.debug and batch_idx == 0:
            print(f"[DEBUG] Hybrid_SDE validation_step: batch_idx={batch_idx}")
        result = self.common_step(batch, batch_idx)

        total_loss = result["total_loss"]
        loss = result["loss"]
        ic_consistency_loss = result["ic_consistency_loss"]
        nll = result["nll"]
        kl_div = result["kl_div"]

        # If graph is degenerate (e.g., encoder='none' and force_no_controls=True), attach a zero-sum reg to create a grad path
        if not total_loss.requires_grad:
            if self.debug:
                print(
                    "[WARN] total_loss has no grad_fn. Likely no trainable path active (encoder='none' with force_no_controls=True)."
                )
            zero_reg = None
            for p in self.parameters():
                if p.requires_grad:
                    zero_reg = (
                        (0.0 * p.sum())
                        if zero_reg is None
                        else zero_reg + 0.0 * p.sum()
                    )
            if zero_reg is None:
                zero_reg = 0.0 * total_loss
            total_loss = total_loss + zero_reg

        # Log metrics
        self.log(
            "train_total_loss",
            total_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "train_main_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "train_ic_consistency_loss",
            ic_consistency_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "train_NLL", nll, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        self.log(
            "train_KL", kl_div, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )

        return total_loss

    def validation_step(self, batch, batch_idx):
        if self.debug and batch_idx == 0:
            print(f"[DEBUG] Hybrid_SDE validation_step: batch_idx={batch_idx}")

        result = self.common_step(batch, batch_idx)

        # Log the individual components
        self.log(
            "val_total_loss",
            result["total_loss"],
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "val_main_loss",
            result["loss"],
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "val_ic_consistency_loss",
            result["ic_consistency_loss"],
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "val_NLL",
            result["nll"],
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "val_KL",
            result["kl_div"],
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        return {
            "val_loss": result["loss"],
            "val_nll": result["nll"],
            "val_kl": result["kl_div"],
            "decoded_traj": result["decoded_traj"],
            "true_traj": result["Y"],
            "mask": result["combined_mask"],
        }

    def test_step(self, batch, batch_idx):
        if self.debug and batch_idx == 0:
            print(f"[DEBUG] Hybrid_SDE validation_step: batch_idx={batch_idx}")

        result = self.common_step(batch, batch_idx)

        total_loss = result["total_loss"]
        loss = result["loss"]
        nll = result["nll"]
        kl_div = result["kl_div"]

        # Log the individual components
        self.log(
            "test_total_loss",
            total_loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "test_main_loss",
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "test_ic_consistency_loss",
            result["ic_consistency_loss"],
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "test_NLL", nll, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )
        self.log(
            "test_KL", kl_div, on_step=False, on_epoch=True, prog_bar=True, logger=True
        )

        # Compute additional test metrics
        decoded_traj = result["decoded_traj"]
        Y = result["Y"]
        combined_mask = result["combined_mask"]

        with torch.no_grad():
            mse_per_sample = ((decoded_traj.mean(1) - Y) ** 2) * combined_mask
            mae_per_sample = torch.abs(decoded_traj.mean(1) - Y) * combined_mask

            valid_elements = combined_mask.sum()
            valid_mse = mse_per_sample.sum() / valid_elements
            valid_mae = mae_per_sample.sum() / valid_elements

            self.log(
                "test_mse",
                valid_mse,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )
            self.log(
                "test_mae",
                valid_mae,
                on_step=False,
                on_epoch=True,
                prog_bar=True,
                logger=True,
            )

        if batch_idx < 3:
            self.plot_nature_style_with_uncertainty(
                decoded_traj, Y, combined_mask, batch_idx
            )

        return_dict = {
            "test_loss": loss,
            "test_nll": nll,
            "test_kl": kl_div,
            "test_mse": valid_mse,
            "test_mae": valid_mae,
            "decoded_traj": decoded_traj,
            "true_traj": Y,
            "mask": combined_mask,
        }
        return return_dict




    def on_test_epoch_end(self):
        """Log final test metrics to wandb"""
        if self.log_wandb:
            # Get the logged metrics
            test_results = {
                "final_test_loss": self.trainer.callback_metrics.get(
                    "test_total_loss", 0
                ),
                "final_test_mse": self.trainer.callback_metrics.get("test_mse", 0),
                "final_test_mae": self.trainer.callback_metrics.get("test_mae", 0),
                "final_test_nll": self.trainer.callback_metrics.get("test_NLL", 0),
                "final_test_kl": self.trainer.callback_metrics.get("test_KL", 0),
            }

            # Log final summary
            wandb.log(test_results)

            # Create a summary table
            test_summary = [
                ["Metric", "Value"],
                ["Total Loss", f"{test_results['final_test_loss']:.4f}"],
                ["MSE", f"{test_results['final_test_mse']:.4f}"],
                ["MAE", f"{test_results['final_test_mae']:.4f}"],
                ["NLL", f"{test_results['final_test_nll']:.4f}"],
                ["KL Divergence", f"{test_results['final_test_kl']:.4f}"],
            ]

            wandb.log(
                {
                    "test_summary_table": wandb.Table(
                        data=test_summary[1:], columns=test_summary[0]
                    )
                }
            )

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

        scheduler = {
            "monitor": "train_total_loss",
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer=optimizer, mode="min", factor=0.5, patience=50
            ),
        }
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def _setup_plot_style(self):
        """Shared plotting style configuration"""
        plt.rcParams.update(
            {
                "font.size": 8,
                "font.family": "sans-serif",
                "axes.linewidth": 0.8,
                "axes.spines.top": False,
                "axes.spines.right": False,
                "legend.frameon": True,
            }
        )

        # Define consistent color scheme
        colors = {
            "arterial_true": "#2E86AB",
            "arterial_pred": "#2E86AB",
            "arterial_baseline": "#2E86AB",
            "venous_true": "#A23B72",
            "venous_pred": "#A23B72",
            "venous_baseline": "#A23B72",
            "control1": "#F18F01",  # Orange
            "control2": "#C73E1D",  # Red
            "derivative1": "#F18F01",
            "derivative2": "#C73E1D",
        }
        return colors

    def plot_nature_style_with_uncertainty(
            self, predictions_full, targets, combined_mask, batch_idx
    ):
        """Nature-style plots with uncertainty bands around predictions"""
        import os

        import matplotlib.pyplot as plt
        import numpy as np

        try:
            plt.switch_backend("Agg")
        except Exception:
            pass

        colors = self._setup_plot_style()
        os.makedirs(os.path.join(self.train_dir, "nature_plots"), exist_ok=True)

        pred_mean = predictions_full.mean(1).detach()
        pred_std = predictions_full.std(1).detach()
        targets = targets.detach() if targets.requires_grad else targets

        for patient_idx in range(min(3, predictions_full.shape[0])):
            patient_mask = combined_mask[patient_idx]
            time_seconds = (torch.arange(patient_mask.shape[0]).cpu().numpy()) * 10
            pred_mean_patient = pred_mean[patient_idx].detach().cpu().numpy()
            pred_std_patient = pred_std[patient_idx].detach().cpu().numpy()
            true_patient = targets[patient_idx].detach().cpu().numpy()

            arterial_mask = patient_mask[:, 0].cpu().numpy().astype(bool)
            venous_mask = patient_mask[:, 1].cpu().numpy().astype(bool)

            # Mask out invalid points per channel (plot with gaps where missing)
            arterial_true = true_patient[:, 0].copy()
            arterial_pred = pred_mean_patient[:, 0].copy()
            arterial_std = pred_std_patient[:, 0].copy()
            arterial_true[~arterial_mask] = np.nan
            arterial_pred[~arterial_mask] = np.nan
            arterial_std[~arterial_mask] = np.nan

            venous_true = true_patient[:, 1].copy()
            venous_pred = pred_mean_patient[:, 1].copy()
            venous_std = pred_std_patient[:, 1].copy()
            venous_true[~venous_mask] = np.nan
            venous_pred[~venous_mask] = np.nan
            venous_std[~venous_mask] = np.nan

            fig, ax = plt.subplots(figsize=(7, 5))
            ax.set_xlim(0, 1200)

            # Use consistent colors and styling
            ax.plot(
                time_seconds,
                arterial_true,
                color=colors["arterial_true"],
                linestyle="-",
                linewidth=2.0,
                label="Arterial pressure (true)",
                zorder=3,
            )
            ax.plot(
                time_seconds,
                arterial_pred,
                color=colors["arterial_pred"],
                linestyle="--",
                linewidth=1.5,
                label="Arterial pressure (predicted)",
                alpha=0.9,
                zorder=2,
            )
            ax.fill_between(
                time_seconds,
                arterial_pred - arterial_std,
                arterial_pred + arterial_std,
                color=colors["arterial_pred"],
                alpha=0.2,
                zorder=1,
            )

            ax.plot(
                time_seconds,
                venous_true,
                color=colors["venous_true"],
                linestyle="-",
                linewidth=2.0,
                label="Venous pressure (true)",
                zorder=3,
            )
            ax.plot(
                time_seconds,
                venous_pred,
                color=colors["venous_pred"],
                linestyle="--",
                linewidth=1.5,
                label="Venous pressure (predicted)",
                alpha=0.9,
                zorder=2,
            )
            ax.fill_between(
                time_seconds,
                venous_pred - venous_std,
                venous_pred + venous_std,
                color=colors["venous_pred"],
                alpha=0.2,
                zorder=1,
            )

            ax.set_xlabel("Time (seconds)", fontweight="bold")
            ax.set_ylabel("Pressure (mmHg)", fontweight="bold")
            epoch_tag = f"epoch{int(getattr(self, 'current_epoch', 0)):03d}_step{int(getattr(self, 'global_step', 0)):06d}"
            ax.set_title(
                f"{epoch_tag} – Patient {patient_idx} (Batch {batch_idx})",
                fontweight="bold",
            )
            ax.legend(
                loc="upper right", fancybox=False, facecolor="white", framealpha=1.0
            )
            ax.grid(True, alpha=0.2)

            if self.log_wandb:
                wandb.log(
                    {
                        f"uncertainty_plot_batch_{batch_idx}_patient_{patient_idx}": wandb.Image(
                            plt
                        )
                    }
                )
            else:
                out_path = os.path.join(
                    self.train_dir,
                    f"nature_plots/{epoch_tag}_patient{patient_idx}_batch{batch_idx}_uncertainty.png",
                )
                plt.savefig(out_path, dpi=300, bbox_inches="tight")
                print(f"[PLOT] Saved: {out_path}")
            plt.close()
