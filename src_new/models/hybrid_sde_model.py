import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
import torch
import torchsde
import wandb
from graph_control_net import GraphControlNet
from lightning import LightningModule
from raindrop import Raindrop_v2
from torch import distributions, nn
from torch.special import erf
from ZenkerModel import ZenkerODE

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "utils"))
from train_utils import zenker_derivatives
from utils_beta import (
    CV_params,
    CV_params_divisors,
    LinearScheduler,
    MLPSimple,
    _stable_division,
    select_tensor_by_index_list_advanced,
)

# <<< Global DEBUG flag for model_beta.py, to be set by instance >>>
# This is more of a placeholder if a module-level default is ever needed,
# but instance-level self.debug passed from main_beta.py is the primary control.
DEBUG = False


class Hybrid_SDE(LightningModule):
    def __init__(
        self,
        use_encoder,
        start_dec_at_treatment,
        variational_sampling,
        # Encoder
        encoder_input_dim,
        encoder_hidden_dim,
        encoder_SDENN_dims,
        expert_latent_dims,
        encoder_num_layers,
        variational_encoder,
        encoder_w_time,
        encoder_reverse_time,
        use_2_5std_encoder_minmax,
        n_medications,
        encoder_context_len,
        # New static fusion params
        static_input_dim,
        static_hidden_dim,
        fusion_hidden_dim,
        # SDE params
        normalise_for_SDENN,
        prior_tx_sigma,
        prior_tx_mu,
        self_reverting_prior_control,
        SDE_input_state,
        include_time,
        theta,
        SDE_control_weighting,
        # Control smoothness/inductive-bias flags
        use_control_lowpass,
        control_lowpass_tau,
        use_control_tv_loss,
        control_tv_weight,
        override_control_scales,
        # SDE model params
        num_samples,
        SDEnet_hidden_dim,
        SDEnet_depth,
        SDEnet_out_dims,
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
        # loss
        KL_weighting_SDE,
        # admin
        train_dir,
        learning_rate,
        log_wandb,
        adjoint,
        plot_every,
        batch_size,
        dataset,
        test_zenker,
        debug=False,  # <<< Add debug flag >>>
        # Diagnostics
        force_zenker_defaults=False,
        force_no_controls=False,
        # Plotting
        plot_outputs_train=True,
        # plotting behavior (overlays auto-enabled for num_samples>1)
        plot_control_samples=False,
        # regularization
        control_energy_weight: float = 1e-4,
        # controller selection
        controller_type: str = "mlp",  # or "gat"
        gat_heads: int = 4,
        gat_layers: int = 2,
        gat_hidden: int = 128,
        gat_dropout: float = 0.0,
    ):
        super().__init__()
        self.debug = debug  # <<< Store debug flag >>>
        global DEBUG
        DEBUG = (
            self.debug
        )  # <<< Update module-level DEBUG if needed, primarily use self.debug >>>

        if self.debug:
            print(
                f"[DEBUG] Hybrid_SDE __init__: Initializing... adjoint={adjoint}, use_encoder={use_encoder}, normalise_for_SDENN={normalise_for_SDENN}"
            )

        self.noise_type = "diagonal"  # required
        self.sde_type = "ito"  # required
        self.sdeint_fn = torchsde.sdeint_adjoint if adjoint else torchsde.sdeint

        ### ADMIN
        self.train_dir = train_dir
        self.learning_rate = learning_rate
        self.log_wandb = log_wandb
        self.plot_every = plot_every

        ### Bifurcation options
        self.use_encoder = use_encoder
        self.normalise_for_SDENN = normalise_for_SDENN
        self.start_dec_at_treatment = start_dec_at_treatment
        self.variational_sampling = variational_sampling
        self.SDE_input_state = SDE_input_state
        self.include_time = include_time

        ### Encoder model
        self.encoder_SDENN_dims = encoder_SDENN_dims
        self.encoder_output_dim = encoder_SDENN_dims + expert_latent_dims

        self.variational_encoder = variational_encoder
        self.use_2_5std_encoder_minmax = use_2_5std_encoder_minmax

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

        ### PRIOR PARAMS
        self.self_reverting_prior_control = self_reverting_prior_control
        self.prior_tx_sigma = prior_tx_sigma
        self.prior_tx_mu = prior_tx_mu

        # sigma_values = torch.tensor(list(CV_params_prior_sigma.values())).float()
        # sigma_values = sigma_values[:expert_latent_dims].view(1, -1)
        # self.register_buffer('sigma', sigma_values.clone())
        self.sigma = (
            torch.tensor(self.prior_tx_sigma, dtype=torch.float32)
            .to(self.device)
            .unsqueeze(0)
        )

        self.theta = (
            torch.tensor(theta, dtype=torch.float)
            .clone()
            .view(1, -1)
            .repeat(1, expert_latent_dims)
        )

        ### LATENT MODEL

        self.num_samples = num_samples
        self.expert_latent_dims = expert_latent_dims
        self.CV_params = CV_params

        self.divisors = torch.tensor(
            [
                CV_params_divisors[key]
                for key in [
                    "pa",
                    "pv",
                    "s",
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
            ],
            dtype=torch.float32,
        )

        self.SDEnet_hidden_dim = SDEnet_hidden_dim
        self.SDEnet_depth = SDEnet_depth
        self.SDEnet_out_dims = SDEnet_out_dims
        self.SDE_control_weighting = SDE_control_weighting
        # Smoothness flags
        self.use_control_lowpass = bool(use_control_lowpass)
        self.control_lowpass_tau = float(control_lowpass_tau)
        self.use_control_tv_loss = bool(use_control_tv_loss)
        self.control_tv_weight = float(control_tv_weight)
        override_scales = override_control_scales
        # Blend between pure low-pass derivative and raw control derivative
        # 1.0 -> pure low-pass du=(u_hat-u)/tau; 0.0 -> raw du=u_hat (interpreted as direct derivative)
        self.control_lowpass_blend = 0.1

        net_input_dims = (
            self.encoder_output_dim
            if SDE_input_state == "full"
            else self.encoder_output_dim - len(encoder_input_dim)
        )
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

        # Append physics-derived feature count (ΔP, r_tpr, f_hr, F, dpa_base, dpv_base, sigma, s_dot)
        self.physics_feat_dims = 8
        net_input_dims = net_input_dims + self.physics_feat_dims

        activations = {"relu": nn.ReLU(), "tanh": nn.Tanh(), "none": None}
        final_activation_real = activations[final_activation.lower()]

        # TODO change net input dims to be 14 + number of meds if there is no encoder, else encoder dim + 14 + meds

        self.SDEnet = MLPSimple(
            input_dim=net_input_dims,
            output_dim=SDEnet_out_dims,
            hidden_dim=SDEnet_hidden_dim,
            depth=SDEnet_depth,
            activations=[nn.Tanh() for _ in range(SDEnet_depth)],
            final_activation=final_activation_real,
            use_batch_norm=use_batch_norm,
            debug=self.debug,
        )

        # Per-head control scales applied to SDE NN outputs (post-final-layer)
        # Head 0 (dpv_dt): ~0.1 mmHg/s
        # Head 1 (dsv_dt): ~0.02 units per time step
        # Head 2 (dt_ca): ~0.01 units per time step
        # Head 3 (dt_r_tpr_mod): ~0.01 units per time step
        default_scales = torch.tensor([0.1, 0.02, 0.01, 0.01], dtype=torch.float32)
        if SDEnet_out_dims != len(default_scales):
            # Fallback: repeat or truncate to match dims
            repeats = (SDEnet_out_dims + len(default_scales) - 1) // len(default_scales)
            default_scales = default_scales.repeat(repeats)[:SDEnet_out_dims]
        # Allow override via CLI
        if isinstance(override_scales, str) and override_scales.strip() != "":
            try:
                pieces = [float(x) for x in override_scales.split(",")]
                ov = torch.tensor(pieces, dtype=torch.float32)
                if ov.numel() < SDEnet_out_dims:
                    reps = (SDEnet_out_dims + ov.numel() - 1) // ov.numel()
                    ov = ov.repeat(reps)[:SDEnet_out_dims]
                else:
                    ov = ov[:SDEnet_out_dims]
                default_scales = ov
                if self.debug:
                    print(
                        f"[DEBUG] Override control scales -> {default_scales.tolist()}"
                    )
            except Exception as e:
                if self.debug:
                    print(
                        f"[WARN] Failed to parse override_control_scales '{override_scales}': {e}"
                    )
        self.register_buffer("control_scales", default_scales)

        # Initialization trick from Glow.
        # self.SDEnet.output_layer[0].weight.data.fill_(0.)
        # self.SDEnet.output_layer[0].bias.data.fill_(0.)

        ### DECODER
        self.decoder_output_dims = decoder_output_dims
        self.normalised_data = normalised_data
        self.dataset = dataset

        self.integration_step_size = integration_step_size
        self.integration_method = integration_method
        self.rtol = rtol
        self.atol = atol
        self.integration_adaptive = integration_adaptive
        self.test_zenker = test_zenker
        # Diagnostics flags
        self.force_zenker_defaults = force_zenker_defaults
        self.force_no_controls = force_no_controls
        # Plotting flags
        self.plot_outputs_train = plot_outputs_train
        self.plot_control_samples = plot_control_samples
        # Regularization weight for control energy λ·E[||u||^2]
        self.control_energy_weight = float(control_energy_weight)
        # Controller config
        self.controller_type = controller_type.lower()
        self.gat_heads = int(gat_heads)
        self.gat_layers = int(gat_layers)
        self.gat_hidden = int(gat_hidden)
        self.gat_dropout = float(gat_dropout)

        # Debug helper state
        self._forward_hook_handles = []
        self._last_sdnet_io_stats = None

        ### LOSS
        self.MSE_loss = nn.MSELoss(reduction="none")
        self.log_lik_output_scale = log_lik_output_scale
        self.KL_weighting_SDE = KL_weighting_SDE
        self.kl_scheduler = LinearScheduler(
            start=70, iters=600, startval=1.0, endval=0.01
        )
        self.save_hyperparameters()
        if self.debug:
            print(
                f"[DEBUG] Hybrid_SDE __init__: Initialization complete. Encoder output dim: {self.encoder_output_dim}, SDEnet input dims: {net_input_dims}"
            )

            # Register a single forward hook on SDEnet to capture I/O stats
            def _sdnet_hook(module, inputs, outputs):
                try:
                    step_val = int(getattr(self, "global_step", 0))
                    if step_val % 10 != 0:
                        return
                except Exception:
                    pass
                x = inputs[0] if isinstance(inputs, (list, tuple)) else inputs
                y = outputs
                self._last_sdnet_io_stats = {
                    "x_shape": tuple(x.shape),
                    "x_mean": float(x.mean().detach().cpu()),
                    "x_std": float(x.std().detach().cpu()),
                    "x_min": float(x.min().detach().cpu()),
                    "x_max": float(x.max().detach().cpu()),
                    "x_nan": bool(torch.isnan(x).any().item()),
                    "x_inf": bool(torch.isinf(x).any().item()),
                    "y_shape": tuple(y.shape),
                    "y_mean": float(y.mean().detach().cpu()),
                    "y_std": float(y.std().detach().cpu()),
                    "y_min": float(y.min().detach().cpu()),
                    "y_max": float(y.max().detach().cpu()),
                    "y_nan": bool(torch.isnan(y).any().item()),
                    "y_inf": bool(torch.isinf(y).any().item()),
                }
                print(
                    f"[DEBUG] SDEnet hook: x_shape={self._last_sdnet_io_stats['x_shape']} x_mean={self._last_sdnet_io_stats['x_mean']:.4e} x_std={self._last_sdnet_io_stats['x_std']:.4e} | y_shape={self._last_sdnet_io_stats['y_shape']} y_mean={self._last_sdnet_io_stats['y_mean']:.4e} y_std={self._last_sdnet_io_stats['y_std']:.4e}"
                )

            try:
                handle = self.SDEnet.register_forward_hook(_sdnet_hook)
                self._forward_hook_handles.append(handle)
            except Exception:
                pass

        # plotting:
        self.mse_data_factual = [[] for _ in range(batch_size)]
        self.mse_data_cf = [[] for _ in range(batch_size)]

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

        # Optional GAT controller (AB test)
        node_feat_dim = 1 + self.physics_feat_dims + (2 if self.include_time else 0)
        self.n_medications = int(n_medications)
        if self.n_medications > 0:
            self.med_proj = nn.Linear(self.n_medications * 2, 8)
            node_feat_dim += 8
        else:
            self.med_proj = None

        if self.controller_type == "gat":
            self.gat_controller = GraphControlNet(
                node_feature_dim=node_feat_dim,
                hidden_dim=self.gat_hidden,
                num_layers=self.gat_layers,
                num_heads=self.gat_heads,
                dropout=self.gat_dropout,
                device=self.device,
            )
        else:
            self.gat_controller = None

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
                t,
                dtype=self.current_med_time.dtype,
                device=self.current_med_time.device,
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
        last_rates = torch.where(
            has_valid_data, last_rates, torch.zeros_like(last_rates)
        )
        recency_weights = torch.where(
            has_valid_data, recency_weights, torch.ones_like(recency_weights)
        )
        result = torch.stack([last_rates, recency_weights], dim=-1).flatten(start_dim=1)
        samples_per_batch = expanded_batch_size // batch_size
        return result.repeat_interleave(samples_per_batch, dim=0)

    def _precompute_med_context(self, ts: torch.Tensor) -> None:
        """Precompute medication context [rate, recency] for each time step.

        Creates `self.med_context_by_time` with shape [B, T, 2*M] and stores
        time grid params `_t0`, `_dt_grid`, `_time_len` for fast indexing.
        """
        B, T, M = self.current_med_values.shape
        device = self.current_med_values.device
        dtype = self.current_med_values.dtype

        ts_dev = ts.to(device=device, dtype=dtype)
        self._t0 = ts_dev[0]
        self._time_len = int(ts_dev.shape[0])
        self._dt_grid = (
            (ts_dev[1] - ts_dev[0]).clamp_min(1e-6)
            if T > 1
            else torch.tensor(1.0, device=device, dtype=dtype)
        )

        valid = self.current_med_mask > 0  # [B,T,M]
        idxs = torch.arange(T, device=device).view(1, T, 1).expand(B, T, M)
        valid_idxs = torch.where(valid, idxs + 1, torch.zeros_like(idxs))  # +1 sentinel
        last_idx_plus1 = torch.cummax(valid_idxs, dim=1).values  # [B,T,M]
        has_valid = last_idx_plus1 > 0
        last_idx = torch.clamp(last_idx_plus1 - 1, min=0).to(torch.long)

        last_rates = torch.gather(
            self.current_med_values, dim=1, index=last_idx
        )  # [B,T,M]
        time_bt = self.current_med_time.unsqueeze(-1).expand(B, T, M)
        last_times = torch.gather(time_bt, dim=1, index=last_idx)  # [B,T,M]

        t_grid = ts_dev.view(1, T, 1).expand(B, T, M)
        time_since = t_grid - last_times
        recency = torch.clamp(time_since / 1200.0 - 1.0 / 1200.0, min=0.0)

        last_rates = torch.where(has_valid, last_rates, torch.zeros_like(last_rates))
        recency = torch.where(has_valid, recency, torch.ones_like(recency))

        ctx = torch.stack([last_rates, recency], dim=-1)  # [B,T,M,2]
        self.med_context_by_time = ctx.view(B, T, 2 * M).contiguous()

    def apply_SDE_fun(self, t, y):
        """
        Normalise data and add time information (if the appropriate options have been set).
        Args:
            t:
            y:

        a) all of the initial ODE variables at t0 (initial conditions) √

        b) for the ODE variables that are dynamic, the ones at time t √

        c) the control at time t √

        d) the med trajectories: these provide at each time t which medication and which dose is given. √

        e) the physio context + med context tensors: this is the physio variables averaged per 10 mins in the hour before t0, so k x 6 if k is the number of physio vars.



        Returns:

        """

        batch_size = y.shape[0]

        if self.debug:
            t_idx = torch.round((t.to(self._t0) - self._t0) / self._dt_grid).to(
                torch.long
            )
            if (t_idx.remainder(10) == 0).item():
                print(
                    f"[DEBUG] Hybrid_SDE apply_SDE_fun: t_idx={int(t_idx)}, y_shape={y.shape}, normalise_for_SDENN={self.normalise_for_SDENN}, include_time={self.include_time}, SDE_input_state={self.SDE_input_state}"
                )

        if self.normalise_for_SDENN:
            SDNN_expert_input_state = self.normalise_sde_inputs(
                y[
                    :,
                    self.SDEnet_out_dims : self.expert_latent_dims
                    + self.SDEnet_out_dims,
                ]
            )
        else:
            SDNN_expert_input_state = y[
                :, self.SDEnet_out_dims : self.expert_latent_dims + self.SDEnet_out_dims
            ] / self.divisors.to(self.device)

        # Always get raw expert variables (physical units) for physics features
        expert_raw = y[
            :, self.SDEnet_out_dims : self.SDEnet_out_dims + self.expert_latent_dims
        ]
        pa = expert_raw[:, 0:1]
        pv = expert_raw[:, 1:2]
        s = expert_raw[:, 2:3]
        sv = expert_raw[:, 3:4]
        r_tpr_mod = expert_raw[:, 4:5]
        f_hr_max = expert_raw[:, 5:6]
        f_hr_min = expert_raw[:, 6:7]
        r_tpr_max = expert_raw[:, 7:8]
        r_tpr_min = expert_raw[:, 8:9]
        ca = expert_raw[:, 9:10]
        cv = expert_raw[:, 10:11]
        k_width = expert_raw[:, 11:12]
        p_aset = expert_raw[:, 12:13]
        tau = expert_raw[:, 13:14]

        # Derived physiology
        fhr = s * (f_hr_max - f_hr_min) + f_hr_min
        r_tpr = s * (r_tpr_max - r_tpr_min) + r_tpr_min + r_tpr_mod
        dP = pa - pv
        F = -dP / torch.clamp(r_tpr, min=1e-9) + sv * fhr
        dpa_base = F / torch.clamp(ca, min=1e-9)
        dpv_base = (dP / (torch.clamp(cv, min=1e-9) * torch.clamp(r_tpr, min=1e-9))) - (
            sv * fhr / torch.clamp(cv, min=1e-9)
        )
        sigma = 1.0 / (1.0 + torch.exp(-k_width * (pa - p_aset)))
        s_dot = (1.0 / torch.clamp(tau, min=1e-9)) * (1.0 - sigma - s)

        physics_feats = torch.cat(
            [dP, r_tpr, fhr, F, dpa_base, dpv_base, sigma, s_dot], dim=-1
        )
        if torch.isnan(physics_feats).any() or torch.isinf(physics_feats).any():
            raise RuntimeError("Non-finite values in physics_feats. Aborting for debugging.")
        # Normalize physics features if we normalize SDEnn inputs (zero-mean, unit-std across batch)
        if self.normalise_for_SDENN:
            pf_mean = physics_feats.mean(dim=0, keepdim=True)
            pf_std = physics_feats.std(dim=0, keepdim=True).clamp_min(1e-6)
            physics_feats = (physics_feats - pf_mean) / pf_std

        if self.debug:
            try:
                print(
                    f"  [DBG] expert_raw: mean={expert_raw.mean().item():.4e} std={expert_raw.std().item():.4e} min={expert_raw.min().item():.4e} max={expert_raw.max().item():.4e}"
                )
                print(
                    f"  [DBG] SDNN_expert_input_state: mean={SDNN_expert_input_state.mean().item():.4e} std={SDNN_expert_input_state.std().item():.4e} min={SDNN_expert_input_state.min().item():.4e} max={SDNN_expert_input_state.max().item():.4e}"
                )
                print(
                    f"  [DBG] physics_feats: mean={physics_feats.mean().item():.4e} std={physics_feats.std().item():.4e} min={physics_feats.min().item():.4e} max={physics_feats.max().item():.4e}"
                )
            except Exception:
                pass

        # print('SDNN_expert_input_state', SDNN_expert_input_state.shape, SDNN_expert_input_state[0, :])

        if self.include_time:
            # Positional encoding in transformers for time-inhomogeneous posterior
            sde_latent_times = torch.full_like(y[:, 0], fill_value=t).unsqueeze(1)
            sin_time = torch.sin(sde_latent_times)
            cos_time = torch.cos(sde_latent_times)

            if self.SDE_input_state == "full":
                input_state = torch.cat(
                    [
                        SDNN_expert_input_state,
                        y[:, self.SDEnet_out_dims + self.expert_latent_dims :],
                    ],
                    dim=-1,
                )
                SDE_NN_input = torch.cat((sin_time, cos_time, input_state), dim=-1)

            elif self.SDE_input_state == "latents":
                input_state = torch.cat(
                    [
                        SDNN_expert_input_state[:, 2:],
                        y[:, self.SDEnet_out_dims + self.expert_latent_dims :],
                    ],
                    dim=-1,
                )
                SDE_NN_input = torch.cat((sin_time, cos_time, input_state), dim=-1)

        else:
            if self.SDE_input_state == "full":
                SDE_NN_input = torch.cat(
                    [
                        SDNN_expert_input_state,
                        y[:, self.SDEnet_out_dims + self.expert_latent_dims :],
                    ],
                    dim=-1,
                )

            elif self.SDE_input_state == "latents":
                SDE_NN_input = torch.cat(
                    [
                        SDNN_expert_input_state[:, 2:],
                        y[:, self.SDEnet_out_dims + self.expert_latent_dims :],
                    ],
                    dim=-1,
                )

        # Medication context (used by either controller)
        if (
            hasattr(self, "current_med_context")
            and self.current_med_context is not None
        ):
            # Index precomputed med context by time step
            if not hasattr(self, "_t0"):
                self._t0 = torch.tensor(0.0, device=self.device)
            if not hasattr(self, "_dt_grid"):
                self._dt_grid = torch.tensor(
                    float(self.integration_step_size), device=self.device
                )
            t_idx = torch.round((t.to(self._t0) - self._t0) / self._dt_grid).to(
                torch.long
            )
            t_idx = torch.clamp(t_idx, 0, self.current_med_context.shape[1] - 1)
            med_context = self.current_med_context[:, t_idx, :]
            # expand over samples if needed
            samples_per_batch = batch_size // med_context.shape[0]
            med_context = med_context.repeat_interleave(samples_per_batch, dim=0)
        else:
            med_context = self.get_medication_context(
                t, batch_size
            )  # (batch, 2*n_meds)
        if torch.isnan(med_context).any() or torch.isinf(med_context).any():
            raise RuntimeError("Non-finite values in med_context. Aborting for debugging.")
        if self.normalise_for_SDENN and med_context.numel() > 0:
            mc_mean = med_context.mean(dim=0, keepdim=True)
            mc_std = med_context.std(dim=0, keepdim=True).clamp_min(1e-6)
            med_context = (med_context - mc_mean) / mc_std

        if self.debug:
            try:
                print(
                    f"  [DBG] med_context: mean={med_context.mean().item():.4e} std={med_context.std().item():.4e} min={med_context.min().item():.4e} max={med_context.max().item():.4e}"
                )
            except Exception:
                pass

        # If GAT controller is selected, build node-wise features and branch here
        if (
            self.controller_type == "gat"
            and getattr(self, "gat_controller", None) is not None
        ):
            # Per-node scalar values for the 14 expert variables
            node_value = expert_raw.unsqueeze(-1)  # [B, 14, 1]
            # Broadcast physics features to all nodes
            phys_b = physics_feats.unsqueeze(1).repeat(
                1, expert_raw.shape[1], 1
            )  # [B,14,8]
            parts = [node_value, phys_b]
            # Optional time features
            if self.include_time:
                time_feats = torch.cat([sin_time, cos_time], dim=-1)  # [B,2]
                time_b = time_feats.unsqueeze(1).repeat(1, expert_raw.shape[1], 1)
                parts.append(time_b)
            # Optional medication embedding shared across nodes
            if (
                getattr(self, "med_proj", None) is not None
                and med_context.shape[1] == self.n_medications * 2
            ):
                med_embed = torch.tanh(self.med_proj(med_context))  # [B,8]
                med_b = med_embed.unsqueeze(1).repeat(1, expert_raw.shape[1], 1)
                parts.append(med_b)
            node_features = torch.cat(parts, dim=-1)  # [B,14,F]
            if torch.isnan(node_features).any() or torch.isinf(node_features).any():
                raise RuntimeError("Non-finite values in GAT node_features. Aborting for debugging.")
            # Additional stabilization for attention: clamp and layer-normalize features per node
            node_features = torch.clamp(node_features, -50.0, 150.0)
            try:
                import torch.nn.functional as F  # already imported at top, safe

                node_features = F.layer_norm(node_features, node_features.shape[-1:])
            except Exception:
                pass

            # Fail fast if any GAT controller parameter has NaN/Inf
            for name, param in self.gat_controller.named_parameters():
                if param is None:
                    continue
                if torch.isnan(param).any() or torch.isinf(param).any():
                    raise RuntimeError(f"Detected non-finite values in GAT parameter '{name}'. Aborting for debugging.")

            u_raw = self.gat_controller(node_features)  # [B,4]
            if self.force_no_controls:
                scaled_output = torch.zeros_like(u_raw)
            else:
                # Apply per-head scales and global weighting
                scaled_output = (
                    u_raw * self.control_scales.to(u_raw.device)
                ) * self.SDE_control_weighting
            scaled_output = torch.nan_to_num(
                scaled_output, nan=0.0, posinf=0.0, neginf=0.0
            )
            scaled_output = torch.clamp(scaled_output, -5.0, 5.0)

            if self.debug:
                t_idx = torch.round((t.to(self._t0) - self._t0) / self._dt_grid).to(
                    torch.long
                )
                if (t_idx.remainder(10) == 0).item():
                    print(
                        f"[DEBUG] Hybrid_SDE apply_SDE_fun (GAT): node_features={node_features.shape}, u_raw={u_raw.shape}, scaled_output={scaled_output.shape}"
                    )
                    try:
                        nf = node_features
                        clamp_frac = (scaled_output.abs() >= 5.0).float().mean().item()
                        print(
                            f"  node_features stats: mean={nf.mean().item():.4f}, std={nf.std().item():.4f}, min={nf.min().item():.4f}, max={nf.max().item():.4f}"
                        )
                        print(
                            f"  u_raw stats: mean={u_raw.mean().item():.4f}, std={u_raw.std().item():.4f}, min={u_raw.min().item():.4f}, max={u_raw.max().item():.4f}"
                        )
                        print(
                            f"  scaled_output stats: mean={scaled_output.mean().item():.4f}, std={scaled_output.std().item():.4f}, min={scaled_output.min().item():.4f}, max={scaled_output.max().item():.4f}, clamp|x|>=5 frac={clamp_frac:.3f}"
                        )
                    except Exception:
                        pass
            return scaled_output

        # Else: MLP controller path — append physics and med context to flat input
        SDE_NN_input = torch.cat([SDE_NN_input, physics_feats], dim=-1)

        # print('SDE_NN_input shape', SDE_NN_input.shape)
        # print('SDE_NN_input example', SDE_NN_input[0,:])
        # NaN/Inf fail-fast for inputs
        if torch.isnan(SDE_NN_input).any() or torch.isinf(SDE_NN_input).any():
            nan_cnt = int(torch.isnan(SDE_NN_input).sum().item())
            inf_cnt = int(torch.isinf(SDE_NN_input).sum().item())
            bad_mask = torch.isnan(SDE_NN_input) | torch.isinf(SDE_NN_input)
            bad_rows = bad_mask.any(dim=1).nonzero(as_tuple=False).flatten()[:5]
            raise RuntimeError(
                f"SDE_NN_input contains non-finite values (NaN={nan_cnt}, Inf={inf_cnt}). Sample bad rows: {bad_rows.tolist()}"
            )

        for name, param in self.SDEnet.named_parameters():
            if param is None or param.data is None:
                continue
            if torch.isnan(param.data).any() or torch.isinf(param.data).any():
                raise RuntimeError(f"Detected non-finite values in SDEnet parameter '{name}'. Aborting for debugging.")

        SDE_NN_input = torch.cat([SDE_NN_input, med_context], dim=-1)

        if self.debug:
            try:
                print(
                    f"  [DBG] SDE_NN_input: shape={tuple(SDE_NN_input.shape)} mean={SDE_NN_input.mean().item():.4e} std={SDE_NN_input.std().item():.4e} min={SDE_NN_input.min().item():.4e} max={SDE_NN_input.max().item():.4e}"
                )
            except Exception:
                pass

        # if self.debug:
        #    print(f"SDE_NN_input shape: {SDE_NN_input.shape}")

        SDE_NN_output_latents = self.SDEnet(SDE_NN_input)

        # TODO do these clamps make sense
        # control_scales = torch.tensor([100.0, 30.0], device=SDE_NN_output_latents.device)
        # scaled_output = SDE_NN_output_latents * control_scales.unsqueeze(0)
        if self.force_no_controls:
            scaled_output = torch.zeros_like(SDE_NN_output_latents)
        else:
            # Apply per-head scales and global weighting
            scaled_output = (
                SDE_NN_output_latents
                * self.control_scales.to(SDE_NN_output_latents.device)
            ) * self.SDE_control_weighting
        if torch.isnan(scaled_output).any() or torch.isinf(scaled_output).any():
            raise RuntimeError("Non-finite values in scaled_output. Aborting for debugging.")
        scaled_output = torch.clamp(scaled_output, -5.0, 5.0)

        if torch.isnan(SDE_NN_output_latents).any() or torch.isinf(SDE_NN_output_latents).any():
            try:
                torch.autograd.set_detect_anomaly(True)
            except Exception:
                pass
            raise RuntimeError("SDE_NN_output contains non-finite values. Aborting for debugging.")
        # print(SDE_NN_output_latents)
        # print(self.SDE_input_state)
        # breakpoint()
        # print('SDE_NN_output_latents example', SDE_NN_output_latents[0, :])
        has_nonzero = SDE_NN_output_latents.ne(0.0).any()
        # print('SDE_NN Has non-0 OUTPUT??', has_nonzero)
        if self.debug:
            t_idx = torch.round((t.to(self._t0) - self._t0) / self._dt_grid).to(
                torch.long
            )
            if (t_idx.remainder(10) == 0).item():
                print(
                    f"[DEBUG] Hybrid_SDE apply_SDE_fun: SDE_NN_input_shape={SDE_NN_input.shape}, SDE_NN_output_latents_shape={SDE_NN_output_latents.shape}"
                )
                print(f"Scaled output: {scaled_output}")
                try:
                    print(
                        f"  inputs: mean={SDE_NN_input.mean().item():.4f}, std={SDE_NN_input.std().item():.4f}, min={SDE_NN_input.min().item():.4f}, max={SDE_NN_input.max().item():.4f}"
                    )
                    print(
                        f"  physics: mean={physics_feats.mean().item():.4f}, std={physics_feats.std().item():.4f}, min={physics_feats.min().item():.4f}, max={physics_feats.max().item():.4f}"
                    )
                    print(
                        f"  dt_u: mean={scaled_output.mean().item():.4f}, std={scaled_output.std().item():.4f}, min={scaled_output.min().item():.4f}, max={scaled_output.max().item():.4f}"
                    )
                except Exception:
                    pass

        # Optionally zero specific control heads (debug/ablation support)
        if hasattr(self, "disabled_control_indices") and self.disabled_control_indices:
            try:
                scaled_output[:, self.disabled_control_indices] = 0.0
                if self.debug:
                    t_idx = torch.round((t.to(self._t0) - self._t0) / self._dt_grid).to(
                        torch.long
                    )
                    if (t_idx.remainder(10) == 0).item():
                        print(
                            f"[DEBUG] apply_SDE_fun: zeroed controls at indices {self.disabled_control_indices}"
                        )
            except Exception as e:
                if self.debug:
                    print(
                        f"[WARN] Failed to zero controls {self.disabled_control_indices}: {e}"
                    )

        return scaled_output

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

    def f(self, t, y, Tx, time_to_treatment):  # Approximate posterior drift.
        if self.debug and t.item() % 10 == 0:
            pass

        batch_size = y.shape[0]

        y_clamped = torch.cat(
            [
                y[:, : self.SDEnet_out_dims],  # Keep i_ext unchanged
                torch.clamp(
                    y[
                        :,
                        self.SDEnet_out_dims : self.SDEnet_out_dims
                        + self.expert_latent_dims,
                    ],
                    min=self.physio_min_vals,
                    max=self.physio_max_vals,
                ),  # Clamp physio vars
                y[
                    :, self.SDEnet_out_dims + self.expert_latent_dims :
                ],  # Keep neural embedding unchanged
            ],
            dim=1,
        )

        # y now contains: [i_ext (2), expert_latents (14), neural_embedding (4)]

        i_ext_SDE_dict = {}
        for i in range(self.SDEnet_out_dims):
            i_ext_SDE_dict[f"i_ext_SDE_{i+1}"] = y_clamped[:, i].unsqueeze(1)

        # Use tensor-safe comparison for time gating (no .item())
        t_tensor = t if torch.is_tensor(t) else torch.tensor(t, device=self.device)
        if (t_tensor >= time_to_treatment).all():
            # this will always be the case when working with mimic data as time to treatment is 0
            dt_i_ext_SDE = self.apply_SDE_fun(t, y_clamped)
            dt_i_ext_SDE_dict = {}
            for i in range(self.SDEnet_out_dims):
                dt_i_ext_SDE_dict[f"dt_i_ext_SDE_{i+1}"] = dt_i_ext_SDE[:, i].unsqueeze(
                    1
                )
        else:
            raise ValueError("Time to treatment should always be <= current time t")

        # Neural embedding derivatives (zeros - they evolve stochastically)
        dt_neural_embedding = torch.zeros([batch_size, self.encoder_SDENN_dims]).to(
            self.device
        )

        # Construct the output in the correct order to match the state vector y
        # The order should be: i_ext (2), expert_latents (14), neural_embedding (4)
        # Total: 20 dimensions

        # For i_ext
        dt_i_ext = torch.cat(
            [
                dt_i_ext_SDE_dict[f"dt_i_ext_SDE_{i+1}"]
                for i in range(self.SDEnet_out_dims)
            ],
            dim=-1,
        )

        # compute the expert latents from
        # Pass only the expert slice to ensure indexing aligns with Zenker state order
        expert_slice = y_clamped[
            :, self.SDEnet_out_dims : self.SDEnet_out_dims + self.expert_latent_dims
        ]
        (
            dpa_dt,
            dpv_dt,
            ds_dt,
            dsv_dt,
            dt_expert,
            dt_r_tpr_mod,
            dt_f_hr_max,
            dt_f_hr_min,
            dt_r_tpr_max,
            dt_r_tpr_min,
            dt_ca,
            dt_cv,
            dt_k_width,
            dt_p_aset,
            dt_tau,
        ) = zenker_derivatives(expert_slice, device=self.device, expert_start_index=0)

        if self.debug:
            t_idx = torch.round((t.to(self._t0) - self._t0) / self._dt_grid).to(
                torch.long
            )
            if (t_idx.remainder(10) == 0).item():
                try:
                    pa_vals = expert_slice[:, 0]
                    pv_vals = expert_slice[:, 1]
                    ca_vals = expert_slice[:, 9]
                    cv_vals = expert_slice[:, 10]
                    print(
                        f"[DEBUG] f(): t_idx={int(t_idx):.0f} | p_a mean={pa_vals.mean().item():.2f}, p_v mean={pv_vals.mean().item():.2f}, c_a mean={ca_vals.mean().item():.2f}, c_v mean={cv_vals.mean().item():.2f}"
                    )
                    print(
                        f"  dpa_dt mean={dpa_dt.mean().item():.4f}, dpv_dt(before ctl) mean={dpv_dt.mean().item():.4f}"
                    )
                except Exception:
                    pass

        # apply model-specific transformations on Zenker model output using control DERIVATIVES
        dpv_dt = dpv_dt + dt_i_ext_SDE_dict["dt_i_ext_SDE_1"]
        dsv_dt = dt_i_ext_SDE_dict["dt_i_ext_SDE_2"]
        dt_ca = dt_ca + dt_i_ext_SDE_dict["dt_i_ext_SDE_3"]
        dt_r_tpr_mod = dt_r_tpr_mod + dt_i_ext_SDE_dict["dt_i_ext_SDE_4"]

        if self.debug:
            t_idx = torch.round((t.to(self._t0) - self._t0) / self._dt_grid).to(
                torch.long
            )
            if (t_idx.remainder(10) == 0).item():
                print(f"[DEBUG] Time idx: {int(t_idx)}")
                print(
                    f"  controls (first sample): {y_clamped[0, :self.SDEnet_out_dims]}"
                )
                print(f"  control derivs (first sample): {dt_i_ext[0]}")
                try:
                    print(
                        f"  dpv_dt(after ctl) mean={dpv_dt.mean().item():.4f}, dt_ca mean={dt_ca.mean().item():.6f}, dt_r_tpr_mod mean={dt_r_tpr_mod.mean().item():.6f}"
                    )
                except Exception:
                    pass

        dt_expert = torch.cat(
            [
                dpa_dt,
                dpv_dt,
                ds_dt,
                dsv_dt,
                dt_r_tpr_mod,
                dt_f_hr_max,
                dt_f_hr_min,  # Next 3 (indices 6-8)
                dt_r_tpr_max,
                dt_r_tpr_min,  # Next 2 (indices 9-10)
                dt_ca,
                dt_cv,
                dt_k_width,
                dt_p_aset,
                dt_tau,
            ],
            dim=-1,
        )

        # Combine all with fail-fast non-finite checks
        if torch.isnan(dt_i_ext).any() or torch.isinf(dt_i_ext).any():
            raise RuntimeError("Non-finite values in dt_i_ext. Aborting for debugging.")
        dt_i_ext = dt_i_ext.clamp_(-10.0, 10.0)
        if torch.isnan(dt_expert).any() or torch.isinf(dt_expert).any():
            raise RuntimeError("Non-finite values in dt_expert. Aborting for debugging.")
        if torch.isnan(dt_neural_embedding).any() or torch.isinf(dt_neural_embedding).any():
            raise RuntimeError("Non-finite values in dt_neural_embedding. Aborting for debugging.")
        final_f_out = torch.cat([dt_i_ext, dt_expert, dt_neural_embedding], dim=-1)

        if self.debug:
            t_idx = torch.round((t.to(self._t0) - self._t0) / self._dt_grid).to(
                torch.long
            )
            if (t_idx.remainder(10) == 0).item():
                expected_dims = (
                    self.SDEnet_out_dims
                    + self.expert_latent_dims
                    + self.encoder_SDENN_dims
                )
                print(
                    f"[DEBUG] f: final_f_out shape = {final_f_out.shape} (expected [batch, {expected_dims}])"
                )

        return final_f_out

    # 3. Fixed f_aug method:
    def f_aug(self, t, y):
        i_ext = y[:, : self.SDEnet_out_dims]
        dt_all_dims = y[
            :,
            : self.SDEnet_out_dims + self.expert_latent_dims + self.encoder_SDENN_dims,
        ]
        Tx = y[:, -2]
        time_to_treatment = y[0, -1]

        # Get normal dynamics
        f_res = self.f(t, dt_all_dims, Tx, time_to_treatment)

        if self.self_reverting_prior_control:
            # Apply OU prior to ALL control heads: i_ext[:, 0:SDEnet_out_dims]
            u2_sum = None
            for ctrl_idx in range(self.SDEnet_out_dims):
                y_ctrl = i_ext[:, ctrl_idx].unsqueeze(1)
                g_ctrl, h_ctrl = self.g(t, y_ctrl), self.h(t, y_ctrl)
                f_ctrl = f_res[:, ctrl_idx].unsqueeze(1)
                u = _stable_division(f_ctrl - h_ctrl, g_ctrl)  # normalized drift diff
                term = 0.5 * (u**2)  # [batch, 1]
                u2_sum = term if u2_sum is None else (u2_sum + term)
            # Resulting per-sample KL integrand
            f_logqp = u2_sum  # [batch, 1]
        else:
            self.mu = None
            f_logqp = torch.zeros_like(y[:, 0]).unsqueeze(1).to(self.device)

        # f_res contains derivatives for: i_ext (4) + expert (14) + neural (encoder dims)
        # We need to add derivatives for: logqp (1) + Tx (1) + time_to_tx (1) = 3 dims

        # Derivatives for Tx and time_to_tx are zero (they don't change)
        dt_tx = torch.zeros_like(f_logqp)
        dt_time_to_tx = torch.zeros_like(f_logqp)

        # Apply mask to freeze dynamics after valid time
        f_out = torch.cat(
            [
                f_res,
                f_logqp,
                dt_tx,
                dt_time_to_tx,
            ],
            dim=1,
        )
        if self.debug:
            t_idx = torch.round((t.to(self._t0) - self._t0) / self._dt_grid).to(
                torch.long
            )
            if (t_idx.remainder(10) == 0).item():
                print(
                    f"[DEBUG] Hybrid_SDE f_aug: t_idx={int(t_idx)}, f_out_shape={f_out.shape}"
                )

        return f_out

    # 4. Fixed h method:
    def h(self, t, y):  # Prior drift.
        if self.debug:
            t_idx = torch.round((t.to(self._t0) - self._t0) / self._dt_grid).to(
                torch.long
            )
            if (t_idx.remainder(10) == 0).item():
                print(
                    f"[DEBUG] Hybrid_SDE h (prior drift): t_idx={int(t_idx)}, y_shape={y.shape}"
                )

        # y here should be just i_ext_2 (single dimension)
        self.mu = torch.tensor([0.0], device=y.device)
        expanded_mu = self.mu.repeat(y.size(0), 1)

        # Get theta value for i_ext_2
        if isinstance(self.theta, (int, float)):
            theta_val = self.theta
        elif self.theta.dim() == 0:
            theta_val = self.theta.item()
        elif self.theta.dim() == 1:
            theta_val = (
                self.theta[0].item() if self.theta.shape[0] > 0 else self.theta.item()
            )
        else:  # 2D
            theta_val = self.theta[0, 0].item()

        theta_for_iext2 = torch.tensor([[theta_val]], device=y.device).repeat(
            y.size(0), 1
        )

        return theta_for_iext2 * (expanded_mu - y)

    # 5. Fixed g method:
    def g(self, t, y):
        if self.debug:
            t_idx = torch.round((t.to(self._t0) - self._t0) / self._dt_grid).to(
                torch.long
            )
            if (t_idx.remainder(10) == 0).item():
                print(
                    f"[DEBUG] Hybrid_SDE g (diffusion): t_idx={int(t_idx)}, y_shape={y.shape}"
                )

        batch_size = y.shape[0]

        # Handle different shapes of self.sigma
        if self.sigma.dim() == 1:
            if self.sigma.shape[0] > 1:
                sigma_val = self.sigma[0].item()
            else:
                sigma_val = self.sigma.item()
        elif self.sigma.dim() == 2:
            sigma_val = self.sigma[0, 0].item()
        else:
            sigma_val = self.sigma.item()

        # Return as a tensor with shape [batch_size, 1]
        return torch.full((batch_size, 1), sigma_val, device=y.device)

    # 6. Fixed g_aug method:
    def g_aug(self, t, y):
        """Diffusion for augmented dynamics (no valid_time masking)."""
        # concise only

        batch_size = y.shape[0]

        # Diffusion only on control states (first 4); zeros elsewhere
        i_ext_1 = y[:, 0].unsqueeze(1)
        g_i_ext_1 = self.g(t, i_ext_1)
        i_ext_2 = y[:, 1].unsqueeze(1)
        g_i_ext_2 = self.g(t, i_ext_2)
        i_ext_3 = y[:, 2].unsqueeze(1)
        g_i_ext_3 = self.g(t, i_ext_3)
        i_ext_4 = y[:, 3].unsqueeze(1)
        g_i_ext_4 = self.g(t, i_ext_4)

        g_expert_dims = torch.zeros([batch_size, self.expert_latent_dims]).to(y.device)
        g_neural_dims = torch.zeros([batch_size, self.encoder_SDENN_dims]).to(y.device)
        g_logqp = torch.zeros([batch_size, 1]).to(y.device)
        g_tx = torch.zeros([batch_size, 1]).to(y.device)
        g_time_to_tx = torch.zeros([batch_size, 1]).to(y.device)

        g_out = torch.cat(
            [
                g_i_ext_1,
                g_i_ext_2,
                g_i_ext_3,
                g_i_ext_4,
                g_expert_dims,
                g_neural_dims,
                g_logqp,
                g_tx,
                g_time_to_tx,
            ],
            dim=1,
        )

        # concise only

        return g_out

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

    def prior_diext_dt(self, t):
        factor = -2 * (t - 5) / 5
        exponential = torch.exp(-(((t - 5) / 5) ** 2))
        diext_dt = torch.tensor([5 / 3 * factor * exponential]).to(self.device)
        # print('diext_dt', diext_dt.shape)
        return diext_dt.unsqueeze(1)

    def prior_rate_of_change_of_flow(self, t, volumes=50, durations=20, width_factor=3):
        volumes = (
            torch.tensor(volumes, dtype=torch.float32).unsqueeze(0).to(self.device)
        )
        durations = (
            torch.tensor(durations, dtype=torch.float32).unsqueeze(0).to(self.device)
        )
        width_factor = torch.tensor(width_factor, dtype=torch.float32).to(self.device)

        # print('t:', t.shape)
        if t.ndim == 0:
            t = t.unsqueeze(0)

        means = durations / 2
        sigmas = width_factor * durations / 10
        A = volumes / (
            torch.sqrt(torch.tensor(np.pi)) * sigmas * erf((durations - means) / sigmas)
        )
        # print('A:', A.shape)

        derivatives = (
            -2
            * A[:, None]
            * (t - means[:, None])
            / sigmas[:, None] ** 2
            * torch.exp(-(((t - means[:, None]) / sigmas[:, None]) ** 2))
        )
        # print('derivatives', derivatives.shape)
        return derivatives

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
        """
        Forward through SDE with batch-compatible variable length support.
        """

        batch_size = init_latents.shape[0]

        # sys.setrecursionlimit(10)
        if self.debug:
            print(f"[DEBUG] forward_latent: init_latents_shape={init_latents.shape}")
            if valid_lengths is not None:
                print(f"[DEBUG] forward_latent: valid_lengths={valid_lengths}")

        # Store for use in apply_SDE_fun
        self.current_med_values = med_traj_values  # (batch, time, n_meds)
        self.current_med_mask = med_traj_mask  # (batch, time, n_meds)
        self.current_med_time = med_traj_time  # (batch, time)

        if self.debug:
            print(
                f"current_med_values shape: {self.current_med_values.shape}. (batch, time, n_meds)"
            )
        # Initialize time grid helpers for debug/indexing (avoid host syncs)
        try:
            self._t0 = ts[0].to(self.device)
            self._time_len = int(ts.shape[0])
            if ts.shape[0] > 1:
                self._dt_grid = (ts[1] - ts[0]).to(self.device).clamp_min(1e-6)
            else:
                self._dt_grid = torch.tensor(
                    float(self.integration_step_size), device=self.device
                )
        except Exception:
            # Fallback defaults
            self._t0 = torch.tensor(0.0, device=self.device)
            self._time_len = 1
            self._dt_grid = torch.tensor(
                float(self.integration_step_size), device=self.device
            )

        # Note: removed runtime precompute; med_context now loaded offline via dataloader

        # Prepare standard augmented state components
        Tx_expanded = (
            Tx.unsqueeze(1).unsqueeze(2).repeat(1, self.num_samples, 1).to(init_latents)
        )
        if self.debug:
            print(f"Tx expanded shape: {Tx_expanded.shape}.")

        time_to_tx_expanded = (
            time_to_tx.unsqueeze(1)
            .unsqueeze(2)
            .repeat(1, self.num_samples, 1)
            .to(init_latents)
        )
        i_ext = torch.zeros(batch_size, self.num_samples, self.SDEnet_out_dims).to(
            init_latents
        )
        log_path = torch.zeros(batch_size, self.num_samples, 1).to(init_latents)

        # Legacy valid_time masking removed; training loss masking already restricts supervision window.

        # Optionally override initial latents with exact Zenker defaults for diagnostics
        if self.force_zenker_defaults:
            # Zenker defaults from ZenkerModel.ZenkerODE
            pa0 = torch.full_like(init_latents[..., 0], 100.0)
            pv0 = torch.full_like(init_latents[..., 1], 8.0)
            s0 = torch.full_like(init_latents[..., 2], 0.5)
            sv0 = torch.full_like(init_latents[..., 3], 70.0)
            r_tpr_mod0 = torch.full_like(init_latents[..., 4], 0.0)
            f_hr_max0 = torch.full_like(init_latents[..., 5], 3.0)
            f_hr_min0 = torch.full_like(init_latents[..., 6], 1.0)
            r_tpr_max0 = torch.full_like(init_latents[..., 7], 2.13)
            r_tpr_min0 = torch.full_like(init_latents[..., 8], 0.53)
            ca0 = torch.full_like(init_latents[..., 9], 4.0)
            cv0 = torch.full_like(init_latents[..., 10], 111.0)
            k_width0 = torch.full_like(init_latents[..., 11], 0.18)
            p_aset0 = torch.full_like(init_latents[..., 12], 70.0)
            tau0 = torch.full_like(init_latents[..., 13], 20.0)
            expert0 = torch.stack(
                [
                    pa0,
                    pv0,
                    s0,
                    sv0,
                    r_tpr_mod0,
                    f_hr_max0,
                    f_hr_min0,
                    r_tpr_max0,
                    r_tpr_min0,
                    ca0,
                    cv0,
                    k_width0,
                    p_aset0,
                    tau0,
                ],
                dim=-1,
            )
            zeros_neural = torch.zeros_like(
                init_latents[..., : self.encoder_SDENN_dims]
            )
            init_latents = torch.cat([expert0, zeros_neural], dim=-1)

        # Create augmented initial state
        aug_y0 = torch.cat(
            [
                i_ext,
                init_latents,
                log_path,
                Tx_expanded,
                time_to_tx_expanded,
            ],
            dim=-1,
        )
        dim_aug = aug_y0.shape[-1]  # 24
        if self.debug:
            print(
                f"aug_y0 shape: {aug_y0.shape}. Expected {batch_size} x {self.num_samples} x {dim_aug}"
            )  # 18 = i_ext (2) + init_latents (14) + 1 each for rest
        # print(f"Aug_y0: {aug_y0[0]}")
        # breakpoint()

        # Reshape for SDE integration
        aug_y0 = aug_y0.reshape(-1, dim_aug)
        if self.debug:
            print(
                f"Aug y0 shape: {aug_y0.shape}"
            )  # 161 x 24. Each element in the batch has 7 samples: 7 x 23 = 161. 24 variables
            print("[DEBUG] aug_y0 stats before SDE:")
            print(f"  Shape: {aug_y0.shape}")
            print(f"  Contains NaN: {torch.isnan(aug_y0).any()}")
            print(f"  Contains Inf: {torch.isinf(aug_y0).any()}")
            print(f"  Min/Max: {aug_y0.min().item()}/{aug_y0.max().item()}")

        # Check for extreme values that might cause numerical issues
        if aug_y0.max().item() > 1e6 or aug_y0.min().item() < -1e6:
            print(
                f"[WARNING] Extreme values in aug_y0: min={aug_y0.min().item()}, max={aug_y0.max().item()}"
            )

        # Run SDE integration
        options = {"dtype": torch.float32}

        aug_ys = self.sdeint_fn(
            sde=self,
            y0=aug_y0,
            ts=ts,
            method=self.integration_method,
            dt=self.integration_step_size,
            adaptive=self.integration_adaptive,
            rtol=self.rtol,
            atol=self.atol,
            options=options,
            names={"drift": "f_aug", "diffusion": "g_aug"},
        )

        if self.debug:
            print(f"Aug_ys shape: {aug_ys.shape}. Expect: [len(ts) x 161 x 24]")

        # Reshape back and extract outputs (excluding valid_time from outputs)
        aug_ys = aug_ys.view(len(ts), batch_size, self.num_samples, dim_aug).permute(
            1, 2, 0, 3
        )
        if self.debug:
            print(f"aug_ys shape: {aug_ys.shape}. Expect: 23 x 7 x 17 x 24")

        # Extract paths (don't include the valid_time in outputs)
        i_ext_path = aug_ys[:, :, :, : self.SDEnet_out_dims]
        latent_out = aug_ys[
            :,
            :,
            :,
            self.SDEnet_out_dims : self.expert_latent_dims + self.SDEnet_out_dims,
        ]
        # Layout: [i_ext (4), expert (14), neural (encoder dims), logqp, Tx, time_to_tx]
        # Select logqp (third from last)
        logqp_path = aug_ys[:, :, -1, -3]

        if self.debug:
            print(f"Latent out: {latent_out.shape}. Expect [23 x 7 x 17 x 14]")

        return latent_out, logqp_path, i_ext_path

    def forward_dec(self, latent_out):
        """
        Selects the output trajectories from the latents
        Args:
            latent_out:

        Returns:

        """
        if self.debug:
            print(
                f"[DEBUG] Hybrid_SDE forward_dec: latent_out_shape={latent_out.shape}"
            )
            # print('latent_out', latent_out[0, 0, :, 0])
            # print('latent_out', latent_out[0, 1, :, 0])
            # print('latent_out', latent_out[1, 0, :, 0])
            # print('latent_out', latent_out[1, 1, :, 0])
            # print('latent device', latent_out.device)

        output_traj = select_tensor_by_index_list_advanced(
            latent_out, self.decoder_output_dims
        )

        pa = torch.clamp(output_traj[..., 0], min=40.0, max=220.0)
        pv = torch.clamp(output_traj[..., 1], min=0, max=39)

        output_traj = torch.stack([pa, pv], dim=-1)

        if self.debug:
            print(
                f"[DEBUG] Hybrid_SDE forward_dec: decoded_mean_shape={output_traj.shape}"
            )
        return output_traj

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

        # TODO compute loss over normlized values

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
            ic_consistency_loss = torch.tensor(0.0, device=init_states.device)

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

            # Additional diagnostics: per-channel masked MSE/MAE on predicted mean
            pred_mean = predicted_traj.mean(dim=1)  # [B,T,C]
            if mask is not None:
                mask_f = mask.to(dtype=pred_mean.dtype, device=pred_mean.device)
                denom = torch.clamp(mask_f.sum(dim=(0, 1)), min=1.0)  # [C]
                se = ((pred_mean - true_traj) ** 2) * mask_f  # [B,T,C]
                ae = (pred_mean - true_traj).abs() * mask_f
                mse_ch = se.sum(dim=(0, 1)) / denom
                mae_ch = ae.sum(dim=(0, 1)) / denom
                print(f"[DEBUG] MSE per channel: {mse_ch.detach().cpu().tolist()}")
                print(f"[DEBUG] MAE per channel: {mae_ch.detach().cpu().tolist()}")
                # Saturation stats (0 and high clip ~220 if present)
                zeros_frac = ((pred_mean == 0.0) * mask_f.bool()).sum(
                    dim=(0, 1)
                ) / torch.clamp(mask_f.sum(dim=(0, 1)), min=1.0)
                high_frac = ((pred_mean >= 210.0) * mask_f.bool()).sum(
                    dim=(0, 1)
                ) / torch.clamp(mask_f.sum(dim=(0, 1)), min=1.0)
                print(
                    f"[DEBUG] Saturation frac@0 per channel: {zeros_frac.detach().cpu().tolist()}"
                )
                print(
                    f"[DEBUG] Saturation frac@>=210 per channel: {high_frac.detach().cpu().tolist()}"
                )
                # Mask coverage
                cov = mask_f.mean(dim=(0, 1))
                print(
                    f"[DEBUG] Mask coverage per channel: {cov.detach().cpu().tolist()}"
                )
                # Per-channel NLL (mean over B,T,S)
                mask_expanded_dbg = mask_f.unsqueeze(1).expand(
                    -1, predicted_traj.shape[1], -1, -1
                )
                logpy_masked = logpy * mask_expanded_dbg
                denom_ch = torch.clamp(mask_f.sum(dim=(0, 1)), min=1.0)
                # Sum over batch, samples, and time to get per-channel NLL
                nll_ch = -logpy_masked.sum(dim=(0, 1, 2)) / denom_ch  # [C]
                nll_ch_mean = nll_ch
                print(
                    f"[DEBUG] NLL per channel (avg over samples): {nll_ch_mean.detach().cpu().tolist() if nll_ch_mean.numel()>1 else float(nll_ch_mean.detach().cpu())}"
                )
                # Small diff snippet first 5 timesteps
                Tsnip = min(pred_mean.shape[1], 5)
                diff_snip = (
                    (pred_mean[:1, :Tsnip, :] - true_traj[:1, :Tsnip, :]).detach().cpu()
                )
                print(
                    f"[DEBUG] y_pred - y_true (first sample, first {Tsnip} steps):\n{diff_snip}"
                )
            else:
                print("[DEBUG] No mask provided; MSE/MAE coverage skipped.")

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

    def on_after_backward(self) -> None:
        # Fail-fast on non-finite gradients (always on)
        total_norm_sq = 0.0
        count = 0
        with torch.no_grad():
            for name, p in self.named_parameters():
                if p.grad is not None:
                    g = p.grad.detach()
                    if torch.isnan(g).any() or torch.isinf(g).any():
                        raise RuntimeError(f"Non-finite gradient detected in parameter '{name}'. Aborting for debugging.")
                    total_norm_sq += float(g.norm(2).item() ** 2)
                    count += 1

    def on_before_optimizer_step(self, optimizer) -> None:
        # Always fail-fast if parameters contain non-finite values before stepping
        with torch.no_grad():
            for name, p in self.named_parameters():
                if p is None or p.data is None:
                    continue
                if torch.isnan(p.data).any() or torch.isinf(p.data).any():
                    raise RuntimeError(f"Non-finite parameter detected before optimizer step: '{name}'. Aborting for debugging.")

    def compute_counterfactual_loss(self, true_fact, true_cf, pred_fact, pred_cf):
        if self.debug:
            print(
                f"[DEBUG] Hybrid_SDE compute_counterfactual_loss: true_fact_shape={true_fact.shape}, pred_cf_shape={pred_cf.shape}"
            )
        # print('true_fact:', true_fact.shape, true_fact[0,:,:] )
        # print('true_cf:', true_cf.shape, true_cf[0,:,:])
        # print('pred_fact:', pred_fact.shape, pred_fact.mean(1)[0,:,:])
        # print('pred_cf:', pred_cf.shape, pred_cf.mean(1)[0,:,:])

        # RECON LOSS
        # MSE loss between the Y and the MEAN of the SDE samples predictions, which includes expert and SDE in hybrid
        mse_cf = torch.sqrt(self.MSE_loss(true_cf, pred_cf.mean(1))).mean()

        # Now find the mean of the standard devs of the predictions across the SDE samples
        std_preds_cf = pred_cf.std(1).mean()

        # Individual Treatment Effect computed as the difference between Y_cf and Y
        ite = true_cf - true_fact
        # print('ite:', ite.shape)

        # Predicted Individual Treatment Effect computed as the difference between the mean predictions of Y_hat_cf and Y_hat
        ite_hat = pred_cf.mean(1) - pred_fact.mean(1)
        # print('ite_hat:', ite_hat.shape)

        # MSE of the ITE
        mse_ite = torch.sqrt(self.MSE_loss(ite, ite_hat)).mean()
        # print('mse_ite:', mse_ite)

        if self.debug:
            print(
                f"[DEBUG] Hybrid_SDE compute_counterfactual_loss: mse_fact={mse_cf.item()}, mse_cf={mse_cf.item()}"
            )
        return mse_cf, mse_ite, std_preds_cf

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

        if torch.isnan(z1_for_sde).any() or torch.isinf(z1_for_sde).any():
            nan_cnt = int(torch.isnan(z1_for_sde).sum().item())
            inf_cnt = int(torch.isinf(z1_for_sde).sum().item())
            raise RuntimeError(
                f"Non-finite values in final z1_for_sde: NaN={nan_cnt}, Inf={inf_cnt}. Aborting for debugging."
            )

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
            if self.debug:
                try:
                    pred_mean_ic = predicted_ode_latents.mean(dim=1)  # [B,14]
                    pred_min_ic, _ = predicted_ode_latents.min(dim=1)
                    pred_max_ic, _ = predicted_ode_latents.max(dim=1)
                    def _dims_str(t: torch.Tensor) -> str:
                        vals = t.detach().mean(dim=0)
                        return (
                            f"pa={vals[0].item():.2f}, pv={vals[1].item():.2f}, s={vals[2].item():.3f}, sv={vals[3].item():.2f}, r_tpr_mod={vals[4].item():.3f}, ca={vals[9].item():.2f}, cv={vals[10].item():.2f}"
                        )
                    print("[DEBUG] Encoder IC@t0 (pred) batch-mean:", _dims_str(pred_mean_ic))
                    print("[DEBUG] Encoder IC@t0 (pred) batch-min:", _dims_str(pred_min_ic))
                    print("[DEBUG] Encoder IC@t0 (pred) batch-max:", _dims_str(pred_max_ic))
                    # Compare to provided init_states on available dims
                    num_ic_vars = init_states.shape[-1]
                    pred_ic_used = pred_mean_ic[:, :num_ic_vars]
                    gt_ic = init_states
                    mask_ic = ic_mask
                    valid = mask_ic > 0
                    if valid.any():
                        diff = (pred_ic_used - gt_ic).abs() * valid
                        denom = torch.clamp(valid.sum(dim=0), min=1)
                        mae_per_dim = diff.sum(dim=0) / denom
                        print(
                            f"[DEBUG] Encoder IC vs data (masked MAE per-dim up to {num_ic_vars}): {mae_per_dim.detach().cpu().tolist()}"
                        )
                        overall_mae = (diff.sum() / torch.clamp(valid.sum(), min=1)).item()
                        print(
                            f"[DEBUG] Encoder IC overall MAE on provided dims: {overall_mae:.3f}"
                        )
                    else:
                        print("[DEBUG] No IC mask provided; IC similarity skipped.")
                    print(
                        "[DEBUG] Using encoder-predicted ICs at t0 (with observed dims overwritten by data via ic_mask) to initialise SDE latents feeding the decoder."
                    )
                except Exception as e:
                    print(f"[WARN] IC debug stats failed: {e}")
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
                    tv_loss = (du**2).mean()
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

        # Control energy regularizer: λ · mean(||u||^2) over batch, time, heads
        try:
            i_ext_path = result.get("i_ext_path", None)
            if i_ext_path is not None:
                control_mean = i_ext_path.mean(1)  # [B, T, D]
                control_energy = (control_mean**2).mean()
            else:
                control_energy = torch.tensor(0.0, device=total_loss.device)
        except Exception:
            control_energy = torch.tensor(0.0, device=total_loss.device)

        total_loss = total_loss + self.control_energy_weight * control_energy

        # Optional training plots (controls only), gated by plot_every
        should_plot_now = False
        try:
            step = int(getattr(self, "global_step", 0))
            interval = max(1, int(getattr(self, "plot_every", 1)))
            should_plot_now = (step % interval) == 0
        except Exception:
            should_plot_now = self.plot_outputs_train

        if self.plot_outputs_train and should_plot_now:
            try:
                self.plot_nature_with_controls(
                    result["decoded_traj"],
                    result["Y"],
                    result["combined_mask"],
                    result["i_ext_path"],
                    batch_idx,
                    result["z1_for_sde"],
                )
            except Exception as e:
                if self.debug:
                    print(f"[WARN] Training control plot failed: {e}")

        # Disabled CSV exporting for speed as requested

        # Log metrics
        self.log(
            "train_total_loss",
            total_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "train_control_energy",
            control_energy.detach(),
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            logger=True,
        )
        self.log(
            "train_main_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "train_ic_consistency_loss",
            ic_consistency_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            "train_NLL", nll, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        self.log(
            "train_KL", kl_div, on_step=True, on_epoch=True, prog_bar=True, logger=True
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

            if self.test_zenker:
                zenker_predictions = torch.zeros_like(Y)
                seq_len = Y.shape[1]

                for patient_idx in range(Y.shape[0]):
                    patient_z1 = result["z1_for_sde"][patient_idx, 0].cpu().numpy()
                    p_a_init, p_v_init, s_reflex_init, sv_init = patient_z1[:4]
                    r_tpr_mod, f_hr_max, f_hr_min, r_tpr_max, r_tpr_min = patient_z1[
                        4:9
                    ]
                    ca, cv, k_width, p_aset, tau = patient_z1[9:14]

                    zenker_model = ZenkerODE(
                        p_a_init=float(p_a_init),
                        p_v_init=float(p_v_init),
                        s_reflex_init=float(s_reflex_init),
                        sv_init=float(sv_init),
                        r_tpr_mod=float(r_tpr_mod),
                        f_hr_max=float(f_hr_max),
                        f_hr_min=float(f_hr_min),
                        r_tpr_max=float(r_tpr_max),
                        r_tpr_min=float(r_tpr_min),
                        ca=float(ca),
                        cv=float(cv),
                        k_width=float(k_width),
                        p_aset=float(p_aset),
                        tau=float(tau),
                        use_physiological_clamping=True,
                    )

                    time_seconds = np.arange(seq_len) * 10
                    t_zenker, solution_zenker = zenker_model.integrate(
                        t_span=time_seconds[-1], dt=self.integration_step_size
                    )

                    zenker_pa = np.interp(time_seconds, t_zenker, solution_zenker[:, 0])
                    zenker_pv = np.interp(time_seconds, t_zenker, solution_zenker[:, 1])
                    zenker_predictions[patient_idx, :, 0] = torch.from_numpy(zenker_pa)
                    zenker_predictions[patient_idx, :, 1] = torch.from_numpy(zenker_pv)

                zenker_predictions = zenker_predictions.to(Y.device)

                zenker_mse_per_sample = ((zenker_predictions - Y) ** 2) * combined_mask
                zenker_mae_per_sample = (
                    torch.abs(zenker_predictions - Y) * combined_mask
                )

                zenker_mse = zenker_mse_per_sample.sum() / valid_elements
                zenker_mae = zenker_mae_per_sample.sum() / valid_elements

                self.log(
                    "test_zenker_mse",
                    zenker_mse,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=True,
                    logger=True,
                )
                self.log(
                    "test_zenker_mae",
                    zenker_mae,
                    on_step=False,
                    on_epoch=True,
                    prog_bar=True,
                    logger=True,
                )

        if batch_idx < 3:
            self.plot_nature_style_with_uncertainty(
                decoded_traj, Y, combined_mask, batch_idx
            )
            self.plot_nature_with_controls(
                decoded_traj,
                Y,
                combined_mask,
                result["i_ext_path"],
                batch_idx,
                result["z1_for_sde"],
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
        if self.test_zenker:
            return_dict.update(
                {"test_zenker_mse": zenker_mse, "test_zenker_mae": zenker_mae}
            )
        return return_dict

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=1e-4)

        scheduler = {
            "monitor": "train_total_loss",
            "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer=optimizer, mode="min", factor=0.5, patience=50
            ),
        }
        return {"optimizer": optimizer, "lr_scheduler": scheduler}

    def on_save_checkpoint(self, checkpoint):
        # print('SAVING CHECKPOINT')
        # Manually add mu, sigma, theta to the checkpoint dictionary
        checkpoint["mu"] = self.mu
        checkpoint["sigma"] = self.sigma
        checkpoint["theta"] = self.theta

    def on_load_checkpoint(self, checkpoint):
        # print('LOADING CHECKPOINT')
        # Load mu, sigma, theta from the checkpoint dictionary if they exist
        if "mu" in checkpoint:
            self.mu = checkpoint["mu"]
        if "sigma" in checkpoint:
            self.sigma = checkpoint["sigma"]
        if "theta" in checkpoint:
            self.theta = checkpoint["theta"]

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

        try:
            plt.switch_backend("Agg")
        except Exception:
            pass

        colors = self._setup_plot_style()
        os.makedirs(os.path.join(self.train_dir, "nature_plots"), exist_ok=True)

        pred_mean = predictions_full.mean(1).detach()
        pred_std = predictions_full.std(1, unbiased=False).detach()
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

    def plot_nature_with_controls(
        self,
        predictions_full,
        targets,
        combined_mask,
        i_ext_path,
        batch_idx,
        z1_for_sde=None,
    ):
        """Plot BP + controls with detailed control analysis (Zenker baseline removed)."""

        colors = self._setup_plot_style()  # Use same style
        os.makedirs(os.path.join(self.train_dir, "control_plots"), exist_ok=True)

        try:
            import matplotlib.pyplot as plt

            plt.switch_backend("Agg")
        except Exception:
            pass

        pred_mean = predictions_full.mean(1).detach()
        pred_std = predictions_full.std(1, unbiased=False).detach()
        control_mean = i_ext_path.mean(1).detach()
        control_std = i_ext_path.std(1, unbiased=False).detach()
        # Approximate drift (derivative of control state) using finite differences over 10-second grid
        control_drift = control_mean.clone()
        if control_drift.shape[1] > 1:
            control_drift[:, 1:, :] = (
                control_mean[:, 1:, :] - control_mean[:, :-1, :]
            ) / 10.0
            control_drift[:, 0, :] = control_drift[:, 1, :]
        else:
            control_drift[:] = 0.0

        for patient_idx in range(min(3, predictions_full.shape[0])):
            patient_mask = combined_mask[patient_idx]
            time_seconds = np.arange(patient_mask.shape[0]) * 10

            pred_mean_patient = pred_mean[patient_idx].detach().cpu().numpy()
            pred_std_patient = pred_std[patient_idx].detach().cpu().numpy()
            true_patient = (
                targets[patient_idx].detach().cpu().numpy()
                if hasattr(targets[patient_idx], "requires_grad")
                else targets[patient_idx].cpu().numpy()
            )
            control_mean_patient = control_mean[patient_idx].detach().cpu().numpy()
            control_std_patient = control_std[patient_idx].detach().cpu().numpy()

            arterial_mask = patient_mask[:, 0].cpu().numpy().astype(bool)
            venous_mask = patient_mask[:, 1].cpu().numpy().astype(bool)
            bp_available_mask = arterial_mask | venous_mask

            # Extract ALL initial conditions from z1_for_sde

            patient_z1 = z1_for_sde[patient_idx, 0].detach().cpu().numpy()
            p_a_init, p_v_init, s_reflex_init, sv_init = patient_z1[:4]
            r_tpr_mod, f_hr_max, f_hr_min, r_tpr_max, r_tpr_min = patient_z1[4:9]
            ca, cv, k_width, p_aset, tau = patient_z1[9:14]

            # # Run Zenker baseline
            # zenker_model = ZenkerODE(
            #     p_a_init=float(p_a_init),
            #     p_v_init=float(p_v_init),
            #     s_reflex_init=float(s_reflex_init),
            #     sv_init=float(sv_init),
            #     r_tpr_mod=float(r_tpr_mod),
            #     f_hr_max=float(f_hr_max),
            #     f_hr_min=float(f_hr_min),
            #     r_tpr_max=float(r_tpr_max),
            #     r_tpr_min=float(r_tpr_min),
            #     ca=float(ca),
            #     cv=float(cv),
            #     k_width=float(k_width),
            #     p_aset=float(p_aset),
            #     tau=float(tau),
            #     use_physiological_clamping=True,
            # )

            # t_zenker, solution_zenker = zenker_model.integrate(
            #     t_span=time_seconds[-1], dt=self.integration_step_size
            # )
            # zenker_pa = np.interp(time_seconds, t_zenker, solution_zenker[:, 0])
            # zenker_pv = np.interp(time_seconds, t_zenker, solution_zenker[:, 1])

            # Calculate derivatives
            sde_derivatives = np.zeros_like(control_mean_patient)
            sde_derivatives[1:] = np.diff(control_mean_patient, axis=0) / 10.0
            if sde_derivatives.shape[0] > 1:
                sde_derivatives[0] = sde_derivatives[1]

            # Create 1 + 2*num_controls subplots stacked vertically: state then derivative per control
            num_controls = (
                control_mean_patient.shape[1] if control_mean_patient.ndim == 2 else 1
            )
            nrows = 1 + 2 * num_controls
            fig, axes = plt.subplots(
                nrows, 1, figsize=(8, 2.0 * nrows + 3), sharex=True
            )
            if nrows == 1:
                axes = [axes]
            ax1 = axes[0]

            # === TOP PANEL: Use same styling as uncertainty plots ===
            arterial_true = true_patient[:, 0].copy()
            arterial_pred = pred_mean_patient[:, 0].copy()
            arterial_std = pred_std_patient[:, 0].copy()
            arterial_true[~arterial_mask] = np.nan
            arterial_pred[~arterial_mask] = np.nan
            arterial_std[~arterial_mask] = np.nan

            ax1.plot(
                time_seconds,
                arterial_true,
                color=colors["arterial_true"],
                linestyle="-",
                linewidth=2.0,
                label="Arterial BP (true)",
            )
            ax1.plot(
                time_seconds,
                arterial_pred,
                color=colors["arterial_pred"],
                linestyle="--",
                linewidth=1.5,
                label="Arterial BP (predicted)",
            )
            # (Zenker arterial baseline removed)
            ax1.fill_between(
                time_seconds,
                arterial_pred - arterial_std,
                arterial_pred + arterial_std,
                color=colors["arterial_pred"],
                alpha=0.3,
                zorder=1,
            )
            # Overlay a few per-sample predicted arterial traces when S>1
            if predictions_full.shape[1] > 1:
                samples_np = (
                    predictions_full[patient_idx].detach().cpu().numpy()
                )  # [S,T,C]
                max_overlays = min(samples_np.shape[0], 6)
                for s_idx in range(max_overlays):
                    sample_line = samples_np[s_idx, :, 0].copy()
                    sample_line[~arterial_mask] = np.nan
                    ax1.plot(
                        time_seconds,
                        sample_line,
                        color=colors["arterial_pred"],
                        linewidth=0.8,
                        linestyle="-",
                        alpha=0.35,
                    )

            venous_true = true_patient[:, 1].copy()
            venous_pred = pred_mean_patient[:, 1].copy()
            venous_std = pred_std_patient[:, 1].copy()
            venous_true[~venous_mask] = np.nan
            venous_pred[~venous_mask] = np.nan
            venous_std[~venous_mask] = np.nan

            ax1.plot(
                time_seconds,
                venous_true,
                color=colors["venous_true"],
                linestyle="-",
                linewidth=2.0,
                label="Venous BP (true)",
            )
            ax1.plot(
                time_seconds,
                venous_pred,
                color=colors["venous_pred"],
                linestyle="--",
                linewidth=1.5,
                label="Venous BP (predicted)",
            )
            # (Zenker venous baseline removed)
            ax1.fill_between(
                time_seconds,
                venous_pred - venous_std,
                venous_pred + venous_std,
                color=colors["venous_pred"],
                alpha=0.3,
                zorder=1,
            )
            # Overlay a few per-sample predicted venous traces when S>1
            if predictions_full.shape[1] > 1:
                samples_np = (
                    predictions_full[patient_idx].detach().cpu().numpy()
                )  # [S,T,C]
                max_overlays = min(samples_np.shape[0], 6)
                for s_idx in range(max_overlays):
                    sample_line = samples_np[s_idx, :, 1].copy()
                    sample_line[~venous_mask] = np.nan
                    ax1.plot(
                        time_seconds,
                        sample_line,
                        color=colors["venous_pred"],
                        linewidth=0.8,
                        linestyle="-",
                        alpha=0.35,
                    )

            ax1.set_xlim(0, 1200)
            ax1.set_ylabel("Pressure (mmHg)", fontweight="bold")
            ax1.legend(
                loc="upper right",
                fancybox=False,
                facecolor="white",
                framealpha=1.0,
                fontsize=7,
            )
            ax1.grid(True, alpha=0.2)

            # === Controls: one subplot per control dimension ===
            control_cmap = plt.cm.get_cmap("tab10", num_controls)
            for control_idx in range(num_controls):
                axc = axes[1 + 2 * control_idx]
                # Integrated control state and its finite-difference derivative
                deriv_values = (
                    control_drift[patient_idx, :, control_idx]
                    .detach()
                    .cpu()
                    .numpy()
                    .copy()
                )
                deriv_values[~bp_available_mask] = np.nan
                control_values = control_mean_patient[:, control_idx].copy()
                control_std_values = control_std_patient[:, control_idx].copy()
                control_values[~bp_available_mask] = np.nan
                control_std_values[~bp_available_mask] = np.nan

                color = control_cmap(control_idx)

                axc.plot(
                    time_seconds,
                    control_values,
                    color=color,
                    linewidth=2.0,
                    linestyle="--",
                    label=f"Control {control_idx + 1}",
                )
                axc.fill_between(
                    time_seconds,
                    control_values - control_std_values,
                    control_values + control_std_values,
                    color=color,
                    alpha=0.35,
                )
                # Overlay per-sample control traces if multiple samples
                if i_ext_path.shape[1] > 1:
                    sample_controls = (
                        i_ext_path[patient_idx, :, :, control_idx]
                        .detach()
                        .cpu()
                        .numpy()
                    )  # [S, T]
                    for s_idx in range(min(sample_controls.shape[0], 6)):
                        axc.plot(
                            time_seconds,
                            sample_controls[s_idx],
                            color=color,
                            linewidth=0.8,
                            alpha=0.35,
                        )
                axc.plot(
                    time_seconds,
                    deriv_values,
                    color=color,
                    linewidth=1.0,
                    linestyle="-",
                    alpha=0.7,
                    label=f"d(Control {control_idx + 1})/dt",
                )
                axc.set_ylabel(f"Ctrl {control_idx + 1}")
                axc.grid(True, alpha=0.2)
                axc.axhline(y=0, color="black", linestyle=":", alpha=0.4)
                # Separate derivative subplot directly under the control state
                axd = axes[1 + 2 * control_idx + 1]
                # Compute per-sample finite-difference derivatives for uncertainty and overlays
                samples_for_ctrl = (
                    i_ext_path[patient_idx, :, :, control_idx].detach().cpu().numpy()
                )  # [S, T]
                if samples_for_ctrl.shape[1] > 1:
                    sample_derivs = np.diff(samples_for_ctrl, axis=1) / 10.0  # [S, T-1]
                    # pad first timestep by copying the first diff to align lengths
                    first_col = sample_derivs[:, :1]
                    sample_derivs = np.concatenate(
                        [first_col, sample_derivs], axis=1
                    )  # [S, T]
                else:
                    sample_derivs = np.zeros_like(samples_for_ctrl)
                deriv_mean_line = sample_derivs.mean(axis=0)
                deriv_std_line = sample_derivs.std(axis=0, ddof=0)

                # Mean derivative line
                axd.plot(
                    time_seconds,
                    deriv_mean_line,
                    color=color,
                    linewidth=1.2,
                    linestyle="-",
                    alpha=0.9,
                    label=f"d(Control {control_idx + 1})/dt (mean)",
                )
                # Std band
                axd.fill_between(
                    time_seconds,
                    deriv_mean_line - deriv_std_line,
                    deriv_mean_line + deriv_std_line,
                    color=color,
                    alpha=0.25,
                )
                # Overlay a few sample derivative traces for visibility
                max_overlays = min(sample_derivs.shape[0], 6)
                for s_idx in range(max_overlays):
                    axd.plot(
                        time_seconds,
                        sample_derivs[s_idx],
                        color=color,
                        linewidth=0.7,
                        linestyle="-",
                        alpha=0.3,
                    )
                axd.set_ylabel(f"dCtrl {control_idx + 1}/dt")
                axd.grid(True, alpha=0.2)
                axd.axhline(y=0, color="black", linestyle=":", alpha=0.4)

            axes[-1].set_xlabel("Time (seconds)", fontweight="bold")
            # Single legend for last controls axis
            handles, labels = axes[-1].get_legend_handles_labels()
            if handles:
                axes[-1].legend(
                    handles,
                    labels,
                    loc="upper right",
                    fancybox=False,
                    facecolor="white",
                    framealpha=1.0,
                    fontsize=7,
                )

            epoch_tag = f"epoch{int(getattr(self, 'current_epoch', 0)):03d}_step{int(getattr(self, 'global_step', 0)):06d}"
            plt.suptitle(
                f"{epoch_tag} – Patient {patient_idx} (Batch {batch_idx})",
                fontweight="bold",
            )
            plt.tight_layout()

            if self.log_wandb:
                wandb.log(
                    {
                        f"enhanced_control_plot_batch_{batch_idx}_patient_{patient_idx}": wandb.Image(
                            plt
                        )
                    }
                )
            else:
                out_path = os.path.join(
                    self.train_dir,
                    f"control_plots/{epoch_tag}_patient{patient_idx}_batch{batch_idx}_controls.png",
                )
                plt.savefig(out_path, dpi=300, bbox_inches="tight")
                print(f"[PLOT] Saved: {out_path}")
            plt.close()

    def plot_mse_evolution(self, chart_type):
        fig = go.Figure()

        # Plot each batch element's factual MSE evolution
        for i, mse_list in enumerate(self.mse_data_factual):
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(mse_list))),
                    y=sorted(mse_list),  # Sort if needed, or just plot as is
                    mode="lines+markers",
                    name=f"Factual Batch {i+1}",
                )
            )

        # Similarly for counterfactual MSEs
        for i, mse_list in enumerate(self.mse_data_cf):
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(mse_list))),
                    y=sorted(mse_list),
                    mode="lines+markers",
                    name=f"Counterfactual Batch {i+1}",
                )
            )

        fig.update_layout(
            title="MSE Evolution Over Validation Steps",
            xaxis_title="Validation Step",
            yaxis_title="Mean Squared Error",
            legend_title="Batch Element",
        )

        plot_filename = os.path.join(
            self.train_dir,
            f"Grouped_MSE_{chart_type}_global_step_{self.global_step}.png",
        )
        fig.write_image(plot_filename, engine="kaleido")
        # print(f'Saved figure at: {plot_filename}')

        # Optionally log the plot to wandb if logging is enabled
        if self.log_wandb:
            wandb.log({"Grouped MSE Plot": fig})

        fig.data = []

    def _save_control_time_series(self, i_ext_path, batch_idx):
        """Save control mean/std over time to CSV for quick inspection."""
        try:
            os.makedirs(os.path.join(self.train_dir, "control_csvs"), exist_ok=True)
            # Stats over SDE samples dimension
            control_mean = i_ext_path.mean(1).detach().cpu().numpy()  # [B, T, D]
            # Use unbiased=False to avoid NaNs when num_samples == 1
            control_std_t = i_ext_path.std(1, unbiased=False)
            # If still degenerate, set std to zeros
            if i_ext_path.shape[1] <= 1:
                control_std_t = torch.zeros_like(control_std_t)
            control_std = control_std_t.detach().cpu().numpy()  # [B, T, D]

            time_seconds = (np.arange(control_mean.shape[1]) * 10).reshape(-1, 1)
            num_dims = control_mean.shape[2]
            for patient_idx in range(min(3, control_mean.shape[0])):
                # Build columns: time, mean_d0..mean_d{D-1}, std_d0..std_d{D-1}
                data = [time_seconds]
                for d in range(num_dims):
                    data.append(control_mean[patient_idx, :, d : d + 1])
                for d in range(num_dims):
                    data.append(control_std[patient_idx, :, d : d + 1])
                mat = np.concatenate(data, axis=1)

                header = (
                    ["time_sec"]
                    + [f"mean_d{d}" for d in range(num_dims)]
                    + [f"std_d{d}" for d in range(num_dims)]
                )
                out_path = os.path.join(
                    self.train_dir,
                    "control_csvs",
                    f"batch_{batch_idx}_patient_{patient_idx}_controls.csv",
                )
                np.savetxt(
                    out_path, mat, delimiter=",", header=",".join(header), comments=""
                )
                print(f"[PLOT] Saved control CSV (stats over samples): {out_path}")
        except Exception as e:
            if self.debug:
                print(f"[WARN] Failed to save control CSVs: {e}")
