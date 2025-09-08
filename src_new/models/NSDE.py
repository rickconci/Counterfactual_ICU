import os
import sys
import math
from collections import defaultdict
import typing as t

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
import torch.nn.functional as F

import torch.utils.checkpoint as checkpoint

# Add the project root and utils directory to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
utils_path = os.path.join(project_root, "src_new", "utils")
sys.path.insert(0, project_root)
sys.path.insert(0, utils_path)

from src_new.utils.train_utils import zenker_derivatives
from src_new.utils.utils_beta import (
    CV_params,
    CV_params_divisors,
    LinearScheduler,
    MLPSimple,
    _stable_division,
    select_tensor_by_index_list_advanced,
    activate_auto_debug_mode,
    check_for_nan_inf,
    fail_on_nan_inf,
)

# <<< Global DEBUG flag for model_beta.py, to be set by instance >>>
# This is more of a placeholder if a module-level default is ever needed,
# but instance-level self.debug passed from main_beta.py is the primary control.
DEBUG = False

# Global auto-debug mode that activates on first NaN/Inf detection
AUTO_DEBUG_ACTIVATED = False


def activate_auto_debug_mode(model_instance, location: str, tensor_name: str, tensor_value=None):
    """Activate global debug mode and log comprehensive state information."""
    global AUTO_DEBUG_ACTIVATED, DEBUG

    if not AUTO_DEBUG_ACTIVATED:
        AUTO_DEBUG_ACTIVATED = True
        DEBUG = True
        model_instance.debug = True

        print("\n" + "=" * 80)
        print("🚨 AUTO-DEBUG MODE ACTIVATED 🚨")
        print("=" * 80)
        print(f"Location: {location}")
        print(f"Tensor: {tensor_name}")
        print(f"Global step: {getattr(model_instance, 'global_step', 'unknown')}")
        print(f"Current epoch: {getattr(model_instance, 'current_epoch', 'unknown')}")
        print("=" * 80)

        # Log tensor statistics if provided
        if tensor_value is not None:
            try:
                print(f"Tensor shape: {tensor_value.shape}")
                print(f"Tensor dtype: {tensor_value.dtype}")
                print(f"Tensor device: {tensor_value.device}")
                print(f"NaN count: {torch.isnan(tensor_value).sum().item()}")
                print(f"Inf count: {torch.isinf(tensor_value).sum().item()}")
                print(f"Min value: {tensor_value.min().item()}")
                print(f"Max value: {tensor_value.max().item()}")
                print(f"Mean value: {tensor_value.mean().item()}")
                print(f"Std value: {tensor_value.std().item()}")
            except Exception as e:
                print(f"Could not log tensor stats: {e}")

        # Log model state
        try:
            print("\nModel State:")
            print(f"  use_encoder: {model_instance.use_encoder}")
            print(f"  normalise_for_SDENN: {model_instance.normalise_for_SDENN}")
            print(f"  controller_type: {model_instance.controller_type}")
            print(f"  SDE_control_weighting: {model_instance.SDE_control_weighting}")
            print(f"  learning_rate: {model_instance.learning_rate}")
            print(f"  SDEnet_out_dims: {model_instance.SDEnet_out_dims}")
            print(f"  expert_latent_dims: {model_instance.expert_latent_dims}")
            print(f"  num_samples: {model_instance.num_samples}")
            print(f"  integration_step_size: {model_instance.integration_step_size}")
            print(f"  integration_method: {model_instance.integration_method}")
        except Exception as e:
            print(f"Could not log model state: {e}")

        # Log parameter statistics
        try:
            print("\nParameter Statistics:")
            total_params = 0
            for name, param in model_instance.named_parameters():
                if param is not None:
                    total_params += param.numel()
                    if torch.isnan(param).any() or torch.isinf(param).any():
                        print(f"  ❌ {name}: {param.shape} - CONTAINS NaN/Inf!")
                    else:
                        print(f"  ✅ {name}: {param.shape} - OK")
            print(f"  Total parameters: {total_params:,}")
        except Exception as e:
            print(f"Could not log parameter stats: {e}")

        try:
            if hasattr(model_instance, '_debug_predicted_traj') and hasattr(model_instance, '_debug_true_traj'):
                print("\n🔍 FULL TRAJECTORIES (First 5 samples):")
                pred_traj = model_instance._debug_predicted_traj
                true_traj = model_instance._debug_true_traj

                batch_size = min(5, pred_traj.shape[0])
                for i in range(batch_size):
                    print(f"\n--- Sample {i} ---")
                    print(f"True trajectory ({true_traj[i].shape}):")
                    print(true_traj[i].detach().cpu().numpy())

                    if pred_traj.shape[1] > 1:  # Multiple samples
                        pred_mean = pred_traj[i].mean(0)
                        print(f"Predicted trajectory (mean over {pred_traj.shape[1]} samples):")
                        print(pred_mean.detach().cpu().numpy())

                        # Also print individual samples for first patient
                        if i == 0:
                            print(f"Individual prediction samples for patient 0:")
                            for s_idx in range(min(3, pred_traj.shape[1])):
                                print(f"  Sample {s_idx}:")
                                print(f"  {pred_traj[i, s_idx].detach().cpu().numpy()}")
                    else:
                        print(f"Predicted trajectory:")
                        print(pred_traj[i, 0].detach().cpu().numpy())
                print("--- End Trajectories ---\n")
        except Exception as e:
            print(f"Could not print trajectories: {e}")

        print("=" * 80 + "\n")

        # Set breakpoint for debugging
        import pdb;
        pdb.set_trace()


def check_for_nan_inf(tensor, name: str, location: str, model_instance=None, raise_error: bool = True):
    """
    Comprehensive NaN/Inf detection with detailed logging.

    Args:
        tensor: Tensor to check
        name: Name of the tensor for logging
        location: Location where the check is performed
        model_instance: Model instance for context
        raise_error: Whether to raise an error on NaN/Inf detection
    """
    if tensor is None:
        return False

    has_nan = torch.isnan(tensor).any()
    has_inf = torch.isinf(tensor).any()

    if has_nan or has_inf:
        print(f"\n🚨 NaN/Inf detected in {name} at {location}")
        print(f"  Shape: {tensor.shape}")
        print(f"  Dtype: {tensor.dtype}")
        print(f"  Device: {tensor.device}")
        print(f"  NaN count: {torch.isnan(tensor).sum().item()}")
        print(f"  Inf count: {torch.isinf(tensor).sum().item()}")

        if has_nan:
            nan_indices = torch.isnan(tensor).nonzero()
            print(f"  NaN indices (first 10): {nan_indices[:10]}")

        if has_inf:
            inf_indices = torch.isinf(tensor).nonzero()
            print(f"  Inf indices (first 10): {inf_indices[:10]}")

        if model_instance is not None:
            activate_auto_debug_mode(model_instance, location, name, tensor)

        if raise_error:
            raise RuntimeError(f"NaN/Inf detected in {name} at {location}")

    return has_nan or has_inf


def fail_on_nan_inf(param, name: str, location: str, model_instance=None):
    """
    Fail fast on NaN/Inf values with comprehensive debugging.

    Args:
        param: Parameter tensor to check
        name: Name of the parameter
        location: Location where check occurs
        model_instance: Model instance for context
    """
    if param is None:
        return

    has_nan = torch.isnan(param).any()
    has_inf = torch.isinf(param).any()

    if has_nan or has_inf:
        print(f"[ERROR] NaN/Inf weights in {name}! FAILING FAST - NO SANITIZATION!")

        if model_instance is not None:
            activate_auto_debug_mode(model_instance, location, name, param)

        raise RuntimeError(f"NaN/Inf detected in {name} at {location}. Failing fast for debugging.")


class NSDE(LightningModule):
    def __init__(
            self,
            use_encoder,
            start_dec_at_treatment,
            variational_sampling,
            # Encoder
            context_input_dim,
            chartevents_input_dim,
            encoder_hidden_dim,
            encoder_SDENN_dims,
            expert_latent_dims,
            encoder_num_layers,
            variational_encoder,
            encoder_w_time,
            encoder_reverse_time,
            use_2_5std_encoder_minmax,
            n_medications,
            med_embed_dim,
            encoder_context_len,
            # New static fusion params
            static_input_dim,
            static_hidden_dim,
            fusion_hidden_dim,
            # SDE params
            num_samples,
            normalise_for_SDENN,
            self_reverting_prior_control,
            prior_tx_sigma_per_control: t.List[float],
            prior_tx_mu,
            theta,
            SDE_control_weighting,
            use_control_lowpass,
            control_lowpass_tau,
            use_control_tv_loss,
            control_tv_weight,
            override_control_scales,
            control_energy_weight,
            # SDE model params
            SDE_input_state,
            include_time,
            SDEnet_hidden_dim,
            SDEnet_depth,
            SDEnet_out_dims,
            use_batch_norm,
            final_activation,
            # SDE Integration
            integration_step_size,
            integration_method,
            atol,
            rtol,
            integration_adaptive,
            # decoder params
            decoder_output_dims,
            normalised_data,
            log_lik_output_scale,
            # admin
            train_dir,
            KL_weighting_SDE,
            loss_type,
            log_lik_scale_mode,
            anneal_iters,
            # Optimizer params
            use_lr_scheduler,
            total_training_steps,
            warmup_steps,
            min_lr,
            learning_rate,
            optimizer_name,
            log_wandb,
            adjoint,
            plot_every,
            batch_size,
            dataset,
            test_zenker,
            debug,
            force_no_controls,
            plot_outputs_train,
            # Controller selection (MLP vs GAT)
            controller_type,
            gat_heads,
            gat_layers,
            gat_hidden,
            gat_dropout,
            sde_burn_in_period,
            log_train_combo_loss_every_n_steps,
            scale_loss_by_variance,
            ic_consistency_weight,
            forward_loss_weight,
            use_wandb_for_logging,
            use_checkpointing,
            debug_level,
            force_zenker_defaults,
            plot_control_samples,
            plot_include_burn_in=True,
            id_to_combo_map=None,
            **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Store all hyperparameters
        self.use_encoder = use_encoder
        self.start_dec_at_treatment = start_dec_at_treatment
        self.variational_sampling = variational_sampling
        self.context_input_dim = context_input_dim
        self.chartevents_input_dim = chartevents_input_dim
        self.encoder_hidden_dim = encoder_hidden_dim
        self.encoder_SDENN_dims = encoder_SDENN_dims
        self.expert_latent_dims = expert_latent_dims
        self.encoder_num_layers = encoder_num_layers
        self.variational_encoder = variational_encoder
        self.encoder_w_time = encoder_w_time
        self.encoder_reverse_time = encoder_reverse_time
        self.use_2_5std_encoder_minmax = use_2_5std_encoder_minmax
        self.n_medications = n_medications
        self.med_embed_dim = med_embed_dim
        self.encoder_context_len = encoder_context_len
        self.static_input_dim = static_input_dim
        self.static_hidden_dim = static_hidden_dim
        self.fusion_hidden_dim = fusion_hidden_dim
        self.num_samples = num_samples
        self.normalise_for_SDENN = normalise_for_SDENN
        self.self_reverting_prior_control = self_reverting_prior_control
        self.prior_tx_sigma_per_control = prior_tx_sigma_per_control
        self.prior_tx_mu = prior_tx_mu
        self.theta = theta
        self.SDE_control_weighting = SDE_control_weighting
        self.use_control_lowpass = use_control_lowpass
        self.control_lowpass_tau = control_lowpass_tau
        self.use_control_tv_loss = use_control_tv_loss
        self.control_tv_weight = control_tv_weight
        self.override_control_scales = override_control_scales
        self.control_energy_weight = control_energy_weight
        self.SDE_input_state = SDE_input_state
        self.include_time = include_time
        self.SDEnet_hidden_dim = SDEnet_hidden_dim
        self.SDEnet_depth = SDEnet_depth
        # Flag: use direct pressure controls (C1->dpa_dt, C2->dpv_dt)
        self.direct_pressure_controls = bool(kwargs.get("direct_pressure_controls", False))
        # Determine control head count based on flag
        self.SDEnet_out_dims = 2
        self.use_batch_norm = use_batch_norm
        self.final_activation = final_activation
        self.integration_step_size = integration_step_size
        self.integration_method = integration_method
        self.atol = atol
        self.rtol = rtol
        self.integration_adaptive = integration_adaptive
        self.decoder_output_dims = decoder_output_dims
        self.normalised_data = normalised_data
        self.log_lik_output_scale = log_lik_output_scale
        self.train_dir = train_dir
        self.KL_weighting_SDE = KL_weighting_SDE
        self.loss_type = loss_type
        self.log_lik_scale_mode = log_lik_scale_mode
        self.anneal_iters = anneal_iters
        self.use_lr_scheduler = use_lr_scheduler
        self.total_training_steps = total_training_steps
        self.warmup_steps = warmup_steps
        self.min_lr = min_lr
        self.learning_rate = learning_rate
        self.optimizer_name = optimizer_name
        self.log_wandb = log_wandb
        self.adjoint = adjoint
        self.plot_every = plot_every
        self.batch_size = batch_size
        self.dataset = dataset
        self.test_zenker = test_zenker
        self.debug = debug
        self.force_no_controls = force_no_controls
        self.plot_outputs_train = plot_outputs_train
        self.controller_type = controller_type
        self.gat_heads = gat_heads
        self.gat_layers = gat_layers
        self.gat_hidden = gat_hidden
        self.gat_dropout = gat_dropout
        self.sde_burn_in_period = sde_burn_in_period
        self.log_train_combo_loss_every_n_steps = log_train_combo_loss_every_n_steps
        self.scale_loss_by_variance = scale_loss_by_variance
        self.ic_consistency_weight = ic_consistency_weight
        self.forward_loss_weight = forward_loss_weight
        self.use_wandb_for_logging = use_wandb_for_logging
        self.use_checkpointing = use_checkpointing
        self.debug_level = debug_level
        self.force_zenker_defaults = force_zenker_defaults
        self.plot_control_samples = plot_control_samples
        self.plot_include_burn_in = bool(plot_include_burn_in)

        self.noise_type = "diagonal"  # required
        self.sde_type = "ito"  # required
        self.sdeint_fn = torchsde.sdeint_adjoint if self.hparams.adjoint else torchsde.sdeint

        ### ADMIN
        self.train_dir = train_dir
        self.learning_rate = learning_rate
        self.log_wandb = log_wandb
        self.plot_every = plot_every
        self.sde_burn_in_period = sde_burn_in_period

        # Store the med combo map from the dataset or explicit argument
        self.id_to_combo_map = {}
        try:
            if id_to_combo_map:
                self.id_to_combo_map = id_to_combo_map
            elif hasattr(dataset, 'id_to_combo_map') and getattr(dataset, 'id_to_combo_map'):
                self.id_to_combo_map = dataset.id_to_combo_map
            elif hasattr(dataset, 'id_to_combo') and getattr(dataset, 'id_to_combo'):
                raw_map = dataset.id_to_combo
                formatted = {}
                for k, combo in raw_map.items():
                    meds = combo if isinstance(combo, (list, tuple)) else (combo,)
                    meds_sorted = sorted(list(meds))
                    formatted[int(k)] = {
                        "str": ", ".join(map(str, meds_sorted)),
                        "details": [{"med_name": str(name)} for name in meds_sorted],
                    }
                self.id_to_combo_map = formatted
        except Exception:
            self.id_to_combo_map = {}

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
            max_len_ctx = (
                120 if encoder_context_len is None else int(encoder_context_len)
            )
            # Encoder for 60-minute physiological context
            self.context_encoder_60m = Raindrop_v2(
                d_inp=context_input_dim,
                d_model=encoder_hidden_dim,
                output_dim=encoder_hidden_dim,
                nhead=4,
                nhid=128,
                max_len=max_len_ctx,
                global_structure=torch.ones(context_input_dim, context_input_dim),
                nlayers=encoder_num_layers,
                static=False,
                debug=self.debug,
            )
            # Encoder for 24-hour chartevents context
            self.context_encoder_24h = Raindrop_v2(
                d_inp=chartevents_input_dim,
                d_model=encoder_hidden_dim,
                output_dim=encoder_hidden_dim,
                nhead=4,
                nhid=128,
                max_len=24,  # 24 hours, 1h interval
                global_structure=torch.ones(chartevents_input_dim, chartevents_input_dim),
                nlayers=encoder_num_layers,
                static=False,
                debug=self.debug,
            )

            # Compute actual encoder output dims from Raindrop internals
            def _rd_out_dim(enc):
                if enc is None:
                    return 0
                # Raindrop returns [B, d_inp*d_ob + 16] when sensor_wise_mask=False
                # and [B, d_inp*(d_ob+16)] when sensor_wise_mask=True
                if getattr(enc, 'sensor_wise_mask', False):
                    return int(enc.d_inp) * int(enc.d_ob + 16)
                return int(enc.d_inp) * int(enc.d_ob) + 16

            rd24_dim = _rd_out_dim(self.context_encoder_24h)
            rd60_dim = _rd_out_dim(self.context_encoder_60m)
            temporal_embedding_dim = rd24_dim + rd60_dim
            self.fused_expected_dim = temporal_embedding_dim + static_hidden_dim
        else:
            self.context_encoder_60m = None
            self.context_encoder_24h = None
            self.fused_expected_dim = static_hidden_dim

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
            # This MLP fuses the temporal and static embeddings to produce the neural_embedding
            self.fusion_mlp = MLPSimple(
                input_dim=self.fused_expected_dim,
                output_dim=encoder_SDENN_dims,  # Directly outputs the neural embedding
                hidden_dim=max(64, self.fused_expected_dim // 2),
                depth=2,
                activations=[nn.ReLU(), nn.ReLU()],
                debug=self.debug,
            )
            # DEPRECATED: Heads for predicting initial conditions are removed
            self.ode_latent_head = None
            self.neural_embedding_head = None
        else:
            self.static_encoder = None
            self.fusion_mlp = None
            self.ode_latent_head = None
            self.neural_embedding_head = None
        # --- End New ---

        ### PRIOR PARAMS
        self.self_reverting_prior_control = self_reverting_prior_control
        if prior_tx_sigma_per_control is None:
            # Defaults based on 10% of original control scales
            self.prior_tx_sigma_per_control = torch.tensor([0.01, 0.002, 0.001, 0.001], dtype=torch.float32)
        else:
            self.prior_tx_sigma_per_control = torch.tensor(prior_tx_sigma_per_control, dtype=torch.float32)

        if self.prior_tx_sigma_per_control.shape[0] != SDEnet_out_dims:
            raise ValueError(
                f"Length of prior_tx_sigma_per_control ({self.prior_tx_sigma_per_control.shape[0]}) must match SDEnet_out_dims ({SDEnet_out_dims})")

        self.prior_tx_mu = prior_tx_mu
        self.noise_scale = prior_tx_sigma

        # sigma_values = torch.tensor(list(CV_params_prior_sigma.values())).float()
        # sigma_values = sigma_values[:expert_latent_dims].view(1, -1)
        # self.register_buffer('sigma', sigma_values.clone())
        self.sigma = (
            torch.tensor(self.prior_tx_sigma_per_control, dtype=torch.float32)
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
        # Do NOT overwrite the earlier direct-control override
        if not getattr(self, "direct_pressure_controls", False):
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
        # Ratio cap for diffusion vs. drift scaling on controls (sigma <= ratio * weight)
        try:
            self.control_sigma_ratio = float(kwargs.get("control_sigma_ratio", 0.1))
        except Exception:
            self.control_sigma_ratio = 0.1
        # Limit number of patients plotted per call to reduce I/O spam in DDP
        try:
            self.max_plotted_patients = int(kwargs.get("max_plotted_patients", 4))
        except Exception:
            self.max_plotted_patients = 4

        net_input_dims = (
            self.encoder_output_dim
            if SDE_input_state == "full"
            else self.encoder_output_dim - len(encoder_input_dim)
        )
        net_input_dims = net_input_dims + 2 if include_time else net_input_dims

        # Medication embedding config (project variable med context dims -> fixed size)
        self.n_medications = int(n_medications)
        self.med_embed_dim = med_embed_dim
        # Initialize explicitly to avoid Lazy params (DDP requires materialized weights)
        # Input dimension equals the number of medication features per time step (per-med features = 5 -> rate, pre_on, cumulative, decay, trigger)
        # So total input dim is 5 * M
        if int(self.n_medications) > 0:
            self.med_proj = nn.Linear(int(self.n_medications) * 5, int(self.med_embed_dim), bias=True)
        else:
            self.med_proj = None

        if self.use_encoder != "none":
            self.ic_consistency_weight = 0
            # add fixed med embedding dims
            net_input_dims = net_input_dims + self.med_embed_dim
        else:
            self.ic_consistency_weight = 0
            # expert latents + fixed med embedding dims + neural embedding
            net_input_dims = self.expert_latent_dims + self.med_embed_dim + encoder_SDENN_dims
            net_input_dims = net_input_dims + 2 if include_time else net_input_dims

        # Append physics-derived feature count (ΔP, r_tpr, f_hr, F, dpa_base, dpv_base, sigma, s_dot)
        net_input_dims = 2  # Current pressures [p_a, p_v]
        net_input_dims += self.encoder_SDENN_dims  # Neural embedding
        net_input_dims += 2 if include_time else 0  # Time encoding [sin(t), cos(t)]
        net_input_dims += self.med_embed_dim

        activations = {"relu": nn.ReLU(), "tanh": nn.Tanh(), "none": None}
        final_activation_real = activations[final_activation.lower()]

        # TODO change net input dims to be 14 + number of meds if there is no encoder, else encoder dim + 14 + meds

        self.SDEnet = MLPSimple(
            input_dim=net_input_dims,
            output_dim=self.SDEnet_out_dims,
            hidden_dim=SDEnet_hidden_dim,
            depth=SDEnet_depth,
            activations=[nn.Tanh() for _ in range(SDEnet_depth)],
            final_activation=final_activation_real,
            use_batch_norm=use_batch_norm,
            debug=self.debug,
        )
        self._initial_sde_weights = None
        for name, param in self.SDEnet.named_parameters():
            if 'weight' in name:
                self._initial_sde_weights = param.data.clone()
                break

                # Per-head control scales applied to SDE NN outputs (post-final-layer)
        # Head 0 (dpv_dt): ~0.1 mmHg/s
        # Head 1 (dsv_dt): ~0.02 units per time step
        # Head 2 (dt_ca): ~0.01 units per time step
        # Head 3 (dt_r_tpr_mod): ~0.01 units per time step
        default_scales = torch.tensor([0.1, 0.02, 0.01, 0.01], dtype=torch.float32)
        if self.SDEnet_out_dims != len(default_scales):
            # Fallback: repeat or truncate to match dims
            repeats = (self.SDEnet_out_dims + len(default_scales) - 1) // len(default_scales)
            default_scales = default_scales.repeat(repeats)[: self.SDEnet_out_dims]
        # Allow override via CLI
        if isinstance(override_scales, str) and override_scales.strip() != "":
            try:
                pieces = [float(x) for x in override_scales.split(",")]
                ov = torch.tensor(pieces, dtype=torch.float32)
                if ov.numel() < self.SDEnet_out_dims:
                    reps = (self.SDEnet_out_dims + ov.numel() - 1) // ov.numel()
                    ov = ov.repeat(reps)[: self.SDEnet_out_dims]
                else:
                    ov = ov[: self.SDEnet_out_dims]
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
        self.sde_burn_in_period = float(sde_burn_in_period)
        self.log_train_combo_loss_every_n_steps = int(log_train_combo_loss_every_n_steps)
        self.scale_loss_by_variance = scale_loss_by_variance

        # For per-med combo loss tracking
        self.train_combo_loss_accumulator = defaultdict(lambda: {'loss': 0.0, 'loss_sq': 0.0, 'count': 0})
        self.val_combo_losses = []
        self.test_combo_losses = []

        # Store optimizer params
        self.use_lr_scheduler = use_lr_scheduler
        self.total_training_steps = total_training_steps
        self.warmup_steps = warmup_steps
        self.min_lr = min_lr

        # Debug helper state
        self._forward_hook_handles = []
        self._last_sdnet_io_stats = None

        ### LOSS
        self.loss_type = loss_type
        self.log_lik_scale_mode = log_lik_scale_mode
        self.MSE_loss = nn.MSELoss(reduction="none")

        # if self.log_lik_scale_mode == 'learnable':
        #     # We learn log(scale) for stability. Initialize with log of the provided value.
        #     initial_log_scale = torch.log(torch.tensor(log_lik_output_scale, dtype=torch.float32))
        #     # Create a learnable parameter for each output dimension
        #     self.log_scale_param = nn.Parameter(
        #         torch.full((self.decoder_output_dims,), initial_log_scale)
        #     )
        #     # The original log_lik_output_scale is now just a config value, not used directly in loss
        #     self.initial_log_lik_output_scale = log_lik_output_scale
        # elif self.log_lik_scale_mode == 'annealing':
        #     # Can be parameterized later if needed
        #     anneal_start_val = 10.0  # Start with high uncertainty
        #     anneal_end_val = log_lik_output_scale  # End at the provided value
        #     self.scale_scheduler = LinearScheduler(iters=anneal_iters, startval=anneal_start_val, endval=anneal_end_val, start=0)
        #     self.log_lik_output_scale = anneal_start_val # Store the initial value for clarity, but scheduler.val is used
        # else: # fixed
        #     self.log_lik_output_scale = log_lik_output_scale

        ### ADMIN
        self.train_dir = train_dir
        self.KL_weighting_SDE = KL_weighting_SDE

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
            "p_a": (40, 180.0),
            "p_v": (0.0, 39),
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


        # Log-likelihood scale configuration
        if self.hparams.log_lik_scale_mode == "learnable":
            # Learn per-output log-scale; initialize from provided scale
            initial_log_scale = torch.log(
                torch.as_tensor(self.hparams.log_lik_output_scale, dtype=torch.float32)
            )
            # Determine number of output channels
            if isinstance(self.hparams.decoder_output_dims, (list, tuple)):
                num_outputs = int(sum(self.hparams.decoder_output_dims))
            else:
                num_outputs = int(self.hparams.decoder_output_dims)
            self.log_scale_param = nn.Parameter(
                torch.full((num_outputs,), initial_log_scale, dtype=torch.float32)
            )
            # Keep a scalar for debug logging purposes
            self.log_lik_output_scale = float(self.hparams.log_lik_output_scale)
        elif self.hparams.log_lik_scale_mode == "annealing":
            anneal_start_val = 10.0
            anneal_end_val = float(self.hparams.log_lik_output_scale)
            self.scale_scheduler = LinearScheduler(
                iters=self.hparams.anneal_iters, startval=anneal_start_val, endval=anneal_end_val, start=0
            )
            self.log_lik_output_scale = anneal_start_val
        else:
            # fixed
            self.log_lik_output_scale = float(self.hparams.log_lik_output_scale)

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

    def get_adaptive_control_scales(self, current_step):
        """Gradually increase control authority as training progresses"""
        base_scales = torch.tensor([1, 1, 0.1, 0.1], dtype=torch.float32)

        # Start with 1% control authority, ramp up to 100% over 1000 steps
        ramp_progress = min(current_step / 50.0, 1.0)
        scale_multiplier = 0.01 + 0.99 * ramp_progress

        return base_scales * scale_multiplier

    def get_current_per_control_weights(self):
        """
        Compute per-control weights applied to controller outputs at the current training step.
        Returns a tensor of shape [SDEnet_out_dims].
        """
        adaptive = self.get_adaptive_control_scales(self.global_step)
        if adaptive.numel() != self.SDEnet_out_dims:
            repeats = (self.SDEnet_out_dims + adaptive.numel() - 1) // adaptive.numel()
            adaptive = adaptive.repeat(repeats)[: self.SDEnet_out_dims]
        return adaptive.to(self.device) * float(self.SDE_control_weighting)

    def get_current_per_control_sigmas(self):
        """
        Compute per-control diffusion sigmas constrained to be a small ratio of the
        current effective control weights to avoid diffusion dominating early training.

        Rule: sigma_i <= control_sigma_ratio * weight_i, applied elementwise.
        Falls back to base prior sigmas if they are already below the cap.
        """
        # Base per-control sigma from prior (vector length SDEnet_out_dims)
        base_sigma = self.prior_tx_sigma_per_control.to(self.device)
        # Current effective control weights (already includes global weighting)
        weights = self.get_current_per_control_weights()
        # Cap per-control sigma as a fraction of current weights
        max_sigma = float(self.control_sigma_ratio) * torch.clamp(weights, min=1e-12)
        # Use the tighter (smaller) of base_sigma and cap
        sigma = torch.minimum(base_sigma, max_sigma)
        # Ensure strictly positive to avoid degenerate diffusion
        sigma = torch.clamp(sigma, min=1e-12)
        return sigma

    def normalize_pressures_only(self, pressures):
        """Normalize only p_a, p_v using their specific mu/sigma"""
        first_two_mu = self.first_two_normalization_mu.to(pressures.device)
        first_two_sigma = self.first_two_normalization_sigma.to(pressures.device)
        normalized_pressures = (pressures - first_two_mu) / first_two_sigma
        return normalized_pressures

    def f(self, t, y):
        """
        Neural SDE drift function
        Args:
            t: current time
            y: current state [batch*samples, 2] representing [p_a, p_v]
        Returns:
            derivatives [batch*samples, 2] representing [dp_a/dt, dp_v/dt]
        """
        batch_size = y.shape[0]

        # Input validation
        if torch.isnan(y).any() or torch.isinf(y).any():
            raise RuntimeError("Non-finite values in input state y")

        # Current pressures [batch*samples, 2]
        current_pressures = y

        # Normalize pressures (using the same normalization as the hybrid model)
        if self.normalise_for_SDENN:
            normalized_pressures = self.normalize_pressures_only(current_pressures)
        else:
            normalized_pressures = current_pressures

        # Start building neural network input
        nn_input = normalized_pressures  # [batch*samples, 2]

        # Add neural embedding context (stored from forward_latent)
        if hasattr(self, 'static_neural_embedding') and self.static_neural_embedding is not None:
            nn_input = torch.cat([nn_input, self.static_neural_embedding], dim=-1)

        # Add time encoding
        if self.include_time:
            sin_time = torch.sin(torch.full_like(y[:, 0], fill_value=t)).unsqueeze(1)
            cos_time = torch.cos(torch.full_like(y[:, 0], fill_value=t)).unsqueeze(1)
            nn_input = torch.cat([nn_input, sin_time, cos_time], dim=-1)

        # Add medication context
        t_idx = torch.floor_divide((t.to(self._t0) - self._t0), 10).to(torch.long)
        t_idx = torch.clamp(t_idx, 0, self.current_med_tensors.shape[1] - 1)
        med_tensor = self.current_med_tensors[:, t_idx, :]

        # Expand over samples if needed
        samples_per_batch = batch_size // med_tensor.shape[0]
        med_tensor = med_tensor.repeat_interleave(samples_per_batch, dim=0)

        # Project medication context to embedding
        med_embed = torch.tanh(self.med_proj(med_tensor))
        nn_input = torch.cat([nn_input, med_embed], dim=-1)

        # Predict pressure derivatives using neural network
        if self.training and self.use_checkpointing and getattr(nn_input, 'requires_grad', False):
            pressure_derivatives = checkpoint.checkpoint(self.SDEnet, nn_input, use_reentrant=False)
        else:
            pressure_derivatives = self.SDEnet(nn_input)

        # Validation and stability
        if torch.isnan(pressure_derivatives).any() or torch.isinf(pressure_derivatives).any():
            raise RuntimeError("Non-finite values in SDEnet output")

        # Optional: clamp derivatives for stability
        pressure_derivatives = torch.clamp(pressure_derivatives, -10.0, 10.0)

        if self.debug:
            t_idx_debug = torch.round((t.to(self._t0) - self._t0) / self._dt_grid).to(torch.long)
            if (t_idx_debug.remainder(10) == 0).item():
                print(f"[DEBUG] Neural SDE f(): t={float(t):.1f}, y_mean=[{y.mean(0)[0]:.2f}, {y.mean(0)[1]:.2f}]")
                print(
                    f"  derivatives_mean=[{pressure_derivatives.mean(0)[0]:.4f}, {pressure_derivatives.mean(0)[1]:.4f}]")

        return pressure_derivatives

    # 5. Fixed g method:
    def g(self, t, y):
        """
        Neural SDE diffusion function with different sigmas per variable
        Args:
            t: current time
            y: current state [batch*samples, 2] representing [p_a, p_v]
        Returns:
            diffusion [batch*samples, 2]
        """
        batch_size = y.shape[0]

        # Different noise for each pressure variable
        sigma_arterial = self.noise_scale  # Full sigma for p_a
        sigma_venous = self.noise_scale / 5.0  # Reduced sigma for p_v

        diffusion = torch.zeros(batch_size, 2, device=y.device)
        diffusion[:, 0] = sigma_arterial  # Arterial pressure noise
        diffusion[:, 1] = sigma_venous  # Venous pressure noise (smaller)

        return diffusion

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
            init_latents,  # [batch, samples, 2] - just [p_a, p_v]
            ts,
            Tx,
            time_to_tx,
            neural_embedding,  # [batch, samples, encoder_dims]
            valid_lengths=None,
            med_traj_values=None,
            med_traj_mask=None,
            med_traj_time=None,
    ):
        """Neural SDE with 2D state evolution"""
        batch_size, num_samples, state_dims = init_latents.shape  # state_dims = 2

        # Store context as attributes for f() to access
        self.current_med_values = med_traj_values
        self.current_med_mask = med_traj_mask
        self.current_med_time = med_traj_time
        self.current_neural_embedding = neural_embedding

        # Store time grid helpers
        self._t0 = ts[0].to(self.device)
        self._time_len = int(ts.shape[0])
        if ts.shape[0] > 1:
            self._dt_grid = (ts[1] - ts[0]).to(self.device).clamp_min(1e-6)
        else:
            self._dt_grid = torch.tensor(float(self.integration_step_size), device=self.device)

        # Attach precomputed med_context if available
        if hasattr(self, 'current_med_tensors'):
            pass  # Already set from common_step

        # Store neural embedding flattened for f() access
        self.static_neural_embedding = neural_embedding.reshape(-1, self.encoder_SDENN_dims)

        # Initial state: just the 2 observables
        y0 = init_latents.reshape(-1, 2)  # [batch*samples, 2]

        if self.debug:
            print(f"[DEBUG] Neural SDE forward_latent:")
            print(f"  init_latents shape: {init_latents.shape}")
            print(f"  y0 shape: {y0.shape}")
            print(f"  neural_embedding shape: {neural_embedding.shape}")
            print(f"  static_neural_embedding shape: {self.static_neural_embedding.shape}")

        # Integrate neural SDE
        options = {"dtype": torch.float32}

        trajectory = self.sdeint_fn(
            sde=self,
            y0=y0,
            ts=ts,
            method=self.integration_method,
            dt=self.integration_step_size,
            adaptive=self.integration_adaptive,
            rtol=self.rtol,
            atol=self.atol,
            options=options,
            names={"drift": "f", "diffusion": "g"},  # Use simpler f, g functions
        )

        # Reshape back: [time, batch*samples, 2] -> [batch, samples, time, 2]
        time_steps = trajectory.shape[0]
        trajectory = trajectory.view(time_steps, batch_size, num_samples, 2)
        trajectory = trajectory.permute(1, 2, 0, 3)  # [batch, samples, time, 2]

        if self.debug:
            print(f"  output trajectory shape: {trajectory.shape}")

        # No control paths or complex logqp for simple neural SDE
        return trajectory, torch.zeros(batch_size), None

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

        pa_raw = latent_out[..., 0]  # [batch, samples, time]
        pv_raw = latent_out[..., 1]

        pa = torch.clamp(pa_raw, 40.0, 180.0)
        pv = torch.clamp(pv_raw, 0.0, 39.0)

        output_traj = torch.stack([pa, pv], dim=-1)

        if self.debug:
            print(
                f"[DEBUG] Hybrid_SDE forward_dec: decoded_mean_shape={output_traj.shape}"
            )

        print("=" * 50 + "\n")
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
        if self.loss_type == "nll":
            # Determine the scale based on the mode
            if self.log_lik_scale_mode == 'learnable':
                # Clamp the log_scale parameter to prevent collapse or explosion, then exponentiate.
                # Range corresponds to a scale of approx. 0.1 to 55 mmHg.
                clamped_log_scale = torch.clamp(self.log_scale_param, min=-2.3, max=4.0)
                current_scale = torch.exp(clamped_log_scale)
            elif self.log_lik_scale_mode == 'annealing':
                current_scale = self.scale_scheduler.val
            else:  # 'fixed'
                current_scale = self.log_lik_output_scale

            # NLL Loss (probabilistic)
            logpy = distributions.Normal(
                loc=predicted_traj, scale=current_scale
            ).log_prob(true_traj_expanded)

            # Per-sample loss calculation
            if mask is not None:
                mask = mask.to(device=logpy.device, dtype=logpy.dtype)
                mask_expanded = mask.unsqueeze(1).expand_as(logpy)
                logpy = logpy * mask_expanded

                # Sum over time and features, then average over samples
                per_sample_logpy = logpy.sum(dim=(2, 3)).mean(dim=1)  # [B]
                valid_count_per_sample = torch.clamp(mask.sum(dim=(1, 2)), min=1.0)  # [B]
                per_sample_loss = -per_sample_logpy / valid_count_per_sample

                # Overall loss for backward pass
                valid_count_total = torch.clamp(mask.sum(), min=1.0)
                recon_loss = -logpy.sum() / valid_count_total
            else:
                per_sample_loss = -logpy.sum(dim=(2, 3)).mean(dim=(1, 0))  # [B]
                recon_loss = per_sample_loss.mean()

        elif self.loss_type == "mse":
            # MSE Loss (deterministic on mean prediction)
            pred_mean = predicted_traj.mean(dim=1)  # [B, T, C]
            sq_error = (pred_mean - true_traj) ** 2

            # Per-sample loss calculation
            if mask is not None:
                mask = mask.to(device=sq_error.device, dtype=sq_error.dtype)
                sq_error_masked = sq_error * mask

                per_sample_sq_error = sq_error_masked.sum(dim=(1, 2))  # [B]
                valid_count_per_sample = torch.clamp(mask.sum(dim=(1, 2)), min=1.0)  # [B]
                per_sample_loss = per_sample_sq_error / valid_count_per_sample

                # Overall loss for backward pass
                valid_count_total = torch.clamp(mask.sum(), min=1.0)
                recon_loss = sq_error_masked.sum() / valid_count_total
            else:
                per_sample_loss = sq_error.mean(dim=(1, 2))  # [B]
                recon_loss = per_sample_loss.mean()
        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}")

        # Compute final reconstruction loss, optionally scaling by variance
        if self.scale_loss_by_variance:
            with torch.no_grad():
                if mask is not None:
                    mask_f = mask.to(dtype=true_traj.dtype)
                    valid_counts_per_channel = torch.clamp(mask_f.sum(dim=1), min=1.0)  # [B, C]
                    masked_traj = true_traj * mask_f
                    mean_per_channel = masked_traj.sum(dim=1) / valid_counts_per_channel  # [B, C]
                    variance_per_channel = (masked_traj ** 2).sum(
                        dim=1) / valid_counts_per_channel - mean_per_channel ** 2  # [B, C]
                else:  # No mask
                    variance_per_channel = torch.var(true_traj, dim=1, unbiased=False)  # [B, C]

                total_variance_per_sample = variance_per_channel.sum(dim=1)  # [B]
                variance_scale_factor = torch.log1p(total_variance_per_sample)

                # Normalize across batch to prevent changing the overall loss magnitude
                if variance_scale_factor.mean() > 1e-6:
                    variance_scale_factor = variance_scale_factor / variance_scale_factor.mean()
                else:
                    variance_scale_factor = torch.ones_like(variance_scale_factor)

            # Log the mean scale factor during training
            if self.training:
                self.log("train/variance_scale_factor_mean", variance_scale_factor.mean().item(), on_step=True,
                         on_epoch=False, prog_bar=False, logger=True)

            # Apply scaling to get the final reconstruction loss for the backward pass
            recon_loss = (per_sample_loss * variance_scale_factor).mean()
        else:
            # Default behavior: simple mean of per-sample losses
            recon_loss = per_sample_loss.mean()

        if self.log_lik_scale_mode == 'annealing':
            self.scale_scheduler.step()

        current_kl_weight = self.kl_scheduler.val
        self.kl_scheduler.step()

        loss = recon_loss + self.KL_weighting_SDE * current_kl_weight * logqp.mean()

        return loss, recon_loss, logqp.mean(), per_sample_loss.detach()

    def on_after_backward(self) -> None:
        # Fail-fast on non-finite gradients (always on)
        total_norm_sq = 0.0
        count = 0
        with torch.no_grad():
            for name, p in self.named_parameters():
                if p.grad is not None:
                    g = p.grad.detach()
                    if torch.isnan(g).any() or torch.isinf(g).any():
                        activate_auto_debug_mode(self, "on_after_backward", f"gradient '{name}'", g)
                        raise RuntimeError(
                            f"Non-finite gradient detected in parameter '{name}'. Auto-debug mode activated.")
                    total_norm_sq += float(g.norm(2).item() ** 2)
                    count += 1

        # Additional check for parameter values after backward pass
        for name, p in self.named_parameters():
            if p is not None:
                check_for_nan_inf(p.data, f"parameter '{name}'", "on_after_backward", self, raise_error=True)

    def on_before_optimizer_step(self, optimizer) -> None:
        # Always fail-fast if parameters contain non-finite values before stepping
        with torch.no_grad():
            for name, p in self.named_parameters():
                if p is None or p.data is None:
                    continue
                if torch.isnan(p.data).any() or torch.isinf(p.data).any():
                    activate_auto_debug_mode(self, "on_before_optimizer_step", f"parameter '{name}'", p.data)
                    raise RuntimeError(
                        f"Non-finite parameter detected before optimizer step: '{name}'. Auto-debug mode activated.")

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
            activate_auto_debug_mode(self, "_prepare_sde_initial_state", "z1_for_sde", z1_for_sde)
            raise RuntimeError(
                f"Non-finite values in final z1_for_sde: NaN={nan_cnt}, Inf={inf_cnt}. Auto-debug mode activated."
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
                ce_rd_src,
                ce_rd_times,
                ce_rd_length,
                static_features,
                init_states,
                ic_mask,
                Y,
                Y_mask,
                t_Y,
                med_trajectory_values,
                med_trajectory_mask,
                med_trajectory_time,
                med_tensors,
                hadm_ids,
                traj_ids,
                med_combo_ids
            ) = batch
        else:
            raise NotImplementedError(
                "Synthetic path not supported in simplified pipeline"
            )

        batch_size = Y.shape[0]
        ts = t_Y[0, :]

        if self.use_encoder != "none":
            if self.use_encoder == "raindrop":
                # Process 24h chartevents context
                ce_src = ce_rd_src.permute(1, 0, 2)
                ce_times = ce_rd_times.permute(1, 0)
                ce_lengths = ce_rd_length

                # Process 60min physio context
                src = rd_src.permute(1, 0, 2)
                times = rd_times.permute(1, 0)
                lengths = rd_length

                if self.training and self.use_checkpointing:
                    chartevents_embedding, _, _ = checkpoint.checkpoint(self.context_encoder_24h, ce_src, None,
                                                                        ce_times, ce_lengths, use_reentrant=False)
                    context_embedding, _, _ = checkpoint.checkpoint(self.context_encoder_60m, src, None, times, lengths,
                                                                    use_reentrant=False)
                    static_embedding = checkpoint.checkpoint(self.static_encoder, static_features, use_reentrant=False)
                else:
                    chartevents_embedding, _, _ = self.context_encoder_24h(src=ce_src, static=None, times=ce_times,
                                                                           lengths=ce_lengths)
                    context_embedding, _, _ = self.context_encoder_60m(src=src, static=None, times=times,
                                                                       lengths=lengths)
                    static_embedding = self.static_encoder(static_features)

                # Ensure concat matches the expected fusion input dim
                fused_embedding = torch.cat([chartevents_embedding, context_embedding, static_embedding], dim=-1)
                if fused_embedding.shape[-1] != self.fused_expected_dim:
                    # Fall back to computing expected size from actual tensors to avoid shape mismatch
                    self.fused_expected_dim = fused_embedding.shape[-1]
                    if self.debug:
                        print(f"[DEBUG] Adjusted fused_expected_dim to {self.fused_expected_dim}")

                if self.training and self.use_checkpointing:
                    neural_embedding = checkpoint.checkpoint(self.fusion_mlp, fused_embedding, use_reentrant=False)
                else:
                    neural_embedding = self.fusion_mlp(fused_embedding)

                logqp0 = 0
            else:
                raise NotImplementedError(
                    "Only raindrop encoder supported in MIMIC pipeline"
                )

            if self.debug:
                print(f"Static features shape: {static_features.shape}")
                print(f"Fused embedding shape for SDE: {neural_embedding.shape}")

            # === ARCHITECTURE CHANGE: START ===
            # The new architecture uses the fused embedding to directly generate a `neural_embedding`
            # that conditions the SDE, instead of predicting the initial ODE latents.

            # --- Legacy Path (Commented Out) ---
            # fused_rep = self.fusion_mlp(fused_embedding)
            # predicted_ode_latents_sigmoid = self.ode_latent_head(fused_rep).unsqueeze(1).repeat(1, self.num_samples, 1)
            # neural_embedding_from_head = self.neural_embedding_head(fused_rep).unsqueeze(1).repeat(1, self.num_samples, 1)
            # predicted_ode_latents = self.transform_sigmoid_to_physiological_ranges(predicted_ode_latents_sigmoid)
            # z1_for_sde = self._prepare_sde_initial_state(predicted_ode_latents, neural_embedding_from_head, init_states, ic_mask)
            # ic_consistency_loss = self.compute_ic_consistency_loss(
            #     predicted_ode_latents_sigmoid=predicted_ode_latents_sigmoid,
            #     init_states=init_states,
            #     ic_mask=ic_mask,
            # )
            # --- End Legacy Path ---

            # --- New Path ---
            # The initial state for the SDE includes expert variables and the neural embedding.
            initial_condition = self._prepare_no_encoder_initial_state(
                init_states, ic_mask
            )  # [B, expert_latent_dims]
            neural_embedding_expanded = neural_embedding.unsqueeze(1).repeat(1, self.num_samples, 1)  # [B, S, D_neural]
            expert_init_expanded = initial_condition.unsqueeze(1).repeat(1, self.num_samples, 1)  # [B, S, 14]
            z1_for_sde = torch.cat([expert_init_expanded, neural_embedding_expanded], dim=-1)  # [B, S, 14 + D_neural]

            # IC consistency loss is no longer applicable in this architecture.
            ic_consistency_loss = torch.tensor(0.0, device=self.device)
            # === ARCHITECTURE CHANGE: END ===

        else:  # No encoder path
            # When no encoder is used, include zeros neural embedding in the initial state for consistency.
            neural_embedding = torch.zeros(batch_size, self.encoder_SDENN_dims, device=self.device)
            initial_condition = self._prepare_no_encoder_initial_state(
                init_states, ic_mask
            )  # [B, expert_latent_dims]
            neural_embedding_expanded = neural_embedding.unsqueeze(1).repeat(1, self.num_samples, 1)  # [B, S, D_neural]
            expert_init_expanded = initial_condition.unsqueeze(1).repeat(1, self.num_samples, 1)  # [B, S, 14]
            z1_for_sde = torch.cat([expert_init_expanded, neural_embedding_expanded], dim=-1)  # [B, S, 14 + D_neural]
            logqp0 = 0
            ic_consistency_loss = torch.tensor(0.0, device=self.device)

        valid_lengths = (Y_mask.sum(dim=2) > 0).sum(dim=1)

        self.current_med_tensors = med_tensors if med_tensors is not None else None

        z1_for_sde = z1_for_sde[:, :, :2]  # [B, S, 2]


        latent_traj, logqp_path, i_ext_path = self.forward_latent(
            init_latents=z1_for_sde,
            ts=ts,
            Tx=torch.ones(batch_size, device=self.device),
            time_to_tx=torch.zeros(batch_size, device=self.device),
            neural_embedding=neural_embedding_expanded,
            valid_lengths=valid_lengths,
            med_traj_values=med_trajectory_values,
            med_traj_mask=med_trajectory_mask,
            med_traj_time=med_trajectory_time,
        )



        try:
            self._debug_predicted_traj = decoded_traj.detach().clone()
            self._debug_true_traj = Y.detach().clone()
            if self.debug:
                print(f"[DEBUG] Stored trajectories in common_step")
        except Exception as e:
            if self.debug:
                print(f"[WARN] Failed to store debug trajectories: {e}")

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
        loss, recon_loss, kl_div, per_sample_loss = self.compute_factual_loss(
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
                    if hasattr(self, 'use_control_tv_loss') and self.use_control_tv_loss:
                        try:
                            # tv_loss should be available from the computation above
                            self.log("train_tv_loss", tv_loss.detach(), on_step=True, on_epoch=True, prog_bar=False,
                                     logger=True)
                            self.log("train_tv_contribution", self.control_tv_weight * tv_loss.detach(), on_step=True,
                                     on_epoch=True, prog_bar=False, logger=True)
                        except:
                            pass
                    total_loss = total_loss + self.control_tv_weight * tv_loss
            except Exception:
                pass

        return {
            "loss": loss,
            "recon_loss": recon_loss,
            "kl_div": kl_div,
            "total_loss": total_loss,
            "ic_consistency_loss": ic_consistency_loss,
            "decoded_traj": decoded_traj,
            "latent_traj": latent_traj,
            "Y": Y,
            "combined_mask": combined_mask,
            "i_ext_path": i_ext_path,
            "z1_for_sde": z1_for_sde,
            "hadm_ids": hadm_ids,
            "traj_ids": traj_ids,
            "per_sample_loss": per_sample_loss,
            "med_combo_ids": med_combo_ids,
        }

    def training_step(self, batch, batch_idx):
        if self.debug and batch_idx == 0:
            print(f"[DEBUG] Hybrid_SDE validation_step: batch_idx={batch_idx}")
        result = self.common_step(batch, batch_idx)
        # Stash IDs for plotting filenames if present
        try:
            self._last_hadm_ids = result.get("hadm_ids", None)
            self._last_traj_ids = result.get("traj_ids", None)
        except Exception:
            pass

        total_loss = result["total_loss"]
        loss = result["loss"]
        ic_consistency_loss = result["ic_consistency_loss"]
        recon_loss = result["recon_loss"]
        kl_div = result["kl_div"]

        # NOTE: Per-medication group loss separation is performed only during validation.
        # The following training-time accumulation and periodic logging are disabled to reduce overhead.
        # per_sample_loss = result["per_sample_loss"]
        # med_combo_ids = result["med_combo_ids"]
        # for i in range(len(per_sample_loss)):
        #     combo_id = med_combo_ids[i].item()
        #     loss_val = per_sample_loss[i].item()
        #     self.train_combo_loss_accumulator[combo_id]['loss'] += loss_val
        #     self.train_combo_loss_accumulator[combo_id]['loss_sq'] += loss_val**2
        #     self.train_combo_loss_accumulator[combo_id]['count'] += 1
        # if self.global_step > 0 and self.log_train_combo_loss_every_n_steps > 0 and self.global_step % self.log_train_combo_loss_every_n_steps == 0:
        #     self._log_train_combo_stats()

        if self.log_lik_scale_mode == 'learnable':
            scales = torch.exp(self.log_scale_param).detach()
            for i, scale in enumerate(scales):
                self.log(f'train_learned_scale_dim_{i}', scale, on_step=True, on_epoch=False, prog_bar=False,
                         logger=True)
        elif self.log_lik_scale_mode == 'annealing':
            self.log('train_annealed_scale', self.scale_scheduler.val, on_step=True, on_epoch=False, prog_bar=False,
                     logger=True)

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
                control_energy = (control_mean ** 2).mean()
            else:
                control_energy = torch.tensor(0.0, device=total_loss.device)
        except Exception:
            control_energy = torch.tensor(0.0, device=total_loss.device)

        total_loss = total_loss + self.control_energy_weight * control_energy
        # Log and print current per-control weights (adaptive ramp * global weighting)
        current_ctrl_weights = self.get_current_per_control_weights().detach().cpu()
        # Log via Lightning so it reaches the configured logger (e.g., wandb)
        for i in range(current_ctrl_weights.numel()):
            try:
                self.log(f"train_ctrl_weight_{i}", float(current_ctrl_weights[i].item()), on_step=True, on_epoch=False,
                         prog_bar=False, logger=True)
            except Exception:
                pass
        # Also log dynamic per-control sigmas and weights succinctly
        current_sigmas = self.get_current_per_control_sigmas().detach().cpu()
        for i in range(current_sigmas.numel()):
            try:
                self.log(f"train_ctrl_sigma_{i}", float(current_sigmas[i].item()), on_step=True, on_epoch=False,
                         prog_bar=False, logger=True)
            except Exception:
                pass
        print(
            f"Step {self.global_step}: control_energy={control_energy.item():.2e}, weight={self.control_energy_weight:.2e}, contribution={self.control_energy_weight * control_energy.item():.2e}, main_loss={loss.item():.2f}, weights={current_ctrl_weights.tolist()}, sigmas={current_sigmas.tolist()}"
        )

        # Optional training plots (controls only), gated by plot_every
        should_plot_now = False
        try:
            step = int(getattr(self, "global_step", 0))
            interval = max(1, int(getattr(self, "plot_every", 1)))
            should_plot_now = (step % interval) == 0
        except Exception:
            should_plot_now = self.plot_outputs_train


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
            f"train_{self.loss_type}", recon_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )
        self.log(
            "train_KL", kl_div, on_step=True, on_epoch=True, prog_bar=True, logger=True
        )

        # Log optimizer learning rate (first param group) and SDEnet Frobenius norm of weights
        try:
            if hasattr(self, "trainer") and self.trainer is not None and self.trainer.optimizers:
                opt = self.trainer.optimizers[0]
                if opt and opt.param_groups:
                    current_lr = float(opt.param_groups[0].get("lr", self.learning_rate))
                    self.log("train_lr", current_lr, on_step=True, on_epoch=False, prog_bar=False, logger=True)
        except Exception:
            pass

        try:
            sum_sq: torch.Tensor = torch.tensor(0.0, device=total_loss.device)
            for name, param in self.SDEnet.named_parameters():
                if param is None:
                    continue
                if "weight" in name and param.data is not None:
                    sum_sq = sum_sq + (param.data.float() ** 2).sum()
            frob_norm = torch.sqrt(sum_sq).detach()
            self.log("train_sdenet_frobenius", frob_norm, on_step=True, on_epoch=False, prog_bar=False, logger=True)
        except Exception:
            pass

        return total_loss

    def validation_step(self, batch, batch_idx):
        if self.debug and batch_idx == 0:
            print(f"[DEBUG] Hybrid_SDE validation_step: batch_idx={batch_idx}")

        result = self.common_step(batch, batch_idx)
        try:
            self._last_hadm_ids = result.get("hadm_ids", None)
            self._last_traj_ids = result.get("traj_ids", None)
        except Exception:
            pass

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
            f"val_{self.loss_type}",
            result["recon_loss"],
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

        # Store per-combo losses for epoch-end analysis
        self.val_combo_losses.append({
            "losses": result["per_sample_loss"],
            "med_combo_ids": result["med_combo_ids"]
        })

        return {
            "val_loss": result["loss"],
            "val_recon_loss": result["recon_loss"],
            "val_kl": result["kl_div"],
            "decoded_traj": result["decoded_traj"],
            "true_traj": result["Y"],
            "mask": result["combined_mask"],
            "per_sample_loss": result["per_sample_loss"],
            "med_combo_ids": result["med_combo_ids"]
        }

    def test_step(self, batch, batch_idx, dataloader_idx=0):
        if self.debug and batch_idx == 0:
            print(f"[DEBUG] Hybrid_SDE validation_step: batch_idx={batch_idx}")

        result = self.common_step(batch, batch_idx)

        total_loss = result["total_loss"]
        loss = result["loss"]
        recon_loss = result["recon_loss"]
        kl_div = result["kl_div"]
        suffix = "_all" if dataloader_idx == 0 else "_filtered"

        # Log with dataset-specific names
        self.log(f"test_total_loss{suffix}", total_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log(f"test_main_loss{suffix}", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log(f"test_ic_consistency_loss{suffix}", result["ic_consistency_loss"], on_step=False, on_epoch=True,
                 prog_bar=True, logger=True)
        self.log(f"test_{self.loss_type}{suffix}", result["recon_loss"], on_step=False, on_epoch=True, prog_bar=True,
                 logger=True)
        self.log(f"test_KL{suffix}", kl_div, on_step=False, on_epoch=True, prog_bar=True, logger=True)

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
            self.log(f"test_mse{suffix}", valid_mse, on_step=False, on_epoch=True, prog_bar=True, logger=True)
            self.log(f"test_mae{suffix}", valid_mae, on_step=False, on_epoch=True, prog_bar=True, logger=True)

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

                self.log(f"test_zenker_mse{suffix}", zenker_mse, on_step=False, on_epoch=True, prog_bar=True,
                         logger=True)
                self.log(f"test_zenker_mae{suffix}", zenker_mae, on_step=False, on_epoch=True, prog_bar=True,
                         logger=True)
        if batch_idx < 3:
            self.plot_nature_style_with_uncertainty(
                decoded_traj, Y, combined_mask, batch_idx, suffix, result["med_combo_ids"]
            )

        # Store per-combo losses for epoch-end analysis
        self.test_combo_losses.append({
            "losses": result["per_sample_loss"],
            "med_combo_ids": result["med_combo_ids"]
        })

        return_dict = {
            "test_loss": loss,
            "test_recon_loss": recon_loss,
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
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.learning_rate, weight_decay=1e-4
        )

        if not self.use_lr_scheduler:
            return optimizer

        def lr_lambda(current_step: int):
            # Linear warmup
            if current_step < self.warmup_steps:
                return float(current_step) / float(max(1, self.warmup_steps))
            # Cosine decay
            progress = float(current_step - self.warmup_steps) / float(
                max(1, self.total_training_steps - self.warmup_steps)
            )
            cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
            # Final learning rate is min_lr
            scaled_decay = (1.0 - self.min_lr / self.learning_rate) * cosine_decay + (
                    self.min_lr / self.learning_rate
            )
            return scaled_decay

        scheduler = {
            "scheduler": torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda),
            "interval": "step",
            "frequency": 1,
            "name": "learning_rate",
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

    def on_test_epoch_end(self):
        """Log final test metrics to wandb with proper handling of multiple test datasets"""
        if self.log_wandb:
            # Extract metrics using the correct key format with dataloader_idx suffixes
            all_metrics = {
                'total_loss': float(self.trainer.callback_metrics.get('test_total_loss_all/dataloader_idx_0', 0)),
                'main_loss': float(self.trainer.callback_metrics.get('test_main_loss_all/dataloader_idx_0', 0)),
                'mse': float(self.trainer.callback_metrics.get('test_mse_all/dataloader_idx_0', 0)),
                'mae': float(self.trainer.callback_metrics.get('test_mae_all/dataloader_idx_0', 0)),
                'nll': float(self.trainer.callback_metrics.get('test_NLL_all/dataloader_idx_0', 0)),
                'kl': float(self.trainer.callback_metrics.get('test_KL_all/dataloader_idx_0', 0)),
                'ic_consistency': float(
                    self.trainer.callback_metrics.get('test_ic_consistency_loss_all/dataloader_idx_0', 0)),
                'zenker_mse': float(self.trainer.callback_metrics.get('test_zenker_mse_all/dataloader_idx_0', 0)),
                'zenker_mae': float(self.trainer.callback_metrics.get('test_zenker_mae_all/dataloader_idx_0', 0))
            }

            filtered_metrics = {
                'total_loss': float(self.trainer.callback_metrics.get('test_total_loss_filtered/dataloader_idx_1', 0)),
                'main_loss': float(self.trainer.callback_metrics.get('test_main_loss_filtered/dataloader_idx_1', 0)),
                'mse': float(self.trainer.callback_metrics.get('test_mse_filtered/dataloader_idx_1', 0)),
                'mae': float(self.trainer.callback_metrics.get('test_mae_filtered/dataloader_idx_1', 0)),
                'nll': float(self.trainer.callback_metrics.get('test_NLL_filtered/dataloader_idx_1', 0)),
                'kl': float(self.trainer.callback_metrics.get('test_KL_filtered/dataloader_idx_1', 0)),
                'ic_consistency': float(
                    self.trainer.callback_metrics.get('test_ic_consistency_loss_filtered/dataloader_idx_1', 0)),
                'zenker_mse': float(self.trainer.callback_metrics.get('test_zenker_mse_filtered/dataloader_idx_1', 0)),
                'zenker_mae': float(self.trainer.callback_metrics.get('test_zenker_mae_filtered/dataloader_idx_1', 0))
            }

            # Create comparison table
            comparison_data = [
                ['Metric', 'All Trajectories', 'Filtered Trajectories', 'Difference (All - Filtered)'],
                ['MSE', f"{all_metrics['mse']:.4f}", f"{filtered_metrics['mse']:.4f}",
                 f"{all_metrics['mse'] - filtered_metrics['mse']:.4f}"],
                ['MAE', f"{all_metrics['mae']:.4f}", f"{filtered_metrics['mae']:.4f}",
                 f"{all_metrics['mae'] - filtered_metrics['mae']:.4f}"],
                ['Total Loss', f"{all_metrics['total_loss']:.4f}", f"{filtered_metrics['total_loss']:.4f}",
                 f"{all_metrics['total_loss'] - filtered_metrics['total_loss']:.4f}"],
                ['Main Loss', f"{all_metrics['main_loss']:.4f}", f"{filtered_metrics['main_loss']:.4f}",
                 f"{all_metrics['main_loss'] - filtered_metrics['main_loss']:.4f}"],
                ['NLL', f"{all_metrics['nll']:.4f}", f"{filtered_metrics['nll']:.4f}",
                 f"{all_metrics['nll'] - filtered_metrics['nll']:.4f}"],
                ['KL Divergence', f"{all_metrics['kl']:.4f}", f"{filtered_metrics['kl']:.4f}",
                 f"{all_metrics['kl'] - filtered_metrics['kl']:.4f}"],
                ['IC Consistency', f"{all_metrics['ic_consistency']:.4f}", f"{filtered_metrics['ic_consistency']:.4f}",
                 f"{all_metrics['ic_consistency'] - filtered_metrics['ic_consistency']:.4f}"],
                ['Zenker MSE', f"{all_metrics['zenker_mse']:.4f}", f"{filtered_metrics['zenker_mse']:.4f}",
                 f"{all_metrics['zenker_mse'] - filtered_metrics['zenker_mse']:.4f}"],
                ['Zenker MAE', f"{all_metrics['zenker_mae']:.4f}", f"{filtered_metrics['zenker_mae']:.4f}",
                 f"{all_metrics['zenker_mae'] - filtered_metrics['zenker_mae']:.4f}"]
            ]

            wandb.log({"test_results_comparison": wandb.Table(data=comparison_data[1:], columns=comparison_data[0])})

            # Print summary to console as well
            print("\n" + "=" * 60)
            print("TEST RESULTS SUMMARY")
            print("=" * 60)
            for row in comparison_data:
                if row == comparison_data[0]:  # Header
                    print(f"{row[0]:<15} {row[1]:<18} {row[2]:<18} {row[3]}")
                    print("-" * 60)
                else:
                    print(f"{row[0]:<15} {row[1]:<18} {row[2]:<18} {row[3]}")
            print("=" * 60)

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
            self, predictions_full, targets, combined_mask, batch_idx, suffix, med_combo_ids
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

        # Optionally skip burn-in period for plotting
        dt = 10  # Time step in seconds between trajectory points
        burn_in_steps = int(self.sde_burn_in_period / dt)

        if (not self.plot_include_burn_in) and targets.shape[1] > burn_in_steps:
            predictions_full = predictions_full[:, :, burn_in_steps:, :]
            targets = targets[:, burn_in_steps:, :]
            combined_mask = combined_mask[:, burn_in_steps:, :]

        pred_mean = predictions_full.mean(1).detach()
        pred_std = predictions_full.std(1, unbiased=False).detach()
        targets = targets.detach() if targets.requires_grad else targets

        # Only allow rank 0 to write plots (avoid DDP duplicates)
        if getattr(self, 'trainer', None) is not None and not self.trainer.is_global_zero:
            return

        for patient_idx in range(min(self.max_plotted_patients, predictions_full.shape[0])):
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
            # Mark t0 after burn-in on main axis
            if self.plot_include_burn_in and burn_in_steps > 0:
                try:
                    ax.axvline(x=burn_in_steps * dt, color="#444", linestyle=":", linewidth=1.0, alpha=0.8)
                except Exception:
                    pass

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

            med_info_str = "Unknown Meds"
            try:
                combo_id = int(med_combo_ids[patient_idx].item())
                med_info_str = self.id_to_combo_map.get(combo_id, {}).get("str", f"ID {combo_id}")
            except (IndexError, AttributeError, ValueError):
                pass

            ax.set_title(
                f"{epoch_tag} – Patient {patient_idx} (Batch {batch_idx})\n{med_info_str}",
                fontweight="bold",
                fontsize=10
            )
            ax.legend(
                loc="upper right", fancybox=False, facecolor="white", framealpha=1.0
            )
            ax.grid(True, alpha=0.2)

            if self.log_wandb and getattr(self, 'trainer',
                                          None) is not None and self.trainer.is_global_zero and getattr(self, 'logger',
                                                                                                        None) is not None and hasattr(
                    self.logger, 'experiment'):
                try:
                    self.logger.experiment.log(
                        {f"uncertainty_plot_batch_{batch_idx}_patient_{patient_idx}{suffix}": wandb.Image(plt)})
                except Exception:
                    pass
            else:
                try:
                    last_hadm = int(getattr(self, "_last_hadm_ids", [None])[patient_idx])
                    last_traj = int(getattr(self, "_last_traj_ids", [None])[patient_idx])
                    id_suffix = f"_hadm{last_hadm}_traj{last_traj}" if last_hadm is not None and last_traj is not None else ""
                except Exception:
                    id_suffix = ""
                out_path = os.path.join(
                    self.train_dir,
                    f"nature_plots/{epoch_tag}_patient{patient_idx}_batch{batch_idx}{id_suffix}{suffix}_uncertainty.png",
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
                    name=f"Factual Batch {i + 1}",
                )
            )

        # Similarly for counterfactual MSEs
        for i, mse_list in enumerate(self.mse_data_cf):
            fig.add_trace(
                go.Scatter(
                    x=list(range(len(mse_list))),
                    y=sorted(mse_list),
                    mode="lines+markers",
                    name=f"Counterfactual Batch {i + 1}",
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
            for patient_idx in range(min(8, control_mean.shape[0])):
                # Build columns: time, mean_d0..mean_d{D-1}, std_d0..std_d{D-1}
                data = [time_seconds]
                for d in range(num_dims):
                    data.append(control_mean[patient_idx, :, d: d + 1])
                for d in range(num_dims):
                    data.append(control_std[patient_idx, :, d: d + 1])
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

    def on_validation_epoch_end(self):
        """Log per-medication combo loss statistics at the end of the validation epoch."""
        self._log_per_combo_stats("val", self.val_combo_losses)
        self.val_combo_losses.clear()  # Reset for next epoch

    def _log_per_combo_stats(self, prefix, combo_losses_list):
        """Helper function to compute and log statistics for per-medication combo losses."""
        if not combo_losses_list:
            return

        all_losses = torch.cat([d["losses"] for d in combo_losses_list])
        all_combo_ids = torch.cat([d["med_combo_ids"] for d in combo_losses_list])

        unique_combo_ids = torch.unique(all_combo_ids)

        stats_data = []
        for combo_id_tensor in unique_combo_ids:
            combo_id = combo_id_tensor.item()
            mask = all_combo_ids == combo_id
            losses = all_losses[mask]

            combo_name = self.id_to_combo_map.get(combo_id, {}).get("str", f"Unknown ID {combo_id}")

            mean_loss = losses.mean().item()
            std_loss = losses.std(unbiased=False).item() if len(losses) > 1 else 0.0
            stats_data.append([
                combo_name,
                mean_loss,
                std_loss,
                len(losses)
            ])

        # Sort by mean loss (descending)
        stats_data.sort(key=lambda x: x[1], reverse=True)

        # Log to console
        print(f"\n--- Per-Medication Combo Loss ({prefix}) ---")
        print(f"{'Medication Combo':<50} | {'Mean Loss':<12} | {'Std Dev':<10} | {'Count':<5}")
        print("-" * 80)
        for row in stats_data:
            print(f"{row[0]:<50} | {row[1]:<12.4f} | {row[2]:<10.4f} | {row[3]:<5}")
        print("-" * 80)

        # Log to wandb
        if self.log_wandb and getattr(self, 'trainer', None) is not None and self.trainer.is_global_zero and getattr(
                self, 'logger', None) is not None and hasattr(self.logger, 'experiment'):
            try:
                columns = ["Medication Combo", "Mean Loss", "Std Dev", "Count"]
                table = wandb.Table(columns=columns, data=stats_data)
                self.logger.experiment.log({f"{prefix}_per_combo_loss": table})
            except Exception:
                pass

    def on_train_epoch_start(self):
        """Reset the training loss accumulator at the start of each epoch."""
        self.train_combo_loss_accumulator.clear()

    def _log_train_combo_stats(self):
        """Computes and logs statistics from the training loss accumulator."""
        if not self.train_combo_loss_accumulator:
            return

        stats_data = []
        for combo_id, data in self.train_combo_loss_accumulator.items():
            if data['count'] > 0:
                mean_loss = data['loss'] / data['count']
                mean_loss_sq = data['loss_sq'] / data['count']
                std_dev = math.sqrt(max(0, mean_loss_sq - mean_loss ** 2))
                combo_name = self.id_to_combo_map.get(combo_id, {}).get("str", f"Unknown ID {combo_id}")
                stats_data.append([
                    combo_name,
                    mean_loss,
                    std_dev,
                    data['count']
                ])

        stats_data.sort(key=lambda x: x[1], reverse=True)

        # Log to console
        print(f"\n--- [Step {self.global_step}] Per-Medication Combo Training Loss ---")
        print(f"{'Medication Combo':<50} | {'Mean Loss':<12} | {'Std Dev':<10} | {'Count':<5}")
        print("-" * 80)
        for row in stats_data:
            print(f"{row[0]:<50} | {row[1]:<12.4f} | {row[2]:<10.4f} | {row[3]:<5}")
        print("-" * 80)

        # Log to wandb
        if self.log_wandb:
            columns = ["Medication Combo", "Mean Loss", "Std Dev", "Count"]
            table = wandb.Table(columns=columns, data=stats_data)
            wandb.log({f"train_per_combo_loss": table})

    def _log_epoch_end_combo_stats(self, prefix, combo_losses_list):
        """Helper function to compute and log statistics for per-medication combo losses at epoch end."""
        if not combo_losses_list:
            return

        all_losses = torch.cat([d["losses"] for d in combo_losses_list])
        all_combo_ids = torch.cat([d["med_combo_ids"] for d in combo_losses_list])

        unique_combo_ids = torch.unique(all_combo_ids)

        stats_data = []
        for combo_id_tensor in unique_combo_ids:
            combo_id = combo_id_tensor.item()
            mask = all_combo_ids == combo_id
            losses = all_losses[mask]

            combo_name = self.id_to_combo_map.get(combo_id, {}).get("str", f"Unknown ID {combo_id}")

            mean_loss = losses.mean().item()
            std_loss = losses.std(unbiased=False).item() if len(losses) > 1 else 0.0
            stats_data.append([
                combo_name,
                mean_loss,
                std_loss,
                len(losses)
            ])

        # Sort by mean loss (descending)
        stats_data.sort(key=lambda x: x[1], reverse=True)

        # Log to console
        print(f"\n--- Per-Medication Combo Loss ({prefix}) ---")
        print(f"{'Medication Combo':<50} | {'Mean Loss':<12} | {'Std Dev':<10} | {'Count':<5}")
        print("-" * 80)
        for row in stats_data:
            print(f"{row[0]:<50} | {row[1]:<12.4f} | {row[2]:<10.4f} | {row[3]:<5}")
        print("-" * 80)

        # Log to wandb
        if self.log_wandb:
            columns = ["Medication Combo", "Mean Loss", "Std Dev", "Count"]
            table = wandb.Table(columns=columns, data=stats_data)
            wandb.log({f"{prefix}_per_combo_loss": table})
