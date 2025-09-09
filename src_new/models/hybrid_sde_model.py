import os
import sys
import math
from collections import defaultdict
import typing as t

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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
import logging

# Add the project root and utils directory to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
utils_path = os.path.join(project_root, "src_new", "utils")
sys.path.insert(0, project_root)
sys.path.insert(0, utils_path)

from src_new.utils.train_utils import (
    zenker_derivatives, 
    compute_stable_equilibrium_batch, 
    denormalize_selected_from_batched_sample, 
    _normalize_label, 
    _build_full_state_from_equilibrium, 
    _dict_to_PhysioRanges,
    _midpoint_pair,
    _EXPERT_ORDER,
    _pick_ic_or_fallback
)

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
    """Handle NaN/Inf detection based on configuration."""
    global AUTO_DEBUG_ACTIVATED, DEBUG
    
    if not AUTO_DEBUG_ACTIVATED:
        AUTO_DEBUG_ACTIVATED = True
        DEBUG = True
        model_instance.debug = True
        
        # Get debug behavior from model instance or default to "kill"
        debug_behavior = getattr(model_instance, 'debug_on_nan_inf', 'kill')
        
        print("\n" + "="*80)
        print("🚨 NaN/Inf DETECTED 🚨")
        print("="*80)
        print(f"Location: {location}")
        print(f"Tensor: {tensor_name}")
        print(f"Global step: {getattr(model_instance, 'global_step', 'unknown')}")
        print(f"Current epoch: {getattr(model_instance, 'current_epoch', 'unknown')}")
        print(f"Debug behavior: {debug_behavior}")
        print("="*80)
        
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
        
        print("="*80)
        
        # Handle different debug behaviors
        if debug_behavior == "kill":
            print("💀 EXITING FOR SWEEP CONTINUATION 💀")
            print("="*80 + "\n")
            
            # Clean exit strategy to avoid NFS issues
            try:
                # Try to cleanup any open files/resources gracefully
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                
                # Force garbage collection
                import gc
                gc.collect()
                
                # Close any open matplotlib figures
                try:
                    import matplotlib.pyplot as plt
                    plt.close('all')
                except:
                    pass
                
                # Try to flush any pending I/O
                try:
                    import sys
                    sys.stdout.flush()
                    sys.stderr.flush()
                except:
                    pass
                
                # Use sys.exit instead of os._exit for cleaner shutdown
                import sys
                sys.exit(1)
            except Exception:
                # If graceful exit fails, force exit
                import os
                os._exit(1)
        elif debug_behavior == "debug":
            # Avoid blocking if stdout is redirected or no TTY is available
            import sys
            has_tty = hasattr(sys, "stdin") and sys.stdin is not None and sys.stdin.isatty()
            if getattr(model_instance, "redirect_output", False) or not has_tty:
                print("[AUTO-DEBUG] No TTY or output redirected; raising instead of entering pdb.")
                raise RuntimeError(f"NaN/Inf detected in {tensor_name} at {location} (non-interactive environment)")
            print("🔧 ENTERING INTERACTIVE DEBUG MODE 🔧")
            print("="*80 + "\n")
            # Set up debug mode - the model will continue with debug prints
            model_instance.debug = True
            DEBUG = True
            
            # Enter interactive debug mode
            import pdb
            pdb.set_trace()
        elif debug_behavior == "raise":
            print("⚠️  RAISING EXCEPTION ⚠️")
            print("="*80 + "\n")
            # Raise a custom exception
            raise RuntimeError(f"NaN/Inf detected in {tensor_name} at {location}")
        else:
            print(f"❌ UNKNOWN DEBUG BEHAVIOR: {debug_behavior} ❌")
            print("="*80 + "\n")
            # Default to kill mode
            import sys
            sys.exit(1)


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


class Hybrid_SDE(LightningModule):
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
        debug_on_nan_inf="kill",
        id_to_combo_map=None,
        redirect_output=False,
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
        self.SDEnet_out_dims = 2 if self.direct_pressure_controls else SDEnet_out_dims
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
        self.debug_on_nan_inf = debug_on_nan_inf
        self.redirect_output = bool(redirect_output)

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
                max_len=24, # 24 hours, 1h interval
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
                output_dim=encoder_SDENN_dims, # Directly outputs the neural embedding
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
            raise ValueError(f"Length of prior_tx_sigma_per_control ({self.prior_tx_sigma_per_control.shape[0]}) must match SDEnet_out_dims ({SDEnet_out_dims})")
            
        self.prior_tx_mu = prior_tx_mu

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
        # Base control scales for adaptive ramp (configurable)
        try:
            base_scales_cfg = kwargs.get("base_control_scales", None)
            if base_scales_cfg is None:
                # Default per original: [1, 1, 0.1, 0.1]
                base_scales_list = [1.0, 1.0, 0.1, 0.1]
            else:
                base_scales_list = [float(x) for x in list(base_scales_cfg)]
        except Exception:
            base_scales_list = [1.0, 1.0, 0.1, 0.1]
        base_scales_tensor = torch.tensor(base_scales_list, dtype=torch.float32)
        # Align base scales to current number of control heads
        if base_scales_tensor.numel() != self.SDEnet_out_dims:
            repeats = (self.SDEnet_out_dims + base_scales_tensor.numel() - 1) // base_scales_tensor.numel()
            base_scales_tensor = base_scales_tensor.repeat(repeats)[: self.SDEnet_out_dims]
        self.register_buffer("base_control_scales", base_scales_tensor)
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

        # Base network input dims start from encoder output dims
        net_input_dims = self.encoder_output_dim
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
            self.ic_consistency_weight = 10
            # add fixed med embedding dims
            net_input_dims = net_input_dims + self.med_embed_dim
        else:
            self.ic_consistency_weight = 0
            # expert latents + fixed med embedding dims + neural embedding
            net_input_dims = self.expert_latent_dims + self.med_embed_dim + encoder_SDENN_dims
            net_input_dims = net_input_dims + 2 if include_time else net_input_dims

        # Append physics-derived feature count (ΔP, r_tpr, f_hr, F, dpa_base, dpv_base, sigma, s_dot)
        self.physics_feat_dims = 8
        net_input_dims = net_input_dims + self.physics_feat_dims


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

        self.indx_interest = {'Heart rate': 0, 'Arterial BP Mean': 6, 'CVP': 8}
        self.df_ranges = df_ranges = pd.DataFrame({
            'item_label': ['Heart rate', 'Arterial BP Mean', 'CVP'],
            'lower_bound': [33.9, 26.7, -5.2],
            'upper_bound': [142.8, 134.5, 28.7],
        }) 

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
        if self.n_medications > 0:
            # med embedding appended per node
            node_feat_dim += self.med_embed_dim
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
            anneal_start_val = 3.0
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
        base_scales = self.base_control_scales

        # Start with 1% control authority, ramp up to 100% over 1000 steps
        ramp_progress = min(current_step / 150.0, 1.0)
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
        
        # Check for NaN/Inf in input
        check_for_nan_inf(y, "SDE_input_y", "apply_SDE_fun", self, raise_error=True)
        check_for_nan_inf(t, "SDE_input_t", "apply_SDE_fun", self, raise_error=True)

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
            activate_auto_debug_mode(self, "apply_SDE_fun", "physics_feats", physics_feats)
            raise RuntimeError("Non-finite values in physics_feats. Auto-debug mode activated.")
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

        # Index precomputed med context by time step
        if not hasattr(self, "_t0"):
            self._t0 = torch.tensor(0.0, device=self.device)
        if not hasattr(self, "_dt_grid"):
            self._dt_grid = torch.tensor(
                float(self.integration_step_size), device=self.device
            )
        t_idx = torch.floor_divide((t.to(self._t0) - self._t0), 10).to(
            torch.long
        )
        t_idx = torch.clamp(t_idx, 0, self.current_med_tensors.shape[1] - 1)

    
        med_tensor = self.current_med_tensors[:, t_idx, :]
        # expand over samples if needed
        samples_per_batch = batch_size // med_tensor.shape[0]
        med_tensor = med_tensor.repeat_interleave(samples_per_batch, dim=0)
        #med_safe = torch.sign(med_tensor) * torch.log1p(torch.abs(med_tensor))
        #med_safe = torch.nan_to_num(med_safe, nan=0.0, posinf=3.0, neginf=-3.0)  # Clean NaN/Inf first
        #med_safe = torch.clamp(med_safe, min=-3.0, max=3.0)
        # Project to fixed-size med embedding (LazyLinear infers input dim on first call)
        def med_forward(med_input):
            return torch.tanh(self.med_proj(med_input))

        if self.training and self.use_checkpointing and getattr(med_tensor, 'requires_grad', False):
            med_embed = checkpoint.checkpoint(med_forward, med_tensor, use_reentrant=False)
        else:
            med_embed = torch.tanh(self.med_proj(med_tensor))
       
        if torch.isnan(med_tensor).any() or torch.isinf(med_tensor).any():
            activate_auto_debug_mode(self, "apply_SDE_fun", "med_context", med_tensor)
            raise RuntimeError("Non-finite values in med_context. Auto-debug mode activated.")
        
        if self.debug:
            
            pass
                #print(
                #)
                ##    f"  [DBG] med_tensor:{med_tensor.shape} {med_tensor[0,:]}"
            

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
            if getattr(self, "med_proj", None) is not None:
                med_b = med_embed.unsqueeze(1).repeat(1, expert_raw.shape[1], 1)
                parts.append(med_b)
            node_features = torch.cat(parts, dim=-1)  # [B,14,F]
            if torch.isnan(node_features).any() or torch.isinf(node_features).any():
                activate_auto_debug_mode(self, "apply_SDE_fun (GAT path)", "node_features", node_features)
                raise RuntimeError("Non-finite values in GAT node_features. Auto-debug mode activated.")
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
                    activate_auto_debug_mode(self, "apply_SDE_fun (GAT path)", f"GAT parameter '{name}'", param)
                    raise RuntimeError(f"Detected non-finite values in GAT parameter '{name}'. Auto-debug mode activated.")

            u_raw = self.gat_controller(node_features)  # [B,4]
            if self.force_no_controls:
                scaled_output = torch.zeros_like(u_raw)
            else:
                # Apply adaptive per-head scales and global weighting
                current_scales = self.get_current_per_control_weights().to(u_raw.device)
                if current_scales.numel() != u_raw.shape[-1]:
                    repeats = (u_raw.shape[-1] + current_scales.numel() - 1) // current_scales.numel()
                    current_scales = current_scales.repeat(repeats)[: u_raw.shape[-1]]
                scaled_output = u_raw * current_scales.unsqueeze(0)
            # Check for NaN/Inf in scaled output - FAIL FAST, NO SANITIZATION
            check_for_nan_inf(scaled_output, "GAT_scaled_output", "apply_SDE_fun", self, raise_error=True)
            
            # Only clamp extreme values, but don't sanitize NaN/Inf
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
            activate_auto_debug_mode(self, "apply_SDE_fun (MLP path)", "SDE_NN_input", SDE_NN_input)
            raise RuntimeError(
                f"SDE_NN_input contains non-finite values (NaN={nan_cnt}, Inf={inf_cnt}). Sample bad rows: {bad_rows.tolist()}. Auto-debug mode activated."
            )

        for name, param in self.SDEnet.named_parameters():
            if param is None or param.data is None:
                continue
            if torch.isnan(param.data).any() or torch.isinf(param.data).any():
                activate_auto_debug_mode(self, "apply_SDE_fun (MLP path)", f"SDEnet parameter '{name}'", param.data)
                raise RuntimeError(f"Detected non-finite values in SDEnet parameter '{name}'. Auto-debug mode activated.")

        # Append compact med embedding rather than raw med tensor to keep dims stable
        SDE_NN_input = torch.cat([SDE_NN_input, med_embed], dim=-1)

        if self.debug:
            try:
                print(
                    f"  [DBG] SDE_NN_input: shape={tuple(SDE_NN_input.shape)} mean={SDE_NN_input.mean().item():.4e} std={SDE_NN_input.std().item():.4e} min={SDE_NN_input.min().item():.4e} max={SDE_NN_input.max().item():.4e}"
                )
                #print(
                #    f"  [DBG] SDE_NN_input[0, :]:{SDE_NN_input[0, :]}"
                #)
            except Exception:
                pass

        # if self.debug:
        #    print(f"SDE_NN_input shape: {SDE_NN_input.shape}")

        # Use gradient checkpointing for control network to prevent accumulation
        def control_forward(sde_input):
            #normalized_input = self.input_layer_norm(sde_input)  # Include in checkpoint
            return self.SDEnet(sde_input)

        if self.training and self.use_checkpointing and getattr(SDE_NN_input, 'requires_grad', False):
            SDE_NN_output_latents = checkpoint.checkpoint(control_forward, SDE_NN_input, use_reentrant=False)
        else:
            #normalized_input = self.input_layer_norm(SDE_NN_input)
            SDE_NN_output_latents = self.SDEnet(SDE_NN_input)

        # TODO do these clamps make sense
        # control_scales = torch.tensor([100.0, 30.0], device=SDE_NN_output_latents.device)
        # scaled_output = SDE_NN_output_latents * control_scales.unsqueeze(0)
        if self.force_no_controls:
            scaled_output = torch.zeros_like(SDE_NN_output_latents)
        else:
            # Apply per-head scales and global weighting
            current_scales = self.get_current_per_control_weights().to(SDE_NN_output_latents.device)
            scaled_output = SDE_NN_output_latents * current_scales.unsqueeze(0)
        if torch.isnan(scaled_output).any() or torch.isinf(scaled_output).any():
            activate_auto_debug_mode(self, "apply_SDE_fun (MLP path)", "scaled_output", scaled_output)
            raise RuntimeError("Non-finite values in scaled_output. Auto-debug mode activated.")
        scaled_output = torch.clamp(scaled_output, -5.0, 5.0)

        if torch.isnan(SDE_NN_output_latents).any() or torch.isinf(SDE_NN_output_latents).any():
            try:
                torch.autograd.set_detect_anomaly(True)
            except Exception:
                pass
            activate_auto_debug_mode(self, "apply_SDE_fun (MLP path)", "SDE_NN_output_latents", SDE_NN_output_latents)
            raise RuntimeError("SDE_NN_output contains non-finite values. Auto-debug mode activated.")
        # print(SDE_NN_output_latents)
        # print(self.SDE_input_state)
        # breakpoint()
        # print('SDE_NN_output_latents example', SDE_NN_output_latents[0, :])
        
        if self.debug:
            t_idx = torch.floor_divide((t.to(self._t0) - self._t0), 10).to(
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
                    t_idx = torch.floor_divide((t.to(self._t0) - self._t0), 10).to(
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

        # Final check for NaN/Inf in output - FAIL FAST, NO SANITIZATION
        check_for_nan_inf(scaled_output, "apply_SDE_fun_output", "apply_SDE_fun", self, raise_error=True)
        # In training_step(), replace the broken lines with:
        
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
        # Check for NaN/Inf in inputs - FAIL FAST, NO SANITIZATION
        check_for_nan_inf(y, "f_input_y", "f_drift", self, raise_error=True)
        check_for_nan_inf(t, "f_input_t", "f_drift", self, raise_error=True)
        
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

        # y now contains: [i_ext (SDEnet_out_dims), expert_latents (14), neural_embedding (encoder_SDENN_dims)]

        i_ext_SDE_dict = {}
        for i in range(self.SDEnet_out_dims):
            i_ext_SDE_dict[f"i_ext_SDE_{i+1}"] = y_clamped[:, i].unsqueeze(1)

        # Use tensor-safe comparison for time gating (no .item())
        t_tensor = t if torch.is_tensor(t) else torch.tensor(t, device=self.device)
        
        # Burn-in logic
        if t_tensor < self.sde_burn_in_period:
            # During burn-in, no neural control is applied.
            dt_i_ext_SDE = torch.zeros(
                batch_size, self.SDEnet_out_dims, device=self.device
            )
        elif (t_tensor >= time_to_treatment).all():
            # this will always be the case when working with mimic data as time to treatment is 0
            dt_i_ext_SDE = self.apply_SDE_fun(t, y_clamped)
            
        else:
            raise ValueError("Time to treatment should always be <= current time t")

        dt_i_ext_SDE_dict = {}
        for i in range(self.SDEnet_out_dims):
            dt_i_ext_SDE_dict[f"dt_i_ext_SDE_{i+1}"] = dt_i_ext_SDE[:, i].unsqueeze(
                1
            )

        # Neural embedding derivatives (zeros - they evolve stochastically)
        dt_neural_embedding = torch.zeros([batch_size, self.encoder_SDENN_dims]).to(
            self.device
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
        if self.direct_pressure_controls:
            # Use two direct control heads: C1->dpa_dt, C2->dpv_dt
            dpa_dt = dpa_dt + dt_i_ext_SDE_dict["dt_i_ext_SDE_1"]
            dpv_dt = dpv_dt + dt_i_ext_SDE_dict["dt_i_ext_SDE_2"]
        else:
            dpv_dt = dpv_dt + dt_i_ext_SDE_dict["dt_i_ext_SDE_1"]
            dsv_dt = dsv_dt + dt_i_ext_SDE_dict["dt_i_ext_SDE_2"]
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
                print(f"  control derivs (first sample): {dt_i_ext_SDE[0]}")
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
        if torch.isnan(dt_i_ext_SDE).any() or torch.isinf(dt_i_ext_SDE).any():
            activate_auto_debug_mode(self, "f()", "dt_i_ext", dt_i_ext_SDE)
            raise RuntimeError("Non-finite values in dt_i_ext_SDE. Auto-debug mode activated.")
        dt_i_ext_SDE = dt_i_ext_SDE.clamp_(-10.0, 10.0)
        if torch.isnan(dt_expert).any() or torch.isinf(dt_expert).any():
            activate_auto_debug_mode(self, "f()", "dt_expert", dt_expert)
            raise RuntimeError("Non-finite values in dt_expert. Auto-debug mode activated.")
        if torch.isnan(dt_neural_embedding).any() or torch.isinf(dt_neural_embedding).any():
            activate_auto_debug_mode(self, "f()", "dt_neural_embedding", dt_neural_embedding)
            raise RuntimeError("Non-finite values in dt_neural_embedding. Auto-debug mode activated.")
        final_f_out = torch.cat([dt_i_ext_SDE, dt_expert, dt_neural_embedding], dim=-1)

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

                # Use dynamic per-control sigma (tied to current weights) for KL computation
                sigma_val = self.get_current_per_control_sigmas()[ctrl_idx].to(y.device)
                g_ctrl = sigma_val.repeat(y_ctrl.size(0), 1)

                h_ctrl = self.h(t, y_ctrl)
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

        # Check for NaN/Inf in output - FAIL FAST, NO SANITIZATION
        check_for_nan_inf(f_out, "f_output", "f_drift", self, raise_error=True)
        
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
        # Check for NaN/Inf in inputs - FAIL FAST, NO SANITIZATION
        check_for_nan_inf(y, "g_input_y", "g_diffusion", self, raise_error=True)
        check_for_nan_inf(t, "g_input_t", "g_diffusion", self, raise_error=True)
        
        if self.debug:
            t_idx = torch.round((t.to(self._t0) - self._t0) / self._dt_grid).to(
                torch.long
            )
            if (t_idx.remainder(10) == 0).item():
                print(
                    f"[DEBUG] Hybrid_SDE g (diffusion): t_idx={int(t_idx)}, y_shape={y.shape}"
                )

        batch_size = y.shape[0]

        # Use dynamic per-control sigmas tied to current control weights
        sigmas = self.get_current_per_control_sigmas()  # [D]
        # Return as tensor with shape [batch_size, 1]; this g() is used for OU prior on a single dim
        # Take the first control's sigma for this scalar dimension
        sigma_val = float(sigmas[0].item())
        g_out = torch.full((batch_size, 1), sigma_val, device=y.device)
        
        # Check for NaN/Inf in output - FAIL FAST, NO SANITIZATION
        check_for_nan_inf(g_out, "g_output", "g_diffusion", self, raise_error=True)
        
        return g_out

    # 6. Fixed g_aug method:
    def g_aug(self, t, y):
        """Diffusion for augmented dynamics (no valid_time masking)."""
        # concise only

        batch_size = y.shape[0]

        # Diffusion only on control states (first SDEnet_out_dims) using dynamic per-control sigmas
        g_controls = self.get_current_per_control_sigmas().to(y.device).repeat(batch_size, 1)

        # Zeros for all other state dimensions
        num_other_dims = y.shape[1] - self.SDEnet_out_dims
        g_zeros = torch.zeros(batch_size, num_other_dims, device=y.device)

        g_out = torch.cat([g_controls, g_zeros], dim=1)
        
        check_for_nan_inf(g_out, "g_aug_output", "g_diffusion", self, raise_error=True)
        
        return g_out


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
        neural_embedding,
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
        self.current_neural_embedding = neural_embedding # [B, S, D_neural]

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

        # Final sanitization to prevent numerical issues in SDE solver
        try:
            aug_y0 = torch.nan_to_num(aug_y0, nan=0.0, posinf=1e6, neginf=-1e6)
            aug_y0 = torch.clamp(aug_y0, -1e6, 1e6)
        except Exception:
            pass

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
            print(f"Latent out: {latent_out.shape}. Expect [B x S x T x {self.expert_latent_dims}]")

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
            else: # 'fixed'
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
                per_sample_logpy = logpy.sum(dim=(2, 3)).mean(dim=1) # [B]
                valid_count_per_sample = torch.clamp(mask.sum(dim=(1,2)), min=1.0) # [B]
                per_sample_loss = -per_sample_logpy / valid_count_per_sample
                
                # Overall loss for backward pass
                valid_count_total = torch.clamp(mask.sum(), min=1.0)
                recon_loss = -logpy.sum() / valid_count_total
            else:
                per_sample_loss = -logpy.sum(dim=(2, 3)).mean(dim=(1,0)) # [B]
                recon_loss = per_sample_loss.mean()

        elif self.loss_type == "mse":
            # MSE Loss (deterministic on mean prediction)
            pred_mean = predicted_traj.mean(dim=1)  # [B, T, C]
            sq_error = (pred_mean - true_traj) ** 2
            
            # Per-sample loss calculation
            if mask is not None:
                mask = mask.to(device=sq_error.device, dtype=sq_error.dtype)
                sq_error_masked = sq_error * mask
                
                per_sample_sq_error = sq_error_masked.sum(dim=(1,2)) # [B]
                valid_count_per_sample = torch.clamp(mask.sum(dim=(1,2)), min=1.0) # [B]
                per_sample_loss = per_sample_sq_error / valid_count_per_sample
                
                # Overall loss for backward pass
                valid_count_total = torch.clamp(mask.sum(), min=1.0)
                recon_loss = sq_error_masked.sum() / valid_count_total
            else:
                per_sample_loss = sq_error.mean(dim=(1,2)) # [B]
                recon_loss = per_sample_loss.mean()
        else:
            raise ValueError(f"Unknown loss_type: {self.loss_type}")

        # Compute final reconstruction loss, optionally scaling by variance
        if self.scale_loss_by_variance:
            with torch.no_grad():
                if mask is not None:
                    mask_f = mask.to(dtype=true_traj.dtype)
                    valid_counts_per_channel = torch.clamp(mask_f.sum(dim=1), min=1.0) # [B, C]
                    masked_traj = true_traj * mask_f
                    mean_per_channel = masked_traj.sum(dim=1) / valid_counts_per_channel # [B, C]
                    variance_per_channel = (masked_traj**2).sum(dim=1) / valid_counts_per_channel - mean_per_channel**2 # [B, C]
                else: # No mask
                    variance_per_channel = torch.var(true_traj, dim=1, unbiased=False) # [B, C]

                total_variance_per_sample = variance_per_channel.sum(dim=1) # [B]
                variance_scale_factor = torch.log1p(total_variance_per_sample)
                
                # Normalize across batch to prevent changing the overall loss magnitude
                if variance_scale_factor.mean() > 1e-6:
                    variance_scale_factor = variance_scale_factor / variance_scale_factor.mean()
                else:
                    variance_scale_factor = torch.ones_like(variance_scale_factor)
            
            # Log the mean scale factor during training
            if self.training:
                self.log("train/variance_scale_factor_mean", variance_scale_factor.mean().item(), on_step=True, on_epoch=False, prog_bar=False, logger=True)
            
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
                        raise RuntimeError(f"Non-finite gradient detected in parameter '{name}'. Auto-debug mode activated.")
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
                    raise RuntimeError(f"Non-finite parameter detected before optimizer step: '{name}'. Auto-debug mode activated.")

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
        """Deprecated: encoder directly consumes Raindrop sources; kept for API compat."""
        return X

    def _prepare_no_encoder_initial_state(self, init_states, ic_mask):
        """Deprecated: replaced by analytical ICs from chartevents + equilibrium."""
        return init_states


    def _prepare_sde_initial_state(
        self, predicted_ode_latents, neural_embedding, init_states, ic_mask
    ):
        """Deprecated: replaced by analytical ICs + fusion with neural embedding."""
        return torch.cat([init_states.unsqueeze(1).repeat(1, self.num_samples, 1), neural_embedding], dim=-1)

    
    def _extract_physio_from_chartevents(self, cd_rd_src, df_ranges, indx_interest, group_col='item_label'):
        """
        Uses your existing `denormalize_selected_from_batched_sample` to get original-scale values.
        Returns dict {normalized_label: [B] tensor}.
        """
        original_values, _, _, _, _, _, _, _ = denormalize_selected_from_batched_sample(
            cd_rd_src, df_ranges, indx_interest, group_col=group_col, use_mask=True, clip=True
        )
        labels = list(indx_interest.keys())
        norm_labels = [_normalize_label(l) for l in labels]
        return {nl: original_values[:, i] for i, nl in enumerate(norm_labels)}

    def _compute_full_analytical_zenker_initial_conditions(
        self,
        cd_rd_src,                 # [B, T, 2*d_inp] (values then mask)
        init_states: torch.Tensor, # [B, 5], IC[:,0]=ABP mean (Pa), IC[:,1]=CVP (Pv), IC[:,2]=Hr (if present)
        ic_mask: torch.Tensor,     # [B, 5], 1 if IC entry present
    ) -> torch.Tensor:
        """
        Return [B, L] with L=len(_EXPERT_ORDER)=14 in the fixed order _EXPERT_ORDER.
        Policy:
        - Pa := IC[:,0] if present else Arterial BP from chartevents
        - Pv := IC[:,1] if present else CVP from chartevents
        - Hr := IC[:,2] if present else Heart Rate from chartevents
        - Remaining parameters set to midpoints of self.physio_ranges
        - p_aset, s_reflex, r_tpr_mod, SV come from the equilibrium solution
        """
        if getattr(self, "debug", False):
            try:
                print("[DEBUG] _compute_full_analytical_zenker_initial_conditions: start")
                cd_shape = tuple(cd_rd_src.shape) if hasattr(cd_rd_src, "shape") else "NA"
                ic_shape = tuple(init_states.shape) if hasattr(init_states, "shape") else "NA"
                mask_shape = tuple(ic_mask.shape) if hasattr(ic_mask, "shape") else "NA"
                print(f"  cd_rd_src shape={cd_shape}  init_states shape={ic_shape}  ic_mask shape={mask_shape}")
                print(f"  physio_ranges keys: {list(self.physio_ranges.keys())[:8]} ... ({len(self.physio_ranges)})")
                try:
                    print(f"  indx_interest size={len(self.indx_interest)} sample keys={list(self.indx_interest.keys())[:8]}")
                except Exception:
                    print("  indx_interest: <unavailable or not a mapping>")
            except Exception as e:
                print(f"[WARN] Debug print failed at start: {e}")
        # 1) denormalized fallbacks from chartevents
        ce_vals = self._extract_physio_from_chartevents(
            cd_rd_src, self.df_ranges, self.indx_interest, group_col='item_label'
        )
        #print('ce_vals:', ce_vals)

        if getattr(self, "debug", False):
            try:
                ce_keys = list(ce_vals.keys())
                print(f"  ce_vals keys (normalized) sample: {ce_keys[:8]} total={len(ce_keys)}")
            except Exception as e:
                print(f"[WARN] Debug print failed on ce_vals: {e}")
        
        
        
        # Prefer the exact key used in indx_interest ('Arterial BP Mean'),
        # but fall back to 'Arterial BP' if present
        pa_key = _normalize_label('Arterial BP Mean')
        Pa_ce  = ce_vals[pa_key]  # [B]
        Pv_ce  = ce_vals[_normalize_label('CVP')]          # [B]
        # handle either "Heart Rate" or "Heart_rate"
        hr_key = _normalize_label('Heart Rate')
        Hr_ce  = ce_vals[hr_key]
        if getattr(self, "debug", False):
            try:
                print(f"  chosen keys -> Pa:{pa_key} Pv:{_normalize_label('CVP')} Hr:{hr_key}")
            except Exception:
                pass

        B = Pa_ce.shape[0]
        device = Pa_ce.device
        dtype = Pa_ce.dtype
     
        # 2) prefer IC for Pa, Pv; always use chartevents for Hr
        Pa = _pick_ic_or_fallback(init_states[:, 0].to(device=device, dtype=dtype),
                                ic_mask[:, 0].to(device=device),
                                Pa_ce)
        Pv = _pick_ic_or_fallback(init_states[:, 1].to(device=device, dtype=dtype),
                                ic_mask[:, 1].to(device=device),
                                Pv_ce)
        Hr = Hr_ce

        #print('pre-sanitize:')
        #print('Pa:', Pa)
        #print('Pv:', Pv)
        #print('Hr:', Hr)

        # 3) equilibrium (vectorized)
        eq = compute_stable_equilibrium_batch(
            Pa, Pv, Hr,
            ranges=_dict_to_PhysioRanges(self.physio_ranges), hr_units="bpm"
        )
        print('consistency_error:', eq['consistency_error'])

        # 4) assemble full [B, L] in fixed order
        expert_state = _build_full_state_from_equilibrium(
            Pa, Pv, Hr, eq, self.physio_ranges, device, dtype
        )

        if getattr(self, "debug", False):
            print("[DEBUG] _compute_full_analytical_zenker_initial_conditions")
            print(f"  expert_state: {expert_state.shape}  (B={B}, L={expert_state.shape[1]})")
            print(f"  first row (subset): {expert_state[0, :]}")
            try:
                print(f"  _EXPERT_ORDER length: {len(_EXPERT_ORDER)}")
            except Exception:
                pass

        return expert_state
            

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
                    chartevents_embedding, _, _ = checkpoint.checkpoint(self.context_encoder_24h, ce_src, None, ce_times, ce_lengths, use_reentrant=False)
                    context_embedding, _, _ = checkpoint.checkpoint(self.context_encoder_60m, src, None, times, lengths, use_reentrant=False)
                    static_embedding = checkpoint.checkpoint(self.static_encoder, static_features, use_reentrant=False)
                else:
                    chartevents_embedding, _, _ = self.context_encoder_24h(src=ce_src, static=None, times=ce_times, lengths=ce_lengths)
                    context_embedding, _, _ = self.context_encoder_60m(src=src, static=None, times=times, lengths=lengths)
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
            # Build expert initial state analytically from chartevents + IC, then fuse with neural embedding.
            expert_state = self._compute_full_analytical_zenker_initial_conditions(
                ce_rd_src, init_states, ic_mask
            )  # [B, 14]
            neural_embedding_expanded = neural_embedding.unsqueeze(1).repeat(1, self.num_samples, 1)  # [B, S, D_neural]
            expert_init_expanded = expert_state.unsqueeze(1).repeat(1, self.num_samples, 1)  # [B, S, 14]
            z1_for_sde = torch.cat([expert_init_expanded, neural_embedding_expanded], dim=-1)  # [B, S, 14 + D_neural]
            # Sanitize initial latents: clamp expert latents and normalize/clamp neural embedding
            try:
                expert_part = z1_for_sde[..., : self.expert_latent_dims]
                neural_part = z1_for_sde[..., self.expert_latent_dims :]
                expert_part = torch.clamp(expert_part, min=self.physio_min_vals, max=self.physio_max_vals)
                if neural_part.numel() > 0:
                    try:
                        neural_part = F.layer_norm(neural_part, neural_part.shape[-1:])
                    except Exception:
                        pass
                    neural_part = torch.clamp(neural_part, -50.0, 150.0)
                z1_for_sde = torch.cat([expert_part, neural_part], dim=-1)
            except Exception:
                pass

            ic_consistency_loss = torch.tensor(0.0, device=self.device)
            # === ARCHITECTURE CHANGE: END ===
            
        else: # No encoder path
            # When no encoder is used, fuse analytical expert state with a zero neural embedding.
            neural_embedding = torch.zeros(batch_size, self.encoder_SDENN_dims, device=self.device)
            expert_state = self._compute_full_analytical_zenker_initial_conditions(
                ce_rd_src, init_states, ic_mask
            )  # [B, 14]
            neural_embedding_expanded = neural_embedding.unsqueeze(1).repeat(1, self.num_samples, 1)  # [B, S, D_neural]
            expert_init_expanded = expert_state.unsqueeze(1).repeat(1, self.num_samples, 1)  # [B, S, 14]
            z1_for_sde = torch.cat([expert_init_expanded, neural_embedding_expanded], dim=-1)  # [B, S, 14 + D_neural]
            # Sanitize initial latents: clamp expert latents and normalize/clamp neural embedding
            try:
                expert_part = z1_for_sde[..., : self.expert_latent_dims]
                neural_part = z1_for_sde[..., self.expert_latent_dims :]
                expert_part = torch.clamp(expert_part, min=self.physio_min_vals, max=self.physio_max_vals)
                if neural_part.numel() > 0:
                    try:
                        neural_part = F.layer_norm(neural_part, neural_part.shape[-1:])
                    except Exception:
                        pass
                    neural_part = torch.clamp(neural_part, -50.0, 150.0)
                z1_for_sde = torch.cat([expert_part, neural_part], dim=-1)
            except Exception:
                pass
            logqp0 = 0
            ic_consistency_loss = torch.tensor(0.0, device=self.device)

        valid_lengths = (Y_mask.sum(dim=2) > 0).sum(dim=1)

        # Attach precomputed med_context for fast per-step indexing
        self.current_med_tensors = med_tensors if med_tensors is not None else None

        # Repeat neural_embedding for each sample
        neural_embedding_expanded = neural_embedding.unsqueeze(1).repeat(1, self.num_samples, 1)

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
        decoded_traj = self.forward_dec(latent_traj)
        try:
            self._debug_predicted_traj = decoded_traj.detach().clone()
            self._debug_true_traj = Y.detach().clone()
            if self.debug:
                print(f"[DEBUG] Stored trajectories in common_step")
        except Exception as e:
            if self.debug:
                print(f"[WARN] Failed to store debug trajectories: {e}")
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
                    tv_loss = (du**2).mean()
                    if hasattr(self, 'use_control_tv_loss') and self.use_control_tv_loss:
                        try:
                            # tv_loss should be available from the computation above
                            self.log("train_tv_loss", tv_loss.detach(), on_step=True, on_epoch=True, prog_bar=False,
                                     logger=True, sync_dist=True)
                            self.log("train_tv_contribution", self.control_tv_weight * tv_loss.detach(), on_step=True,
                                     on_epoch=True, prog_bar=False, logger=True, sync_dist=True)
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
                self.log(f'train_learned_scale_dim_{i}', scale, on_step=True, on_epoch=False, prog_bar=False, logger=True)
        elif self.log_lik_scale_mode == 'annealing':
            self.log('train_annealed_scale', self.scale_scheduler.val, on_step=True, on_epoch=False, prog_bar=False, logger=True)

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
        # Log and print current per-control weights (adaptive ramp * global weighting)
        current_ctrl_weights = self.get_current_per_control_weights().detach().cpu()
        # Log via Lightning so it reaches the configured logger (e.g., wandb)
        for i in range(current_ctrl_weights.numel()):
            try:
                self.log(f"train_ctrl_weight_{i}", float(current_ctrl_weights[i].item()), on_step=True, on_epoch=False, prog_bar=False, logger=True)
            except Exception:
                pass
        # Also log dynamic per-control sigmas and weights succinctly
        current_sigmas = self.get_current_per_control_sigmas().detach().cpu()
        for i in range(current_sigmas.numel()):
            try:
                self.log(f"train_ctrl_sigma_{i}", float(current_sigmas[i].item()), on_step=True, on_epoch=False, prog_bar=False, logger=True)
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
            if self.debug:
                print(f"[DEBUG] Plotting check: step={step}, interval={interval}, should_plot_now={should_plot_now}, plot_outputs_train={self.plot_outputs_train}")
        except Exception as e:
            should_plot_now = self.plot_outputs_train
            if self.debug:
                print(f"[DEBUG] Plotting check failed: {e}, using plot_outputs_train={self.plot_outputs_train}")

        if self.plot_outputs_train and should_plot_now and self._is_rank_zero():
            if self.debug:
                print(f"[DEBUG] Generating control plots at step {getattr(self, 'global_step', 'unknown')} (rank 0 only)")
            try:
                self.plot_nature_with_controls(
                    result["decoded_traj"],
                    result["Y"],
                    result["combined_mask"],
                    result["i_ext_path"],
                    batch_idx,
                    "",
                    result["z1_for_sde"],
                    result.get("latent_traj"),
                    result.get("med_combo_ids")
                )
                if self.debug:
                    print(f"[DEBUG] Control plots generated successfully")
            except Exception as e:
                if self.debug:
                    print(f"[WARN] Training control plot failed: {e}")
                    import traceback
                    print(f"[WARN] Full traceback: {traceback.format_exc()}")

        # Disabled CSV exporting for speed as requested

        # Log metrics
        self.log(
            "train_total_loss",
            total_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        self.log(
            "train_control_energy",
            control_energy.detach(),
            on_step=True,
            on_epoch=True,
            prog_bar=False,
            logger=True,
            sync_dist=True,
        )
        self.log(
            "train_main_loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        self.log(
            "train_ic_consistency_loss",
            ic_consistency_loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        self.log(
            f"train_{self.loss_type}", recon_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True
        )
        self.log(
            "train_KL", kl_div, on_step=True, on_epoch=True, prog_bar=True, logger=True, sync_dist=True
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
            sync_dist=True,
        )
        self.log(
            "val_main_loss",
            result["loss"],
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        self.log(
            "val_ic_consistency_loss",
            result["ic_consistency_loss"],
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        self.log(
            f"val_{self.loss_type}",
            result["recon_loss"],
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
        )
        self.log(
            "val_KL",
            result["kl_div"],
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
            sync_dist=True,
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

    def test_step(self, batch, batch_idx, dataloader_idx = 0):
        if self.debug and batch_idx == 0:
            print(f"[DEBUG] Hybrid_SDE validation_step: batch_idx={batch_idx}")

        result = self.common_step(batch, batch_idx)

        total_loss = result["total_loss"]
        loss = result["loss"]
        recon_loss = result["recon_loss"]
        kl_div = result["kl_div"]
        suffix = "_all" if dataloader_idx == 0 else "_filtered"

        # Log with dataset-specific names
        self.log(f"test_total_loss{suffix}", total_loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log(f"test_main_loss{suffix}", loss, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log(f"test_ic_consistency_loss{suffix}", result["ic_consistency_loss"], on_step=False, on_epoch=True,
                 prog_bar=True, logger=True, sync_dist=True)
        self.log(f"test_{self.loss_type}{suffix}", result["recon_loss"], on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
        self.log(f"test_KL{suffix}", kl_div, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)


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
            self.log(f"test_mse{suffix}", valid_mse, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)
            self.log(f"test_mae{suffix}", valid_mae, on_step=False, on_epoch=True, prog_bar=True, logger=True, sync_dist=True)

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
                         logger=True, sync_dist=True)
                self.log(f"test_zenker_mae{suffix}", zenker_mae, on_step=False, on_epoch=True, prog_bar=True,
                         logger=True, sync_dist=True)
        if batch_idx < 3 and self._is_rank_zero():
            self.plot_nature_style_with_uncertainty(
                decoded_traj, Y, combined_mask, batch_idx, suffix, result["med_combo_ids"]
            )
            self.plot_nature_with_controls(
                decoded_traj,
                Y,
                combined_mask,
                result["i_ext_path"],
                batch_idx,
                suffix,
                result["z1_for_sde"],
                result["med_combo_ids"]
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

    def _is_rank_zero(self):
        """Check if this is the main process (rank 0) in DDP training."""
        try:
            if hasattr(self, 'trainer') and self.trainer is not None:
                rank = self.trainer.global_rank
                if self.debug:
                    print(f"[DEBUG] DDP rank check: global_rank={rank}, is_rank_zero={rank == 0}")
                return rank == 0
        except Exception as e:
            if self.debug:
                print(f"[DEBUG] DDP rank check failed: {e}, defaulting to rank 0")
        return True  # Fallback to True if we can't determine rank
    
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

            if self.log_wandb and getattr(self, 'trainer', None) is not None and self.trainer.is_global_zero and getattr(self, 'logger', None) is not None and hasattr(self.logger, 'experiment'):
                try:
                    self.logger.experiment.log({f"uncertainty_plot_batch_{batch_idx}_patient_{patient_idx}{suffix}": wandb.Image(plt)})
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

    def plot_nature_with_controls(
        self,
        predictions_full,
        targets,
        combined_mask,
        i_ext_path,
        batch_idx,
        suffix = "",
        z1_for_sde=None,
        latent_traj=None,
        med_combo_ids=None
    ):
        """Plot BP + controls with detailed control analysis (Zenker baseline removed)."""

        if self.debug:
            print(f"[DEBUG] plot_nature_with_controls called with shapes: pred={predictions_full.shape}, targets={targets.shape}, mask={combined_mask.shape}")
            print(f"[DEBUG] i_ext_path shape: {i_ext_path.shape if i_ext_path is not None else 'None'}")
            print(f"[DEBUG] train_dir: {self.train_dir}")

        colors = self._setup_plot_style()  # Use same style
        control_plots_dir = os.path.join(self.train_dir, "control_plots")
        os.makedirs(control_plots_dir, exist_ok=True)
        
        if self.debug:
            print(f"[DEBUG] Created control_plots directory: {control_plots_dir}")

        # Optionally skip burn-in period for plotting
        dt = 10  # Time step in seconds between trajectory points
        burn_in_steps = int(self.sde_burn_in_period / dt)
        
        if (not self.plot_include_burn_in) and targets.shape[1] > burn_in_steps:
            predictions_full = predictions_full[:, :, burn_in_steps:, :]
            targets = targets[:, burn_in_steps:, :]
            combined_mask = combined_mask[:, burn_in_steps:, :]
            if i_ext_path is not None:
                i_ext_path = i_ext_path[:, :, burn_in_steps:, :]

        try:
            import matplotlib.pyplot as plt
            plt.switch_backend("Agg")
            if self.debug:
                print(f"[DEBUG] Matplotlib backend set to Agg")
        except Exception as e:
            if self.debug:
                print(f"[WARN] Failed to set matplotlib backend: {e}")
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

        # Determine how many latent panels we'll actually draw (avoid blank subplots)
        selected_latent_indices = [0, 1, 2, 3, 9, 10, 4]  # p_a, p_v, s, sv, ca, cv, r_tpr_mod
        selected_latent_names = ["p_a", "p_v", "s", "sv", "ca", "cv", "r_tpr_mod"]
        num_latent_panels = len(selected_latent_indices) if latent_traj is not None else 0

        # Only allow rank 0 to write plots (avoid DDP duplicates)
        if getattr(self, 'trainer', None) is not None and not self.trainer.is_global_zero:
            if self.debug:
                print(f"[DEBUG] Skipping plot generation - not global zero rank")
            return

        num_patients = min(self.max_plotted_patients, predictions_full.shape[0])
        if self.debug:
            print(f"[DEBUG] Starting control plot generation for {num_patients} patients")

        for patient_idx in range(num_patients):
            if self.debug:
                print(f"[DEBUG] Processing patient {patient_idx}/{num_patients}")
            
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
            
            if self.debug:
                print(f"[DEBUG] Patient {patient_idx} data shapes: pred_mean={pred_mean_patient.shape}, true={true_patient.shape}, control_mean={control_mean_patient.shape}")

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

            # Create top BP panel + selected expert latents + 2*num_controls derivative/state panels
            num_controls = (
                control_mean_patient.shape[1] if control_mean_patient.ndim == 2 else 1
            )
            nrows = 1 + num_latent_panels + 2 * num_controls
            
            if self.debug:
                print(f"[DEBUG] Patient {patient_idx}: Creating figure with {nrows} rows, {num_controls} controls")
            
            fig, axes = plt.subplots(
                nrows, 1, figsize=(8, 2.0 * nrows + 3), sharex=True
            )
            if nrows == 1:
                axes = [axes]
            ax1 = axes[0]
            
            if self.debug:
                print(f"[DEBUG] Patient {patient_idx}: Figure created successfully, starting plot generation")

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
                if self.debug_level >= 2:
                    print(f"[DEBUG] Drawing {max_overlays} arterial sample overlays for patient {patient_idx}")
                for s_idx in range(max_overlays):
                    sample_line = samples_np[s_idx, :, 0].copy()
                    sample_line[~arterial_mask] = np.nan
                    ax1.plot(
                        time_seconds,
                        sample_line,
                        color=colors["arterial_pred"],
                        linewidth=1.2,
                        linestyle="-",
                        alpha=0.6,
                        label="Sample traces" if s_idx == 0 else "",
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
                if self.debug_level >= 2:
                    print(f"[DEBUG] Drawing {max_overlays} venous sample overlays for patient {patient_idx}")
                for s_idx in range(max_overlays):
                    sample_line = samples_np[s_idx, :, 1].copy()
                    sample_line[~venous_mask] = np.nan
                    ax1.plot(
                        time_seconds,
                        sample_line,
                        color=colors["venous_pred"],
                        linewidth=1.2,
                        linestyle="-",
                        alpha=0.6,
                        label="Sample traces" if s_idx == 0 else "",
                    )

            ax1.set_xlim(0, 1200)
            ax1.set_ylabel("Pressure (mmHg)", fontweight="bold")
            # Draw t0 marker (after burn-in) across all subplots when including burn-in
            if self.plot_include_burn_in and burn_in_steps > 0:
                try:
                    ax1.axvline(x=burn_in_steps * dt, color="#444", linestyle=":", linewidth=1.0, alpha=0.8)
                except Exception:
                    pass
            ax1.legend(
                loc="upper right",
                fancybox=False,
                facecolor="white",
                framealpha=1.0,
                fontsize=7,
            )
            ax1.grid(True, alpha=0.2)

            # === Expert latent variable panels (selected Zenker state variables) ===
            if latent_traj is not None and num_latent_panels > 0:
                try:
                    latent_mean = latent_traj.mean(1).detach().cpu().numpy()  # [B,T,14]
                    latent_patient = latent_mean[patient_idx]
                    for panel_idx, latent_idx in enumerate(selected_latent_indices):
                        axi = axes[1 + panel_idx]
                        series = latent_patient[:, latent_idx].copy()
                        # Use bp availability to blank gaps if desired
                        series[~bp_available_mask] = np.nan
                        axi.plot(time_seconds, series, color="#555", linewidth=1.2)
                        if self.plot_include_burn_in and burn_in_steps > 0:
                            try:
                                axi.axvline(x=burn_in_steps * dt, color="#444", linestyle=":", linewidth=1.0, alpha=0.8)
                            except Exception:
                                pass
                        axi.set_ylabel(selected_latent_names[panel_idx])
                        axi.grid(True, alpha=0.2)
                except Exception:
                    pass

            # === Controls: one subplot per control dimension ===
            control_cmap = plt.cm.get_cmap("tab10", num_controls)
            control_short = ["dpv", "sv", "ca", "r_tpr_mod"]
            control_pretty = [
                "Venous pressure",
                "Stroke volume",
                "Arterial compliance",
                "TPR modifier",
            ]
            for control_idx in range(num_controls):
                axc = axes[1 + num_latent_panels + 2 * control_idx]
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

                target_suffix = (
                    f" ({control_short[control_idx]})"
                    if control_idx < len(control_short)
                    else ""
                )
                axc.plot(
                    time_seconds,
                    control_values,
                    color=color,
                    linewidth=2.0,
                    linestyle="--",
                    label=f"Control {control_idx + 1}{target_suffix}",
                )
                if self.plot_include_burn_in and burn_in_steps > 0:
                    try:
                        axc.axvline(x=burn_in_steps * dt, color="#444", linestyle=":", linewidth=1.0, alpha=0.8)
                    except Exception:
                        pass
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
                    max_control_overlays = min(sample_controls.shape[0], 6)
                    if self.debug_level >= 2:
                        print(f"[DEBUG] Drawing {max_control_overlays} control {control_idx} sample overlays for patient {patient_idx}")
                    for s_idx in range(max_control_overlays):
                        axc.plot(
                            time_seconds,
                            sample_controls[s_idx],
                            color=color,
                            linewidth=1.0,
                            alpha=0.5,
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
                axd = axes[1 + num_latent_panels + 2 * control_idx + 1]
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
                    label=f"d(Control {control_idx + 1}{target_suffix})/dt (mean)",
                )
                if self.plot_include_burn_in and burn_in_steps > 0:
                    try:
                        axd.axvline(x=burn_in_steps * dt, color="#444", linestyle=":", linewidth=1.0, alpha=0.8)
                    except Exception:
                        pass
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
                if self.debug_level >= 2:
                    print(f"[DEBUG] Drawing {max_overlays} derivative sample overlays for control {control_idx}, patient {patient_idx}")
                for s_idx in range(max_overlays):
                    axd.plot(
                        time_seconds,
                        sample_derivs[s_idx],
                        color=color,
                        linewidth=0.9,
                        linestyle="-",
                        alpha=0.4,
                    )
                axd.set_ylabel(
                    f"dCtrl {control_idx + 1}/dt"
                    + (f" ({control_short[control_idx]})" if control_idx < len(control_short) else "")
                )
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
            
            med_info_str = "Unknown Meds"
            if med_combo_ids is not None:
                try:
                    combo_id = int(med_combo_ids[patient_idx].item())
                    med_info_str = self.id_to_combo_map.get(combo_id, {}).get("str", f"ID {combo_id}")
                except (IndexError, AttributeError, ValueError):
                    pass

            mapping_line = "Controls: " + ", ".join(
                [
                    (f"C{idx+1}→{control_short[idx]}" if idx < len(control_short) else f"C{idx+1}")
                    for idx in range(num_controls)
                ]
            )
            plt.suptitle(
                f"{epoch_tag} – Patient {patient_idx} (Batch {batch_idx})\n{med_info_str}\n{mapping_line}",
                fontweight="bold",
                fontsize=10
            )
            plt.tight_layout()

            wandb_condition = self.log_wandb and getattr(self, 'trainer', None) is not None and self.trainer.is_global_zero and getattr(self, 'logger', None) is not None and hasattr(self.logger, 'experiment')
            if self.debug:
                print(f"[DEBUG] Patient {patient_idx}: wandb_condition={wandb_condition}, log_wandb={self.log_wandb}, trainer={getattr(self, 'trainer', None) is not None}, is_global_zero={getattr(self.trainer, 'is_global_zero', 'no_trainer') if getattr(self, 'trainer', None) is not None else 'no_trainer'}")
            
            # Always save locally first
            if self.debug:
                print(f"[DEBUG] Patient {patient_idx}: Saving to file")
            try:
                last_hadm = int(getattr(self, "_last_hadm_ids", [None])[patient_idx])
                last_traj = int(getattr(self, "_last_traj_ids", [None])[patient_idx])
                id_suffix = f"_hadm{last_hadm}_traj{last_traj}" if last_hadm is not None and last_traj is not None else ""
            except Exception:
                id_suffix = ""
            out_path = os.path.join(
                self.train_dir,
                f"control_plots/{epoch_tag}_patient{patient_idx}_batch{batch_idx}{id_suffix}{suffix}_controls.png",
            )
            if self.debug:
                print(f"[DEBUG] About to save plot to: {out_path}")
            plt.savefig(out_path, dpi=300, bbox_inches="tight")
            print(f"[PLOT] Saved: {out_path}")
            
            # Also log to wandb if enabled
            if wandb_condition:
                if self.debug:
                    print(f"[DEBUG] Patient {patient_idx}: Also logging to wandb")
                try:
                    self.logger.experiment.log({f"enhanced_control_plot_batch_{batch_idx}_patient_{patient_idx}{suffix}": wandb.Image(plt)})
                except Exception as e:
                    if self.debug:
                        print(f"[WARN] Failed to log to wandb: {e}")
            
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
            for patient_idx in range(min(8, control_mean.shape[0])):
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
        if self.log_wandb and getattr(self, 'trainer', None) is not None and self.trainer.is_global_zero and getattr(self, 'logger', None) is not None and hasattr(self.logger, 'experiment'):
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
                std_dev = math.sqrt(max(0, mean_loss_sq - mean_loss**2))
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
