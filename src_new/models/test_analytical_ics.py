import sys
import os
import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import numpy as np
import matplotlib.pyplot as plt

# Add the parent directory (src_new) to the Python path to resolve imports
# This allows running the script from the project root `Counterfactual_ICU`
script_dir = os.path.dirname(os.path.abspath(__file__))
src_dir = os.path.dirname(script_dir)
if src_dir not in sys.path:
    sys.path.insert(0, src_dir)

# Change to the models directory where the configs are located
os.chdir(script_dir)

from dataloaders.MIMIC_data import MIMICDataModule
from models.hybrid_sde_model import Hybrid_SDE
from models.ZenkerModel import ZenkerODE
from utils.train_utils import _EXPERT_ORDER

# Define a function to map the expert state vector to ZenkerODE kwargs
def map_expert_state_to_zenker_kwargs(expert_state_np: np.ndarray) -> dict:
    """Maps a 14-element numpy array to ZenkerODE constructor arguments."""
    state_dict = {key: val for key, val in zip(_EXPERT_ORDER, expert_state_np)}
    
    # Map to the names expected by ZenkerODE __init__
    return {
        'p_a_init': state_dict.get('p_a', 0),
        'p_v_init': state_dict.get('p_v', 0),
        's_reflex_init': state_dict.get('s_reflex', 0),
        'sv_init': state_dict.get('sv', 0),
        'r_tpr_mod': state_dict.get('r_tpr_mod', 0),
        'f_hr_max': state_dict.get('f_hr_max', 0),
        'f_hr_min': state_dict.get('f_hr_min', 0),
        'r_tpr_max': state_dict.get('r_tpr_max', 0),
        'r_tpr_min': state_dict.get('r_tpr_min', 0),
        'ca': state_dict.get('ca', 0),
        'cv': state_dict.get('cv', 0),
        'k_width': state_dict.get('k_width', 0),
        'p_aset': state_dict.get('p_aset', 0),
        'tau': state_dict.get('tau', 0),
    }

@hydra.main(version_base=None, config_path="configs", config_name="config")
def test_analytical_ics(cfg: DictConfig) -> None:
    """
    Test script to validate _compute_full_analytical_zenker_initial_conditions.
    """
    print("--- Test Script: Analytical Initial Conditions ---")
    
    # 1. Load Data
    print("Initializing DataModule...")
    data_module = MIMICDataModule(
        data_root=cfg.data_config.data_root,
        icu_stays_path=cfg.data_config.icu_stays_path,
        batch_size=4,
        num_workers=0,
        max_samples=cfg.data_config.max_samples,
        split_mode=cfg.data_config.split_mode,
        random_state=cfg.seed
    )
    print("Setting up data...")
    data_module.setup('fit')
    
    train_loader = data_module.train_dataloader()
    batch = next(iter(train_loader))
    print("Loaded one batch of data.")

    # 2. Instantiate the Hybrid_SDE model
    print("Instantiating Hybrid_SDE model...")
    model = Hybrid_SDE(
        use_encoder=cfg.model.use_encoder,
        start_dec_at_treatment=cfg.model.start_dec_at_treatment,
        variational_sampling=cfg.model.variational_sampling,
        # Encoder
        context_input_dim=data_module.context_input_dim,
        chartevents_input_dim=data_module.chartevents_input_dim,
        encoder_hidden_dim=cfg.model.encoder_hidden_dim,
        expert_latent_dims=14,  # Fixed by the ODE model
        encoder_SDENN_dims=0 if cfg.model.use_encoder == "none" else cfg.model.encoder_SDENN_dims,
        n_medications=data_module.n_medications,
        med_embed_dim=cfg.model.med_embed_dim,
        encoder_context_len=data_module.context_max_len,
        use_2_5std_encoder_minmax=cfg.model.use_2_5std_encoder_minmax,
        encoder_num_layers=cfg.model.encoder_num_layers,
        variational_encoder=cfg.model.variational_encoder,
        encoder_w_time=cfg.model.encoder_w_time,
        encoder_reverse_time=cfg.model.encoder_reverse_time,
        # Integration
        integration_step_size=cfg.model.integration_step_size,
        integration_method=cfg.model.integration_method,
        atol=cfg.model.integration_atol,
        rtol=cfg.model.integration_rtol,
        integration_adaptive=cfg.model.integration_adaptive,
        # Static Fusion
        static_input_dim=data_module.static_input_dim,
        static_hidden_dim=cfg.model.static_hidden_dim,
        fusion_hidden_dim=cfg.model.fusion_hidden_dim,
        # SDE params
        num_samples=cfg.model.num_samples,
        normalise_for_SDENN=cfg.model.normalise_for_SDENN,
        self_reverting_prior_control=cfg.model.self_reverting_prior_control,
        prior_tx_sigma_per_control=cfg.model.prior_tx_sigma_per_control,
        prior_tx_mu=cfg.model.prior_tx_mu,
        theta=cfg.model.theta,
        SDE_control_weighting=cfg.model.SDE_control_weighting,
        use_control_lowpass=cfg.model.use_control_lowpass,
        control_lowpass_tau=cfg.model.control_lowpass_tau,
        use_control_tv_loss=cfg.model.use_control_tv_loss,
        control_tv_weight=cfg.model.control_tv_weight,
        override_control_scales=cfg.model.override_control_scales,
        control_energy_weight=cfg.model.control_energy_weight,
        # SDE model params
        SDE_input_state=cfg.model.SDE_input_state,
        include_time=cfg.model.include_time,
        SDEnet_hidden_dim=cfg.model.SDEnet_hidden_dim,
        SDEnet_depth=cfg.model.SDEnet_depth,
        SDEnet_out_dims=cfg.model.SDEnet_out_dims,
        use_batch_norm=cfg.model.use_batch_norm,
        final_activation=cfg.model.final_activation,
        # decoder params
        decoder_output_dims=[0, 1],
        normalised_data=False, # This is handled by the datamodule now
        log_lik_output_scale=cfg.model.output_scale,
        # admin
        train_dir=".",
        KL_weighting_SDE=cfg.model.KL_weighting_SDE,
        loss_type=cfg.model.loss_type,
        log_lik_scale_mode=cfg.model.log_lik_scale_mode,
        anneal_iters=5000,
        # Optimizer params
        use_lr_scheduler=cfg.model.use_lr_scheduler,
        total_training_steps=10000,
        warmup_steps=1000,
        min_lr=cfg.model.min_lr,
        learning_rate=cfg.model.learning_rate,
        optimizer_name=cfg.model.optimizer_name,
        log_wandb=False,
        adjoint=cfg.model.adjoint,
        plot_every=cfg.model.plot_every,
        batch_size=cfg.data_config.batch_size,
        dataset=cfg.data_config.dataset_type,
        test_zenker=cfg.model.test_zenker,
        debug=cfg.model.debug_level,
        force_no_controls=cfg.model.force_no_controls,
        plot_outputs_train=cfg.model.plot_outputs_train,
        # Controller selection (MLP vs GAT)
        controller_type=cfg.model.controller_type,
        gat_heads=cfg.model.gat_heads,
        gat_layers=cfg.model.gat_layers,
        gat_hidden=cfg.model.gat_hidden,
        gat_dropout=cfg.model.gat_dropout,
        sde_burn_in_period=cfg.model.sde_burn_in_period,
        log_train_combo_loss_every_n_steps=cfg.model.log_train_combo_loss_every_n_steps,
        scale_loss_by_variance=cfg.model.scale_loss_by_variance,
        ic_consistency_weight=cfg.model.ic_consistency_weight,
        forward_loss_weight=cfg.model.forward_loss_weight,
        use_wandb_for_logging=False,
        use_checkpointing=cfg.model.use_checkpointing,
        debug_level=cfg.model.debug_level,
        force_zenker_defaults=cfg.model.force_zenker_defaults,
        plot_control_samples=cfg.model.plot_control_samples,
        plot_include_burn_in=getattr(cfg.model, "plot_include_burn_in", True),
    )
    
    # 3. Compute Initial Conditions
    print("Computing analytical initial conditions...")
    print(f"Batch contains {len(batch)} elements")
    print("Batch structure:")
    for i, item in enumerate(batch):
        if hasattr(item, 'shape'):
            print(f"  {i}: {type(item).__name__} with shape {item.shape}")
        else:
            print(f"  {i}: {type(item).__name__} with value {item}")
    
    # Unpack the batch - let's be more careful about this
    if len(batch) >= 9:  # Make sure we have enough elements
        ce_rd_src = batch[3]  # Chartevents data
        init_states = batch[7]  # Initial states
        ic_mask = batch[8]  # Initial condition mask
    else:
        raise ValueError(f"Expected at least 9 elements in batch, got {len(batch)}")
    
    expert_states = model._compute_full_analytical_zenker_initial_conditions(
        ce_rd_src, init_states, ic_mask
    )
    print(f"Computed expert states with shape: {expert_states.shape}")
    
    # 4. & 5. Simulate and Plot
    output_dir = "zenker_ic_test_plots"
    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving plots to '{output_dir}/'")

    num_patients_to_plot = min(expert_states.shape[0], 5)
    
    for i in range(num_patients_to_plot):
        print(f"--- Simulating Patient {i} ---")
        patient_ic_np = expert_states[i].detach().cpu().numpy()
        zenker_kwargs = map_expert_state_to_zenker_kwargs(patient_ic_np)
        
        print("Initial conditions:")
        for k, v in zenker_kwargs.items():
            print(f"  {k}: {v:.3f}")
            
        zenker_model = ZenkerODE(**zenker_kwargs)
        t, solution = zenker_model.integrate(t_span=1200, dt=1.0)
        
        fig, axes = plt.subplots(4, 4, figsize=(20, 18))
        fig.suptitle(f'Zenker Simulation from Analytical ICs - Patient {i}', fontsize=16)
        axes_flat = axes.flatten()

        solution_map = {
            "p_a": solution[:, 0],
            "p_v": solution[:, 1],
            "s_reflex": solution[:, 2],
            "sv": solution[:, 3]
        }

        for idx, key in enumerate(_EXPERT_ORDER):
            if idx >= len(axes_flat): break
            ax = axes_flat[idx]
            
            if key in solution_map:
                y_data = solution_map[key]
                ax.plot(t, y_data, label=f'State: {key}')
                
                # Set appropriate y-axis limits and formatting
                y_min, y_max = np.min(y_data), np.max(y_data)
                y_range = y_max - y_min
                
                if y_range > 0:
                    # Add 10% padding to y-axis
                    padding = 0.1 * y_range
                    ax.set_ylim(y_min - padding, y_max + padding)
                
                # Use scientific notation for very small or very large values
                if y_max > 1e4 or (y_max < 1e-2 and y_max > 0):
                    ax.ticklabel_format(style='scientific', axis='y', scilimits=(-2, 4))
                else:
                    # Use regular formatting with appropriate precision
                    if y_range < 1:
                        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.3f}'))
                    elif y_range < 10:
                        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.2f}'))
                    else:
                        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1f}'))
                        
            else:
                param_key_in_zenker = key if key in zenker_kwargs else f"{key}_init"
                if param_key_in_zenker in zenker_kwargs:
                    value = zenker_kwargs[param_key_in_zenker]
                    ax.plot(t, np.full_like(t, value), label=f'Param: {key} = {value:.3f}', linestyle='--')
                    
                    # Set y-axis limits for constant parameters with some padding
                    if value != 0:
                        padding = abs(value) * 0.1
                        ax.set_ylim(value - padding, value + padding)
                    else:
                        ax.set_ylim(-0.1, 0.1)
                    
                    # Format y-axis for parameter values
                    if abs(value) > 1e4 or (abs(value) < 1e-2 and value != 0):
                        ax.ticklabel_format(style='scientific', axis='y', scilimits=(-2, 4))
                    else:
                        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.3f}'))
                else:
                     print(f"[WARN] Parameter '{key}' not in Zenker kwargs.")

            ax.set_title(key, fontsize=12, fontweight='bold')
            ax.set_xlabel('Time (s)', fontsize=10)
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper right', fontsize=9)

        for idx in range(len(_EXPERT_ORDER), len(axes_flat)):
            axes_flat[idx].set_visible(False)
            
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        save_path = os.path.join(output_dir, f'patient_{i}_simulation.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved plot to {save_path}")

if __name__ == "__main__":
    test_analytical_ics()
