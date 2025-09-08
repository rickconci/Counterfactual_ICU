import argparse
import os
import random
import tempfile
import sys
from datetime import datetime

import numpy as np
import torch

# <<< New global DEBUG variable >>>
DEBUG = False

from dataloaders.MIMIC_data import MIMICDataModule
from NSDE import NSDE
from synthetic_data import create_load_save_data, CVDataModule_IID, CVDataModule_OOD
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import WandbLogger

import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, EarlyStopping, TQDMProgressBar
from lightning.pytorch.loggers import WandbLogger
import wandb
import hydra
from omegaconf import DictConfig, OmegaConf

# Helper to suppress annoying pandas future warnings
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)


# def _parse_args():
#     parser = argparse.ArgumentParser(description="Train and evaluate the Hybrid SDE model.")
#     # Add all previous argparse arguments here...
#     return parser.parse_args()

def str2bool(v):
    """Parse flexible boolean values from CLI.

    Accepts: true/false, yes/no, y/n, 1/0 (case-insensitive). If provided without a value,
    it evaluates to True when used with nargs='?'.
    """
    if isinstance(v, bool):
        return v
    if v is None:
        return True
    s = str(v).strip().lower()
    if s in ("yes", "true", "t", "y", "1"):
        return True
    if s in ("no", "false", "f", "n", "0"):
        return False
    raise argparse.ArgumentTypeError("Boolean value expected (true/false)")


def set_seed(seed):
    seed_everything(seed, workers=True)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@hydra.main(version_base=None, config_path="configs", config_name="nsde_config")
def main(cfg: DictConfig) -> None:
    # Set seed for reproducibility
    set_seed(cfg.seed)

    # Print the full configuration
    print("--- Configuration ---")
    print(OmegaConf.to_yaml(cfg))
    print("-----------------------")

    # Initialize DataModule
    print("Initializing DataModule...")
    data_module = MIMICDataModule(
        data_root=cfg.data.data_root,
        icu_stays_path=cfg.data.icu_stays_path,
        batch_size=cfg.data.batch_size,
        num_workers=cfg.data.num_workers,
        max_samples=cfg.data.max_samples,
        split_mode=cfg.data.split_mode,
        ood_holdout_ratio=cfg.data.ood_holdout_ratio,
        filter_flat_trajectories=cfg.data.filter_flat_trajectories,
        test_both_filtered_and_unfiltered=cfg.data.test_both_filtered_and_unfiltered,
        use_raindrop_context=cfg.data.use_raindrop_context,
        expert_latent_dim=cfg.data.expert_latent_dim,
        random_state=cfg.seed
    )
    print("Setting up data...")
    data_module.setup()

    # Calculate scheduler iterations based on total training steps
    try:
        num_batches_per_epoch = len(data_module.train_dataloader())
        total_training_steps = cfg.trainer.max_epochs * num_batches_per_epoch
        warmup_steps = int(total_training_steps * cfg.model.warmup_fraction)
        print(f"Total training steps: {total_training_steps}. LR scheduler warmup over {warmup_steps} steps.")
    except Exception as e:
        print(f"[WARN] Could not determine train dataloader length to calculate scheduler steps: {e}")
        print("[WARN] Falling back to default steps.")
        total_training_steps = 10000
        warmup_steps = 1000

    # Calculate annealing iterations based on total training steps
    if cfg.model.log_lik_scale_mode == 'annealing':
        anneal_iters = int(total_training_steps * cfg.model.anneal_fraction)
        print(f"Annealing NLL scale over {anneal_iters} steps ({cfg.model.anneal_fraction * 100:.1f}% of training).")
    else:
        anneal_iters = 2000  # Default value, will not be used by the model in other modes

    # Hydra handles the output directory automatically
    train_dir_final = hydra.core.hydra_config.HydraConfig.get().runtime.output_dir
    print(f"All outputs will be saved to: {train_dir_final}")

    model = NSDE(
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
        prior_tx_sigma=cfg.model.prior_tx_sigma,
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
        normalised_data=False,  # This is handled by the datamodule now
        log_lik_output_scale=cfg.model.output_scale,
        # admin
        train_dir=train_dir_final,
        KL_weighting_SDE=cfg.model.KL_weighting_SDE,
        loss_type=cfg.model.loss_type,
        log_lik_scale_mode=cfg.model.log_lik_scale_mode,
        anneal_iters=anneal_iters,
        # Optimizer params
        use_lr_scheduler=cfg.model.use_lr_scheduler,
        total_training_steps=total_training_steps,
        warmup_steps=warmup_steps,
        min_lr=cfg.model.min_lr,
        learning_rate=cfg.model.learning_rate,
        optimizer_name=cfg.model.optimizer_name,
        log_wandb=not cfg.disable_wandb,
        adjoint=cfg.model.adjoint,
        plot_every=cfg.model.plot_every,
        batch_size=cfg.data.batch_size,
        dataset=cfg.data.dataset_type,
        id_to_combo_map=(
            getattr(data_module.train_dataset, 'id_to_combo_map', None)
            if hasattr(data_module, 'train_dataset') and data_module.train_dataset is not None
            else None
        ),
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
        use_wandb_for_logging=not cfg.disable_wandb,
        use_checkpointing=cfg.model.use_checkpointing,
        debug_level=cfg.model.debug_level,
        force_zenker_defaults=cfg.model.force_zenker_defaults,
        plot_control_samples=cfg.model.plot_control_samples,
        plot_include_burn_in=getattr(cfg.model, "plot_include_burn_in", True),
    )

    # Setup Logging & Callbacks
    logger = None
    if not cfg.disable_wandb:
        run_name = cfg.run_name or f"{cfg.model.use_encoder}_{cfg.data.split_mode}_lr{cfg.model.learning_rate}_seed{cfg.seed}"
        logger = WandbLogger(
            project=cfg.experiment_name,
            name=run_name,
            notes=cfg.notes,
            log_model=True,
            config=OmegaConf.to_container(cfg, resolve=True),
        )

    callbacks = []
    if not cfg.trainer.disable_model_checkpoint:
        checkpoint_callback = ModelCheckpoint(
            monitor="val_total_loss",
            dirpath=os.path.join(train_dir_final, "checkpoints"),
            filename="best_model-{epoch:02d}-{val_loss:.2f}",
            save_top_k=1,
            mode="min",
            save_last=True,
            verbose=True,
        )
        callbacks.append(checkpoint_callback)

    if not cfg.trainer.disable_early_stopping:
        early_stop_callback = EarlyStopping(
            monitor="val_total_loss",
            min_delta=0.00,
            patience=cfg.trainer.early_stopping_patience,
            verbose=True,
            mode="min",
        )
        callbacks.append(early_stop_callback)

    progress_bar = TQDMProgressBar(refresh_rate=10)
    callbacks.append(progress_bar)

    # Initialize Trainer
    print("Initializing Trainer...")
    trainer = L.Trainer(
        max_epochs=cfg.trainer.max_epochs,
        min_epochs=cfg.trainer.min_epochs,
        max_steps=cfg.trainer.max_steps,
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        strategy=cfg.trainer.strategy,
        gradient_clip_val=cfg.trainer.gradient_clip_val,
        gradient_clip_algorithm=cfg.trainer.gradient_clip_algorithm,
        log_every_n_steps=cfg.trainer.log_every_n_steps,
        check_val_every_n_epoch=cfg.trainer.check_val_every_n_epoch,
        deterministic=cfg.trainer.deterministic,
        precision=cfg.trainer.precision,
        accumulate_grad_batches=cfg.trainer.accumulate_grad_batches,
        limit_train_batches=cfg.trainer.limit_train_batches,
        overfit_batches=cfg.trainer.overfit_batches,
        logger=logger,
        callbacks=callbacks,
        num_sanity_val_steps=0 if cfg.trainer.disable_sanity_check else 1,
        limit_val_batches=0 if cfg.trainer.disable_sanity_check else 1.0,
    )
    if DEBUG:
        print("[DEBUG] main_beta.py: Trainer initialized. Starting fit...")
    trainer.fit(model, data_module)
    if DEBUG:
        print("[DEBUG] main_beta.py: Trainer fit completed.")
    # At the end of main(), after trainer.fit():
    if cfg.run_eval:
        print("Running evaluation on test set...")
        test_results = trainer.test(ckpt_path="best", dataloaders=data_module)

        if cfg.data.test_both_filtered_and_unfiltered:
            print(f"Test results (All trajectories): {test_results[0]}")
            print(f"Test results (Filtered trajectories): {test_results[1]}")

            # Log results to wandb if enabled
            if not cfg.disable_wandb and logger is not None:
                # Log metrics from all trajectories dataset
                for key, value in test_results[0].items():
                    logger.experiment.log({f"test_all/{key}": value})

                # Log metrics from filtered trajectories dataset
                for key, value in test_results[1].items():
                    logger.experiment.log({f"test_filtered/{key}": value})

                # Also log summary comparison metrics
                if "test_mse" in test_results[0] and "test_mse" in test_results[1]:
                    logger.experiment.log({
                        "test_comparison/mse_all_vs_filtered": test_results[0]["test_mse"] - test_results[1]["test_mse"]
                    })
                if "test_mae" in test_results[0] and "test_mae" in test_results[1]:
                    logger.experiment.log({
                        "test_comparison/mae_all_vs_filtered": test_results[0]["test_mae"] - test_results[1]["test_mae"]
                    })
        else:
            print(f"Test results: {test_results}")
            if not cfg.disable_wandb and logger is not None:
                for key, value in test_results[0].items():
                    logger.experiment.log({f"test/{key}": value})

    # Train the model
    print("Starting training...")
    trainer.fit(model, datamodule=data_module)
    print("Training finished.")

    # Test the model
    print("Starting testing...")
    trainer.test(model, datamodule=data_module)
    print("Testing finished.")

    if not cfg.disable_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
