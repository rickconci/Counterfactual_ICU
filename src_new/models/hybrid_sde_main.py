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
from hybrid_sde_model import Hybrid_SDE
from synthetic_data import create_load_save_data, CVDataModule_IID, CVDataModule_OOD
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import WandbLogger


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


def main(args):
    # <<< Update global DEBUG based on args >>>
    global DEBUG
    DEBUG = args.debug
    if DEBUG:
        print(f"[DEBUG] main_beta.py: Starting main function with args: {args}")

    # Redirect stdout/stderr to file if requested
    if getattr(args, "redirect_output", False):
        try:
            base_dir = args.train_dir if args.train_dir else os.path.join(os.getcwd(), "../../results")
            os.makedirs(base_dir, exist_ok=True)
            log_path = args.log_file if getattr(args, "log_file", "") else os.path.join(
                base_dir, f"train_{datetime.now().strftime('%Y%m%d-%H%M%S')}.log"
            )
            log_fh = open(log_path, "a", buffering=1)
            sys.stdout = log_fh
            sys.stderr = log_fh
            print(f"[LOG] Redirected stdout/stderr to: {log_path}")
        except Exception as e:
            # If redirection fails, continue without crashing
            print(f"[WARN] Failed to redirect output to file: {e}")

    print("CUDA GPUs present?", torch.cuda.is_available())
    if args.HPC_work:
        torch.set_float32_matmul_precision(
            "medium"
        )  # Faster computations with less precision

    saving_dir = "../../results/"
    os.environ["TMPDIR"] = os.path.join(os.getcwd(), "../../results/Tempdir")
    os.makedirs(os.environ["TMPDIR"], exist_ok=True)
    if DEBUG:
        print(f"[DEBUG] main_beta.py: Saving directory set to: {saving_dir}")
    print("Temporary directory set to:", tempfile.gettempdir())
    os.environ["WANDB_DIR"] = os.path.join(os.getcwd(), "../../results/Wandbdir")
    os.makedirs(os.environ["WANDB_DIR"], exist_ok=True)
    if DEBUG:
        print(f"[DEBUG] main_beta.py: WANDB_DIR set to: {os.environ['WANDB_DIR']}")
    print("Setting WANDB_DIR to:", os.environ["WANDB_DIR"])

    set_seed(args.seed)
    if DEBUG:
        print(f"[DEBUG] main_beta.py: Seed set to {args.seed}")

    if args.log_wandb:
        wandb_logger = WandbLogger(
            project=args.project_name,
            log_model=False,
            save_dir=os.path.join(saving_dir, "model_logs"),
        )
        wandb_logger.log_hyperparams(args)
        if DEBUG:
            print("[DEBUG] main_beta.py: Wandb logger initialized.")
    else:
        wandb_logger = None
        if DEBUG:
            print("[DEBUG] main_beta.py: Wandb logger not used.")

    if args.dataset_type == "synthetic":
        dataset_params = {
            "fixed_tx": args.fixed_tx,
            "include_all_inputs": args.include_all_inputs,
            "gamma": args.gamma,
            "sigma_tx": 0.01,
            "confounder_type": args.confounder_type,
            "non_confounded_effect": False,
            "noise_std": 0.0,
            "t_span": 60,
            "t_treatment": 45,
            "t_cutoff": 40,
            "seed": args.seed,
            "pre_treatment_dims": [0, 1],
            "post_treatment_dims": [0],
            "normalize": False,
            "N": 1280,
            "debug": args.debug,  # <<< Pass debug flag >>>
        }
        if DEBUG:
            print(f"[DEBUG] main_beta.py: Dataset params created: {dataset_params}")

        unique_dir_name = "_".join(
            [
                f"sd={args.seed}",
                f"gm={args.gamma}",
                f"ftx={args.fixed_tx}",
                f"allin={args.include_all_inputs}" f"cnf={args.confounder_type}",
                f"enc={args.use_encoder}",
                f"txsig={args.prior_tx_sigma}",
                f"revert={args.self_reverting_prior_control}",
                f"klw={args.KL_weighting_SDE}",
                f"SDEhd={args.SDEnet_hidden_dim}",
            ]
        )

        print("dataset_params", dataset_params)
        data_path = os.path.join(saving_dir, "data_created")
        if DEBUG:
            print(f"[DEBUG] main_beta.py: Data path set to: {data_path}")

        dataset_params["r_tpr_mod"] = -0.5
        if DEBUG:
            print(
                "[DEBUG] main_beta.py: Calling create_load_save_data for train/val data."
            )
        train_val_data = create_load_save_data(dataset_params, data_path)
        if DEBUG:
            print(
                f"[DEBUG] main_beta.py: train_val_data loaded. Type: {type(train_val_data)}"
            )
        dataset_params["r_tpr_mod"] = +0.2
        if DEBUG:
            print("[DEBUG] main_beta.py: Calling create_load_save_data for test data.")
        test_data = create_load_save_data(dataset_params, data_path)
        if DEBUG:
            print(f"[DEBUG] main_beta.py: test_data loaded. Type: {type(test_data)}")

        if DEBUG:
            print("[DEBUG] main_beta.py: Initializing CVDataModule_IID.")
        cv_data_module_IID = CVDataModule_IID(
            train_val_data=train_val_data,
            batch_size=args.batch_size,
            num_workers=0,
            debug=args.debug,
        )  # <<< Pass debug flag >>>
        if DEBUG:
            print("[DEBUG] main_beta.py: CVDataModule_IID initialized.")
        if DEBUG:
            print("[DEBUG] main_beta.py: Initializing CVDataModule_OOD.")
        cv_data_module_OOD = CVDataModule_OOD(
            OOD_test_data=test_data,
            batch_size=args.batch_size,
            num_workers=0,
            debug=args.debug,
        )  # <<< Pass debug flag >>>
        if DEBUG:
            print("[DEBUG] main_beta.py: CVDataModule_OOD initialized.")
        data_module = cv_data_module_IID
        data_module.setup()  # Need to call this to get dims

    elif args.dataset_type == "mimic":
        num_workers = 0
        if args.HPC_work:
            num_workers = 4
        data_module = MIMICDataModule(
            data_root=args.data_root,
            icu_stays_path=args.icu_stays_path,
            batch_size=args.batch_size,
            num_workers=num_workers,
            random_state=args.seed,
            max_samples=args.max_samples,
            use_raindrop_context=True,
            expert_latent_dim=14,
        )
        data_module.setup()
        unique_dir_name = f"MIMIC_DATA_seed={args.seed}"  # Simplified name for now

        dataset_params = dict()
        # predict MAP and CVP
        dataset_params["post_treatment_dims"] = 2
        dataset_params["normalize"] = True
    else:
        raise ValueError(
            "Invalid dataset_type specified. Choose 'synthetic' or 'mimic'."
        )

    if DEBUG:
        print("[DEBUG] main_beta.py: Initializing Hybrid_VAE_SDE model.")
    # Determine final train_dir; if not provided, compose from variable params
    if getattr(args, "train_dir", None):
        train_dir_final = args.train_dir
    else:
        # Compose name from key variable params with safe formatting
        enc_tag = f"enc-{args.use_encoder}"
        sdew_tag = f"sdew-{args.SDE_control_weighting}"
        lr_tag = f"lr-{args.learning_rate}"
        sigma_tag = f"sigma-{args.prior_tx_sigma}"
        ns_tag = f"ns-{args.num_samples}"
        seed_tag = f"seed-{args.seed}"
        auto_name = "_".join([enc_tag, sdew_tag, lr_tag, sigma_tag, ns_tag, seed_tag])
        train_dir_final = os.path.join(saving_dir, "experiments", auto_name)
    os.makedirs(train_dir_final, exist_ok=True)
    model = Hybrid_SDE(
        use_encoder=args.use_encoder,
        start_dec_at_treatment=args.start_dec_at_treatment,
        variational_sampling=args.variational_sampling,
        # Encoder
        encoder_input_dim=data_module.encoder_input_dim,
        encoder_hidden_dim=args.encoder_hidden_dim,
        expert_latent_dims=14,  # Fixed by the ODE model
        encoder_SDENN_dims=0 if args.use_encoder == "none" else args.encoder_SDENN_dims,
        n_medications=21,
        med_embed_dim=args.med_embed_dim,
        encoder_context_len=data_module.context_max_len,
        use_2_5std_encoder_minmax=args.use_2_5std_encoder_minmax,
        encoder_num_layers=args.encoder_num_layers,
        variational_encoder=args.variational_encoder,
        encoder_w_time=args.encoder_w_time,
        encoder_reverse_time=args.encoder_reverse_time,
        integration_step_size=args.integration_step_size,
        integration_method=args.integration_method,
        atol=args.integration_atol,
        rtol=args.integration_rtol,
        integration_adaptive=args.integration_adaptive,
        # New static fusion params
        static_input_dim=data_module.static_input_dim,
        static_hidden_dim=args.static_hidden_dim,
        fusion_hidden_dim=args.fusion_hidden_dim,
        # SDE params
        num_samples=args.num_samples,
        normalise_for_SDENN=args.normalise_for_SDENN,
        self_reverting_prior_control=args.self_reverting_prior_control,
        prior_tx_sigma=args.prior_tx_sigma,
        prior_tx_mu=args.prior_tx_mu,
        theta=args.theta,
        SDE_control_weighting=args.SDE_control_weighting,
        use_control_lowpass=args.use_control_lowpass,
        control_lowpass_tau=args.control_lowpass_tau,
        use_control_tv_loss=args.use_control_tv_loss,
        control_tv_weight=args.control_tv_weight,
        override_control_scales=args.override_control_scales,
        control_energy_weight=args.control_energy_weight,
        # SDE model params
        SDE_input_state=args.SDE_input_state,
        include_time=args.include_time,
        SDEnet_hidden_dim=args.SDEnet_hidden_dim,
        SDEnet_depth=args.SDEnet_depth,
        SDEnet_out_dims=args.SDEnet_out_dims,
        use_batch_norm=args.use_batch_norm,
        final_activation=args.final_activation,
        # decoder params
        decoder_output_dims=[0, 1],
        normalised_data=dataset_params["normalize"],
        log_lik_output_scale=args.output_scale,
        # admin
        train_dir=train_dir_final,
        KL_weighting_SDE=args.KL_weighting_SDE,
        learning_rate=args.learning_rate,
        log_wandb=args.log_wandb,
        adjoint=args.adjoint,
        plot_every=args.plot_every,
        batch_size=args.batch_size,
        dataset=args.dataset_type,
        test_zenker=args.test_zenker,
        debug=args.debug,  # <<< Pass debug flag >>>
        force_no_controls=args.force_no_controls,
        plot_outputs_train=args.plot_outputs_train,
        # Controller selection (MLP vs GAT)
        controller_type=args.controller_type,
        gat_heads=args.gat_heads,
        gat_layers=args.gat_layers,
        gat_hidden=args.gat_hidden,
        gat_dropout=args.gat_dropout,
    )
    # Optionally disable specific control heads for ablation
    if args.disable_controls:
        try:
            disabled = [
                int(x) for x in args.disable_controls.split(",") if x.strip() != ""
            ]
            model.disabled_control_indices = disabled
            if DEBUG:
                print(f"[DEBUG] main: Disabled control indices = {disabled}")
        except Exception as e:
            print(
                f"[WARN] Failed to parse --disable_controls='{args.disable_controls}': {e}"
            )
    os.makedirs(model.train_dir, exist_ok=True)
    if DEBUG:
        print("[DEBUG] main_beta.py: Hybrid_VAE_SDE model initialized.")

    callbacks = []

    if args.model_checkpoint:
        checkpoint_callback = ModelCheckpoint(
            monitor="val_total_loss",  # Ensure this is the exact name used in your logging
            dirpath=os.path.join(
                saving_dir, "model_checkpoints", unique_dir_name
            ),  # Directory to save checkpoints
            filename="best-{epoch:02d}-{val_loss:.2f}",
            save_top_k=1,
            mode="min",  # Minimize the monitored value
            save_last=True,  # Save the last model to resume training
            verbose=True,
        )
        callbacks.append(checkpoint_callback)

    if args.early_stopping:
        early_stopping = EarlyStopping(
            min_delta=0.00,
            monitor="val_total_loss",  # Ensure this is the exact name used in your logging
            patience=args.early_stopping_patience,  # num epochs with a val loss not improving before it stops
            mode="min",  # Minimize the monitored value
            verbose=True,
        )
        callbacks.append(early_stopping)
        if DEBUG:
            print("[DEBUG] main_beta.py: Early stopping callback added.")

    if DEBUG:
        print("[DEBUG] main_beta.py: Initializing Trainer.")
    trainer = Trainer(
        max_epochs=args.max_epochs,
        accelerator=args.accelerator,
        precision=args.precision,
        logger=wandb_logger,
        log_every_n_steps=args.log_every_n_steps,
        callbacks=callbacks,
        gradient_clip_val=1.0,  # Increased to 2.0 for extra stability
        gradient_clip_algorithm="norm",
        overfit_batches=args.overfit_batches,
        num_sanity_val_steps=0 if args.disable_sanity_check else 2,
        # Keep this >=1 to avoid modulo by zero; use limit_val_batches=0 to disable
        check_val_every_n_epoch=1,
        limit_val_batches=0 if args.disable_sanity_check else 1.0,
        # fast_dev_run = True,
        # overfit_batches = 1
        # deterministic=True,
        # check_val_every_n_epoch=1,
        # profiler="simple"   #this helps to identify bottlenecks
    )
    if DEBUG:
        print("[DEBUG] main_beta.py: Trainer initialized. Starting fit...")
    trainer.fit(model, data_module)
    if DEBUG:
        print("[DEBUG] main_beta.py: Trainer fit completed.")
    # At the end of main(), after trainer.fit():
    if args.run_eval:
        print("Running evaluation on test set...")
        test_results = trainer.test(ckpt_path="best", dataloaders=data_module)
        print(f"Test results: {test_results}")

    # test_results_IID = trainer.test(ckpt_path='last', dataloaders = cv_data_module_IID.test_dataloader())
    # test_results_OOD = trainer.test(ckpt_path='last', dataloaders = cv_data_module_OOD.test_dataloader())


if __name__ == "__main__":
    # <<< Comment out or control stdout redirection for DEBUG >>>
    # sys.stdout = open('Hybrid_SDE_output_beta.txt', 'w')

    parser = argparse.ArgumentParser(description="Train a model on CV dataset")
    # <<< Add debug argument >>>
    parser.add_argument(
        "--debug", action="store_true", help="Enable debug print statements"
    )
    # Logging specific args
    parser.add_argument(
        "--HPC_work",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
        help="Where to save if HPC",
    )
    parser.add_argument(
        "--seed", type=int, default=64, help="Random seed for initialization"
    )
    parser.add_argument(
        "--project_name",
        type=str,
        default="sdehybrid_rc_2",
        help="Wandb project name",
    )
    parser.add_argument(
        "--log_wandb",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
        help="Whether to log to Weights & Biases",
    )
    parser.add_argument(
        "--early_stopping",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
        help="Enable early stopping",
    )
    parser.add_argument(
        "--model_checkpoint",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
        help="Enable model checkpointing",
    )
    parser.add_argument(
        "--plot_every", type=int, default=10, help="Plot every how many global steps? "
    )

    # Data specific args
    parser.add_argument(
        "--dataset_type",
        type=str,
        default="mimic",
        choices=["synthetic", "mimic"],
        help="Which dataset to use.",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="/n/netscratch/mzitnik_lab/Lab/rconci/BIOMM/processed_data",
        help="Root directory for MIMIC preprocessed data.",
    )
    parser.add_argument(
        "--icu_stays_path",
        type=str,
        default="/n/netscratch/mzitnik_lab/Lab/rconci/BIOMM/input_data/ICUSTAYS.csv",
        help="Path to icustays.csv file.",
    )
    parser.add_argument(
        "--static_hidden_dim",
        type=int,
        default=64,
        help="Hidden dimension for the static encoder MLP.",
    )
    parser.add_argument(
        "--fusion_hidden_dim",
        type=int,
        default=128,
        help="Hidden dimension for the fusion MLP.",
    )
    parser.add_argument(
        "--normalise",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="Whether to normalise the data. Recommended ONLY if using an Encoder",
    )
    parser.add_argument(
        "--noise_std",
        type=float,
        default=0.0,
        help="Noise defines how noisy the data is ",
    )
    parser.add_argument(
        "--non_confounded_effect",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="Whether to add non-confounded unsee effect on the treatment (increases the noise of the prediction)",
    )
    parser.add_argument(
        "--fixed_tx",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
        help="Whether all patients receive the same treatment ",
    )
    parser.add_argument(
        "--include_all_inputs",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
        help="Whether to create data with all variables in the X as input",
    )

    # PRIMARY Bifurcation args
    parser.add_argument(
        "--gamma",
        type=int,
        default=6,
        help="Gamma defines how confounded the data is. the higher, the less overlap. the lower the more overlap",
    )
    parser.add_argument(
        "--confounder_type",
        type=str,
        default="partial_hard",
        choices=["visible", "partial", "partial_hard", "invisible"],
        help="the type of confounding present",
    )
    parser.add_argument(
        "--use_encoder",
        type=str,
        default="raindrop",
        choices=["full", "partial", "none", "raindrop"],
        help="what to do with the encoder!",
    )

    parser.add_argument(
        "--SDEnet_hidden_dim", 
        type=int, default=512, 
        help="Hidden dim for SDE NN  "
    )
    parser.add_argument(
        "--SDEnet_depth", type=int, default=6, help="Num layeres for SDE NN  "
    )
    parser.add_argument(
        "--use_batch_norm",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="Whether to include batch norm within the SDE NN network )",
    )
    parser.add_argument(
        "--include_time",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
        help="Whether to include encoded time in the SDE NN inputs)",
    )

    parser.add_argument(
        "--integration_step_size",
        type=float,
        default=2.0,
        help="Parameter dt for SDE integration",
    )
    parser.add_argument(
        "--integration_method", type=str, default="euler", help="SDE integration method"
    )
    parser.add_argument(
        "--integration_rtol", type=float, default=1e-3, help="SDE integration rtol"
    )
    parser.add_argument(
        "--integration_atol", type=float, default=1e-3, help="SDE integration atol"
    )
    parser.add_argument(
        "--integration_adaptive",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="Use adaptive SDE integration?",
    )

    parser.add_argument(
        "--prior_tx_sigma",
        type=float,
        default=0.00001,
        help="prior_tx_sigma defines our assumed prior noise of the stochastic control ",
    )
    parser.add_argument(
        "--self_reverting_prior_control",
        action="store_true",
        help="Whether the control has a self reverting prior to it with a functional prior",
    )
    parser.add_argument(
        "--KL_weighting_SDE",
        type=float,
        default=0.0001,
        help="Defines the weighting to the KL loss for the SDE",
    )

    parser.add_argument(
        "--use_2_5std_encoder_minmax",
        type=bool,
        default=False,
        help="pushes the outputs of the encoder into a narrower range. BUT will mean some are NOT reached appropriately. ",
    )
    parser.add_argument(
        "--encoder_SDENN_dims",
        type=int,
        default=128,
        help="Encoder output used by SDENN",
    )

    # Default args _not be changed_
    parser.add_argument(
        "--num_samples",
        type=int,
        default=3,
        help="Number of SDE samples- is affected if sigma >0 ",
    )
    parser.add_argument(
        "--prior_tx_mu",
        type=float,
        default=0.01,
        help="prior_tx_mu defines our assumed prior Dt_iexternal of the stochastic control ",
    )
    parser.add_argument(
        "--theta",
        type=float,
        default=0.0001,
        help="Theta defines how the impact of the mean reverting process correction on the SDE",
    )
    parser.add_argument(
        "--SDE_control_weighting",
        type=float,
        default=1.5,
        help="how much to scale the output of the SDE NN",
    )
    # Smoothness/inductive bias flags
    parser.add_argument(
        "--use_control_lowpass",
        action="store_true",
        help="Use low-pass control dynamics du/dt=(u_hat-u)/tau",
    )
    parser.add_argument(
        "--control_lowpass_tau",
        type=float,
        default=30.0,
        help="Time constant tau (seconds) for control low-pass",
    )
    parser.add_argument(
        "--use_control_tv_loss",
        action="store_true",
        help="Add TV/L2 smoothness penalty on control path",
    )
    parser.add_argument(
        "--control_tv_weight",
        type=float,
        default=1e-3,
        help="Weight for control TV/L2 smoothness penalty",
    )
    parser.add_argument(
        "--override_control_scales",
        type=str,
        default="",
        help="Comma-separated per-head control scales to override defaults (e.g., '0.1,0.02,0.01,0.01')",
    )
    parser.add_argument(
        "--control_energy_weight",
        type=float,
        default=1e-6,
        help="Weight for control energy regularizer added to total loss",
    )

    parser.add_argument(
        "--force_no_controls",
        action="store_true",
        help="Whether to force the SDE to not use controls",
    )

    parser.add_argument(
        "--plot_outputs_train",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
        help="Whether to plot the outputs of the SDE during training",
    )

    # Model specific args
    parser.add_argument(
        "--start_dec_at_treatment",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
        help="Whether to encode the data until treatment and the decode or decode from the beginning!)",
    )
    parser.add_argument(
        "--variational_encoder",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="Whether encoder is variational or not - not finished variational)",
    )
    parser.add_argument(
        "--encoder_hidden_dim",
        type=int,
        default=128,
        help="Output of the encoder into a latent space. This needs to match the total SDE input dims ",
    )
    parser.add_argument(
        "--encoder_num_layers",
        type=int,
        default=2,
        help="Number of layers in encoder GRU",
    )
    parser.add_argument(
        "--encoder_w_time",
        type=bool,
        default=False,
        help="Whether encoder includes time in its inputs)",
    )
    parser.add_argument(
        "--encoder_reverse_time",
        type=bool,
        default=False,
        help="Whether encoder runs with inputs backwards in time)",
    )
    parser.add_argument(
        "--variational_sampling",
        type=bool,
        default=False,
        help="If NOT using encoder, to learn a variational q distribution for the unobserved dims)",
    )

    parser.add_argument(
        "--final_activation",
        type=str,
        default="tanh",
        choices=["relu", "none", "tanh"],
        help="Which nonlinearity to add as a final layer to the NN!",
    )
    parser.add_argument(
        "--normalise_for_SDENN",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
        help="Whether to normalise data when handing it to the SDE NN or just scale it )",
    )
    parser.add_argument(
        "--SDEnet_out_dims", 
        type=int, 
        default=4, 
        help="Num output dims for SDE NN  "
    )
    parser.add_argument(
        "--med_embed_dim", 
        type=int, 
        default=32, 
        help="Num output dims for med embedding"
    )
    parser.add_argument(
        "--output_scale",
        type=float,
        default=2,
        help="Standard Deviation when computing GaussianNegLL between Y_true and Y_hat",
    )
    parser.add_argument(
        "--SDE_input_state",
        type=str,
        default="full",
        choices=["full", "partial"],
        help="which dims to include in the SDE NN - always do full!",
    )

    # Solver args
    parser.add_argument("--adjoint", type=bool, default=False, const=True, nargs="?")
    # parser.add_argument('--adaptive', type=bool, default=False, const=True, nargs="?")
    # parser.add_argument('--method', type=str, default='euler', choices=('euler', 'milstein', 'srk'), help='Name of numerical solver.')
    # parser.add_argument('--dt', type=float, default=1e-2)
    # parser.add_argument('--rtol', type=float, default=1e-3)
    # parser.add_argument('--atol', type=float, default=1e-3)

    # Training specific args
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="Learning rate for the optimizer",
    )
    parser.add_argument("--batch_size", 
        type=int, default=32,
        help="Training batch size")
    
    parser.add_argument(
        "--max_epochs", type=int, default=40, help="Maximum number of epochs to train"
    )
    parser.add_argument(
        "--accelerator",
        type=str,
        default="auto",
        choices=["gpu", "mps", "cpu", "auto"],
        help="Which accelerator to use",
    )
    parser.add_argument(
        "--precision",
        type=str,
        default="32-true",
        choices=["32-true", "16-mixed", "bf16-mixed", "64-true"],
        help="Lightning precision (use 16-mixed or bf16-mixed for speed)",
    )
    parser.add_argument(
        "--log_every_n_steps",
        type=int,
        default=4,
        help="Lightning logging frequency in steps",
    )
    parser.add_argument(
        "--redirect_output",
        action="store_true",
        help="Redirect stdout/stderr to a log file instead of terminal",
    )
    parser.add_argument(
        "--log_file",
        type=str,
        default="",
        help="Optional explicit path to log file (default: train_<timestamp>.log under --train_dir if set)",
    )
    
    parser.add_argument(
        "--max_samples",
        type=str,
        default=None,
        help="Max dataset length (None for production)",
    )
    parser.add_argument(
        "--run_eval", 
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
        help="Run evaluation after training"
    )
    parser.add_argument(
        "--early_stopping_patience", 
        type=int, 
        default=10,
        help="Early stopping patience"
    )
    parser.add_argument(
        "--test_zenker",
        type=bool,
        default=False,
        help="Run the pure Zenker baseline as comparison?",
    )

    # Controller selection
    parser.add_argument(
        "--controller_type",
        type=str,
        default="mlp",
        choices=["mlp", "gat"],
        help="Controller type: standard MLP or GAT-based",
    )
    parser.add_argument(
        "--gat_heads", type=int, default=4, help="Number of attention heads in GAT"
    )
    parser.add_argument(
        "--gat_layers", type=int, default=2, help="Number of GAT layers"
    )
    parser.add_argument(
        "--gat_hidden", type=int, default=128, help="Hidden size per GAT layer"
    )
    parser.add_argument(
        "--gat_dropout", type=float, default=0.0, help="Dropout in GAT layers"
    )

    # Overfitting/debug controls
    parser.add_argument(
        "--overfit_batches",
        type=float,
        default=0.0,
        help="Lightning overfit_batches setting: 0.0=disabled, 0.01=1%, 1=single batch, 10=ten batches",
    )
    parser.add_argument(
        "--disable_controls",
        type=str,
        default="",
        help="Comma-separated zero-based indices of control heads to disable (e.g., '0,3').",
    )
    parser.add_argument(
        "--train_dir",
        type=str,
        default="",
        help="Override output directory for plots and logs (default: results/<unique_dir>).",
    )
    parser.add_argument(
        "--disable_sanity_check",
        action="store_true",
        help="Disable sanity validation to start training immediately (sets num_sanity_val_steps=0).",
    )
    parser.add_argument(
        "--audit_one_batch",
        action="store_true",
        help="Run a one-batch audit of dataloader and model inputs then exit",
    )
    parser.add_argument(
        "--audit_print_samples",
        type=int,
        default=3,
        help="How many batch elements to print during audit",
    )

    args = parser.parse_args()
    if args.audit_one_batch and args.dataset_type == "mimic":
        # Lightweight in-place audit without training
        dm = MIMICDataModule(
            data_root=args.data_root,
            icu_stays_path=args.icu_stays_path,
            batch_size=args.batch_size,
            num_workers=0,
            random_state=args.seed,
            max_samples=args.max_samples,
            use_raindrop_context=True,
            expert_latent_dim=14,
        )
        dm.setup("fit")
        loader = dm.train_dataloader()
        batch = next(iter(loader))
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
            med_values,
            med_mask,
            med_time,
            med_tensors,
            hadm_ids,
            traj_ids,
        ) = batch
        print("=== One-batch audit (MIMIC) ===")
        def stat(name, t):
            t = t.detach() if hasattr(t, "detach") else t
            print(f"{name:>16s}: shape={tuple(t.shape)}, NaN={(~torch.isfinite(t)).sum().item()}, min={torch.nanmin(t.float()) if t.dtype!=torch.bool else 'NA'}, max={torch.nanmax(t.float()) if t.dtype!=torch.bool else 'NA'}")
        stat("rd_src", rd_src)
        stat("rd_times", rd_times)
        stat("rd_length", rd_length)
        stat("static", static_features)
        stat("init_state", init_states)
        stat("ic_mask", ic_mask)
        stat("Y", Y)
        stat("Y_mask", Y_mask)
        stat("t_Y", t_Y)
        stat("med_values", med_values)
        stat("med_mask", med_mask)
        stat("med_time", med_time)
        stat("med_tensors", med_tensors)
        print("IDs:", hadm_ids[: args.audit_print_samples].tolist(), traj_ids[: args.audit_print_samples].tolist())
        # Monotonicity checks
        diffs = t_Y[:, 1:] - t_Y[:, :-1]
        nonmono = (diffs <= 0).any(dim=1)
        print("Non-monotonic t_Y samples idx:", torch.where(nonmono)[0].tolist())
        # Mask validity
        mask_bad = ((Y_mask < 0) | (Y_mask > 1)).any()
        print("Y_mask outside [0,1]?:", bool(mask_bad))
        # Basic consistency lengths
        print("rd_length min/max:", int(rd_length.min()), int(rd_length.max()))
        print("Audit complete. Exiting.")
    else:
        main(args)
