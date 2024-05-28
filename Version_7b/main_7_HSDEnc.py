import os
import torch
import numpy as np
import random
import argparse
import wandb
import sys
import tempfile


import lightning as L
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.profilers import SimpleProfiler, AdvancedProfiler


from CV_data_7b import create_load_save_data, CVDataModule_IID, CVDataModule_OOD

from model_7_HSDEnc import Hybrid_VAE_SDE_Encoder




def set_seed(seed):
    seed_everything(seed, workers=True)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multi-GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


    
def main(args):
    print('CUDA GPUs present?', torch.cuda.is_available())
    if args.HPC_work:
        torch.set_float32_matmul_precision('medium')  # Faster computations with less precision
        
    saving_dir = os.getcwd()
   

    set_seed(args.seed)

    if args.log_wandb:
        wandb_logger = WandbLogger(project=args.project_name, 
                                   log_model=False, 
                                   save_dir = os.path.join(saving_dir, 'model_logs'))
        wandb_logger.log_hyperparams(args)
    else:
        wandb_logger = None

    dataset_params = {
        'include_all_inputs':True, 
        'gamma': args.gamma,
        'sigma_tx': 2,
        'confounder_type': args.confounder_type,

        'non_confounded_effect': False,
        'noise_std': 0.0,
        't_span': 60,
        't_treatment': 45,
        't_cutoff':40,
        'seed': 1,
        'pre_treatment_dims': [0, 1], # = pa + pv
        'post_treatment_dims': [0], # = pa 
        'normalize': False,
        'N': 1280
    }

    filename_parts = [
        f"sd={args.seed}",
        f"gm={args.gamma}",
        f"cnf={args.confounder_type}",
        f"txsig={args.prior_path_sigma}",
        f"revert={args.self_reverting_prior_control}",
        f"klw={args.KL_weighting_SDE}",
        f"SDEhd={args.SDEnet_hidden_dim}"
    ]
    unique_dir_name = "_".join(filename_parts)

    print('dataset_params',dataset_params)
    data_path = os.path.join(saving_dir, 'data_created')

    dataset_params['r_tpr_mod'] = 0.0
    train_val_data = create_load_save_data(dataset_params, data_path)
    dataset_params['r_tpr_mod'] = -0.5 
    test_data = create_load_save_data(dataset_params, data_path)
    
    cv_data_module_IID = CVDataModule_IID(train_val_data = train_val_data, batch_size=args.batch_size, num_workers = 4)
    cv_data_module_OOD = CVDataModule_OOD(OOD_test_data = test_data, batch_size=128, num_workers = 4)
    

    model = Hybrid_VAE_SDE_Encoder(expert_latent_dims=cv_data_module_IID.expert_latent_dim,
                                   path_control_dim=args.path_control_dim,
                                   apply_path_SDE=args.apply_path_SDE,
                                   prior_path_sigma=args.prior_path_sigma,
                                   num_samples=args.num_samples,
                                   self_reverting_prior_control=args.self_reverting_prior_control,
                                   theta=args.theta,
                                   SDE_control_weighting=args.SDE_control_weighting,
                                   normalise_for_SDENN=args.normalise_for_SDENN,
                                   include_time=args.include_time,
                                   SDEnet_hidden_dim=args.SDEnet_hidden_dim,
                                   SDEnet_depth=args.SDEnet_depth,
                                   final_activation=args.final_activation,
                                   KL_weighting_SDE=args.KL_weighting_SDE,
                                   l1_lambda=args.l1_lambda,
                                   log_lik_output_scale=args.log_lik_output_scale,
                                   normalised_data=dataset_params['normalize'],
                                   train_dir=os.path.join(saving_dir, 'figures', unique_dir_name),
                                   learning_rate=args.learning_rate,
                                   log_wandb=args.log_wandb,
                                   adjoint=args.adjoint,
                                   plot_every=args.plot_every
                                   )

    callbacks = []


    if args.model_checkpoint:
        checkpoint_callback = ModelCheckpoint(
            monitor='val_total_loss',        # Ensure this is the exact name used in your logging
            dirpath= os.path.join(saving_dir, 'model_checkpoints', unique_dir_name),  # Directory to save checkpoints
            filename=f'best-{{epoch:02d}}-{{val_loss:.2f}}-{unique_dir_name}',
            save_top_k=1,
            mode='min',                     # Minimize the monitored value
            save_last=True,                # Save the last model to resume training
            verbose = True
        )
        callbacks.append(checkpoint_callback)

    if args.early_stopping:
        early_stopping = EarlyStopping(
            min_delta=0.00,
            monitor='val_total_loss',        # Ensure this is the exact name used in your logging
            patience=100,                    # num epochs with a val loss not improving before it stops 
            mode='min',                     # Minimize the monitored value
            verbose=True
        )
        callbacks.append(early_stopping)

    trainer = Trainer(
        max_epochs=args.max_epochs,
        accelerator=args.accelerator,
        logger=wandb_logger,
        log_every_n_steps=6,
        callbacks=callbacks,
        #fast_dev_run = True,
        #overfit_batches = 1
        #deterministic=True,
        #check_val_every_n_epoch=1,  
        #profiler="simple"   #this helps to identify bottlenecks 
    )
    trainer.fit(model, cv_data_module_IID)
    
    #test_results_IID = trainer.test(ckpt_path='last', dataloaders = cv_data_module_IID.test_dataloader())
    #test_results_OOD = trainer.test(ckpt_path='last', dataloaders = cv_data_module_OOD.test_dataloader())

if __name__ == '__main__':
    sys.stdout = open('Hybrid_encoder', 'w')

    parser = argparse.ArgumentParser(description="Train a model on CV dataset")
    # Logging specific args 
    parser.add_argument('--HPC_work', type=bool, default=False, help='Where to save if HPC')
    parser.add_argument('--seed', type=int, default=44, help='Random seed for initialization')
    parser.add_argument('--project_name', type=str, default='YAY_sdehybrid_2', help='Wandb project name')
    parser.add_argument('--log_wandb', type=bool, default=False, help='Whether to log to Weights & Biases')
    parser.add_argument('--early_stopping', type=bool, default=False, help='Enable early stopping')
    parser.add_argument('--model_checkpoint', type=bool, default=False, help='Enable model checkpointing')
    parser.add_argument('--plot_every', type=int, default=70, help='Plot every how many global steps? ')

    # Data specific args
    parser.add_argument('--normalise', type=bool, default=False, help='Whether to normalise the data. Recommended ONLY if using an Encoder')
    parser.add_argument('--noise_std', type=float, default=0.0, help='Noise defines how noisy the data is ')
    parser.add_argument('--non_confounded_effect', type=bool, default=False, help='Whether to add non-confounded unsee effect on the treatment (increases the noise of the prediction)')
    parser.add_argument('--gamma', type=int, default=10, help='Gamma defines how confounded the data is. the higher, the less overlap. the lower the more overlap')
    parser.add_argument('--confounder_type', type=str, default='partial', choices=['visible', 'partial', 'invisible'], help='the type of confounding present')

    #PRIMARY Bifurcation args
    # Specified model parameters
    parser.add_argument('--path_control_dim', type=int, default=5, help='Dimensionality of the path control variable')
    parser.add_argument('--apply_path_SDE', type=bool, default=False, help='Whether to apply the path-based SDE formulation')
    
    parser.add_argument('--prior_path_sigma', type=float, default=0.00, help='Standard deviation of the prior path process')
    parser.add_argument('--num_samples', type=int, default=1, help='Number of samples for the SDE process')
    parser.add_argument('--self_reverting_prior_control', type=bool, default=False, help='Whether the control has a self-reverting prior')
    parser.add_argument('--theta', type=float, default=0.01, help='Theta for the mean-reverting process')
    parser.add_argument('--SDE_control_weighting', type=float, default=1.0, help='Weighting factor for the SDE control term')
    
    parser.add_argument('--normalise_for_SDENN', type=bool, default=True, help='Whether to normalise inputs for the SDENN')
    parser.add_argument('--include_time', type=bool, default=True, help='Whether to include time in model inputs')
    parser.add_argument('--SDEnet_hidden_dim', type=int, default=400, help='Hidden dimensions for SDE network')
    parser.add_argument('--SDEnet_depth', type=int, default=6, help='Depth of layers in the SDE network')
    parser.add_argument('--final_activation', type=str, default='none', choices=['relu', 'none'], help='Final activation function for the SDE network')
    
    parser.add_argument('--KL_weighting_SDE', type=float, default=0.001, help='Weighting for the KL divergence in SDE terms')
    parser.add_argument('--l1_lambda', type=float, default=0.1, help='Lambda for L1 regularization')
    parser.add_argument('--log_lik_output_scale', type=float, default=0.01, help='Output scale for log-likelihood calculation')


    # Training specific args
    parser.add_argument('--adjoint', type=bool, default=False, help='Whether to use adjoint method for backpropagation')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for the optimizer')
    parser.add_argument('--batch_size', type=int, default=128, help='Training batch size')
    parser.add_argument('--max_epochs', type=int, default=200, help='Maximum number of epochs to train')
    parser.add_argument('--accelerator', type=str, default='auto', choices=['gpu', 'mps', 'cpu', 'auto'], help='Which accelerator to use')

    
    args = parser.parse_args()
    main(args)
