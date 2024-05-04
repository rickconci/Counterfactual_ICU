import os
import torch
import numpy as np
import random
import argparse
import wandb

import lightning as L
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.profilers import SimpleProfiler, AdvancedProfiler

from CV_data_4 import CVDataModule
from models_4 import SDE_VAE_Lightning



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
    print(torch.cuda.is_available())
    if args.HPC_work:
        saving_dir = r'/home/rc667/rds/hpc-work/CF_ICU'
        torch.set_float32_matmul_precision('medium')  # Faster computations with less precision

    else:
        saving_dir = r'.'


    set_seed(args.seed)

    if args.log_wandb:
        wandb_logger = WandbLogger(project=args.project_name, log_model='all', save_dir = os.path.join(saving_dir, 'model_logs'))
    else:
        wandb_logger = None

    cv_data_module = CVDataModule(batch_size=args.batch_size, seed=args.seed, N_ts=6400,
                                  gamma=args.gamma, noise_std=args.noise_std, t_span=30, t_treatment=15)
    cv_data_module.prepare_data()
    post_tx_ode_len = cv_data_module.post_tx_ode_len
    Tx_dim = cv_data_module.Tx_dim
    obs_dim = cv_data_module.X.shape[1]
    input_dim = obs_dim  

    # -> adjust CV data module to output Tx with time 
    # -> adjust encoder to take in Obs + Tx w time

    if args.latent_model == 'expert':
        #if expert only, the latent dim need to match those of the expert ODEs
        encoder_output_dim = cv_data_module.expert_ODE_size
    elif args.latent_model == 'hybrid_SDE':
         #if hybrid only, the latent dim need to match those of the expert ODEs + extra SDE dims 
         encoder_output_dim = cv_data_module.expert_ODE_size + args.latent_dim
    elif args.latent_model == 'SDE':
         #if SDE only, the latent dim can be any size 
        encoder_output_dim = args.latent_dim
    

    ode_vae_model = SDE_VAE_Lightning(encoder_model = args.encoder_model, 
                                      latent_type = args.latent_type, 
                                      expert_ODE_size =cv_data_module.expert_ODE_size , 
                                      #models vars
                                      input_dim = input_dim, 
                                      hidden_dim = args.hidden_dim, 
                                      latent_dim = encoder_output_dim, 
                                      output_dim = 2,
                                      Tx_dim = Tx_dim, 
                                      dropout_p = args.dropout_p,
                                      #SDE vars
                                      num_samples = args.num_samples, 
                                      theta = args.theta, 
                                      mu = args.mu, 
                                      sigma = args.sigma_sde, 
                                      output_scale = args.output_scale, 
                                      #output function vars 
                                      use_whole_trajectory = args.use_whole_traj, 
                                      post_tx_ode_len = post_tx_ode_len, 
                                      # loss vars
                                      KL_weighting_SDE = args.KL_weighting_SDE, 
                                      KL_weighting_var_encoder = args.KL_weighting_var_encoder,  
                                      learning_rate = args.learning_rate, 
                                      log_wandb = args.log_wandb) 

    callbacks = []

    if args.model_checkpoint:
        checkpoint_callback = ModelCheckpoint(
            monitor='val_SDE_loss',        # Ensure this is the exact name used in your logging
            dirpath= os.path.join(saving_dir, 'model_checkpoints'),  # Directory to save checkpoints
            filename='best-checkpoint-{epoch:02d}-{val_SDE_loss:.2f}',
            save_top_k=1,                   # Save only the top 1 model
            mode='min',                     # Minimize the monitored value
            save_last=True                  # Save the last model to resume training
        )
        callbacks.append(checkpoint_callback)

    if args.early_stopping:
        early_stopping = EarlyStopping(
            monitor='val_SDE_loss',        # Ensure this is the exact name used in your logging
            patience=200,                    # Patience of 10 epochs
            mode='min',                     # Minimize the monitored value
            verbose=True
        )
        callbacks.append(early_stopping)

    trainer = Trainer(
        max_epochs=args.max_epochs,
        accelerator=args.accelerator,
        devices= 'auto',
        deterministic=True,
        logger=wandb_logger,
        log_every_n_steps=50,
        check_val_every_n_epoch=10,   #this is so the model isn't clogged up by loggers 
        callbacks=callbacks,  # Add only initialized callbacks
        profiler="simple"   #this helps to identify bottlenecks 

    )
    
    trainer.fit(ode_vae_model, cv_data_module)
    
    trainer.test(ckpt_path='best', dataloaders = cv_data_module.test_dataloader())


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train a model on CV dataset")
    # Logging specific args 
    parser.add_argument('--HPC_work', action='store_true', help='HPC run or not')
    parser.add_argument('--seed', type=int, default=1234, help='Random seed for initialization')
    parser.add_argument('--project_name', type=str, default='SDE_CV', help='Wandb project name')
    parser.add_argument('--log_wandb', type=bool, default=False, help='Whether to log to Weights & Biases')
    parser.add_argument('--early_stopping', type=bool, default=False, help='Enable early stopping')
    parser.add_argument('--model_checkpoint', type=bool, default=False, help='Enable model checkpointing')

    # Data specific args
    parser.add_argument('--gamma', type=float, default=10, help='Gamma defines how confounded the data is. the higher, the less overlap. the lower the more overlap')
    parser.add_argument('--noise_std', type=float, default=0.005, help='Noise defines how noisy the data is ')

    # Model specific args
    parser.add_argument('--encoder_model', type=str, default='LSTM', choices=['LSTM', 'GRU', 'RNN'], help='the type of encoder model')
    parser.add_argument('--latent_type', type=str, default='SDE', choices=['expert', 'hybrid_SDE', 'SDE'], help='the type of model')
    parser.add_argument('--latent_dim', type=int, default=64, help='Latent dimension for the CDE')
    parser.add_argument('--hidden_dim', type=int, default=146, help='Hidden dimension across models')
    parser.add_argument('--use_whole_traj', type=bool, default=False, help='Whether to use the whole trajectory to convert from latent to observed or pointwise')
    parser.add_argument('--dropout_p', type=float, default=0.0, help='Drop out in MLP models ')

    ## SDE specific argrs
    parser.add_argument('--num_samples', type=int, default=5, help='Number of SDE samples  ')
    parser.add_argument("--output_scale",type=float,default = 0.01, help = "Standard Deviation when computing GaussianNegLL between Y_true and Y_hat")
    parser.add_argument('--theta', type=float, default=0.1, help='Theta defines how the impact of the mean reverting process correction on the CDE')
    parser.add_argument('--mu', type=float, default=0, help='Mu defines where the mean of the mean reverting process is ')
    parser.add_argument("--sigma_sde",type=float,default = 0.1, help = "Diffusion parameter in the SDE prior")

    # Training specific args
    parser.add_argument('--KL_weighting_SDE', type=float, default=0.001, help='Defines the weighting to the KL loss for the SDE')
    parser.add_argument('--KL_weighting_var_encoder', type=float, default=0.001, help='Defines the weighting to the KL loss for the variational encoder')

    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for the optimizer')
    parser.add_argument('--batch_size', type=int, default=128, help='Training batch size')
    parser.add_argument('--max_epochs', type=int, default=300, help='Maximum number of epochs to train')
    parser.add_argument('--accelerator', type=str, default='auto', choices=['gpu', 'mps', 'cpu', 'auto'], help='Which accelerator to use')
    #parser.add_argument('--devices', type=str, default='auto', choices=['gpu', 'mps', 'cpu', 'auto'], help='Which number of devices to use')

    
    args = parser.parse_args()
    main(args)
