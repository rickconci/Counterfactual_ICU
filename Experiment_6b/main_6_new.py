import os
import torch
import numpy as np
import random
import argparse
import wandb
import sys

import lightning as L
from lightning.pytorch import Trainer, seed_everything
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.profilers import SimpleProfiler, AdvancedProfiler

#from CV_data_6_new import create_load_save_data, CVDataModule_final
from CV_data_6 import create_load_save_data, CVDataModule_final

from model_6_new_nonvar_2 import Hybrid_VAE_SDE
from utils_6 import process_input




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
        saving_dir = r'/home/rc667/rds/hpc-work/CF_ICU'
        torch.set_float32_matmul_precision('medium')  # Faster computations with less precision

    else:
        saving_dir = r'.'


    set_seed(args.seed)

    if args.log_wandb:
        wandb_logger = WandbLogger(project=args.project_name, 
                                   log_model=False, 
                                   save_dir = os.path.join(saving_dir, 'model_logs'))
    else:
        wandb_logger = None

    dataset_params = {
        'N': 1000,
        'gamma': 0,
        'noise_std': 0.0,
        'sigma_tx': 4,
        'confounder_type': 'partial',
        'non_confounded_effect': False,
        't_span': 26,
        't_treatment': 15,
        'seed': 1,
        'pre_treatment_dims': [0, 1],
        'post_treatment_dims': [1],
        'normalize': False
    }


    print('dataset_params',dataset_params)

    data_path = '/Users/riccardoconci/Local_documents/ACS submissions/THESIS/Counterfactual_ICU/Experiment_6b/data_created'

    data = create_load_save_data(dataset_params, data_path)
    cv_data_module = CVDataModule_final(data,  batch_size=args.batch_size, num_workers=9)


    model = Hybrid_VAE_SDE(use_encoder = args.use_encoder,
                           start_dec_at_treatment = args.start_dec_at_treatment, 
                           variational_sampling = args.variational_sampling,

                           #Encoder
                           encoder_input_dim =  len(dataset_params['pre_treatment_dims']), 
                           encoder_hidden_dim = args.encoder_hidden_dim,
                           encoder_output_dim = args.expert_latent_dims + args.encoder_SDENN_dims,

                           encoder_num_layers = args.encoder_num_layers,
                           variational_encoder = args.variational_encoder,
                           encoder_w_time = args.encoder_w_time,
                           encoder_reverse_time = args.encoder_reverse_time,

                           #SDE params
                           expert_latent_dims  = args.expert_latent_dims,
                           num_samples = args.num_samples,
                           normalise_for_SDENN = args.normalise_for_SDENN,
                           self_reverting_prior_control = args.self_reverting_prior_control,
                           prior_tx_sigma = args.prior_tx_sigma,
                           prior_tx_mu = args.prior_tx_mu,
                           theta = args.theta,
                           SDE_control_weighting = args.SDE_control_weighting,
                           
                           #SDE model params
                           SDE_input_state = args.SDE_input_state, 
                           include_time = args.include_time, 
                           SDEnet_hidden_dim = args.SDEnet_hidden_dim, 
                           SDEnet_depth = args.SDEnet_depth,
                           SDEnet_out_dims = args.SDEnet_out_dims, 

                           #decoder params
                           decoder_output_dims = dataset_params['post_treatment_dims'],
                           normalised_data = dataset_params['normalize'],
                           log_lik_output_scale = args.output_scale,

                           #admin
                           train_dir = os.path.join(saving_dir, 'figures'), 
                           KL_weighting_SDE = args.KL_weighting_SDE, 
                           learning_rate = args.learning_rate,
                           log_wandb = args.log_wandb,
                           adjoint = args.adjoint,
                           plot_every = args.plot_every

    )


    callbacks = []

    if args.model_checkpoint:
        checkpoint_callback = ModelCheckpoint(
            monitor='val_total_loss',        # Ensure this is the exact name used in your logging
            dirpath= os.path.join(saving_dir, 'model_checkpoints'),  # Directory to save checkpoints
            filename='best-checkpoint-{epoch:02d}-{val_loss:.2f}',
            mode='min',                     # Minimize the monitored value
            save_last=True                  # Save the last model to resume training
        )
        callbacks.append(checkpoint_callback)

    if args.early_stopping:
        early_stopping = EarlyStopping(
            min_delta=1.0,
            monitor='val_total_loss',        # Ensure this is the exact name used in your logging
            patience=10,                    # num epochs with a val loss not improving before it stops 
            mode='min',                     # Minimize the monitored value
            verbose=True
        )
        callbacks.append(early_stopping)

    trainer = Trainer(
        max_epochs=args.max_epochs,
        accelerator=args.accelerator,
        logger=wandb_logger,
        log_every_n_steps=5,
        callbacks=callbacks
        #deterministic=True,
        #check_val_every_n_epoch=1,  
        #devices= 'auto',
        #profiler="simple"   #this helps to identify bottlenecks 
    )
    trainer.fit(model, cv_data_module)
    
    #trainer.test(ckpt_path='best', dataloaders = cv_data_module.test_dataloader())
    test_results = trainer.test(ckpt_path='best', dataloaders = cv_data_module.test_dataloader())

if __name__ == '__main__':
    #sys.stdout = open('Hybrid_SDE_output.txt', 'w')

    parser = argparse.ArgumentParser(description="Train a model on CV dataset")
    # Logging specific args 
    parser.add_argument('--HPC_work', action='store_true', help='HPC run or not')
    parser.add_argument('--seed', type=int, default=44, help='Random seed for initialization')
    parser.add_argument('--project_name', type=str, default='maybe_YAY_sdehybrid', help='Wandb project name')
    parser.add_argument('--log_wandb', type=bool, default=True, help='Whether to log to Weights & Biases')
    parser.add_argument('--early_stopping', type=bool, default=True, help='Enable early stopping')
    parser.add_argument('--model_checkpoint', type=bool, default=True, help='Enable model checkpointing')
    parser.add_argument('--plot_every', type=int, default=21, help='Plot every how many global steps? ')

    # Data specific args
    parser.add_argument('--gamma', type=float, default=10, help='Gamma defines how confounded the data is. the higher, the less overlap. the lower the more overlap')
    parser.add_argument('--confounder_type', type=str, default='partial', choices=['visible', 'partial', 'invisible'], help='the type of confounding present')
    parser.add_argument('--normalise', type=bool, default=False, help='Whether to normalise the data. Recommended ONLY if using an Encoder')
    parser.add_argument('--noise_std', type=float, default=0.0, help='Noise defines how noisy the data is ')
    parser.add_argument('--non_confounded_effect', type=bool, default=False, help='Whether to add non-confounded unsee effect on the treatment (increases the noise of the prediction)')

    #PRIMARY Bifurcation args
    parser.add_argument('--use_encoder', type=str, default='none', choices=['full', 'partial', 'none'], help='what to do with the encoder!')
    parser.add_argument('--normalise_for_SDENN', type=bool, default=False, help='Whether to normalise data when handing it to the SDE NN or just scale it )')

    parser.add_argument('--prior_tx_sigma', type=float, default=0.0, help='prior_tx_sigma defines our assumed prior noise of the stochastic control ')
    parser.add_argument('--self_reverting_prior_control', type=bool, default=False, help='Whether the control has a self reverting prior to it with a mu of 0)')
    parser.add_argument('--theta', type=float, default=0.01, help='Theta defines how the impact of the mean reverting process correction on the SDE')
    parser.add_argument('--prior_tx_mu', type=float, default=0.0, help='prior_tx_mu defines our assumed prior Dt_iexternal of the stochastic control ')
    parser.add_argument('--SDE_control_weighting', type=float, default=1, help='how much to scale the output of the SDE NN')

    #SECONDARY bifurcation argrs
    parser.add_argument('--encoder_SDENN_dims', type=int, default=4, help='Encoder output used by SDENN')
    parser.add_argument('--include_time', type=bool, default=True, help='Whether to include encoded time in the SDE NN inputs)')

    #Default args _not be changed_
    parser.add_argument('--expert_latent_dims', type=int, default=4, help='Encoder output used by expert model.. minimum 4 max ?10')
    parser.add_argument('--num_samples', type=int, default=2, help='Number of SDE samples- is affected if sigma >0 ')

    # Model specific args
    parser.add_argument('--start_dec_at_treatment', type=bool, default=True, help='Whether to encode the data until treatment and the decode or decode from the beginning!)')
    parser.add_argument('--variational_encoder', type=bool, default=False, help='Whether encoder is variational or not - not finished variational)')
    parser.add_argument('--encoder_hidden_dim', type=int, default=64, help='Output of the encoder into a latent space. This needs to match the total SDE input dims ')
    parser.add_argument('--encoder_num_layers', type=int, default=2, help='Number of layers in encoder GRU')
    parser.add_argument('--encoder_w_time', type=bool, default=False, help='Whether encoder includes time in its inputs)')
    parser.add_argument('--encoder_reverse_time', type=bool, default=False, help='Whether encoder runs with inputs backwards in time)')
    parser.add_argument('--variational_sampling', type=bool, default=False, help='If NOT using encoder, to learn a variational q distribution for the unobserved dims)')

    parser.add_argument('--SDE_input_state', type=str, default='full', choices=['full', 'partial'], help='which dims to include in the SDE NN - always do full!')
    parser.add_argument('--SDEnet_hidden_dim', type=int, default=200, help='Hidden dim for SDE NN  ')
    parser.add_argument('--SDEnet_depth', type=int, default=4, help='Num layeres for SDE NN  ')
    parser.add_argument('--SDEnet_out_dims', type=int, default=1, help='Num output dims for SDE NN  ')
    parser.add_argument("--output_scale",type=float,default = 0.01, help = "Standard Deviation when computing GaussianNegLL between Y_true and Y_hat")
    parser.add_argument('--KL_weighting_SDE', type=float, default=0.1, help='Defines the weighting to the KL loss for the SDE')

    # Solver args
    parser.add_argument('--adjoint', type=bool, default=False, const=True, nargs="?")
    #parser.add_argument('--adaptive', type=bool, default=False, const=True, nargs="?")
    #parser.add_argument('--method', type=str, default='euler', choices=('euler', 'milstein', 'srk'), help='Name of numerical solver.')
    #parser.add_argument('--dt', type=float, default=1e-2)
    #parser.add_argument('--rtol', type=float, default=1e-3)
    #parser.add_argument('--atol', type=float, default=1e-3)

    # Training specific args
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate for the optimizer')
    parser.add_argument('--batch_size', type=int, default=64, help='Training batch size')
    parser.add_argument('--max_epochs', type=int, default=50, help='Maximum number of epochs to train')
    parser.add_argument('--accelerator', type=str, default='auto', choices=['gpu', 'mps', 'cpu', 'auto'], help='Which accelerator to use')
    #parser.add_argument('--devices', type=str, default='auto', choices=['gpu', 'mps', 'cpu', 'auto'], help='Which number of devices to use')

    
    args = parser.parse_args()
    main(args)
