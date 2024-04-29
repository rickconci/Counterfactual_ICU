
import pytorch_lightning as pl
#from lightning.pytorch.tuner import Tuner
import torch
from pytorch_lightning.loggers import WandbLogger


from CV_data_1 import CVDataModule
from models_1 import ODEVAELightning




def main():
    for i in range(torch.cuda.device_count()):
        print(torch.cuda.get_device_name(i))

    wandb_logger = WandbLogger(project='ode_vae_project', log_model='all')

    cv_data_module = CVDataModule(batch_size=50, seed=1234, N_ts=1000, gamma=0.1, noise_std=0.1, t_span=30, t_treatment=15)

    # Instantiate the model
    ode_vae_model = ODEVAELightning(output_dim=2, hidden_dim=64, latent_dim=8, kl_coeff=0.5, learning_rate=1e-3)

    # Instantiate the PyTorch Lightning trainer with the WandbLogger
    trainer = pl.Trainer(max_epochs=600, accelerator='gpu', logger=wandb_logger)

    # Fit the model
    trainer.fit(ode_vae_model, cv_data_module)

    wandb_logger.experiment.finish()



if __name__ == '__main__':
    

    main()





