import math
import numpy as np

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import torch
from torch import Tensor
from torch import nn
from torch.nn  import functional as F 
from torch.autograd import Variable
from torch import float32


import pytorch_lightning as pl
import wandb

import torch
from torch.utils.data import Dataset, DataLoader, Subset
import os
import argparse
import numpy as np

use_cuda = torch.cuda.is_available()


#from helper_func_1 import NNODEF, NeuralODE



from torchdiffeq import odeint_adjoint as odeint


class ODEFunc(nn.Module):
    def __init__(self, latent_dim, hidden_dim):
        super(ODEFunc, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, latent_dim)
        )

    def forward(self, t, y):
        return self.net(y.float())

class NeuralODEDecoder(nn.Module):
    def __init__(self, output_dim, hidden_dim, latent_dim):
        super(NeuralODEDecoder, self).__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        # ODE function
        self.ode_func = ODEFunc(latent_dim, hidden_dim)
        
        # Linear layers for processing outputs
        self.l2h = nn.Linear(latent_dim, hidden_dim)
        self.relu1 = nn.ReLU()  
        self.h2o = nn.Linear(hidden_dim, output_dim)

    def forward(self, z0, t):
        #print('z0 :', z0.shape) #[10,3] Now correct 

        t = t[0] if t.ndim > 1 else t
        t = t.flatten() # [30]
        #print('t :', t.shape) #[30] now correct 

        options = {
            'dtype': torch.float32
        }

        # Solve the ODE with specified options
        zs = odeint(self.ode_func, z0.float(), t.float(), method = 'bosh3', rtol=1e-2, atol = 1e-3,  options=options)
        
        #print('zs', zs.shape)  #[30, 10, 3] instead of [10, 30, 3], where 10 is the batch size, 30 the seq len and 3 the latent dim
        zs = zs.permute(1, 0, 2)
        #print('Permutated zs', zs.shape)  #[10, 30, 3] 

        hs = self.relu1(self.l2h(zs))
        xs = self.h2o(hs)
        #print('xs', xs.shape)  #[10, 30, 2]

        return xs

        
class RNNEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(RNNEncoder, self).__init__()
        self.input_dim = input_dim  # input = output = 2
        self.hidden_dim = hidden_dim  # hidden_dim = 64
        self.latent_dim = latent_dim  # latent_dim = 3

        self.rnn = nn.GRU(input_dim + 1, hidden_dim, batch_first=True)
        self.hid2lat = nn.Linear(hidden_dim, 2 * latent_dim)

    def forward(self, x, t):
        #print('Initial x shape:', x.shape)  # Expected: [batch_size, seq_length, input_dim]
        #print('Initial t shape:', t.shape)  # Expected: [batch_size, seq_length, 1]

        # Calculate the time differences
        t_diff = torch.zeros_like(t)
        t_diff[:, 1:] = t[:, 1:] - t[:, :-1]  # Forward differences
        t_diff[:, 0] = 0.
        #print('Time differences shape:', t_diff.shape)  # Should match t's shape

        xt = torch.cat((x, t_diff), dim=-1)  # Concatenate along the feature dimension
        #print('Concatenated xt shape:', xt.shape)  # Expected: [batch_size, seq_length, input_dim + 1]

        # Reverse the sequence along the time dimension
        xt_reversed = xt.flip(dims=[1])
        #print('Reversed xt shape:', xt_reversed.shape)  # Should match xt's shape

        # Apply the RNN
        _, h0 = self.rnn(xt_reversed)
        #print('Output hidden state h0 shape:', h0.shape)  # Expected: [1, batch_size, hidden_dim]

        # Process the last hidden state to produce latent variables
        z0 = self.hid2lat(h0.squeeze(0))  # Remove the first dimension
        #print('Latent variable z0 shape:', z0.shape)  # Expected: [batch_size, 2 * latent_dim]

        # Split the output into mean and log-variance components
        z0_mean = z0[:, :self.latent_dim]
        z0_log_var = z0[:, self.latent_dim:]
        #print('z0_mean shape:', z0_mean.shape)  # Expected: [batch_size, latent_dim]
        #print('z0_log_var shape:', z0_log_var.shape)  # Expected: [batch_size, latent_dim]

        return z0_mean, z0_log_var



class ODEVAE(nn.Module):
    def __init__(self, output_dim, hidden_dim, latent_dim):
        super(ODEVAE, self).__init__()
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        self.encoder = RNNEncoder(output_dim, hidden_dim, latent_dim)
        self.decoder = NeuralODEDecoder(output_dim, hidden_dim, latent_dim)

    def forward(self, x, t, MAP=False):
        z_mean, z_log_var = self.encoder(x, t)
        if MAP:
            z = z_mean
        else:
            z = z_mean + torch.randn_like(z_mean) * torch.exp(0.5 * z_log_var)
        x_p = self.decoder(z, t)

        return x_p, z, z_mean, z_log_var

    def generate_with_seed(self, seed_x, t):
        seed_t_len = seed_x.shape[0]
        z_mean, z_log_var = self.encoder(seed_x, t[:seed_t_len])
        x_p = self.decoder(z_mean, t)
        return x_p

class ODEVAELightning(pl.LightningModule):
    def __init__(self, output_dim, hidden_dim, latent_dim, learning_rate= 1e-3, kl_coeff=1.0):
        super().__init__()
        self.learning_rate = learning_rate
        self.kl_coeff = kl_coeff
        self.model = ODEVAE(output_dim, hidden_dim, latent_dim)
        self.save_hyperparameters()

    def forward(self, x, t, MAP=False):
        return self.model(x, t, MAP)

    def training_step(self, batch, batch_idx):
        y, t = batch
        #print('y batch', y.shape)
        #print('t batch', t.shape)
        x_p, z, z_mean, z_log_var = self.model(y, t)
        #print('x_p', x_p.shape)
        #print('y', y.shape)
        # Reconstruction Loss
        recon_loss = F.mse_loss(x_p, y)
        # KL Divergence Loss
        kl_loss = -0.5 * torch.sum(1 + z_log_var - z_mean.pow(2) - z_log_var.exp())
        # Total Loss
        loss = recon_loss +self.kl_coeff*kl_loss
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return loss

    def validation_step(self, batch, batch_idx):
        y, t = batch
        #print('t', t.shape)
        #print('y', y.shape)
        x_p, _, _, _ = self.model(y, t, MAP=True)
        #print('x_p', x_p.shape)

        loss = F.mse_loss(x_p, y)
        self.log('val_loss', loss, on_epoch=True, prog_bar=True)
        return loss

    def test_step(self, batch, batch_idx):
        y, t = batch
        x_p, _, _, _ = self.model(y, t, MAP=True)
        loss = F.mse_loss(x_p, y)
        self.log('test_loss', loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
    
    def plot_and_log(self, y, y_pred, epoch_idx):
        num_samples = min(y.size(0), 3)  
        fig, axes = plt.subplots(num_samples, 1, figsize=(10, num_samples * 3))
        if num_samples == 1:
            axes = [axes]  

        for i in range(num_samples):
            axes[i].plot(y[i, :, :].cpu().numpy(), label='True')
            axes[i].plot(y_pred[i, :, :].cpu().numpy(), label='Predicted')
            axes[i].set_title(f'Sample {i + 1}')
            axes[i].legend()

        plt.tight_layout()
        plt.close(fig)

        # Log the figure to wandb
        wandb.log({"predictions_vs_actuals": wandb.Image(fig)}, commit=False)

    def on_train_epoch_end(self):
        n = 10
        if self.current_epoch % n == 0:  
            sample_batch = next(iter(self.trainer.datamodule.train_dataloader()))
            y, t = sample_batch
            with torch.no_grad():
                y, t = y.to(self.device), t.to(self.device)  
                y_pred, _, _, _ = self.model(y, t, MAP=True)

                self.plot_and_log(y, y_pred, self.current_epoch)
        
       



'''

class NeuralODEDecoder(nn.Module):
    def __init__(self, output_dim, hidden_dim, latent_dim):
        super(NeuralODEDecoder, self).__init__()
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

        func = NNODEF(latent_dim, hidden_dim, time_invariant=True)
        self.ode = NeuralODE(func)
        self.l2h = nn.Linear(latent_dim, hidden_dim)
        self.h2o = nn.Linear(hidden_dim, output_dim)

    def forward(self, z0, t):
        #print('z0 :', z0.shape) #[30,3]
        #print('t :', t.shape) #[30,3]

        zs = self.ode(z0, t, return_whole_sequence=True)
        #print('zs', zs.shape)  
        hs = self.l2h(zs)
        xs = self.h2o(hs)
        #print('xs', xs.shape)  #[30, 30, 3] instead of [10, 30, 3]
        return xs



class RNNEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(RNNEncoder, self).__init__()
        self.input_dim = input_dim # input = output = 2
        self.hidden_dim = hidden_dim #hidden_dim = 64
        self.latent_dim = latent_dim #latent_dim = 3

        self.rnn = nn.GRU(input_dim+1, hidden_dim)
        self.hid2lat = nn.Linear(hidden_dim, 2*latent_dim)

    def forward(self, x, t):
        # Concatenate time to input

        #print('x encoder', x.shape) #[10, 30, 2]
        #print('t encoder', t.shape) #[10, 30, 1]
        t = t.clone()
        t[1:] = t[:-1] - t[1:]
        t[0] = 0.
        #print('x', x.shape)
        #print('t', t.shape)
        xt = torch.cat((x, t), dim=-1)

        _, h0 = self.rnn(xt.flip((0,)))  # Reversed
        # Compute latent dimension
        z0 = self.hid2lat(h0[0]) 
        #print('z0', z0.shape) #[30, 6]
        z0_mean = z0[:, :self.latent_dim]
        z0_log_var = z0[:, self.latent_dim:]

        #print('z0_mean', z0_mean.shape) # [30, 3]
        #print('z0_log_var', z0_log_var.shape) #[30, 3]
        return z0_mean, z0_log_var


''' 