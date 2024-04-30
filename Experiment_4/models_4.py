
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go


import os
import argparse
import numpy as np
import pandas as pd
from functools import partial


import torch
from torch import Tensor
from torch import nn
from torch.nn  import functional as F 
from torch.autograd import Variable
from torch import float32
from torch.utils.data import Dataset, DataLoader, Subset

import lightning as L
from lightning import LightningModule

from torchdiffeq import odeint_adjoint as odeint
import wandb

import torchsde


from utils_3 import _stable_division, GaussianNLLLoss, LinearScheduler, interpolate_colors

use_cuda = torch.cuda.is_available()



class MLPSimple(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, depth, activations=None, dropout_p=None):
        super().__init__()
        #print(f"Initializing MLPSimple: input_dim={input_dim}, output_dim={output_dim}, hidden_dim={hidden_dim}, depth={depth}")
        self.input_layer = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU())
        self.output_dim = output_dim
        
        self.output_layer = nn.Sequential(nn.Linear(hidden_dim, output_dim))
        
        if activations is None:
            activations = [nn.ReLU() for _ in range(depth)]
        
        if dropout_p is None:
            dropout_p = [0. for _ in range(depth)]
        
        assert len(activations) == depth, "Mismatch between depth and number of provided activations."
        
        self.layers = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Dropout(dropout_p[i]), activations[i])
            for i in range(depth)
        ])

    def forward(self, x):
        ##print(f"Forward pass started with input shape: {x.shape}")
        x = self.input_layer(x)
        ##print(f"Post input layer shape: {x.shape}")
        
        for i, mod in enumerate(self.layers):
            x_old_shape = x.shape
            x = mod(x)
            ##print(f"Layer {i+1}: input shape: {x_old_shape}, output shape: {x.shape}")
        
        x = self.output_layer(x)
        ##print(f"Post output layer shape: {x.shape}")
        return x



class RNNEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim):
        super(RNNEncoder, self).__init__()
        self.input_dim = input_dim  # input = output = 2
        self.hidden_dim = hidden_dim  # hidden_dim = 64
        self.latent_dim = latent_dim  # latent_dim = 

        self.rnn = nn.GRU(input_dim + 1, hidden_dim, batch_first=True)
        self.hid2lat = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x, t):
        #print('Initial x shape:', x.shape)  # Expected: [batch_size, seq_length, input_dim]
        #print('Initial t shape:', t.shape)  # Expected: [batch_size, seq_length, 1]


        # Calculate the time differences
        t_diff = torch.zeros_like(t)
        t_diff[:, 1:] = t[:, 1:] - t[:, :-1]  # Forward differences
        t_diff[:, 0] = 0.
        t_diff = t_diff.unsqueeze(-1) 
        #print('Time differences shape:', t_diff.shape)  # Should match t's shape

        xt = torch.cat((x, t_diff), dim=-1)  # Concatenate along the feature dimension
        #print('Concatenated xt shape:', xt.shape)  # Expected: [batch_size, seq_length, input_dim + 1]

        # Reverse the sequence along the time dimension
        #xt = xt.flip(dims=[1])
        ##print('reversed xt shape:', xt.shape)  # Should match xt's shape

        # Apply the RNN
        _, h0 = self.rnn(xt)
        #print('Output hidden state h0 shape:', h0.shape)  # Expected: [1, batch_size, hidden_dim]

        # Process the last hidden state to produce latent variables
        z0 = self.hid2lat(h0.squeeze(0))  # Remove the first dimension
        #print('Latent variable z0 shape:', z0.shape)  # Expected: [batch_size, 2 * latent_dim]

        # Split the output into mean and log-variance components
        z0_mean = z0[:, :self.latent_dim]
        #z0_log_var = z0[:, self.latent_dim:]
        #print('z0_mean shape:', z0_mean.shape)  # Expected: [batch_size, latent_dim]
        ##print('z0_log_var shape:', z0_log_var.shape)  # Expected: [batch_size, latent_dim]

        return z0_mean #z0_log_var



class LatentSDE(torchsde.SDEIto):
    def __init__(self,latent_dim, hidden_dim, Tx_dim, theta, mu, sigma):

        super().__init__(noise_type="diagonal")
    
        self.theta = theta
        self.mu = mu
        self.register_buffer("sigma", torch.tensor([[sigma]]))
        
        u_dim = int(latent_dim/5)
        self.treatment_fun = MLPSimple(input_dim = 1, output_dim = u_dim, hidden_dim = 20, depth = 4, activations = [nn.ReLU() for _ in range(4)] )
        
        self.sde_drift = MLPSimple(input_dim = latent_dim + u_dim, output_dim = latent_dim, hidden_dim = 4*latent_dim, depth = 4, activations = [nn.Tanh() for _ in range(4)])
   
    
    def fun_treatment(self,t):
        return self.treatment_fun(t)

    def g(self,t,y):
        return self.sigma.repeat(y.shape)

    def h(self,t,y):
        return self.theta * (self.mu-y)

    def f(self,t,y, T):
        u = self.fun_treatment(t.repeat(y.shape[0],1))
        u_t = u * T[:,None]
        y_and_u = torch.cat((y,u_t),-1)
        return self.sde_drift(y_and_u) - self.h(t,y)
    
    def f_aug(self, t, y):  # Drift for augmented dynamics with logqp term.
        T = y[:,-1]
        y = y[:, 0:-2]
        f, g, h = self.f(t, y, T), self.g(t, y), self.h(t, y)
        u = _stable_division(f - h, g)
        f_logqp = .5 * (u ** 2).sum(dim=1, keepdim=True)
        return torch.cat([f, f_logqp, torch.zeros_like(f_logqp)], dim=1)
    
    def g_aug(self, t, y):  # Diffusion for augmented dynamics with logqp term.
        y = y[:, 0:-2]
        g = self.g(t, y)
        g_logqp = torch.zeros((y.shape[0],2), device = y.device)
        return torch.cat([g, g_logqp], dim=1)



class Neural_SDE_integrator(nn.Module):
    def __init__(self, latent_dim, hidden_dim, num_samples, Tx_dim, theta, mu, sigma):
        super(Neural_SDE_integrator, self).__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.num_samples = num_samples
        self.Tx_dim = Tx_dim
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
                        #theta,mu, sigma, embedding_dim
        self.sde_func = LatentSDE(latent_dim= latent_dim, 
                                      hidden_dim=hidden_dim, 
                                      Tx_dim= Tx_dim,
                                      theta = theta, 
                                      mu = mu, 
                                      sigma = sigma)

    def forward(self, z0, t, Tx):

        batch_size = z0.shape[0]
    
        

        t = t[0] if t.ndim > 1 else t
        t = t.flatten()  
        #print('time dim for ODE', t.shape) #= [time]

        #print('Z0',z0.shape )
        #print('aug_num_samples', torch.zeros(batch_size,self.num_samples, 1).shape)

        #print('Tx', Tx.shape) # shape = batch
        # let's convert now to batch x num_samples x 1
        Tx = Tx.unsqueeze(1).unsqueeze(2)
        Tx = Tx.repeat(1, self.num_samples, 1)  
        #print('Aug Tx', Tx.shape)

        # For each SDE sample, adding Tx as part of the initial state + an empty dim for the LogQ for that SDE sample
        aug_y0 = torch.cat([z0, torch.zeros(batch_size,self.num_samples, 1).to(z0), Tx.to(z0)], dim=-1)

        #print('aug_y0', aug_y0.shape)
        dim_aug = aug_y0.shape[-1]
        aug_y0 = aug_y0.reshape(-1,dim_aug)
        #print('augmented_init_state', aug_y0.shape)



        options = {'dtype': torch.float32} 

        aug_ys = torchsde.sdeint(sde=self.sde_func,
                y0=aug_y0,
                ts=t,
                method="euler",
                dt=0.05,
                adaptive=False,
                rtol=1e-3,
                atol=1e-3,
                names={'drift': "f_aug", 'diffusion': 'g_aug'},
                options = options)
                
        aug_ys = aug_ys.reshape(-1, self.num_samples, len(t),dim_aug) # reshape for # batch_size x num_samples x times x dim

        ys, logqp_path = aug_ys[:, :, :, 0:-2], aug_ys[:,: , -1, -2]
        logqp0 = 0
        logqp = (logqp0 + logqp_path)  # KL(t=0) + KL(path).

        return ys, logqp

    


class SDE_VAE(nn.Module):
    def __init__(self, 
                 input_dim, output_dim, hidden_dim, latent_dim, num_samples,  
                 use_whole_trajectory, post_tx_ode_len, Tx_dim,
                 theta, mu, sigma, dropout_p= 0.2):
        
        super(SDE_VAE, self).__init__()
        self.input_dim = input_dim #dim of input in observed space 
        self.output_dim = output_dim #dim of the output in the observed space
        self.latent_dim = latent_dim #dim of the latent space  
        self.hidden_dim = hidden_dim #dim of the hidden layers in NNs
        self.num_samples = num_samples #number of latent SDE samples

        self.post_tx_ode_len = post_tx_ode_len
        self.use_whole_trajectory = use_whole_trajectory #whether the output fun is applied pointwise to the latent ODE or takes in the whole traj and then converts to observed
        
        self.encoder = RNNEncoder(input_dim, 
                                  hidden_dim, 
                                  latent_dim)
        
        self.forward_SDE_latent = Neural_SDE_integrator(latent_dim = latent_dim, 
                                                   hidden_dim = hidden_dim, 
                                                   Tx_dim= Tx_dim,
                                                   num_samples = num_samples, 
                                                   theta= theta, 
                                                   mu=mu, 
                                                   sigma = sigma) 

        self.output_fun = MLPSimple(input_dim=self.latent_dim * (post_tx_ode_len if self.use_whole_trajectory else 1),
                            output_dim=self.output_dim * (post_tx_ode_len if self.use_whole_trajectory else 1),
                            hidden_dim=self.hidden_dim,
                            depth=3,
                            activations=[nn.ReLU() for _ in range(3)],
                            dropout_p=[0.5 for _ in range(3)])
        

        

    def forward(self, x, Tx, time_in, time_out,  MAP=True):
        # takes in:
        # x:  observed trajectory until treatment time 
        # time_in: time from start to treatment time 
        # time_out: time form treatment time to finish
        # Tx: treatment presence: binary vector with 1 = treated, 0 = untreated

        #We instead have the RNN to have the option of having a variational RNN latent, so split between the mean and var 
        z_mean = self.encoder(x, time_in)
        z = z_mean #if MAP else z_mean + torch.randn_like(z_mean) * torch.exp(0.5 * z_log_var) #if MAP then we do NOT sample, so effectively it's not longer variational, otherwise we sample


        # z shape starts as batch x latent, and I want to convert it to batch x num_samples x latent 
        z = z.unsqueeze(1)  # Add an extra dimension: shape becomes [batch, 1, latent]
        z = z.repeat(1, self.num_samples, 1) 
    
        # Generating latent dynamics using the SDE
        latent_traj, logqp = self.forward_SDE_latent(z, time_out, Tx)
        #print('latentSDEoutput', latent_traj.shape) #batch_size x num_samples x times x latent_dim
        

        # Apply the output function based on the configuration (need to correct!!)
        if self.use_whole_trajectory:
            batch_size, num_samples, seq_len , _ = latent_traj.shape
            latent_traj_flat = latent_traj.reshape(batch_size, -1)  # Flatten the entire trajectory
            pred_traj_flat = self.output_fun(latent_traj_flat)  # Process the whole trajectory
            predicted_traj = pred_traj_flat.reshape(batch_size, num_samples, seq_len, self.output_dim)  # Reshape back
        else:
            # Flatten for pointwise processing
            batch_size, num_samples, seq_len , _ = latent_traj.shape
            latent_traj_flat = latent_traj.reshape(-1, self.latent_dim)  
            pred_traj_flat = self.output_fun(latent_traj_flat)
            predicted_traj = pred_traj_flat.reshape(batch_size, num_samples, seq_len, self.output_dim)


        #may also want to adjust output so it's still:  batch x num_samples x times x dims

        return z, latent_traj, logqp, predicted_traj
        

    



class SDE_VAE_Lightning(LightningModule):
    def __init__(self, input_dim, output_dim, hidden_dim, latent_dim, 
                 use_whole_trajectory, post_tx_ode_len, Tx_dim, 
                 theta, mu, sigma,  KL_weighting, learning_rate, 
                 num_samples=5, 
                 output_scale = 0.01,
                 start_scheduler = 200, 
                 iter_scheduler = 600):
        super().__init__()
        
        self.VAE_model = SDE_VAE(input_dim= input_dim,
                                output_dim = output_dim, 
                                hidden_dim = hidden_dim, 
                                latent_dim = latent_dim, 
                                num_samples = num_samples, 
                                use_whole_trajectory = use_whole_trajectory, 
                                post_tx_ode_len = post_tx_ode_len,
                                Tx_dim = Tx_dim, 
                                theta = theta, 
                                mu=mu, 
                                sigma = sigma)
        

        self.post_tx_ode_len = post_tx_ode_len
        self.num_samples = num_samples

        self.loss = GaussianNLLLoss(reduction = "none")
        self.MSE_loss = nn.MSELoss(reduction = "none")
        self.output_scale = torch.tensor([output_scale], requires_grad = False, device = self.device)

        self.KL_weighting = KL_weighting
        self.learning_rate = learning_rate

        self.kl_scheduler = LinearScheduler(start = start_scheduler, iters = iter_scheduler)
        self.save_hyperparameters()

    def forward(self, x, t, MAP=False):
        return self.VAE_model(x, t, MAP)
    
    def compute_factual_loss(self, Y, Y_hat, logqp):
        #print('Y initial:', Y.shape)  # Shape of ground truth data
        #print('Y_hat initial:', Y_hat.shape)  # Shape of predicted data from SDE

        # Convert Y_true to match Y_hat shape: batch x num_samples x times x dims, by repeating for each num_sample 
        # so that each sample can be compared to the ground truth. Also make sure that the Y sequence length is as
        # previously established (the gap between t* and the end), given by post_tx_ode_len
        Y_expanded = Y[:, :self.post_tx_ode_len, :].unsqueeze(1)
        Y_true = Y_expanded.repeat(1, self.num_samples, 1, 1)  
        #print('Y_true after repeat:', Y_true.shape)

        # Apply the negative gaussian log likelihood loss between the true trajectory and the predicted SDE trajectories,
        # with a Standard dev preset by output_scale (why preset?)
        fact_loss = self.loss(Y_true, Y_hat, self.output_scale.repeat(Y_hat.shape).to(self.device))
        fact_loss = fact_loss.sum((2, 3))  # sum across times and dims (keeping for each batch and SDE sample)
        #print('fact_loss after sum:', fact_loss.shape)

        # MSE loss between the Y true and the MEAN of the SDE samples predictions
        mse_recon_loss = torch.sqrt(self.MSE_loss(Y[:, :self.post_tx_ode_len, :], Y_hat.mean(1))).mean()
        # Now find the mean of the standard devs of the predictions across the SDE samples
        std_preds = Y_hat.std(1).mean()

        # Now find the total loss: the average gaussian log likelihood across SDE samples for the batch + mean logQP across samples & batch
        SDE_loss = fact_loss.mean() + self.KL_weighting * logqp.mean() * self.kl_scheduler.val 
        #print('Final SDE loss:', SDE_loss)

        return SDE_loss, mse_recon_loss, std_preds

    
    def compute_counterfactual_loss(self, Y, Y_cf, Y_hat, Y_hat_cf):
        #print('Y:', Y.shape)
        #print('Y_cf:', Y_cf.shape)
        #print('Y_hat:', Y_hat.shape)
        #print('Y_hat_cf:', Y_hat_cf.shape)

        # MSE loss between the counterfactual Y and the MEAN of the SDE counterfactual predictions
        mse_cf = torch.sqrt(self.MSE_loss(Y_cf[:, :self.post_tx_ode_len, :], Y_hat_cf.mean(1))).mean()
        #print('mse_cf:', mse_cf)

        # Standard deviation of predictions across the SDE counterfactual samples
        std_preds_cf = Y_hat_cf.std(1).mean()
        #print('std_preds_cf:', std_preds_cf)

        # Individual Treatment Effect computed as the difference between Y_cf and Y
        ite = (Y_cf[:, :self.post_tx_ode_len, :] - Y[:, :self.post_tx_ode_len, :])
        #print('ite:', ite.shape)

        # Predicted Individual Treatment Effect computed as the difference between the mean predictions of Y_hat_cf and Y_hat
        ite_hat = (Y_hat_cf.mean(1) - Y_hat.mean(1))
        #print('ite_hat:', ite_hat.shape)

        # MSE of the ITE
        mse_ite = torch.sqrt(self.MSE_loss(ite, ite_hat)).mean()
        #print('mse_ite:', mse_ite)

        return mse_cf, mse_ite, std_preds_cf


    def training_step(self, batch, batch_idx):
        #print("Variatinal SDE BEGIN TRAINING STEP")
        X, Y, T, Y_cf, p, thetas_0, time_X, time_Y = batch

        z, latent_traj, logqp, Y_hat = self.VAE_model(X, T, 
                                                             time_in = time_X, 
                                                             time_out = time_Y, 
                                                             MAP=True) 
        #MAP=True means that the encoder is no longer variational
        
        SDE_loss, mse, std_preds = self.compute_factual_loss(Y, Y_hat, logqp)
        
        
        self.log('train_SDE_loss', SDE_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_recon_loss', mse, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_std_preds', std_preds, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        self.kl_scheduler.step()

        return SDE_loss

    def validation_step(self, batch, batch_idx):
        #print('VALIDATION')
        X, Y, T, Y_cf, p, thetas_0, time_X, time_Y = batch
        #print('X', X.shape)
        #print('Y', Y.shape)
        
       
        #MAP = true as you don't sample the encoder variational latents during validation step 
        z, latent_traj, logqp, Y_hat  = self.VAE_model(X, T, 
                                                             time_in = time_X, 
                                                             time_out = time_Y,
                                                             MAP=True)
        
        #print('Y_hat',Y_hat.shape )
        #print('logqp',logqp.shape )
        #print('latent_traj', latent_traj.shape)
        
        SDE_loss, mse, std_preds = self.compute_factual_loss(Y, Y_hat, logqp)

        #print('SDE_loss', SDE_loss.shape)
        #print('mse_loss', mse.shape)
        #print('std_preds', std_preds.shape)

        
        T_cf = (~T.bool()).long()
        _, latent_traj_cf, _, Y_hat_cf = self.VAE_model(X, T_cf, 
                                                             time_in = time_X, 
                                                             time_out = time_Y,
                                                             MAP=True)
        
        mse_cf, mse_ite, std_preds_cf = self.compute_counterfactual_loss(Y,Y_cf, Y_hat,Y_hat_cf)



        if batch_idx ==0:
            #self.plot_trajectories( X, Y, Y_hat,latent_traj,  chart_type = "val" )
            self.plot_SDE_trajectories( X, Y, Y_hat,latent_traj,  chart_type = "val" )
            

        self.log('val_SDE_loss', SDE_loss, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_recon_loss', mse, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_std_preds', std_preds, on_epoch=True, prog_bar=True, logger=True)

        self.log('val_mse_cf', mse_cf, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_PEHE', mse_ite, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_std_preds_cf', std_preds_cf, on_epoch=True, prog_bar=True, logger=True)


        return SDE_loss

    def test_step(self, batch, batch_idx):
        X, Y, T, Y_cf, p, thetas_0, time_X, time_Y = batch
       
       
        #MAP = true as you don't sample the encoder variational latents during validation step 
        z, latent_traj, logqp, Y_hat = self.VAE_model(X, T, 
                                                             time_in = time_X, 
                                                             time_out = time_Y,
                                                             MAP=True)
        
        SDE_loss, mse, std_preds = self.compute_factual_loss(Y, Y_hat, logqp)

        
        T_cf = (~T.bool()).long()
        _, latent_traj_cf, _, Y_hat_cf = self.VAE_model(X, T_cf, 
                                                             time_in = time_X, 
                                                             time_out = time_Y,
                                                             MAP=True)
        
        mse_cf, mse_ite, std_preds_cf = self.compute_counterfactual_loss(Y,Y_cf, Y_hat,Y_hat_cf)

        if batch_idx ==0:
            self.plot_SDE_trajectories( X, Y, Y_hat,latent_traj,  chart_type = "test" )

        self.log('test_SDE_loss', SDE_loss, on_epoch=True, prog_bar=True, logger=True)
        self.log('test_recon_loss', mse, on_epoch=True, prog_bar=True, logger=True)
        self.log('test_std_preds', std_preds, on_epoch=True, prog_bar=True, logger=True)

        self.log('test_mse_cf', mse_cf, on_epoch=True, prog_bar=True, logger=True)
        self.log('test_PEHE', mse_ite, on_epoch=True, prog_bar=True, logger=True)
        self.log('test_std_preds_cf', std_preds_cf, on_epoch=True, prog_bar=True, logger=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr = self.learning_rate)
        
        scheduler = {"monitor": "val_SDE_loss", "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode = "min", factor = 0.5, patience = 50, verbose = True)}
        return {"optimizer": optimizer, "lr_scheduler":scheduler}
    

    
    
    def plot_SDE_trajectories(self, X, Y, Y_hat,latent_traj,  chart_type = "val"):

        pass


    def plot_trajectories(self, X, Y, Y_hat, latent_traj, chart_type="val"):
        # Ensure tensors are on CPU for plotting
        X = X.cpu()
        Y = Y.cpu()
        Y_hat = Y_hat.cpu()
        latent_traj = latent_traj.cpu()  # Assume latent_traj is passed or correctly fetched earlier

        # Only take the first item of the batch
        X = X[0]
        Y = Y[0]
        Y_hat = Y_hat[0]
        latent_traj = latent_traj[0]

        # Define a color palette that provides distinct colors for multiple dimensions
        base_colors = ['dodgerblue', 'sandybrown', 'mediumseagreen', 'mediumorchid', 'coral', 'slategray']
        light_colors = ['lightblue', 'peachpuff', 'lightgreen', 'plum', 'lightsalmon', 'lightgray']
        dark_colors = ['darkblue', 'darkorange', 'darkgreen', 'darkorchid', 'darkred', 'darkslategray']

        start_color = (0,255,127)  # Royal blue
        end_color = (238,130,238)     # Dark orange
        latent_colors = interpolate_colors(start_color, end_color, latent_traj.shape[1])


        fig = make_subplots(rows=2, cols=1, subplot_titles=("Latent Trajectories", "Observed Trajectories"))

        # Time axes for observed and latent data
        time_x = np.arange(X.shape[0])
        time_y = time_x[-1] + 1 + np.arange(Y.shape[0])
        time_latent = np.arange(latent_traj.shape[0])

        # Plot latent trajectories
        for i in range(latent_traj.shape[1]):  # Assuming second dim is the feature dimension of latent space
            fig.add_trace(
                go.Scatter(x=time_latent, y=latent_traj[:, i], mode='lines', name=f'Latent Dim {i}', line=dict(color=latent_colors[i])),
                row=1, col=1
            )

        # Plot X, Y, and Y_hat trajectories
        for i in range(max(X.shape[1], Y.shape[1])):  # Handle different dimensions for X and Y/Y_hat
            if i < X.shape[1]:
                fig.add_trace(
                    go.Scatter(x=time_x, y=X[:, i], mode='lines', name=f'Input_{i}', line=dict(color=base_colors[i % len(base_colors)])),
                    row=2, col=1
                )
            if i < Y.shape[1]:
                fig.add_trace(
                    go.Scatter(x=time_y, y=Y[:, i], mode='lines', name=f'Factual_{i}', line=dict(color=light_colors[i % len(light_colors)])),
                    row=2, col=1
                )
                fig.add_trace(
                    go.Scatter(x=time_y, y=Y_hat[:, i], mode='lines', name=f'Predicted_{i}', line=dict(color=dark_colors[i % len(dark_colors)])),
                    row=2, col=1
                )

        # Update layout
        fig.update_layout(height=600, width=800, title_text=f"{chart_type} Validation Trajectories")
        fig.update_xaxes(title_text="Time Steps")
        fig.update_yaxes(title_text="Values", row=1, col=1)
        fig.update_yaxes(title_text="Values", row=2, col=1)

        # Optionally show the figure, useful for interactive sessions or debugging
        # fig.show()

        # Log the figure to wandb
        wandb.log({"predictions_vs_actuals": fig})


'''

    def plot_trajectories(self, X, Y, Y_hat,latent_traj, chart_type="val"):
        # Ensure tensors are on CPU for plotting
        X = X.cpu()
        Y = Y.cpu()
        Y_hat = Y_hat.cpu()
        latent_traj = latent_traj.cpu()

        # Only take the first item of the batch
        X = X[0]
        Y = Y[0]
        Y_hat = Y_hat[0]
        latent_traj = latent_traj[0]  
        # Define a color palette that provides distinct colors for multiple dimensions
        base_colors = ['dodgerblue', 'sandybrown', 'mediumseagreen', 'mediumorchid', 'coral', 'slategray']
        light_colors = ['lightblue', 'peachpuff', 'lightgreen', 'plum', 'lightsalmon', 'lightgray']
        dark_colors = ['darkblue', 'darkorange', 'darkgreen', 'darkorchid', 'darkred', 'darkslategray']

        fig = make_subplots(rows=2, cols=1, subplot_titles=("Latent Trajectories", "Observed Trajectories"))

        # Prepare DataFrame list for the input features X and corresponding outputs
        all_data = []
        time_x = np.arange(X.shape[0])
        time_y = time_x[-1] + 1 + np.arange(Y.shape[0])
        time_latent = np.arange(latent_traj.shape[0])

        for i in range(latent_traj.shape[1]):  # Assuming second dim is the feature dimension of latent space
            fig.add_trace(
                go.Scatter(x=time_latent, y=latent_traj[:, i], mode='lines', name=f'Latent Dim {i}'),
                row=1, col=1
            )

        # Check dimensionality and adjust Y and Y_hat if necessary
        if Y.ndim == 1:
            Y = Y.unsqueeze(1)
            Y_hat = Y_hat.unsqueeze(1)

        max_dims = max(X.shape[-1], Y.shape[-1])

        for dim in range(max_dims):
            if dim < X.shape[-1]:
                # Input data
                X_df = pd.DataFrame({
                    "Blood Pressure": X[:, dim],
                    "time": time_x,
                    "type": f"Input_{dim}",
                    "color": base_colors[dim % len(base_colors)]
                })
                all_data.append(X_df)

            if dim < Y.shape[-1]:
                # Observed output data
                Y_df = pd.DataFrame({
                    "Blood Pressure": Y[:, dim],
                    "time": time_y,
                    "type": f"Factual_{dim}",
                    "color": light_colors[dim % len(light_colors)]
                })
                all_data.append(Y_df)

                # Predicted output data
                Y_hat_df = pd.DataFrame({
                    "Blood Pressure": Y_hat[:, dim],
                    "time": time_y,
                    "type": f"Predicted_{dim}",
                    "color": dark_colors[dim % len(dark_colors)]
                })
                all_data.append(Y_hat_df)

        # Combine all dimensions data
        df = pd.concat(all_data)

        # Plotting with a color scheme
        fig = px.line(df, x="time", y="Blood Pressure", color='type',
                    color_discrete_map={row['type']: row['color'] for index, row in df.iterrows()},
                    title=f"{chart_type} longitudinal predictions")
        fig.update_traces(dict(mode='markers+lines'))

        # Log the figure to wandb
        wandb.log({"predictions_vs_actuals": fig})


'''
       



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
        ##print('z0 :', z0.shape) #[30,3]
        ##print('t :', t.shape) #[30,3]

        zs = self.ode(z0, t, return_whole_sequence=True)
        ##print('zs', zs.shape)  
        hs = self.l2h(zs)
        xs = self.h2o(hs)
        ##print('xs', xs.shape)  #[30, 30, 3] instead of [10, 30, 3]
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

        ##print('x encoder', x.shape) #[10, 30, 2]
        ##print('t encoder', t.shape) #[10, 30, 1]
        t = t.clone()
        t[1:] = t[:-1] - t[1:]
        t[0] = 0.
        ##print('x', x.shape)
        ##print('t', t.shape)
        xt = torch.cat((x, t), dim=-1)

        _, h0 = self.rnn(xt.flip((0,)))  # Reversed
        # Compute latent dimension
        z0 = self.hid2lat(h0[0]) 
        ##print('z0', z0.shape) #[30, 6]
        z0_mean = z0[:, :self.latent_dim]
        z0_log_var = z0[:, self.latent_dim:]

        ##print('z0_mean', z0_mean.shape) # [30, 3]
        ##print('z0_log_var', z0_log_var.shape) #[30, 3]
        return z0_mean, z0_log_var


class ControlledODE(nn.Module):
    def __init__(self, latent_dim, hidden_dim, Tx_dim, theta, mu):
        super().__init__()
        
        self.theta = theta
        self.mu = mu
        
        self.Tx_dim = Tx_dim
        u_dim = int(latent_dim / 2)
        self.treatment_fun = MLPSimple(input_dim=Tx_dim, output_dim=u_dim, hidden_dim=hidden_dim, depth=4, activations=[nn.ReLU() for _ in range(4)])
        self.ode_drift = MLPSimple(input_dim=latent_dim + u_dim, output_dim=latent_dim, hidden_dim=4*hidden_dim, depth=4, activations=[nn.Tanh() for _ in range(4)])
    
    def fun_treatment(self, t):
        #this is the control process 
        return self.treatment_fun(t)

    def h(self, t, y):
        #this is the mean reverting process
        return self.theta * (self.mu - y)

    def f_aug(self, t, y, T):
        #this is the drift process 
        u = self.fun_treatment(t.repeat(y.shape[0], 1))
        u_t = u * T[:, None]
        #print('u_t', u_t.shape)
        y_and_u = torch.cat((y, u_t), -1)

        drift = self.ode_drift(y_and_u)
        #print('drift', drift.shape)
        correction = self.h(t, y) #the correction is a mean reverting process
        #print('correction', correction.shape) 
        corrected_drift = drift - correction

        null_dim = torch.zeros(corrected_drift.shape[0], self.Tx_dim, device=corrected_drift.device)
        augmented_corrected_drift = torch.cat([corrected_drift, null_dim], dim=-1)
        #print('augmented_corrected_drift', augmented_corrected_drift.shape)

        return augmented_corrected_drift
    
    def forward(self, t, y):
        T = y[:, -1]  # Last dimension for control variable
        y = y[:, 0:-1]  # Remaining dimensions for state
        #print('T', T.shape)
        #print('y', y.shape)
        return self.f_aug(t, y, T)





class ODEFunc(nn.Module):
    def __init__(self, latent_dim, hidden_dim):
        super(ODEFunc, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, latent_dim)
        )

    def forward(self, t, y):
        return self.net(y.float())



class ODEFunc(nn.Module):
    def __init__(self, latent_dim, hidden_dim, treatment_fun):
        super(ODEFunc, self).__init__()
        self.treatment_fun = treatment_fun
        self.fc1 = nn.Linear(latent_dim + treatment_fun.output_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, latent_dim)
        self.relu = nn.ReLU()

    def forward(self, t, y):
        # Splitting the state y into actual state and treatment indicator
        actual_state, tx = y[:, :-1], y[:, -1]

        # Assuming last value in y is the treatment indicator
        actual_state, tx = y[:, :-1], y[:, -1].unsqueeze(1)
        # Ensure t is correctly sized for batch and stacked with tx
        t_repeated = t.repeat(tx.size(0), 1)
        tx_and_t = torch.cat([tx, t_repeated], dim=1)  # Ensure dimensions align here

        # Calculate the treatment effect
        u = self.treatment_fun(tx_and_t)
        # Combine the treatment effect with the actual state
        combined_input = torch.cat([actual_state, u], dim=1)

        #print("tx shape:", tx.shape)
        #print("t repeated shape:", t_repeated.shape)
        #print("treatment output shape:", u.shape)
        #print("combined input shape:", combined_input.shape)


        return self.fc2(self.relu(self.fc1(combined_input)))

class NeuralODEDecoder(nn.Module):
    def __init__(self, output_dim, hidden_dim, latent_dim):
        super(NeuralODEDecoder, self).__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        u_dim = int(latent_dim/5)
        augmented_latent_dim = latent_dim + u_dim

        #2 inputs for treatment_fun, one is the Tx value, another is the time value 
        self.treatment_fun = MLPSimple(input_dim = 2, output_dim = u_dim, hidden_dim = 20, depth = 4, activations = [nn.ReLU() for _ in range(4)] )
        self.ode_func = MLPSimple(input_dim = augmented_latent_dim, output_dim = latent_dim, hidden_dim = 4*latent_dim, depth = 4, activations = [nn.Tanh() for _ in range(4)])

        
        # Linear layers for processing outputs
        self.l2h = nn.Linear(augmented_latent_dim, hidden_dim)
        self.h2o = nn.Linear(hidden_dim, output_dim)

    

    def forward(self, z0, t, Tx):
        #print('z0 :', z0.shape) #[batch, latent_dim] Now correct 

        t = t[0] if t.ndim > 1 else t
        t = t.flatten() # [30= timepoints]
        #print('t :', t.shape) #[30] now correct 

        batch_size = z0.shape[0]
        #print(f"Batch size extracted: {batch_size}")
        
        #num_samples = number of SDE samples to draw at each pass
        self.num_samples = 1 #for ODE
        
        u = self.fun_treatment(Tx[..., None])


        # Preparing augmented initial condition
        #aug_y0 = torch.cat([z0, torch.zeros(self.num_samples, batch_size, 1).to(z0), Tx.repeat(self.num_samples, 1)[..., None].to(z0)], dim=-1)
        aug_y0 = torch.cat([z0, u.to(z0.device)], dim=-1)
        #print('aug_y0', aug_y0.shape)
        dim_aug = aug_y0.shape[-1]
        #print(f"Augmented initial condition prepared with dimension: {dim_aug}")


        #need to convert to float32 to run on Mac MPS GPU (my local machine)
        options = {'dtype': torch.float32} 
        # Solve the ODE with bosh3 and not very detailed rtol and atol as computationally expensive
        aug_ys = odeint(self.ode_func, 
                        aug_y0.float(), 
                        t.float(), 
                        method = 'bosh3', 
                        rtol=1e-2, 
                        atol = 1e-3,  
                        options=options)

        ##print('zs', zs.shape)  #[TS, BS, LD] instead of [BS, TS, LD], where BS is the batch size, TS the seq len and LD the latent dim
        aug_ys = aug_ys.permute(1, 0, 2)
        #print(f"Reshaped ODE integration output: {aug_ys.shape}")


        #print("Exiting FORWARD SDE function.")
        return aug_ys

        

'''