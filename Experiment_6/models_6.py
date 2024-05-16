import os
import argparse
import numpy as np
from collections import namedtuple
from typing import Optional, Union
import math

import wandb

import torch
from torch import Tensor
from torch import nn
from torch.nn  import functional as F 
from torch.autograd import Variable
from torch import float32
from torch.utils.data import Dataset, DataLoader, Subset
from torch import distributions, nn, optim

import lightning as L
from lightning import LightningModule

from torchdiffeq import odeint_adjoint as odeint
import torchsde




from utils_6 import _stable_division, GaussianNLLLoss, LinearScheduler, interpolate_colors, CV_params, print_model_details
from plotting_6 import plot_trajectories

use_cuda = torch.cuda.is_available()



class MLPSimple(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, depth, activations=None, dropout_p=None):
        super().__init__()
        print(f"Initializing MLPSimple: input_dim={input_dim}, output_dim={output_dim}, hidden_dim={hidden_dim}, depth={depth}")
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
        #print(f"Forward pass started with input shape: {x.shape}")
        x = self.input_layer(x)
        #print(f"Post input layer shape: {x.shape}")
        
        for i, mod in enumerate(self.layers):
            x_old_shape = x.shape
            x = mod(x)
            #print(f"Layer {i+1}: input shape: {x_old_shape}, output shape: {x.shape}")
        
        x = self.output_layer(x)
        #print(f"Post output layer shape: {x.shape}")
        return x



class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, variational,encode_with_time_dim, encoder_num_layers, encoder_model = 'GRU', reverse=False):
        super(Encoder, self).__init__()
        self.input_dim = input_dim  # obs dim + tx dim
        self.hidden_dim = hidden_dim   
        self.latent_dim = latent_dim  # latent_dim depends on the latent model
        self.encoder_model = encoder_model 
        self.variational = variational
        self.reverse = reverse
        self.encode_with_time_dim = encode_with_time_dim
        self.encoder_num_layers = encoder_num_layers
        
        if encoder_model == 'LSTM':
            self.rnn = nn.LSTM(input_dim + 1 if encode_with_time_dim else input_dim, hidden_dim, num_layers = encoder_num_layers, batch_first=True)
        elif encoder_model == 'GRU':
            self.rnn = nn.GRU(input_dim + 1 if encode_with_time_dim else input_dim, hidden_dim, num_layers = encoder_num_layers, batch_first=True)

        if variational:
            self.hid2lat = nn.Linear(hidden_dim, 2*latent_dim)
        else:
            self.hid2lat = nn.Linear(hidden_dim, latent_dim)

        

    def forward(self, x, t):
        print('Initial x shape:', x.shape)  # Expected: [batch_size, seq_length, input_dim]
        print('Initial t shape:', t.shape)  # Expected: [batch_size, seq_length, 1]

        if self.encode_with_time_dim: # this is how VDS does it 
            # Calculate the time differences
            t_diff = torch.zeros_like(t)
            t_diff[:, 1:] = t[:, 1:] - t[:, :-1]  # Forward differences
            t_diff[:, 0] = 0.
            t_diff = t_diff.unsqueeze(-1) 
            print('Time differences shape:', t_diff.shape)  # Should match t's shape

            xt = torch.cat((x, t_diff), dim=-1)  # Concatenate along the feature dimension
            print('Concatenated xt shape:', xt.shape)  # Expected: [batch_size, seq_length, input_dim + 1]
        
        else: # this is how Hyland does it 
            xt = x

        # Reverse the sequence along the time dimension
        if self.reverse:
            xt = xt.flip(dims=[1])
            print('reversed xt shape:', xt.shape)  # Should match xt's shape

        _, h0 = self.rnn(xt)
        print('Output hidden state h0 shape:', h0.shape)  # Expected: [depth, batch_size, hidden_dim]
        print('output_last_dim', h0[-1].shape)
        
        # Process the last hidden state to produce latent variables
        z0 = self.hid2lat(h0)[-1]

        # Split the output into mean and log-variance components
        if self.variational:
            
            z0_mean = z0[:, :self.latent_dim]
            z0_log_var = z0[:, self.latent_dim:]
            print('z0_mean shape:', z0_mean.shape)  # Expected: [batch_size, latent_dim]
            print('z0_log_var shape:', z0_log_var.shape)  # Expected: [batch_size, latent_dim]
            
            return z0_mean, z0_log_var
        
        else:
            z0_mean = z0[:, :]
            print('z0_mean shape:', z0_mean.shape)  # Expected: [batch_size, latent_dim]

            return z0_mean



class LatentSDE(torchsde.SDEIto):
    def __init__(self,latent_dim, hidden_dim, Tx_dim, theta, mu, sigma):

        super().__init__(noise_type="diagonal")
    
        self.theta = theta
        self.mu = mu
        self.register_buffer("sigma", torch.tensor([[sigma]]))
        
        u_dim = int(latent_dim/5)
        print('Initialising Treatment function')
        self.treatment_fun = MLPSimple(input_dim = 1, output_dim = u_dim, hidden_dim = 20, depth = 4, activations = [nn.ReLU() for _ in range(4)] )
        print('Initialising SDE_drift function')
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
        print('u', u.shape)
        f_logqp = .5 * (u ** 2).sum(dim=1, keepdim=True)
        print('f_logqp', f_logqp.shape)
        return torch.cat([f, f_logqp, torch.zeros_like(f_logqp)], dim=1)
    
    def g_aug(self, t, y):  # Diffusion for augmented dynamics with logqp term.
        y = y[:, 0:-2]
        g = self.g(t, y)
        g_logqp = torch.zeros((y.shape[0],2), device = y.device)
        return torch.cat([g, g_logqp], dim=1)



class CV_expert_ODE(nn.Module):
    def __init__(self, params):
        super(CV_expert_ODE, self).__init__()
        # Store parameters
        for key, value in params.items():
            setattr(self, key, nn.Parameter(torch.tensor(value, dtype=torch.float32), requires_grad=True))

    def fluids_input(self, t):   # => This AGAIN assumes that the treatment is given at a specific point in time and then goes down... 
        # but obviously fluid is not given AT a point in time for FOR a DURATION of time 
        return 5 * torch.exp(-((t-5)/5)**2)

    def v_fun(self, x): # => WHY v_fun!??? it's a WEIRD function 
        return 0.02 * (torch.cos(5*x-0.2) * (5-x)**2)**2

    def sigmoid(self, x):
        return 1 / (1 + torch.exp(-x))
    
    def scale_encoder_state_aug(self, state_aug):
        p_a, p_v, s_reflex , sv = state_aug[:, 0], state_aug[:, 1], state_aug[:, 2], state_aug[:, 3]

        p_a = self.sigmoid(p_a)*(self.max_pa - self.min_pa) + self.min_pa
        p_v = self.sigmoid(p_v)*(self.max_pv - self.min_pv) + self.min_pv
        s_reflex = self.sigmoid(s_reflex)*(self.max_s - self.min_s) + self.min_s
        sv = self.sigmoid(sv)*(self.max_sv - self.min_sv) + self.min_sv

        return p_a, p_v, s_reflex, sv


    def forward(self, t, state_aug):
       
        # Unpack state variables
        p_a, p_v, s_reflex, sv = state_aug[:, 0], state_aug[:, 1], state_aug[:, 2], state_aug[:, 3]
        print('pa',p_a, 'pv', p_v, 's_reflex', s_reflex, 'sv', sv)

        p_a, p_v, s_reflex, sv  = self.scale_encoder_state_aug(state_aug)
        print('pa',p_a, 'pv', p_v, 's_reflex', s_reflex, 'sv', sv)

        Tx = state_aug[:, 4][:,None]
        t_expanded = t.expand(state_aug.shape[0])
        print('t_expanded', t_expanded.shape)
        print('Tx', Tx.shape)
        print('state', state_aug.shape)
        Tx_dim = Tx.shape[1]
        
        
        #p_a = 0.5 + (p_a - 0.75) / 0.1
        #A_ = self.v_fun(p_a)
        i_ext =  self.fluids_input(t_expanded) # * A_ 
        i_ext = i_ext[:, None]  
        i_ext = i_ext * Tx  #Adjust the I_ext by the DOSE of the treatment 

        print('i_ext', i_ext)

        # Calculate f_hr and r_tpr
        f_hr = s_reflex * (self.f_hr_max - self.f_hr_min) + self.f_hr_min
        r_tpr = s_reflex * (self.r_tpr_max - self.r_tpr_min) + self.r_tpr_min - self.r_tpr_mod
        
        # Calculate changes in volumes and pressures
        dva_dt = -1. * (p_a - p_v) / r_tpr + sv * f_hr
        dvv_dt = -1. * dva_dt +  i_ext.squeeze()

        print('f_hr', f_hr)
        print('dva_dt', dva_dt)
        print('r_tpr', r_tpr)
        print('dvv_dt', dvv_dt)

        # Calculate derivative of state variables

        dpa_dt = dva_dt / (self.ca * 100.)
        dpv_dt = dvv_dt / (self.cv * 10.)
        ds_dt = (1. / self.tau) * (1. - self.sigmoid(self.k_width * (p_a - self.p_aset)) - s_reflex)
        print('i_ext.squeeze()', i_ext.squeeze().shape)
        print('self.sv_mod', self.sv_mod.shape)
        dsv_dt = i_ext * self.sv_mod

        print('dpa_dt', dpa_dt)
        print('dpv_dt', dpv_dt)
        print('ds_dt', ds_dt)
        print('dsv_dt', dsv_dt)

        diff_res = torch.cat([dpa_dt[:, None], dpv_dt[:, None], ds_dt[:, None], dsv_dt], dim=-1)


        null_dim = torch.zeros(diff_res.shape[0], Tx.shape[1], device=diff_res.device)
        diff_res_aug = torch.cat([diff_res, null_dim], dim=-1)
        print('diff_res_aug', diff_res_aug.shape)

        return diff_res_aug

    


class Hybrid_LatentSDE(torchsde.SDEIto):

    def __init__(self, expert_dims, SDE_latents_dim, CV_params, mu=None, sigma=None, theta=1.0 ):
        super(Hybrid_LatentSDE, self).__init__(noise_type="diagonal")
        #self.sde_type="ito"

        self.SDE_latents_dim = SDE_latents_dim
        self.expert_dims = expert_dims
        self.resid_dims = SDE_latents_dim - expert_dims
        
        if mu is None:
            mu = np.zeros(SDE_latents_dim)
        if sigma is None:
            sigma = np.ones(SDE_latents_dim) * 0.5
        
        logvar = np.log(sigma ** 2 / (2. * theta))

        # Prior drift.
        self.register_buffer("theta", torch.tensor(theta).view(1, -1).expand(1, SDE_latents_dim).float())
        self.register_buffer("mu", torch.tensor(mu).view(1, -1).float())
        self.register_buffer("sigma", torch.tensor(sigma).view(1, -1).float())

        #print('theta init', self.theta.shape)
        #print('mu init ', self.mu.shape)
        #print('sigma init', self.sigma.shape)


        # p(y0).
        self.register_buffer("py0_mean", torch.tensor(mu).view(1, -1).float())
        self.register_buffer("py0_logvar", torch.tensor(logvar).view(1, -1).float())

        # q(y0).
        self.qy0_mean = nn.Parameter(torch.tensor(mu).view(1, -1).float(), requires_grad=True)
        self.qy0_logvar = nn.Parameter(torch.tensor(logvar).view(1, -1).float(), requires_grad=True)

        # Approximate posterior drift: Takes in 2 positional encodings and the state.
        self.net = nn.Sequential(
            nn.Linear(self.resid_dims, 200),
            nn.Tanh(),
            nn.Linear(200, 200),
            nn.Tanh(),
            nn.Linear(200, self.resid_dims)
        )
        # Initialization trick from Glow.
        self.net[-1].weight.data.fill_(0.)
        self.net[-1].bias.data.fill_(0.)
        
       

        for key, value in CV_params.items():
            setattr(self, key, nn.Parameter(torch.tensor(value, dtype=torch.float32), requires_grad=True))
    
    def sigmoid(self, x):
        return 1 / (1 + torch.exp(-x))
    
    def scale_init_expert(self, expert_var):
        init_pa = expert_var[:, 0] 
        init_p_v = expert_var[:,1] 
        p_a = (100.0 * init_pa - self.min_pa) / (self.max_pa - self.min_pa)
        p_v = (10.0 * init_p_v - self.min_pa) / (self.max_pa - self.min_pa)
        
        return(p_a, p_v)
    
    def reverse_scale_init_expert(self, scaled_vars):
        scaled_p_a = scaled_vars[:, :, 0]
        scaled_p_v = scaled_vars[:, :, 1]
        # Undo the normalization and scaling for p_a
        normalised_pa = ((scaled_p_a * (self.max_pa - self.min_pa)) + self.min_pa) / 100.0

        # Undo the normalization and scaling for p_v
        normalised_pv = ((scaled_p_v * (self.max_pa - self.min_pa)) + self.min_pa) / 10.0
        return torch.stack([normalised_pa, normalised_pv, scaled_vars[:, :, 2], scaled_vars[:, :, 3]], dim=-1)

    def f(self, t, y):  # Approximate posterior drift.
        if t.dim() == 0:
            t = torch.full_like(y, fill_value=t)
        # Positional encoding in transformers for time-inhomogeneous posterior.
        
        p_a, p_v = self.scale_init_expert(y[:,:2])
        sde_latent_params = y[:, 2:] #the last two of the y are for the SDE
        s_reflex_sde = sde_latent_params[:, 0]
        sv_sde = sde_latent_params[:, 0]

        #convert all to 'num samples x 1'
        s_reflex_sde = s_reflex_sde.unsqueeze(1)
        sv_sde = sv_sde.unsqueeze(1)
        p_a = p_a.unsqueeze(1)
        p_v = p_v.unsqueeze(1)
        SDE_samples = p_v.shape[0]

        #print('s_reflex_sde', s_reflex_sde.shape)
        #print('sv_sde', sv_sde.shape)
        #print('p_a', p_a.shape)
        #print('p_v', p_v.shape)


        i_ext = torch.zeros(SDE_samples, 1, device=y.device) 
        f_hr = s_reflex_sde * (self.f_hr_max - self.f_hr_min) + self.f_hr_min
        r_tpr = s_reflex_sde * (self.r_tpr_max - self.r_tpr_min) + self.r_tpr_min - self.r_tpr_mod
        
        # Calculate changes in volumes and pressures
        dva_dt = -1. * (p_a - p_v) / r_tpr + sv_sde * f_hr
        dvv_dt = -1. * dva_dt + i_ext


        #print('f_hr', f_hr.shape)
        #print('dva_dt', dva_dt.shape)
        #print('r_tpr', r_tpr.shape)
        #print('dvv_dt', dvv_dt.shape)

        # Calculate derivative of state variables

        dpa_dt = dva_dt / (self.ca * 100.)
        dpv_dt = dvv_dt / (self.cv * 10.)
        ds_dt = (1. / self.tau) * (1. - self.sigmoid(self.k_width * (p_a - self.p_aset)) - s_reflex_sde)
        dsv_dt = i_ext * self.sv_mod
    
        #print('dpa_dt', dpa_dt.shape)
        #print('dpv_dt', dpv_dt.shape)
        #print('ds_dt', ds_dt.shape)
        #print('dsv_dt', dsv_dt.shape)

        #sde_latent_times = t[:, :1]  #the time is shared across all 
        #SDE_latents = self.net(torch.cat((torch.sin(sde_latent_times), torch.cos(sde_latent_times), sde_latent_params), dim=-1))
        SDE_latents = self.net(sde_latent_params) 
        #print('SDE_drift_fun_output', SDE_latents.shape)
        Dt_s_reflex_sde = SDE_latents[:, 0].unsqueeze(1)
        Dt_sv_sde = SDE_latents[:, 1].unsqueeze(1)

        #print('Dt_s_reflex_sde', Dt_s_reflex_sde.shape)
        #print('Dt_sv_sde', Dt_sv_sde.shape)


        ds_dt = ds_dt + Dt_s_reflex_sde
        dsv_dt = dsv_dt + Dt_sv_sde

        #print('ds_dt', ds_dt.shape)
        #print('dsv_dt', dsv_dt.shape)


        diff_results = torch.cat([dpa_dt, dpv_dt, ds_dt, dsv_dt], dim=-1)
        #print('diff_results', diff_results.shape)

        return diff_results 

    def g(self, t, y):  # Shared diffusion.
        expanded_sigma = self.sigma.expand(y.size(0), -1)
        #print('sigma g', expanded_sigma.shape)
        return expanded_sigma

    def h(self, t, y):  # Prior drift.
        #print('theta h', self.theta.shape)
        #print('mu h', self.mu.shape)
        #print('y in h', y.shape)
        return self.theta * (self.mu - y)

    def f_aug(self, t, y):  # Drift for augmented dynamics with logqp term.
        y = y[:, :self.SDE_latents_dim]

        #print('Y f_aug', y.shape) # num_samples x sde_dims 

        f, g, h = self.f(t, y), self.g(t, y), self.h(t, y)
        #print('doing stable division!')
        #print('f', f.shape, 'g', g.shape, 'h', h.shape)
        u = _stable_division(f - h, g)
        #print('u', u.shape)
        f_logqp = .5 * (u ** 2).sum(dim=1, keepdim=True)
        #print('f_logqp', f_logqp.shape)
        f_out = torch.cat([f, f_logqp], dim=1)
        #print('f_aug out', f_out.shape)
        return f_out

    def g_aug(self, t, y):  # Diffusion for augmented dynamics with logqp term.
        y = y[:, :self.SDE_latents_dim]
        g = self.g(t, y)
        #print('g', g.shape)
        g_logqp = torch.zeros(y.size(0), 1).to(y.device)
        g_out = torch.cat([g, g_logqp], dim=1)
        #print('g out', g_out.shape)
        return g_out




class Neural_Expert_SDE_integrator(nn.Module):
    def __init__(self, latent_dim, hidden_dim, num_samples, Tx_dim, theta, mu, sigma, expert_ODE_size):
        super(Neural_Expert_SDE_integrator, self).__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.num_samples = num_samples
        self.Tx_dim = Tx_dim
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.expert_ODE_size = expert_ODE_size
                        #theta,mu, sigma, embedding_dim
        self.sde_func = LatentSDE(latent_dim= latent_dim, 
                                      hidden_dim=hidden_dim, 
                                      Tx_dim= Tx_dim,
                                      theta = theta, 
                                      mu = mu, 
                                      sigma = sigma)
        self.CV_expert_ODE = CV_expert_ODE(CV_params)

    def forward_expert(self, z_expt, t, Tx):
        aug_init_state = torch.cat([z_expt, Tx.unsqueeze(1).to(z_expt)], dim=-1)
        
        print('aug_init_state.float()', aug_init_state.float())

        options = {'solver': 'LSODA'} 
        predicted_latents = odeint(self.CV_expert_ODE, 
                    aug_init_state.float(), 
                    t.float(), 
                    method = 'scipy_solver', 
                    rtol=1e-6, 
                    atol = 1e-9,  
                    options=options)
        
        predicted_latents = predicted_latents.permute(1, 0, 2) #now latent traj should be [batch, seq len, dim]
        predicted_latents = predicted_latents[:, :, :-self.Tx_dim]
        print('predicted_latents_Expert', predicted_latents.shape)

        return predicted_latents

    def forward_SDE(self,z_SDE, t, Tx):
        batch_size = z_SDE.shape[0]
        # let's convert now to batch x num_samples x 1
        Tx = Tx.unsqueeze(1).unsqueeze(2)
        Tx = Tx.repeat(1, self.num_samples, 1)  
        print('Aug Tx', Tx.shape)

        # For each SDE sample, adding Tx as part of the initial state + an empty dim for the LogQ for that SDE sample
        aug_y0 = torch.cat([z_SDE, torch.zeros(batch_size,self.num_samples, 1).to(z_SDE), Tx.to(z_SDE)], dim=-1) 
        print('aug_y0', aug_y0.shape)
        dim_aug = aug_y0.shape[-1]
        aug_y0 = aug_y0.reshape(-1,dim_aug)

        dim_change = dim_aug - z_SDE.shape[-1]
        print('augmented_init_state', aug_y0.shape)


        #options = {'dtype': torch.float32} 

        aug_ys = torchsde.sdeint(sde=self.sde_func,
                y0=aug_y0,
                ts=t,
                method="euler",
                dt=0.05,
                adaptive=False,
                rtol=1e-3,
                atol=1e-3,
                names={'drift': "f_aug", 'diffusion': 'g_aug'})
                #options = options)
                
        aug_ys = aug_ys.reshape(-1, self.num_samples, len(t),dim_aug) # reshape for # batch_size x num_samples x times x dim

        print('aug_ys', aug_ys.shape)

        #effectively get rid of the Tx dim by not including it (it's the last one)
        ys, logqp_path = aug_ys[:, :, :, :-2], aug_ys[:,: , -1, -2]  
        print('ys_extracted', ys.shape)
        print('logqp_path_extracted', logqp_path.shape)
        logqp0 = 0
        logqp = (logqp0 + logqp_path)  # KL(t=0) + KL(path).

        return ys, logqp


    def forward(self, z_expt, z_SDE, t, Tx, latent_type):
        
        t = t[0] if t.ndim > 1 else t
        t = t.flatten()  
        print('time dim for ODE', t.shape) #= [time]
        print('z_expt',z_expt.shape if z_expt != None else None)
        print('z_SDE', z_SDE.shape if z_SDE != None else None)
        print('Tx', Tx.shape) # shape = batch


        if latent_type == 'expert':
            assert z_expt.shape[1] == self.expert_ODE_size
            predicted_latent_traj =  self.forward_expert( z_expt, t, Tx)
            return predicted_latent_traj
        
        elif latent_type == 'SDE':
            assert z_SDE.shape[2] == self.latent_dim
            predicted_latent_traj, logqp = self.forward_SDE(z_SDE, t, Tx)
            return predicted_latent_traj, logqp
        
        elif latent_type == 'hybrid_SDE':
            assert z_expt.shape[1] + z_SDE.shape[2] == self.expert_ODE_size + self.latent_dim

            predicted_latent_traj_expert =  self.forward_expert( z_expt, t, Tx)
            predicted_latent_traj_SDE, logqp_SDE = self.forward_SDE(z_SDE, t, Tx)

            return predicted_latent_traj_expert, predicted_latent_traj_SDE, logqp_SDE


    




class SDE_VAE(nn.Module):
    def __init__(self, 
                 input_dim, output_dim, hidden_dim, latent_dim, post_tx_ode_len, Tx_dim,
                 encoder_model, latent_type, expert_ODE_size,
                 num_samples, theta, mu, sigma, 
                 use_whole_trajectory, output_then_join, output_fun_depth =1,  dropout_p= 0.2):
        
        super(SDE_VAE, self).__init__()
        self.input_dim = input_dim #dim of input in observed space 
        self.output_dim = output_dim #dim of the output in the observed space
        self.latent_dim = latent_dim #dim of the latent space  
        self.hidden_dim = hidden_dim #dim of the hidden layers in NNs

        self.dim_difference = latent_dim - expert_ODE_size
        
        self.encoder_model = encoder_model
        self.latent_type = latent_type

        self.num_samples = num_samples #number of latent SDE samples
        self.post_tx_ode_len = post_tx_ode_len
        
        self.use_whole_trajectory = use_whole_trajectory #whether the output fun is applied pointwise to the latent ODE or takes in the whole traj and then converts to observed
        self.output_then_join = output_then_join
        self.output_fun_depth = output_fun_depth


        print('Initialising Encoder variational')
        self.encoder_Var = Encoder(input_dim = input_dim, 
                               hidden_dim = hidden_dim, 
                               latent_dim = expert_ODE_size, 
                               variational = True,
                               encoder_model = encoder_model, 
                               encode_with_time_dim = True, ### Should this be true or false?
                               encoder_num_layers = 2, ## Set this to be flexible??
                               reverse=False)
        print('Initialising Encoder non-variational')
        self.encoder_NonVar = Encoder(input_dim = input_dim, 
                               hidden_dim = hidden_dim, 
                               latent_dim = latent_dim, 
                               variational = False,
                               encoder_model = encoder_model, 
                               encode_with_time_dim = False, 
                               encoder_num_layers = 2, ## Set this to be flexible??
                               reverse=False)
        print('Initialising Neural_Expert_SDE_integrator')
        self.latent_model = Neural_Expert_SDE_integrator(latent_dim = latent_dim, 
                                                   hidden_dim = hidden_dim, 
                                                   Tx_dim= Tx_dim,
                                                   num_samples = num_samples, 
                                                   theta= theta, 
                                                   mu=mu, 
                                                   sigma = sigma, 
                                                   expert_ODE_size = expert_ODE_size) 
                
        
        if latent_type == 'hybrid_SDE':
            output_fun_input_dim = expert_ODE_size + self.dim_difference
        else:
            output_fun_input_dim = latent_dim
        output_fun_input_dim_expert = expert_ODE_size
        output_fun_input_dim_SDE = latent_dim

        print('Initialising Output Function')

        self.output_fun_expert = MLPSimple(input_dim=output_fun_input_dim_expert * (post_tx_ode_len if self.use_whole_trajectory else 1),
                output_dim=self.output_dim * (post_tx_ode_len if self.use_whole_trajectory else 1),
                hidden_dim=output_fun_input_dim_expert,
                depth=output_fun_depth,
                activations=[nn.ReLU() for _ in range(output_fun_depth - 1)] + [nn.Tanh()], 
                dropout_p=[dropout_p for _ in range(output_fun_depth)])
        
        self.output_fun_SDE = MLPSimple(input_dim=output_fun_input_dim_SDE * (post_tx_ode_len * num_samples if self.use_whole_trajectory else 1),
                output_dim=self.output_dim * (post_tx_ode_len * num_samples if self.use_whole_trajectory else 1),
                hidden_dim=output_fun_input_dim_SDE,
                depth=output_fun_depth,
                activations=[nn.ReLU() for _ in range(output_fun_depth - 1)] + [nn.Tanh()], 
                dropout_p=[dropout_p for _ in range(output_fun_depth)])
        
        self.output_fun = MLPSimple(input_dim=output_fun_input_dim * (post_tx_ode_len if self.use_whole_trajectory else 1),
                        output_dim=self.output_dim * (post_tx_ode_len if self.use_whole_trajectory else 1),
                        hidden_dim=output_fun_input_dim,
                        depth=output_fun_depth,
                        activations=[nn.ReLU() for _ in range(output_fun_depth - 1)] + [nn.Tanh()], 
                        dropout_p=[dropout_p for _ in range(output_fun_depth)])

        
    def rescale_down_expert_ode(self,predictions):


        pass


    def forward_VAE_expert(self, x, Tx, time_in, time_out,  MAP):
        
        z_mean, z_log_var_exp = self.encoder_Var(x, time_in)
        z_expt = z_mean if MAP else z_mean + torch.randn_like(z_mean) * torch.exp(0.5 * z_log_var_exp) #if MAP then we do NOT sample, so effectively it's not longer variational, otherwise we sample

        z_SDE = None
        latent_traj_expt = self.latent_model(z_expt, z_SDE, time_out, Tx, self.latent_type)
        print('z_mean',z_expt.shape if z_expt != None else None)
        print('z_expt',z_expt.shape if z_expt != None else None)
        print('z_SDE', z_SDE.shape if z_expt!= None else None)

        

        return z_expt, z_mean,  z_log_var_exp, latent_traj_expt

    def forward_VAE_hybrid(self, x, Tx, time_in, time_out,  MAP):

        z_mean, z_log_var_exp = self.encoder_Var(x, time_in)
        z_expt = z_mean if MAP else z_mean + torch.randn_like(z_mean) * torch.exp(0.5 * z_log_var_exp) #if MAP then we do NOT sample, so effectively it's not longer variational, otherwise we sample
        z_SDE = self.encoder_NonVar(x, time_in)
        z_SDE = z_SDE.unsqueeze(1)  # Add an extra dimension: shape becomes [batch, 1, latent]
        z_SDE = z_SDE.repeat(1, self.num_samples, 1) #now repeat by num_samples

        print('z_mean',z_expt.shape if z_expt != None else None)
        print('z_expt',z_expt.shape if z_expt != None else None)
        print('z_SDE', z_SDE.shape if z_expt!= None else None)

        latent_traj_expt, latent_traj_SDE, logqp_SDE = self.latent_model(z_expt, z_SDE, time_out, Tx, self.latent_type)

        return z_expt, z_mean, z_log_var_exp, latent_traj_expt, z_SDE, latent_traj_SDE, logqp_SDE

    def forward_VAE_SDE(self, x, Tx, time_in, time_out,  MAP):

        z_SDE = self.encoder_NonVar(x, time_in)

        # z shape starts as batch x latent, and I want to convert it to batch x num_samples x latent 
        z_SDE = z_SDE.unsqueeze(1)  # Add an extra dimension: shape becomes [batch, 1, latent]
        z_SDE = z_SDE.repeat(1, self.num_samples, 1) #now repeat by num_samples 

        z_expt = None
        # Generating latent dynamics using the SDE
        print('z_mean',z_expt.shape if z_expt != None else None)
        print('z_expt',z_expt.shape if z_expt != None else None)
        print('z_SDE', z_SDE.shape if z_expt!= None else None)

        latent_traj_SDE, logqp_SDE = self.latent_model(z_expt, z_SDE, time_out, Tx, self.latent_type)
        print('latentSDEoutput', latent_traj_SDE.shape) #batch_size x num_samples x times x latent_dim

        return z_SDE, latent_traj_SDE, logqp_SDE
    
    def output_fun_flexy(self, output_function_any, latent_traj, use_whole_trajectory):

        if use_whole_trajectory:
            batch_size, num_samples, seq_len, dim = latent_traj.shape
            latent_traj_flat = latent_traj.reshape(batch_size, -1)
            pred_traj_flat = output_function_any(latent_traj_flat)  # Process the whole trajectory
            predicted_traj = pred_traj_flat.reshape(batch_size, num_samples, seq_len, self.output_dim)

        else:
            predicted_traj = output_function_any(latent_traj)

        return predicted_traj # here Hyland drops the frist time step 


    def forward(self, x, Tx, time_in, time_out,  MAP=True):
        # takes in:
        # x:  observed trajectory until treatment time 
        # time_in: time from start to treatment time 
        # time_out: time form treatment time to finish
        # Tx: treatment presence: binary vector with 1 = treated, 0 = untreated
        

        #it would be good to change this so you can be flexible and have the SDE also with a variational encoder!
        if self.latent_type == 'expert':
            z_expt, z_expt_mean, z_log_var_exp, latent_traj_expt = self.forward_VAE_expert(x, Tx, time_in, time_out,  MAP)
            latent_traj_expt = latent_traj_expt.unsqueeze(1)
            print('latent_traj_expt',latent_traj_expt.shape)
            z_SDE, latent_traj_SDE, logqp_SDE, predicted_traj_SDE, predicted_traj_hybrid = (None, None, None, None, None)
            predicted_traj_exp = self.output_fun_flexy(self.output_fun_expert, latent_traj_expt, self.use_whole_trajectory)

        elif self.latent_type == 'SDE':
            z_SDE, latent_traj_SDE, logqp_SDE = self.forward_VAE_SDE(x, Tx, time_in, time_out,  MAP)
            print('latent_traj_SDE', latent_traj_SDE.shape)
            z_expt,z_expt_mean,  z_log_var_exp, latent_traj_expt, predicted_traj_exp,  predicted_traj_hybrid= (None, None, None, None, None, None)
            predicted_traj_SDE = self.output_fun_flexy(self.output_fun_SDE, latent_traj_SDE, self.use_whole_trajectory) 


        elif self.latent_type == 'hybrid_SDE':
            z_expt, z_expt_mean,  z_log_var_exp, latent_traj_expt, z_SDE, latent_traj_SDE, logqp_SDE = self.forward_VAE_hybrid(x, Tx, time_in, time_out,  MAP)
            latent_traj_expt = latent_traj_expt.unsqueeze(1)
            print('latent_traj_expt', latent_traj_expt.shape)
            print('latent_traj_SDE', latent_traj_SDE.shape)

            if self.output_then_join: #convert to output then join 
                predicted_traj_exp = self.output_fun_flexy(self.output_fun_expert, latent_traj_expt, self.use_whole_trajectory)  
                predicted_traj_SDE = self.output_fun_flexy(self.output_fun_SDE, latent_traj_SDE, self.use_whole_trajectory)  
                predicted_traj_hybrid = torch.cat([predicted_traj_exp, predicted_traj_SDE], dim=-1)

            else: #join the latents then output to observed dim 
                predicted_traj_exp = self.output_fun_flexy(self.output_fun_expert, latent_traj_expt, self.use_whole_trajectory)  
                predicted_traj_SDE = self.output_fun_flexy(self.output_fun_SDE, latent_traj_SDE, self.use_whole_trajectory) 

                latent_traj_expt_padded = F.pad(latent_traj_expt, (0, self.dim_difference))
                print('latent_traj_expt_padded', latent_traj_expt_padded.shape)
                latent_traj_hybrid = torch.cat([latent_traj_expt_padded,latent_traj_SDE], dim=1 ) #join at the num_samples dim 
                print('latent_traj_hybrid', latent_traj_hybrid.shape)
                predicted_traj_hybrid = self.output_fun_flexy(self.output_fun,latent_traj_hybrid, self.use_whole_trajectory)


        print('predicted_traj_hybrid',predicted_traj_hybrid.shape if predicted_traj_hybrid != None else None)
        predicted_traj_exp = predicted_traj_exp.contiguous() if predicted_traj_exp != None else None
        predicted_traj_SDE = predicted_traj_SDE.contiguous() if predicted_traj_SDE != None else None
        predicted_traj_hybrid = predicted_traj_hybrid.contiguous() if predicted_traj_hybrid != None else None

        return z_expt, z_expt_mean, z_log_var_exp, latent_traj_expt, predicted_traj_exp, z_SDE, latent_traj_SDE, logqp_SDE, predicted_traj_SDE, predicted_traj_hybrid
        

    



class SDE_VAE_Lightning(LightningModule):
    def __init__(self, 
                # Experiment vars
                encoder_model, latent_type, expert_ODE_size, 
                #models vars
                input_dim, output_dim, hidden_dim, latent_dim, Tx_dim, dropout_p,
                #SDE vars
                num_samples, theta, mu, sigma, output_scale, 
                #output function vars 
                use_whole_trajectory, post_tx_ode_len, output_then_join, 
                # loss vars
                KL_weighting_SDE, KL_weighting_var_encoder, mse_weighting,  learning_rate, log_wandb, start_scheduler = 200, iter_scheduler = 600,):
        super().__init__()
        
        self.VAE_model = SDE_VAE(#experiment vars 
                                encoder_model = encoder_model,
                                latent_type = latent_type, 
                                expert_ODE_size = expert_ODE_size, 
                                #models vars
                                input_dim= input_dim,
                                output_dim = output_dim, 
                                hidden_dim = hidden_dim, 
                                latent_dim = latent_dim,
                                Tx_dim = Tx_dim,
                                dropout_p= dropout_p,
                                #SDE vars
                                num_samples = num_samples,
                                theta = theta, 
                                mu=mu, 
                                sigma = sigma, 
                                #output function vars 
                                use_whole_trajectory = use_whole_trajectory, 
                                post_tx_ode_len = post_tx_ode_len,
                                output_then_join = output_then_join
                                )
    
        self.loss = GaussianNLLLoss(reduction = "none")
        self.MSE_loss = nn.MSELoss(reduction = "none")
        self.kl_scheduler = LinearScheduler(start = start_scheduler, iters = iter_scheduler)
        self.KL_weighting_SDE = KL_weighting_SDE
        self.KL_weighting_var_encoder = KL_weighting_var_encoder
        self.mse_weighting = mse_weighting

        self.latent_type = latent_type
        self.expert_ODE_size = expert_ODE_size
        self.num_samples = num_samples
        self.output_scale = torch.tensor([output_scale], requires_grad = False, device = self.device)
        self.post_tx_ode_len = post_tx_ode_len
        
        self.learning_rate = learning_rate
        self.log_wandb = log_wandb
        self.save_hyperparameters()

    def on_fit_start(self):
        print_model_details(self.VAE_model)


    def forward(self, x, t, MAP=False):
        return self.VAE_model(x, t, MAP)
    

    def SDE_KL_loss(self, Y, Y_hat_SDE, logqp_SDE):

        # Convert Y_true to match Y_hat_SDE shape: batch x num_samples x times x dims, by repeating for each num_sample 
        # so that each sample can be compared to the ground truth.
        Y_expanded = Y[:, :self.post_tx_ode_len, :].unsqueeze(1)
        Y_true_SDE_shape = Y_expanded.repeat(1, self.num_samples, 1, 1)  
        print('Y_true after repeat:', Y_true_SDE_shape.shape)


        ## Apply the negative gaussian log likelihood loss between the enhanced true trajectory and the predicted SDE trajectories,
        ## with a Standard dev preset by output_scale (why preset?)
        fact_loss = self.loss(Y_true_SDE_shape, Y_hat_SDE, self.output_scale.repeat(Y_hat_SDE.shape).to(self.device))
        fact_loss = fact_loss.sum((2, 3))  # sum across times and dims (keeping for each batch and SDE sample)
        print('fact_loss after sum:', fact_loss.shape)

        # Now find the total loss: the average gaussian log likelihood across SDE samples for the batch + mean logQP across samples & batch
        SDE_loss = fact_loss.mean() + self.KL_weighting_SDE * logqp_SDE.mean() * self.kl_scheduler.val #val for value

        return SDE_loss, fact_loss.mean(), logqp_SDE.mean()

    def expert_KL_loss(self, z_expt_mean, z_log_var_exp):

        kl_expt_loss = -0.5 * torch.sum(1 + z_log_var_exp - z_expt_mean.pow(2) - z_log_var_exp.exp())

        return  self.KL_weighting_var_encoder* kl_expt_loss
    
    def compute_factual_loss(self, Y, Y_hat, z_expt_mean, z_log_var_exp, logqp_SDE, Y_hat_SDE = None):
        print('Y initial:', Y.shape)  # Shape of ground truth data
        print('Y_hat initial:', Y_hat.shape)  # Shape of predicted data from SDE


        # RECON LOSS
        # MSE loss between the Y and the MEAN of the SDE samples predictions, which includes expert and SDE in hybrid 
        mse_recon_loss = torch.sqrt(self.MSE_loss(Y[:, :self.post_tx_ode_len, :], Y_hat.mean(1))).mean() * self.mse_weighting
        # Now find the mean of the standard devs of the predictions across the SDE samples
        std_preds = Y_hat.std(1).mean()

        
        if self.latent_type == 'expert':
            kl_expt_loss = self.expert_KL_loss(z_expt_mean, z_log_var_exp)
            SDE_loss, fact_loss, logqp_SDE_mean = (0, 0, 0)

        elif self.latent_type == 'SDE':
            SDE_loss, fact_loss, logqp_SDE_mean = self.SDE_KL_loss(Y, Y_hat, logqp_SDE)
            kl_expt_loss = 0
    
        elif self.latent_type == 'hybrid_SDE':
            SDE_loss, fact_loss, logqp_SDE_mean = self.SDE_KL_loss(Y, Y_hat_SDE, logqp_SDE)
            kl_expt_loss = self.expert_KL_loss(z_expt_mean, z_log_var_exp)


        return mse_recon_loss, std_preds, kl_expt_loss, SDE_loss, fact_loss, logqp_SDE_mean

    
    def compute_counterfactual_loss(self, Y, Y_cf, Y_hat, Y_hat_cf):
        print('Y:', Y.shape)
        print('Y_cf:', Y_cf.shape)
        print('Y_hat:', Y_hat.shape)
        print('Y_hat_cf:', Y_hat_cf.shape)

        # RECON LOSS
        # MSE loss between the Y and the MEAN of the SDE samples predictions, which includes expert and SDE in hybrid 
        mse_cf = torch.sqrt(self.MSE_loss(Y_cf[:, :self.post_tx_ode_len, :], Y_hat_cf.mean(1))).mean()
        # Now find the mean of the standard devs of the predictions across the SDE samples
        std_preds_cf = Y_hat_cf.std(1).mean()

        # Individual Treatment Effect computed as the difference between Y_cf and Y
        ite = (Y_cf[:, :self.post_tx_ode_len, :] - Y[:, :self.post_tx_ode_len, :])
        print('ite:', ite.shape)

        # Predicted Individual Treatment Effect computed as the difference between the mean predictions of Y_hat_cf and Y_hat
        ite_hat = (Y_hat_cf.mean(1) - Y_hat.mean(1))
        print('ite_hat:', ite_hat.shape)

        # MSE of the ITE
        mse_ite = torch.sqrt(self.MSE_loss(ite, ite_hat)).mean()
        print('mse_ite:', mse_ite)    


        return mse_cf, mse_ite, std_preds_cf


    def training_step(self, batch, batch_idx):
        print("TRAINING")
        X, Y, T, Y_cf, p, thetas_0, time_X, time_Y = batch
        z_expt, z_expt_mean, z_log_var_exp, latent_traj_expt, predicted_traj_exp, z_SDE, latent_traj_SDE, logqp_SDE, predicted_traj_SDE, predicted_traj_hybrid = self.VAE_model(X, T, time_in = time_X, time_out = time_Y, MAP=False) 
        
        if self.latent_type == 'expert':
            Y_hat = predicted_traj_exp
        elif self.latent_type == 'SDE':
            Y_hat = predicted_traj_SDE
        elif self.latent_type == 'hybrid_SDE':
            Y_hat = predicted_traj_hybrid

        weighted_mse_recon_loss, std_preds, weighted_kl_expt_loss, weighted_SDE_loss, fact_loss, logqp_SDE_mean= self.compute_factual_loss(Y, Y_hat, z_expt_mean, z_log_var_exp, logqp_SDE, Y_hat_SDE = predicted_traj_SDE)
        
        if self.latent_type == 'hybrid_SDE':
            total_loss = weighted_SDE_loss + weighted_kl_expt_loss + weighted_mse_recon_loss
        elif self.latent_type == 'expert':
            total_loss = weighted_kl_expt_loss + weighted_mse_recon_loss
        elif self.latent_type == 'SDE':
            total_loss = weighted_SDE_loss

        self.log('train_total_loss', total_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_SDE_loss', weighted_SDE_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_Fact_loss', fact_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_logqp_SDE_mean', logqp_SDE_mean, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        self.log('train_recon_loss', weighted_mse_recon_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_std_preds', std_preds, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_kl_expt_loss', weighted_kl_expt_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        self.kl_scheduler.step()

        return total_loss

    def validation_step(self, batch, batch_idx):
        print('VALIDATION')
        X, Y, T, Y_cf, p, thetas_0, time_X, time_Y = batch
        print('X', X.shape)
        print('Y', Y.shape)
        
       
        #MAP = true as you don't sample the encoder variational latents during validation step 
        z_expt, z_expt_mean, z_log_var_exp, latent_traj_expt, predicted_traj_exp, z_SDE, latent_traj_SDE, logqp_SDE, predicted_traj_SDE, predicted_traj_hybrid = self.VAE_model(X, T, 
                                                             time_in = time_X, 
                                                             time_out = time_Y, 
                                                             MAP=True)

        if self.latent_type == 'expert':
            Y_hat = predicted_traj_exp
        elif self.latent_type == 'SDE':
            Y_hat = predicted_traj_SDE
        elif self.latent_type == 'hybrid_SDE':
            Y_hat = predicted_traj_hybrid 
        
        weighted_mse_recon_loss, std_preds, weighted_kl_expt_loss, weighted_SDE_loss, fact_loss, logqp_SDE_mean = self.compute_factual_loss(Y, Y_hat, z_expt_mean, z_log_var_exp, logqp_SDE, Y_hat_SDE = predicted_traj_SDE)
        if self.latent_type == 'hybrid_SDE':
            total_loss = weighted_SDE_loss + weighted_kl_expt_loss + weighted_mse_recon_loss
        elif self.latent_type == 'expert':
            total_loss = weighted_kl_expt_loss + weighted_mse_recon_loss
        elif self.latent_type == 'SDE':
            total_loss = weighted_SDE_loss

        
        T_cf = (~T.bool()).long()
        z_expt, z_expt_mean, z_log_var_exp, latent_traj_expt_CF, predicted_traj_exp_CF, z_SDE, latent_traj_SDE_CF, logqp_SDE, predicted_traj_SDE_CF,predicted_traj_hybrid_CF = self.VAE_model(X, T_cf, 
                                                             time_in = time_X, 
                                                             time_out = time_Y,
                                                             MAP=True)
        if self.latent_type == 'expert':
            Y_hat_cf = predicted_traj_exp_CF
        elif self.latent_type == 'SDE':
            Y_hat_cf = predicted_traj_SDE_CF
        elif self.latent_type == 'hybrid_SDE':
            Y_hat_cf = predicted_traj_hybrid_CF
        
        mse_cf, mse_ite, std_preds_cf = self.compute_counterfactual_loss(Y,Y_cf, Y_hat,Y_hat_cf)

        
        if batch_idx ==0:
            #self.plot_trajectories( X, Y, Y_hat,latent_traj,  chart_type = "val" )
            if self.log_wandb:
                #latent_traj = torch.cat([latent_traj_expt, latent_traj_SDE], dim=-1)
                latent_traj = latent_traj_expt if latent_traj_expt != None else latent_traj_SDE
                plot_trajectories( X, Y, Y_hat, Y_cf, Y_hat_cf, latent_traj,  chart_type = "val" )

        self.log('val_total_loss', total_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_SDE_loss', weighted_SDE_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_Fact_loss', fact_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_logqp_SDE_mean', logqp_SDE_mean, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        self.log('val_recon_loss', weighted_mse_recon_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_std_preds', std_preds, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_kl_expt_loss', weighted_kl_expt_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)


        self.log('val_mse_cf', mse_cf, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_PEHE', mse_ite, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_std_preds_cf', std_preds_cf, on_epoch=True, prog_bar=True, logger=True)

        return total_loss

    def test_step(self, batch, batch_idx):
        X, Y, T, Y_cf, p, thetas_0, time_X, time_Y = batch
        #MAP = true as you don't sample the encoder variational latents during validation step 
        #MAP = true as you don't sample the encoder variational latents during validation step 
        z_expt, z_expt_mean, z_log_var_exp, latent_traj_expt, predicted_traj_exp, z_SDE, latent_traj_SDE, logqp_SDE, predicted_traj_SDE, predicted_traj_hybrid = self.VAE_model(X, T, 
                                                             time_in = time_X, 
                                                             time_out = time_Y, 
                                                             MAP=True)

        if self.latent_type == 'expert':
            Y_hat = predicted_traj_exp
        elif self.latent_type == 'SDE':
            Y_hat = predicted_traj_SDE
        elif self.latent_type == 'hybrid_SDE':
            Y_hat = predicted_traj_hybrid 
        
        weighted_mse_recon_loss, std_preds, weighted_kl_expt_loss, weighted_SDE_loss, fact_loss, logqp_SDE_mean = self.compute_factual_loss(Y, Y_hat, z_expt_mean, z_log_var_exp, logqp_SDE, Y_hat_SDE = predicted_traj_SDE)
        if self.latent_type == 'hybrid_SDE':
            total_loss = weighted_SDE_loss + weighted_kl_expt_loss + weighted_mse_recon_loss
        elif self.latent_type == 'expert':
            total_loss = weighted_kl_expt_loss + weighted_mse_recon_loss
        elif self.latent_type == 'SDE':
            total_loss = weighted_SDE_loss

        
        T_cf = (~T.bool()).long()
        z_expt, z_expt_mean, z_log_var_exp, latent_traj_expt_CF, predicted_traj_exp_CF, z_SDE, latent_traj_SDE_CF, logqp_SDE, predicted_traj_SDE_CF,predicted_traj_hybrid_CF = self.VAE_model(X, T_cf, 
                                                             time_in = time_X, 
                                                             time_out = time_Y,
                                                             MAP=True)
        if self.latent_type == 'expert':
            Y_hat_cf = predicted_traj_exp_CF
        elif self.latent_type == 'SDE':
            Y_hat_cf = predicted_traj_SDE_CF
        elif self.latent_type == 'hybrid_SDE':
            Y_hat_cf = predicted_traj_hybrid_CF
        
        
        mse_cf, mse_ite, std_preds_cf = self.compute_counterfactual_loss(Y,Y_cf, Y_hat,Y_hat_cf)

        
        if batch_idx ==0:
            #self.plot_trajectories( X, Y, Y_hat,latent_traj,  chart_type = "val" )
            if self.log_wandb:
                #latent_traj = torch.cat([latent_traj_expt, latent_traj_SDE], dim=-1)
                latent_traj = latent_traj_expt if latent_traj_expt != None else latent_traj_SDE
                plot_trajectories( X, Y, Y_hat, Y_cf, Y_hat_cf, latent_traj,  chart_type = "test" )

        self.log('test_total_loss', total_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('test_SDE_loss', weighted_SDE_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('test_Fact_loss', fact_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('test_logqp_SDE_mean', logqp_SDE_mean, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        self.log('test_recon_loss', weighted_mse_recon_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('test_std_preds', std_preds, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('test_kl_expt_loss', weighted_kl_expt_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)


        self.log('test_mse_cf', mse_cf, on_epoch=True, prog_bar=True, logger=True)
        self.log('test_PEHE', mse_ite, on_epoch=True, prog_bar=True, logger=True)
        self.log('test_std_preds_cf', std_preds_cf, on_epoch=True, prog_bar=True, logger=True)       
       

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr = self.learning_rate)
        
        scheduler = {"monitor": "train_total_loss", "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode = "min", factor = 0.5, patience = 50)}
        return {"optimizer": optimizer, "lr_scheduler":scheduler}
    

    