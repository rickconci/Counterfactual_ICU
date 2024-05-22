# Copyright 2020 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Latent SDE fit to a single time series with uncertainty quantification."""
import argparse
import logging
import math
import os
import random
from collections import namedtuple
from typing import Optional, Union
import sys
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import wandb

import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm
from torch import distributions, nn, optim

import torchsde

import lightning as L
from lightning import LightningModule


#from CV_data_6_new import create_cv_data
from CV_data_6 import create_cv_data

from utils_6 import CV_params, _stable_division, LinearScheduler, MLPSimple, CV_params_prior_mu, CV_params_prior_sigma, CV_params_max_min_2_5STD, CV_params_max_min_2STD
from plotting_6 import plot_trajectories_normal, plotting_config




def manual_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)




class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, expert_latent_dims, variational, encode_with_time_dim, encoder_num_layers, reverse=False):
        super(Encoder, self).__init__()
        self.input_dim = input_dim  # obs dim + tx dim
        self.hidden_dim = hidden_dim   
        self.latent_dim = latent_dim  # latent_dim depends on the latent model
        self.expert_latent_dims = expert_latent_dims
        non_expert_latent_dims = latent_dim - expert_latent_dims

        self.variational = variational
        self.reverse = reverse
        self.encode_with_time_dim = encode_with_time_dim
        self.encoder_num_layers = encoder_num_layers
        
        self.rnn = nn.GRU(input_dim + 1 if encode_with_time_dim else input_dim, hidden_dim, num_layers = encoder_num_layers, batch_first=True)

        if variational:
            self.hid2lat = nn.Linear(hidden_dim, 2*expert_latent_dims + non_expert_latent_dims )
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

        
        #rediscover the data mean and std so can convert in encoder output 
        input_mean_obs_dim =  x.mean([0,1]) #mean across batch & seq len
        input_std_obs_dim = x.std([0,1])    #std across batch & seq len
        

        # Reverse the sequence along the time dimension
        if self.reverse:
            xt = xt.flip(dims=[1])
            print('reversed xt shape:', xt.shape)  # Should match xt's shape

        _, h0 = self.rnn(xt)
        print('Output hidden state h0 shape:', h0.shape)  # Expected: [depth, batch_size, hidden_dim]
        print('output_last_dim', h0[-1].shape)
        
        # Process the last hidden state to produce latent variables
        z0 = self.hid2lat(h0[-1])
        print('z0 from hid to lat', z0.shape)
        if self.variational:
            
            z0_mean_expert = z0[:, :self.expert_latent_dims]
            z0_log_var_expert = z0[:, self.expert_latent_dims:self.expert_latent_dims ]
            z0_rest = z0[:, 2*self.expert_latent_dims:]

            scaled_expert_latents = self.sigmoid_scale(z0_mean_expert,input_mean_obs_dim,  input_std_obs_dim)
            z0_means = torch.cat([scaled_expert_latents, z0_rest], dim=-1)
            
            print('z0_mean shape:', z0_mean_expert.shape)  # Expected: [batch_size, latent_dim]
            print('z0_log_var shape:', z0_log_var_expert.shape)  # Expected: [batch_size, latent_dim]
            
            return z0_means, z0_log_var_expert
        
        else:
            z0_mean_expert = z0[:, :self.expert_latent_dims]
            z0_rest = z0[:, self.expert_latent_dims:]

            print('z0_mean_expert', z0_mean_expert[0,:4])
            print('z0_rest', z0_rest.shape)

            #scaled_expert_latents = self.sigmoid_scale(z0_mean_expert)
            #scaled_expert_latents = z0_mean_expert
            z0_means = torch.cat([z0_mean_expert, z0_rest], dim=-1)

            print('z0_mean shape:', z0_means.shape)  # Expected: [batch_size, latent_dim]
            #print('z0_means',z0_means[0,:4] )
            
            return z0_means



class Hybrid_VAE_SDE(LightningModule):

    def __init__(self, use_encoder, start_dec_at_treatment, variational_sampling, 
                 #Encoder
                 encoder_input_dim, encoder_hidden_dim, encoder_output_dim,
                 encoder_num_layers, variational_encoder, encoder_w_time, encoder_reverse_time,
                 use_2_5std_encoder_minmax, 
                 #SDE params
                 normalise_for_SDENN, 
                 expert_latent_dims, prior_tx_sigma, prior_tx_mu, self_reverting_prior_control, 
                 SDE_input_state, include_time, 
                 theta, SDE_control_weighting, 
                #SDE model params
                num_samples, SDEnet_hidden_dim, SDEnet_depth, SDEnet_out_dims,
                #decoder params
                decoder_output_dims, log_lik_output_scale, normalised_data, 
                #loss
                KL_weighting_SDE,
                #admin
                train_dir, learning_rate, log_wandb, adjoint, plot_every, 
                 start_scheduler = 200 , iter_scheduler = 600):
        super().__init__()

        self.noise_type = "diagonal"  # required
        self.sde_type = "ito"  # required
        self.sdeint_fn = torchsde.sdeint_adjoint if adjoint else torchsde.sdeint

        ### ADMIN
        self.train_dir = train_dir
        self.learning_rate = learning_rate
        self.log_wandb = log_wandb
        self.plot_every = plot_every

        ### Bifurcation options 
        self.use_encoder = use_encoder
        self.normalise_for_SDENN = normalise_for_SDENN
        self.start_dec_at_treatment = start_dec_at_treatment
        self.variational_sampling = variational_sampling
        self.SDE_input_state = SDE_input_state
        self.include_time = include_time
        
        ### Encoder model
        self.variational_encoder = variational_encoder
        self.use_2_5std_encoder_minmax = use_2_5std_encoder_minmax
        self.enc_model = Encoder(input_dim = encoder_input_dim,  
                                 hidden_dim = encoder_hidden_dim,
                                 latent_dim = encoder_output_dim, 
                                 expert_latent_dims = expert_latent_dims,
                                 variational = variational_encoder,
                                 encode_with_time_dim = encoder_w_time, 
                                 encoder_num_layers = encoder_num_layers,
                                 reverse = encoder_reverse_time)
        
        self.encoder_output_dim = encoder_output_dim
        self.encoder_SDENN_dims = encoder_output_dim - expert_latent_dims
        print('encoder_SDENN_dims', self.encoder_SDENN_dims)
        
        ### PRIOR PARAMS
        self.self_reverting_prior_control = self_reverting_prior_control
        self.prior_tx_sigma = prior_tx_sigma
        self.prior_tx_mu = prior_tx_mu

        mu_values = torch.tensor(list(CV_params_prior_mu.values())).float()
        mu_values = mu_values[:expert_latent_dims].view(1, -1)
        self.register_buffer('mu', mu_values)

        sigma_values = torch.tensor(list(CV_params_prior_sigma.values())).float()
        sigma_values = sigma_values[:expert_latent_dims].view(1, -1)
        self.register_buffer('sigma', sigma_values)


        self.register_buffer("theta", torch.tensor(theta).view(1, -1).expand(1, expert_latent_dims).float())

        ### LATENT MODEL  

        self.sdeint_fn = torchsde.sdeint_adjoint if adjoint else torchsde.sdeint

        self.num_samples = num_samples
        self.expert_latent_dims = expert_latent_dims
        self.CV_params = CV_params
        
        self.divisors = torch.tensor([CV_params['pa_divisor'], CV_params['pv_divisor'], CV_params['s_divisor'], CV_params['sv_divisor']], dtype=torch.float32)
        for key, value in CV_params.items():
            #setattr(self, key, nn.Parameter(torch.tensor(value, dtype=torch.float32), requires_grad=False))
            print(f'{key}: {value}')
            setattr(self, key, torch.tensor(value, dtype=torch.float32))

        self.SDEnet_hidden_dim = SDEnet_hidden_dim
        self.SDEnet_depth = SDEnet_depth
        self.SDEnet_out_dims = SDEnet_out_dims
        self.SDE_control_weighting = SDE_control_weighting

        net_input_dims = encoder_output_dim if SDE_input_state == 'full' else encoder_output_dim - len(encoder_input_dim)
        net_input_dims = net_input_dims + 2 if include_time else net_input_dims 

        #self.SDEnet = MLPSimple(input_dim = net_input_dims, 
        #                        output_dim = SDEnet_out_dims, 
        #                        hidden_dim = SDEnet_hidden_dim, 
        #                        depth = SDEnet_depth, 
        #                        activations = [nn.Tanh() for _ in range(SDEnet_depth)] )

        self.SDEnet = nn.Sequential(
            nn.Linear(net_input_dims, SDEnet_hidden_dim),
            nn.Tanh(),
            nn.Linear(SDEnet_hidden_dim, SDEnet_hidden_dim),
            nn.Tanh(),
            nn.Linear(SDEnet_hidden_dim, SDEnet_out_dims),
            nn.Tanh()
        )
        # Initialization trick from Glow.
        self.SDEnet[-2].weight.data.fill_(0.)
        self.SDEnet[-2].bias.data.fill_(0.)

        ### DECODER
        self.decoder_output_dims = decoder_output_dims
        self.normalised_data = normalised_data

        ### LOSS
        self.MSE_loss = nn.MSELoss(reduction = "none")
        self.log_lik_output_scale = log_lik_output_scale
        self.KL_weighting_SDE = KL_weighting_SDE
        self.kl_scheduler = LinearScheduler(start = start_scheduler, iters = iter_scheduler)

    
    def sigmoid_scale(self, z0):
        print('unnormalising the data and scaling it back to its appropriate values ')
        batch_size = z0.shape[0]
        
        # Splitting the input tensor into individual variables, one for each column
        p_a, p_v, s_reflex, sv = z0[:, 0], z0[:, 1], z0[:, 2], z0[:, 3]

        # Selecting the appropriate dictionary based on the condition
        params_dict = CV_params_max_min_2_5STD if self.use_2_5std_encoder_minmax else CV_params_max_min_2STD

        # Assigning values from the chosen dictionary
        pa_max = params_dict["max_pa"]
        pa_min = params_dict["min_pa"]
        pv_max = params_dict["max_pv"]
        pv_min = params_dict["min_pv"]
        s_max = params_dict["max_s"]
        s_min = params_dict["min_s"]
        sv_max = params_dict["max_sv"]
        sv_min = params_dict["min_sv"]

        print('CV_params', params_dict) 

        
        # Applying sigmoid and scaling to each dimension
        p_a = torch.sigmoid(p_a).unsqueeze(1) * (pa_max - pa_min) + pa_min
        p_v = torch.sigmoid(p_v).unsqueeze(1) * (pv_max - pv_min) + pv_min
        s_reflex = torch.sigmoid(s_reflex).unsqueeze(1) * (s_max - s_min) + s_min
        sv = torch.sigmoid(sv).unsqueeze(1) * (sv_max - sv_min) + sv_min

        # Concatenating along the second dimension (columns)
        scaled_out = torch.cat([p_a, p_v, s_reflex, sv], dim=1)
        print('scaled_out shape:', scaled_out.shape, scaled_out[:3, :])

        return scaled_out
    
    def normalise_expert_inputs(self, y):
        print('y to normalise', y.shape, y[0, :])
        print('CV_params_prior_mu', CV_params_prior_mu)
        print('CV_params_prior_sigma', CV_params_prior_sigma)
        pa = ((y[:, 0]- CV_params_prior_mu['pa'])/CV_params_prior_sigma['pa']).unsqueeze(1)
        pv = ((y[:, 1]- CV_params_prior_mu['pv'])/CV_params_prior_sigma['pv']).unsqueeze(1)
        s = ((y[:, 2]- CV_params_prior_mu['s'])/CV_params_prior_sigma['s']).unsqueeze(1)
        sv = ((y[:, 3]- CV_params_prior_mu['sv'])/CV_params_prior_sigma['sv']).unsqueeze(1)

        SDE_NN_norm_inputs = torch.cat([pa, pv, s, sv], dim=-1)
        return SDE_NN_norm_inputs

    def sigmoid(self, x):
        return 1 / (1 + torch.exp(-x))
    
    def apply_SDE_fun(self, t, y):

        if self.normalise_for_SDENN:
            SDNN_expert_input_state = self.normalise_expert_inputs(y[:, 1:self.expert_latent_dims+1])
        else:
            SDNN_expert_input_state = y[:, 1:self.expert_latent_dims+1]/self.divisors.to(self.device)

        print('SDNN_expert_input_state', SDNN_expert_input_state.shape, SDNN_expert_input_state[0, :])

        if self.include_time:
            # Positional encoding in transformers for time-inhomogeneous posterior
            sde_latent_times = torch.full_like(y[:, 0], fill_value=t).unsqueeze(1)
            sin_time = torch.sin(sde_latent_times)
            cos_time = torch.cos(sde_latent_times)

            if self.SDE_input_state == 'full':
                input_state = torch.cat([SDNN_expert_input_state, y[:, 5:]], dim=-1)
                SDE_NN_input = torch.cat((sin_time, cos_time, input_state), dim=-1)

            elif self.SDE_input_state == 'latents':
                input_state = torch.cat([SDNN_expert_input_state[:,2:], y[:, 5:]], dim=-1)
                SDE_NN_input = torch.cat((sin_time, cos_time, input_state), dim=-1)

        else:
            if self.SDE_input_state == 'full':
                SDE_NN_input = torch.cat([SDNN_expert_input_state, y[:, 5:]], dim=-1)

            elif self.SDE_input_state == 'latents':
                SDE_NN_input = torch.cat([SDNN_expert_input_state[:,2:], y[:, 5:]], dim=-1)

        print('SDE_NN_input shape', SDE_NN_input.shape)
        print('SDE_NN_input example', SDE_NN_input[0,:])
        SDE_NN_output_latents = self.SDEnet(SDE_NN_input) 
        print('SDE_NN_output_latents', SDE_NN_output_latents.shape)
        #print('SDE_NN_output_latents example', SDE_NN_output_latents[0, :])
        has_nonzero = SDE_NN_output_latents.ne(0.).any()
        print('SDE_NN Has non-0 OUTPUT??', has_nonzero)
        return SDE_NN_output_latents
    
    def scale_unnormalised_experts(self, expert_dims):
        pa = (100*expert_dims[:, 0]).unsqueeze(1)
        pv = (100*expert_dims[:, 1]).unsqueeze(1)
        s =  (1*expert_dims[:, 2]).unsqueeze(1)
        sv = (100*expert_dims[:, 3]).unsqueeze(1)
        return torch.cat([pa, pv, s, sv], dim=-1)
    
    def f(self, t, y, Tx, time_to_treatment):  # Approximate posterior drift.

        i_ext = y[:,0].unsqueeze(1) 
        p_a = y[:,1].unsqueeze(1) 
        p_v = y[:,2].unsqueeze(1) 
        s_reflex = y[:, 3] .unsqueeze(1) 
        sv = y[:, 4].unsqueeze(1)
            

        print('time, i_ext, p_a pv, s, sv', t.item(), i_ext[0].item(), p_a[0].item(), p_v[0].item(), s_reflex[0].item(), sv[0].item())   
        print('y', y.shape)
        print('t, time_to_treatment', t.item(), time_to_treatment.shape)
        if t.item() >= time_to_treatment:
            print('Treatment has started! Estimating effect', t.item(), time_to_treatment )
            #the neural network is trying to learn the ultimate treatment effect!! this means both fluid function AND the v_fun. V_fun determines the unknown tx_effect multiplied (beyond) the model, hence to be learned
            dt_i_ext_SDE = self.apply_SDE_fun(t, y) * self.SDE_control_weighting
            print('dt_i_ext_SDE NN', dt_i_ext_SDE.shape)
            print('dt_i_ext_SDE', dt_i_ext_SDE[:3, :])

        else:
            dt_i_ext_SDE = torch.zeros_like(y[:,0]).unsqueeze(1)
        
        #T is binary and indicates whether a treatment was given or not. 
        #this is an important step to then create the counterfactuals when we set T as the opposite of what it's trained on (T_cf)
        #print('Tx', Tx)
        i_ext_SDE = Tx[:,None] * i_ext 

        has_nonzero = i_ext_SDE.ne(0.).any()
        print('i_ext_SDE Has non-0 OUTPUT??', has_nonzero)
        
        #print('i_ext', i_ext_SDE.shape)
        #print('i_ext example', i_ext_SDE[:5, :])
        
        #print('fixed params', self.f_hr_min.item(), self.f_hr_max.item(), self.r_tpr_max.item(), self.r_tpr_min.item(),self.r_tpr_mod.item() )
        #print('fixed params', self.ca.item(), self.cv.item(), self.tau.item(), self.k_width.item(),self.p_aset.item(),self.sv_mod.item())

        f_hr = s_reflex * (self.f_hr_max - self.f_hr_min) + self.f_hr_min
        r_tpr = s_reflex * (self.r_tpr_max - self.r_tpr_min) + self.r_tpr_min - self.r_tpr_mod
        
        # Calculate changes in volumes and pressures
        dva_dt = -1. * (p_a - p_v) / (r_tpr + 1e-7)  + sv * f_hr
        dvv_dt = -1. * dva_dt + i_ext_SDE
        #print('f_hr, dva_dt, r_tpr, dvv_dt', f_hr.shape, dva_dt.shape, r_tpr.shape, dvv_dt.shape)
        #print('f_hr, dva_dt, r_tpr, dvv_dt', f_hr[0], dva_dt[0], r_tpr[0], dvv_dt[0])

        # Calculate derivative of state variables
        dpa_dt = dva_dt / (self.ca * 100.)
        dpv_dt = dvv_dt / (self.cv * 10.)
        ds_dt = (1. / self.tau) * (1. - 1. / (1 + torch.exp(-self.k_width * (p_a - self.p_aset))) - s_reflex)
                                  #(1. - 1. / (1 + torch.exp(-self.params["k_width"] * (p_a - self.params["p_aset"]))) - s)
        #self.sigmoid(self.k_width * (p_a - self.p_aset)) - s_reflex)
        dsv_dt = i_ext_SDE * self.sv_mod
        print('dpa_dt, dpv_dt, ds_dt, dsv_dt', dpa_dt.shape, dpv_dt.shape, ds_dt.shape, dsv_dt.shape)

        
        diff_results = torch.cat([dt_i_ext_SDE, dpa_dt, dpv_dt, ds_dt, dsv_dt], dim=-1)
        print('diff_results example', diff_results[0,:])
        print('diff_results', diff_results.shape)

        return diff_results 

    def h(self, t, y):  # Prior drift.
        mu = torch.tensor(self.prior_tx_mu).to(self.device)
        expanded_mu = mu.repeat(y.size(0), 1)
        print('theta h', self.theta.shape, self.theta[0,:])
        print('mu h', expanded_mu.shape, expanded_mu[0,:])
        print('y in h', y.shape, y[0,:])

        return self.theta * (expanded_mu - y)

    def f_aug(self, t, y):  # Drift for augmented dynamics with logqp term.
        i_ext = y[:, 0].unsqueeze(1)
        dt_expert_dims = y[:, 1:self.expert_latent_dims] # this is pa, pv, s, sv
        dt_all_dims = y[:, :self.encoder_output_dim + 1]  # this is i_ext, pa, pv, s, sv and other encoder latents that go straight to the SDENN
        Tx = y[:, -2]
        time_to_treatment = y[0, -1]
        encoder_to_SDENN_latents = torch.zeros(y.shape[0], self.encoder_output_dim - self.expert_latent_dims).to(self.device)
        print('inputs to f', dt_all_dims.shape) # num_samples x sde_dims 
        
        f_res = self.f(t, dt_all_dims, Tx, time_to_treatment)
        
        if self.self_reverting_prior_control:
            g_iext, h_iext  = self.g(t, i_ext), self.h(t, i_ext)
            f_iext = f_res[:,0].unsqueeze(1)
            print('f', f_iext.shape, 'g', g_iext.shape, 'h', h_iext.shape)
            print('f mean', f_iext.mean(), 'g mean', g_iext.mean(), 'h mean', h_iext.mean())
            print('f', f_iext[:3,:], 'g ', g_iext[:3,:], 'h ', h_iext[:3,:] )
            
            print('doing stable division!')
            u = _stable_division(f_iext - h_iext, g_iext)
            print('u shape', u.shape)
            f_logqp = .5 * (u ** 2).sum(dim=1, keepdim=True)
        else:
            f_logqp = torch.zeros_like(y[:, 0]).unsqueeze(1).to(self.device)
        
        print('f_logqp', f_logqp.shape)
        print('f_logqp mean', f_logqp[:3,:])        

        f_out = torch.cat([f_res, encoder_to_SDENN_latents, f_logqp, torch.zeros_like(f_logqp), torch.zeros_like(f_logqp)], dim=1)
        #f_out now has 7 +K dims + : 4 dims for dt, K dims of latents going straight to SDENN, 1 dim for SDENN output, 1 dim for logqp, 1 dim for T, 1 dim for time_to_T  
        print('f_aug out', f_out.shape, f_out[0])
        return f_out
    
    def g(self, t, y):  
        #sigma is different for each values here! 
        sigma = torch.tensor(self.prior_tx_sigma, dtype=torch.float32).to(self.device).unsqueeze(0)
        expanded_sigma = sigma.expand(y.size(0), 1)
        print('sigma g', expanded_sigma.shape, expanded_sigma[0] )
        return expanded_sigma

    def g_aug(self, t, y):  # Diffusion for augmented dynamics with logqp term.
        
        i_ext = y[:, 0].unsqueeze(1)
        dt_expert_dims = torch.zeros(y.shape[0], self.expert_latent_dims).to(self.device)
        encoder_to_SDENN_latents = torch.zeros(y.shape[0], self.encoder_output_dim - self.expert_latent_dims).to(self.device)
        
        g_res = self.g(t, i_ext)
        print('g', g_res.shape, g_res[0])
        g_logqp = torch.zeros(y.size(0), 1).to(y.device)

        g_out = torch.cat([g_res, dt_expert_dims, encoder_to_SDENN_latents,  g_logqp, torch.zeros_like(g_logqp), torch.zeros_like(g_logqp)], dim=1)
        print('g out', g_out.shape, g_out[0])
        return g_out

    def forward_latent(self, init_latents, ts, Tx, time_to_tx):
        #inputs of shape [batch x num_samples x dim ]
        batch_size = init_latents.shape[0]
        Tx_expanded = Tx.unsqueeze(1).unsqueeze(2).expand(-1, self.num_samples, -1).to(init_latents)
        time_to_tx = time_to_tx.unsqueeze(1).unsqueeze(2).expand(batch_size, self.num_samples, -1).to(init_latents)
        i_ext = torch.zeros(batch_size,self.num_samples, 1).to(init_latents)
        log_path = torch.zeros(batch_size,self.num_samples, 1).to(init_latents)
        print('ts',ts.shape)
        print('i_ext ', i_ext.shape)
        print('init_latents ', init_latents.shape)
        print('log_path', log_path.shape)
        print(f"Tx_expanded shape: {Tx_expanded.shape}")
        print(f"time_to_tx shape: {time_to_tx.to(init_latents).shape}")

        
        aug_y0 = torch.cat([i_ext, init_latents,  log_path, Tx_expanded, time_to_tx], dim=-1) 
        print('aug_y0', aug_y0.shape, aug_y0[0, 0,:])
        dim_aug = aug_y0.shape[-1]
        aug_y0 = aug_y0.reshape(-1,dim_aug) # this is because the SDEint can only accept [M x dim], so M becomes batch_size * num_samples

        print('aug_y0', aug_y0.shape) #this will be num_samples x dim = 512 x 4
        options = {'dtype': torch.float32}
        aug_ys = self.sdeint_fn(
            sde=self,
            y0=aug_y0,
            ts=ts,
            method='euler',
            dt=0.05,
            adaptive=False,
            rtol=1e-3,
            atol=1e-3,
            options = options, 
            names={'drift': 'f_aug', 'diffusion': 'g_aug'}
        )
        
        print('len(ts)', len(ts))
        print('dim_aug', dim_aug)
        print('aug_ys pre_reshape',aug_ys.shape)
        
        aug_ys = aug_ys.view(len(ts), batch_size, self.num_samples, dim_aug).permute(1,2,0,3)
        #reshape(self.num_samples,-1,  len(ts),dim_aug) # reshape for # batch_size x num_samples x times x dim
        
        i_ext_path = aug_ys[:, :, :, 0]
        latent_out = aug_ys[:, :, :, 1:self.expert_latent_dims+1]
        logqp_path = aug_ys[:,: , -1, -3]    #.mean(dim=0)  # KL(t=0) + KL(path).
       
        print('latent_out end of latent ', latent_out[0, 0, :, :])
        
        print('i_ext_path', i_ext_path.shape, i_ext_path[:4,:2,:])
        print('ys_extracted', latent_out.shape)
        print('logqp_path_extracted', logqp_path.shape)
       
        return latent_out, logqp_path, i_ext_path


    def forward_enc(self,input_vals, time_in):

        if self.start_dec_at_treatment: #here we use an encoder to take our observed variables and infer the latents at the moment we want to give the treatment & initial
                        # Q1: WHICH latents!? there are MANY. # Q2: how can we make the encoder 'variational'? not on the stating values but the ones just before treatment. 
            
            if self.variational_encoder:
                z1_mean, z1_logvar = self.enc_model(input_vals, time_in)
                z1 = z1_mean.unsqueeze(1).repeat(1, self.num_samples, 1) # Add an extra dimension: shape becomes [batch, 1, latent]
                logqp0 = 0
                print('encoder_output', z1.shape)

            else:
                z1 = self.enc_model(input_vals, time_in)
                z1 = torch.cat([self.sigmoid_scale(z1[:,:self.expert_latent_dims]), z1[:, self.expert_latent_dims:] ], dim =-1)
                z1 = z1.unsqueeze(1).repeat(1, self.num_samples, 1) # Add an extra dimension: shape becomes [batch, 1, latent]
                logqp0 = 0
                z1_logvar = 0
                print('encoder_output', z1.shape)


        else: #Here we do NOT use an encoder, as we run the models on the FULL trajectory by giving it WHEN the tx will occur 
            if self.variational_sampling: # here we can see the OBSERVED initial values clearly, but we need INFER the LATENT values from a Prior
                                     # Question 1: WHICH latents!? there are many as part of the model. Do we try and sample them all here and then give them to the model? or the model takes them as params w grad?
                                     # Question 2: Each latent has its own prior distribution which is not necessarily gaussian, 
                #  NOT FINISHED!! The priors need to be accurate for the expert dims
                #eps = torch.randn(self.num_samples, self.total_SDE_input_dims).to(self.qy0_std) if eps is None else eps
                #z_SDE = self.qy0_mean + eps * self.qy0_std
                #qy0 = distributions.Normal(loc=self.qy0_mean, scale=self.qy0_std)
                #py0 = distributions.Normal(loc=self.py0_mean, scale=self.py0_std)
                #logqp0 = distributions.kl_divergence(qy0, py0).sum(dim=1)  # KL(t=0).
                pass
            else: #This is the simplest and least correct: we are saying that we can see ALL initial values perfectly correctly! So this is both the Observed and ALL the LATENTS. 
                  # This is because currently the SDE_hybrid model is NOT trying to learn the PARAMS. If the model does try to learn them, then this is saying that we have full access to the 4 MAIN vars: Pa, Pv, S, Sv (which normally would be 2 observed and 2 latents)
                #this is shape of [batch x dims] , where dims = Pa, Pv, S, Sv
                # Add an extra dimension: shape becomes [batch, 1, latent] & repeat by num_samples
                z1 = input_vals.unsqueeze(1).repeat(1, self.num_samples, 1)
                
                logqp0 = 0

        return  z1, z1_logvar, logqp0 

    def select_tensor_by_index_list_advanced(self,tensor, index_list):
        # Convert index list to a tensor of type long
        indices = torch.tensor(index_list, dtype=torch.long).to(self.device)
        
        # Select along the last dimension using torch.index_select
        selected_tensor = torch.index_select(tensor, -1, indices)
        
        return selected_tensor
    
    def forward_dec(self, latent_out):
        print('latent_out', latent_out[0, 0, :, 1])
        print('latent_out', latent_out[0, 1, :, 1])
        print('latent_out', latent_out[1, 0, :, 1])
        print('latent_out', latent_out[1, 1, :, 1])
       
        if self.normalised_data:
            latent_out[:,:,:,0] = (latent_out[:,:,:,0] - CV_params_prior_mu['pa'])/CV_params_prior_sigma['pa']
            latent_out[:,:,:,1] = (latent_out[:,:,:,1] - CV_params_prior_mu['pv'])/CV_params_prior_sigma['pv']
            latent_out[:,:,:,2] = (latent_out[:,:,:,2] - CV_params_prior_mu['s'])/CV_params_prior_sigma['s']
            latent_out[:,:,:,3] = (latent_out[:,:,:,3] - CV_params_prior_mu['sv'])/CV_params_prior_sigma['sv']
        
        else:
            #now we just need to rescale down instead to renormalise 
            latent_out = latent_out/self.divisors.view(1, 1, 1, 4).to(self.device)

        output_traj = self.select_tensor_by_index_list_advanced(latent_out, self.decoder_output_dims)
        print('output_traj', output_traj.shape, output_traj[0, 0, :, :])

        return output_traj

    def compute_factual_loss(self, predicted_traj, true_traj, logqp):
        
        true_traj_expanded = true_traj.unsqueeze(1).repeat(1, self.num_samples, 1, 1) 
        print('true_traj_expanded', true_traj_expanded.shape, true_traj_expanded[0,0,:,:])
        
        print('predicted_traj', predicted_traj.shape, predicted_traj[0,0,:,:])
        output_scale = torch.full(predicted_traj.shape, self.log_lik_output_scale, device=self.device)
        print('output_scale', output_scale.shape)

        #calculating the gaussian log prob between the predicted and the true traj. We want this to be as big as possible, so we will minimise its negative.
        print('likelihood')
        likelihood = distributions.Normal(loc=predicted_traj, scale=output_scale)
        print('log prob')
        logpy = likelihood.log_prob(true_traj_expanded)
        print('logpy', logpy.shape)
        logpy = logpy.sum((2,3)) #sum across times and dims, keeping a loss for each sde sample and batch size 
        print('logpy', logpy.shape, logpy.mean())

        print('FACT LOSS', logpy.mean())

        # calculating final loss 
        loss = -logpy.mean() + self.KL_weighting_SDE* (logqp.mean() * self.kl_scheduler.val)
        loss = loss.squeeze() 
        print("TOTAL LOSS ", loss)

        return loss, -logpy.mean(), logqp.mean()

    def compute_counterfactual_loss(self, true_fact, true_cf, pred_fact, pred_cf):
        print('true_fact:', true_fact.shape, true_fact[0,:,:] )
        print('true_cf:', true_cf.shape, true_cf[0,:,:])
        print('pred_fact:', pred_fact.shape, pred_fact.mean(1)[0,:,:])
        print('pred_cf:', pred_cf.shape, pred_cf.mean(1)[0,:,:])

        # RECON LOSS
        # MSE loss between the Y and the MEAN of the SDE samples predictions, which includes expert and SDE in hybrid 
        mse_cf = torch.sqrt(self.MSE_loss(true_cf, pred_cf.mean(1))).mean()
        # Now find the mean of the standard devs of the predictions across the SDE samples
        std_preds_cf = pred_cf.std(1).mean()

        # Individual Treatment Effect computed as the difference between Y_cf and Y
        ite = (true_cf- true_fact)
        print('ite:', ite.shape)

        # Predicted Individual Treatment Effect computed as the difference between the mean predictions of Y_hat_cf and Y_hat
        ite_hat = (pred_cf.mean(1) - pred_fact.mean(1))
        print('ite_hat:', ite_hat.shape)

        # MSE of the ITE
        mse_ite = torch.sqrt(self.MSE_loss(ite, ite_hat)).mean()
        print('mse_ite:', mse_ite)    


        return mse_cf, mse_ite, std_preds_cf
        

    def training_step(self, batch, batch_idx):
        print("TRAINING")
        X, Y, T, Y_cf, p, init_states, time_pre, time_post, time_FULL, full_fact_traj, full_cf_traj = batch

        if self.use_encoder == 'full':
            print('time_post', time_post.shape)
            input_vals = X if self.start_dec_at_treatment else init_states[:, :4] 
            z1_input, z_logvar, logqp0 =   self.forward_enc(input_vals, time_pre)
            

        elif self.use_encoder == 'partial':
            input_vals = X if self.start_dec_at_treatment else init_states[:, :4] 
            z1_enc, z_logvar, logqp0 =   self.forward_enc(input_vals, time_pre)
            z1_ML = z1_enc[:, 0, self.expert_latent_dims:] 
            #if z1_ML.nelement() != 0: to be extra robust 

            z1_real = full_fact_traj[:, 15, :]
            if self.normalised_data: 
                z1_real = self.sigmoid_scale(z1_real)
            else: #need to scale only! YAY
                z1_real = self.scale_unnormalised_experts(z1_real)
            print('z1_real', z1_real.shape)
            print('z1_ML', z1_ML.shape)
            z1_input = torch.cat([z1_real, z1_ML  ], dim =-1)
            z1_input = z1_input.unsqueeze(1).repeat(1, self.num_samples, 1) 

        elif self.use_encoder == 'none':
            print('full_fact_traj', full_fact_traj.shape)
            z1_real = full_fact_traj[:, 15, :]
            print('z1_real', z1_real.shape, z1_real[0, :])
            if self.normalised_data: 
                z1_real = self.sigmoid_scale(z1_real)
            else: #need to scale only! YAY
                z1_real = self.scale_unnormalised_experts(z1_real)

            z1_ML = torch.zeros(z1_real.shape[0],self.encoder_SDENN_dims).to(self.device)
            z1_input = torch.cat([z1_real, z1_ML  ], dim =-1)
            z1_input = z1_input.unsqueeze(1).repeat(1, self.num_samples, 1) 
            logqp0 = 0

        print('z1_input', z1_input.shape, z1_input[0, 0, :])

        latent_traj, logqp_path, i_ext_path = self.forward_latent(init_latents = z1_input, 
                                                      ts = time_post[0] if self.start_dec_at_treatment else time_FULL[0], 
                                                      Tx= T, 
                                                      time_to_tx = torch.tensor([0]) if self.start_dec_at_treatment else torch.tensor([time_pre.shape[1] - 1]))
        predicted_traj = self.forward_dec(latent_traj )

        loss, fact_loss, kl_loss = self.compute_factual_loss(predicted_traj = predicted_traj, 
                                                             true_traj = Y if self.start_dec_at_treatment else  self.select_tensor_by_index_list_advanced(full_fact_traj, self.decoder_output_dims), 
                                                             logqp=  logqp0 + logqp_path)
        
        
        self.log('train_total_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_fact_loss', fact_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_kl_loss', kl_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.kl_scheduler.step()

        print("LOSS STEP")
        return loss


    def validation_step(self, batch, batch_idx):
        print("VALIDATION")
        X, Y, T, Y_cf, p, init_states, time_pre, time_post, time_FULL, full_fact_traj, full_cf_traj = batch
        
        if self.use_encoder == 'full':
            print('time_post', time_post.shape)
            input_vals = X if self.start_dec_at_treatment else init_states[:, :4] 
            z1_input, z_logvar, logqp0 =   self.forward_enc(input_vals, time_pre)
            
        elif self.use_encoder == 'partial':
            input_vals = X if self.start_dec_at_treatment else init_states[:, :4] 
            z1_enc, z_logvar, logqp0 =   self.forward_enc(input_vals, time_pre)
            z1_ML = z1_enc[:, 0, self.expert_latent_dims:] 
            #if z1_ML.nelement() != 0: to be extra robust 

            z1_real = full_fact_traj[:, 15, :]
            if self.normalised_data: 
                z1_real = self.sigmoid_scale(z1_real)
            else: #need to scale only! YAY
                z1_real = self.scale_unnormalised_experts(z1_real)
            print('z1_real', z1_real.shape)
            print('z1_ML', z1_ML.shape)
            z1_input = torch.cat([z1_real, z1_ML  ], dim =-1)
            z1_input = z1_input.unsqueeze(1).repeat(1, self.num_samples, 1) 

        elif self.use_encoder == 'none':
            print('full_fact_traj', full_fact_traj.shape)
            z1_real = full_fact_traj[:, 15, :]
            print('z1_real', z1_real.shape, z1_real[0, :])
            if self.normalised_data: 
                z1_real = self.sigmoid_scale(z1_real)
            else: #need to scale only! YAY
                z1_real = self.scale_unnormalised_experts(z1_real)

            z1_ML = torch.zeros(z1_real.shape[0],self.encoder_SDENN_dims).to(self.device)
            z1_input = torch.cat([z1_real, z1_ML  ], dim =-1)
            z1_input = z1_input.unsqueeze(1).repeat(1, self.num_samples, 1) 
            logqp0 = 0

        print('z1_input', z1_input.shape, z1_input[0, 0, :])

        


        latent_traj, logqp_path, i_ext_path = self.forward_latent(init_latents = z1_input, 
                                                      ts = time_post[0] if self.start_dec_at_treatment else time_FULL[0], 
                                                      Tx= T, 
                                                      time_to_tx = torch.tensor([0]) if self.start_dec_at_treatment else torch.tensor([time_pre.shape[1] - 1]))
        predicted_traj = self.forward_dec(latent_traj )
        
        loss, fact_loss, kl_loss = self.compute_factual_loss(predicted_traj = predicted_traj, 
                                                             true_traj = Y if self.start_dec_at_treatment else  self.select_tensor_by_index_list_advanced(full_fact_traj, self.decoder_output_dims), 
                                                             logqp=  logqp0 + logqp_path)
        
        ### now the COUNTERFACTUAL
        print('NOW COUNTERFACTUAL')
        #z1_cf, logqp0_cf =   self.forward_enc(input_vals, time_pre)
        latent_traj_cf, logqp_path_cf, _ = self.forward_latent(init_latents = z1_input, 
                                                      ts = time_post[0] if self.start_dec_at_treatment else time_FULL[0], 
                                                      Tx= (~T.bool()).long(), 
                                                      time_to_tx = torch.tensor([0]) if self.start_dec_at_treatment else torch.tensor([time_pre.shape[1] - 1]))
        predicted_traj_cf = self.forward_dec(latent_traj_cf )

        mse_cf, mse_ite, std_preds_cf = self.compute_counterfactual_loss(true_fact = Y if self.start_dec_at_treatment else self.select_tensor_by_index_list_advanced(full_fact_traj, self.decoder_output_dims),
                                                                         true_cf = Y_cf if self.start_dec_at_treatment else  self.select_tensor_by_index_list_advanced(full_cf_traj, self.decoder_output_dims), 
                                                                         pred_fact= predicted_traj,
                                                                         pred_cf= predicted_traj_cf)

        if self.global_step % self.plot_every == 0:
            if self.start_dec_at_treatment:
                self.plot_trajectories_hyland( X, Y, Y_cf, predicted_traj.permute(0, 2, 1, 3), predicted_traj_cf.permute(0, 2, 1, 3), chart_type = "val" )
                self.plot_SDENN_output(i_ext_path)
            else:
                plot_trajectories_normal(ts = time_FULL[0] , 
                                        zs = predicted_traj[0], 
                                        ys = full_fact_traj[0], 
                                        plotting_config= plotting_config, 
                                        img_path = os.path.join(self.train_dir, f'Factual_{self.global_step}.png'),
                                        log_to_wandb=self.log_wandb)
                
                plot_trajectories_normal(ts = time_FULL[0], 
                                        zs = predicted_traj_cf[0], 
                                        ys = full_cf_traj[0], 
                                        plotting_config = plotting_config, 
                                        img_path = os.path.join(self.train_dir, f'Counterfactual_{self.global_step}.png'),
                                        log_to_wandb=self.log_wandb)

        self.log('val_total_loss', loss,   on_epoch=True, prog_bar=True, logger=True)
        self.log('val_fact_loss', fact_loss, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_kl_loss', kl_loss,  on_epoch=True, prog_bar=True, logger=True)

        self.log('val_mse_cf', mse_cf, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_PEHE', mse_ite, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_std_preds_cf', std_preds_cf, on_epoch=True, prog_bar=True, logger=True)
        

        return loss
   
         
    def test_step(self, batch, batch_idx):
        print("TEST")
        X, Y, T, Y_cf, p, init_states, time_pre, time_post, time_FULL, full_fact_traj, full_cf_traj = batch
        
        if self.use_encoder == 'full':
            print('time_post', time_post.shape)
            input_vals = X if self.start_dec_at_treatment else init_states[:, :4] 
            z1_input, z_logvar, logqp0 =   self.forward_enc(input_vals, time_pre)
            
        elif self.use_encoder == 'partial':
            input_vals = X if self.start_dec_at_treatment else init_states[:, :4] 
            z1_enc, z_logvar, logqp0 =   self.forward_enc(input_vals, time_pre)
            z1_ML = z1_enc[:, 0, self.expert_latent_dims:] 
            #if z1_ML.nelement() != 0: to be extra robust 

            z1_real = full_fact_traj[:, 15, :]
            if self.normalised_data: 
                z1_real = self.sigmoid_scale(z1_real)
            else: #need to scale only! YAY
                z1_real = self.scale_unnormalised_experts(z1_real)
            print('z1_real', z1_real.shape)
            print('z1_ML', z1_ML.shape)
            z1_input = torch.cat([z1_real, z1_ML  ], dim =-1)
            z1_input = z1_input.unsqueeze(1).repeat(1, self.num_samples, 1) 

        elif self.use_encoder == 'none':
            print('full_fact_traj', full_fact_traj.shape)
            z1_real = full_fact_traj[:, 15, :]
            print('z1_real', z1_real.shape, z1_real[0, :])
            if self.normalised_data: 
                z1_real = self.sigmoid_scale(z1_real)
            else: #need to scale only! YAY
                z1_real = self.scale_unnormalised_experts(z1_real)

            z1_ML = torch.zeros(z1_real.shape[0],self.encoder_SDENN_dims).to(self.device)
            z1_input = torch.cat([z1_real, z1_ML  ], dim =-1)
            z1_input = z1_input.unsqueeze(1).repeat(1, self.num_samples, 1) 
            logqp0 = 0

        print('z1_input', z1_input.shape, z1_input[0, 0, :])


        latent_traj, logqp_path, i_ext_path = self.forward_latent(init_latents = z1_input, 
                                                      ts = time_post[0] if self.start_dec_at_treatment else time_FULL[0], 
                                                      Tx= T, 
                                                      time_to_tx = torch.tensor([0]) if self.start_dec_at_treatment else torch.tensor([time_pre.shape[1] - 1]))
        predicted_traj = self.forward_dec(latent_traj )
        
        loss, fact_loss, kl_loss = self.compute_factual_loss(predicted_traj = predicted_traj, 
                                                             true_traj = Y[:, :, self.decoder_output_dims] if self.start_dec_at_treatment else full_fact_traj[:,:,self.decoder_output_dims], 
                                                             logqp=  logqp0 + logqp_path)
        
        ### now the COUNTERFACTUAL
        print('NOW COUNTERFACTUAL')
        #z1_cf, logqp0_cf =   self.forward_enc(input_vals, time_pre)
        latent_traj_cf, logqp_path_cf, _ = self.forward_latent(init_latents = z1_input, 
                                                      ts = time_post[0] if self.start_dec_at_treatment else time_FULL[0], 
                                                      Tx= (~T.bool()).long(), 
                                                      time_to_tx = torch.tensor([0]) if self.start_dec_at_treatment else torch.tensor([time_pre.shape[1] - 1]))
        predicted_traj_cf = self.forward_dec(latent_traj_cf )
        true_traj = Y if self.start_dec_at_treatment else full_cf_traj

        mse_cf, mse_ite, std_preds_cf = self.compute_counterfactual_loss(true_fact = Y if self.start_dec_at_treatment else full_fact_traj,
                                                                         true_cf= Y_cf[:,:,self.decoder_output_dims] if self.start_dec_at_treatment else full_cf_traj[:,:,self.decoder_output_dims], 
                                                                         pred_fact= predicted_traj,
                                                                         pred_cf= predicted_traj_cf)

        if batch_idx == 0:
            if self.start_dec_at_treatment:
                self.plot_trajectories_hyland( X, Y, Y_cf, predicted_traj.permute(0, 2, 1, 3), predicted_traj_cf.permute(0, 2, 1, 3), chart_type = "val" )
                self.plot_SDENN_output(i_ext_path)
            else:
                plot_trajectories_normal(ts = time_FULL[0] , 
                                        zs = predicted_traj[0], 
                                        ys = full_fact_traj[0], 
                                        plotting_config= plotting_config, 
                                        img_path = os.path.join(self.train_dir, f'Factual_{self.global_step}.png'),
                                        log_to_wandb=self.log_wandb)
                
                plot_trajectories_normal(ts = time_FULL[0], 
                                        zs = predicted_traj_cf[0], 
                                        ys = full_cf_traj[0], 
                                        plotting_config = plotting_config, 
                                        img_path = os.path.join(self.train_dir, f'Counterfactual_{self.global_step}.png'),
                                        log_to_wandb=self.log_wandb)

        self.log('test_total_loss', loss,   on_epoch=True, prog_bar=True, logger=True)
        self.log('test_fact_loss', fact_loss, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_kl_loss', kl_loss,  on_epoch=True, prog_bar=True, logger=True)

        self.log('test_mse_cf', mse_cf, on_epoch=True, prog_bar=True, logger=True)
        self.log('test_PEHE', mse_ite, on_epoch=True, prog_bar=True, logger=True)
        self.log('test_std_preds_cf', std_preds_cf, on_epoch=True, prog_bar=True, logger=True)

        return loss
   


    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr = self.learning_rate)
        
        scheduler = {"monitor": "train_total_loss", "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode = "min", factor = 0.5, patience = 50)}
        return {"optimizer": optimizer, "lr_scheduler":scheduler}

    def plot_trajectories_hyland(self, X, Y, Y_cf, Y_hat, Y_hat_cf, chart_type = "val" ):
        print('PLOTTING')
        if not os.path.exists(self.train_dir):
            os.makedirs(self.train_dir)
            print(f'Created directory: {self.train_dir}')

        X_df_list = []
        for dim in range(X.shape[-1]):
            X_df = pd.DataFrame(X.cpu().numpy()[0,:,dim], columns = ["PaPv"])
            X_df["time"]  = np.arange(X_df.shape[0])
            X_df["type"] = f"Input_{dim}"
            X_df_list.append(X_df)
        X_df = pd.concat(X_df_list)
        time_x_end = X_df.time.max() + 1

        for dim in range(Y.shape[-1]):
            std_hat = 0
            std_hat_cf = 0

            Y_fact_df = pd.DataFrame(Y.cpu().numpy()[0,:,dim], columns = ["PaPv"])
            Y_fact_df["time"]  = time_x_end + np.arange(Y_fact_df.shape[0])
            Y_fact_df["type"] = "Factual"

            Y_cfact_df = pd.DataFrame(Y_cf.cpu().numpy()[0,:,dim], columns = ["PaPv"])
            Y_cfact_df["time"]  = time_x_end + np.arange(Y_cfact_df.shape[0])
            Y_cfact_df["type"] = "Counterfactual"

            Y_hat_fact_df = pd.DataFrame(Y_hat.cpu().numpy()[0,:,dim], columns = ["PaPv"])
            Y_hat_fact_df["time"]  = time_x_end + np.arange(Y_hat_fact_df.shape[0])
            Y_hat_fact_df["type"] = "Factual Prediction"
            Y_hat_fact_df["std"] = std_hat

            Y_hat_cfact_df = pd.DataFrame(Y_hat_cf.cpu().numpy()[0,:,dim], columns = ["PaPv"])
            Y_hat_cfact_df["time"]  = time_x_end + np.arange(Y_hat_cfact_df.shape[0])
            Y_hat_cfact_df["type"] = "Counterfactual Prediction"
            Y_hat_cfact_df["std"] = std_hat_cf
            
            df = pd.concat([X_df, Y_fact_df, Y_cfact_df, Y_hat_fact_df, Y_hat_cfact_df])
            fig = px.line(df, x = "time",y = "PaPv",color="type", error_y ="std", title = f"{chart_type} longitudinal predictions - dimension {dim}") 

            plot_filename = os.path.join(self.train_dir, f'Predictions_global_step_{self.global_step}.png')
            fig.write_image(plot_filename, engine="kaleido")
            print(f'Saved figure at: {plot_filename}')

            # Optionally log the plot to wandb if logging is enabled
            if self.log_wandb:
                wandb.log({"Predictions": fig})
        
    def plot_SDENN_output(self, SDE_output):
        print('PLOTTING')
        # Ensure the 'train_dir' exists, which already includes the 'figures' subdirectory
        if not os.path.exists(self.train_dir):
            os.makedirs(self.train_dir)
            print(f'Created directory: {self.train_dir}')

        # Convert the PyTorch tensor to NumPy
        data_np = SDE_output.cpu().numpy()
        data_np = data_np.mean(1)  # Mean across the samples, keep separate for time and batch
        print('data_np', data_np.shape)

        # Create a new figure using Plotly
        fig = go.Figure()
        
        # Number of samples (lines) and time steps
        batch_dim, num_time_steps = data_np.shape
        
        # Adding each sample as a separate trace in the plot
        for i in range(batch_dim):
            fig.add_trace(go.Scatter(x=list(range(num_time_steps)), y=data_np[i],
                                    mode='lines', name=f'Sample {i+1}'))
        
        # Update layout
        fig.update_layout(title='SDE Model Output Over Time',
                        xaxis_title='Time Step',
                        yaxis_title='Output Value',
                        legend_title='Samples')
        
        # Save the plot to the directory
        plot_filename = os.path.join(self.train_dir, f'sde_output_global_step_{self.global_step}.png')
        fig.write_image(plot_filename, engine="kaleido")
        print(f'Saved figure at: {plot_filename}')

        # Optionally log the plot to wandb if logging is enabled
        if self.log_wandb:
            wandb.log({"SDE_Output_Over_Time": fig})