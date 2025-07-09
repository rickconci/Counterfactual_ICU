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
from torch.special import erf

import torchsde

import lightning as L
from lightning import LightningModule


#from CV_data_6_new import create_cv_data
from CV_data_beta import create_cv_data
from models_rd import Raindrop_v2

from utils_beta import CV_params, CV_params_divisors,  _stable_division, LinearScheduler, MLPSimple, CV_params_prior_mu, CV_params_prior_sigma, CV_params_max_min_2_5STD, CV_params_max_min_2STD, sigmoid_scale, normalize_latent_output, sigmoid
from utils_beta import select_tensor_by_index_list_advanced, scale_unnormalised_experts, normalise_expert_data
from plotting_beta import plot_trajectories_simple, plot_factuals_counterfactuals, plot_SDENN_output, plot_grouped_mse


# <<< Global DEBUG flag for model_beta.py, to be set by instance >>>
# This is more of a placeholder if a module-level default is ever needed,
# but instance-level self.debug passed from main_beta.py is the primary control.
DEBUG = False 


class Hybrid_VAE_SDE(LightningModule):

    def __init__(self, use_encoder, start_dec_at_treatment, variational_sampling, 
                 #Encoder
                 encoder_input_dim, encoder_hidden_dim, encoder_SDENN_dims,expert_latent_dims,
                 encoder_num_layers, variational_encoder, encoder_w_time, encoder_reverse_time,
                 use_2_5std_encoder_minmax, 
                 # New static fusion params
                 static_input_dim, static_hidden_dim, fusion_hidden_dim,
                 #SDE params
                 normalise_for_SDENN, prior_tx_sigma, prior_tx_mu, self_reverting_prior_control, 
                 SDE_input_state, include_time, 
                 theta, SDE_control_weighting, 
                #SDE model params
                num_samples, SDEnet_hidden_dim, SDEnet_depth, SDEnet_out_dims, final_activation, use_batch_norm,
                #decoder params
                decoder_output_dims, log_lik_output_scale, normalised_data, 
                #loss
                KL_weighting_SDE,
                #admin
                train_dir, learning_rate, log_wandb, adjoint, plot_every, batch_size,
                dataset,
                debug=False # <<< Add debug flag >>>
                ):
        super().__init__()
        self.debug = debug # <<< Store debug flag >>>
        global DEBUG
        DEBUG = self.debug # <<< Update module-level DEBUG if needed, primarily use self.debug >>>

        if self.debug: print(f"[DEBUG] Hybrid_VAE_SDE __init__: Initializing... adjoint={adjoint}, use_encoder={use_encoder}, normalise_for_SDENN={normalise_for_SDENN}")

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
        self.encoder_SDENN_dims = encoder_SDENN_dims
        self.encoder_output_dim = encoder_SDENN_dims + expert_latent_dims

        self.variational_encoder = variational_encoder
        self.use_2_5std_encoder_minmax = use_2_5std_encoder_minmax
        
        temporal_embedding_dim = 0 # To store the output dim of the temporal encoder

        if use_encoder == 'raindrop':
            # For Raindrop, d_model is its internal feature size.
            # Its output before projection will be d_model + d_pe if sensor_wise_mask is False
            temporal_embedding_dim = encoder_hidden_dim + 16 # d_model + d_pe
            self.temporal_encoder = Raindrop_v2(
                d_inp=encoder_input_dim,
                d_model=encoder_hidden_dim, 
                output_dim=temporal_embedding_dim, # Not used since we commented out final layer
                nhead=4, 
                nhid=128, 
                nlayers=encoder_num_layers,
                static=False, 
                debug=self.debug
            )
        elif use_encoder != 'none':
            temporal_embedding_dim = encoder_hidden_dim
            self.temporal_encoder = Encoder(input_dim = encoder_input_dim,
                                     hidden_dim = encoder_hidden_dim,
                                     latent_dim = temporal_embedding_dim, # GRU output is hidden_dim
                                     expert_latent_dims = expert_latent_dims,
                                     variational = variational_encoder,
                                     encode_with_time_dim = encoder_w_time,
                                     encoder_num_layers = encoder_num_layers,
                                     reverse = encoder_reverse_time,
                                     debug = self.debug) 
        else:
            self.temporal_encoder = None

        # --- New Static Encoder and Fusion Heads ---
        if use_encoder != 'none':
            self.static_encoder = MLPSimple(
                input_dim=static_input_dim,
                output_dim=static_hidden_dim,
                hidden_dim=static_hidden_dim,
                depth=2,
                activations=[nn.ReLU(), nn.ReLU()],
                debug=self.debug
            )
            # This MLP fuses the temporal and static embeddings
            self.fusion_mlp = MLPSimple(
                input_dim=temporal_embedding_dim + static_hidden_dim,
                output_dim=fusion_hidden_dim,
                hidden_dim=(temporal_embedding_dim + static_hidden_dim) // 2,
                depth=2,
                activations=[nn.ReLU(), nn.ReLU()],
                debug=self.debug
            )
            # Head 1: Predicts the initial state for all 16 expert ODE variables
            self.ode_latent_head = nn.Linear(fusion_hidden_dim, expert_latent_dims)
            # Head 2: Predicts the separate embedding for the neural SDE component
            self.neural_embedding_head = nn.Linear(fusion_hidden_dim, encoder_SDENN_dims)
        else:
            self.static_encoder = None
            self.fusion_mlp = None
            self.ode_latent_head = None
            self.neural_embedding_head = None
        # --- End New ---
        
        
        ### PRIOR PARAMS
        self.self_reverting_prior_control = self_reverting_prior_control
        self.prior_tx_sigma = prior_tx_sigma
        self.prior_tx_mu = prior_tx_mu


        #sigma_values = torch.tensor(list(CV_params_prior_sigma.values())).float()
        #sigma_values = sigma_values[:expert_latent_dims].view(1, -1)
        #self.register_buffer('sigma', sigma_values.clone())
        self.sigma = torch.tensor(self.prior_tx_sigma, dtype=torch.float32).to(self.device).unsqueeze(0)


        self.theta = torch.tensor(theta, dtype=torch.float).clone().view(1, -1).repeat(1, expert_latent_dims)

        ### LATENT MODEL  

        self.num_samples = num_samples
        self.expert_latent_dims = expert_latent_dims
        self.CV_params = CV_params
        
        self.divisors = torch.tensor([CV_params_divisors[key] for key in ['pa', 'pv', 's', 'sv', 
            'r_tpr_mod', 'f_hr_max', 'f_hr_min', 
            'r_tpr_max', 'r_tpr_min', 
            'ca', 'cv', 'k_width', 'p_aset', 'tau']], dtype=torch.float32)
        
       
        self.SDEnet_hidden_dim = SDEnet_hidden_dim
        self.SDEnet_depth = SDEnet_depth
        self.SDEnet_out_dims = SDEnet_out_dims
        self.SDE_control_weighting = SDE_control_weighting

        net_input_dims = self.encoder_output_dim if SDE_input_state == 'full' else self.encoder_output_dim - len(encoder_input_dim)
        net_input_dims = net_input_dims + 2 if include_time else net_input_dims 

        activations = {
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
            'none': None
        }
        final_activation_real = activations[final_activation.lower()]
        
        self.SDEnet = MLPSimple(input_dim = net_input_dims, 
                                output_dim = SDEnet_out_dims, 
                                hidden_dim = SDEnet_hidden_dim, 
                                depth = SDEnet_depth, 
                                activations = [nn.Tanh() for _ in range(SDEnet_depth)], 
                                final_activation=final_activation_real, 
                                use_batch_norm=use_batch_norm,
                                debug=self.debug) # <<< Pass debug flag (if MLPSimple is modified) >>>

        # Initialization trick from Glow.
        self.SDEnet.output_layer[0].weight.data.fill_(0.)
        self.SDEnet.output_layer[0].bias.data.fill_(0.)

        ### DECODER
        self.decoder_output_dims = decoder_output_dims
        self.normalised_data = normalised_data
        self.dataset = dataset

        ### LOSS
        self.MSE_loss = nn.MSELoss(reduction = "none")
        self.log_lik_output_scale = log_lik_output_scale
        self.KL_weighting_SDE = KL_weighting_SDE
        self.kl_scheduler = LinearScheduler(start=70, iters=600, startval=1.0, endval=0.01)
        self.save_hyperparameters()
        if self.debug: print(f"[DEBUG] Hybrid_VAE_SDE __init__: Initialization complete. Encoder output dim: {self.encoder_output_dim}, SDEnet input dims: {net_input_dims}")

        #plotting:
        self.mse_data_factual = [[] for _ in range(batch_size)]  
        self.mse_data_cf = [[] for _ in range(batch_size)]
        
    def forward_enc(self, input_vals, time_in, static=None, lengths=None):
        if self.debug: print(f"[DEBUG] Hybrid_VAE_SDE forward_enc: input_vals_shape={input_vals.shape}, time_in_shape={time_in.shape}, use_encoder={self.use_encoder}")

        if self.use_encoder == 'raindrop':
            # Raindrop expects src shape [maxlen, batch_size, features]
            # It also needs static features and lengths.
            z1, _, _ = self.enc_model(src=input_vals, static=static, times=time_in, lengths=lengths)
            # Raindrop doesn't produce a logvar, and its second output is 'distance' which we ignore here.
            return z1, None, 0

        elif self.use_encoder != 'none':
            if self.start_dec_at_treatment:
                if self.variational_encoder:
                    z1_mean, z1_logvar = self.enc_model(input_vals, time_in)
                    z1 = z1_mean.unsqueeze(1).repeat(1, self.num_samples, 1)
                    logqp0 = 0
                else:
                    z1 = self.enc_model(input_vals, time_in)
                    if self.debug: print(f"[DEBUG] Hybrid_VAE_SDE forward_enc (non-variational): Encoder output z1_shape (before repeat): {z1.shape}")
                    # The following line seems incorrect as sigmoid_scale is not a method of this class. Assuming it's a typo from original code.
                    # z1 = torch.cat([self.sigmoid_scale(z1[:,:self.expert_latent_dims], self.use_2_5std_encoder_minmax), z1[:, self.expert_latent_dims:] ], dim =-1)
                    z1 = z1.unsqueeze(1).repeat(1, self.num_samples, 1)
                    logqp0 = 0
                    z1_logvar = None
            else:
                z1 = input_vals.unsqueeze(1).repeat(1, self.num_samples, 1)
                logqp0 = 0
                z1_logvar = None
        else: # No encoder
            z1 = input_vals.unsqueeze(1).repeat(1, self.num_samples, 1)
            if self.debug: print(f"[DEBUG] Hybrid_VAE_SDE forward_enc (no encoder, no variational sampling): z1_shape={z1.shape}")
            logqp0 = 0
            z1_logvar = None


        if self.debug: print(f"[DEBUG] Hybrid_VAE_SDE forward_enc: Returning z1_shape={z1.shape}, z1_logvar_type={type(z1_logvar)}, logqp0_type={type(logqp0)}")
        return z1, z1_logvar, logqp0
    
    def apply_SDE_fun(self, t, y):
        if self.debug and t.item() % 10 == 0: # Avoid excessive printing
            print(f"[DEBUG] Hybrid_VAE_SDE apply_SDE_fun: t={t.item()}, y_shape={y.shape}, normalise_for_SDENN={self.normalise_for_SDENN}, include_time={self.include_time}, SDE_input_state={self.SDE_input_state}")

        if self.normalise_for_SDENN:
            SDNN_expert_input_state = normalise_expert_data(y[:, self.SDEnet_out_dims:self.expert_latent_dims+self.SDEnet_out_dims])
        else:
            SDNN_expert_input_state = y[:, self.SDEnet_out_dims:self.expert_latent_dims+self.SDEnet_out_dims]/self.divisors.to(self.device)

        #print('SDNN_expert_input_state', SDNN_expert_input_state.shape, SDNN_expert_input_state[0, :])

        if self.include_time:
            # Positional encoding in transformers for time-inhomogeneous posterior
            sde_latent_times = torch.full_like(y[:, 0], fill_value=t).unsqueeze(1)
            sin_time = torch.sin(sde_latent_times)
            cos_time = torch.cos(sde_latent_times)

            if self.SDE_input_state == 'full':
                input_state = torch.cat([SDNN_expert_input_state, y[:, self.SDEnet_out_dims+self.expert_latent_dims:]], dim=-1)
                SDE_NN_input = torch.cat((sin_time, cos_time, input_state), dim=-1)

            elif self.SDE_input_state == 'latents':
                input_state = torch.cat([SDNN_expert_input_state[:,2:], y[:, self.SDEnet_out_dims+self.expert_latent_dims:]], dim=-1)
                SDE_NN_input = torch.cat((sin_time, cos_time, input_state), dim=-1)

        else:
            if self.SDE_input_state == 'full':
                SDE_NN_input = torch.cat([SDNN_expert_input_state, y[:, self.SDEnet_out_dims+self.expert_latent_dims:]], dim=-1)

            elif self.SDE_input_state == 'latents':
                SDE_NN_input = torch.cat([SDNN_expert_input_state[:,2:], y[:, self.SDEnet_out_dims+self.expert_latent_dims:]], dim=-1)

        #print('SDE_NN_input shape', SDE_NN_input.shape)
        #print('SDE_NN_input example', SDE_NN_input[0,:])
        SDE_NN_output_latents = self.SDEnet(SDE_NN_input) 
        #print('SDE_NN_output_latents', SDE_NN_output_latents.shape)
        #print('SDE_NN_output_latents example', SDE_NN_output_latents[0, :])
        has_nonzero = SDE_NN_output_latents.ne(0.).any()
        #print('SDE_NN Has non-0 OUTPUT??', has_nonzero)
        if self.debug and t.item() % 10 == 0:
            print(f"[DEBUG] Hybrid_VAE_SDE apply_SDE_fun: SDE_NN_input_shape={SDE_NN_input.shape}, SDE_NN_output_latents_shape={SDE_NN_output_latents.shape}")
        return SDE_NN_output_latents
    

    def f(self, t, y, Tx, time_to_treatment):  # Approximate posterior drift.
        if self.debug and t.item() % 10 == 0:
            print(f"[DEBUG] Hybrid_VAE_SDE f (posterior drift): t={t.item()}, y_shape={y.shape}, Tx_example={Tx[0].item() if Tx.numel() > 0 else 'N/A'}, time_to_treatment_example={time_to_treatment[0].item() if time_to_treatment.numel() > 0 else 'N/A'}")
        batch_size = y.shape[0]
        #print('y', y.shape)
        i_ext_1 = y[:,0].unsqueeze(1)
        i_ext_2 = y[:,1].unsqueeze(1) 
        p_a = y[:,2].unsqueeze(1) 
        p_v = y[:,3].unsqueeze(1) 
        s_reflex = y[:, 4] .unsqueeze(1) 
        sv = y[:, 5].unsqueeze(1)
        r_tpr_mod = y[:, 6].unsqueeze(1)
        f_hr_max = y[:, 7].unsqueeze(1)
        f_hr_min = y[:, 8].unsqueeze(1)
        r_tpr_max = y[:, 9].unsqueeze(1)
        r_tpr_min= y[:, 10].unsqueeze(1)
        ca = y[:, 11].unsqueeze(1)
        cv = y[:, 12].unsqueeze(1)
        k_width = y[:, 13].unsqueeze(1)
        p_aset = y[:, 14].unsqueeze(1)
        tau = y[:, 15].unsqueeze(1)

        ##print('fixed params 1', f_hr_min[0].item(), f_hr_max[0].item(), r_tpr_max[0].item(), r_tpr_min[0].item(), r_tpr_mod[0].item())
        ##print('fixed params 2', ca[0].item(), cv[0].item(), tau[0].item(), k_width[0].item(), p_aset[0].item(), )
            

        #print('TIME, i_ext, p_a pv, s, sv', t.item(), i_ext_1[0].item(),i_ext_2[0].item(), p_a[0].item(), p_v[0].item(), s_reflex[0].item(), sv[0].item())   
        
        if t.item() >= time_to_treatment:
            #print('Treatment has started! Estimating effect', t.item(), time_to_treatment )
            #the neural network is trying to learn the ultimate treatment effect!! this means both fluid function AND the v_fun. V_fun determines the unknown tx_effect multiplied (beyond) the model, hence to be learned
            dt_i_ext_SDE = self.apply_SDE_fun(t, y) * self.SDE_control_weighting
            #print('dt_i_ext_SDE NN', dt_i_ext_SDE.shape)
            #print('dt_i_ext_SDE', dt_i_ext_SDE[:3, :])
            dt_i_ext_SDE_1 = dt_i_ext_SDE[:,0].unsqueeze(1)
            dt_i_ext_SDE_2 = dt_i_ext_SDE[:,1].unsqueeze(1)

        else:
            dt_i_ext_SDE = torch.zeros([y.shape[0], self.SDEnet_out_dims])
        
        #T is binary and indicates whether a treatment was given or not. 
        #this is an important step to then create the counterfactuals when we set T as the opposite of what it's trained on (T_cf)
        i_ext_SDE_1 = Tx[:,None] * i_ext_1
        i_ext_SDE_2 = Tx[:,None] * i_ext_2 

        ##print('i_ext_SDE Has non-0 OUTPUT??', has_nonzero)
        
        ##print('i_ext', i_ext_SDE.shape)
        ##print('i_ext example', i_ext_SDE[:4, :])

        #print('time, i_ext, dt_i_ext_SDE, pa, pv, sv', t.item(),i_ext_1[0].item(),i_ext_2[0].item(), dt_i_ext_SDE[0,0].item(), dt_i_ext_SDE[0,1].item(), p_a[0].item(), p_v[0].item(), sv[0].item())   

    
        f_hr = s_reflex * (f_hr_max - f_hr_min) + f_hr_min 
        r_tpr = s_reflex * (r_tpr_max - r_tpr_min) + r_tpr_min + r_tpr_mod
        
        dva_dt = -1. * (p_a - p_v) / (r_tpr + 1e-7)  + sv * f_hr
        
        ##print('dvv_dt pre iext', -1.*dva_dt[:4, 0])
        dvv_dt = -1. * dva_dt + i_ext_SDE_1
        ##print('dvv_dt post iext', dvv_dt[:4, 0])

        ##print('f_hr, dva_dt, r_tpr, dvv_dt', f_hr.shape, dva_dt.shape, r_tpr.shape, dvv_dt.shape)
        ##print('f_hr, dva_dt, r_tpr, dvv_dt', f_hr[0], dva_dt[0], r_tpr[0], dvv_dt[0])

        dpa_dt = dva_dt / (ca )
        dpv_dt = dvv_dt / (cv ) 
        ds_dt = (1. / tau) * (1. - 1. / (1 + torch.exp(-k_width * (p_a - p_aset))) - s_reflex)
                                  #(1. - 1. / (1 + torch.exp(-k_width * (p_a - p_aset))) - s)
        #self.sigmoid(k_width * (p_a - p_aset)) - s_reflex)
        
        dsv_dt = i_ext_SDE_2 
        ##print('i_ext_SDE, dsv_dt post iext, dt_i_ext_SDE', i_ext_SDE[0].item(), dsv_dt[0].item(), dt_i_ext_SDE[0].item())

        ##print('dpa_dt, dpv_dt, ds_dt, dsv_dt', dpa_dt.shape, dpv_dt.shape, ds_dt.shape, dsv_dt.shape)
        dt_r_tpr_mod = torch.zeros([batch_size, 1])  
        dt_f_hr_max = torch.zeros([batch_size, 1])
        dt_f_hr_min = torch.zeros([batch_size, 1])
        dt_r_tpr_max = torch.zeros([batch_size, 1])
        dt_r_tpr_min = torch.zeros([batch_size, 1])
        dt_ca = torch.zeros([batch_size, 1])
        dt_cv = torch.zeros([batch_size, 1])
        dt_k_width = torch.zeros([batch_size, 1])
        dt_p_aset = torch.zeros([batch_size, 1])
        dt_tau = torch.zeros([batch_size, 1])
        underlying_params = torch.cat([dt_r_tpr_mod, dt_f_hr_max, dt_f_hr_min, dt_r_tpr_max, dt_r_tpr_min, dt_ca, dt_cv, dt_k_width, dt_p_aset, dt_tau ], dim=-1).to(self.device)
        diff_results = torch.cat([dt_i_ext_SDE_1, dt_i_ext_SDE_2, dpa_dt, dpv_dt, ds_dt, dsv_dt, ], dim=-1)
        final_f_out = torch.cat([diff_results, underlying_params], dim=-1)
        
        ##print('diff_results ', diff_results.shape, diff_results[0,:])
        #print('final_f_out ', final_f_out.shape, final_f_out[0,:])
        return final_f_out 
    
    def prior_diext_dt(self,t):
        factor = -2 * (t - 5) / 5
        exponential = torch.exp(-((t - 5) / 5) ** 2)
        diext_dt = torch.tensor([5/3 * factor * exponential]).to(self.device)
        #print('diext_dt', diext_dt.shape)
        return diext_dt.unsqueeze(1)
    
    def prior_rate_of_change_of_flow(self, t, volumes=50, durations=20, width_factor=3):
        volumes = torch.tensor(volumes, dtype=torch.float32).unsqueeze(0).to(self.device)
        durations = torch.tensor(durations, dtype=torch.float32).unsqueeze(0).to(self.device)
        width_factor = torch.tensor(width_factor, dtype=torch.float32).to(self.device)

        #print('t:', t.shape)
        if t.ndim == 0:  
            t = t.unsqueeze(0)

        means = durations / 2
        sigmas = width_factor * durations / 10
        A = volumes / (torch.sqrt(torch.tensor(np.pi)) * sigmas * erf((durations - means) / sigmas))
        #print('A:', A.shape)

        derivatives = -2 * A[:, None] * (t - means[:, None]) / sigmas[:, None]**2 * torch.exp(-((t - means[:, None]) / sigmas[:, None])**2)
        #print('derivatives', derivatives.shape)
        return derivatives

    def h(self, t, y):  # Prior drift.
        if self.debug and t.item() % 10 == 0:
            print(f"[DEBUG] Hybrid_VAE_SDE h (prior drift): t={t.item()}, y_shape={y.shape}")
        #self.mu = self.prior_rate_of_change_of_flow(t)
        self.mu = torch.tensor([0]).unsqueeze(1).to(self.device)
        expanded_mu = self.mu.repeat(y.size(0), y.shape[1])
        #print('theta h', self.theta.shape, self.theta[0,:])
        #print('mu h', expanded_mu[0,:])
        #print('y in h', y.shape, y[0,:])
        #print('mu -y ', expanded_mu[0,:] - y[0,:])

        return self.theta.to(self.device) * (expanded_mu - y)

    def f_aug(self, t, y):  # Drift for augmented dynamics with logqp term.
        if self.debug and t.item() % 10 == 0:
            print(f"[DEBUG] Hybrid_VAE_SDE f_aug: t={t.item()}, y_shape={y.shape}")
        i_ext = y[:, :self.SDEnet_out_dims]
        dt_all_dims = y[:, :self.encoder_output_dim + self.SDEnet_out_dims]  # this is i_ext, pa, pv, s, sv and other encoder latents that go straight to the SDENN
        Tx = y[:, -2]
        time_to_treatment = y[0, -1]
        
        #print('inputs to f', dt_all_dims.shape) # num_samples x sde_dims 
        
        f_res = self.f(t, dt_all_dims, Tx, time_to_treatment)
        
        if self.self_reverting_prior_control:
            i_ext_2 = i_ext[:, 1].unsqueeze(1)
            g_iext, h_iext  = self.g(t, i_ext_2), self.h(t, i_ext_2)
            f_iext = f_res[:,1].unsqueeze(1)
            #print('f', f_iext.shape, 'g', g_iext.shape, 'h', h_iext.shape)
            #print('f mean', f_iext.mean(), 'g mean', g_iext.mean(), 'h mean', h_iext.mean())
            #print('f', f_iext[:3,:], 'g ', g_iext[:3,:], 'h ', h_iext[:3,:] )
            
            #print('doing stable division!')
            u = _stable_division(f_iext - h_iext, g_iext)
            #print('u shape', u.shape)
            f_logqp = .5 * (u ** 2).sum(dim=1, keepdim=True)
        else:
            self.mu = None
            f_logqp = torch.zeros_like(y[:, 0]).unsqueeze(1).to(self.device)
        
        #print('f_logqp', f_logqp.shape)
        #print('f_logqp mean', f_logqp[:3,:])   

        encoder_to_SDENN_latents = torch.zeros(y.shape[0], self.encoder_output_dim - self.expert_latent_dims).to(self.device)
        f_out = torch.cat([f_res, encoder_to_SDENN_latents, f_logqp, torch.zeros_like(f_logqp), torch.zeros_like(f_logqp)], dim=1)
        #print('f_aug out', f_out.shape, f_out[0])
        return f_out
    
    def g(self, t, y):  
        if self.debug and t.item() % 10 == 0:
            print(f"[DEBUG] Hybrid_VAE_SDE g (diffusion): t={t.item()}, y_shape={y.shape}")
        #sigma is different for each values here! 
        expanded_sigma = self.sigma.repeat(y.size(0), y.shape[1]).to(self.device)
        #print('sigma g', expanded_sigma.shape, expanded_sigma[0] )
        return expanded_sigma

    def g_aug(self, t, y):  # Diffusion for augmented dynamics with logqp term.
        if self.debug and t.item() % 10 == 0:
            print(f"[DEBUG] Hybrid_VAE_SDE g_aug: t={t.item()}, y_shape={y.shape}")
        y_SDE = y[:, 0:self.SDEnet_out_dims+self.expert_latent_dims+self.encoder_SDENN_dims]
        Tx = y[:, -2] # Tx is the second to last element
        dt_expert_dims = torch.zeros(y.shape[0], self.expert_latent_dims).to(self.device)
        encoder_to_SDENN_latents = torch.zeros(y.shape[0], self.encoder_output_dim - self.expert_latent_dims).to(self.device)

        #print('dt_expert_dims gaug', dt_expert_dims.shape)
        #print('encoder_to_SDENN_latents gaug', encoder_to_SDENN_latents.shape)
        g_res = self.g(t, y_SDE)
        #print('g', g_res.shape, g_res[0])
        g_logqp = torch.zeros(y.size(0), 1).to(y.device)

        g_out = torch.cat([g_res, dt_expert_dims, encoder_to_SDENN_latents,  g_logqp, torch.zeros_like(g_logqp), torch.zeros_like(g_logqp)], dim=1)
        #print('g out', g_out.shape, g_out[0])
        return g_out

    def forward_latent(self, init_latents, ts, Tx, time_to_tx):
        if self.debug:
            print(f"[DEBUG] Hybrid_VAE_SDE forward_latent: init_latents_shape={init_latents.shape}, ts_shape={ts.shape}, Tx_example={Tx[0].item() if Tx.numel() > 0 else 'N/A'}, time_to_tx_example={time_to_tx[0].item() if time_to_tx.numel() > 0 else 'N/A'}")
            print(f"[DEBUG] Hybrid_VAE_SDE forward_latent: sdeint_fn is {'adjoint' if self.sdeint_fn == torchsde.sdeint_adjoint else 'standard'}")
        #inputs of shape [batch x num_samples x dim ]
        batch_size = init_latents.shape[0]
        Tx_expanded = Tx.unsqueeze(1).unsqueeze(2).repeat(1, self.num_samples, 1).to(init_latents)
        time_to_tx = time_to_tx.unsqueeze(1).unsqueeze(2).repeat(batch_size, self.num_samples, 1).to(init_latents)
        i_ext = torch.zeros(batch_size,self.num_samples, self.SDEnet_out_dims).to(init_latents)
        log_path = torch.zeros(batch_size,self.num_samples, 1).to(init_latents)
        #print('ts',ts.shape)
        #print('i_ext ', i_ext.shape)
        #print('init_latents ', init_latents.shape)
        #print('log_path', log_path.shape)
        #print(f"Tx_expanded shape: {Tx_expanded.shape}")
        #print(f"time_to_tx shape: {time_to_tx.to(init_latents).shape}")

        
        aug_y0 = torch.cat([i_ext, init_latents,  log_path, Tx_expanded, time_to_tx], dim=-1) 
        #print('aug_y0', aug_y0.shape, aug_y0[0, 0,:])
        dim_aug = aug_y0.shape[-1]
        aug_y0 = aug_y0.reshape(-1,dim_aug)

        #print('aug_y0', aug_y0.shape) #this will be num_samples x dim = 512 x 4
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
        
        #print('len(ts)', len(ts))
        #print('dim_aug', dim_aug)
        #print('aug_ys pre_reshape',aug_ys.shape)
        
        aug_ys = aug_ys.view(len(ts), batch_size, self.num_samples, dim_aug).permute(1,2,0,3)
        #reshape(self.num_samples,-1,  len(ts),dim_aug) # reshape for # batch_size x num_samples x times x dim
        
        i_ext_path = aug_ys[:, :, :, :self.SDEnet_out_dims]
        latent_out = aug_ys[:, :, :, self.SDEnet_out_dims:self.expert_latent_dims+self.SDEnet_out_dims]
        logqp_path = aug_ys[:,: , -1, -3]    #.mean(dim=0)  # KL(t=0) + KL(path).
       
        #print('latent_out end of latent ', latent_out.shape, latent_out[0, 0, :, 0])
        
        #print('i_ext_path', i_ext_path.shape, i_ext_path[:4,:2,:])
        #print('logqp_path_extracted', logqp_path.shape)
       
        if self.debug: print(f"[DEBUG] Hybrid_VAE_SDE forward_latent: SDE integration complete. zs_shape={latent_out.shape}, logqp_shape={logqp_path.shape}")
        return latent_out, logqp_path, i_ext_path

    def forward_dec(self, latent_out):
        if self.debug: 
            print(f"[DEBUG] Hybrid_VAE_SDE forward_dec: latent_out_shape={latent_out.shape}")
            #print('latent_out', latent_out[0, 0, :, 0])
            #print('latent_out', latent_out[0, 1, :, 0])
            #print('latent_out', latent_out[1, 0, :, 0])
            #print('latent_out', latent_out[1, 1, :, 0])
            #print('latent device', latent_out.device)
            if self.normalised_data:
                latent_out = self.normalise_expert_data(latent_out)
            else:
                divisors = self.divisors.view(1, 1, 1, self.expert_latent_dims).to(latent_out.device)
                latent_out = latent_out / divisors

            output_traj = select_tensor_by_index_list_advanced(latent_out, self.decoder_output_dims)
            #print('output_traj', output_traj.shape, output_traj[0, 0, :, :])

            if self.debug: print(f"[DEBUG] Hybrid_VAE_SDE forward_dec: decoded_mean_shape={output_traj.shape}")
            return output_traj

  

    def compute_factual_loss(self, predicted_traj, true_traj, logqp):
        if self.debug: print(f"[DEBUG] Hybrid_VAE_SDE compute_factual_loss: Shapes: predicted_traj={predicted_traj.shape}, true_traj={true_traj.shape}, logqp_mean={logqp.mean().item() if logqp.numel() > 0 else 'N/A'}")
        
        true_traj_expanded = true_traj.unsqueeze(1).repeat(1, self.num_samples, 1, 1) 
        
        #print("predicted_traj, true_traj_expanded",predicted_traj.shape, true_traj_expanded.shape)
        mse_val = self.MSE_loss(predicted_traj, true_traj_expanded)
        #print(\"mse_val mean\\", mse_val.mean(dim=(1,2,3)).shape)
        
        # Gaussian NLL
        #var = torch.full_like(predicted_traj, self.log_lik_output_scale**2)
        logpy = distributions.Normal(loc=predicted_traj, scale=self.log_lik_output_scale).log_prob(true_traj_expanded).sum(dim=(2,3)).mean(dim=1)
        #print(\\"logpy mean\\", logpy.mean().shape)

        current_kl_weight = self.kl_scheduler.val
        self.kl_scheduler.step()
        
        loss = -logpy.mean() + self.KL_weighting_SDE * current_kl_weight * logqp.mean() # NLL + KL
        
        #print(\\"-logpy.mean() \\", -logpy.mean())
        #print(\\\"KL term \\\", self.KL_weighting_SDE * logqp.mean())
        #print(\"TOTAL LOSS \", loss)

        # Corrected debug print statement
        total_loss_val = loss.item()
        nll_val = (-logpy.mean()).item()
        kl_val = (logqp.mean()).item() # Assuming logqp is the KL divergence term directly
        if self.debug: 
            print(f"[DEBUG] Hybrid_VAE_SDE compute_factual_loss: total_loss={total_loss_val:.4f}, nll_component={nll_val:.4f}, kl_component={kl_val:.4f}, kl_weight={current_kl_weight:.4f}")
        
        return loss, -logpy.mean(), logqp.mean()

    def compute_counterfactual_loss(self, true_fact, true_cf, pred_fact, pred_cf):
        if self.debug: print(f"[DEBUG] Hybrid_VAE_SDE compute_counterfactual_loss: true_fact_shape={true_fact.shape}, pred_cf_shape={pred_cf.shape}")
        #print('true_fact:', true_fact.shape, true_fact[0,:,:] )
        #print('true_cf:', true_cf.shape, true_cf[0,:,:])
        #print('pred_fact:', pred_fact.shape, pred_fact.mean(1)[0,:,:])
        #print('pred_cf:', pred_cf.shape, pred_cf.mean(1)[0,:,:])
       
        # RECON LOSS
        # MSE loss between the Y and the MEAN of the SDE samples predictions, which includes expert and SDE in hybrid 
        mse_cf = torch.sqrt(self.MSE_loss(true_cf, pred_cf.mean(1))).mean()
        
        # Now find the mean of the standard devs of the predictions across the SDE samples
        std_preds_cf = pred_cf.std(1).mean()

        # Individual Treatment Effect computed as the difference between Y_cf and Y
        ite = (true_cf- true_fact)
        #print('ite:', ite.shape)

        # Predicted Individual Treatment Effect computed as the difference between the mean predictions of Y_hat_cf and Y_hat
        ite_hat = (pred_cf.mean(1) - pred_fact.mean(1))
        #print('ite_hat:', ite_hat.shape)

        # MSE of the ITE
        mse_ite = torch.sqrt(self.MSE_loss(ite, ite_hat)).mean()
        #print('mse_ite:', mse_ite)    

        if self.debug: print(f"[DEBUG] Hybrid_VAE_SDE compute_counterfactual_loss: mse_fact={mse_cf.item()}, mse_cf={mse_cf.item()}")
        return mse_cf, mse_ite, std_preds_cf
        
    def _prepare_encoder_input(self, X, init_states):
        """Prepares the input for the `forward_enc` method based on whether an encoder is used."""
        if self.use_encoder != 'none':
            # When using an encoder, provide only the observable variables.
            # The encoder will infer the full latent state.
            X_for_encoder = select_tensor_by_index_list_advanced(X, [0, 1, 2, 3])
        else: 
            # When not using an encoder, we manually construct the initial latent state.
            # This state must match the dimensions expected by the SDE dynamics.
            # It consists of the control signal (i_ext, starts at 0) and the expert variables.
            batch_size = X.shape[0]
            zeros_for_i_ext = torch.zeros(batch_size, self.SDEnet_out_dims, device=self.device)
            expert_inits = init_states[:, :self.expert_latent_dims]
            X_for_encoder = torch.cat([zeros_for_i_ext, expert_inits], dim=1)
        
        if self.debug:
            print(f"[DEBUG] _prepare_encoder_input: use_encoder='{self.use_encoder}', output_shape={X_for_encoder.shape}")
        
        return X_for_encoder

    def training_step(self, batch, batch_idx):
        if self.debug and batch_idx == 0:
            print(f"[DEBUG] Hybrid_VAE_SDE training_step: batch_idx={batch_idx}, first_element_type={type(batch[0])}")

        if self.dataset == "mimic":
            # Assuming mimic loader provides: X, X_mask, Y, T, ... , meds_in, static
            X, X_mask, Y, T, Y_cf, p, init_states, time_pre, time_post, time_FULL, full_fact_traj, full_cf_traj, meds_in, static_features = batch
        else:
            # Synthetic data path
            X, Y, T, Y_cf, p, init_states, time_pre, time_post, time_FULL, full_fact_traj, full_cf_traj = batch
            X_mask = torch.ones_like(X) # Create dummy mask for synthetic
            static_features = None
        batch_size = X.shape[0]
        
        if self.debug and batch_idx == 0:
            print(f"  X shape: {X.shape}, X snippet:\n{X[0, :2, :2]}")
            print(f"  X_mask shape: {X_mask.shape}, X_mask snippet:\n{X_mask[0, :2, :2]}")
            if static_features is not None:
                print(f"  static_features shape: {static_features.shape}, snippet: {static_features[0]}")

        if self.use_encoder != 'none':
            if self.use_encoder == 'raindrop':
                # Raindrop expects src shape: [seq_len, batch_size, features]
                X_t = X.permute(1, 0, 2)
                time_pre_t = time_pre.permute(1, 0)
                mask_t = X_mask.permute(1, 0, 2)
                X_with_mask = torch.cat([X_t, mask_t], dim=2)
                lengths = X_mask.sum(dim=1)[:, 0]
                
                # Get temporal embedding from Raindrop
                temporal_embedding, _, _ = self.temporal_encoder(X_with_mask, time_pre_t, static=None, lengths=lengths)
                if self.debug and batch_idx == 0: print(f"  temporal_embedding (from Raindrop) shape: {temporal_embedding.shape}, snippet: {temporal_embedding[0, :4]}")

            else: # GRU Encoder path
                X_for_encoder = self._prepare_encoder_input(X, init_states)
                temporal_embedding, z1_logvar, logqp0 = self.forward_enc(X_for_encoder, time_pre)
                if self.debug and batch_idx == 0: print(f"  temporal_embedding (from GRU) shape: {temporal_embedding.shape}, snippet: {temporal_embedding[0, :4]}")

            # Get static and fused embedding
            static_embedding = self.static_encoder(static_features)
            if self.debug and batch_idx == 0: print(f"  static_embedding shape: {static_embedding.shape}, snippet: {static_embedding[0, :4]}")
            
            fused_embedding = torch.cat([temporal_embedding, static_embedding], dim=-1)
            if self.debug and batch_idx == 0: print(f"  fused_embedding shape: {fused_embedding.shape}, snippet: {fused_embedding[0, :4]}")

            fused_rep = self.fusion_mlp(fused_embedding)
            if self.debug and batch_idx == 0: print(f"  fused_rep shape: {fused_rep.shape}, snippet: {fused_rep[0, :4]}")

            # Get the two separate outputs from the heads
            predicted_ode_latents = self.ode_latent_head(fused_rep).unsqueeze(1).repeat(1, self.num_samples, 1)
            if self.debug and batch_idx == 0: print(f"  predicted_ode_latents shape: {predicted_ode_latents.shape}, snippet:\n{predicted_ode_latents[0, 0, :4]}")
            
            neural_embedding = self.neural_embedding_head(fused_rep).unsqueeze(1).repeat(1, self.num_samples, 1)
            if self.debug and batch_idx == 0: print(f"  neural_embedding shape: {neural_embedding.shape}, snippet:\n{neural_embedding[0, 0, :4]}")

            logqp0 = 0 # Not variational for this setup
        
        elif self.use_encoder != 'none': # GRU Encoder path
            X_for_encoder = self._prepare_encoder_input(X, init_states)
            temporal_embedding, z1_logvar, logqp0 = self.forward_enc(X_for_encoder, time_pre)
            
            static_embedding = self.static_encoder(static_features)

            fused_embedding = torch.cat([temporal_embedding, static_embedding], dim=-1)
            fused_rep = self.fusion_mlp(fused_embedding)
            
            predicted_ode_latents = self.ode_latent_head(fused_rep).unsqueeze(1).repeat(1, self.num_samples, 1)
            neural_embedding = self.neural_embedding_head(fused_rep).unsqueeze(1).repeat(1, self.num_samples, 1)
            
        else:
            # No encoder path
            X_for_encoder = self._prepare_encoder_input(X, init_states)
            z1_encoder, z1_logvar, logqp0 = self.forward_enc(X_for_encoder, time_pre) # z1_encoder is just init_states here
            predicted_ode_latents = z1_encoder[:, :, :self.expert_latent_dims]
            neural_embedding = z1_encoder[:, :, self.expert_latent_dims:]


        # Prepare the SDE initial state using the consistent hybrid approach
        z1_for_sde = self._prepare_sde_initial_state(predicted_ode_latents, neural_embedding, init_states)
        
        latent_traj, logqp_path, i_ext_path = self.forward_latent(
            init_latents=z1_for_sde,
            ts=time_post[0, :],
            Tx=T,
            time_to_tx=torch.zeros(batch_size).to(self.device)
        )

        decoded_traj = self.forward_dec(latent_traj)
        
        # Use logqp0 from GRU if available, otherwise it's 0
        kl_logqp = logqp0 if self.use_encoder not in ['raindrop', 'none'] else 0
        total_logqp = kl_logqp + logqp_path

        loss, nll, kl_div = self.compute_factual_loss(predicted_traj=decoded_traj, true_traj=Y, logqp=total_logqp)
        
        # LOGGING
        self.log('train_total_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_NLL', nll, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_KL', kl_div, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        
        return loss

    def _prepare_sde_initial_state(self, predicted_ode_latents, neural_embedding, init_states):
        """
        Prepares the initial state for the SDE by combining the interpolated initial
        conditions with the encoder's two-headed output.
        """
        if self.debug:
            print(f"[DEBUG] _prepare_sde_initial_state:")
            print(f"  predicted_ode_latents shape: {predicted_ode_latents.shape}, snippet:\n{predicted_ode_latents[0, 0, :4]}")
            print(f"  neural_embedding shape: {neural_embedding.shape}, snippet:\n{neural_embedding[0, 0, :4]}")
            print(f"  init_states shape: {init_states.shape}, snippet: {init_states[0, :4]}")

        # Number of variables provided by the IC tensor
        num_ic_vars = init_states.shape[-1]

        # Part 1: Take the accurate interpolated values
        interpolated_part = init_states.unsqueeze(1).repeat(1, self.num_samples, 1)

        # Part 2: Take the inferred values for the remaining ODE variables from the specific head
        inferred_part = predicted_ode_latents[:, :, num_ic_vars:]
        
        # Part 3: The separate neural embedding
        neural_part = neural_embedding

        # Concatenate to form the full initial state for the SDE.
        # Note: The neural part comes *after* the expert ODE part.
        expert_part = torch.cat([interpolated_part, inferred_part], dim=-1)
        z1_for_sde = torch.cat([expert_part, neural_part], dim=-1)


        if self.debug:
            print(f"  final z1_for_sde shape: {z1_for_sde.shape}, snippet:\n{z1_for_sde[0, 0, :6]}")

        return z1_for_sde

    def validation_step(self, batch, batch_idx):
        if self.debug and batch_idx == 0: print(f"[DEBUG] Hybrid_VAE_SDE validation_step: batch_idx={batch_idx}")
        #print("VALIDATION")
        if self.dataset == "mimic":
            X, X_mask, Y, T, Y_cf, p, init_states, time_pre, time_post, time_FULL, full_fact_traj, full_cf_traj, meds_in, static_features = batch
        else:
            X, Y, T, Y_cf, p, init_states, time_pre, time_post, time_FULL, full_fact_traj, full_cf_traj = batch
            X_mask = torch.ones_like(X)
            static_features = None
        batch_size = X.shape[0]

        if self.use_encoder != 'none':
            if self.use_encoder == 'raindrop':
                X_t = X.permute(1, 0, 2)
                time_pre_t = time_pre.permute(1, 0)
                mask_t = X_mask.permute(1, 0, 2)
                X_with_mask = torch.cat([X_t, mask_t], dim=2)
                lengths = X_mask.sum(dim=1)[:, 0]
                temporal_embedding, _, _ = self.temporal_encoder(X_with_mask, time_pre_t, static=None, lengths=lengths)
                logqp0 = 0
            else: # GRU Encoder
                X_for_encoder = self._prepare_encoder_input(X, init_states)
                temporal_embedding, z1_logvar, logqp0 = self.forward_enc(X_for_encoder, time_pre)

            static_embedding = self.static_encoder(static_features)
            fused_embedding = torch.cat([temporal_embedding, static_embedding], dim=-1)
            fused_rep = self.fusion_mlp(fused_embedding)

            predicted_ode_latents = self.ode_latent_head(fused_rep).unsqueeze(1).repeat(1, self.num_samples, 1)
            neural_embedding = self.neural_embedding_head(fused_rep).unsqueeze(1).repeat(1, self.num_samples, 1)

        else: # No encoder
            X_for_encoder = self._prepare_encoder_input(X, init_states)
            z1_encoder, z1_logvar, logqp0 = self.forward_enc(X_for_encoder, time_pre)
            predicted_ode_latents = z1_encoder[:, :, :self.expert_latent_dims]
            neural_embedding = z1_encoder[:, :, self.expert_latent_dims:]


        # Prepare the SDE initial state using the consistent hybrid approach
        z1_for_sde = self._prepare_sde_initial_state(predicted_ode_latents, neural_embedding, init_states)

        latent_traj, logqp_path, i_ext_path = self.forward_latent(init_latents=z1_for_sde,
                                                                    ts = time_post[0, :],
                                                                    Tx = T,
                                                                    time_to_tx = torch.zeros(batch_size).to(self.device))
        
        decoded_traj = self.forward_dec(latent_traj)

        total_logqp = logqp0 + logqp_path
        loss, nll, kl_div = self.compute_factual_loss(predicted_traj=decoded_traj, true_traj=Y, logqp=total_logqp)
        self.log('val_total_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_NLL', nll, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_KL', kl_div, on_step=False, on_epoch=True, prog_bar=True, logger=True)

        # ... (rest of validation_step remains)
        
           
    def test_step(self, batch, batch_idx):
        if self.debug and batch_idx == 0: print(f"[DEBUG] Hybrid_VAE_SDE test_step: batch_idx={batch_idx}")
        #print("TEST")
        if self.dataset == "mimic":
            X, X_mask, Y, T, Y_cf, p, init_states, time_pre, time_post, time_FULL, full_fact_traj, full_cf_traj, meds_in, static_features = batch
        else:
            X, Y, T, Y_cf, p, init_states, time_pre, time_post, time_FULL, full_fact_traj, full_cf_traj = batch
            X_mask = torch.ones_like(X)
            static_features = None
        batch_size = X.shape[0]

        if self.use_encoder != 'none':
            if self.use_encoder == 'raindrop':
                X_t = X.permute(1, 0, 2)
                time_pre_t = time_pre.permute(1, 0)
                mask_t = X_mask.permute(1, 0, 2)
                X_with_mask = torch.cat([X_t, mask_t], dim=2)
                lengths = X_mask.sum(dim=1)[:, 0]
                temporal_embedding, _, _ = self.temporal_encoder(X_with_mask, time_pre_t, static=None, lengths=lengths)
                logqp0 = 0
            else: # GRU
                X_for_encoder = self._prepare_encoder_input(X, init_states)
                temporal_embedding, z1_logvar, logqp0 = self.forward_enc(X_for_encoder, time_pre)

            static_embedding = self.static_encoder(static_features)
            fused_embedding = torch.cat([temporal_embedding, static_embedding], dim=-1)
            fused_rep = self.fusion_mlp(fused_embedding)

            predicted_ode_latents = self.ode_latent_head(fused_rep).unsqueeze(1).repeat(1, self.num_samples, 1)
            neural_embedding = self.neural_embedding_head(fused_rep).unsqueeze(1).repeat(1, self.num_samples, 1)

        else: # No encoder
            X_for_encoder = self._prepare_encoder_input(X, init_states)
            z1_encoder, z1_logvar, logqp0 = self.forward_enc(X_for_encoder, time_pre)
            predicted_ode_latents = z1_encoder[:, :, :self.expert_latent_dims]
            neural_embedding = z1_encoder[:, :, self.expert_latent_dims:]


        # Prepare the SDE initial state using the consistent hybrid approach
        z1_for_sde = self._prepare_sde_initial_state(predicted_ode_latents, neural_embedding, init_states)

        # ... (rest of test_step remains)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr = self.learning_rate)
        
        scheduler = {"monitor": "train_total_loss", "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode = "min", factor = 0.5, patience = 50)}
        return {"optimizer": optimizer, "lr_scheduler":scheduler}
    
    def on_save_checkpoint(self, checkpoint):
        #print('SAVING CHECKPOINT')
        # Manually add mu, sigma, theta to the checkpoint dictionary
        checkpoint['mu'] = self.mu
        checkpoint['sigma'] = self.sigma
        checkpoint['theta'] = self.theta

    def on_load_checkpoint(self, checkpoint):
        #print('LOADING CHECKPOINT')
        # Load mu, sigma, theta from the checkpoint dictionary if they exist
        if 'mu' in checkpoint:
            self.mu = checkpoint['mu']
        if 'sigma' in checkpoint:
            self.sigma = checkpoint['sigma']
        if 'theta' in checkpoint:
            self.theta = checkpoint['theta']


    def plot_mse_evolution(self, chart_type):
    
        fig = go.Figure()

        # Plot each batch element's factual MSE evolution
        for i, mse_list in enumerate(self.mse_data_factual):
            fig.add_trace(go.Scatter(
                x=list(range(len(mse_list))),
                y=sorted(mse_list),  # Sort if needed, or just plot as is
                mode='lines+markers',
                name=f'Factual Batch {i+1}'
            ))

        # Similarly for counterfactual MSEs
        for i, mse_list in enumerate(self.mse_data_cf):
            fig.add_trace(go.Scatter(
                x=list(range(len(mse_list))),
                y=sorted(mse_list),
                mode='lines+markers',
                name=f'Counterfactual Batch {i+1}'
            ))

        fig.update_layout(title="MSE Evolution Over Validation Steps",
                        xaxis_title="Validation Step",
                        yaxis_title="Mean Squared Error",
                        legend_title="Batch Element")

        plot_filename = os.path.join(self.train_dir, f'Grouped_MSE_{chart_type}_global_step_{self.global_step}.png')
        fig.write_image(plot_filename, engine="kaleido")
        #print(f'Saved figure at: {plot_filename}')

        # Optionally log the plot to wandb if logging is enabled
        if self.log_wandb:
            wandb.log({"Grouped MSE Plot": fig})

        fig.data = []







class Encoder(nn.Module):
    def __init__(self, input_dim, hidden_dim, latent_dim, expert_latent_dims, variational, encode_with_time_dim, encoder_num_layers, reverse=False, debug=False): # <<< Add debug flag >>>
        super(Encoder, self).__init__()
        self.debug = debug # <<< Store debug flag >>>
        if self.debug: print(f"[DEBUG] Encoder __init__: input_dim={input_dim}, hidden_dim={hidden_dim}, latent_dim={latent_dim}, variational={variational}, reverse={reverse}")

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
        if self.debug:
            print(f"[DEBUG] Encoder forward:")
            print(f"  x_shape={x.shape}, x snippet:\n{x[0, :2, :2]}")
            print(f"  t_shape={t.shape}, t snippet:\n{t[0, :2]}")

        if self.encode_with_time_dim: # this is how VDS does it 
            # Calculate the time differences
            t_diff = torch.zeros_like(t)
            t_diff[:, 1:] = t[:, 1:] - t[:, :-1]  # Forward differences
            t_diff[:, 0] = 0.
            t_diff = t_diff.unsqueeze(-1) 
            #print('Time differences shape:', t_diff.shape)  # Should match t's shape

            xt = torch.cat((x, t_diff), dim=-1)  # Concatenate along the feature dimension
            #print('Concatenated xt shape:', xt.shape)  # Expected: [batch_size, seq_length, input_dim + 1]
        
        else: # this is how Hyland does it 
            xt = x

        
        #rediscover the data mean and std so can convert in encoder output 
        input_mean_obs_dim =  x.mean([0,1]) #mean across batch & seq len
        input_std_obs_dim = x.std([0,1])    #std across batch & seq len
        

        # Reverse the sequence along the time dimension
        if self.reverse:
            xt = xt.flip(dims=[1])
            #print('reversed xt shape:', xt.shape)  # Should match xt's shape

        _, h0 = self.rnn(xt)
        #print('Output hidden state h0 shape:', h0.shape)  # Expected: [depth, batch_size, hidden_dim]
        #print('output_last_dim', h0[-1].shape)
        
        # Process the last hidden state to produce latent variables
        z0 = self.hid2lat(h0[-1])
        if self.debug:
            print(f"  h0 (last layer) shape: {h0[-1].shape}, snippet: {h0[-1][0, :4]}")
            print(f"  z0 (output) shape: {z0.shape}, snippet: {z0[0, :4]}")
        #print('z0 from hid to lat', z0.shape)
        if self.variational:
            
            z0_mean_expert = z0[:, :self.expert_latent_dims]
            z0_log_var_expert = z0[:, self.expert_latent_dims:self.expert_latent_dims ]
            z0_rest = z0[:, 2*self.expert_latent_dims:]

            scaled_expert_latents = self.sigmoid_scale(z0_mean_expert,input_mean_obs_dim,  input_std_obs_dim)
            z0_means = torch.cat([scaled_expert_latents, z0_rest], dim=-1)
            
            #print('z0_mean shape:', z0_mean_expert.shape)  # Expected: [batch_size, latent_dim]
            #print('z0_log_var shape:', z0_log_var_expert.shape)  # Expected: [batch_size, latent_dim]
            
            if self.debug: print(f"[DEBUG] Encoder forward (variational): z_mean_shape={z0_means.shape}, z_log_var_shape={z0_log_var_expert.shape}")
            return z0_means, z0_log_var_expert
        
        else:
            z0_mean_expert = z0[:, :self.expert_latent_dims]
            z0_rest = z0[:, self.expert_latent_dims:]

            #print('z0_mean_expert', z0_mean_expert[0,:4])
            #print('z0_rest', z0_rest.shape)

            #scaled_expert_latents = self.sigmoid_scale(z0_mean_expert)
            #scaled_expert_latents = z0_mean_expert
            z0_means = torch.cat([z0_mean_expert, z0_rest], dim=-1)

            #print('z0_mean shape:', z0_means.shape)  # Expected: [batch_size, latent_dim]
            #print('z0_means',z0_means[0,:4] )
            
            if self.debug: print(f"[DEBUG] Encoder forward (non-variational): out_shape={z0_means.shape}")
            return z0_means
