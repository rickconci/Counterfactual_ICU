import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
from torch import distributions, nn, optim

import torchsde
import wandb
import lightning as L
from lightning import LightningModule


from utils_7b import CV_params, CV_params_divisors,  _stable_division, LinearScheduler, MLPSimple, CV_params_prior_mu, CV_params_prior_sigma, CV_params_max_min_2_5STD, CV_params_max_min_2STD, sigmoid_scale, normalize_latent_output, sigmoid
from utils_7b import select_tensor_by_index_list_advanced, scale_unnormalised_experts, normalise_expert_data






class Hybrid_VAE_SDE_Encoder(LightningModule):

    def __init__(self, expert_latent_dims, path_control_dim, apply_path_SDE, 
                 prior_path_sigma, num_samples, self_reverting_prior_control, theta, SDE_control_weighting,
                 normalise_for_SDENN, include_time, SDEnet_hidden_dim, SDEnet_depth, final_activation,
                 KL_weighting_SDE, l1_lambda, log_lik_output_scale, 
                 normalised_data, train_dir, learning_rate, log_wandb, adjoint, plot_every
                 ):
        super().__init__()
        
        self.noise_type = "diagonal"  # required
        self.sde_type = "ito"  # required
        self.sdeint_fn = torchsde.sdeint_adjoint if adjoint else torchsde.sdeint

        self.expert_latent_dims = expert_latent_dims
        self.path_control_dim = path_control_dim
        self.apply_path_SDE = apply_path_SDE

        self.prior_path_sigma = prior_path_sigma
        self.num_samples = num_samples
        self.self_reverting_prior_control = self_reverting_prior_control
        
        ### PRIOR PARAMS
        mu_dict = CV_params_prior_mu.copy()
        sigma_dict = CV_params_prior_sigma.copy()
        keys_order = ['pa', 'pv', 's', 'sv', 'r_tpr_mod', 'f_hr_max', 'f_hr_min','r_tpr_max', 'r_tpr_min', 'ca', 'cv', 'k_width', 'p_aset', 'tau']
        mu_tensor = torch.tensor([mu_dict[key].item() for key in keys_order]).float().view(1, -1)
        sigma_tensor = torch.tensor([sigma_dict[key] for key in keys_order]).float().view(1, -1)
        logvar_tensor = 2 * torch.log(sigma_tensor)
        
        self.qy0_mean = nn.Parameter(mu_tensor, requires_grad=True)
        self.qy0_logvar = nn.Parameter(logvar_tensor, requires_grad=True)
        self.qy0_std = torch.exp(0.5 * self.qy0_logvar)
        
        self.py0_mean = mu_tensor.detach()  
        self.py0_logvar = logvar_tensor.detach()  
        self.py0_std = torch.exp(0.5 * self.py0_logvar)

        self.sigma = torch.tensor(self.prior_path_sigma, dtype=torch.float32).to(self.device).unsqueeze(0)
        self.theta = torch.tensor(theta, dtype=torch.float).clone().view(1, -1).repeat(1, expert_latent_dims)

        ### LATENT MODEL  

        net_input_dims = expert_latent_dims + 2 if include_time else expert_latent_dims 
        self.SDEnet_hidden_dim = SDEnet_hidden_dim
        self.SDEnet_depth = SDEnet_depth
        self.normalise_for_SDENN = normalise_for_SDENN
        self.include_time = include_time

        self.SDE_control_weighting = SDE_control_weighting

        activations = {
            'relu': nn.ReLU(),
            'tanh': nn.Tanh(),
            'none': None
        }
        final_activation_real = activations[final_activation.lower()]
        
        self.SDEnet = MLPSimple(input_dim = net_input_dims, 
                                output_dim = self.path_control_dim, 
                                hidden_dim = SDEnet_hidden_dim, 
                                depth = SDEnet_depth, 
                                activations = [nn.Tanh() for _ in range(SDEnet_depth)], 
                                final_activation=final_activation_real, 
                                use_batch_norm=False)
        
        # Initialization trick from Glow.
        self.SDEnet.output_layer[0].weight.data.fill_(0.)
        self.SDEnet.output_layer[0].bias.data.fill_(0.)
    
        ### LOSS
        self.l1_lambda = l1_lambda
        self.MSE_loss = nn.MSELoss(reduction = "none")
        self.log_lik_output_scale = log_lik_output_scale
        self.KL_weighting_SDE = KL_weighting_SDE
        self.kl_scheduler = LinearScheduler(start=70, iters=600, startval=1.0, endval=0.01)
        

        ### ADMIN
        self.train_dir = train_dir
        self.learning_rate = learning_rate
        self.log_wandb = log_wandb
        self.plot_every = plot_every

        ### Utils 
        self.normalised_data = normalised_data
        self.sdeint_fn = torchsde.sdeint_adjoint if adjoint else torchsde.sdeint
        self.divisors = torch.tensor([CV_params_divisors[key] for key in ['pa', 'pv', 's', 'sv', 
                'r_tpr_mod', 'f_hr_max', 'f_hr_min', 
                'r_tpr_max', 'r_tpr_min', 
                'ca', 'cv', 'k_width', 'p_aset', 'tau']], dtype=torch.float32)
        

        self.save_hyperparameters()
        
    def forward_enc(self,input_vals, time_in, MAP=False):

        batch_size = input_vals.shape[0]
        if len(input_vals.shape)==3:
            true_input = input_vals[:, 0, :]
        else:
            true_input = input_vals

        if MAP == False:
            eps = torch.randn(batch_size, self.qy0_mean.shape[1]).to(self.device)
            y0 = self.qy0_mean.to(self.device) + eps * self.qy0_std.to(self.device)
        else:
            y0 = self.qy0_mean.to(self.device).repeat(batch_size, 1)
        print('y0 sampled', y0.shape, y0[0, :])

        z1 = torch.cat((true_input[:, :2], y0[:, 2:]), dim=1)
        print('z1 variational', z1.shape)

        mean_q = self.qy0_mean.to(self.device)
        std_q = torch.clamp(self.qy0_std.to(self.device), min=1e-6)
        mean_p = self.py0_mean.to(self.device)
        std_p = torch.clamp(self.py0_std.to(self.device), min=1e-6)
        qy0 = distributions.Normal(loc=mean_q, scale=std_q)
        py0 = distributions.Normal(loc=mean_p, scale=std_p)
        logqp0 = distributions.kl_divergence(qy0, py0).sum(dim=1)

        z1 = z1.unsqueeze(1).repeat(1, self.num_samples, 1)
        print('qy0,py0, logqp0 ',qy0, py0, logqp0 )
        print('z1 variational', z1.shape)

        return  z1, logqp0, qy0, py0


    
    def apply_SDE_fun(self, t, y):

        if self.normalise_for_SDENN:
            SDNN_expert_input_state = normalise_expert_data(y[:, :self.expert_latent_dims])
        else:
            SDNN_expert_input_state = y[:, :self.expert_latent_dims]/self.divisors.to(self.device)

        #print('SDNN_expert_input_state', SDNN_expert_input_state.shape, SDNN_expert_input_state[0, :])

        if self.include_time:
            # Positional encoding in transformers for time-inhomogeneous posterior
            sde_latent_times = torch.full_like(y[:, 0], fill_value=t).unsqueeze(1)
            sin_time = torch.sin(sde_latent_times)
            cos_time = torch.cos(sde_latent_times)

            input_state = torch.cat([SDNN_expert_input_state, y[:, self.expert_latent_dims+self.path_control_dim:]], dim=-1)
            SDE_NN_input = torch.cat((sin_time, cos_time, input_state), dim=-1)
        else:
            SDE_NN_input = torch.cat([SDNN_expert_input_state, y[:, self.expert_latent_dims+self.path_control_dim:]], dim=-1)

        print('SDE_NN_input shape', SDE_NN_input.shape)
        print('SDE_NN_input example', SDE_NN_input[0,:])
        SDE_NN_output_latents = self.SDEnet(SDE_NN_input) 
        print('SDE_NN_output_latents', SDE_NN_output_latents.shape)
        print('SDE_NN_output_latents example', SDE_NN_output_latents[0, :])

        return SDE_NN_output_latents
    
    def l1_regularization_output_layer(self, model, lambda_l1):
        # Assuming the first component of output_layer is the nn.Linear layer
        l1_norm = model.output_layer[0].weight.abs().sum()
        return lambda_l1 * l1_norm

    def f(self, t, y):  # Approximate posterior drift.
        batch_size = y.shape[0]
        #print('y', y.shape)
        p_a = y[:,0].unsqueeze(1) 
        p_v = y[:,1].unsqueeze(1) 
        s_reflex = y[:, 2] .unsqueeze(1) 
        sv = y[:, 3].unsqueeze(1)
        r_tpr_mod = y[:, 4].unsqueeze(1)
        f_hr_max = y[:, 5].unsqueeze(1)
        f_hr_min = y[:, 6].unsqueeze(1)
        r_tpr_max = y[:, 7].unsqueeze(1)
        r_tpr_min= y[:, 8].unsqueeze(1)
        ca = y[:, 9].unsqueeze(1)
        cv = y[:, 10].unsqueeze(1)
        k_width = y[:, 11].unsqueeze(1)
        p_aset = y[:, 12].unsqueeze(1)
        tau = y[:, 13].unsqueeze(1)

        path_r_tpr_mod = y[:, 14].unsqueeze(1)
        path_ca = y[:, 15].unsqueeze(1)
        path_cv = y[:, 16].unsqueeze(1)
        path_sv = y[:,17].unsqueeze(1)
        path_hr = y[:,18].unsqueeze(1)
        print('time:p_a pv, s, sv', t.item(), p_a[0].item(), p_v[0].item(), s_reflex[0].item(), sv[0].item())   
        print('time: params 1',t.item(),  r_tpr_mod[0].item(), f_hr_max[0].item(), f_hr_min[0].item(), r_tpr_max[0].item(), r_tpr_min[0].item(), )
        print('time: params 2', t.item(), ca[0].item(), cv[0].item(), k_width[0].item(), p_aset[0].item(), tau[0].item())
        print('time:pathrmod, path_ca, path_cv', t.item(), path_r_tpr_mod[0].item(), path_ca[0].item(), path_cv[0].item())   

        
        #the neural network is trying to learn the pathology control
        if self.apply_path_SDE:
            dt_path_SDE = self.apply_SDE_fun(t, y) * self.SDE_control_weighting
        else:
            dt_path_SDE = torch.zeros([batch_size, self.path_control_dim])

        f_hr = s_reflex * (f_hr_max - f_hr_min) + f_hr_min + path_hr
        r_tpr = s_reflex * (r_tpr_max - r_tpr_min) + r_tpr_min + (r_tpr_mod + path_r_tpr_mod)
        
        dva_dt = -1. * (p_a - p_v) / (r_tpr + 1e-7)  + sv * f_hr
        dvv_dt = -1. * dva_dt 
        dpa_dt = dva_dt / (ca + path_ca) 
        dpv_dt = dvv_dt / (cv + path_cv) 
        ds_dt = (1. / tau) * (1. - 1. / (1 + torch.exp(-k_width * (p_a - p_aset))) - s_reflex)
        dsv_dt = path_sv 

        print('dpa_dt, dpv_dt, ds_dt, dsv_dt', dpa_dt.shape, dpv_dt.shape, ds_dt.shape, dsv_dt.shape)
      
        diff_results = torch.cat([dpa_dt, dpv_dt, ds_dt, dsv_dt ], dim=-1)
        underlying_params = torch.zeros([batch_size, self.expert_latent_dims-diff_results.shape[1]])
        
        final_f_out = torch.cat([diff_results.to(self.device), underlying_params.to(self.device), dt_path_SDE.to(self.device)], dim=-1)
        
        print('final_f_out ', final_f_out.shape, final_f_out[0,:])
        return final_f_out 
    

    def h(self, t, y):  # Prior drift.
        self.mu = torch.tensor([0]).unsqueeze(1)
        expanded_mu = self.mu.repeat(y.size(0), self.path_control_dim)
        ##print('theta h', self.theta.shape, self.theta[0,:])
        ##print('mu h', expanded_mu[0,:])
        ##print('y in h', y.shape, y[0,:])
        ##print('mu -y ', expanded_mu[0,:] - y[0,:])

        return self.theta.to(self.device) * (expanded_mu - y)

    def f_aug(self, t, y):  # Drift for augmented dynamics with logqp term.
        
        dt_all_dims = y[:, :self.expert_latent_dims+self.path_control_dim]  # this is i_ext, pa, pv, s, sv and other encoder latents that go straight to the SDENN
        path_control = y[:, self.expert_latent_dims:self.expert_latent_dims +self.path_control_dim ]
        ##print('inputs to f', dt_all_dims.shape) # num_samples x sde_dims 
        
        f_res = self.f(t, dt_all_dims)
        
        if self.self_reverting_prior_control:
            g_path, h_path  = self.g(t, path_control), self.h(t, path_control)
            f_path = f_res[:,self.expert_latent_dims:]
            print('f', f_path.shape, 'g', g_path.shape, 'h', h_path.shape)
            print('f mean', f_path.mean(), 'g mean', g_path.mean(), 'h mean', h_path.mean())
            print('f', f_path[:3,:], 'g ', g_path[:3,:], 'h ', h_path[:3,:] )
            
            print('doing stable division!')
            u = _stable_division(f_path - h_path, g_path)
            print('u shape', u.shape)
            f_logqp = .5 * (u ** 2).sum(dim=1, keepdim=True)
        else:
            f_logqp = torch.zeros_like(y[:, 0]).unsqueeze(1).to(self.device)
        
        print('f_logqp', f_logqp.shape)
        print('f_logqp example', f_logqp[:3,:])   

        f_out = torch.cat([f_res, f_logqp], dim=1)
        print('f_aug out', f_out.shape, f_out[0])
        return f_out
    
    def g(self, t, y):  
        #sigma is different for each values here! 
        
        expanded_sigma = self.sigma.repeat(y.size(0), self.path_control_dim).to(self.device)
        #print('sigma g', expanded_sigma.shape, expanded_sigma[0] )
        return expanded_sigma

    def g_aug(self, t, y):  # Diffusion for augmented dynamics with logqp term.
        path_control = y[:, self.expert_latent_dims:self.expert_latent_dims +self.path_control_dim ]
        expert_dims = torch.zeros(y.shape[0], self.expert_latent_dims)

        g_res = self.g(t, path_control)
        print('g', g_res.shape, g_res[0])
        g_logqp = torch.zeros(y.size(0), 1).to(y.device)

        g_out = torch.cat([expert_dims.to(self.device), g_res.to(self.device), g_logqp.to(self.device)], dim=1)
        print('g out', g_out.shape, g_out[0])
        return g_out

    def forward_latent(self, init_latents, ts):
        #inputs of shape [batch x num_samples x dim ]
        batch_size = init_latents.shape[0]
        
        pathology_control = torch.zeros(batch_size,self.num_samples, self.path_control_dim).to(init_latents)
        log_path = torch.zeros(batch_size,self.num_samples, 1).to(init_latents)
        print('ts',ts.shape)
        print('init_latents ', init_latents.shape)
        print('pathology_control ', pathology_control.shape)
        print('log_path', log_path.shape)
        
        aug_y0 = torch.cat([init_latents, pathology_control, log_path,], dim=-1) 
        #print('aug_y0', aug_y0.shape, aug_y0[0, 0,:])
        dim_aug = aug_y0.shape[-1]
        aug_y0 = aug_y0.reshape(-1,dim_aug) # this is because the SDEint can only accept [M x dim], so M becomes batch_size * num_samples

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


        aug_ys = aug_ys.view(len(ts), batch_size, self.num_samples, dim_aug).permute(1,2,0,3)
        print('aug_ys out ', aug_ys.shape)
        
        latent_out = aug_ys[:, :, :, :self.expert_latent_dims]
        path_control = aug_ys[:, :, :, self.expert_latent_dims:self.expert_latent_dims+self.path_control_dim]
        logqp_path = aug_ys[:,: , -1, -1]    #.mean(dim=0)  # KL(t=0) + KL(path).
       
        print('latent_out end of latent ', latent_out.shape, latent_out[0, 0, :, 0])
        
        print('path_control', path_control.shape, path_control[0,0,:])
        print('logqp_path_extracted', logqp_path.shape)
       
        return latent_out, logqp_path, path_control

    def forward_dec(self, latent_out):
          
        if self.normalised_data:
            latent_out = self.normalise_expert_data(latent_out)
        
        else:
            divisors = self.divisors.view(1, 1, 1, self.expert_latent_dims).to(latent_out.device)
            latent_out = latent_out / divisors

        output_traj = select_tensor_by_index_list_advanced(latent_out, [0,1,2,3])
        print('output_traj', output_traj.shape, output_traj[0, 0, :, :])

        return output_traj
  

    def compute_factual_loss(self, predicted_traj, true_traj, logqp):
        
        predicted_traj = predicted_traj[:, :,:, :2]
        true_traj_expanded = true_traj.unsqueeze(1).repeat(1, self.num_samples, 1, 1) 
        print('true_traj_expanded', true_traj_expanded.shape, true_traj_expanded[0,0,:,:] )
        
        print('predicted_traj', predicted_traj.shape, predicted_traj.mean(1)[0,:,:])
        output_scale = torch.full(predicted_traj.shape, self.log_lik_output_scale, device=self.device)
        ##print('output_scale', output_scale.shape)

        #calculating the gaussian log prob between the predicted and the true traj. We want this to be as big as possible, so we will minimise its negative.
        print('likelihood')
        likelihood = distributions.Normal(loc=predicted_traj, scale=output_scale)
        print('log prob')
        logpy = likelihood.log_prob(true_traj_expanded)
        print('logpy', logpy.shape)
        logpy = logpy.sum((2,3)) #sum across times and dims, keeping a loss for each sde sample and batch size 

        L1_loss = self.l1_regularization_output_layer(self.SDEnet, self.l1_lambda )
        
        print('FACT log likelihood', logpy.shape, logpy.mean())
        print('KL_logqp',logqp.mean() )
        print('kl_scheduler', self.kl_scheduler.val)
        print('total KL', self.KL_weighting_SDE* (logqp.mean() * self.kl_scheduler.val))
        print('L1 loss', L1_loss)
        
        loss = -logpy.mean() + self.KL_weighting_SDE* (logqp.mean() * self.kl_scheduler.val) + L1_loss
        loss = loss.squeeze() 
        #print("TOTAL LOSS ", loss)

        return loss, -logpy.mean(), logqp.mean(), L1_loss


    def training_step(self, batch, batch_idx):
        print("TRAINING")
        X, Y, T, Y_cf, p, init_states, time_pre, time_post, time_FULL, full_fact_traj, full_cf_traj = batch
       
        z0 = self.sigmoid_scale(X[:, 0, :2]) if self.normalised_data else scale_unnormalised_experts(X[:, 0, :])
        print('z0',z0.shape, z0[0, :])
        print(torch.isnan(z0).any(), torch.isinf(z0).any())
        z1_input, logqp0, qy0, py0 =   self.forward_enc(z0[:, :2], time_pre)
        print('z1_input', z1_input.shape, z1_input[0, 0, :])

        latent_traj, logqp_path, path_control = self.forward_latent(init_latents = z1_input, 
                                                      ts = time_pre[0])
        predicted_traj = self.forward_dec(latent_traj )

        loss, fact_loss, kl_loss, L1_loss = self.compute_factual_loss(predicted_traj = predicted_traj, 
                                                             true_traj = X[:, :, :2], 
                                                             logqp=  logqp0 + logqp_path)
        
        
        self.log('train_total_loss', loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_fact_loss', fact_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_kl_loss', kl_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log('train_L1_loss', L1_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        self.kl_scheduler.step()

        print("LOSS STEP")
        return loss


    def validation_step(self, batch, batch_idx):
        print("VALIDATION")
        X, Y, T, Y_cf, p, init_states, time_pre, time_post, time_FULL, full_fact_traj, full_cf_traj = batch
       
        z0 = self.sigmoid_scale(X[:, 0, :2]) if self.normalised_data else scale_unnormalised_experts(X[:, 0, :])
        print('z0',z0.shape, z0[0, :])
        print(torch.isnan(z0).any(), torch.isinf(z0).any())

        z1_input, logqp0, qy0, py0 =   self.forward_enc(z0[:, :2], time_pre, MAP=True)
        print('z1_input', z1_input.shape, z1_input[0, 0, :])

        latent_traj, logqp_path, path_control = self.forward_latent(init_latents = z1_input, 
                                                      ts = time_pre[0])
        predicted_traj = self.forward_dec(latent_traj )

        loss, fact_loss, kl_loss, L1_loss = self.compute_factual_loss(predicted_traj = predicted_traj, 
                                                             true_traj = X[:, :, :2], 
                                                             logqp=  logqp0 + logqp_path)

        if self.global_step % self.plot_every == 0:
           self.visualize_distributions(qy0, py0)
           self.plot_trajectories(predicted_traj, X[:, :, :4])


        self.log('val_total_loss', loss,   on_epoch=True, prog_bar=True, logger=True)
        self.log('val_fact_loss', fact_loss, on_epoch=True, prog_bar=True, logger=True)
        self.log('val_kl_loss', kl_loss,  on_epoch=True, prog_bar=True, logger=True)
        self.log('val_L1_loss', L1_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)


        return loss
   
         
    def test_step(self, batch, batch_idx):
        print("TEST")
        X, Y, T, Y_cf, p, init_states, time_pre, time_post, time_FULL, full_fact_traj, full_cf_traj = batch
       
        z0 = self.sigmoid_scale(X[:, 0, :2]) if self.normalised_data else scale_unnormalised_experts(X[:, 0, :])
        print('z0',z0.shape, z0[0, :])
        z1_input, logqp0, qy0, py0 =   self.forward_enc(z0[:, :2], time_pre, MAP=True)
        print('z1_input', z1_input.shape, z1_input[0, 0, :])

        latent_traj, logqp_path, path_control = self.forward_latent(init_latents = z1_input, 
                                                      ts = time_pre[0])
        predicted_traj = self.forward_dec(latent_traj )

        loss, fact_loss, kl_loss, L1_loss = self.compute_factual_loss(predicted_traj = predicted_traj, 
                                                             true_traj = X[:, :, :2], 
                                                             logqp=  logqp0 + logqp_path)
        if batch_idx == 0:
            self.visualize_distributions(qy0, py0)
            self.plot_trajectories(predicted_traj, X[:, :, :4])

        self.log('test_total_loss', loss,   on_epoch=True, prog_bar=True, logger=True)
        self.log('test_fact_loss', fact_loss, on_epoch=True, prog_bar=True, logger=True)
        self.log('test_kl_loss', kl_loss,  on_epoch=True, prog_bar=True, logger=True)
        self.log('test_L1_loss', L1_loss,  on_epoch=True, prog_bar=True, logger=True)

        return loss
   

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr = self.learning_rate)
        
        scheduler = {"monitor": "train_total_loss", "scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer=optimizer, mode = "min", factor = 0.5, patience = 50)}
        return {"optimizer": optimizer, "lr_scheduler":scheduler}
    
    def on_save_checkpoint(self, checkpoint):
        print('SAVING CHECKPOINT')
        # Manually add mu, sigma, theta to the checkpoint dictionary
        checkpoint['mu'] = self.mu
        checkpoint['sigma'] = self.sigma
        checkpoint['theta'] = self.theta
        checkpoint['py0_mean'] = self.py0_mean
        checkpoint['py0_logvar']  = self.py0_logvar

    def on_load_checkpoint(self, checkpoint):
        print('LOADING CHECKPOINT')
        # Load mu, sigma, theta from the checkpoint dictionary if they exist
        if 'mu' in checkpoint:
            self.mu = checkpoint['mu']
        if 'sigma' in checkpoint:
            self.sigma = checkpoint['sigma']
        if 'theta' in checkpoint:
            self.theta = checkpoint['theta']
        if 'py0_mean' in checkpoint:
            self.py0_mean = checkpoint['py0_mean']
        if 'py0_logvar' in checkpoint:
            self.py0_logvar = checkpoint['py0_logvar']

    def visualize_distributions(self, qy0, py0, num_samples=1000):
        if not os.path.exists(self.train_dir):
            os.makedirs(self.train_dir)
        # Generate samples from both distributions
        with torch.no_grad():
            qy0_samples = qy0.sample((num_samples,))
            py0_samples = py0.sample((num_samples,))
            
            # Convert to numpy for plotting
            qy0_samples = qy0_samples.cpu().detach().numpy()
            py0_samples = py0_samples.cpu().detach().numpy()

            # Assume dimensions from the shape of samples
            num_dims = qy0_samples.shape[1]

            # Plotting using matplotlib and seaborn
            fig, axes = plt.subplots(nrows=num_dims, figsize=(10, num_dims * 5))
            if num_dims == 1:
                axes = [axes]  # Make it iterable
            
            for i, ax in enumerate(axes):
                sns.histplot(qy0_samples[:, i], kde=True, stat="density", color="blue", label="qy0 (Posterior)", ax=ax)
                sns.histplot(py0_samples[:, i], kde=True, stat="density", color="red", alpha=0.6, label="py0 (Prior)", ax=ax)
                ax.set_title(f'Distribution of Dimension {i+1}')
                ax.legend()

            plt.tight_layout()

            # Save the plot
            plot_path = os.path.join(self.train_dir, 'distribution_comparison.png')
            plt.savefig(plot_path)
            plt.close(fig)

            # Log to wandb if required
            if self.log_wandb:
                wandb.log({"Distribution Comparison": wandb.Image(plot_path)})

    def plot_trajectories(self, predicted_traj, true_traj, k=3):
        if not os.path.exists(self.train_dir):
            os.makedirs(self.train_dir)
        with torch.no_grad():
            idx = np.random.choice(predicted_traj.shape[0], k, replace=False)
            num_dims = true_traj.shape[2]

            fig = make_subplots(rows=k, cols=1, subplot_titles=[f'Batch {i+1}' for i in range(k)])

            for i, b in enumerate(idx):
                for d in range(num_dims):
                    pred_samples = predicted_traj[b, :, :, d].detach().cpu().numpy()
                    true_values = true_traj[b, :, d].detach().cpu().numpy()
                    lower_bound = np.min(pred_samples, axis=0)
                    upper_bound = np.max(pred_samples, axis=0)
                    times = np.arange(pred_samples.shape[1])

                    # Concatenate for plotly area plot (shaded region for prediction bounds)
                    fig.add_trace(
                        go.Scatter(
                            x=np.concatenate([times, times[::-1]]),
                            y=np.concatenate([upper_bound, lower_bound[::-1]]),
                            fill='toself',
                            fillcolor=f'rgba(0,100,80,{0.2 if d == 0 else 0.1})',  # vary opacity by dimension
                            line=dict(color='rgba(255,255,255,0)'),
                            showlegend=False,
                            legendgroup=f'Batch {b}',
                            name=f'Bounds Dim {d+1}'
                        ),
                        row=i+1, col=1
                    )

                    # True trajectory
                    fig.add_trace(
                        go.Scatter(
                            x=times, y=true_values, mode='lines+markers',
                            line=dict(color='blue', width=2),
                            marker=dict(size=4, color='red'),
                            name=f'True Trajectory Dim {d+1}',
                            legendgroup=f'Batch {b}',
                        ),
                        row=i+1, col=1
                    )

            fig.update_layout(
                height=300*k,  # Adjust height based on the number of batches
                title=f'Trajectories Comparison Across Batches',
                showlegend=True
            )
            plot_filename = os.path.join(self.train_dir, f'Predictions_global_step_{self.global_step}.png')
            fig.write_image(plot_filename, engine="kaleido")
            #print(f'Saved figure at: {plot_filename}')

            # Optionally log the plot to wandb if logging is enabled
            if self.log_wandb:
                wandb.log({"Predictions and factual": fig})






