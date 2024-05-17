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

import matplotlib.pyplot as plt
import numpy as np
import torch
import tqdm
from torch import distributions, nn, optim

import torchsde

from CV_data_6 import create_cv_data
from utils_6 import CV_params, GaussianNLLLoss
# w/ underscore -> numpy; w/o underscore -> torch.
Data = namedtuple('Data', ['ts_', 'ts_ext_', 'ts_vis_', 'ts', 'ts_ext', 'ts_vis', 'ys', 'ys_'])



def ensure_scalar(value):
    """Converts a single-element tensor or array to a Python scalar if necessary."""
    if isinstance(value, torch.Tensor):
        return value.item() if value.numel() == 1 else value
    elif isinstance(value, np.ndarray):
        return value.item() if value.size == 1 else value
    else:
        return value


class LinearScheduler(object):
    def __init__(self, iters, maxval=1.0):
        self._iters = max(1, iters)
        self._val = maxval / self._iters
        self._maxval = maxval

    def step(self):
        self._val = min(self._maxval, self._val + self._maxval / self._iters)

    @property
    def val(self):
        return self._val


class EMAMetric(object):
    def __init__(self, gamma: Optional[float] = .99):
        super(EMAMetric, self).__init__()
        self._val = 0.
        self._gamma = gamma

    def step(self, x: Union[torch.Tensor, np.ndarray]):
        x = x.detach().cpu().numpy() if torch.is_tensor(x) else x
        self._val = self._gamma * self._val + (1 - self._gamma) * x
        return self._val

    @property
    def val(self):
        return self._val


def str2bool(v):
    """Used for boolean arguments in argparse; avoiding `store_true` and `store_false`."""
    if isinstance(v, bool): return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'): return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'): return False
    else: raise argparse.ArgumentTypeError('Boolean value expected.')


def manual_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _stable_division(a, b, epsilon=1e-7):
    b = torch.where(b.abs().detach() > epsilon, b, torch.full_like(b, fill_value=epsilon) * b.sign())
    return a / b


class LatentSDE(torchsde.SDEIto):

    def __init__(self, expert_dims, SDE_latents_dim, CV_params, mu=None, sigma=None, theta=1.0 ):
        super(LatentSDE, self).__init__(noise_type="diagonal")
        #self.sde_type="ito"

        self.SDE_latents_dim = SDE_latents_dim
        self.expert_dims = expert_dims
        self.resid_dims = SDE_latents_dim - expert_dims
        
        if mu is None:
            mu = np.zeros(SDE_latents_dim)
        if sigma is None:
            sigma = np.ones(SDE_latents_dim) * 0.5
        
        # Prior drift.
        self.register_buffer("theta", torch.tensor(theta).view(1, -1).expand(1, SDE_latents_dim).float())
        self.register_buffer("mu", torch.tensor(mu).view(1, -1).float())
        self.register_buffer("sigma", torch.tensor(sigma).view(1, -1).float())

        #print('theta init', self.theta.shape)
        #print('mu init ', self.mu.shape)
        #print('sigma init', self.sigma.shape)

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
        print('f', f.shape, 'g', g.shape, 'h', h.shape)
        print('f mean', f.mean(), 'g mean', g.mean(), 'h mean', h.mean())
        print('f', f, 'g ', g, 'h ', h )
        u = _stable_division(f - h, g)
        print('u', u.shape)
        print('u mean', u)
        f_logqp = .5 * (u ** 2).sum(dim=1, keepdim=True)
        print('f_logqp', f_logqp.shape)
        print('f_logqp mean', f_logqp)
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

    def forward(self,ts, SDE_samples, eps=None):
        #eps = torch.randn(SDE_samples, self.SDE_latents_dim).to(self.qy0_std) if eps is None else eps
        
        y0 = self.mu.expand(SDE_samples, -1)  #starting vals has shape [batch x dims ]
        #print('y0 sampled', y0.shape)
        #qy0 = distributions.Normal(loc=self.qy0_mean, scale=self.qy0_std)
        #py0 = distributions.Normal(loc=self.py0_mean, scale=self.py0_std)
        #logqp0 = distributions.kl_divergence(qy0, py0).sum(dim=1)  # KL(t=0).
        logqp0 = 0

        aug_y0 = torch.cat([y0, torch.zeros(SDE_samples, 1).to(y0)], dim=-1)
        #print('aug_y0', aug_y0.shape) #this will be num_samples x dim = 512 x 4

        aug_ys = sdeint_fn(
            sde=self,
            y0=aug_y0,
            ts=ts,
            method=args.method,
            dt=args.dt,
            adaptive=args.adaptive,
            rtol=args.rtol,
            atol=args.atol,
            names={'drift': 'f_aug', 'diffusion': 'g_aug'}
        )
        ys, logqp_path = aug_ys[:, :, :self.SDE_latents_dim], aug_ys[-1, :, self.SDE_latents_dim:]
        
        #print('ys post sde', ys.shape)
        ys = self.reverse_scale_init_expert(ys)
        #print('ys_out', ys.shape)
        #print('logqp_path_out', logqp_path.shape)
        logqp = (logqp0 + logqp_path) #.mean(dim=0)  # KL(t=0) + KL(path).
        return ys, logqp

   


def make_segmented_cosine_data():
    ts_ = np.concatenate((np.linspace(0.3, 0.8, 10), np.linspace(1.2, 1.5, 10)), axis=0)
    ts_ext_ = np.array([0.] + list(ts_) + [2.0])
    ts_vis_ = np.linspace(0., 2.0, 300)
    ys_ = np.cos(ts_ * (2. * math.pi))[:, None]

    ts = torch.tensor(ts_).float()
    ts_ext = torch.tensor(ts_ext_).float()
    ts_vis = torch.tensor(ts_vis_).float()
    ys = torch.tensor(ys_).float().to(device)
    return Data(ts_, ts_ext_, ts_vis_, ts, ts_ext, ts_vis, ys, ys_)


def make_irregular_sine_data():
    ts_ = np.sort(np.random.uniform(low=0.4, high=1.6, size=16))
    ts_ext_ = np.array([0.] + list(ts_) + [2.0])
    ts_vis_ = np.linspace(0., 2.0, 300)
    ys_ = np.sin(ts_ * (2. * math.pi))[:, None] * 0.8

    ts = torch.tensor(ts_).float()
    ts_ext = torch.tensor(ts_ext_).float()
    ts_vis = torch.tensor(ts_vis_).float()
    ys = torch.tensor(ys_).float().to(device)
    return Data(ts_, ts_ext_, ts_vis_, ts, ts_ext, ts_vis, ys, ys_)




def make_data():
    data_constructor = {
        'segmented_cosine': make_segmented_cosine_data,
        'irregular_sine': make_irregular_sine_data
    }[args.data]
    return data_constructor()



def main():
    # Dataset.
    if args.data != "cv_data":
        ts_, ts_ext_, ts_vis_, ts, ts_ext, ts_vis, ys, ys_ = make_data()
    else:
        X, X_static, T, Y_fact, Y_cf, p, init_state, t_X, t_Y, expert_ODE_size = create_cv_data(N = 1000, gamma = 10, noise_std = 0, t_span = 25, t_treatment=10, seed = 123, output_dims=[0,1])
        
        ys =  X[0,:, :]
        ys_ = X[0,:].cpu().numpy()
        ts_ = t_X[0, :].cpu().numpy()
        ts_vis = t_X[0, :]
        ts_ext = t_X[0, :]
        #print('ts_vis',ts_vis.shape)

        ys_to_compare = ys.unsqueeze(1) #going from [times x dim] to [times x num_samples x dim] 
        ys_to_compare = ys_to_compare[:, :, :].repeat(1, args.SDE_samples, 1)

        SDE_latents_dim = 4

    #print('ys', ys.shape)
    #print('ts_ext', ts_ext.shape)
    #print('ts_vis', ts_vis.shape)


    # Plotting parameters.
    vis_num_samples = args.SDE_samples
    ylims = (-1.75, 1.75)
    alphas = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55]
    percentiles = [0.999, 0.99, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    vis_idx = np.random.permutation(vis_num_samples)
    # From https://colorbrewer2.org/.
    if args.color == "blue":
        sample_colors = ('#8c96c6', '#8c6bb1', '#810f7c')
        fill_color = '#9ebcda'
        mean_color = '#4d004b'
        num_samples = len(sample_colors)
    else:
        sample_colors = ('#fc4e2a', '#e31a1c', '#bd0026')
        fill_color = '#fd8d3c'
        mean_color = '#800026'
        num_samples = len(sample_colors)

    eps = torch.randn(vis_num_samples, SDE_latents_dim).to(device)  # Fix seed for the random draws used in the plots.
    bm = torchsde.BrownianInterval(
        t0=ts_vis[0],
        t1=ts_vis[-1],
        size=(vis_num_samples, SDE_latents_dim),
        device=device,
        levy_area_approximation='space-time'
    )  # We need space-time Levy area to use the SRK solver

    # Model.
    first_pa_pv = ys[0,:2]
    print('first_pa_pv', first_pa_pv)
    mu_latents = np.concatenate((first_pa_pv, np.array([0.2, 0.95])))    # shall i give the first values for this?
    sigma_latents = np.array([0.01, 0.01, 0.2, 0.2])
    print('mu_latents', mu_latents)
    #print('mu_latents', mu_latents.shape , 'sigma_latents', sigma_latents.shape)

    model = LatentSDE(expert_dims=2, SDE_latents_dim = SDE_latents_dim, mu=mu_latents, sigma=sigma_latents,CV_params =CV_params).to(device)


    optimizer = optim.Adam(model.parameters(), lr=1e-2)
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=.999)
    kl_scheduler = LinearScheduler(iters=args.kl_anneal_iters)

    logpy_metric = EMAMetric()
    kl_metric = EMAMetric()
    loss_metric = EMAMetric()

    if args.show_prior:
        with torch.no_grad():
            zs = model.sample_p(ts=ts_vis, SDE_samples=vis_num_samples, eps=eps, bm=bm).squeeze()
            ts_vis_, zs_ = ts_vis.cpu().numpy(), zs.cpu().numpy()
            zs_ = np.sort(zs_, axis=1)  # Sort along the sample dimension
            #print('ts_vis_', ts_vis_.shape, 'zs_', zs_.shape,'ys_', ys_.shape,'ts_', ts_.shape  )

            img_dir = os.path.join(args.train_dir, 'prior.png')
            # Determine the number of observed dimensions in ys_
            num_obs_dims = ys_.shape[-1]

            # Setting up the plot
            fig, axs = plt.subplots(nrows=4, ncols=1, figsize=(10, 20), sharex=True)  # Assuming zs_ has 4 dimensions

            colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']  # Extend this list if you have more than 7 observed dimensions

            for i in range(4):  # Loop through each dimension of zs_
                for alpha, percentile in zip(alphas, percentiles):
                    idx = int((1 - percentile) / 2. * vis_num_samples)
                    zs_bot_ = zs_[:, idx, i]
                    zs_top_ = zs_[:, -idx, i]
                    axs[i].fill_between(ts_vis_, zs_bot_, zs_top_, alpha=alpha, color=fill_color)

                # Plot each dimension of ys_ using a different color
                for j in range(num_obs_dims):
                    axs[i].scatter(ts_, ys_[:, j], marker='x', zorder=3, color=colors[j % len(colors)], s=35, label=f'Dim {j+1}')

                axs[i].set_ylim(ylims)  # Ensure ylims is appropriately defined
                axs[i].set_ylabel(f'$Z_{{t,{i+1}}}$')
                axs[i].legend(loc='upper right')

            axs[-1].set_xlabel('$t$')
            plt.tight_layout()
            plt.savefig(img_dir, dpi=args.dpi)
            plt.close()
            logging.info(f'Saved prior figure at: {img_dir}')

    for global_step in tqdm.tqdm(range(args.train_iters)):
        # Plot and save.

        # Train.
        optimizer.zero_grad()
        zs, kl = model(ts=ts_ext, SDE_samples=args.SDE_samples)
        print('zs', zs.shape)
        print('kl', kl.shape)
        #zs = zs[1:-1, :, :]  # Drop first and last which are only used to penalize out-of-data region and spread uncertainty.
        ##print('zs edited', zs.shape)
        
        
        zs_to_compare = zs[:, :, :2]
        
        #print('example zs', zs_to_compare[:,0,0])
        #print('zs_to_compare', zs_to_compare.shape)
        #print('ys_to_compare', ys_to_compare.shape)
        
        #need to convert back to normalised!!
        #loss_fun = GaussianNLLLoss(reduction = "none")
        #output_scale = torch.tensor([args.scale], requires_grad = False, device = device)


        #fact_loss = loss_fun(ys_to_compare, zs_to_compare, output_scale.repeat(ys_to_compare.shape).to(device))
        #logpy = fact_loss.sum((0, 2))  # sum across times and dims (keeping for each batch and SDE sample)
        #print('fact_loss after sum:', fact_loss.shape)

        likelihood_constructor = {"laplace": distributions.Laplace, "normal": distributions.Normal}[args.likelihood]
        likelihood = likelihood_constructor(loc=zs_to_compare, scale=args.scale)
        logpy = likelihood.log_prob(ys_to_compare)#.sum(dim=0).mean(dim=0)
        
        print('logpy', logpy.shape)
        logpy = logpy.sum((0,2)) #sum across times and dims, keeping a loss for each sde sample
        print('logpy', logpy.shape)

        
        loss = -logpy.mean() + kl.mean() * kl_scheduler.val
        loss = loss.squeeze() 
        #print(loss)
        #print('loss', loss.shape)
        loss.backward()

        optimizer.step()
        scheduler.step()
        kl_scheduler.step()

        # Calculate the mean to reduce logpy to a scalar
        logpy_val = logpy.mean() 
        logpy_metric.step(logpy_val) 
        
        kl_val = ensure_scalar(kl.mean())
        kl_metric.step(kl_val)

        loss_val = ensure_scalar(loss)
        loss_metric.step(loss_val)

        #print(f"logpy_val type: {type(logpy_val)}, value: {logpy_val}")
        #print(f"kl_val type: {type(kl_val)}, value: {kl_val}")
        #print(f"loss_val type: {type(loss_val)}, value: {loss_val}")

        # Logging information
        logging.info(
            f'global_step: {global_step}, '
            f'logpy: {logpy_val:.3f}, '
            f'kl: {kl_val:.3f}, '
            f'loss: {loss_val:.3f}'
        )

        if global_step % args.pause_iters == 0:
            img_path = os.path.join(args.train_dir, f'global_step_{global_step}.png')

            with torch.no_grad():
                if args.sample_q == True:
                    zs = model.sample_q(ts=ts_vis, SDE_samples=vis_num_samples, eps=eps, bm=bm).squeeze()

                samples = zs[:, vis_idx, :]
                #print('samples', samples.shape)
                                
                # Assuming `zs` and `ys_` have been computed and are on the appropriate device
                ts_vis_, zs_, samples_ = ts_vis.cpu().numpy(), zs.cpu().numpy(), samples.cpu().numpy()
                zs_ = np.sort(zs_, axis=1)  # Sort along the sample dimension

                # Initialize the plot
                plt.figure(figsize=(10, 6))

                # Define colors and styles for clarity
                colors = ['blue', 'green', 'red', 'purple', 'cyan', 'magenta']
                styles = ['-', '--', '-.', ':', '-', '--']
                markers = ['o', 'x']

                # Plot each SDE latent dimension
                for i in range(4):  # There are four SDE latent dimensions
                    if args.show_percentiles:
                        for alpha, percentile in zip(alphas, percentiles):
                            idx = int((1 - percentile) / 2. * vis_num_samples)
                            zs_bot_, zs_top_ = zs_[:, idx, i], zs_[:, -idx, i]
                            plt.fill_between(ts_vis_, zs_bot_, zs_top_, alpha=alpha, color=colors[i], label=f'SDE Latent Dim {i+1} (Percentiles)' if i == 0 else "")

                    if args.show_mean:
                        plt.plot(ts_vis_, zs_.mean(axis=1)[:, i], color=colors[i], linestyle=styles[i], label=f'SDE Latent Dim {i+1} (Mean)')

                    if args.show_samples:
                        for j in range(num_samples):
                            plt.plot(ts_vis_, samples_[:, j, i], color=colors[i], linestyle=styles[i], linewidth=0.5, label=f'SDE Latent Dim {i+1} (Sample {j+1})' if j == 0 else "")

                # Plot observed dimensions
                for k in range(2):  # Assuming there are two observed dimensions
                    plt.scatter(ts_, ys_[:, k], color=colors[4+k], marker=markers[k], s=35, zorder=3, label=f'Observed Dim {k+1}')

                # Adding plot embellishments
                plt.xlabel('$t$')
                plt.ylabel('$Y_t$')
                plt.ylim(ylims)
                plt.legend()
                plt.tight_layout()
                plt.savefig(img_path, dpi=args.dpi)
                plt.close()
                logging.info(f'Saved figure at: {img_path}')

if __name__ == '__main__':
    sys.stdout = open('Hybrid_SDE_output.txt', 'w')
    # The argparse format supports both `--boolean-argument` and `--boolean-argument True`.
    # Trick from https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse.
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-gpu', type=str2bool, default=False, const=True, nargs="?")
    parser.add_argument('--debug', type=str2bool, default=True, const=True, nargs="?")
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--train-dir', type=str, required=True)
    parser.add_argument('--save-ckpt', type=str2bool, default=False, const=True, nargs="?")

    parser.add_argument('--data', type=str, default='cv_data', choices=['segmented_cosine', 'irregular_sine','cv_data'])
    parser.add_argument('--kl-anneal-iters', type=int, default=100, help='Number of iterations for linear KL schedule.')
    parser.add_argument('--train-iters', type=int, default=5000, help='Number of iterations for training.')
    parser.add_argument('--pause_iters', type=int, default=30, help='Number of iterations before pausing.')
    parser.add_argument('--SDE_samples', type=int, default=3, help='num_samples for training.')
    parser.add_argument('--likelihood', type=str, choices=['normal', 'laplace'], default='normal')
    parser.add_argument('--scale', type=float, default=0.05, help='Scale parameter of Normal and Laplace.')

    parser.add_argument('--adjoint', type=str2bool, default=False, const=True, nargs="?")
    parser.add_argument('--adaptive', type=str2bool, default=False, const=True, nargs="?")
    parser.add_argument('--method', type=str, default='euler', choices=('euler', 'milstein', 'srk'),
                        help='Name of numerical solver.')
    parser.add_argument('--dt', type=float, default=1e-2)
    parser.add_argument('--rtol', type=float, default=1e-3)
    parser.add_argument('--atol', type=float, default=1e-3)

    parser.add_argument('--sample_q', type=str2bool, default=False, const=True, nargs="?")
    parser.add_argument('--show-prior', type=str2bool, default=False, const=True, nargs="?")
    parser.add_argument('--show-samples', type=str2bool, default=False, const=True, nargs="?")
    parser.add_argument('--show-percentiles', type=str2bool, default=False, const=True, nargs="?")
    parser.add_argument('--show-arrows', type=str2bool, default=True, const=True, nargs="?")
    parser.add_argument('--show-mean', type=str2bool, default=True, const=True, nargs="?")
    parser.add_argument('--hide-ticks', type=str2bool, default=False, const=True, nargs="?")
    parser.add_argument('--dpi', type=int, default=300)
    parser.add_argument('--color', type=str, default='blue', choices=('blue', 'red'))
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_gpu else 'cpu')
    manual_seed(args.seed)

    if args.debug:
        logging.getLogger().setLevel(logging.INFO)

    ckpt_dir = os.path.join(args.train_dir, 'ckpts')
    os.makedirs(ckpt_dir, exist_ok=True)

    sdeint_fn = torchsde.sdeint_adjoint if args.adjoint else torchsde.sdeint

    main()