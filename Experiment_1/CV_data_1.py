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


import pytorch_lightning as pl

import torch
from torch.utils.data import Dataset, DataLoader, Subset
import os
import numpy as np
from scipy.integrate import odeint

use_cuda = torch.cuda.is_available()





def fluids_input(t):
    return 5*np.exp(-((t-5)/5)**2)

def v_fun(x):
    return 0.02*(np.cos(5*x-0.2) * (5-x)**2)**2

def sigmoid(x):
    return 1/(1 + np.exp(-x))

def dx_dt(state, t, params):
    # Parameters:
    f_hr_max = params["f_hr_max"]
    f_hr_min = params["f_hr_min"]
    r_tpr_max = params["r_tpr_max"]
    r_tpr_min = params["r_tpr_min"]
    ca = params["ca"]
    cv = params["cv"]
    k_width = params["k_width"]
    p_aset = params["p_aset"]
    tau = params["tau"]
    t_treatment = params["t_treatment"]

    # Unknown parameters:
    
    if (params["treatment"]) and (t>=t_treatment):
        initp_transform  = 0.5+(params["init_pressure"]-0.75)/0.1
        A_ = v_fun(initp_transform)
        #A_ = 1
        i_ext = A_ * fluids_input(t-t_treatment)
    else:
        i_ext = 0
    
    
    r_tpr_mod = params["r_tpr_mod"]
    sv_mod = params["sv_mod"]

    # State variables
    p_a = 100. * state[0]
    p_v = 10. * state[1]
    s = state[2]
    sv = 100. * state[3]

    # Calculating modified stroke volume based on venous pressure
    #sv = sv_base + sv_mod_factor * (p_v - pv_base) 
    #cardiac_output = sv * f_hr


    # Building f_hr and r_tpr:
    f_hr = s * (f_hr_max - f_hr_min) + f_hr_min
    r_tpr = s * (r_tpr_max - r_tpr_min) + r_tpr_min - r_tpr_mod

    # Building dp_a/dt and dp_v/dt:
    dva_dt = -1. * (p_a - p_v) / r_tpr + sv * f_hr
    dvv_dt = -1. * dva_dt + i_ext
    dpa_dt = dva_dt / (ca * 100.)
    dpv_dt = dvv_dt / (cv * 10.)

    # Building dS/dt:
    ds_dt = (1. / tau) * (1. - 1. / (1 + np.exp(-1 * k_width * (p_a - p_aset))) - s)

    # Building dSV/dt:


    dsv_dt = i_ext * sv_mod

    # State derivative
    return np.array([dpa_dt, dpv_dt, ds_dt, dsv_dt])

def init_random_state():
    max_ves = 64.0 - 10.0
    min_ves = 36.0 + 10.0

    max_ved = 167.0 - 10.0
    min_ved = 121.0 + 10.0

    max_sv = 1.0
    min_sv = 0.9

    max_pa = 85.0
    min_pa = 75.0

    max_pv = 7.0
    min_pv = 3.0

    max_s = 0.25
    min_s = 0.15

    init_ves = (np.random.rand() * (max_ves - min_ves) + min_ves) / 100.0
    # init_ves = 50.0 / 100.0

    init_ved = (np.random.rand() * (max_ved - min_ved) + min_ved) / 100.0
    # init_ved = 144.0 / 100.0

    init_sv = (np.random.rand() * (max_sv - min_sv) + min_sv)
    init_pa = (np.random.rand() * (max_pa - min_pa) + min_pa) / 100.0
    init_pv = (np.random.rand() * (max_pv - min_pv) + min_pv) / 10.0
    init_s = (np.random.rand() * (max_s - min_s) + min_s)

    init_state = np.array([init_pa, init_pv, init_s, init_sv])
    return init_state


def create_cv_data(N,gamma,noise_std, t_span = 30, t_treatment = 15, seed = 421, normalize = True, output_dims = [0, 1], input_dims = [0,1] ):

    np.random.seed(seed)

    X = []
    Y_combined = []
    init_state_list = []
    
    params = {"r_tpr_mod": 0.,
            "f_hr_max": 3.0,
            "f_hr_min": 2.0 / 3.0,
            "r_tpr_max": 2.134,
            "r_tpr_min": 0.5335,
            "sv_mod": 0.001,
            "ca": 4.0,
            "cv": 111.0,

            # dS/dt parameters
            "k_width": 0.1838,
            "p_aset": 70,
            "tau": 20,
            "p_0lv": 2.03,
            "r_valve": 0.0025,
            "k_elv": 0.066,
            "v_ed0": 7.14,
            "T_sys": 4. / 15.,
            "cprsw_max": 103.8,
            "cprsw_min": 25.9,
            "t_treatment" : t_treatment
            }
    
    params_treatment = params.copy()
    params_treatment["treatment"]=True
    params_notreatment = params.copy()
    params_notreatment["treatment"]=False
    
    t = np.arange(t_span).astype(float)
    
    for i in range(N):
        init_state = init_random_state()
        params_treatment["init_pressure"] = init_state[0]
        params_treatment["cv"] = np.random.rand() * 100 + 10
        y = odeint(dx_dt,init_state,t,args=tuple([params_treatment]))
        
        X.append(torch.Tensor(init_state))
        Y_combined.append(torch.Tensor(y))
        init_state_list.append(torch.Tensor(init_state))
    
    init_state = torch.stack(init_state_list)
    X = torch.stack(X)
    Y_combined = torch.stack(Y_combined)
    
    X += noise_std * torch.randn(X.shape)
    Y_combined += noise_std * torch.randn(Y_combined.shape)

    if normalize:
            mu = Y_combined.mean([0, 1])
            std = Y_combined.std([0, 1])
            Y_combined = (Y_combined - mu) / std
            mu_X = X.mean([0, 1])
            std_X = X.std([0, 1])
            X = (X - mu_X) / std_X

        # No need for separate masks if the entire trajectory is used
    t_X = torch.Tensor(np.tile(t[None, :], (X.shape[0], 1)))
    t_X = t_X.unsqueeze(-1)
    Y_combined = Y_combined[:, :, output_dims]
    
    return Y_combined, t_X

class CVDataset(Dataset):
    def __init__(self,N, gamma,noise_std, t_span, t_treatment, seed):
        
        Y_combined, t_X = create_cv_data(N = N, gamma = gamma, noise_std = noise_std, t_span = t_span, t_treatment = t_treatment, seed = seed)

        self.Y_combined = Y_combined
        self.t_X = t_X
    

    def __getitem__(self,idx):
        return self.Y_combined[idx], self.t_X[idx]
    
    def __len__(self):
        return self.t_X.shape[0]



class CVDataModule(pl.LightningDataModule):
    def __init__(self,batch_size, seed, N_ts, gamma, noise_std, t_span, t_treatment, num_workers = 4, **kwargs):
        
        super().__init__()
        self.batch_size = batch_size
        self.seed = seed
        self.num_workers = num_workers

        self.train_shuffle = True

        self.input_dim = 2
        self.output_dim = 1 # number of dimensions to reconstruct in the time series

        self.N = N_ts
        self.gamma = gamma
        self.noise_std = noise_std
        self.t_span = t_span
        self.t_treatment = t_treatment

    def prepare_data(self):

        dataset = CVDataset(N = self.N, gamma = self.gamma,noise_std =  self.noise_std, seed = self.seed, t_span = self.t_span, t_treatment = self.t_treatment)       
        
        train_idx = np.arange(len(dataset))[:int(0.5*len(dataset))]
        val_idx = np.arange(len(dataset))[int(0.5*len(dataset)):]
        test_idx = val_idx[int(len(val_idx)/2):]
        val_idx = val_idx[:int(len(val_idx)/2)]


        self.train = Subset(dataset,train_idx)
        self.val = Subset(dataset,val_idx)
        self.test = Subset(dataset,test_idx)
    
    def train_dataloader(self):
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            shuffle=self.train_shuffle,
            num_workers=self.num_workers,
            persistent_workers=True, 
            drop_last=True,
            pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True,
            drop_last=False,
            pin_memory=True
        )

    def test_dataloader(self):
        return DataLoader(
            self.test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=False,
            pin_memory=True
            )

