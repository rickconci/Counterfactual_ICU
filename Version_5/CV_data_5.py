import math
import numpy as np
from tqdm import tqdm_notebook as tqdm
import os
import pickle
from pathlib import Path


import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import torch
from torch import Tensor
from torch import nn
from torch.nn  import functional as F 
from torch.autograd import Variable


import lightning as L

import torch
from torch.utils.data import Dataset, DataLoader, Subset
import os
import argparse
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
        #initp_transform  = 0.5+(params["init_pressure"]-0.75)/0.1
        #A_ = v_fun(initp_transform)
        A_ = 1
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


def create_cv_data(N,gamma,noise_std, t_span, t_treatment, seed, output_dims, input_dims = [0,1], normalize = True):

    np.random.seed(seed)

    X = []
    Y_0 = []
    Y_1 = []
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
        #create the treated trajectory (with treatment starting at t_treatment = 15 seconds )
        init_state = init_random_state()
        params_treatment["init_pressure"] = init_state[0]
        params_treatment["cv"] = np.random.rand() * 100 + 10
        y1 = odeint(dx_dt,init_state,t,args=tuple([params_treatment]))
        
        #create the untreated trajectory (with same baseline until 15 seconds)
        params_notreatment["init_pressure"] = init_state[0]
        y0 = odeint(dx_dt,init_state,t,args=tuple([params_notreatment]))
        
        
        X.append(torch.Tensor(init_state))
        Y_0.append(torch.Tensor(y0))
        Y_1.append(torch.Tensor(y1))
        init_state_list.append(torch.Tensor(init_state))
    
    #stack all the initial states 
    init_state = torch.stack(init_state_list)
    #identify the probability of treatment based on the initial states (this is the confounder!)
    # note however, that the probability of tx is affected by the 'initial' state, rather than in the states just before the tx as would be in real life (or at least a combination)
    # gamma determines the extent of the confounding 
    p = torch.sigmoid(gamma*((init_state[:,0]-0.75)/0.1-0.5))
    T = torch.zeros(N)
    #T determines which trajectories as selected as treated (this is confounded by p and gamma)
    T[torch.rand(N)<p] = 1

    Y_0 = torch.stack(Y_0)
    Y_1 = torch.stack(Y_1)
    Y_0 += noise_std * torch.randn(Y_0.shape)
    Y_1 += noise_std * torch.randn(Y_1.shape)
    X = torch.stack(X)
    X += noise_std * torch.randn(X.shape)

    expert_ODE_size = X.shape[1]

    # the 'factual' trajectories are the UNtreated outcome (Y0) for the not Treated (1-T) and the Treated outcome (Y1) for the factually Treated (T)
    Y_fact = Y_0 * (1-T)[:,None,None] + Y_1 * T[:,None,None]
    # the 'COUNTERfactual' trajectories are the UNtreated outcome (Y0) for the Treated (T) and the Treated outcome (Y1) for the factually UNtreated (1-T)
    # we would never actually have access to the counterfactual other than in this situation where we are simulating it 
    Y_cf = Y_0 * (T)[:,None,None] + Y_1 * (1-T)[:,None,None]

    
    Y_fact_np = Y_fact.detach().cpu().numpy()
    states_mean = Y_fact_np.mean(axis=(0, 1))
    states_min = Y_fact_np.min(axis=(0, 1))
    states_max = Y_fact_np.max(axis=(0, 1))
    print('states_mean', states_mean, 'states_min', states_min, 'states_max', states_max)

    if normalize:
        mu = Y_fact.mean([0,1])
        std = Y_fact.std([0,1])

        Y_fact = (Y_fact - mu)/std
        Y_cf = (Y_cf - mu)/std
        
        mu_X = X.mean([0,1])
        std_X = X.std([0,1])
        X = (X-mu_X)/std_X
    
    # Now split these factual and counterfactual trajectories by the 'before' and 'after treatment' so we have a baseline 
    pre_treat_mask = (t<=t_treatment)
    post_treat_mask = (t>t_treatment)
    
    # We define X as the Factual trajectory BEFORE treatment, and X_ as the COUNTERfactual traj BEFORE treatment 
    X_static = X
    X = Y_fact[:,pre_treat_mask][:,:,input_dims]
    X_ = Y_cf[:,pre_treat_mask][:,:,input_dims]

    # We redfine Y_fact as the Factual trajectory AFTER treatment, and Y_cf as the COUNTERfactual traj AFTER treatment 
    # We are selecting the DIASTOLIC BP (output dim = 1) as the one to maintain.. this is because the fluid is only really affecting this within the time values
    Y_fact = Y_fact[:,post_treat_mask][:,:,output_dims] 
    Y_cf = Y_cf[:,post_treat_mask][:,:,output_dims]

    # we split the time vector also as before and after treatment 
    t_x = t[pre_treat_mask]
    t_y = t[post_treat_mask]
    # and get it to match the dimensions, so it's not a vector t, but a matrix of dimensions N by before_tx and N by after_tx
    t_X = torch.Tensor(np.tile(t_x[None,:],(X.shape[0],1)))
    t_Y = torch.Tensor(np.tile(t_y[None,:],(Y_fact.shape[0],1))) - t_x[-1]


    return X, X_static, T, Y_fact, Y_cf, p, init_state, t_X, t_Y, expert_ODE_size

class CVDataset(Dataset):
    def __init__(self,N, gamma,noise_std, t_span, t_treatment, seed, output_dims):

        X, X_static, T, Y_fact, Y_cf, p, init, t_X, t_Y, expert_ODE_size = create_cv_data(N = N, gamma = gamma, noise_std = noise_std, t_span = t_span, t_treatment = t_treatment, seed = seed, output_dims = output_dims)

        self.X = X
        self.T = T
        self.Y_fact = Y_fact
        self.Y_cf = Y_cf
        self.T_cf = (~T.bool()).float()
        self.p = p
        self.init = init
        self.t_X = t_X
        self.t_Y = t_Y
        
        self.post_tx_ode_len = t_Y.shape[1]
        self.Tx_dim = T.shape[1] if len(T.shape) > 1 else 1
        self.expert_ODE_size = expert_ODE_size



    def __getitem__(self,idx):
        return self.X[idx], self.Y_fact[idx], self.T[idx], self.Y_cf[idx], self.p[idx], self.init[idx], self.t_X[idx], self.t_Y[idx]
    def __len__(self):
        return self.X.shape[0]

class CVDataModule(L.LightningDataModule):
    def __init__(self,batch_size, seed, N_ts, gamma, noise_std, t_span, t_treatment, output_dims, data_dir, num_workers,  **kwargs):
        
        super().__init__()
        self.batch_size = batch_size
        self.seed = seed
        self.num_workers = num_workers

        self.train_shuffle = True

        self.input_dim = 2
        self.output_dims = len(output_dims)
        

        self.N_ts = N_ts
        self.gamma = gamma
        self.noise_std = noise_std
        self.t_span = t_span
        self.t_treatment = t_treatment

        self.data_dir = Path(data_dir)
        self.data_file = self.data_dir/"cv_data_module.pkl"

    def prepare_data(self):
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        if self.data_file.exists():
            print("Loading dataset from pickle file!")
            with open(self.data_file, "rb") as file:
                dataset = pickle.load(file)
        else:
            print("Creating dataset!")
            dataset = self.create_data()  # This should return an instance of CVDataset
            with open(self.data_file, "wb") as file:
                pickle.dump(dataset, file)
            print("Data saved to", self.data_file)

        # Splitting the dataset
        self.setup_splits(dataset)

        return dataset

    def create_data(self):
        # Your data creation logic here
        return CVDataset(N=self.N_ts, gamma=self.gamma, noise_std=self.noise_std, t_span=self.t_span, t_treatment=self.t_treatment, seed=self.seed, output_dims=self.output_dims)

    def setup_splits(self, dataset):
        
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
