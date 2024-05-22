

import numpy as np
import os
from pathlib import Path
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import torch
from torch.utils.data import Dataset, DataLoader, Subset
import torchsde
import lightning as L

from utils_6 import CV_params, CV_params_prior_mu, CV_params_prior_sigma

use_cuda = torch.cuda.is_available()


class PhysiologicalSDE_batched(torchsde.SDEIto):
    def __init__(self, sigma, sigma_tx, params, confounder_type,non_confounded_effect):
        super(PhysiologicalSDE_batched, self).__init__(noise_type="diagonal")
        self.params = params
        self.sigma = sigma
        self.sigma_tx = sigma_tx
        self.confounder_type = confounder_type
        self.non_confounded_effect = non_confounded_effect

    def calculate_diext_dt(self,t, batch_size):
        # Assuming 't' might be a tensor with the shape [batch_size] or a scalar
        # Ensure 't' is at least 1D tensor with the shape [batch_size]
        t = t * torch.ones(batch_size)  # This line is only necessary if t might be a scalar

        # Calculate factor
        factor = -2 * (t - 5) / 5
        # Calculate exponential
        exponential = torch.exp(-((t - 5) / 5) ** 2)

        # Calculate diext_dt and ensure it's of shape [batch x 1]
        diext_dt = 5 * factor * exponential

        return diext_dt.unsqueeze(1)
    
    def create_treatment_effect(self):
        if self.confounder_type == 'visible':
            #confound on the initial NORMALISED PA - STATIC CONFOUNDER (not time dep)
            initp_transform  = 0.5+(self.params["confounding_pressure"]-0.75)/0.1
            ##print('initp_transform', initp_transform.shape)
            A_ = self.v_fun(initp_transform)
        elif self.confounder_type == 'partial':
            #confound on the initial NORMALISED SV - STATIC CONFOUNDER (not time dep)
            init_sv_transform  = 0.5+(self.params["confounding_sv"]-0.2)/0.1 
            A_ = self.v_fun(init_sv_transform)
        elif self.confounder_type == 'invisible':
            A_ = self.v_fun(self.params["confounder_random_number"])
        self.treatment_effect = A_
        return A_

    def v_fun(self, x):
        cos_term = torch.cos(5 * x - 0.2)
        square_term = (5 - x) ** 2
        return 0.02 * (cos_term * square_term) ** 2

    def f(self, t, y):
        ##print('y', y.shape)
        p_a = 100. * y[:, 0].unsqueeze(1)
        p_v = 100. * y[:, 1].unsqueeze(1)
        s = y[:, 2].unsqueeze(1)
        sv = 100. * y[:, 3].unsqueeze(1)
        i_ext = y[:, 4].unsqueeze(1)
        batch_size = y.shape[0]

        ##print('pa', p_a.shape)
        #print(t.item())


        if t.item() == 0:
            ##print(t)
            #print('saving params')
            self.params["confounding_pressure"] = y[:, 0]
            self.params["confounding_sv"] = y[:, 3]
            self.params["cv"] = torch.rand_like(y[:, 0]) * 100 + 10 if self.non_confounded_effect else self.params["cv"] 
            self.params["confounder_random_number"] = torch.rand_like(y[:, 0])
            ##print('self.params["init_pressure"]', self.params["init_pressure"].shape)
            ##print('self.params["init_sv"]', self.params["init_sv"].shape)
        
            #defining our treatment effect A_ (which we cannot explain with our model), and has a functional dep on the original blood pressure and a form of v_fun
        
        A_ = self.create_treatment_effect()
        #print('A_, i_ext', A_, i_ext)
        i_ext_tx_effect = A_.unsqueeze(1) * i_ext
        ##print('i_ext_tx_effect', i_ext_tx_effect.shape)

        #print('time, i_ext, p_a pv, s, sv', t.item(), i_ext_tx_effect[0].item(), p_a[0].item(), p_v[0].item(), s[0].item(), sv[0].item())   

        #print('t, i_ext_effect', t,i_ext_tx_effect )

        f_hr = s * (self.params["f_hr_max"] - self.params["f_hr_min"]) + self.params["f_hr_min"]
        r_tpr = s * (self.params["r_tpr_max"] - self.params["r_tpr_min"]) + self.params["r_tpr_min"] - self.params["r_tpr_mod"]

        dva_dt = -1. * (p_a - p_v) / r_tpr + sv * f_hr
        dvv_dt = -1. * dva_dt + i_ext_tx_effect
        dpa_dt = dva_dt / (self.params["ca"] * 100.)
        dpv_dt = dvv_dt / (self.params["cv"] * 10.)
        ds_dt = (1. / self.params["tau"]) * (1. - 1. / (1 + torch.exp(-self.params["k_width"] * (p_a - self.params["p_aset"]))) - s)
        dsv_dt = i_ext_tx_effect * self.params["sv_mod"]

        if self.params["treatment"] and (t >= self.params["t_treatment"]):
            #note that the 
            time_since_treatment = t - self.params["t_treatment"]
            diext_dt = self.calculate_diext_dt(time_since_treatment, batch_size)
            diext_dt = torch.relu(i_ext + diext_dt) - i_ext 
        else:
            #diext_dt = torch.full((batch_size, 1), 1)
            diext_dt = torch.zeros_like(dpa_dt)

        ##print('dpa_dt, dpv_dt, ds_dt, dsv_dt, diext_dt',dpa_dt.shape, dpv_dt.shape, ds_dt.shape, dsv_dt.shape, diext_dt.shape )

        
        diff_res = torch.concat([dpa_dt, dpv_dt, ds_dt, dsv_dt, diext_dt], dim=-1)

        ##print('diff_res', diff_res.shape)
        return diff_res
    
    def g(self, t, y):
        diffusion = torch.full_like(y, self.sigma)
        if self.params["treatment"] and (t >= self.params["t_treatment"]):
            diffusion[:, 4] = self.sigma_tx
        return diffusion





def scale_numbers(x, original_min, original_max, target_min, target_max):
    return target_min + ((target_max - target_min) * (x - original_min) / (original_max - original_min))


def init_random_state():
    max_ves = 64.0 - 10.0
    min_ves = 36.0 + 10.0

    max_ved = 167.0 - 10.0
    min_ved = 121.0 + 10.0

    max_pa = 85.0
    min_pa = 75.0

    max_pv = 70.0
    min_pv = 30.0

    max_s = 0.25
    min_s = 0.15

    max_sv = 100
    min_sv = 90

    init_ves = (np.random.rand() * (max_ves - min_ves) + min_ves) / 100.0
    # init_ves = 50.0 / 100.0

    init_ved = (np.random.rand() * (max_ved - min_ved) + min_ved) / 100.0
    # init_ved = 144.0 / 100.0

    init_pa = (np.random.rand() * (max_pa - min_pa) + min_pa) / 100.0
    init_pv = (np.random.rand() * (max_pv - min_pv) + min_pv) / 100.0
    init_s = (np.random.rand() * (max_s - min_s) + min_s)
    init_sv = (np.random.rand() * (max_sv - min_sv) + min_sv) / 100.0

    init_i_ext = 0

    init_state = np.array([init_pa, init_pv, init_s, init_sv, init_i_ext])
    ##print('init_state', init_state)
    return init_state



def create_cv_data(N,gamma,noise_std, sigma_tx, confounder_type, non_confounded_effect, t_span, t_treatment, seed, post_treatment_dims, pre_treatment_dims, normalize = False):

    np.random.seed(seed)

    X = []
    Y_0 = []
    Y_1 = []
    init_state_list = []
    
    params = {"r_tpr_mod": 0.0, #the mod is in case we want to simulate decreasing the total peripheral resistance i.e. shock
            "f_hr_max": 3.0,
            "f_hr_min": 2.0 / 3.0,
            "r_tpr_max": 2.134,
            "r_tpr_min": 0.5335,
            "sv_mod": 0.001,  ## this is also added on from the original model to simulate effect of fluid directly on the stroke volume
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
            "t_treatment" : t_treatment,
            }
    
    params_treatment = params.copy()
    params_treatment["treatment"]=True
    params_notreatment = params.copy()
    params_notreatment["treatment"]=False
    

    t = np.arange(t_span).astype(float)
    t_tensor = torch.tensor(t, dtype=torch.float32)

    #print(f"N: {N}")
    #print(f"Gamma: {gamma}")
    #print(f"Noise Std: {noise_std}")
    #print(f"Sigma Tx: {sigma_tx}")
    #print(f"Confounder Type: {confounder_type}")
    #print(f"Non Confounded Effect: {non_confounded_effect}")
    #print(f"t_span: {t_span}")
    #print(f"t_treatment: {t_treatment}")
    #print(f"Seed: {seed}")
    #print(f"Post Treatment Dims: {post_treatment_dims}")
    #print(f"Pre Treatment Dims: {pre_treatment_dims}")
    #print(f"Normalize: {normalize}")

    #print('creating initial random states for both treated and untreated ')
    init_state_list = []
    for i in range(N):
        init_state_list.append(init_random_state())
    init_state_tensor = torch.tensor(np.array(init_state_list), dtype=torch.float32)

    #creating treated from those initial random states
    #print('creating treated')
    print('init_state_tensor', init_state_tensor.shape, 't_tensor', t_tensor.shape, )
    sde = PhysiologicalSDE_batched(sigma = noise_std, sigma_tx=sigma_tx, confounder_type=confounder_type , non_confounded_effect = non_confounded_effect, params=params_treatment)
    Y_1 = torchsde.sdeint(sde, init_state_tensor, t_tensor, method='euler').squeeze(1)

    #print('creating untreated')
    #created untreated from the same initial random satates
    sde = PhysiologicalSDE_batched(sigma = noise_std, sigma_tx=sigma_tx, confounder_type=confounder_type , non_confounded_effect = non_confounded_effect, params=params_notreatment)
    Y_0 = torchsde.sdeint(sde, init_state_tensor, t_tensor, method='euler').squeeze(1)
        
    X = init_state_tensor
    init_state = init_state_tensor
        
    #print('Assigning confounding factors')
    ##print('init_state', init_state.shape)
    if confounder_type == 'visible':
        scaled_pa = scale_numbers(x=init_state[:, 0], original_min=0.75, original_max=0.85, target_min=0, target_max=1)
        p = torch.sigmoid(gamma*scaled_pa) # use the arterial pressure as visible confounder
    
    elif confounder_type == 'partial':# use the stroke volume as a partially visible confounder
        scaled_sv = scale_numbers(x=init_state[:, 3], original_min=0.9, original_max=1, target_min=0, target_max=1)
        p  = torch.sigmoid(gamma*scaled_sv) 

    elif confounder_type == 'invisible':
        p =  torch.sigmoid(gamma*torch.rand(N)) 

    ##print('p', p.shape)
    T = torch.zeros(N)
    #T determines which trajectories as selected as treated (overlap level is controlled by gamma)
    T[torch.rand(N)<p] = 1

    #all_trajectories = torch.cat([Y_0, Y_1], dim = 1)
    Y_0 = Y_0[:, :, :4].permute(1, 0, 2)  # drop the dim used to create I-external and permute to [batch, seq_len, dim]
    Y_1 = Y_1[:, :, :4].permute(1, 0, 2)
    T_expanded = T[:, None, None]
    #print('Y0, Y1, T_expanded', Y_0.shape, Y_1.shape, T_expanded.shape)

    # the 'factual' trajectories are the UNtreated outcome (Y0) for the not Treated (1-T) and the Treated outcome (Y1) for the factually Treated (T)
    Y_fact = Y_0 * (1-T_expanded)+ Y_1 *T_expanded

    # the 'COUNTERfactual' trajectories are the UNtreated outcome (Y0) for the Treated (T) and the Treated outcome (Y1) for the factually UNtreated (1-T)
    # we would never actually have access to the counterfactual other than in this situation where we are simulating it 
    Y_cf = Y_0 * T_expanded + Y_1 * (1-T_expanded)

    
    Y_fact_np = Y_fact.detach().cpu().numpy()
    states_mean = Y_fact_np.mean(axis=(0, 1))
    states_min = Y_fact_np.min(axis=(0, 1))
    states_max = Y_fact_np.max(axis=(0, 1))
    #print('states_mean', states_mean, 'states_min', states_min, 'states_max', states_max)

    
    Y_fact_until_t = Y_fact[:, :t_treatment, :]
    mu = Y_fact_until_t.mean([0,1])
    std = Y_fact_until_t.std([0,1])
    
    if normalize:
        Y_fact = (Y_fact - mu)/std
        Y_cf = (Y_cf - mu)/std
        mu_X = X.mean([0,1])
        std_X = X.std([0,1])
        X = (X-mu_X)/std_X

    CV_params_prior_mu['pa'] = mu[0]*100
    CV_params_prior_mu['pv'] = mu[1] *100
    CV_params_prior_mu['s'] = mu[2]
    CV_params_prior_mu['sv'] = mu[3] *100

    CV_params_prior_sigma['pa'] = std[0]*100
    CV_params_prior_sigma['pv'] = std[1]*100
    CV_params_prior_sigma['s'] = std[2]
    CV_params_prior_sigma['sv'] = std[3]*100

    CV_params['max_pa'] = (mu[0] + 2.5 * std[0]) * 100
    CV_params['min_pa'] = (mu[0] - 2.5 * std[0]) * 100
    CV_params['max_pv'] = (mu[1] + 2.5 * std[1]) * 100
    CV_params['min_pv'] = (mu[1] - 2.5 * std[1]) * 100
    CV_params['max_s'] = (mu[2] + 2.5 * std[2]) 
    CV_params['min_s'] = (mu[2] - 2.5 * std[2]) 
    CV_params['max_sv'] = (mu[3] + 2.5 * std[3]) * 100
    CV_params['min_sv'] = (mu[3] - 2.5 * std[3]) * 100

    
    # Now split these factual and counterfactual trajectories by the 'before' and 'after treatment' so we have a baseline 
    pre_treat_mask = (t<=t_treatment)
    post_treat_mask = (t>t_treatment)
    
    # We define X as the Factual trajectory BEFORE treatment, and X_ as the COUNTERfactual traj BEFORE treatment 
    X_static = X
    X = Y_fact[:,pre_treat_mask][:,:,pre_treatment_dims]
    X_ = Y_cf[:,pre_treat_mask][:,:,pre_treatment_dims]

    # We redfine Y_fact as the Factual trajectory AFTER treatment, and Y_cf as the COUNTERfactual traj AFTER treatment 
    # We are selecting the DIASTOLIC BP (output dim = 1) as the one to maintain.. this is because the fluid is only really affecting this within the time values
    full_fact_traj = Y_fact
    full_CF_traj = Y_cf

    Y_fact = Y_fact[:,post_treat_mask][:,:,post_treatment_dims] 
    Y_cf = Y_cf[:,post_treat_mask][:,:,post_treatment_dims]

    # we split the time vector also as before and after treatment 
    t_x = t[pre_treat_mask]
    t_y = t[post_treat_mask]
    # and get it to match the dimensions, so it's not a vector t, but a matrix of dimensions N by before_tx and N by after_tx
    t_X = torch.Tensor(np.tile(t_x[None,:],(X.shape[0],1)))
    t_Y = torch.Tensor(np.tile(t_y[None,:],(Y_fact.shape[0],1))) - t_x[-1]
    t_FULL = t_tensor.unsqueeze(0).repeat(X.shape[0], 1)
    expert_ODE_size = X.shape[1] - 1 #need to remove the i_ext which is only for creating data not models


    return X, X_static, T, Y_fact, Y_cf, p, init_state, t_X, t_Y, expert_ODE_size, t_FULL, full_fact_traj,full_CF_traj, sde




def create_load_save_data(dataset_params, data_path):
    if not os.path.exists(data_path):
        os.makedirs(data_path)
        print(f"Created directory: {data_path}")

    non_confounded_effect_str = 'true' if dataset_params['non_confounded_effect'] else 'false'
    normalize = 'true' if dataset_params['normalize'] else 'false'
    post_treatment_dims_str = ''.join(map(str, dataset_params['post_treatment_dims']))
    pre_treatment_dims_str = ''.join(map(str, dataset_params['pre_treatment_dims']))

    # Create a descriptive file name
    filename = f"{dataset_params['confounder_type']}_RCE{non_confounded_effect_str}_N{dataset_params['N']}_G{dataset_params['gamma']}_Dstd{dataset_params['noise_std']}_Tstd{dataset_params['sigma_tx']}_Pre{pre_treatment_dims_str}_Post{post_treatment_dims_str}_Norm{normalize}.pt"
    final_data_path = os.path.join(data_path, filename)

    if os.path.exists(final_data_path):
        print("Loading existing dataset.")
        data = torch.load(final_data_path)

    else:
        print("Creating and saving a new dataset.")
        # Generate data
        X, X_static, T, Y_fact, Y_cf, p, init_state, t_X, t_Y, expert_ODE_size, t_FULL, full_fact_traj,full_CF_traj, sde = create_cv_data(N = dataset_params['N'],
                                                                                                                                      gamma = dataset_params['gamma'],
                                                                                                                                      noise_std = dataset_params['noise_std'], 
                                                                                                                                      sigma_tx = dataset_params['sigma_tx'], 
                                                                                                                                      confounder_type = dataset_params['confounder_type'], 
                                                                                                                                      non_confounded_effect = dataset_params['non_confounded_effect'], 
                                                                                                                                      t_span = dataset_params['t_span'], 
                                                                                                                                      t_treatment = dataset_params['t_treatment'], 
                                                                                                                                      seed = dataset_params['seed'], 
                                                                                                                                      post_treatment_dims = dataset_params['post_treatment_dims'], 
                                                                                                                                      pre_treatment_dims = dataset_params['pre_treatment_dims'], 
                                                                                                                                      normalize = dataset_params['normalize'])
    
        data = {'X': X, 'T': T,'Y_fact': Y_fact, 'Y_cf': Y_cf,'p': p,'init_state': init_state,'t_X': t_X,'t_Y': t_Y,'t_full': t_FULL,'full_fact_traj': full_fact_traj,'full_CF_traj': full_CF_traj }
        # Save the dataset
        torch.save(data, final_data_path)
    
    return data




class CVDataset_loaded(Dataset):
    def __init__(self, data):
        # Unpack the data
        self.X = data['X']
        self.T = data['T']
        self.Y_fact = data['Y_fact']
        self.Y_cf = data['Y_cf']
        self.p = data['p']
        self.init_state = data['init_state']
        self.t_X = data['t_X']
        self.t_Y = data['t_Y']
        self.t_full = data['t_full']
        self.full_fact_traj = data['full_fact_traj']
        self.full_CF_traj = data['full_CF_traj']

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx], self.Y_fact[idx], self.T[idx], self.Y_cf[idx], self.p[idx], self.init_state[idx],self.t_X[idx], self.t_Y[idx], self.t_full[idx], self.full_fact_traj[idx],self.full_CF_traj[idx]




class CVDataModule_final(L.LightningDataModule):
    def __init__(self, dataset_path, batch_size=32, num_workers =1):
        super().__init__()
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.dataset = None
        self.num_workers = num_workers

    def setup(self, stage=None):
        # Load the dataset
        data = torch.load(self.dataset_path)
        self.dataset = CVDataset_loaded(data)
        
        train_idx = np.arange(len(self.dataset))[:int(0.5*len(self.dataset))]
        val_idx = np.arange(len(self.dataset))[int(0.5*len(self.dataset)):]
        test_idx = val_idx[int(len(val_idx)/2):]
        val_idx = val_idx[:int(len(val_idx)/2)]

        self.train = Subset(self.dataset,train_idx)
        self.val = Subset(self.dataset,val_idx)
        self.test = Subset(self.dataset,test_idx)

    def train_dataloader(self):
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            shuffle=True,
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


