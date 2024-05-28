from lightning import LightningModule
import torch
from model_7 import Hybrid_VAE_SDE
from CV_data_7 import CVDataModule_IID,CVDataModule_OOD, create_load_save_data
from utils_7 import scale_unnormalised_experts
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt

device = torch.device("mps")  # Use "cpu" for CPU

def update_dict(main_dict, new_dict):
    for key in new_dict.keys():
        main_dict[key].append(new_dict[key])
    return main_dict

def load_model_checkpoint(path):
    checkpoint = torch.load(path, map_location='cpu')  # Adjust map_location as needed
    model = Hybrid_VAE_SDE.load_from_checkpoint(path)
    model.to(device)

    return model


def load_data_loader(dataset_params):
    
    data_path = '/Users/riccardoconci/Local_documents/ACS submissions/THESIS/Counterfactual_ICU/Version_7/data_created'
    dataset_params['r_tpr_mod'] = 0.0
    train_val_data = create_load_save_data(dataset_params, data_path)
    dataset_params['r_tpr_mod'] = -0.5 
    test_data = create_load_save_data(dataset_params, data_path)

    cv_data_module_IID = CVDataModule_IID(train_val_data = train_val_data, batch_size=128, num_workers = 4)
    cv_data_module_OOD = CVDataModule_OOD(OOD_test_data = test_data, batch_size=128, num_workers = 4)
    cv_data_module_IID.setup()
    cv_data_module_OOD.setup()
    return [cv_data_module_IID, cv_data_module_OOD]


def run_through_test_set(model, cv_data_module_list, plot_with_trimming=False):

    curve_cf_dict = {"random" : [], "uncertainty": [], "propensity": []}
    curve_f_dict = {"random" : [], "uncertainty": [],  "propensity": []}
    curve_pehe_dict = {"random" : [], "uncertainty": [],  "propensity": []}
    
    model.eval()

    repeats = 1
    t_lim_trimming_start = 0
    t_lim_trimming_end = 14
    last_model = True

    print('Running model through test set')

    
    with torch.no_grad():
        for cv_mod in cv_data_module_list:
            print('using ', cv_mod)
            print("Number of batches in test dataloader:", len(list(cv_mod.test_dataloader())))
            Y_hat_samples_list = []
            Y_alea_std_list = []
            Y_hat_cf_samples_list = []
            Y_alea_std_cf_list = []
            Y_list = []
            Y_cf_list = []
            p_list = []
            T_list = []
            X_list = []
            
            for i,batch in enumerate(cv_mod.test_dataloader()):
                
                print('batch: ', i )
                batch = [t.to(device) if isinstance(t, torch.Tensor) else t for t in batch]
                X, Y, T, Y_cf, p, init_states, time_pre, time_post, time_FULL, full_fact_traj, full_cf_traj = batch
                z1_real = full_fact_traj[:, X.shape[1], :]
                z1_real = scale_unnormalised_experts(z1_real).to(device)
                z1_ML = torch.zeros(z1_real.shape[0],model.encoder_SDENN_dims).to(device)
                z1_input = torch.cat([z1_real, z1_ML  ], dim =-1)
                z1_input = z1_input.unsqueeze(1).repeat(1, model.num_samples, 1) 

                
                Y_list.append(Y)
                Y_cf_list.append(Y_cf)
                p_list.append(p)
                T_list.append(T)
                X_list.append(z1_input)
                times = torch.arange(Y.shape[1]).float()
                #times = torch.arange(11).float()


                Y_hat_list = []
                Y_hat_cf_list = []
                Y_alea_std = []
                Y_alea_cf_std = []
                

                for _ in tqdm(range(repeats), desc="Processing", unit="iteration"):
                    #print('factual')
                    latent_traj, logqp_path, i_ext_path = model.forward_latent(init_latents = z1_input, 
                                                            ts = time_post[0], 
                                                            Tx= T, 
                                                            time_to_tx = torch.tensor([0]) )
                    predicted_traj = model.forward_dec(latent_traj.to('mps') )
                    Y_std = torch.zeros_like(predicted_traj)
                    #print(f"Batch {i} predicted_traj shape: {predicted_traj.shape}")

                    Y_hat_list.append(predicted_traj)
                    Y_alea_std.append(torch.sqrt(torch.sigmoid(Y_std)))

                    #print('counterfactual')
                    T_cf = (~T.bool()).float()
                    latent_traj_cf, logqp_path_cf, i_ext_path_cf = model.forward_latent(init_latents = z1_input, 
                                                            ts = time_post[0], 
                                                            Tx= T_cf, 
                                                            time_to_tx = torch.tensor([0]) )
                    predicted_traj_cf = model.forward_dec(latent_traj )

                    Y_std_cf = torch.zeros_like(predicted_traj_cf)
                        
                    Y_hat_cf_list.append(predicted_traj_cf)
                    Y_alea_cf_std.append(torch.sqrt(torch.sigmoid(Y_std_cf)))

                #combining across repeats
                Y_hat_samples = torch.cat(Y_hat_list, 1)
                Y_alea_std = torch.cat(Y_alea_std, 1)
                Y_hat_cf_samples = torch.cat(Y_hat_cf_list, 1)
                Y_alea_std_cf = torch.cat(Y_alea_cf_std, 1)

                Y_hat_samples_list.append(Y_hat_samples)
                Y_alea_std_list.append(Y_alea_std)
                Y_hat_cf_samples_list.append(Y_hat_cf_samples)
                Y_alea_std_cf_list.append(Y_alea_std_cf)

            #combining across batches 
            Y_hat_samples = torch.cat(Y_hat_samples_list, 0)
            Y_alea_std = torch.cat(Y_alea_std_list, 0)
            Y_hat_cf_samples = torch.cat(Y_hat_cf_samples_list, 0)
            Y_alea_std_cf = torch.cat(Y_alea_std_cf_list, 0)

            Y = torch.cat(Y_list, 0)
            Y_cf = torch.cat(Y_cf_list, 0)
            p = torch.cat(p_list, 0)
            T = torch.cat(T_list, 0)
            X = torch.cat(X_list, 0)
            

            factual_mse = calculate_rmse(Y, Y_hat_samples.mean(1))
            counterfactual_mse = calculate_rmse(Y_cf, Y_hat_cf_samples.mean(1))
            pehe = calculate_PEHE(Y, Y_hat_samples.mean(1), Y_cf, Y_hat_cf_samples.mean(1))
            print('factual_rmse', factual_mse)
            print('counterfactual_mse', counterfactual_mse)
            print('pehe', pehe)


            if plot_with_trimming:
                print('Finished running through test set and collecting data')

                Y_hat_mean = Y_hat_samples.mean(1) #averaging across samples
                Y_hat_std  = Y_hat_samples.std(1) #averaging across samples

                print('Y_hat_mean', Y_hat_mean.shape)
                print('Y_hat_std', Y_hat_std.shape)

                Y_hat_up = (Y_hat_samples+1.96*Y_alea_std).max(1)[0]
                Y_hat_down = (Y_hat_samples-1.96*Y_alea_std).min(1)[0]
                Y_hat_diff = Y_hat_up - Y_hat_down

                Y_hat_cf_mean = Y_hat_cf_samples.mean(1)
                Y_hat_cf_std  = Y_hat_cf_samples.std(1)

                Y_hat_cf_up = (Y_hat_cf_samples+1.96*Y_alea_std_cf).max(1)[0]
                Y_hat_cf_down = (Y_hat_cf_samples-1.96*Y_alea_std_cf).min(1)[0]
                Y_hat_cf_diff = Y_hat_cf_up - Y_hat_cf_down

                print('now onto plotting!!')

                times_plot = times
                print('plotting counterfactual')
                _, cf_dict = plot_trimming_prop(Y_cf, Y_hat_cf_mean, Y_hat_cf_diff, data_type="CFactual", p=(p * T + (1 - p) * (1 - T)), normalize=False)
                print('plotting factual')
                _, f_dict = plot_trimming_prop(Y,Y_hat_mean,Y_hat_diff, data_type="Factual", p = -(p*T +(1-p)*(1-T)), normalize=False)
                print('plotting PEHE')
                _, pehe_dict = plot_trimming_prop(Y-Y_cf,Y_hat_mean-Y_hat_cf_mean,Y_hat_diff+Y_hat_cf_diff, data_type="PEHE", p=torch.abs(p-0.5), normalize=False)

                curve_cf_dict = update_dict(curve_cf_dict, cf_dict)
                curve_f_dict = update_dict(curve_f_dict, f_dict)
                curve_pehe_dict = update_dict(curve_pehe_dict, pehe_dict)

                return curve_cf_dict, curve_f_dict, curve_pehe_dict


def calculate_rmse(true,predicted ):
    if true.shape != predicted.shape:
        raise ValueError("Both tensors must have the same shape")
    mse = torch.mean((true - predicted) ** 2)
    rmse = torch.sqrt(mse)
    return rmse.item()

def calculate_PEHE(true, true_cf, predicted, predicted_cf):
    ite = (true_cf- true)
    ite_hat = (predicted_cf - predicted)

    PEHE = calculate_rmse(ite, ite_hat)
    return PEHE



def compute_mean_squared_error(Y, mu):
    """Compute mean squared error between Y and mu for all time points."""
    errors = (Y - mu).pow(2).mean(1)
    return errors

def sort_errors(errors, std_devs, method='uncertainty'):
    """Sort errors based on standard deviations or randomly."""
    if method == 'uncertainty':
        sort_idx = torch.sort(std_devs, 0, descending=False)[1]
    else:
        sort_idx = np.random.permutation(len(std_devs))
    return errors[sort_idx]

def compute_cumulative_mean(errors):
    """Compute cumulative mean of errors."""
    if isinstance(errors, torch.Tensor):
        errors = errors.cpu()  # Move tensor to CPU if it's not already.
    cumulative_means = np.cumsum(errors.numpy()) / np.arange(1, len(errors) + 1)
    return cumulative_means

def plot_trimming_prop(Y, mu, std, data_type, p=None, normalize=False, filename="plot.png"):
    """Plot trimming proportions for various sorting methods using the full timeline, and save the figure."""
    Y, mu = Y.cpu(), mu.cpu()  # Ensure data and predictions are on CPU
    mse = compute_mean_squared_error(Y, mu)
    if normalize:
        mse /= mse.mean()
    
    std_devs = std.cpu().mean(1)  # Move std to CPU and compute mean
    mse_sorted = sort_errors(mse, std_devs)
    
    mse_random = sort_errors(mse, std_devs, method='random')
    
    if p is not None:
        p = p.cpu()  # Move propensity scores to CPU
        p_idx = torch.sort(p, 0)[1]
        mse_p = mse[p_idx]
    else:
        mse_p = np.zeros_like(mse_random)
    
    # Calculate cumulative means
    mse_kept = compute_cumulative_mean(mse_sorted)
    mse_kept_random = compute_cumulative_mean(mse_random)
    mse_kept_p = compute_cumulative_mean(mse_p)

    # Plotting
    fig, ax = plt.subplots()
    proportions = np.linspace(0, 1, len(mse_sorted))
    ax.plot(proportions, mse_kept, label="Uncertainty based")
    ax.plot(proportions, mse_kept_random, label="Random")
    ax.plot(proportions, mse_kept_p, label="Propensity")
    ax.set_xlabel("Proportion of data kept in the dataset")
    ax.set_ylabel("Average MSE")
    ax.set_title(f"Comparison of approaches for data trimming - {data_type}")
    ax.legend()
    plt.show()
    fig.savefig(data_type + '_evaluation_plot.png')

    return fig, {
        "uncertainty": mse_kept,
        "random": mse_kept_random,
        "propensity": mse_kept_p
    }


if __name__ == '__main__':
    dataset_params = {
            'include_all_inputs':True, 
            'gamma': 10,
            'sigma_tx': 2,
            'confounder_type': 'partial',

            'non_confounded_effect': False,
            'noise_std': 0.0,
            't_span': 60,
            't_treatment': 45,
            't_cutoff':40,
            'seed': 1,
            'pre_treatment_dims': [0, 1], # = pa + pv
            'post_treatment_dims': [0], # = pa 
            'normalize': False,
            'N': 1000
        }

    chk_last_path = '/Users/riccardoconci/Local_documents/ACS submissions/THESIS/Counterfactual_ICU/Version_7/model_checkpoints/sd=44_gm=10_cnf=partial_enc=none_txsig=0.1_revert=True_klw=0.001_SDEhd=400 (1)/last-v1.ckpt'
    check_best_path = '/Users/riccardoconci/Local_documents/ACS submissions/THESIS/Counterfactual_ICU/Version_7/model_checkpoints/sd=44_gm=10_cnf=partial_enc=none_txsig=0.1_revert=True_klw=0.001_SDEhd=400 (1)/best-epoch=104-val_loss=0.00.ckpt'
    '/Users/riccardoconci/Local_documents/ACS submissions/THESIS/Core_paper_implementations/cf-ode/causalode/CausalODE_RC_data/80z3owyv/checkpoints/epoch=495-step=5952.ckpt'
    print('Loading model!')
    model = load_model_checkpoint(check_best_path)
    print('loading dataloder!')
    cv_data_module = load_data_loader(dataset_params)
    
    run_through_test_set(model, cv_data_module)