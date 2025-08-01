# Counterfactual_ICU

## ACS MPhil thesis June 2024 

### To create the meds_matrix dataset

- Create a folder named 'data' with the following files from MIMICS-IV:
  - icu/icustays.csv saved as 'icustays.csv'
  - icu/chartevents.csv saved as 'chartevents.csv'
  - icu/inputevents.csv saved as 'inputevents.csv'
  - hosp/patients.csv saved as 'patients.csv'
- run `python src/data_processing/preprocessing.py`

### To run 

1) Set up environment & log into Weight and Biases to track experiments. 

```
conda env create -f Final_Env.yaml
conda activate ExpLight
wandb login
```

2) Run batch to recreate results from Experiment 1 and 2:

```
./exp_HybridDE.sh
```

3) For more targeted approach use the following command, run ```python main_beta.py```. The default arguments were the same as those used in the thesis. 




## Latest versions

- **Version_7_beta_2** is the final version used for this project. It includes:
  - `main_beta`: this organises the dataset and model parameters to train and test on Lightning
  - `CV_data_beta`: a dyamic data creation setup with a hard visible confounding and in and out-of-distribution testing. It takes as control inputs the folloiwng dataset parameters:
    - 'fixed_tx': [True] whether to simulate trajectories all with the same treamtent, or varying both in volume of infusion and duration. 
    - 'include_all_inputs':[True] whether to include all the expert cardiovascular parameters in the observed data or not. 
    - 'gamma':[6] Defines the amount of overlap. The higher gamma, the lower the overlap. 
    - 'sigma_tx': [0.01]the diffusion term present in the creation of the simulated data. 
    - 'confounder_type': ['partial_hard']The type of confounder. 'partial_hard' matches the confounder description in the written thesis. 
    - 'non_confounded_effect': [False] Whether to include a variable that impacts outcome but not treatment assignment, in effect noise. 
    - 'noise_std': [0] the amount of gaussian noise added post-hoc to the observed time series data
    - 't_span': [60] total simulated seconds for data creation 
    - 't_treatment': [45] the time at which treatment is given 
    - 't_cutoff':[40] we cut the first 40 seconds of the simulated data as the cardiovascular model is self-adjusting from random inputs. 
    - 'seed': seed to set up both the creation of the data and the split into training, validation and test. 
    - 'pre_treatment_dims': [0,1]: 0 speficies the arterial pressure, 1 the venous pressure. 
    - 'post_treatment_dims': [0] again specifies the arterioal pressure as the only output trajectory that we use to predict and train. 
    - 'normalize': [False] whether the input data should be normalised or not. 
    - 'N': [1280] the total number of datapoints created. 
  
  - `model_beta.py`. In this version, the HybridSDE has *two* control outputs that are integrated with the expert cardiovascular model. it can take in inputs that include just the observed physiological data at time of treatment, or also a neural encoder that can provide both the expert variables and further information only to the neural SDE. 
  - `evals` which takes in the model checkpoints and performs further analyses and plots, including uncertainty quantification. 


- **Version_7_beta_1** 
  - this is a complete implementation of the Hybrid SDE as both a decoder and *variational encoder* trained to infer the expert physiologival variables that need to be inferred. It was not implemented in conjunction with the decoder as an end-to-end model for two main reasons: first the causal estimation task setup requires accurate values of the confounded variables SV, and second an intrinsic challenge to inverse physiology is that ultimate many latent expert variables are described by distributions until further information arrives. The transition form a distribution of initial values to a distribution of neural SDE samples and expert ODEs is very interesitng but also challenging and for future work. 



## Historical additions:

- **Version_7**
  - continuation of 6b. the decoder HybridSDE takes in all the params as input y rather than the 'fixed' CV params from the dictionary. Therefore can it can either have an encoder that feeds these in, or it is fed in the correct inputs.

- **Version 6b**:
  - considerable improvements on version 6. Version 6 doesn't in fact even have the model or dataset because it's converted to 6b. 

- **Version_6**:
  - adjusting hybrid SDE!! so that when there is a control acting on the latents (hidden or not) the SDEs can identify it
  - Adjust CV_data to include external control as pathology... (or can even assume that the treatment is a kind of external control )
  - adjustment of expert ode to have truncated gaussian prior with linear rescaling on all (or maybe match with hybrid?) latents not just some of them
  - control of observed data to be irregular based on hawkes process
  - clarify CV dataset creation to have various clinically relevant trajectories i.e. normal then infection then treatment then another treatment etc

- **Version_5**:
  - Fixing SDE bugs especially RNN encoder... still not quite right but runs much better
  - adjusted expert ODE with sigmoid rescaling - although better should be for truncated gaussian prior with linear rescaling!!

- **Version_4**:
  - Generalising latent to be either expert only, hybrid SDE or SDE only

- **Version_3**: Stochastic Differential equation, with a non-variational RNN, that takes in non-tx baseline traj, passes to SDE w treatment_fun exerting control. Multiple samples of the latents are created which are then brought back to observed by pointwise output_fun MLP. 3 separate plots with plotly, and uses Lightning instead of Pytorch-Lightning. Experiment_3 has it's OWN environment yaml to use.

- **Version_2** : neural ODE, with variational RNN (silenced by the KL_param), that takes in the non-treatment baseline trajectory, passes to an ODE which is itself modified by the controlled treatment_fun as it creates the latents forward in time, and then pointwise output function to recreate the post-treatment trajectory. Plotly on wandb. Uses pytorch lightning.

- **Version_1** : Standard ODE with variational RNN with pytorch lightning. RNN takes in the whole trajectory, passess to ODE which creates latents, which are then passed to output function to match the trajectory.


-----





