# Counterfactual_ICU

ACS MPhil thesis project

- **Experiment_1** : Standard ODE with variational RNN with pytorch lightning. RNN takes in the whole trajectory, passess to ODE which creates latents, which are then passed to output function to match the trajectory.

- **Experiment_2** : Controlled Ordinary Differential Equation, with variational RNN (silenced by the KL_param), that takes in the non-treatment baseline trajectory, passes to an ODE which is itself modified by the controlled treatment_fun as it creates the latents forward in time, and then pointwise output function to recreate the post-treatment trajectory. Plotly on wandb. Uses pytorch lightning.

- **Experiment_3**: Controlled Stochastic Differential equation, with a non-variational RNN, that takes in non-tx baseline traj, passes to SDE w treatment_fun exerting control. Multiple samples of the latents are created which are then brought back to observed by pointwise output_fun MLP. 3 separate plots with plotly, and uses Lightning instead of Pytorch-Lightning. Experiment_3 has it's OWN environment yaml to use.

- **Experiment_4**:
  - Generalising data to include treatments at different times, and encoder to take that into account
  - Generalising latent to be either expert only, hybrid SDE or SDE only

**Experiment ideas**

Currently CV data is being created with no gaps between the datapoints... which is unlikely in real clinical dataset
so need to convert to having observations at specific time points and the encoder taking these into account

also we are assuming that data is fixed and then we go about analysing it, when actually data is always coming and we need to rapidly place it in a correct latent space interactively
do we need to resubmit the whole trajectory up to that new datapoint every time we get something new?
or if instead we learn an amortised latent model as new data comes in we can just apply it
so we train on full dataset style and then we apply to data on the fly...

but if we train on full dataset then we should have a mask?
