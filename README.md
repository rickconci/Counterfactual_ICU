# Counterfactual_ICU

ACS MPhil thesis project

- **Experiment_1** : Standard ODE with variational RNN with pytorch lightning. RNN takes in the whole trajectory, passess to ODE which creates latents, which are then passed to output function to match the trajectory.

- **Experiment_2** : Controlled Ordinary Differential Equation, with variational RNN (silenced by the KL_param), that takes in the non-treatment baseline trajectory, passes to an ODE which is itself modified by the controlled treatment_fun as it creates the latents forward in time, and then pointwise output function to recreate the post-treatment trajectory. Plotly on wandb. Uses pytorch lightning.

- **Experiment_3**: Controlled Stochastic Differential equation, with a non-variational RNN, that takes in non-tx baseline traj, passes to SDE w treatment_fun exerting control. Multiple samples of the latents are created which are then brought back to observed by pointwise output_fun MLP. 3 separate plots with plotly, and uses Lightning instead of Pytorch-Lightning. Experiment_3 has it's OWN environment yaml to use.
