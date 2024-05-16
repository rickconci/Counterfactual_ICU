# Counterfactual_ICU

ACS MPhil thesis project

- **Experiment_1** : Standard ODE with variational RNN with pytorch lightning. RNN takes in the whole trajectory, passess to ODE which creates latents, which are then passed to output function to match the trajectory.

- **Experiment_2** : neural ODE, with variational RNN (silenced by the KL_param), that takes in the non-treatment baseline trajectory, passes to an ODE which is itself modified by the controlled treatment_fun as it creates the latents forward in time, and then pointwise output function to recreate the post-treatment trajectory. Plotly on wandb. Uses pytorch lightning.

- **Experiment_3**: Stochastic Differential equation, with a non-variational RNN, that takes in non-tx baseline traj, passes to SDE w treatment_fun exerting control. Multiple samples of the latents are created which are then brought back to observed by pointwise output_fun MLP. 3 separate plots with plotly, and uses Lightning instead of Pytorch-Lightning. Experiment_3 has it's OWN environment yaml to use.

- **Experiment_4**:
  - Generalising latent to be either expert only, hybrid SDE or SDE only

- **Experiment_5**:
  - Fixing SDE bugs especially RNN encoder... still not quite right but runs much better
  - adjusted expert ODE with sigmoid rescaling - although better should be for truncated gaussian prior with linear rescaling!!

- **Experiment_6**:
  - adjusting hybrid SDE!! so that when there is a control acting on the latents (hidden or not) the SDEs can identify it
    - OPTIONS:
      - Tx: we know the time & dose , YES  functional dependence YES the functional form:
      - Tx: we know the time & dose, YES functional dep, NO the functional form:
      - Tx: we know the time & dose, NO the functional dep, YES the functional form:
      - Tx: we know the time & dose, NO the functional dep, NO the functional form:

      - Path: we DON'T konw the time & dose, YES the functional dep, YES the functional form
      - Path: we DON'T konw the time & dose, YES the functional dep, NOT the functional form
      - Path: we DON'T konw the time & dose, NO the functional dep, YES the functional form
      - Path: we DON'T konw the time & dose, NO the functional dep, NO the functional form

  - Adjust CV_data to include external control as pathology... (or can even assume that the treatment is a kind of external control )

  - adjustment of expert ode to have truncated gaussian prior with linear rescaling on all (or maybe match with hybrid?) latents not just some of them

  - control of observed data to be irregular based on hawkes process
  - clarify CV dataset creation to have various clinically relevant trajectories i.e. normal then infection then treatment then another treatment etc

**Experiment ideas**

Currently CV data is being created with no gaps between the datapoints... which is unlikely in real clinical dataset
so need to convert to having observations at specific time points and the encoder taking these into account

also we are assuming that data is fixed and then we go about analysing it, when actually data is always coming and we need to rapidly place it in a correct latent space interactively
do we need to resubmit the whole trajectory up to that new datapoint every time we get something new?
or if instead we learn an amortised latent model as new data comes in we can just apply it
so we train on full dataset style and then we apply to data on the fly...

but if we train on full dataset then we should have a mask?
