
## TO DO

**Deadline 7th May**:

- [ ] fix output function of experiment 3 (make sure it's the same as hyland)
- [ ] rerun exp3 on HPC as batch job using Hyland values
- [ ] get hyland paper also on HPC so can run to double check that doing the same
- [ ] plot the tx function learned by the model
- [ ] Read and understand VDS code + meet w Francesco
- [ ] Integrate VDS w Hyland and show uncertainty drop in experiments
- [ ] Integrate basic cardiovascular modelling equations as priors in CDEs
- [ ] Go through causal ML to clarify link between current work and standard double ML methods
- [ ] Write up first section of Diss!

**Deadline 14th May**:

- [ ] Create better CV model to get data from with hidden states to infer from observed
- [ ] Speak to Ari re physiological equations
- [ ] Expand to no-baseline multi-treatment set-up
- [ ] Learn baseline ODE + Path control function + Tx control function
- [ ] Write section & present

**Deadline 21st May**:

- [ ] Expand into ICU dataset... transformer as probabilistic dynamics model?
- [ ] link transformer w previous Temporal CI
- [ ] put everything together to
- [ ] Write up section & present

**Deadline 28th May**:

- [ ] Write write write + repeat experiments if need

**Things to think about before submission**

- CV splits
- multi-seed runs

**Improve code running**

- [x] place code on github & have strict version control
- [x] complete full run on test set as well to make sure that fully saves
- [x] make sure model checkpoints will reload the best for test/ further comparison
- [x] set up HPC cluster access!!
- [x] Switch from pytorch lightning to lightning
- [x] set up multi GPU processing -> turns out GPU is slower than CPU as my model is TINY...
- [ ] adjust logs so they don't occur so often?
- [ ] add batch norm? not in hyland paper
- [ ] check pytorch profiler to see how can optimise training speed & efficiency

- [ ] Find best ways of clustering pts through time (read paper)

## Questions

- **Time forwards or time backwords for the RNN?**
In predictive tasks where you're trying to forecast future states based on past observations, encoding with time forwards is generally more appropriate. However, if you're trying to infer the current state based on a full history that includes more relevant recent events, then time backwards might be beneficial.

- **KL loss for the encoder or L1/L2 loss?**

- **How to best integrate the treatment function in the ODE?**

currently the treatment_fun takes in the tx and the time, and outputs a function for each timestep... but as an MLP...
not completely sure if the tx is being applied at every input, and therefore if it could be adjusted if for example the dose changes as well over time (in a continuous setting)...

- **Does it make more sense to apply the output function as a pointwise transformation from the latent to the observed... or to take the whole latent as an input and give the whole observed as output??**

 Pointwise Transformation
Advantages:
Simplicity: Applying a transformation pointwise is straightforward and often easier to implement and debug. Each latent state is transformed independently, which can simplify the learning process.
Independence: This approach assumes that each observed point is conditionally independent of others given its corresponding latent state. This can be advantageous if the time points are relatively independent or if the dependencies are already captured within the latent states.
Disadvantages:
Lack of Temporal Context: Pointwise transformations do not consider temporal dependencies between consecutive points in a sequence. If the observed data have strong temporal correlations, this method might not capture them effectively unless the latent trajectory already encodes all necessary temporal information.
Potential Overfitting: Since each point is treated independently, the model might overfit to noise in individual points without considering smoother trends over time.
2. Whole Trajectory Transformation
Advantages:
Temporal Dynamics: Processing the entire latent trajectory at once allows the model to take into account the temporal dynamics across the sequence. This is particularly useful in time series data where the future state is dependent on past states.
Complex Mappings: This method can model more complex mappings from latent to observed states, potentially integrating learned features across time points to make more informed predictions.
Disadvantages:
Complexity: Implementing a function that takes the whole trajectory as input and outputs another trajectory can be more complex. It might require more sophisticated network architectures like RNNs, CNNs, or Transformers that can handle sequential data.
Increased Computational Resources: Handling entire sequences might increase the computational cost, especially for long sequences, due to the complexity of the models needed to process them.
