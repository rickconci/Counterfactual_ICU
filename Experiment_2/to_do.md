
Improve code running
[] place code on github & have strict version control
[] complete full run on test set as well to make sure that fully saves
[] make sure model checkpoints will reload the best for test/ further comparison
[] set up HPC cluster access!!

[] Switch from pytorch lightning to lightning
[] set up multi GPU processing
[] check pytorch profiler to see how can optimise training speed & efficiency

-- Assuming we have baseline!

Run BO CDE on HPC
[] adjust to include L1, L2 and KL penalty on RNN
[] adjust to have BO loop for hyperparam optim
[] adjust to have plots occur not so often
[] adjust plots so they include whether treated or not + other..?

Improve the quality of code/models
[] add batch norm?
[] add learning rate scheduler if not already there

Convert from CDE to SCDE
[] adjust hyland SDE code to CDE version
[] understand aspects of SDE code in case missed something important

Integrate basic cardiovascular modelling equations as priors in CDEs

--- Assuming we DON'T HAVE baselines

--- Applying to REAL ICU Dataset

[] Find best ways of clustering pts through time (read paper)

TO DO

DOING

DONE

##  time forwards or time backwords for the RNN?
In predictive tasks where you're trying to forecast future states based on past observations, encoding with time forwards is generally more appropriate. However, if you're trying to infer the current state based on a full history that includes more relevant recent events, then time backwards might be beneficial.

## KL loss for the encoder or L1/L2 loss?

## How to best integrate the treatment function in the ODE?

currently the treatment_fun takes in the tx and the time, and outputs a function for each timestep... but as an MLP...
not completely sure if the tx is being applied at every input, and therefore if it could be adjusted if for example the dose changes as well over time (in a continuous setting)...

## Does it make more sense to apply the output function as a pointwise transformation from the latent to the observed... or to take the whole latent as an input and give the whole observed as output??

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
