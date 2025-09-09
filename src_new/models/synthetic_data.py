"""
IMPORTANT: THIS FILE IS NOT USED IN THE CURRENT IMPLEMENTATION.
PLEASE IGNORE AND USE SYNTHETIC DIR INSTEAD.
"""

import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, Subset
import lightning as L


def create_load_save_data(dataset_params: dict, data_path: str) -> dict:
    """
    Create or load synthetic cardiovascular dataset.
    
    Args:
        dataset_params: Dictionary containing dataset generation parameters
        data_path: Path to save/load the dataset
        
    Returns:
        Dictionary containing the generated/loaded dataset
    """
    if not os.path.exists(data_path):
        os.makedirs(data_path)
        print(f"Created directory: {data_path}")

    include_all = "T" if dataset_params["include_all_inputs"] else "F"
    non_confounded_effect_str = "T" if dataset_params["non_confounded_effect"] else "F"
    normalize = "T" if dataset_params["normalize"] else "F"
    post_treatment_dims_str = "".join(map(str, dataset_params["post_treatment_dims"]))
    pre_treatment_dims_str = "".join(map(str, dataset_params["pre_treatment_dims"]))

    # Create a descriptive file name
    filename = f"allIn{include_all}_{dataset_params['confounder_type']}_RCE{non_confounded_effect_str}_N{dataset_params['N']}_G{dataset_params['gamma']}_Dstd{dataset_params['noise_std']}_Tstd{dataset_params['sigma_tx']}_Pre{pre_treatment_dims_str}_Post{post_treatment_dims_str}_Norm{normalize}_Rtpr{dataset_params['r_tpr_mod']}.pt"
    final_data_path = os.path.join(data_path, filename)

    if os.path.exists(final_data_path):
        print("Loading existing dataset.")
        data = torch.load(final_data_path)
    else:
        print("Creating and saving a new dataset.")
        # For now, create dummy data - this should be replaced with actual SDE generation        data = _create_dummy_data(dataset_params)
        
        # Save the dataset
        torch.save(data, final_data_path)

    return data


def _create_dummy_data(dataset_params: dict) -> dict:
    """
    Create dummy data for testing purposes.
    
    Args:
        dataset_params: Dictionary containing dataset generation parameters
        
    Returns:
        Dictionary containing dummy dataset
    """
    N = dataset_params.get("N", 1000)
    t_span = dataset_params.get("t_span", 60)
    t_treatment = dataset_params.get("t_treatment", 45)
    t_cutoff = dataset_params.get("t_cutoff", 40)
    
    # Create dummy data with appropriate shapes
    X = torch.randn(N, t_span, 4)  # Time series data
    T = torch.randint(0, 2, (N,))  # Treatment assignment
    Y_fact = torch.randn(N, t_span - t_treatment, 2)  # Factual outcomes
    Y_cf = torch.randn(N, t_span - t_treatment, 2)  # Counterfactual outcomes
    p = torch.rand(N)  # Propensity scores
    init_state = torch.randn(N, 4)  # Initial states
    t_X = torch.linspace(0, t_span, t_span)
    t_Y = torch.linspace(t_treatment, t_span, t_span - t_treatment)
    t_full = torch.linspace(0, t_span, t_span)
    full_fact_traj = torch.randn(N, t_span, 14)  # Full factual trajectory
    full_CF_traj = torch.randn(N, t_span, 14)  # Full counterfactual trajectory
    
    return {
        "X": X,
        "T": T,
        "Y_fact": Y_fact,
        "Y_cf": Y_cf,
        "p": p,
        "init_state": init_state,
        "t_X": t_X,
        "t_Y": t_Y,
        "t_full": t_full,
        "full_fact_traj": full_fact_traj,
        "full_CF_traj": full_CF_traj,
    }


class CVDataset_loaded(Dataset):
    """Dataset class for loaded cardiovascular data."""
    
    def __init__(self, data: dict):
        """
        Initialize dataset with loaded data.
        
        Args:
            data: Dictionary containing the dataset components
        """
        # Unpack the data
        self.X = data["X"]
        self.T = data["T"]
        self.Y_fact = data["Y_fact"]
        self.Y_cf = data["Y_cf"]
        self.p = data["p"]
        self.init_state = data["init_state"]
        self.t_X = data["t_X"]
        self.t_Y = data["t_Y"]
        self.t_full = data["t_full"]
        self.full_fact_traj = data["full_fact_traj"]
        self.full_CF_traj = data["full_CF_traj"]

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return self.X.shape[0]

    def __getitem__(self, idx: int) -> tuple:
        """
        Get a single sample from the dataset.
        
        Args:
            idx: Index of the sample to retrieve
            
        Returns:
            Tuple containing all components of the sample
        """
        return (
            self.X[idx],
            self.Y_fact[idx],
            self.T[idx],
            self.Y_cf[idx],
            self.p[idx],
            self.init_state[idx],
            self.t_X[idx],
            self.t_Y[idx],
            self.t_full[idx],
            self.full_fact_traj[idx],
            self.full_CF_traj[idx],
        )


class CVDataModule_IID(L.LightningDataModule):
    """PyTorch Lightning data module for in-distribution cardiovascular data."""
    
    def __init__(
        self, 
        train_val_data: dict, 
        batch_size: int = 32, 
        num_workers: int = 1,
        debug: bool = False
    ):
        """
        Initialize the IID data module.
        
        Args:
            train_val_data: Dictionary containing training/validation data
            batch_size: Batch size for data loaders
            num_workers: Number of worker processes for data loading
            debug: Whether to enable debug mode
        """
        super().__init__()
        self.train_val_data = train_val_data
        self.batch_size = batch_size
        self.dataset = None
        self.num_workers = num_workers
        self.debug = debug
        self.encoder_input_dim = train_val_data["X"].shape[-1]
        self.expert_latent_dim = train_val_data["full_fact_traj"].shape[-1]

    def setup(self, stage: str = None) -> None:
        """
        Set up the dataset splits.
        
        Args:
            stage: Lightning stage ('fit', 'test', etc.)
        """
        # Load the dataset
        self.dataset_train_val = CVDataset_loaded(self.train_val_data)
        dataset_size = len(self.dataset_train_val)

        train_ratio = 0.7
        val_ratio = 0.15
        test_ratio = 0.15
        
        # Calculate the number of examples for each set
        train_size = int(train_ratio * dataset_size)
        val_size = int(val_ratio * dataset_size)

        # Indices for training, validation, and testing
        train_idx = np.arange(0, train_size)
        val_idx = np.arange(train_size, train_size + val_size)
        test_idx = np.arange(train_size + val_size, dataset_size)

        # Create subsets
        self.train = Subset(self.dataset_train_val, train_idx)
        self.val = Subset(self.dataset_train_val, val_idx)
        self.in_dist_test = Subset(self.dataset_train_val, test_idx)

    def train_dataloader(self) -> DataLoader:
        """Return training data loader."""
        return DataLoader(
            self.train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            persistent_workers=True,
            drop_last=True,
            pin_memory=True,
        )

    def val_dataloader(self) -> DataLoader:
        """Return validation data loader."""
        return DataLoader(
            self.val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            persistent_workers=True,
            drop_last=False,
            pin_memory=True,
        )

    def test_dataloader(self) -> DataLoader:
        """Return test data loader."""
        return DataLoader(
            self.in_dist_test,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True,
        )


class CVDataModule_OOD(L.LightningDataModule):
    """PyTorch Lightning data module for out-of-distribution cardiovascular data."""
    
    def __init__(
        self, 
        OOD_test_data: dict, 
        batch_size: int = 32, 
        num_workers: int = 1,
        debug: bool = False
    ):
        """
        Initialize the OOD data module.
        
        Args:
            OOD_test_data: Dictionary containing out-of-distribution test data
            batch_size: Batch size for data loaders
            num_workers: Number of worker processes for data loading
            debug: Whether to enable debug mode
        """
        super().__init__()
        self.OOD_test_data = OOD_test_data
        self.batch_size = batch_size
        self.dataset = None
        self.num_workers = num_workers
        self.debug = debug
        self.encoder_input_dim = OOD_test_data["X"].shape[-1]
        self.expert_latent_dim = OOD_test_data["full_fact_traj"].shape[-1]

    def setup(self, stage: str = None) -> None:
        """
        Set up the OOD test dataset.
        
        Args:
            stage: Lightning stage ('fit', 'test', etc.)
        """
        self.OOD_test_dataset = CVDataset_loaded(self.OOD_test_data)

    def test_dataloader(self) -> DataLoader:
        """Return OOD test data loader."""
        return DataLoader(
            self.OOD_test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True,
        )
