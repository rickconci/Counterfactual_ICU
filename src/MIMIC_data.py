import os
import pickle
import torch
from torch.utils.data import Dataset, DataLoader, Subset
import lightning as L
import numpy as np
import pandas as pd
from torch.nn.utils.rnn import pad_sequence

class MIMICDataset(Dataset):
    def __init__(self, data_root, icu_stays_path, split='train', train_ratio=0.7, val_ratio=0.15, random_state=42):
        self.data_root = data_root
        self.p_tensor_dir = os.path.join(data_root, 'p_tensors')
        self.m_tensor_dir = os.path.join(data_root, 'med_tensors')
        self.ic_tensor_dir = os.path.join(data_root, 'initial_conditions')
        
        # Load metadata
        ic_metadata_path = os.path.join(self.ic_tensor_dir, 'initial_conditions_metadata.pkl')
        if not os.path.exists(ic_metadata_path):
            raise FileNotFoundError(f"Initial conditions metadata not found at {ic_metadata_path}. Please run preprocessing first.")
        with open(ic_metadata_path, 'rb') as f:
            ic_metadata = pickle.load(f)
        self.all_initial_conditions = ic_metadata['all_initial_conditions']

        # Each trajectory from the metadata (which corresponds to a t0) is a sample
        self.samples = []
        for stay_id, trajectories in self.all_initial_conditions.items():
            if not trajectories: continue
            
            # The preprocessing script that creates p_in/p_out only runs for trajectories that have a "next" one.
            # So, we only consider trajectories up to the second to last one for each patient stay.
            # The 'trajectory_num' in the metadata corresponds to the t0 event.
            sorted_trajs = sorted(trajectories, key=lambda x: x['trajectory_num'])
            if len(sorted_trajs) > 1:
                for i in range(len(sorted_trajs) - 1):
                    traj_info = sorted_trajs[i]
                    self.samples.append({
                        'stay_id': stay_id,
                        'traj_num': traj_info['trajectory_num'],
                    })

        # Shuffle all samples and split into train, val, test sets
        np.random.seed(random_state)
        shuffled_samples = list(self.samples)
        np.random.shuffle(shuffled_samples)

        n_samples = len(shuffled_samples)
        train_end = int(train_ratio * n_samples)
        val_end = train_end + int(val_ratio * n_samples)

        if split == 'train':
            self.samples = shuffled_samples[:train_end]
        elif split == 'val':
            self.samples = shuffled_samples[train_end:val_end]
        elif split == 'test':
            self.samples = shuffled_samples[val_end:]
        else:
            raise ValueError(f"Invalid split '{split}'. Must be one of 'train', 'val', or 'test'.")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_info = self.samples[idx]
        stay_id = sample_info['stay_id']
        traj_num = sample_info['traj_num']

        p_in_path = os.path.join(self.p_tensor_dir, f"p_tensor_in_{stay_id}_traj_{traj_num:03d}.pt")
        p_in_values, p_in_mask, p_in_abs_time, p_in_rel_time, p_in_len = torch.load(p_in_path)
        
        ic_path = os.path.join(self.ic_tensor_dir, f"ic_tensor_{stay_id}_traj_{traj_num:03d}.pt")
        ic_tensor, ic_mask_tensor = torch.load(ic_path)
        
        m_path = os.path.join(self.m_tensor_dir, f"med_tensor_{stay_id}_traj_{traj_num:03d}.pt")
        med_tensor_in, _, _, _, _ = torch.load(m_path)
        
        p_out_path = os.path.join(self.p_tensor_dir, f"p_tensor_out_{stay_id}_traj_{traj_num:03d}.pt")
        p_out_values, _, p_out_abs_time, p_out_rel_time, _ = torch.load(p_out_path)
        
        # The output trajectory Y_fact is just p_out_values
        Y_fact = p_out_values
        # The time for Y should be relative to t0
        t_Y = p_out_rel_time - p_out_rel_time[0] if len(p_out_rel_time) > 0 else p_out_rel_time

        # The full trajectory for evaluation/plotting is p_in and p_out concatenated
        full_fact_traj = torch.cat([p_in_values, p_out_values], dim=0)
        t_full = torch.cat([p_in_rel_time, p_out_rel_time])

        return {
            "X": p_in_values,
            "Y_fact": Y_fact,
            "T": torch.tensor(1.0), # Assuming treatment for all MIMIC data for now
            "Y_cf": torch.zeros_like(Y_fact), # No counterfactuals in real data
            "p": torch.tensor(0.0), # Propensity score not applicable here
            "init_state": ic_tensor,
            "t_X": p_in_rel_time,
            "t_Y": t_Y,
            "t_full": t_full,
            "full_fact_traj": full_fact_traj,
            "full_CF_traj": torch.zeros_like(full_fact_traj), # No counterfactuals
            "meds_in": med_tensor_in
        }

class MIMICDataModule(L.LightningDataModule):
    def __init__(self, data_root, icu_stays_path, batch_size=32, num_workers=1):
        super().__init__()
        self.data_root = data_root
        self.icu_stays_path = icu_stays_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.encoder_input_dim = None
        self.expert_latent_dim = None

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_dataset = MIMICDataset(self.data_root, self.icu_stays_path, split='train')
            self.val_dataset = MIMICDataset(self.data_root, self.icu_stays_path, split='val')

            if len(self.train_dataset) > 0:
                sample0 = self.train_dataset[0]
                self.encoder_input_dim = sample0['X'].shape[-1]
                self.expert_latent_dim = sample0['full_fact_traj'].shape[-1]
        
        if stage == 'test' or stage is None:
            self.test_dataset = MIMICDataset(self.data_root, self.icu_stays_path, split='test')
            if len(self.test_dataset) > 0 and self.encoder_input_dim is None:
                sample0 = self.test_dataset[0]
                self.encoder_input_dim = sample0['X'].shape[-1]
                self.expert_latent_dim = sample0['full_fact_traj'].shape[-1]

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, collate_fn=self.collate_fn)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=self.collate_fn)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=self.collate_fn)

    @staticmethod
    def collate_fn(batch):
        # Separate keys that need padding from those that don't
        pad_keys = ["X", "Y_fact", "Y_cf", "t_X", "t_Y", "t_full", "full_fact_traj", "full_CF_traj", "meds_in"]
        no_pad_keys = ["T", "p", "init_state"]

        collated = {}
        
        # Pad sequences
        for key in pad_keys:
            sequences = [item[key] for item in batch]
            collated[key] = pad_sequence(sequences, batch_first=True, padding_value=0.0)
            
        # Stack tensors that don't need padding
        for key in no_pad_keys:
            collated[key] = torch.stack([item[key] for item in batch])

        # The model expects a tuple/list of tensors, not a dict
        # The order must match the unpacking in the training_step
        ordered_keys = [
            'X', 'Y_fact', 'T', 'Y_cf', 'p', 'init_state', 
            't_X', 't_Y', 't_full', 'full_fact_traj', 'full_CF_traj'
        ]
        
        # We also have 'meds_in' which is not in the original model input
        # We will append it at the end and modify the model to accept it.
        
        return_list = [collated[k] for k in ordered_keys]
        return_list.append(collated['meds_in'])
        return tuple(return_list) 