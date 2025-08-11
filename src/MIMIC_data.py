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
        self.target_tensor_dir = os.path.join(data_root, 'prediction_targets')
        self.baseline_tensor_dir = os.path.join(data_root, 'baseline_tensors') # New
        
        # Load metadata
        ic_metadata_path = os.path.join(self.ic_tensor_dir, 'ic_metadata.pkl')
        if not os.path.exists(ic_metadata_path):
            raise FileNotFoundError(f"Initial conditions metadata not found at {ic_metadata_path}. Please run preprocessing first.")
        with open(ic_metadata_path, 'rb') as f:
            ic_metadata = pickle.load(f)
        self.all_initial_conditions = ic_metadata['all_initial_conditions']

        # Each trajectory from the metadata (which corresponds to a t0) is a sample
        self.samples = []
        for hadm_id, trajectories in self.all_initial_conditions.items():
            if not trajectories:
                 continue
            
            # Check if a baseline tensor exists for this hadm_id
            baseline_path = os.path.join(self.baseline_tensor_dir, f"baseline_{hadm_id}.pt")
            if not os.path.exists(baseline_path):
                continue

            # The preprocessing script that creates p_in/p_out only runs for trajectories that have a "next" one.
            # So, we only consider trajectories up to the second to last one for each patient stay.
            # The 'trajectory_num' in the metadata corresponds to the t0 event.
            sorted_trajs = sorted(trajectories, key=lambda x: x['trajectory_num'])
            if len(sorted_trajs) > 1:
                for i in range(len(sorted_trajs) - 1):
                    traj_info = sorted_trajs[i]
                    self.samples.append({
                        'hadm_id': hadm_id,
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
        hadm_id = sample_info['hadm_id']
        traj_num = sample_info['traj_num']

        p_in_path = os.path.join(self.p_tensor_dir, f"p_tensor_in_{hadm_id}_traj_{traj_num:03d}.pt")
        p_in_values, p_in_mask, p_in_abs_time, p_in_rel_time, p_in_len = torch.load(p_in_path)
        
        ic_path = os.path.join(self.ic_tensor_dir, f"ic_tensor_{hadm_id}_traj_{traj_num:03d}.pt")
        ic_tensor, ic_mask_tensor = torch.load(ic_path)
        
        m_path = os.path.join(self.m_tensor_dir, f"med_tensor_{hadm_id}_traj_{traj_num:03d}.pt")
        med_tensor_in, _, _, _, _ = torch.load(m_path)
        
        p_out_path = os.path.join(self.target_tensor_dir, f"prediction_target_{hadm_id}_traj_{traj_num:03d}.pt")
        p_out_values, _, p_out_rel_time, _ = torch.load(p_out_path)

        # The time for Y should be relative to t0: commented out because MIMIC-iii pred tensors start uniformly at t_0 +5s
        #t_Y = p_out_rel_time - p_out_rel_time[0] if len(p_out_rel_time) > 0 else p_out_rel_time
        t_Y = p_out_rel_time


        # --- New: Load static features from pre-made tensor ---
        baseline_path = os.path.join(self.baseline_tensor_dir, f"baseline_{hadm_id}.pt")
        static_feats = torch.load(baseline_path)
        # --- End new ---

        # The full trajectory for evaluation/plotting is p_in and p_out concatenated
        # Pad p_out_values with zeros for the missing 3 features
        print(f"P in shape: {p_in_values.shape}")
        print(f"P out shape: {p_out_values.shape}")
        #changed dim here for MIMIC-III (higher diversity of itemids in MIMIC-III leads to more physio params
        p_out_padded = torch.cat([
            p_out_values,
            torch.zeros(p_out_values.shape[0], 11)  # or torch.full((p_out_values.shape[0], 3), float('nan'))
        ], dim=1)
        full_fact_traj = torch.cat([p_in_values, p_out_padded], dim=0)
        t_full = torch.cat([p_in_rel_time, p_out_rel_time])


        return {
            "X": p_in_values,
            "X_mask": p_in_mask,
            "Y_fact": p_out_values,
            "T": torch.tensor(1.0), # Assuming treatment for all MIMIC data for now
            "Y_cf": torch.zeros_like(p_out_values), # No counterfactuals in real data
            "p": torch.tensor(0.0), # Propensity score not applicable here
            "init_state": ic_tensor,
            "ic_mask": ic_mask_tensor,
            "static": static_feats, # New
            "t_X": p_in_rel_time,
            "t_Y": t_Y,
            "t_full": t_full,
            "full_fact_traj": full_fact_traj,
            "full_CF_traj": torch.zeros_like(full_fact_traj), # No counterfactuals
            "meds_in": med_tensor_in
        }

class MIMICDataModule(L.LightningDataModule):
    def __init__(self, data_root, icu_stays_path, batch_size=32, num_workers=1, random_state = 42):
        super().__init__()
        self.data_root = data_root
        self.icu_stays_path = icu_stays_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.encoder_input_dim = None
        self.expert_latent_dim = None
        self.random_state = random_state
        self.static_input_dim = None # New

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_dataset = MIMICDataset(self.data_root, self.icu_stays_path, split='train', random_state=self.random_state)
            self.val_dataset = MIMICDataset(self.data_root, self.icu_stays_path, split='val', random_state=self.random_state)

            # --- New: Get static dim from metadata ---
            baseline_metadata_path = os.path.join(self.data_root, 'baseline_tensors', 'baseline_metadata.pkl')
            if os.path.exists(baseline_metadata_path):
                with open(baseline_metadata_path, 'rb') as f:
                    baseline_meta = pickle.load(f)
                    self.static_input_dim = baseline_meta['feature_dim']
            # --- End New ---

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
                # Also set static dim if not set
                if self.static_input_dim is None:
                    baseline_metadata_path = os.path.join(self.data_root, 'baseline_tensors', 'baseline_metadata.pkl')
                    if os.path.exists(baseline_metadata_path):
                        with open(baseline_metadata_path, 'rb') as f:
                            baseline_meta = pickle.load(f)
                            self.static_input_dim = baseline_meta['feature_dim']

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers, collate_fn=self.collate_fn)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=self.collate_fn)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers, collate_fn=self.collate_fn)

    @staticmethod
    def collate_fn(batch):
        # Define the expected max_len (should match the model's max_len parameter)
        # TODO this needs to be fixed
        MAX_LEN = 215  # or get this from somewhere consistent

        pad_keys = ["X", "X_mask", "Y_fact", "Y_cf", "t_X", "t_Y", "t_full", "full_fact_traj", "full_CF_traj",
                    "meds_in"]
        no_pad_keys = ["T", "p", "init_state", "ic_mask", "static"]

        collated = {}

        # Compute valid lengths from t_Y before padding
        valid_lengths = []
        for item in batch:
            t_Y = item['t_Y']
            # Find where time stops increasing
            if len(t_Y) > 1:
                time_diffs = t_Y[1:] - t_Y[:-1]
                non_increasing = torch.where(time_diffs <= 0)[0]

                if len(non_increasing) > 0:
                    valid_len = non_increasing[0].item() + 1
                else:
                    valid_len = len(t_Y)
            else:
                valid_len = 1
            # we fixed the predictions to all be of same length when using MIMIC-III, so if that does not hold we raise an Exception
            if valid_len != 299:
                raise ValueError("Unexpected prediction length")
            valid_lengths.append(valid_len)

        collated['valid_lengths'] = torch.tensor(valid_lengths, dtype=torch.long)

        # Pad sequences to fixed MAX_LEN
        for key in pad_keys:
            sequences = [item[key] for item in batch]

            if key == "t_Y":  # Special handling for time to ensure monotonicity
                padded_sequences = []
                for i, seq in enumerate(sequences):
                    valid_len = valid_lengths[i]
                    seq_valid = seq[:valid_len]  # Only the valid part

                    if valid_len < MAX_LEN:
                        # Create monotonically increasing padding
                        last_valid_time = seq_valid[-1]
                        # Add 1.0 time unit for each padded step to ensure strict monotonicity
                        time_increment = 1.0
                        padding_times = torch.arange(1, MAX_LEN - valid_len + 1) * time_increment + last_valid_time
                        padded_seq = torch.cat([seq_valid, padding_times], dim=0)
                    else:
                        # Truncate if longer (shouldn't happen with your data)
                        padded_seq = seq_valid[:MAX_LEN]

                    padded_sequences.append(padded_seq)
                collated[key] = torch.stack(padded_sequences)

            elif key in ["X", "X_mask", "t_X", "meds_in"]:  # Input sequences
                # Pad or truncate to MAX_LEN
                padded_sequences = []
                for seq in sequences:
                    if seq.shape[0] < MAX_LEN:
                        # Pad with zeros
                        padding = torch.zeros((MAX_LEN - seq.shape[0],) + seq.shape[1:])
                        padded_seq = torch.cat([seq, padding], dim=0)
                    else:
                        # Truncate if longer
                        padded_seq = seq[:MAX_LEN]
                    padded_sequences.append(padded_seq)
                collated[key] = torch.stack(padded_sequences)

            elif key in ["Y_fact", "Y_cf", "t_full", "full_fact_traj", "full_CF_traj"]:  # FIX: Add explicit handling
                # Pad these to MAX_LEN as well
                padded_sequences = []
                for seq in sequences:
                    if seq.shape[0] < MAX_LEN:
                        # Pad with zeros (or last value)
                        padding = torch.zeros((MAX_LEN - seq.shape[0],) + seq.shape[1:])
                        # Alternative: pad with last value
                        # padding = seq[-1:].repeat(MAX_LEN - seq.shape[0], *([1] * (seq.ndim - 1)))
                        padded_seq = torch.cat([seq, padding], dim=0)
                    else:
                        # Truncate if longer
                        padded_seq = seq[:MAX_LEN]
                    padded_sequences.append(padded_seq)
                collated[key] = torch.stack(padded_sequences)

            else:
                # For any remaining sequences, use regular padding to MAX_LEN
                padded_sequences = []
                for seq in sequences:
                    if seq.shape[0] < MAX_LEN:
                        padding = torch.zeros((MAX_LEN - seq.shape[0],) + seq.shape[1:])
                        padded_seq = torch.cat([seq, padding], dim=0)
                    else:
                        padded_seq = seq[:MAX_LEN]
                    padded_sequences.append(padded_seq)
                collated[key] = torch.stack(padded_sequences)

        # Stack tensors that don't need padding
        for key in no_pad_keys:
            collated[key] = torch.stack([item[key] for item in batch])

        ordered_keys = [
            'X', 'X_mask', 'Y_fact', 'T', 'Y_cf', 'p', 'init_state', 'ic_mask',
            't_X', 't_Y', 't_full', 'full_fact_traj', 'full_CF_traj',
            'valid_lengths'
        ]

        return_list = [collated[k] for k in ordered_keys]
        return_list.append(collated['meds_in'])
        return_list.append(collated['static'])
        # Debug print
        # print(f"[DEBUG] Collated Y_fact shape: {collated['Y_fact'].shape}, t_Y shape: {collated['t_Y'].shape}")

        return tuple(return_list)