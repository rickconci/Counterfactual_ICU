import os
import pickle
import torch
from torch.utils.data import Dataset, DataLoader, Subset
import lightning as L
import numpy as np
import pandas as pd
from torch.nn.utils.rnn import pad_sequence


class MIMICDataset(Dataset):
    def __init__(self, data_root, icu_stays_path, split='train', train_ratio=0.7, val_ratio=0.15,
                 random_state=42, max_samples = None):
        self.data_root = data_root
        self.m_tensor_dir = os.path.join(data_root, 'med_tensors_output')
        self.ic_tensor_dir = os.path.join(data_root, 'physio_tensors_output/ic_tensors')
        self.target_tensor_dir = os.path.join(data_root, 'physio_tensors_output/prediction_targets')

        # New: Context tensors directories
        self.context_tensors_dir = os.path.join(data_root, 'context_tensors_output')
        self.meds_context_dir = os.path.join(self.context_tensors_dir, 'meds_context')
        self.p_tensor_dir = os.path.join(self.context_tensors_dir, 'physio_context')
        self.baseline_tensor_dir = os.path.join(self.context_tensors_dir, 'baseline_tensors')
        self.max_samples = max_samples

        physio_metadata_path = os.path.join(data_root, 'physio_tensors_output/physio_tensors_metadata.pkl')

        # Load physio metadata from provided path
        with open(physio_metadata_path, 'rb') as f:
            physio_metadata = pickle.load(f)
        self.ic_tensors_metadata = physio_metadata['ic_tensors']
        self.pred_targets_metadata = physio_metadata['prediction_targets']

        # Load med tensors metadata to get action_cluster_id mapping
        med_metadata_path = os.path.join(self.m_tensor_dir, 'med_tensors_metadata.pkl')
        if not os.path.exists(med_metadata_path):
            raise FileNotFoundError(f"Med tensors metadata not found at {med_metadata_path}")
        with open(med_metadata_path, 'rb') as f:
            med_metadata = pickle.load(f)
        self.med_trajectories = med_metadata['trajectories']

        # Each trajectory from the IC metadata is a sample
        # First, collect all trajectories
        all_trajectories = []
        for traj_key, ic_traj_info in self.ic_tensors_metadata.items():
            hadm_id = ic_traj_info['hadm_id']
            action_cluster_id = ic_traj_info['action_cluster_id']

            # Check if a baseline tensor exists for this hadm_id
            baseline_path = os.path.join(self.baseline_tensor_dir, f"baseline_{hadm_id}.pt")

            #TODO add this back in when we have baselines
            #if not os.path.exists(baseline_path):
                #continue

            # Check if corresponding med trajectory exists
            if traj_key not in self.med_trajectories:
                continue

            # Add this trajectory
            all_trajectories.append({
                'hadm_id': hadm_id,
                'action_cluster_id': action_cluster_id,
                'traj_key': traj_key
            })

        # Split by hadm_id to prevent data leakage, but balance trajectory counts
        # Step 1: Count trajectories per patient
        patient_trajectory_counts = {}
        for traj in all_trajectories:
            hadm_id = traj['hadm_id']
            patient_trajectory_counts[hadm_id] = patient_trajectory_counts.get(hadm_id, 0) + 1

        # Step 2: Randomly shuffle patients (no sorting bias)
        patients_with_counts = [(hadm_id, count) for hadm_id, count in patient_trajectory_counts.items()]
        np.random.seed(random_state)
        np.random.shuffle(patients_with_counts)  # Completely random order

        # Step 3: Greedy assignment to balance trajectory counts
        total_trajectories = len(all_trajectories)
        target_train = int(train_ratio * total_trajectories)
        target_val = int(val_ratio * total_trajectories)
        target_test = total_trajectories - target_train - target_val

        # Initialize split counters
        train_patients = []
        val_patients = []
        test_patients = []
        train_count = 0
        val_count = 0
        test_count = 0

        # Greedy assignment: assign each patient to the split that needs trajectories most
        for hadm_id, traj_count in patients_with_counts:
            # Calculate how far each split is from its target
            train_deficit = target_train - train_count
            val_deficit = target_val - val_count
            test_deficit = target_test - test_count

            # Assign to split with largest deficit (that can still accept this patient)
            if train_deficit >= val_deficit and train_deficit >= test_deficit and train_deficit > 0:
                train_patients.append(hadm_id)
                train_count += traj_count
            elif val_deficit >= test_deficit and val_deficit > 0:
                val_patients.append(hadm_id)
                val_count += traj_count
            else:
                test_patients.append(hadm_id)
                test_count += traj_count

        # Select patients for this split
        if split == 'train':
            split_hadm_ids = set(train_patients)
            split_traj_count = train_count
        elif split == 'val':
            split_hadm_ids = set(val_patients)
            split_traj_count = val_count
        elif split == 'test':
            split_hadm_ids = set(test_patients)
            split_traj_count = test_count
        else:
            raise ValueError(f"Invalid split '{split}'. Must be one of 'train', 'val', or 'test'.")

        # Filter trajectories to only include those from the selected hadm_ids
        self.samples = [traj for traj in all_trajectories if traj['hadm_id'] in split_hadm_ids]
        if self.max_samples is not None:
            self.samples = self.samples[:int(max_samples)]
            print(f"[DEBUG] Limited to {len(self.samples)} samples for testing")

        print(f"Split '{split}': {len(split_hadm_ids)} patients, {len(self.samples)} trajectories")
        print(f"Target trajectory split: {target_train}/{target_val}/{target_test}")
        print(f"Actual trajectory split: {train_count}/{val_count}/{test_count}")
        print(
            f"Trajectory percentages: {train_count / total_trajectories * 100:.1f}%/{val_count / total_trajectories * 100:.1f}%/{test_count / total_trajectories * 100:.1f}%")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_info = self.samples[idx]
        hadm_id = sample_info['hadm_id']
        action_cluster_id = sample_info['action_cluster_id']
        traj_key = sample_info['traj_key']

        # Load physio context (replaces p_tensors)
        physio_context_dir = os.path.join(self.context_tensors_dir, 'physio_context')
        physio_context_path = os.path.join(physio_context_dir, f"physio_context_{traj_key}.pt")
        if os.path.exists(physio_context_path):
            p_in_values, p_in_mask, _, p_in_rel_time, _ = torch.load(physio_context_path)
        else:
            # TODO REMOVE THIS FOR PROD
            # Create dummy physio context for development
            print(f"Warning: Using dummy physio context for {traj_key}")
            n_context_intervals = 6
            n_physio_measurements = 5  # ABP MEAN, NBP MEAN, CVP, HR, RESP
            p_in_values = torch.zeros(n_context_intervals, n_physio_measurements)
            p_in_mask = torch.zeros(n_context_intervals, n_physio_measurements)
            p_in_rel_time = torch.arange(-6, 0, dtype=torch.float32) * (10 / 60)  # -1.0, -0.83, etc.

        # Load IC tensor (new format)
        ic_path = os.path.join(self.ic_tensor_dir, f"ic_tensor_{traj_key}.pt")
        if not os.path.exists(ic_path):
            raise FileNotFoundError(f"IC tensor not found: {ic_path}")
        ic_tensor, ic_mask_tensor = torch.load(ic_path)

        # Load prediction targets (new format)
        p_out_path = os.path.join(self.target_tensor_dir, f"pred_targets_{traj_key}.pt")
        if not os.path.exists(p_out_path):
            raise FileNotFoundError(f"Prediction targets not found: {p_out_path}")
        p_out_values, p_out_mask, p_out_rel_time, _, _ = torch.load(p_out_path)

        # Load static features

        # TODO ADD BACK IN WHEN WE HAVE BASELINES
        #baseline_path = os.path.join(self.baseline_tensor_dir, f"baseline_{hadm_id}.pt")
        #if not os.path.exists(baseline_path):
          #  raise FileNotFoundError(f"Baseline tensor not found: {baseline_path}")

        # TODO placeholder for baseline tensors
        static_feats = torch.zeros(10)

        # Load main trajectory med tensor (t₀ forward)
        med_traj_path = os.path.join(self.m_tensor_dir, f"med_tensor_{traj_key}.pt")
        if not os.path.exists(med_traj_path):
            raise FileNotFoundError(f"Med trajectory tensor not found: {med_traj_path}")
        med_traj_values, med_traj_mask, med_traj_time_sec, med_traj_time_hr, _ = torch.load(med_traj_path)

        # Load context med tensor (hour before t₀)
        context_meds_path = os.path.join(self.meds_context_dir, f"meds_context_{traj_key}.pt")

        if os.path.exists(context_meds_path):
            context_meds_values, context_meds_mask, context_meds_time_sec, context_meds_time_hr, _ = torch.load(
                context_meds_path)
        else:
            # TODO REMOVE FOR PROD
            # Create dummy context for development
            print(f"Warning: Using dummy meds context for {traj_key}")
            n_context_intervals = 6
            n_medications = med_traj_values.shape[1] if hasattr(med_traj_values, 'shape') and len(
                med_traj_values.shape) > 1 else 20
            context_meds_values = torch.zeros(n_context_intervals, n_medications)
            context_meds_mask = torch.zeros(n_context_intervals, n_medications)
            context_meds_time_hr = torch.arange(-6, 0, dtype=torch.float32) * (10 / 60)  # -1.0, -0.83, etc.
            context_meds_time_sec = context_meds_time_hr * 3600  # Convert hours to seconds

        # The time for Y should be relative to t0
        t_Y = p_out_rel_time

        # The full trajectory for evaluation/plotting is p_in and p_out concatenated
        p_out_padded = torch.cat([
            p_out_values,
            torch.zeros(p_out_values.shape[0], 3)
        ], dim=1)
        full_fact_traj = torch.cat([p_in_values, p_out_padded], dim=0)
        t_full = torch.cat([p_in_rel_time, p_out_rel_time])

        return {
            "X": p_in_values,
            "X_mask": p_in_mask,
            "Y_fact": p_out_values,
            "Y_fact_mask": p_out_mask,
            "T": torch.tensor(1.0),
            "Y_cf": torch.zeros_like(p_out_values),
            "p": torch.tensor(0.0),
            "init_state": ic_tensor,
            "ic_mask": ic_mask_tensor,
            "static": static_feats,
            "t_X": p_in_rel_time,
            "t_Y": t_Y,
            "t_full": t_full,
            "full_fact_traj": full_fact_traj,
            "full_CF_traj": torch.zeros_like(full_fact_traj),

            # NEW: Medication tensors only (physio context replaces existing p_tensors)
            "med_trajectory_values": med_traj_values,  # Main trajectory meds (t₀ forward)
            "med_trajectory_mask": med_traj_mask,
            "med_trajectory_time": med_traj_time_sec,
            "meds_context_values": context_meds_values,  # Context meds (hour before t₀)
            "meds_context_mask": context_meds_mask,
            "meds_context_time": context_meds_time_hr,
        }


class MIMICDataModule(L.LightningDataModule):
    def __init__(self, data_root, icu_stays_path, batch_size=32, num_workers=1, random_state=42, max_samples = None):
        super().__init__()
        self.data_root = data_root
        self.icu_stays_path = icu_stays_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.encoder_input_dim = None
        self.expert_latent_dim = None
        self.random_state = random_state
        self.static_input_dim = None
        self.max_samples = max_samples

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            self.train_dataset = MIMICDataset(self.data_root, self.icu_stays_path,
                                              split='train', random_state=self.random_state, max_samples=self.max_samples)
            self.val_dataset = MIMICDataset(self.data_root, self.icu_stays_path, split='val',
                                            random_state=self.random_state, max_samples=self.max_samples)

            # Get static dim from metadata
            baseline_metadata_path = os.path.join(self.data_root, 'baseline_tensors', 'baseline_metadata.pkl')
            if os.path.exists(baseline_metadata_path):
                with open(baseline_metadata_path, 'rb') as f:
                    baseline_meta = pickle.load(f)
                    self.static_input_dim = baseline_meta['feature_dim']

            if len(self.train_dataset) > 0:
                sample0 = self.train_dataset[0]
                self.encoder_input_dim = sample0['X'].shape[-1]
                self.expert_latent_dim = sample0['full_fact_traj'].shape[-1]

        if stage == 'test' or stage is None:
            self.test_dataset = MIMICDataset(self.data_root, self.icu_stays_path,
                                             split='test', random_state=self.random_state, max_samples=self.max_samples)
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
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers,
                          collate_fn=self.collate_fn)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers,
                          collate_fn=self.collate_fn)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers,
                          collate_fn=self.collate_fn)

    @staticmethod
    def collate_fn(batch):
        # TODO fix this (eventually)
        MAX_LEN = 120

        # Updated pad keys to include new medication tensors
        pad_keys = ["X", "X_mask", "Y_fact","Y_fact_mask", "Y_cf", "t_X", "t_Y", "t_full", "full_fact_traj", "full_CF_traj",
                    "med_trajectory_values", "med_trajectory_mask", "med_trajectory_time"]

        # Context tensors have fixed size (6 intervals), so they don't need padding
        stack_keys = ["meds_context_values", "meds_context_mask", "meds_context_time"]

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

            elif key in ["X", "X_mask", "t_X"]:  # Input sequences
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

            elif key in ["med_trajectory_values", "med_trajectory_mask",
                         "med_trajectory_time"]:  # Med trajectory tensors
                # These need special handling as they're from t₀ forward with 10-second intervals
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

            elif key in ["Y_fact", "Y_fact_mask", "Y_cf", "t_full", "full_fact_traj", "full_CF_traj"]:  # Target sequences
                # Pad these to MAX_LEN as well
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

        # Stack context tensors (fixed size, no padding needed)
        for key in stack_keys:
            collated[key] = torch.stack([item[key] for item in batch])

        # Stack tensors that don't need padding
        for key in no_pad_keys:
            collated[key] = torch.stack([item[key] for item in batch])

        # Return in the order expected by your model
        ordered_keys = [
            'X', 'X_mask', 'Y_fact', "Y_fact_mask",'T', 'Y_cf', 'p', 'init_state', 'ic_mask',
            't_X', 't_Y', 't_full', 'full_fact_traj', 'full_CF_traj',
            'valid_lengths'
        ]

        return_list = [collated[k] for k in ordered_keys]

        # Add medication and context tensors
        return_list.extend([
            collated['med_trajectory_values'],
            collated['med_trajectory_mask'],
            collated['med_trajectory_time'],
            collated['meds_context_values'],
            collated['meds_context_mask'],
            collated['meds_context_time'],
            collated['static']
        ])

        return tuple(return_list)