import os
import platform
import pickle

import lightning as L
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import typing as t


class MIMICDataset(Dataset):
    def __init__(
        self,
        data_root,
        icu_stays_path,
        split="train",
        train_ratio=0.7,
        val_ratio=0.15,
        random_state=42,
        max_samples=None,
        filter_flat_trajectories=False,  # New parameter
        flatness_std_threshold=2.0,  # New parameter
        flatness_range_threshold=5.0,
        min_valid_points = 5
    ):
        self.data_root = data_root
        self.m_tensor_dir = os.path.join(data_root, "med_tensors_output")
        self.ic_tensor_dir = os.path.join(data_root, "physio_tensors_output/ic_tensors")
        self.target_tensor_dir = os.path.join(
            data_root, "physio_tensors_output/prediction_targets"
        )
        self.filter_flat_trajectories = filter_flat_trajectories
        self.flatness_std_threshold = flatness_std_threshold
        self.flatness_range_threshold = flatness_range_threshold
        self.min_valid_points = min_valid_points

        # New: Context tensors directories
        self.context_tensors_dir = os.path.join(data_root, "context_tensors_output")
        # Baseline tensors live under baseline_tensors_output/baseline_tensors
        self.baseline_tensor_dir = os.path.join(
            self.data_root, "baseline_tensors_output", "baseline_tensors"
        )
        self.max_samples = max_samples

        physio_metadata_path = os.path.join(
            data_root, "physio_tensors_output/physio_tensors_metadata.pkl"
        )

        # Load physio metadata from provided path
        with open(physio_metadata_path, "rb") as f:
            physio_metadata = pickle.load(f)
        self.ic_tensors_metadata = physio_metadata["ic_tensors"]
        self.pred_targets_metadata = physio_metadata["prediction_targets"]

        # Load med tensors metadata to get action_cluster_id mapping
        med_metadata_path = os.path.join(self.m_tensor_dir, "med_tensors_metadata.pkl")
        if not os.path.exists(med_metadata_path):
            raise FileNotFoundError(
                f"Med tensors metadata not found at {med_metadata_path}"
            )
        with open(med_metadata_path, "rb") as f:
            med_metadata = pickle.load(f)
        self.med_trajectories = med_metadata["trajectories"]
        # Store med feature names for normalization
        self.med_feature_names = med_metadata.get("med_feature_names", [
            "rate_hr_weight_norm",
            "pre_on_hours_log1p", 
            "cumulative_since_start_hours_log1p",
            "post_stop_effect",
            "trigger_flag",
        ])

        # Each trajectory from the IC metadata is a sample
        # First, collect all trajectories
        # Load ADMISSIONS.csv to get HADM_ID -> SUBJECT_ID mapping
        admissions_path = os.path.join(
            os.path.dirname(data_root), "input_data", "ADMISSIONS.csv"
        )
        if not os.path.exists(admissions_path):
            raise FileNotFoundError(f"ADMISSIONS.csv not found at {admissions_path}")

        admissions_df = pd.read_csv(admissions_path)
        hadm_to_subject = dict(
            zip(admissions_df["HADM_ID"], admissions_df["SUBJECT_ID"])
        )

        # Each trajectory from the IC metadata is a sample
        all_trajectories = []
        for traj_key, ic_traj_info in self.ic_tensors_metadata.items():
            hadm_id = ic_traj_info["hadm_id"]
            action_cluster_id = ic_traj_info["action_cluster_id"]

            # Get subject_id for this hadm_id
            if hadm_id not in hadm_to_subject:
                continue
            subject_id = hadm_to_subject[hadm_id]

            # Check if corresponding med trajectory exists
            if traj_key not in self.med_trajectories:
                continue

            # Add this trajectory (now with subject_id)
            all_trajectories.append(
                {
                    "hadm_id": hadm_id,
                    "subject_id": subject_id,  # Add subject_id
                    "action_cluster_id": action_cluster_id,
                    "traj_key": traj_key,
                }
            )
        # Filter out flat trajectories if enabled
        if self.filter_flat_trajectories:
            print(f"Before filtering: {len(all_trajectories)} trajectories")

            filtered_trajectories = []

            for traj in all_trajectories:
                traj_key = traj["traj_key"]

                # Load prediction targets to check flatness
                p_out_path = os.path.join(self.target_tensor_dir, f"pred_targets_{traj_key}.pt")
                if not os.path.exists(p_out_path):
                    continue

                try:
                    p_out_values, p_out_mask, _, _, _ = torch.load(p_out_path)

                    # Use the class method to check flatness
                    if not self.is_trajectory_flat(p_out_values, p_out_mask):
                        filtered_trajectories.append(traj)

                except Exception as e:
                    print(f"Error loading trajectory {traj_key}: {e}")
                    continue

            print(f"After filtering: {len(filtered_trajectories)} trajectories")
            print(f"Filtered out {len(all_trajectories)-len(filtered_trajectories)} flat trajectories")
            all_trajectories = filtered_trajectories
        else:
            print("Flat trajectory filtering is disabled")

        # Split by subject_id instead of hadm_id to prevent data leakage
        subject_trajectory_counts = {}
        for traj in all_trajectories:
            subject_id = traj["subject_id"]
            subject_trajectory_counts[subject_id] = (
                subject_trajectory_counts.get(subject_id, 0) + 1
            )

        # Randomly shuffle subjects (not hadm_ids)
        subjects_with_counts = [
            (subject_id, count)
            for subject_id, count in subject_trajectory_counts.items()
        ]
        np.random.seed(random_state)
        np.random.shuffle(subjects_with_counts)

        total_trajectories = len(all_trajectories)
        target_train = int(train_ratio * total_trajectories)
        target_val = int(val_ratio * total_trajectories)
        target_test = total_trajectories - target_train - target_val

        # Greedy assignment by subject
        train_subjects = []
        val_subjects = []
        test_subjects = []
        train_count = val_count = test_count = 0

        for subject_id, traj_count in subjects_with_counts:
            train_deficit = target_train - train_count
            val_deficit = target_val - val_count
            test_deficit = target_test - test_count

            if (
                train_deficit >= val_deficit
                and train_deficit >= test_deficit
                and train_deficit > 0
            ):
                train_subjects.append(subject_id)
                train_count += traj_count
            elif val_deficit >= test_deficit and val_deficit > 0:
                val_subjects.append(subject_id)
                val_count += traj_count
            else:
                test_subjects.append(subject_id)
                test_count += traj_count

        # Select subjects for this split
        if split == "train":
            split_subject_ids = set(train_subjects)
        elif split == "val":
            split_subject_ids = set(val_subjects)
        elif split == "test":
            split_subject_ids = set(test_subjects)

        # Filter trajectories by subject_id (not hadm_id)
        self.samples = [
            traj for traj in all_trajectories if traj["subject_id"] in split_subject_ids
        ]

        if self.max_samples is not None:
            self.samples = self.samples[: int(max_samples)]
            print(f"[DEBUG] Limited to {len(self.samples)} samples for testing")

        print(f"Split '{split}': {len(self.samples)} trajectories")
        print(f"Target trajectory split: {target_train}/{target_val}/{target_test}")
        print(f"Actual trajectory split: {train_count}/{val_count}/{test_count}")
        print(
            f"Trajectory percentages: {train_count / total_trajectories * 100:.1f}%/{val_count / total_trajectories * 100:.1f}%/{test_count / total_trajectories * 100:.1f}%"
        )

        # Compute normalization stats and load normalized tensors into memory
        #self.med_norm_stats, self.med_tensors_cache = self._compute_and_load_normalized_tensors()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_info = self.samples[idx]
        hadm_id = sample_info["hadm_id"]
        action_cluster_id = sample_info["action_cluster_id"]
        traj_key = sample_info["traj_key"]

        # Load Raindrop-ready context (required)
        rd_context_dir = os.path.join(self.context_tensors_dir, "raindrop_context")
        rd_context_path = os.path.join(rd_context_dir, f"rd_context_{traj_key}.pt")

        if not os.path.exists(rd_context_path):
            raise FileNotFoundError(
                f"Raindrop context not found for {traj_key}: {rd_context_path}"
            )
        rd_src, rd_times, rd_length = torch.load(rd_context_path)
        rd_src = rd_src.to(torch.float32)
        rd_times = rd_times.to(torch.float32)
        rd_length = torch.tensor(int(rd_length), dtype=torch.long)
        # Build X/X_mask/t_X from raindrop context
        d_inp = rd_src.shape[-1] // 2
        p_in_values = rd_src[:, :d_inp]
        rd_missing = rd_src[:, d_inp:]
        p_in_mask = 1.0 - torch.clamp(rd_missing, 0.0, 1.0)
        p_in_rel_time = rd_times

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
        baseline_path = os.path.join(self.baseline_tensor_dir, f"baseline_{hadm_id}.pt")
        if not os.path.exists(baseline_path):
            raise FileNotFoundError(f"Baseline tensor not found: {baseline_path}")
        # static_feats = torch.zeros(10)
        static_feats = torch.load(baseline_path)


        # Fallback: load from disk if not in cache
        med_traj_path = os.path.join(self.m_tensor_dir, f"med_tensor_{traj_key}.pt")
        if not os.path.exists(med_traj_path):
            raise FileNotFoundError(f"Med trajectory tensor not found: {med_traj_path}")
        loaded = torch.load(med_traj_path)
        # Backward/forward compatible unpacking (optionally includes med_context)
        if len(loaded) == 6:
            (
                med_traj_values,
                med_traj_mask,
                med_traj_time_sec,
                med_traj_time_hr,
                _n_intervals,
                med_tensors,
            ) = loaded
        else:
            med_traj_values, med_traj_mask, med_traj_time_sec, med_traj_time_hr, _ = loaded
            # Create a placeholder med_context (zeros) if not present
            med_tensors = torch.zeros(
                med_traj_values.shape[0], med_traj_values.shape[1] * 2
            )


        # Legacy meds context removed

        # The time for Y should be relative to t0
        t_Y = p_out_rel_time

        # For plotting/eval, use factual targets only to keep feature dims consistent (2)
        full_fact_traj = p_out_values
        t_full = p_out_rel_time



        return {
            # Raindrop-ready context
            "rd_src": rd_src,  # [T_ctx, 2*d_inp]
            "rd_times": rd_times,  # [T_ctx]
            "rd_length": rd_length,  # scalar length
            # Static and ICs at t0
            "static": static_feats,  # [d_static]
            "init_state": ic_tensor,  # [d_ic]
            "ic_mask": ic_mask_tensor,  # [d_ic]
            # Forward factual targets
            "Y": p_out_values,  # [T_fwd, 2]
            "Y_mask": p_out_mask,  # [T_fwd, 2]
            "t_Y": t_Y,  # [T_fwd]
            # Med trajectory forward
            "med_values": med_traj_values,  # [T_fwd, M]
            "med_mask": med_traj_mask,  # [T_fwd, M]
            "med_time": med_traj_time_sec,  # [T_fwd]
            # Precomputed med context per time [T_fwd, 2*M]
            "med_tensors": med_tensors,
            # IDs for traceability
            "hadm_id": torch.tensor(int(hadm_id), dtype=torch.long),
            "traj_id": torch.tensor(int(action_cluster_id), dtype=torch.long),
        }

    def is_trajectory_flat(self, p_out_values, p_out_mask):
        """
        Determine if a trajectory is too flat to be useful for training.
        Uses channel-specific thresholds:
        - Channel 0 (arterial): requires range >= 15
        - Channel 1 (venous): requires range >= 5
        """
        # Channel-specific range thresholds
        channel_range_thresholds = [15.0, 5.0]  # arterial, venous

        for channel in range(p_out_values.shape[1]):
            channel_values = p_out_values[:, channel]
            channel_mask = p_out_mask[:, channel]

            valid_values = channel_values[channel_mask > 0]

            if len(valid_values) < self.min_valid_points:
                continue

            channel_range = (valid_values.max() - valid_values.min()).item()

            # Get threshold for this channel (default to last threshold if more channels)
            threshold = channel_range_thresholds[min(channel, len(channel_range_thresholds) - 1)]

            # If any channel passes its threshold, trajectory is not flat
            if channel_range >= threshold:
                return False
        return True


class MIMICDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_root,
        icu_stays_path,
        filter_flat_trajectories=False,
        test_both_filtered_and_unfiltered=False,
        batch_size=32,
        num_workers=1,
        random_state=42,
        max_samples=None,
        use_raindrop_context: bool = True,
        expert_latent_dim: int | None = None,
    ):
        super().__init__()
        self.data_root = data_root
        self.icu_stays_path = icu_stays_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.encoder_input_dim = None
        self.random_state = random_state
        self.static_input_dim = None
        self.max_samples = max_samples
        self.use_raindrop_context = use_raindrop_context
        # Derived lengths from generation config (filled in setup from combined metadata)
        self.context_max_len: int | None = None
        self.forward_max_len: int | None = None
        self.expert_latent_dim = expert_latent_dim
        # Store nominal interval seconds if present in metadata (fallback to 10)
        self.interval_seconds: int = 10
        self.filter_flat_trajectories = filter_flat_trajectories
        self.test_both_filtered_and_unfiltered = test_both_filtered_and_unfiltered

    def _resolve_num_workers(self) -> int:
        """Choose efficient default workers: 6-8 on non-macOS, 1 on macOS."""
        if platform.system() == "Darwin":
            return 1
        cpu_count = os.cpu_count() or 8
        # Aim for 6-8 to avoid dataloader contention and CPU throttling
        return min(8, max(6, cpu_count // 2))

    def _loader_common_kwargs(self) -> dict:
        # Honor explicit num_workers if provided; otherwise, use heuristic
        eff_workers = (
            int(self.num_workers)
            if self.num_workers is not None
            else self._resolve_num_workers()
        )
        # Pin memory if using CUDA; persistent workers when >0 workers
        use_pin = torch.cuda.is_available() and platform.system() != "Darwin"
        return {
            "num_workers": eff_workers,
            "pin_memory": use_pin,
            "persistent_workers": eff_workers > 0,
            "prefetch_factor": 4 if eff_workers > 0 else None,
        }

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_dataset = MIMICDataset(
                self.data_root,
                self.icu_stays_path,
                split="train",
                random_state=self.random_state,
                max_samples=self.max_samples,
                filter_flat_trajectories=self.filter_flat_trajectories
            )
            self.val_dataset = MIMICDataset(
                self.data_root,
                self.icu_stays_path,
                split="val",
                random_state=self.random_state,
                max_samples=self.max_samples,
                filter_flat_trajectories=self.filter_flat_trajectories
            )

            # Get static dim from metadata
            baseline_metadata_path = os.path.join(
                self.data_root,
                "baseline_tensors_output",
                "baseline_tensors",
                "baseline_metadata.pkl",
            )
            if os.path.exists(baseline_metadata_path):
                with open(baseline_metadata_path, "rb") as f:
                    baseline_meta = pickle.load(f)
                    self.static_input_dim = baseline_meta["feature_dim"]

            # Load generation parameters from combined metadata if available
            combined_meta_path = os.path.join(
                self.data_root, "combined_tensors_metadata.pkl"
            )
            if os.path.exists(combined_meta_path):
                try:
                    with open(combined_meta_path, "rb") as f:
                        _meta = pickle.load(f)
                    params = _meta.get("params", {})
                    ctx_minutes = int(params.get("context_duration_minutes", 60))
                    ctx_bin = int(params.get("context_interval_minutes", 10))
                    fwd_minutes = int(params.get("trajectory_duration_minutes", 20))
                    interval_seconds = int(params.get("interval_seconds", 10))
                    # Derived lengths
                    self.context_max_len = int(ctx_minutes // ctx_bin)
                    self.forward_max_len = int((fwd_minutes * 60) // interval_seconds)
                    self.interval_seconds = interval_seconds
                except Exception:
                    self.context_max_len = None
                    self.forward_max_len = None
                    self.interval_seconds = 10

            if len(self.train_dataset) > 0:
                sample0 = self.train_dataset[0]
                # If using raindrop context, set d_inp from rd_src (half the last dim)
                if self.use_raindrop_context and "rd_src" in sample0:
                    self.encoder_input_dim = int(sample0["rd_src"].shape[-1] // 2)
                else:
                    self.encoder_input_dim = int(sample0["rd_src"].shape[-1] // 2)

        if stage == "test" or stage is None:
            if self.test_both_filtered_and_unfiltered:
                self.test_dataset_all = MIMICDataset(
                    self.data_root,
                    self.icu_stays_path,
                    split="test",
                    random_state=self.random_state,
                    max_samples=self.max_samples,
                    filter_flat_trajectories=False  # All data
                )
                self.test_dataset_filtered = MIMICDataset(
                    self.data_root,
                    self.icu_stays_path,
                    split="test",
                    random_state=self.random_state,
                    max_samples=self.max_samples,
                    filter_flat_trajectories=True  # Filtered data only
                )
                if len(self.test_dataset_all) > 0 and self.encoder_input_dim is None:
                    sample0 = self.test_dataset[0]
                    if self.use_raindrop_context and "rd_src" in sample0:
                        self.encoder_input_dim = int(sample0["rd_src"].shape[-1] // 2)
                    else:
                        self.encoder_input_dim = sample0["X"].shape[-1]
            else:
                # Single test dataset using the original filter setting
                self.test_dataset = MIMICDataset(
                    self.data_root,
                    self.icu_stays_path,
                    split="test",
                    random_state=self.random_state,
                    max_samples=self.max_samples,
                    filter_flat_trajectories=self.filter_flat_trajectories)
                if len(self.test_dataset) > 0 and self.encoder_input_dim is None:
                    sample0 = self.test_dataset[0]
                    if self.use_raindrop_context and "rd_src" in sample0:
                        self.encoder_input_dim = int(sample0["rd_src"].shape[-1] // 2)
                    else:
                        self.encoder_input_dim = sample0["X"].shape[-1]
                    # Also set static dim if not set
            if self.static_input_dim is None:
                baseline_metadata_path = os.path.join(
                    self.data_root,
                    "baseline_tensors_output",
                    "baseline_tensors",
                    "baseline_metadata.pkl",
                )
                if os.path.exists(baseline_metadata_path):
                    with open(baseline_metadata_path, "rb") as f:
                        baseline_meta = pickle.load(f)
                        self.static_input_dim = baseline_meta["feature_dim"]

    def train_dataloader(self):
        kwargs = self._loader_common_kwargs()
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            collate_fn=self.collate_fn,
            **{k: v for k, v in kwargs.items() if v is not None},
        )

    def val_dataloader(self):
        kwargs = self._loader_common_kwargs()
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            collate_fn=self.collate_fn,
            **{k: v for k, v in kwargs.items() if v is not None},
        )

    def test_dataloader(self):
        kwargs = self._loader_common_kwargs()
        if self.test_both_filtered_and_unfiltered:
            # Return list of dataloaders - Lightning will test on both
            return [
                DataLoader(
                    self.test_dataset_all,
                    batch_size=self.batch_size,
                    collate_fn=self.collate_fn,
                    **{k: v for k, v in kwargs.items() if v is not None},
                ),
                DataLoader(
                    self.test_dataset_filtered,
                    batch_size=self.batch_size,
                    collate_fn=self.collate_fn,
                    **{k: v for k, v in kwargs.items() if v is not None},
                )
            ]
        else:
            # Single test dataloader
            return DataLoader(
                self.test_dataset,
                batch_size=self.batch_size,
                collate_fn=self.collate_fn,
                **{k: v for k, v in kwargs.items() if v is not None},
            )

    def collate_fn(self, batch):
        # Dynamic lengths
        forward_len_cfg = (
            int(self.forward_max_len) if self.forward_max_len is not None else None
        )

        # Group pad keys by sequence family (simplified API)
        pad_forward_keys = [
            "Y",
            "Y_mask",
            "t_Y",
            "med_values",
            "med_mask",
            "med_time",
            "med_tensors",
        ]
        # Context keys derived from raindrop context
        pad_context_keys = []  # no legacy X/X_mask/t_X

        # Raindrop-specific padding keys (built from context tensors)
        rd_pad_keys = ["rd_src", "rd_times"]

        # No fixed-size legacy context tensors to stack
        stack_keys = []

        no_pad_keys = ["init_state", "ic_mask", "static", "rd_length", "hadm_id", "traj_id"]

        collated = {}

        # Compute valid forward lengths from t_Y before padding
        valid_lengths = []
        for item in batch:
            t_Y = item["t_Y"]
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

        collated["valid_lengths"] = torch.tensor(valid_lengths, dtype=torch.long)

        # No legacy context sequences to pad
        rd_lengths = [int(item["rd_length"]) for item in batch]
        rd_max_len = max(rd_lengths) if len(rd_lengths) > 0 else 0

        # Pad forward sequences to batch max or configured forward length
        y_max_len = max(valid_lengths) if len(valid_lengths) > 0 else 0
        if forward_len_cfg is not None:
            y_max_len = max(y_max_len, forward_len_cfg)
        for key in pad_forward_keys:
            sequences = [item[key] for item in batch]
            padded_sequences = []
            for i, seq in enumerate(sequences):
                if key == "t_Y":
                    # Ensure monotonic padding for times
                    valid_len = valid_lengths[i]
                    seq_valid = seq[:valid_len]
                    if valid_len < y_max_len:
                        last_valid_time = seq_valid[-1]
                        # Use per-sample dt if available; fallback to module interval_seconds
                        if valid_len >= 2:
                            dt = (seq_valid[-1] - seq_valid[-2]).item()
                            if not torch.isfinite(torch.tensor(dt)) or dt <= 0:
                                dt = float(self.interval_seconds)
                        else:
                            dt = float(self.interval_seconds)
                        padding_times = (
                            torch.arange(1, y_max_len - valid_len + 1, dtype=seq.dtype)
                            * dt
                            + last_valid_time
                        )
                        padded_seq = torch.cat([seq_valid, padding_times], dim=0)
                    else:
                        padded_seq = seq_valid[:y_max_len]
                elif seq.dim() == 1:
                    # 1D sequences (times)
                    if seq.shape[0] < y_max_len:
                        padding = torch.zeros(
                            (y_max_len - seq.shape[0],), dtype=seq.dtype
                        )
                        padded_seq = torch.cat([seq, padding], dim=0)
                    else:
                        padded_seq = seq[:y_max_len]
                else:
                    # 2D+ sequences (values, masks, src, etc.)
                    if seq.shape[0] < y_max_len:
                        padding = torch.zeros(
                            (y_max_len - seq.shape[0],) + tuple(seq.shape[1:]),
                            dtype=seq.dtype,
                        )
                        padded_seq = torch.cat([seq, padding], dim=0)
                    else:
                        padded_seq = seq[:y_max_len]
                padded_sequences.append(padded_seq)
            collated[key] = torch.stack(padded_sequences)

        # Pad Raindrop-specific sequences to the context length, not MAX_LEN
        for key in rd_pad_keys:
            sequences = [item[key] for item in batch]
            padded_sequences = []
            for seq in sequences:
                if seq.dim() == 1:
                    # times: [T]
                    if seq.shape[0] < rd_max_len:
                        padding = torch.zeros(
                            (rd_max_len - seq.shape[0],), dtype=seq.dtype
                        )
                        padded_seq = torch.cat([seq, padding], dim=0)
                    else:
                        padded_seq = seq[:rd_max_len]
                else:
                    # src: [T, 2*d_inp]
                    if seq.shape[0] < rd_max_len:
                        padding = torch.zeros(
                            (rd_max_len - seq.shape[0],) + tuple(seq.shape[1:]),
                            dtype=seq.dtype,
                        )
                        padded_seq = torch.cat([seq, padding], dim=0)
                    else:
                        padded_seq = seq[:rd_max_len]
                padded_sequences.append(padded_seq)
            collated[key] = torch.stack(padded_sequences)

        # No legacy fixed-size context tensors to stack

        # Stack tensors that don't need padding
        for key in no_pad_keys:
            collated[key] = torch.stack([item[key] for item in batch])

        # Return in simplified order expected by the model
        return (
            collated["rd_src"],  # [B, T_ctx, 2*d_inp]
            collated["rd_times"],  # [B, T_ctx]
            collated["rd_length"],  # [B]
            collated["static"],  # [B, d_static]
            collated["init_state"],  # [B, d_ic]
            collated["ic_mask"],  # [B, d_ic]
            collated["Y"],  # [B, T_fwd, 2]
            collated["Y_mask"],  # [B, T_fwd, 2]
            collated["t_Y"],  # [B, T_fwd]
            collated["med_values"],  # [B, T_fwd, M]
            collated["med_mask"],  # [B, T_fwd, M]
            collated["med_time"],  # [B, T_fwd]
            collated["med_tensors"],  # [B, T_fwd, 2*M]
            collated["hadm_id"],  # [B]
            collated["traj_id"],  # [B]
        )
