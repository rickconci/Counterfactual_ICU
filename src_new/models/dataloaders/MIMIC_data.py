
import os
import platform
import pickle
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import lightning as L
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import typing as t
from collections import defaultdict


class MIMICDataset(Dataset):
    def __init__(
        self,
        data_root,
        split="train",
        split_mode="random",  # 'random', 'in_distribution', 'ood'
        train_ratio=0.7,
        val_ratio=0.15,
        random_state=42,
        max_samples=None,
        ood_holdout_ratio=0.1, # Percentage of combos to hold out for OOD test
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

        # Load consolidated metadata
        consolidated_metadata_path = os.path.join(
            data_root, "consolidated_trajectory_metadata.pkl"
        )
        if not os.path.exists(consolidated_metadata_path):
            raise FileNotFoundError(
                f"Consolidated metadata not found at {consolidated_metadata_path}"
            )
        with open(consolidated_metadata_path, "rb") as f:
            self.metadata = pickle.load(f)
        
        self.trajectories_metadata = self.metadata["trajectories"]
        self.med_combos_info = self.metadata.get("med_combos", {})
        self.id_to_combo = self.med_combos_info.get("id_to_combo", {})
        self.combo_to_id = self.med_combos_info.get("combo_to_id", {})
        self.combo_df = pd.DataFrame(self.med_combos_info.get("combo_df", []))

        self.med_feature_names = med_metadata.get("med_feature_names", [
            "rate_hr_weight_norm",
            "pre_on_hours_log1p", 
            "cumulative_since_start_hours_log1p",
            "post_stop_effect",
            "trigger_flag",
        ])

        # Post-process id_to_combo to create a more useful representation
        self.formatted_id_to_combo = {}
        # This loop correctly processes the `id_to_combo` mapping.
        # It iterates through the medication names within each combo tuple.
        # Since rate information is not available in the consolidated metadata, 
        # it creates a simple string representation and a list of medication details.
        for combo_id, combo_tuple in self.id_to_combo.items():
            # Ensure combo_tuple is iterable (handles single-med combos)
            if not isinstance(combo_tuple, (list, tuple)):
                combo_tuple = (combo_tuple,)
            
            # Sort for consistent representation
            sorted_combo = sorted(combo_tuple)

            combo_str = ", ".join(map(str, sorted_combo))
            med_details = [{"med_name": name} for name in sorted_combo]
            
            self.formatted_id_to_combo[int(combo_id)] = {
                "str": combo_str,
                "details": med_details
            }

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

        # Each trajectory with complete data is a sample
        all_trajectories = []
        for traj_key, traj_info in self.trajectories_metadata.items():
            if not traj_info.get("has_complete_data", True): # Assume complete if key missing
                continue
                
            hadm_id = traj_info["hadm_id"]
            action_cluster_id = traj_info["action_cluster_id"]
            
            # Get subject_id for this hadm_id
            if hadm_id not in hadm_to_subject:
                continue
            subject_id = hadm_to_subject[hadm_id]

            all_trajectories.append(
                {
                    "hadm_id": hadm_id,
                    "subject_id": subject_id,
                    "action_cluster_id": traj_info["action_cluster_id"],
                    "traj_key": traj_key,
                    "med_combo_id": traj_info.get("med_combo_id", -1)
                }
            )

        # Filter out flat trajectories if enabled
        if self.filter_flat_trajectories:

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

            all_trajectories = filtered_trajectories
        else:
            #print("Flat trajectory filtering is disabled")
            pass

        # Split by subject_id instead of hadm_id to prevent data leakage
        subject_trajectory_counts = {}
        for traj in all_trajectories:
            subject_id = traj["subject_id"]
            subject_trajectory_counts[subject_id] = (
                subject_trajectory_counts.get(subject_id, 0) + 1
            )


        

        # --- Data Splitting Logic ---
        np.random.seed(random_state)
        
        if split_mode == 'random':
            self.samples = self._random_subject_split(all_trajectories, split, train_ratio, val_ratio, random_state)
        elif split_mode == 'in_distribution_meds_combo':
            self.samples = self._in_distribution_split(all_trajectories, split, train_ratio, val_ratio, random_state)
        elif split_mode == 'ood_meds_combo':
            self.samples = self._ood_split(all_trajectories, split, train_ratio, val_ratio, ood_holdout_ratio, random_state)
        elif split_mode == 'variance_test_only':
            self.samples = self._variance_test_only_split(all_trajectories, split, train_ratio, val_ratio, random_state)
        else:
            raise ValueError(f"Unknown split mode: {split_mode}")


        if self.max_samples is not None:
            self.samples = self.samples[: int(max_samples)]
            print(f"[DEBUG] Limited to {len(self.samples)} samples for testing")

        print(f"Split '{split}' ({split_mode}): {len(self.samples)} trajectories from {len(set(s['subject_id'] for s in self.samples))} subjects.")

        # Compute normalization stats and load normalized tensors into memory
        #self.med_norm_stats, self.med_tensors_cache = self._compute_and_load_normalized_tensors()

    @property
    def id_to_combo_map(self) -> t.Dict[int, t.Dict[str, t.Any]]:
        """Provides a formatted mapping from med_combo_id to medication details."""
        return self.formatted_id_to_combo

    def _random_subject_split(self, trajectories, split, train_ratio, val_ratio, random_state):
        """Original random split by subject ID."""
        subject_to_trajs = defaultdict(list)
        for traj in trajectories:
            subject_to_trajs[traj['subject_id']].append(traj)

        subjects = list(subject_to_trajs.keys())
        np.random.shuffle(subjects)

        n_train = int(len(subjects) * train_ratio)
        n_val = int(len(subjects) * val_ratio)

        if split == 'train':
            split_subjects = subjects[:n_train]
        elif split == 'val':
            split_subjects = subjects[n_train : n_train + n_val]
        else: # test
            split_subjects = subjects[n_train + n_val:]
            
        split_samples = []
        for subj_id in split_subjects:
            split_samples.extend(subject_to_trajs[subj_id])
        return split_samples

    def _in_distribution_split(self, trajectories, split, train_ratio, val_ratio, random_state):
        """Ensures all med combos are represented in each split, subject-wise."""
        combo_to_subjects = defaultdict(set)
        for traj in trajectories:
            combo_to_subjects[traj['med_combo_id']].add(traj['subject_id'])
            
        train_subjects, val_subjects, test_subjects = set(), set(), set()

        for combo_id, subjects in combo_to_subjects.items():
            subject_list = sorted(list(subjects))
            np.random.shuffle(subject_list)
            
            n_train = int(len(subject_list) * train_ratio)
            n_val = int(len(subject_list) * val_ratio)
            
            train_subjects.update(subject_list[:n_train])
            val_subjects.update(subject_list[n_train : n_train + n_val])
            test_subjects.update(subject_list[n_train + n_val:])
        
        # Resolve overlaps - a subject can only be in one split
        val_subjects -= train_subjects
        test_subjects -= (train_subjects | val_subjects)
        
        if split == 'train':
            split_subject_ids = train_subjects
        elif split == 'val':
            split_subject_ids = val_subjects
        else: # test
            split_subject_ids = test_subjects
            
        return [t for t in trajectories if t['subject_id'] in split_subject_ids]

    def _ood_split(self, trajectories, split, train_ratio, val_ratio, ood_holdout_ratio, random_state):
        """Holds out a fraction of med combos entirely for the test set."""
        combo_df = self.med_combos_info.get("combo_df")
        if combo_df is None or combo_df.empty:
            print("Warning: Combo DF not found for OOD split. Falling back to random split.")
            return self._random_subject_split(trajectories, split, train_ratio, val_ratio, random_state)

        # Sort combos by frequency (most common first)
        sorted_combos = combo_df.sort_values('count', ascending=False)
        
        # Determine number of combos to hold out
        n_holdout = int(len(sorted_combos) * ood_holdout_ratio)
        if n_holdout == 0 and len(sorted_combos) > 0:
            n_holdout = 1 # Hold out at least one
        
        # Get IDs of combos to hold out (least common ones)
        ood_combo_tuples = list(sorted_combos.tail(n_holdout)['combo'])
        combo_to_id = self.med_combos_info.get("combo_to_id", {})
        ood_combo_ids = {combo_to_id[combo] for combo in ood_combo_tuples if combo in combo_to_id}
        
        print(f"OOD Split: Holding out {len(ood_combo_ids)} least common medication combos for testing.")

        test_trajectories = [t for t in trajectories if t['med_combo_id'] in ood_combo_ids]
        
        if split == 'test':
            return test_trajectories
        
        # The rest are for train/val
        remaining_trajs = [t for t in trajectories if t['med_combo_id'] not in ood_combo_ids]
        
        # Split remaining trajectories by subject for train/val
        # Adjust ratio because test set is already removed
        remaining_total = len(remaining_trajs)
        if remaining_total == 0:
            return [] # No data left for train/val
            
        new_train_ratio = train_ratio / (train_ratio + val_ratio)

        subject_to_trajs = defaultdict(list)
        for traj in remaining_trajs:
            subject_to_trajs[traj['subject_id']].append(traj)

        subjects = list(subject_to_trajs.keys())
        np.random.shuffle(subjects)
        
        n_train = int(len(subjects) * new_train_ratio)
        
        if split == 'train':
            split_subjects = subjects[:n_train]
        else: # val
            split_subjects = subjects[n_train:]
            
        split_samples = []
        for subj_id in split_subjects:
            split_samples.extend(subject_to_trajs[subj_id])
        return split_samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_info = self.samples[idx]
        hadm_id = sample_info["hadm_id"]
        action_cluster_id = sample_info["action_cluster_id"]
        traj_key = sample_info["traj_key"]
        med_combo_id = sample_info["med_combo_id"]

        # Load Raindrop-ready context (60-min physio)
        rd_context_dir = os.path.join(self.context_tensors_dir, "raindrop_context")
        rd_context_path = os.path.join(rd_context_dir, f"rd_context_{traj_key}.pt")

        if not os.path.exists(rd_context_path):
            raise FileNotFoundError(
                f"Raindrop context not found for {traj_key}: {rd_context_path}"
            )
        rd_src, rd_times, rd_length = torch.load(rd_context_path)

        # Load Raindrop-ready chartevents context (24h)
        traj_info = self.trajectories_metadata[traj_key]
        chartevents_context_path = traj_info.get("chartevents_context_path")

        if chartevents_context_path and os.path.exists(chartevents_context_path):
            ce_rd_src, ce_rd_times, ce_rd_length = torch.load(chartevents_context_path)
            if chartevents_context_path and os.path.exists(chartevents_context_path):
                ce_rd_src, ce_rd_times, ce_rd_length = torch.load(chartevents_context_path)
        elif chartevents_context_path:
            # Fallback path construction
            fallback_path = os.path.join(self.data_root, "chartevents_tensors_output", "chartevents_context",
                                         f"chartevents_context_{traj_key}.pt")
            if os.path.exists(fallback_path):
                ce_rd_src, ce_rd_times, ce_rd_length = torch.load(fallback_path)
        else:
            print(f"Warning: Chartevents context tensor not found for trajectory {traj_key}. Searched at '{chartevents_context_path}'. Using zero tensor as fallback.")
            # Default values assume 24h context, 1h interval, 100 features
            ce_rd_src = torch.zeros((24, 2 * 100), dtype=torch.float32)
            ce_rd_times = torch.zeros((24,), dtype=torch.float32)
            ce_rd_length = torch.tensor(0, dtype=torch.long)

        rd_src = rd_src.to(torch.float32)
        rd_times = rd_times.to(torch.float32)
        rd_length = torch.tensor(int(rd_length), dtype=torch.long)
        
        ce_rd_src = ce_rd_src.to(torch.float32)
        ce_rd_times = ce_rd_times.to(torch.float32)
        ce_rd_length = torch.tensor(int(ce_rd_length), dtype=torch.long)
        
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
            # Raindrop-ready context (60min)
            "rd_src": rd_src,
            "rd_times": rd_times,
            "rd_length": rd_length,
            # Raindrop-ready chartevents context (24h)
            "ce_rd_src": ce_rd_src,
            "ce_rd_times": ce_rd_times,
            "ce_rd_length": ce_rd_length,
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
            # Precomputed med context per time [T_fwd, 5*M] (rate, pre_on, cumulative, decay, trigger per med)
            "med_tensors": med_tensors,
            # IDs for traceability
            "hadm_id": torch.tensor(int(hadm_id), dtype=torch.long),
            "traj_id": torch.tensor(int(action_cluster_id), dtype=torch.long),
            "med_combo_id": torch.tensor(int(med_combo_id), dtype=torch.long),
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
        split_mode: str = "random",
        ood_holdout_ratio: float = 0.1,
        use_raindrop_context: bool = True,
        expert_latent_dim: int | None = None,
    ):
        super().__init__()
        self.data_root = data_root
        self.icu_stays_path = icu_stays_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.context_input_dim = None
        self.chartevents_input_dim = None
        self.n_medications = None
        self.random_state = random_state
        self.static_input_dim = None
        self.max_samples = max_samples
        self.split_mode = split_mode
        self.ood_holdout_ratio = ood_holdout_ratio
        self.use_raindrop_context = use_raindrop_context
        # Derived lengths from generation config (filled in setup from combined metadata)
        self.context_max_len: int | None = None
        self.forward_max_len: int | None = None
        self.expert_latent_dim = expert_latent_dim
        # Store nominal interval seconds if present in metadata (fallback to 10)
        self.interval_seconds: int = 10
        # For logging and plotting
        self.id_to_combo: dict = {}
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
                split="train",
                split_mode=self.split_mode,
                random_state=self.random_state,
                max_samples=self.max_samples,
                ood_holdout_ratio=self.ood_holdout_ratio,
                filter_flat_trajectories=self.filter_flat_trajectories
            )
            self.val_dataset = MIMICDataset(
                self.data_root,
                split="val",
                split_mode=self.split_mode,
                random_state=self.random_state,
                max_samples=self.max_samples,
                ood_holdout_ratio=self.ood_holdout_ratio,
                filter_flat_trajectories=self.filter_flat_trajectories
            )
            # Store combo map from one of the datasets (they are the same)
            self.id_to_combo = self.train_dataset.id_to_combo


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
                if self.use_raindrop_context and "rd_src" in sample0:
                    self.context_input_dim = int(sample0["rd_src"].shape[-1] // 2)
                    self.chartevents_input_dim = int(sample0["ce_rd_src"].shape[-1] // 2)
                
                if "med_values" in sample0 and sample0["med_values"] is not None:
                    self.n_medications = sample0["med_values"].shape[-1]

        if stage == "test" or stage is None:
            if self.test_both_filtered_and_unfiltered:
                self.test_dataset_all = MIMICDataset(
                    data_root=self.data_root,
                    split="test",
                    split_mode=self.split_mode,
                    random_state=self.random_state,
                    max_samples=self.max_samples,
                    filter_flat_trajectories=False
                )
                self.test_dataset_filtered = MIMICDataset(
                    self.data_root,
                    split="test",
                    split_mode=self.split_mode,
                    random_state=self.random_state,
                    max_samples=self.max_samples,
                    ood_holdout_ratio=self.ood_holdout_ratio,
                    filter_flat_trajectories=True  # Filtered data only
                )
                if len(self.test_dataset_all) > 0 and self.context_input_dim is None:
                    sample0 = self.test_dataset_all[0]
                    if self.use_raindrop_context and "rd_src" in sample0:
                        self.context_input_dim = int(sample0["rd_src"].shape[-1] // 2)
                        self.chartevents_input_dim = int(sample0["ce_rd_src"].shape[-1] // 2)
                    
                    if "med_values" in sample0 and sample0["med_values"] is not None:
                        self.n_medications = sample0["med_values"].shape[-1]
            else:
                # Single test dataset using the original filter setting
                self.test_dataset = MIMICDataset(
                    self.data_root,
                        split="test",
                split_mode=self.split_mode,
                    random_state=self.random_state,
                    max_samples=self.max_samples,
                ood_holdout_ratio=self.ood_holdout_ratio,
                    filter_flat_trajectories=self.filter_flat_trajectories)
                if len(self.test_dataset) > 0 and self.context_input_dim is None:
                    sample0 = self.test_dataset[0]
                    if self.use_raindrop_context and "rd_src" in sample0:
                        self.context_input_dim = int(sample0["rd_src"].shape[-1] // 2)
                        self.chartevents_input_dim = int(sample0["ce_rd_src"].shape[-1] // 2)

                    if "med_values" in sample0 and sample0["med_values"] is not None:
                        self.n_medications = sample0["med_values"].shape[-1]

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
        ce_pad_keys = ["ce_rd_src", "ce_rd_times"]

        # No fixed-size legacy context tensors to stack
        stack_keys = []

        no_pad_keys = ["init_state", "ic_mask", "static", "rd_length", "ce_rd_length", "hadm_id", "traj_id", "med_combo_id"]

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
        
        ce_lengths = [int(item["ce_rd_length"]) for item in batch]
        ce_max_len = max(ce_lengths) if len(ce_lengths) > 0 else 0

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

        # Pad Chartevents Raindrop-specific sequences
        for key in ce_pad_keys:
            sequences = [item[key] for item in batch]
            padded_sequences = []
            for seq in sequences:
                if seq.dim() == 1:
                    if seq.shape[0] < ce_max_len:
                        padding = torch.zeros((ce_max_len - seq.shape[0],), dtype=seq.dtype)
                        padded_seq = torch.cat([seq, padding], dim=0)
                    else:
                        padded_seq = seq[:ce_max_len]
                else:
                    if seq.shape[0] < ce_max_len:
                        padding = torch.zeros((ce_max_len - seq.shape[0],) + tuple(seq.shape[1:]), dtype=seq.dtype)
                        padded_seq = torch.cat([seq, padding], dim=0)
                    else:
                        padded_seq = seq[:ce_max_len]
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
            collated["ce_rd_src"],
            collated["ce_rd_times"],
            collated["ce_rd_length"],
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
            collated["med_combo_id"], # [B]
        )
