import os
import json
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from .LLM_utils import run_LLM  
import torch
from torch.utils.data import Dataset, DataLoader
import nltk
from nltk.tokenize import word_tokenize
import re
nltk.download('punkt', quiet=True)  # Suppress nltk download messages
from datetime import timedelta
from tqdm import tqdm
import lmdb
import time
import joblib  # Add joblib for memory mapping
import shutil
from pathlib import Path
from train_utils import get_device



# Add a global debug flag to control verbosity
DEBUG_PRINT = True

def debug_print(*args, **kwargs):
    """Print only if DEBUG_PRINT is True"""
    if DEBUG_PRINT:
        print(*args, **kwargs)

class MIMICContrastivePairsDatasetLite(Dataset):
    def __init__(self,
                 split="train",
                 cache_dir="./cache",
                 task_mode="CONTRASTIVE",
                 chunk_hours=12,
                 label_window=24,
                 T=80,
                 test_ds_only=False):

        self.split           = split
        self.cache_dir       = cache_dir
        self.task_mode       = task_mode
        self.chunk_hours     = chunk_hours
        self.label_window    = label_window
        self.T               = T
        self.test_ds_only    = test_ds_only
        
        # ────────────────── label cache ──────────────────
        label_dir = os.path.join(self.cache_dir, "label_cache")

        # 1) row map
        with open(os.path.join(label_dir, "hadm_row_map.pkl"), "rb") as f:
            self._row = pickle.load(f)             # dict[int -> int]

        N = len(self._row)

        # 2) scalar labels
        self._label_mm = np.memmap(os.path.join(label_dir, "labels_scalar.bin"),
                                   dtype=np.uint8, mode="r").reshape(N, 2)
        # 3) padded index matrices (mmap)
        self._cur_mat = np.load(os.path.join(label_dir, "current_idx_mat.npy"), mmap_mode="r")
        self._cur_len = np.load(os.path.join(label_dir, "current_len.npy"),     mmap_mode="r")
        self._nxt_mat = np.load(os.path.join(label_dir, "next_idx_mat.npy"),    mmap_mode="r")
        self._nxt_len = np.load(os.path.join(label_dir, "next_len.npy"),        mmap_mode="r")

        self.K = self._cur_mat.shape[1]
        self.P = int(self._cur_mat.max())

        #

        self.phecode_size = pickle.load(open(os.path.join(cache_dir, "phecode_size.pkl"), "rb"))
        # ────────────────── baseline, embeddings, etc. ──────────────────
        baseline_path = os.path.join(cache_dir, "processed_baseline_df.pkl")
        try:
            self.baseline_df = pd.read_pickle(baseline_path)
            print(f"Loaded baseline dataframe: {baseline_path}")
        except Exception as e:
            print(f"Baseline DF not found ({e}); continuing without.")
            self.baseline_df = None

        # ────────────────── ds embeddings ──────────────────
        self.ds_embeddings_dir = os.path.join(cache_dir, "DS_embeddings")
        os.makedirs(self.ds_embeddings_dir, exist_ok=True)

        merged_path = os.path.join(cache_dir, "merged_with_disch_df_final_filtered.pkl")
        self.merged_with_disch_df = pd.read_pickle(merged_path)
        
        # Ensure we have hadm_id as index for fast lookups
        if self.merged_with_disch_df.index.name != "hadm_id":
            self.merged_with_disch_df.set_index("hadm_id", inplace=True)
        
        if 'hadm_id' not in self.merged_with_disch_df.columns:
            self.merged_with_disch_df['hadm_id'] = self.merged_with_disch_df.index

        self.hadm_ids = self.get_split_hadm_ids(split)

        if self.task_mode == "NEXT_24h":
            self.compute_samples()
        elif self.test_ds_only:
            self.samples = list(self.ds_embeddings_dir.iterdir())
        else: 
            self.samples = self.hadm_ids
            
        # Filter samples to only include those with valid cache files
        if self.task_mode == 'CONTRASTIVE':
            self.filter_valid_samples()

    def filter_valid_samples(self):
        """Filter out hadm_ids that don't have corresponding cache files"""
        joint_cache_dir = os.path.join(self.cache_dir, "precomputed_tensors")
        os.makedirs(joint_cache_dir, exist_ok=True)
        full_cache_fn = lambda hid: f"tensor_cache_{int(hid)}.pt"
        
        valid_samples = []
        invalid_count = 0
        
        for hadm_id in tqdm(self.samples, desc="Filtering valid hadm_ids"):
            path = os.path.join(joint_cache_dir, full_cache_fn(hadm_id))
            if os.path.exists(path):
                valid_samples.append(hadm_id)
            else:
                invalid_count += 1
                if invalid_count <= 10:  # Only print first 10 to avoid spam
                    print(f"Skipping hadm_id {hadm_id} - no cache file found")
                elif invalid_count == 11:
                    print("... more invalid hadm_ids ...")
        
        print(f"Filtered {invalid_count} invalid hadm_ids, {len(valid_samples)} remaining")
        self.samples = valid_samples

    def _fetch_labels(self, hadm_id: int):
        """O(1) look-up of scalars + padded indices"""
        row = self._row.get(int(hadm_id), None)
        if row is None:
            # unseen id (shouldn't happen after de-dup)
            return 0, 0, torch.full((self.K,), self.P, dtype=torch.long), 0, \
                   torch.full((self.K,), self.P, dtype=torch.long), 0

        mort, readm = self._label_mm[row]
        # Create copies of the NumPy arrays before converting to tensors
        cur_pad = torch.from_numpy(self._cur_mat[row].copy()).long()
        cur_len = int(self._cur_len[row])
        nxt_pad = torch.from_numpy(self._nxt_mat[row].copy()).long()
        nxt_len = int(self._nxt_len[row])
        return mort, readm, cur_pad, cur_len, nxt_pad, nxt_len


  
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # Prepare joint_cache_dir and full_cache_fn for all but test_ds_only
        joint_cache_dir = os.path.join(self.cache_dir, "precomputed_tensors")
        os.makedirs(joint_cache_dir, exist_ok=True)
        full_cache_fn   = lambda hid: f"tensor_cache_{hid}.pt"

        if self.test_ds_only:
            #print(f"Getting item {idx} for test_ds_only")
            ds_embeddings_list = os.listdir(self.ds_embeddings_dir)
            hadm_id = ds_embeddings_list[idx].split('_')[1].split('.')[0]
            #print(f"hadm_id: {hadm_id}")

            embedding_path = os.path.join(self.ds_embeddings_dir, f"embedding_{hadm_id}.pt")
            embedding = torch.load(embedding_path)
            #print(f"embedding: {embedding.shape}")
            mort, readm, cur_pad, cur_len, nxt_pad, nxt_len = self._fetch_labels(hadm_id)

            return {
                "hadm_id": int(hadm_id),
                'values': torch.tensor([]),
                'mask': torch.tensor([]),
                'static': torch.tensor([]),
                'times': torch.tensor([]),
                'length': 0,
                "mortality_label":   float(mort),
                "readmission_label": float(readm),
                "current_idx_padded": cur_pad,   # (K,)
                "current_len":        cur_len,   # int
                "next_idx_padded":    nxt_pad,   # (K,)
                "next_len":           nxt_len,   # int
                "ds_embedding":       embedding,
            }
        if self.task_mode == 'CONTRASTIVE' or self.task_mode == 'MULTITASK':
            hadm_id = self.samples[idx]
            path = os.path.join(joint_cache_dir, full_cache_fn(hadm_id))
            # No need to check if path exists since we've filtered the samples
            physio, mask, abs_time, rel_time, full_length = torch.load(path)
            ds_path = os.path.join(self.ds_embeddings_dir, f"embedding_{hadm_id}.pt")
            # Return empty tensor instead of empty list when file not found
            ds_emb = torch.load(ds_path) if os.path.exists(ds_path) else torch.zeros(0, 768)

            # Get post-discharge labels
            mort, readm, cur_pad, cur_len, nxt_pad, nxt_len = self._fetch_labels(hadm_id)

            return {
                'hadm_id':      int(hadm_id),
                'values':       physio,
                'mask':         mask,
                'static':       self.get_baseline(hadm_id),
                'times':        abs_time,
                'length':       int(full_length),
                "mortality_label":   float(mort),
                "readmission_label": float(readm),
                "current_idx_padded": cur_pad,   # (K,)
                "current_len":        cur_len,   # int
                "next_idx_padded":    nxt_pad,   # (K,)
                "next_len":           nxt_len,   # int
                "ds_embedding":       ds_emb,
            }
        elif self.task_mode == 'NEXT_24h':
            hadm_id, chunk_idx, label = self.samples[idx]
            cutoff_hr = chunk_idx * self.chunk_hours

            path = os.path.join(joint_cache_dir, full_cache_fn(hadm_id))
            if not os.path.exists(path):
                return None
            physio, mask, abs_time, rel_time, _ = torch.load(path)

            # compute slice_end
            rel_np    = rel_time.cpu().numpy()
            slice_end = np.searchsorted(rel_np, cutoff_hr, side='right')
            slice_end = max(1, slice_end)

            # slice + pad
            physio_slice = self.pad_tensor(physio[:slice_end], self.T)
            mask_slice   = self.pad_tensor(mask[:slice_end],   self.T)
            time_slice   = self.pad_tensor(abs_time[:slice_end], self.T)

            length = int((mask_slice.sum(dim=1) > 0).sum().item())

            # Get post-discharge labels
            mort, readm, cur_pad, cur_len, nxt_pad, nxt_len = self._fetch_labels(hadm_id)

            ds_path = os.path.join(self.ds_embeddings_dir, f"embedding_{hadm_id}.pt")
            ds_emb  = torch.load(ds_path) if os.path.exists(ds_path) else torch.zeros(0, 768)

            return {
                'hadm_id':      int(hadm_id),
                'values':       physio_slice,
                'mask':         mask_slice,
                'static':       self.get_baseline(hadm_id),
                'times':        time_slice,
                'length':       length,
                'label':        int(label),
                "mortality_label":   float(mort),
                "readmission_label": float(readm),
                "current_idx_padded": cur_pad,   # (K,)
                "current_len":        cur_len,   # int
                "next_idx_padded":    nxt_pad,   # (K,)
                "next_len":           nxt_len,   # int
                "ds_embedding":       ds_emb,
            }
        else:
            raise ValueError(f"Bad mode {self.task_mode}")
    
    def compute_samples(self, cache_file=None):
        """
        Compute and cache samples for the NEXT_24h task.
        Uses adaptive stride to reduce sample count while maintaining resolution near important events.
        """
        if cache_file is None:
            cache_file = os.path.join(
                self.cache_dir, 
                f"samples_{self.split}_chunk{self.chunk_hours}_label{self.label_window}_adaptive.pkl"
            )
            
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'rb') as f:
                    self.samples = pickle.load(f)
                print(f"Loaded {len(self.samples)} samples from cache: {cache_file}")
                pos_samples = sum(1 for _, _, label in self.samples if label == 1)
                print(f"Found {pos_samples} positive samples ({pos_samples/len(self.samples)*100:.2f}%)")
                return
            except Exception as e:
                print(f"Failed to load cache: {e}; recomputing...")
        
        df = self.merged_with_disch_df.copy()
        df['admittime'] = pd.to_datetime(df['admittime'], errors='coerce')
        if 'dischtime' in df.columns:
            df['dischtime'] = pd.to_datetime(df['dischtime'], errors='coerce')
        if 'deathtime' in df.columns:
            df['deathtime'] = pd.to_datetime(df['deathtime'], errors='coerce')
            
        meta_lookup = df.set_index('hadm_id').sort_index()
        self.samples = []
        
        for hadm_id in tqdm(self.hadm_ids, desc="Computing sample windows"):
            if hadm_id not in meta_lookup.index:
                continue

            row_data = meta_lookup.loc[hadm_id]
            row = row_data.iloc[0] if isinstance(row_data, pd.DataFrame) else row_data
            admittime = row['admittime']
            deathtime = row.get('deathtime', None)
            if pd.isnull(deathtime):
                deathtime = None
            dischtime = row.get('dischtime', None)
            if pd.isnull(dischtime):
                dischtime = None

            if deathtime is not None:
                chunk_end_cutoff = deathtime
            else:
                if dischtime is None:
                    continue
                chunk_end_cutoff = dischtime

            if chunk_end_cutoff <= admittime:
                continue

            i = 1
            while True:
                chunk_end_time = admittime + pd.Timedelta(hours=self.chunk_hours * i)
                if chunk_end_time > chunk_end_cutoff:
                    break

                if deathtime is not None:
                    hours_to_death = (deathtime - chunk_end_time).total_seconds() / 3600
                    if hours_to_death <= 24:
                        stride = 1
                    elif hours_to_death <= 72:
                        stride = 2
                    else:
                        stride = 3
                else:
                    hours_to_discharge = (dischtime - chunk_end_time).total_seconds() / 3600
                    if hours_to_discharge <= 48:
                        stride = 2
                    else:
                        stride = 3

                if i % stride == 0:
                    label_start_time = chunk_end_time
                    label_end_time = chunk_end_time + pd.Timedelta(hours=self.label_window)
                    if deathtime is not None and label_start_time <= deathtime < label_end_time:
                        label = 1
                    else:
                        label = 0

                    self.samples.append((hadm_id, i, label))
                i += 1

        try:
            os.makedirs(os.path.dirname(cache_file), exist_ok=True)
            with open(cache_file, 'wb') as f:
                pickle.dump(self.samples, f)
            pos_samples = sum(1 for _, _, label in self.samples if label == 1)
            print(f"Saved {len(self.samples)} samples to cache: {cache_file}")
            print(f"Found {pos_samples} positive samples ({pos_samples/len(self.samples)*100:.2f}%)")
        except Exception as e:
            print(f"Failed to save cache: {e}")

    def pad_tensor(self, tensor, target_length):
        current_length = tensor.shape[0]
        if current_length < target_length:
            pad_size = target_length - current_length
            # Assuming you want to pad along the first dimension.
            padding = torch.zeros(pad_size, *tensor.shape[1:], dtype=tensor.dtype)
            tensor = torch.cat([tensor, padding], dim=0)
        return tensor

    def select_if_exists(self, df, hadm_id):
        if df is None or df.empty:
            return None
        if df.index.name == 'hadm_id' and hadm_id in df.index:
            return df.loc[[hadm_id]]
        elif 'hadm_id' in df.columns:
            filtered = df[df['hadm_id'] == hadm_id]
            return filtered if not filtered.empty else None
        else:
            return None

    def get_baseline(self, hadm_id):
        """
        Extracts the baseline features from the processed_baseline_df for the given hadm_id.
        The DataFrame is assumed to already contain one-hot encoded features.
        Returns a tensor of the one-hot encoded baseline features.
        """
        meta = self.select_if_exists(self.baseline_df, hadm_id)
        if meta is None or meta.empty:
            # If baseline information is not found, return an empty tensor.
            return torch.tensor([])
        # Convert the first (and assumed only) row to a tensor.
        baseline_tensor = torch.tensor(meta.values[0], dtype=torch.float)
        return baseline_tensor
    
    def split_by_subject_id(self, df, train_ratio=0.7, val_ratio=0.29, test_ratio=0.01, random_state=42):
        """
        Split hospital admissions into train/val/test sets based on subject_id.
        This ensures that all admissions for a given patient are in the same split.
        
        Args:
            train_ratio: Ratio of patients to use for training
            val_ratio: Ratio of patients to use for validation
            test_ratio: Ratio of patients to use for testing
            random_state: Random seed for reproducibility
            
        Returns:
            tuple: (train_hadm_ids, val_hadm_ids, test_hadm_ids)
        """
        if df is None:
            raise ValueError("Merged dataframe not loaded. Call load_demographics() first.")
            
        # Verify the ratios sum to 1.0
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-10, "Ratios must sum to 1.0"
            
        # Get unique subject_ids
        unique_subjects = df['subject_id'].unique()
        
        # Shuffle the subject_ids
        np.random.seed(random_state)
        np.random.shuffle(unique_subjects)
        
        # Calculate split indices
        n_subjects = len(unique_subjects)
        train_idx = int(train_ratio * n_subjects)
        val_idx = train_idx + int(val_ratio * n_subjects)
        
        # Split subject_ids
        train_subjects = unique_subjects[:train_idx]
        val_subjects = unique_subjects[train_idx:val_idx]
        test_subjects = unique_subjects[val_idx:]
        
        # Get hadm_ids for each split - now we can access hadm_id as a column
        train_hadm_ids = df[df['subject_id'].isin(train_subjects)]['hadm_id'].tolist()
        val_hadm_ids = df[df['subject_id'].isin(val_subjects)]['hadm_id'].tolist()
        test_hadm_ids = df[df['subject_id'].isin(test_subjects)]['hadm_id'].tolist()
            
        # Store the splits
        self.train_hadm_ids = train_hadm_ids
        self.val_hadm_ids = val_hadm_ids
        self.test_hadm_ids = test_hadm_ids
        
        # Print split statistics
        print(f"Split by subject_id: {len(train_subjects)} train, {len(val_subjects)} val, {len(test_subjects)} test subjects")
        print(f"Corresponding to: {len(train_hadm_ids)} train, {len(val_hadm_ids)} val, {len(test_hadm_ids)} test admissions")
        
        return train_hadm_ids, val_hadm_ids, test_hadm_ids
        
    def get_split_hadm_ids(self, split='train'):
        """
        Get hospital admission IDs for a specific split.
        
        Args:
            split: One of 'train', 'val', 'test'
            
        Returns:
            list: Hospital admission IDs for the specified split
        """
        # Use a copy to avoid modifying the original DataFrame
        self.split_by_subject_id(self.merged_with_disch_df.copy())
        if split == 'train':
            return self.train_hadm_ids
        elif split == 'val':
            return self.val_hadm_ids
        elif split == 'test':
            return self.test_hadm_ids
        else:
            raise ValueError(f"Unknown split: {split}. Must be one of 'train', 'val', 'test'")





