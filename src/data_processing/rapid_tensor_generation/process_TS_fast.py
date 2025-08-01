import os
import time
import numpy as np
import pandas as pd
import torch
import pickle
from tqdm import tqdm
import concurrent.futures
from functools import partial
from pathlib import Path

# =============================================================================
# 1. Helper: Pad or truncate an array to exactly T rows.
# =============================================================================
def pad_or_truncate(arr, T, dtype=np.float32):
    M = arr.shape[0]
    if M < T:
        if arr.ndim == 1:
            padded = np.zeros((T,), dtype=dtype)
            padded[:M] = arr
        else:
            padded = np.zeros((T, arr.shape[1]), dtype=dtype)
            padded[:M, :] = arr
        return padded
    else:
        return arr[:T]

# =============================================================================
# 2. Optimized pivot function (dense representation, 2 time tensors)
# =============================================================================
def pivot_and_pad_physio_optimized(physio_df, name_list, T):
    """
    Process the physiology DataFrame for a single hadm_id as follows:
    
      1. Pivot the data using the already-created 'rounded_charttime' column.
         (Since we rounded to 3h, this groups events into 3-hour bins.)
      2. Because pivot_table only returns rows where there is at least one measurement,
         the resulting dense index represents the actual measurement times.
      3. Compute two time offsets:
          - abs_time: difference (in hours) relative to midnight of the admission day.
          - rel_time: difference (in hours) relative to the admission time (first event).
      4. Pad (or truncate) all outputs to exactly T rows.
         The "length" equals the number of actual events (capped at T).
    
    Parameters:
      - physio_df: a DataFrame for one hadm_id.
      - name_list: the master list of names in desired order (most to least common).
      - T: the fixed number of time steps (events) to output.
    
    Returns:
       (values_tensor, mask_tensor, abs_time_tensor, rel_time_tensor, length)
    """
    pivot_df = physio_df.pivot_table(
        index='rounded_charttime',
        columns='name',
        values='value',
        aggfunc='first'
    ).sort_index()
    pivot_df = pivot_df.reindex(columns=name_list)
    
    values_arr = pivot_df.fillna(0).to_numpy(dtype=np.float32)
    mask_arr = pivot_df.notnull().astype(np.float32).to_numpy()
    
    # Compute time offsets:
    first_time = pivot_df.index[0]
    admission_midnight = first_time.normalize()
    baseline_abs = np.int64(admission_midnight.value)
    time_ints = pivot_df.index.astype(np.int64).to_numpy()
    abs_time = ((time_ints - baseline_abs) / 1e9 / 3600).astype(np.float32)
    
    baseline_rel = np.int64(first_time.value)
    rel_time = ((time_ints - baseline_rel) / 1e9 / 3600).astype(np.float32)
    
    M = values_arr.shape[0]
    length = M if M < T else T
    
    values_arr = pad_or_truncate(values_arr, T, dtype=np.float32)
    mask_arr = pad_or_truncate(mask_arr, T, dtype=np.float32)
    abs_time = pad_or_truncate(abs_time, T, dtype=np.float32)
    rel_time = pad_or_truncate(rel_time, T, dtype=np.float32)
    
    values_tensor = torch.from_numpy(values_arr).float()
    mask_tensor = torch.from_numpy(mask_arr).float()
    abs_time_tensor = torch.from_numpy(abs_time).float()
    rel_time_tensor = torch.from_numpy(rel_time).float()
    
    return (values_tensor.clone(), mask_tensor.clone(),
            abs_time_tensor.clone(), rel_time_tensor.clone(), length)

# =============================================================================
# 3. Function to process each hadm_id group
# =============================================================================
def process_hadm(hadm_group, name_list, T, cache_dir):
    """
    Process a single hadm_id group:
      - Sort by charttime.
      - Compute the dense tensor representation.
      - Save the resulting tensors and length to a .pt file.
    
    Returns:
      A tuple (hadm_id, length)
    """
    hadm, physio_sub_df = hadm_group
    physio_sub_df = physio_sub_df.sort_values(by='charttime')
    
    (values_tensor, mask_tensor, abs_time_tensor,
     rel_time_tensor, length) = pivot_and_pad_physio_optimized(physio_sub_df, name_list, T)
    
    file_path = os.path.join(cache_dir, f"tensor_cache_{int(hadm)}.pt")
    torch.save(
        (values_tensor, mask_tensor, abs_time_tensor, rel_time_tensor, length), 
        file_path
    )
    
    # Return the hadm id and its corresponding length.
    return (hadm, length)

# =============================================================================
# 4. Main workflow: Load DataFrame, preprocess, and run parallel processing
# =============================================================================
if __name__ == '__main__':
    # Setup paths relative to current working directory.
    ROOT_DIR = Path.cwd()
    temp_dfs_dir = ROOT_DIR / "temp_dfs_lite"
    input_pkl = temp_dfs_dir / "sorted_filtered_df.pkl"
    cache_dir = temp_dfs_dir / "precomputed_tensors"
    
    top_how_many = 80
    max_T_events = 80
    
    combined_physio_df = pd.read_pickle(input_pkl)
    
    if not np.issubdtype(combined_physio_df['charttime'].dtype, np.datetime64):
        combined_physio_df['charttime'] = pd.to_datetime(combined_physio_df['charttime'], errors='coerce')
    combined_physio_df['rounded_charttime'] = combined_physio_df['charttime'].dt.round('3h')
    
    name_list = combined_physio_df['name'].value_counts().head(top_how_many).index.tolist()
    T = max_T_events
    
    if not cache_dir.exists():
        cache_dir.mkdir(parents=True)
    
    hadm_groups = list(combined_physio_df.groupby('hadm_id'))
    process_func = partial(process_hadm, name_list=name_list, T=T, cache_dir=str(cache_dir))
    
    # Prepare a dictionary to store the length for each hadm_id.
    lengths_dict = {}
    
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
        futures = {executor.submit(process_func, grp): grp[0] for grp in hadm_groups}
        for future in tqdm(concurrent.futures.as_completed(futures),
                           total=len(futures), desc="Processing HADM IDs"):
            hadm = futures[future]
            try:
                result = future.result()
                # result is a tuple of (hadm, length)
                lengths_dict[result[0]] = result[1]
            except Exception as exc:
                print(f'HADM {hadm} generated an exception: {exc}')
    
    # Save the aggregated lengths dictionary as a pickle.
    lengths_file = temp_dfs_dir / "hadm_lengths.pkl"
    with open(lengths_file, "wb") as f:
        pickle.dump(lengths_dict, f)
    
    print(f"Saved lengths for {len(lengths_dict)} hadm_ids to {lengths_file}")