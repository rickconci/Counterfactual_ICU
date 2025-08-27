import pandas as pd
import numpy as np
import torch
from pathlib import Path
import pickle
from datetime import datetime, timedelta
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional
import os



def create_context_tensors(
        waveforms_parquet_path: str,
        med_tensors_metadata_path: str,
        med_data_parquet_path: str,
        output_dir: str = "./context_tensors_output",
        context_duration_minutes: int = 60,
        context_interval_minutes: int = 10
) -> Dict:
    """
    Create context tensors for the hour before each t₀.
    Modified to always save tensors, even if they're all zeros.
    """

    # Calculate context parameters
    n_context_intervals = int(context_duration_minutes / context_interval_minutes)

    print(f"=== Creating Context Tensors ===")
    print(f"Context window: {context_duration_minutes} minutes before t₀")
    print(f"Context intervals: {n_context_intervals} intervals of {context_interval_minutes} minutes each")

    # Create output directories
    output_path = Path(output_dir)
    physio_context_dir = output_path / "physio_context"
    meds_context_dir = output_path / "meds_context"
    physio_context_dir.mkdir(parents=True, exist_ok=True)
    meds_context_dir.mkdir(parents=True, exist_ok=True)

    # Load existing trajectory metadata
    print("Loading med tensor metadata...")
    with open(med_tensors_metadata_path, 'rb') as f:
        med_metadata = pickle.load(f)

    trajectories = med_metadata['trajectories']
    print(f"Found {len(trajectories)} trajectories to create context for")

    # Load waveforms data
    print("Loading waveforms data...")
    waveforms_df = load_waveforms_data(waveforms_parquet_path)

    # Load medication data
    print("Loading medication data...")
    med_df = load_medication_data(med_data_parquet_path)

    # Create context tensors for each trajectory
    physio_context_metadata = {}
    meds_context_metadata = {}

    # Changed: No more skipping - all tensors are created
    for traj_key, traj_info in tqdm(trajectories.items(), desc="Creating context tensors"):

        # Create physio context tensor (always returns tensors now)
        physio_values, physio_mask = create_physio_context_tensor(
            traj_key, traj_info, waveforms_df,
            n_context_intervals, context_interval_minutes
        )

        # Always save physio context tensor
        physio_filepath = save_context_tensor(
            traj_key, physio_values, physio_mask,
            n_context_intervals, context_interval_minutes,
            physio_context_dir, "physio_context"
        )

        physio_context_metadata[traj_key] = {
            'hadm_id': traj_info['hadm_id'],
            'action_cluster_id': traj_info['action_cluster_id'],
            't0_time': traj_info['t0_time'],
            'file_path': physio_filepath,
            'n_intervals': n_context_intervals,
            'interval_minutes': context_interval_minutes,
            'total_measurements': int(torch.sum(physio_mask > 0).item()),
            'has_data': bool(torch.sum(physio_mask > 0).item() > 0)  # Track if tensor has actual data
        }

        # Create meds context tensor (always returns tensors now)
        meds_values, meds_mask = create_meds_context_tensor(
            traj_key, traj_info, med_df,
            n_context_intervals, context_interval_minutes,
            med_metadata['item_labels']
        )

        # Always save meds context tensor
        meds_filepath = save_context_tensor(
            traj_key, meds_values, meds_mask,
            n_context_intervals, context_interval_minutes,
            meds_context_dir, "meds_context"
        )

        meds_context_metadata[traj_key] = {
            'hadm_id': traj_info['hadm_id'],
            'action_cluster_id': traj_info['action_cluster_id'],
            't0_time': traj_info['t0_time'],
            'file_path': meds_filepath,
            'n_intervals': n_context_intervals,
            'interval_minutes': context_interval_minutes,
            'total_measurements': int(torch.sum(meds_mask > 0).item()),
            'has_data': bool(torch.sum(meds_mask > 0).item() > 0)  # Track if tensor has actual data
        }

    # Save metadata
    context_metadata = {
        'physio_context': physio_context_metadata,
        'meds_context': meds_context_metadata,
        'n_context_intervals': n_context_intervals,
        'context_interval_minutes': context_interval_minutes,
        'context_duration_minutes': context_duration_minutes,
        'total_trajectories': len(trajectories),
        'physio_tensors_created': len(physio_context_metadata),  # Now always equals total_trajectories
        'meds_tensors_created': len(meds_context_metadata),      # Now always equals total_trajectories
        'physio_tensors_with_data': sum(1 for m in physio_context_metadata.values() if m['has_data']),  # New: count non-zero tensors
        'meds_tensors_with_data': sum(1 for m in meds_context_metadata.values() if m['has_data']),      # New: count non-zero tensors
        'created_at': datetime.now().isoformat(),
        'source_trajectories': med_tensors_metadata_path
    }

    metadata_file = output_path / "context_tensors_metadata.pkl"
    with open(metadata_file, 'wb') as f:
        pickle.dump(context_metadata, f)

    # Print summary
    print(f"\n=== Context Tensors Summary ===")
    print(f"Total trajectories: {len(trajectories)}")
    print(f"Physio context tensors created: {len(physio_context_metadata)} (100%)")
    print(f"Meds context tensors created: {len(meds_context_metadata)} (100%)")
    print(f"Physio tensors with data: {sum(1 for m in physio_context_metadata.values() if m['has_data'])}")
    print(f"Meds tensors with data: {sum(1 for m in meds_context_metadata.values() if m['has_data'])}")

    print(f"\nFiles saved to: {output_path}")
    print(f"Metadata saved to: {metadata_file}")

    return context_metadata


def load_waveforms_data(parquet_path: str) -> pd.DataFrame:
    """
    Load waveforms data for context tensor creation.
    """
    print(f"Loading waveforms data from {parquet_path}...")
    df = pd.read_parquet(parquet_path)

    # Ensure absolute_timestamp is datetime
    if not pd.api.types.is_datetime64_any_dtype(df['absolute_timestamp']):
        df['absolute_timestamp'] = pd.to_datetime(df['absolute_timestamp'])

    # Check for required physio columns
    required_physio = ['ABP MEAN_z', 'CVP_z', 'HR_z', 'RESP_z']
    available_physio = [col for col in required_physio if col in df.columns]
    missing_physio = [col for col in required_physio if col not in df.columns]

    print(f"Available physio measurements: {available_physio}")
    if missing_physio:
        print(f"Missing physio measurements: {missing_physio}")

    return df


def load_medication_data(parquet_path: str) -> pd.DataFrame:
    """
    Load medication data for context tensor creation.

    Supports parquet and pickle-like binaries (.pkl/.pickle/.bin). Normalizes
    schema to include a 'rate/weight_normalized' column by mapping from common
    alternatives when needed.
    """
    print(f"Loading medication data from {parquet_path}...")

    df: Optional[pd.DataFrame] = None

    # Try to load based on extension, with fallbacks
    lower_path = parquet_path.lower()
    try:
        if lower_path.endswith('.parquet'):
            df = pd.read_parquet(parquet_path)
        elif lower_path.endswith(('.pkl', '.pickle', '.bin')):
            try:
                df = pd.read_pickle(parquet_path)
            except Exception:
                import pickle as _pkl
                with open(parquet_path, 'rb') as f:
                    obj = _pkl.load(f)
                if isinstance(obj, pd.DataFrame):
                    df = obj
                else:
                    raise ValueError("Pickle file did not contain a pandas DataFrame")
        else:
            # Heuristic: try parquet first, then pickle
            try:
                df = pd.read_parquet(parquet_path)
            except Exception:
                df = pd.read_pickle(parquet_path)
    except Exception as e:
        raise ValueError(f"Failed to load medication data from {parquet_path}: {e}")

    # Normalize column names where possible (map common variants to canonical)
    cols_lower_map = {c.lower(): c for c in df.columns}

    def _ensure_column(df_in: pd.DataFrame, canonical: str, candidates: List[str]) -> None:
        for cand in candidates:
            cand_lower = cand.lower()
            if cand_lower in cols_lower_map:
                src = cols_lower_map[cand_lower]
                if src != canonical and canonical not in df_in.columns:
                    df_in.rename(columns={src: canonical}, inplace=True)
                return

    _ensure_column(df, 'hadm_id', ['hadm_id', 'HADM_ID'])
    _ensure_column(df, 'start_time', ['start_time', 'START_TIME', 'STARTTIME'])
    _ensure_column(df, 'end_time', ['end_time', 'END_TIME', 'ENDTIME'])
    _ensure_column(df, 'item_label', ['item_label', 'ITEM_LABEL'])

    # Normalize rate column -> create 'rate/weight_normalized' if missing
    rate_canonical = 'rate/weight_normalized'
    if rate_canonical not in df.columns:
        # Candidates: common variants and fuzzy contains('rate') & contains('weight')
        candidate_order = [
            'rate_weight_normalized',
            'rate_per_kg',
            'rate_per_weight',
            'rate/weight',
            'rate',
            'RATE',
        ]
        picked_src: Optional[str] = None

        for cand in candidate_order:
            if cand in df.columns:
                picked_src = cand
                break
            if cand.lower() in cols_lower_map:
                picked_src = cols_lower_map[cand.lower()]
                break

        if picked_src is None:
            # Fuzzy search: any col name containing both 'rate' and 'weight'
            fuzzy = [c for c in df.columns if ('rate' in c.lower() and 'weight' in c.lower())]
            if len(fuzzy) > 0:
                picked_src = fuzzy[0]

        if picked_src is not None:
            df[rate_canonical] = pd.to_numeric(df[picked_src], errors='coerce')
        else:
            # As a last resort, if a plain 'rate' exists, use it
            if 'rate' in df.columns:
                df[rate_canonical] = pd.to_numeric(df['rate'], errors='coerce')

    # Validate required columns
    required_cols = ['hadm_id', 'start_time', 'end_time', 'item_label', rate_canonical]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(
            "Missing required medication columns: {}. Available: {}".format(
                missing_cols, list(df.columns)
            )
        )

    # Ensure datetime types
    if not pd.api.types.is_datetime64_any_dtype(df['start_time']):
        df['start_time'] = pd.to_datetime(df['start_time'], errors='coerce')
    if not pd.api.types.is_datetime64_any_dtype(df['end_time']):
        df['end_time'] = pd.to_datetime(df['end_time'], errors='coerce')

    print(f"Loaded {len(df)} medication events")
    return df


def create_physio_context_tensor(
        traj_key: str,
        traj_info: Dict,
        waveforms_df: pd.DataFrame,
        n_intervals: int,
        interval_minutes: int
) -> Tuple[torch.Tensor, torch.Tensor]:  # Changed: Always returns tensors, never None
    """
    Create physiological context tensor for the hour before t₀.
    Always returns tensors, even if all zeros.

    Returns:
        tuple: (physio_values, physio_mask) - always returns, may be all zeros
    """
    hadm_id = traj_info['hadm_id']
    t0_time = pd.to_datetime(traj_info['t0_time'])

    # Define context window: 1 hour before t₀
    context_start_time = t0_time - pd.Timedelta(minutes=interval_minutes * n_intervals)
    context_end_time = t0_time

    # Initialize arrays: [ABP MEAN, NBP MEAN, CVP, HR, RESP]
    physio_measurements = ['ABP MEAN', 'CVP', 'HR', 'RESP']
    n_measurements = len(physio_measurements)

    values_array = np.zeros((n_intervals, n_measurements), dtype=np.float32)
    mask_array = np.zeros((n_intervals, n_measurements), dtype=np.float32)

    # Get patient waveforms data in context window
    patient_waveforms = waveforms_df[
        (waveforms_df['hadm_id'] == hadm_id) &
        (waveforms_df['absolute_timestamp'] >= context_start_time) &
        (waveforms_df['absolute_timestamp'] < context_end_time)
        ].copy()

    if len(patient_waveforms) == 0:
        print(f"    No physio data in context window for {traj_key} - saving zero tensor")
        return torch.from_numpy(values_array), torch.from_numpy(mask_array)  # Return zeros instead of None

    # Calculate time indices for context window
    patient_waveforms['time_from_context_start'] = (
            patient_waveforms['absolute_timestamp'] - context_start_time
    ).dt.total_seconds()
    patient_waveforms['context_interval_idx'] = (
            patient_waveforms['time_from_context_start'] // (interval_minutes * 60)
    ).astype(int)

    # Process each measurement type
    for meas_idx, measurement in enumerate(physio_measurements):
        if measurement not in patient_waveforms.columns:
            continue

        # Group by context interval and calculate mean
        for interval_idx in range(n_intervals):
            interval_data = patient_waveforms[
                patient_waveforms['context_interval_idx'] == interval_idx
                ]

            if len(interval_data) > 0:
                measurement_values = interval_data[measurement].dropna()
                if len(measurement_values) > 0:
                    mean_value = float(measurement_values.mean())

                    # Check for NaN
                    if np.isnan(mean_value):
                        print(f"    Warning: NaN in {measurement} for {traj_key} at interval {interval_idx}")
                        continue

                    values_array[interval_idx, meas_idx] = mean_value
                    mask_array[interval_idx, meas_idx] = 1.0

    # Always return tensors (even if all zeros)
    total_measurements = int(np.sum(mask_array))
    if total_measurements == 0:
        print(f"    No valid physio measurements for {traj_key} - saving zero tensor")
    else:
        print(f"    Physio context: {total_measurements} total measurements across {n_measurements} types")

    return torch.from_numpy(values_array), torch.from_numpy(mask_array)


def create_meds_context_tensor(
        traj_key: str,
        traj_info: Dict,
        med_df: pd.DataFrame,
        n_intervals: int,
        interval_minutes: int,
        item_labels: List[str]
) -> Tuple[torch.Tensor, torch.Tensor]:  # Changed: Always returns tensors, never None
    """
    Create medication context tensor for the hour before t₀.
    Always returns tensors, even if all zeros.

    Returns:
        tuple: (meds_values, meds_mask) - always returns, may be all zeros
    """
    hadm_id = traj_info['hadm_id']
    t0_time = pd.to_datetime(traj_info['t0_time'])

    # Define context window: 1 hour before t₀
    context_start_time = t0_time - pd.Timedelta(minutes=interval_minutes * n_intervals)
    context_end_time = t0_time

    # Initialize arrays
    n_medications = len(item_labels)
    values_array = np.zeros((n_intervals, n_medications), dtype=np.float32)
    mask_array = np.zeros((n_intervals, n_medications), dtype=np.float32)

    # Create item_label to index mapping
    item_to_idx = {item: idx for idx, item in enumerate(item_labels)}

    # Get patient medication data
    patient_meds = med_df[med_df['hadm_id'] == hadm_id].copy()

    if len(patient_meds) == 0:
        print(f"    No medication data for patient {hadm_id} - saving zero tensor")
        return torch.from_numpy(values_array), torch.from_numpy(mask_array)  # Return zeros instead of None

    # Filter to infusions that overlap with context window
    relevant_infusions = patient_meds[
        (patient_meds['end_time'] > context_start_time) &
        (patient_meds['start_time'] < context_end_time)
        ].copy()

    if len(relevant_infusions) == 0:
        print(f"    No medication infusions in context window for {traj_key} - saving zero tensor")
        return torch.from_numpy(values_array), torch.from_numpy(mask_array)  # Return zeros instead of None

    # Process each medication type separately (to handle overlaps correctly)
    for item_label in item_labels:
        if item_label not in item_to_idx:
            continue

        item_idx = item_to_idx[item_label]

        # Get all infusions for this medication, sorted by start_time
        item_infusions = relevant_infusions[
            relevant_infusions['item_label'] == item_label
            ].sort_values('start_time')

        if len(item_infusions) == 0:
            continue

        for _, row in item_infusions.iterrows():
            # Get rate value from normalized column
            rate_value = 0.0
            rate_col = 'rate/weight_normalized'
            if rate_col in row and pd.notna(row[rate_col]):
                rate_value = float(row[rate_col])

            if rate_value == 0.0:
                continue

            # Clip infusion to context window
            effective_start = max(context_start_time, row['start_time'])
            effective_end = min(context_end_time, row['end_time'])

            # Convert to context interval indices
            start_seconds = (effective_start - context_start_time).total_seconds()
            end_seconds = (effective_end - context_start_time).total_seconds()

            start_idx = int(start_seconds // (interval_minutes * 60))  # Round down
            end_idx = int(np.ceil(end_seconds / (interval_minutes * 60)))  # Round up

            # Fill all intervals in range (most recent wins due to sort order)
            for time_idx in range(start_idx, end_idx):
                if 0 <= time_idx < n_intervals:
                    values_array[time_idx, item_idx] = rate_value
                    mask_array[time_idx, item_idx] = 1.0

    # Always return tensors (even if all zeros)
    total_medications = int(np.sum(mask_array))
    if total_medications == 0:
        print(f"    No valid medication data in context window for {traj_key} - saving zero tensor")
    else:
        print(f"    Meds context: {total_medications} total medication intervals")

    return torch.from_numpy(values_array), torch.from_numpy(mask_array)


def save_context_tensor(
        traj_key: str,
        values_tensor: torch.Tensor,
        mask_tensor: torch.Tensor,
        n_intervals: int,
        interval_minutes: int,
        output_dir: Path,
        tensor_type: str
) -> str:
    """
    Save context tensor to disk in same format as existing tensors.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create time arrays (relative to t₀, negative values since this is before t₀)
    time_minutes_from_t0 = torch.arange(n_intervals, dtype=torch.float32) * interval_minutes - (
                n_intervals * interval_minutes)
    time_hours_from_t0 = time_minutes_from_t0 / 60.0  # [-1.0, -0.83, -0.67, -0.5, -0.33, -0.17]
    time_seconds_from_t0 = time_minutes_from_t0 * 60

    filepath = output_dir / f"{tensor_type}_{traj_key}.pt"

    # Save in same format as existing tensors: (values, mask, time_seconds, time_hours, length)
    torch.save(
        (values_tensor, mask_tensor, time_seconds_from_t0, time_hours_from_t0, n_intervals),
        filepath
    )

    return str(filepath)


def inspect_context_tensors(metadata: Dict, n_samples: int = 3) -> None:
    """
    Inspect sample context tensors.
    """
    print(f"\n=== Inspecting {n_samples} Sample Context Tensors ===")

    # Get trajectories that have both physio and meds context
    physio_keys = set(metadata['physio_context'].keys())
    meds_keys = set(metadata['meds_context'].keys())
    common_keys = list(physio_keys.intersection(meds_keys))[:n_samples]

    for traj_key in common_keys:
        print(f"\nTrajectory: {traj_key}")

        # Load physio context
        physio_info = metadata['physio_context'][traj_key]
        physio_values, physio_mask, time_sec, time_hr, n_intervals = torch.load(physio_info['file_path'])

        print(f"Physio Context:")
        print(f"  Shape: {physio_values.shape}")
        print(f"  Time range: {time_hr[0]:.2f} to {time_hr[-1]:.2f} hours from t₀")
        print(f"  Total measurements: {torch.sum(physio_mask > 0).item()}")

        # Load meds context
        meds_info = metadata['meds_context'][traj_key]
        meds_values, meds_mask, time_sec, time_hr, n_intervals = torch.load(meds_info['file_path'])

        print(f"Meds Context:")
        print(f"  Shape: {meds_values.shape}")
        print(f"  Time range: {time_hr[0]:.2f} to {time_hr[-1]:.2f} hours from t₀")
        print(f"  Total medication intervals: {torch.sum(meds_mask > 0).item()}")

def preprocess_baseline_values(df,
                               id_col='hadm_id',
                               categorical_features=None,
                               continuous_features=None,
                               binary_features=None,
                               drop_first=True):
    """
    Preprocess baseline values for deep learning input while preserving a unique identifier.
    """
    # Create a copy of the input dataframe and set the identifier as the index.
    processed_df = df.copy()
    if id_col in processed_df.columns:
        processed_df.set_index(id_col, inplace=True)
    print(f"Categorical features: {categorical_features}")
    # Process categorical features with one-hot encoding and ensure 0/1 integer values.
    if categorical_features is not None:
        dummies = pd.get_dummies(processed_df[categorical_features], drop_first=drop_first, dummy_na=False).astype(int)
    else:
        dummies = pd.DataFrame(index=processed_df.index)

    print(f"Dummies shape: {dummies.shape}")
    print(f"Dummies columns: {list(dummies.columns)}")

    # Process continuous features ensuring they are numeric and scale them.
    if continuous_features is not None:
        continuous = processed_df[continuous_features].astype(float)
        # Normalize continuous features
        for col in continuous.columns:
            mean = continuous[col].mean()
            std = continuous[col].std()
            continuous[col] = (continuous[col] - mean) / std
    else:
        continuous = pd.DataFrame(index=processed_df.index)

    # Process binary features and convert them to integer 0 or 1.
    if binary_features is not None:
        binary = processed_df[binary_features].apply(pd.to_numeric, errors='coerce').fillna(0).astype(int)
    else:
        binary = pd.DataFrame(index=processed_df.index)

    # Combine the processed parts into one DataFrame.
    final_df = pd.concat([continuous, binary, dummies], axis=1)

    return final_df


def create_baseline_tensors(input_dir, output_dir, trajectory_metadata_path):
    """
    Load, merge, preprocess, and save static baseline features for each patient.
    """
    print("\nCreating baseline feature tensors...")

    # Load trajectory metadata to get the list of hadm_id's we actually have trajectories for.
    with open(trajectory_metadata_path, 'rb') as f:
        trajectory_data = pickle.load(f)
    valid_hadm_ids = list(trajectory_data['trajectories'].keys())
    valid_hadm_ids = [int(s.split('_')[0]) if '_' in s else int(s) for s in valid_hadm_ids]
    print(f"Found {len(valid_hadm_ids)} patients with trajectories.")

    # Load raw MIMIC data tables from the 'hosp' module
    patients_path = os.path.join(input_dir, 'PATIENTS.csv')
    admission_path = os.path.join(input_dir, 'ADMISSIONS.csv')
    transfers_path = os.path.join(input_dir, 'TRANSFERS.csv')

    try:
        patients_df = pd.read_csv(patients_path)
        admission_df = pd.read_csv(admission_path)
        transfers_df = pd.read_csv(transfers_path)
    except FileNotFoundError as e:
        print(f"Error: Could not find a required MIMIC 'hosp' file. Make sure your input directory is correct.")
        print(f"File not found: {e.filename}")
        return

    # Merge admissions with patients to get age and gender
    merged_df = pd.merge(admission_df, patients_df, on='SUBJECT_ID', how='left')

    # Calculate patient age at the time of admission
    merged_df['ADMITTIME'] = pd.to_datetime(merged_df['ADMITTIME'])
    merged_df['DOB'] = pd.to_datetime(merged_df['DOB'], errors='coerce')
    prov_age =  merged_df['ADMITTIME'].dt.year - merged_df['DOB'].dt.year
    merged_df['AGE'] = prov_age.where(prov_age < 300, 89)

    # Merge with transfers to get discharge time
    merged_df = pd.merge(merged_df, transfers_df, on=['HADM_ID', 'SUBJECT_ID'], how='left')

    # Calculate admission duration in days
    merged_df['DISCHTIME'] = pd.to_datetime(merged_df['DISCHTIME'])
    merged_df['ADMIT_DURATION'] = (merged_df['DISCHTIME'] - merged_df['ADMITTIME']).dt.total_seconds() / (3600 * 24)


    # Filter by the valid hadm_ids that have trajectories
    merged_df_final_filtered = merged_df[merged_df['HADM_ID'].isin(valid_hadm_ids)]


    # Apply additional filters as requested
    merged_with_disch_df_final_filtered = merged_df_final_filtered[merged_df_final_filtered['ADMIT_DURATION'] <= 10]
    merged_with_disch_df_final_filtered = merged_with_disch_df_final_filtered[
        merged_with_disch_df_final_filtered['ADMIT_DURATION'] >= 2]

    print(f"Found {len(merged_with_disch_df_final_filtered)} patients after duration filtering (2-10 days).")
    print("Available columns in merged data:", merged_df_final_filtered.columns.tolist())

    # Define feature lists for preprocessing
    categorical_features = ['GENDER', 'MARITAL_STATUS',
                            'INSURANCE', 'ADMISSION_LOCATION', 'ADMISSION_TYPE']
    continuous_features = ['AGE', 'ADMIT_DURATION']
    binary_features = []  # None specified in example

    # Preprocess the baseline data
    processed_baseline_df = preprocess_baseline_values(
        merged_with_disch_df_final_filtered,
        id_col='HADM_ID',
        categorical_features=categorical_features,
        continuous_features=continuous_features,
        binary_features=binary_features,
        drop_first=True
    )

    # Save a tensor for each patient
    baseline_dir = Path(output_dir) / "baseline_tensors"
    baseline_dir.mkdir(parents=True, exist_ok=True)

    saved_count = 0
    for hadm_id, row in processed_baseline_df.iterrows():
        tensor = torch.tensor(row.values, dtype=torch.float32)
        torch.save(tensor, baseline_dir / f"baseline_{hadm_id}.pt")
        saved_count += 1

    print(f"Saved {saved_count} baseline tensors to {baseline_dir}")

    # Save metadata for the dataloader and model
    baseline_metadata = {
        'feature_names': list(processed_baseline_df.columns),
        'feature_dim': len(processed_baseline_df.columns)
    }
    metadata_file = baseline_dir / "baseline_metadata.pkl"
    with open(metadata_file, "wb") as f:
        pickle.dump(baseline_metadata, f)

    print(f"Saved baseline metadata to {metadata_file}")
    print(f"Baseline feature dimension: {baseline_metadata['feature_dim']}")




# Example usage
if __name__ == "__main__":
    # Essential data processing
    waveforms_path = "../../../data/mimic_3_data/processed_data/combined_waveforms_cleaned_smooth.parquet"
    med_metadata_path = "../../../data/mimic_3_data/processed_data/med_tensors_output/med_tensors_metadata.pkl"
    med_data_path = "../../../data/mimic_3_data/processed_data/mv_filtered_10min.parquet"
    output_dir = "../../../data/mimic_3_data/processed_data/context_tensors_output"

    if all(os.path.exists(p) for p in [waveforms_path, med_metadata_path, med_data_path]):

        # Create context tensors
        """context_metadata = create_context_tensors(
            waveforms_parquet_path=waveforms_path,
            med_tensors_metadata_path=med_metadata_path,
            med_data_parquet_path=med_data_path,
            output_dir=output_dir,
            context_duration_minutes=60,  # 1 hour before t₀
            context_interval_minutes=10  # 10-minute intervals (6 total)
        )

        # Inspect sample tensors
        inspect_context_tensors(context_metadata, n_samples=3)"""

        create_baseline_tensors(input_dir="../../../data/mimic_3_data/input_data", output_dir=output_dir, trajectory_metadata_path="../../../data/mimic_3_data/processed_data/med_tensors_output/med_tensors_metadata.pkl")

    else:
        for p in [waveforms_path, med_metadata_path, med_data_path]:
            if not os.path.exists(p):
                print(p)
        print(f"Please ensure the following files exist:")
        print(f"  - Waveforms: {waveforms_path}")
        print(f"  - Med metadata: {med_metadata_path}")
        print(f"  - Med data: {med_data_path}")