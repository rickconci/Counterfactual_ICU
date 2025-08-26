import pandas as pd
import numpy as np
import torch
from pathlib import Path
import pickle
from datetime import datetime, timedelta
from tqdm import tqdm
from typing import Dict, List, Tuple, Optional
import os


def load_and_prepare_data(parquet_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load the mv_filtered_10min.parquet file and prepare it for processing.
    Returns both the full dataset and the filtered dataset for trajectory identification.
    """
    print(f"Loading data from {parquet_path}...")
    df_full = pd.read_parquet(parquet_path)

    # Ensure start_time is datetime
    if not pd.api.types.is_datetime64_any_dtype(df_full['start_time']):
        df_full['start_time'] = pd.to_datetime(df_full['start_time'])

    print(f"Loaded {len(df_full)} total medication rows")
    print(f"Unique hadm_ids: {df_full['hadm_id'].nunique()}")
    print(f"Unique item_labels: {df_full['item_label'].nunique()}")

    # Create filtered dataset for trajectory identification (only non-NaN action_cluster_id)
    df_triggers = df_full.dropna(subset=['action_cluster_id'])

    print(f"Rows with action_cluster_id (for trajectory identification): {len(df_triggers)}")
    print(f"Unique action_cluster_ids: {df_triggers['action_cluster_id'].nunique()}")

    return df_full, df_triggers


def identify_trajectories(df_full: pd.DataFrame, df_triggers: pd.DataFrame, trajectory_duration_minutes: int = 20) -> Dict:
    """
    Group by hadm_id and action_cluster_id to identify trajectories.
    Each trajectory lasts trajectory_duration_minutes OR until the start of the next action_cluster_id.
    """
    print("Identifying trajectories and calculating trajectory windows...")

    trajectories = {}

    # First, get all action_cluster_ids per hadm_id with their t0 times
    action_starts = df_triggers.groupby(['hadm_id', 'action_cluster_id'])['start_time'].min().reset_index()
    action_starts.columns = ['hadm_id', 'action_cluster_id', 't0_time']

    # Sort by hadm_id and t0_time to find next action starts
    action_starts = action_starts.sort_values(['hadm_id', 't0_time'])

    print(f"Found {len(action_starts)} unique action_cluster_ids across all patients")

    # For each hadm_id, determine trajectory end times
    for hadm_id in tqdm(action_starts['hadm_id'].unique(), desc="Processing patients"):
        patient_actions = action_starts[action_starts['hadm_id'] == hadm_id].copy()

        for idx, row in patient_actions.iterrows():
            action_cluster_id = row['action_cluster_id']
            t0_time = row['t0_time']

            # Calculate trajectory end time
            # Option 1: configurable minutes after t0
            end_time_20min = t0_time + pd.Timedelta(minutes=trajectory_duration_minutes)

            # Option 2: Start of next action_cluster_id
            next_actions = patient_actions[patient_actions['t0_time'] > t0_time]
            if len(next_actions) > 0:
                next_action_start = next_actions['t0_time'].min()
                trajectory_end_time = min(end_time_20min, next_action_start)
            else:
                trajectory_end_time = end_time_20min

            # Get all data for this patient within the trajectory window
            patient_data = df_full[df_full['hadm_id'] == hadm_id].copy()
            trajectory_data = patient_data[
                (patient_data['end_time'] > t0_time) &  # Still ongoing at/after t0
                (patient_data['start_time'] < trajectory_end_time)  # Starts before trajectory ends
                ]

            # Store trajectory info
            traj_key = f"{hadm_id}_{int(action_cluster_id)}"
            trajectories[traj_key] = {
                'hadm_id': hadm_id,
                'action_cluster_id': action_cluster_id,
                't0_time': t0_time,
                'trajectory_end_time': trajectory_end_time,
                'duration_minutes': (trajectory_end_time - t0_time).total_seconds() / 60,
                'data': trajectory_data
            }

    print(f"Created {len(trajectories)} trajectories")

    # Print some statistics about trajectory durations
    durations = [t['duration_minutes'] for t in trajectories.values()]
    print(f"Trajectory duration statistics:")
    print(f"  Mean: {np.mean(durations):.2f} minutes")
    print(f"  Std: {np.std(durations):.2f} minutes")
    print(f"  Min: {np.min(durations):.2f} minutes")
    print(f"  Max: {np.max(durations):.2f} minutes")
    print(f"  Full {trajectory_duration_minutes}min trajectories: {sum(1 for d in durations if d >= (trajectory_duration_minutes - 0.1))}")
    print(f"  Truncated trajectories: {sum(1 for d in durations if d < (trajectory_duration_minutes - 0.1))}")

    return trajectories


def create_time_grid(trajectories: Dict, interval_seconds: int = 10, trajectory_duration_minutes: int = 20) -> Dict:
    """
    Create time grid parameters using the configured maximum forward duration.
    """
    print("Creating time grid parameters...")

    # Maximum trajectory duration is configurable (default 20 minutes)
    max_duration_seconds = trajectory_duration_minutes * 60
    n_intervals = int(np.ceil(max_duration_seconds / interval_seconds))

    # Verify this covers all trajectories
    actual_max_duration = max(traj['duration_minutes'] for traj in trajectories.values()) * 60

    print(f"Fixed trajectory duration: {max_duration_seconds} seconds ({trajectory_duration_minutes} minutes)")
    print(f"Actual max trajectory duration: {actual_max_duration:.1f} seconds")
    print(f"Time grid will have {n_intervals} intervals of {interval_seconds} seconds each")

    return {
        'n_intervals': n_intervals,
        'interval_seconds': interval_seconds,
        'max_duration': max_duration_seconds
    }


def process_single_trajectory(
        traj_key: str,
        traj_info: Dict,
        item_labels: List[str],
        n_intervals: int,
        interval_seconds: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Process a single trajectory and create medication tensor.
    Only includes data within the trajectory window (t0 to trajectory_end_time).

    Returns:
        values_array: (n_intervals, n_medications) array of medication rates
        mask_array: (n_intervals, n_medications) array indicating data presence
    """
    data = traj_info['data']
    t0_time = traj_info['t0_time']
    trajectory_end_time = traj_info['trajectory_end_time']

    # Initialize arrays
    n_medications = len(item_labels)
    values_array = np.zeros((n_intervals, n_medications), dtype=np.float32)
    mask_array = np.zeros((n_intervals, n_medications), dtype=np.float32)

    # Create item_label to index mapping
    item_to_idx = {item: idx for idx, item in enumerate(item_labels)}

    # Calculate the trajectory duration in intervals
    trajectory_duration_seconds = (trajectory_end_time - t0_time).total_seconds()
    max_interval_for_trajectory = min(n_intervals, int(np.ceil(trajectory_duration_seconds / interval_seconds)))

    # Process each medication type separately to handle overlaps correctly
    for item_label in item_labels:
        if item_label not in item_to_idx:
            continue

        item_idx = item_to_idx[item_label]

        # Get all infusions for this medication, sorted by start_time
        item_infusions = data[data['item_label'] == item_label].sort_values('start_time')

        for _, row in item_infusions.iterrows():
            # Get rate value
            rate_value = 0.0
            for rate_col in ['rate/weight_normalized']:
                if rate_col in row and pd.notna(row[rate_col]):
                    rate_value = float(row[rate_col])
                    break

            if rate_value == 0.0:
                continue

            # Clip infusion to trajectory window
            effective_start = max(t0_time, row['start_time'])
            effective_end = min(trajectory_end_time, row['end_time'])

            # Convert to intervals (round down start, round up end)
            start_seconds = (effective_start - t0_time).total_seconds()
            end_seconds = (effective_end - t0_time).total_seconds()

            start_idx = int(start_seconds // interval_seconds)  # Round down
            end_idx = int(np.ceil(end_seconds / interval_seconds))  # Round up

            # Fill all intervals in range (most recent wins due to sort order)
            for time_idx in range(start_idx, end_idx):
                if 0 <= time_idx < max_interval_for_trajectory:
                    values_array[time_idx, item_idx] = rate_value
                    mask_array[time_idx, item_idx] = 1.0

    return values_array, mask_array


def save_trajectory_tensor(
        traj_key: str,
        values_array: np.ndarray,
        mask_array: np.ndarray,
        n_intervals: int,
        interval_seconds: int,
        output_dir: Path
) -> str:
    """
    Save a trajectory tensor to disk.
    """
    # Create time arrays
    time_seconds = np.arange(n_intervals) * interval_seconds
    time_hours = time_seconds / 3600.0

    # Convert to tensors
    values_tensor = torch.from_numpy(values_array).float()
    mask_tensor = torch.from_numpy(mask_array).float()
    time_seconds_tensor = torch.from_numpy(time_seconds).float()
    time_hours_tensor = torch.from_numpy(time_hours).float()

    # Create filename
    filename = f"med_tensor_{traj_key}.pt"
    filepath = output_dir / filename

    # Save tensor
    torch.save(
        (values_tensor, mask_tensor, time_seconds_tensor, time_hours_tensor, n_intervals),
        filepath
    )

    return str(filepath)


def create_med_tensors_from_parquet(
        parquet_path: str,
        output_dir: str = "./med_tensors_output",
        interval_seconds: int = 10,
        trajectory_duration_minutes: int = 20,
        n_workers: int = 1
) -> Dict:
    """
    Main function to create med_tensors from mv_filtered_10min.parquet file.

    Args:
        parquet_path: Path to the mv_filtered_10min.parquet file
        output_dir: Directory to save the tensor files
        interval_seconds: Time interval in seconds (default: 10)
        n_workers: Number of parallel workers (not implemented yet, placeholder)

    Returns:
        Dictionary with metadata about created tensors
    """

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Step 1: Load and prepare data
    df_full, df_triggers = load_and_prepare_data(parquet_path)

    # Step 2: Identify trajectories using trigger data, but get all meds from full data
    trajectories = identify_trajectories(
        df_full,
        df_triggers,
        trajectory_duration_minutes=trajectory_duration_minutes,
    )

    # Step 3: Get unique item labels from FULL dataset (all medications)
    unique_item_labels = sorted(df_full['item_label'].unique().tolist())
    print(f"Found {len(unique_item_labels)} unique item labels:")
    for i, label in enumerate(unique_item_labels):
        print(f"  {i}: {label}")

    # Step 4: Create time grid parameters respecting configurable forward duration
    grid_params = create_time_grid(
        trajectories,
        interval_seconds,
        trajectory_duration_minutes=trajectory_duration_minutes,
    )
    n_intervals = grid_params['n_intervals']

    # Step 5: Process each trajectory
    print(f"\nProcessing {len(trajectories)} trajectories...")

    trajectory_metadata = {}
    saved_files = []

    for traj_key, traj_info in tqdm(trajectories.items(), desc="Creating tensors"):
        # Process trajectory
        values_array, mask_array = process_single_trajectory(
            traj_key=traj_key,
            traj_info=traj_info,
            item_labels=unique_item_labels,
            n_intervals=n_intervals,
            interval_seconds=interval_seconds
        )

        # Save tensor
        filepath = save_trajectory_tensor(
            traj_key=traj_key,
            values_array=values_array,
            mask_array=mask_array,
            n_intervals=n_intervals,
            interval_seconds=interval_seconds,
            output_dir=output_path
        )

        # Store metadata
        trajectory_metadata[traj_key] = {
            'hadm_id': traj_info['hadm_id'],
            'action_cluster_id': traj_info['action_cluster_id'],
            't0_time': traj_info['t0_time'],
            'n_intervals': n_intervals,
            'interval_seconds': interval_seconds,
            'trajectory_end_time': traj_info['trajectory_end_time'],
            'duration_minutes': traj_info['duration_minutes'],
            'n_medications': len(unique_item_labels),
            'file_path': filepath,
            'has_data': np.any(mask_array > 0),
            'total_nonzero_values': int(np.sum(mask_array))
        }

        saved_files.append(filepath)

    # Step 6: Save metadata
    metadata = {
        'trajectories': trajectory_metadata,
        'item_labels': unique_item_labels,
        'n_intervals': n_intervals,
        'interval_seconds': interval_seconds,
        'total_trajectories': len(trajectories),
        'created_at': datetime.now().isoformat(),
        'source_file': parquet_path
    }

    metadata_file = output_path / "med_tensors_metadata.pkl"
    with open(metadata_file, 'wb') as f:
        pickle.dump(metadata, f)

    # Step 7: Save summary statistics
    summary_stats = {
        'total_trajectories': len(trajectories),
        'total_tensor_files': len(saved_files),
        'unique_hadm_ids': df_triggers['hadm_id'].nunique(),
        'unique_action_cluster_ids': df_triggers['action_cluster_id'].nunique(),
        'unique_item_labels': len(unique_item_labels),
        'time_grid_intervals': n_intervals,
        'interval_seconds': interval_seconds,
        'max_duration_hours': grid_params['max_duration'] / 3600.0
    }

    print(f"\n=== Summary ===")
    for key, value in summary_stats.items():
        print(f"{key}: {value}")

    print(f"\nTensor files saved to: {output_path}")
    print(f"Metadata saved to: {metadata_file}")

    return metadata


def load_and_inspect_tensor(tensor_path: str) -> None:
    """
    Load and inspect a saved tensor file.
    """
    print(f"Loading tensor from: {tensor_path}")

    values_tensor, mask_tensor, time_seconds_tensor, time_hours_tensor, n_intervals = torch.load(tensor_path)

    print(f"Values tensor shape: {values_tensor.shape}")
    print(f"Mask tensor shape: {mask_tensor.shape}")
    print(f"Time intervals: {n_intervals}")
    print(f"Duration: {time_hours_tensor[-1]:.2f} hours")
    print(f"Non-zero values: {torch.sum(mask_tensor).item()}")
    print(f"Medications with data: {torch.sum(torch.any(mask_tensor > 0, dim=0)).item()}")


# Example usage
if __name__ == "__main__":
    # Example usage - adjust paths as needed
    parquet_path = "mv_filtered_10min.parquet"
    output_dir = "data/med_tensors_output"

    if os.path.exists(parquet_path):
        metadata = create_med_tensors_from_parquet(
            parquet_path=parquet_path,
            output_dir=output_dir,
            interval_seconds=10
        )

        # Inspect a sample tensor
        if metadata['trajectories']:
            sample_traj_key = list(metadata['trajectories'].keys())[0]
            sample_file = metadata['trajectories'][sample_traj_key]['file_path']
            print(f"\n=== Sample Tensor Inspection ===")
            load_and_inspect_tensor(sample_file)
    else:
        print(f"Please ensure {parquet_path} exists in the current directory")
        print("You can modify the parquet_path variable to point to your file location")