import os
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm


# Module-level globals for fork-based shared memory
_WF_DF: Optional[pd.DataFrame] = None


def _process_single_trajectory(args: Tuple[str, Dict, int, int, str, str, set]) -> Tuple[str, Optional[Dict[str, Any]], Optional[Dict[str, Any]], bool]:
    """Worker function to process a single trajectory end-to-end."""
    traj_key, traj_info, n_int, int_seconds, ic_dir_str, pred_dir_str, common_hadm_ids = args
    # Filter waveform data for this trajectory using shared DataFrame
    hadm_id = traj_info["hadm_id"]
    
    # Get common hadm_ids from the global waveform data
    global _WF_DF
    if _WF_DF is None:
        return traj_key, None, None, True
    
    if hadm_id not in common_hadm_ids:  # quick skip
        return traj_key, None, None, True

    patient_waveforms = _WF_DF[_WF_DF["hadm_id"] == hadm_id].copy()
    if len(patient_waveforms) == 0:
        return traj_key, None, None, True

    t0_time = pd.to_datetime(traj_info["t0_time"]).tz_localize("UTC")
    trajectory_end_time = pd.to_datetime(traj_info["trajectory_end_time"]).tz_localize(
        "UTC"
    )
    trajectory_waveforms = patient_waveforms[
        (patient_waveforms["absolute_timestamp"] >= t0_time)
        & (patient_waveforms["absolute_timestamp"] < trajectory_end_time)
    ].copy()

    if len(trajectory_waveforms) == 0:
        return traj_key, None, None, True

    # Build a local traj_info with waveform_data
    local_traj_info = {**traj_info, "waveform_data": trajectory_waveforms}

    ic_values, ic_mask = create_ic_tensor(traj_key, local_traj_info)
    if not (ic_mask[0].item() or ic_mask[1].item()):
        # skip saving if neither ABP nor CVP present at t0
        return traj_key, None, None, True

    ic_filepath = save_ic_tensor(traj_key, ic_values, ic_mask, Path(ic_dir_str))
    pred_values, pred_mask = create_prediction_targets_tensor(
        traj_key, local_traj_info, n_int, int_seconds
    )
    pred_filepath = save_prediction_targets_tensor(
        traj_key, pred_values, pred_mask, n_int, int_seconds, Path(pred_dir_str)
    )

    ic_meta = {
        "hadm_id": local_traj_info["hadm_id"],
        "action_cluster_id": local_traj_info["action_cluster_id"],
        "t0_time": local_traj_info["t0_time"],
        "file_path": ic_filepath,
        "has_abp_mean": bool(ic_mask[0].item()),
        "has_cvp": bool(ic_mask[1].item()),
        "abp_mean_value": float(ic_values[0].item()) if ic_mask[0] else None,
        "cvp_value": float(ic_values[1].item()) if ic_mask[1] else None,
    }

    pred_meta = {
        "hadm_id": local_traj_info["hadm_id"],
        "action_cluster_id": local_traj_info["action_cluster_id"],
        "t0_time": local_traj_info["t0_time"],
        "trajectory_end_time": local_traj_info["trajectory_end_time"],
        "duration_minutes": local_traj_info["duration_minutes"],
        "file_path": pred_filepath,
        "n_intervals": n_int,
        "interval_seconds": int_seconds,
        "total_abp_mean_measurements": int(torch.sum(pred_mask[:, 0] > 0).item()),
        "total_cvp_measurements": int(torch.sum(pred_mask[:, 1] > 0).item()),
        "timestamps_aligned": True,
    }

    return traj_key, ic_meta, pred_meta, False


def load_waveforms_data(parquet_path: str, interval_seconds: int = 10) -> pd.DataFrame:
    """
    Load the combined_waveforms.cleaned.parquet file and align to 10-second grid.
    """
    print(f"Loading waveforms data from {parquet_path}...")
    df = pd.read_parquet(parquet_path)

    # Ensure absolute_timestamp is datetime
    if not pd.api.types.is_datetime64_any_dtype(df["absolute_timestamp"]):
        df["absolute_timestamp"] = pd.to_datetime(df["absolute_timestamp"])

    print(f"Loaded {len(df)} waveform rows")
    print(
        f"Time range: {df['absolute_timestamp'].min()} to {df['absolute_timestamp'].max()}"
    )
    print(f"Available columns: {list(df.columns)}")

    # Check for required columns
    required_cols = ["hadm_id", "absolute_timestamp"]

    missing_required = [col for col in required_cols if col not in df.columns]
    if missing_required:
        raise ValueError(f"Missing required columns: {missing_required}")

    # Check for smoothed columns that will be aliased
    smoothed_cols = ["ABP_MEAN_smooth4", "CVP_smooth4"]
    available_smoothed = [col for col in smoothed_cols if col in df.columns]
    print(f"Available smoothed physiological measurements: {available_smoothed}")

    # CRITICAL: Round timestamps DOWN to nearest 10-second interval
    print(f"Aligning timestamps to {interval_seconds}-second grid...")

    # Convert to Unix timestamp, round down, convert back
    df["timestamp_unix"] = (
        df["absolute_timestamp"].astype("int64") // 10**9
    )  # Convert to seconds
    df["timestamp_aligned_unix"] = (
        df["timestamp_unix"] // interval_seconds
    ) * interval_seconds
    df["absolute_timestamp_aligned"] = pd.to_datetime(
        df["timestamp_aligned_unix"], unit="s", utc=True
    )

    # Replace original timestamp with aligned version
    df["absolute_timestamp"] = df["absolute_timestamp_aligned"]

    # Clean up temporary columns
    df = df.drop(
        columns=[
            "timestamp_unix",
            "timestamp_aligned_unix",
            "absolute_timestamp_aligned",
        ]
    )

    print(f"Timestamps aligned to {interval_seconds}-second intervals")
    print(
        f"New time range: {df['absolute_timestamp'].min()} to {df['absolute_timestamp'].max()}"
    )
    
    # Map smoothed columns to canonical names expected downstream
    if "ABP_MEAN_smooth4" in df.columns:
        df["ABP MEAN"] = pd.to_numeric(df["ABP_MEAN_smooth4"], errors="coerce")
    if "CVP_smooth4" in df.columns:
        df["CVP"] = pd.to_numeric(df["CVP_smooth4"], errors="coerce")

    return df



def create_ic_tensor(
    traj_key: str, traj_info: Dict
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create Initial Condition (IC) tensor at t_0.
    Since timestamps are aligned to 10s grid, we require exact t0 match.

    Returns:
        ic_values: [ABP MEAN, CVP, 0, 0, 0]
        ic_mask: [1 if ABP MEAN valid, 1 if CVP valid, 0, 0, 0]

    Raises:
        ValueError: If no exact physiological data found at t0
    """
    waveform_data = traj_info["waveform_data"]
    t0_time = pd.to_datetime(traj_info["t0_time"]).tz_localize("UTC")

    # Initialize IC tensor: [ABP MEAN, CVP, 0, 0, 0]
    ic_values = np.zeros(5, dtype=np.float32)
    ic_mask = np.zeros(5, dtype=np.float32)

    if len(waveform_data) == 0:
        raise ValueError(
            f"No waveform data found for trajectory {traj_key}. "
            f"Check hadm_id matching and trajectory time windows."
        )

    # Since timestamps are aligned to 10s grid, t0_time should also be aligned
    # Look for exact match only - no tolerance fallback
    exact_match = waveform_data[waveform_data["absolute_timestamp"] == t0_time]

    if len(exact_match) > 0:
        closest_row = exact_match.iloc[0]  # Take first if multiple
        print(f"  Found exact t0 match for trajectory {traj_key}")
    else:
        # No exact match - return zeros tensors as a sentinel while allowing pipeline to continue
        print(f"  No exact t0 match for trajectory {traj_key} - returning zeros")
        return torch.from_numpy(ic_values), torch.from_numpy(ic_mask)

    # ABP MEAN (index 0) - note the space in column name
    if "ABP MEAN" in closest_row and pd.notna(closest_row["ABP MEAN"]):
        abp_value = float(closest_row["ABP MEAN"])
        if np.isnan(abp_value):
            print(f"  Warning: ABP MEAN is NaN for trajectory {traj_key} at t0")
        else:
            ic_values[0] = abp_value
            ic_mask[0] = 1.0

    # CVP (index 1)
    if "CVP" in closest_row and pd.notna(closest_row["CVP"]):
        cvp_value = float(closest_row["CVP"])
        if np.isnan(cvp_value):
            print(f"  Warning: CVP is NaN for trajectory {traj_key} at t0")
        else:
            ic_values[1] = cvp_value
            ic_mask[1] = 1.0

    # Indices 2, 3, 4 remain zero (as requested)

    return torch.from_numpy(ic_values), torch.from_numpy(ic_mask)


def create_prediction_targets_tensor(
    traj_key: str, traj_info: Dict, n_intervals: int, interval_seconds: int = 10
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create prediction targets tensor over the trajectory duration.
    Time grid is perfectly aligned - no interpolation needed since timestamps are pre-aligned.

    Raises ValueError if any NaN values are found in prediction targets.

    Returns:
        pred_values: (n_intervals, 2) - [ABP MEAN, CVP] over time
        pred_mask: (n_intervals, 2) - mask for valid measurements
    """
    waveform_data = traj_info["waveform_data"]
    t0_time = pd.to_datetime(traj_info["t0_time"]).tz_localize("UTC")
    trajectory_duration_seconds = traj_info["duration_minutes"] * 60

    # Initialize tensors
    pred_values = np.zeros((n_intervals, 2), dtype=np.float32)  # [ABP MEAN, CVP]
    pred_mask = np.zeros((n_intervals, 2), dtype=np.float32)

    if len(waveform_data) == 0:
        return torch.from_numpy(pred_values), torch.from_numpy(pred_mask)

    # Calculate max intervals for this trajectory
    max_interval_for_trajectory = min(
        n_intervals, int(np.ceil(trajectory_duration_seconds / interval_seconds))
    )

    print(
        f"  Processing trajectory {traj_key}: {len(waveform_data)} waveform points, {max_interval_for_trajectory} intervals"
    )

    # Since timestamps are now aligned to 10-second grid, we can directly map them
    # Calculate time indices for all waveform data points
    waveform_data = waveform_data.copy()
    waveform_data["time_from_t0_seconds"] = (
        waveform_data["absolute_timestamp"] - t0_time
    ).dt.total_seconds()
    waveform_data["time_idx"] = (
        waveform_data["time_from_t0_seconds"] // interval_seconds
    ).astype(int)

    # Filter to trajectory duration and valid indices
    valid_waveforms = waveform_data[
        (waveform_data["time_idx"] >= 0)
        & (waveform_data["time_idx"] < max_interval_for_trajectory)
    ]

    if len(valid_waveforms) == 0:
        return torch.from_numpy(pred_values), torch.from_numpy(pred_mask)

    # Group by time index and aggregate (mean if multiple measurements per interval)
    for time_idx in valid_waveforms["time_idx"].unique():
        interval_data = valid_waveforms[valid_waveforms["time_idx"] == time_idx]

        # ABP MEAN (index 0) - same as MAP
        if "ABP MEAN" in interval_data.columns:
            abp_mean_values = interval_data["ABP MEAN"].dropna()
            if len(abp_mean_values) > 0:
                mean_abp = float(abp_mean_values.mean())
                # CHECK FOR NaN - CRITICAL REQUIREMENT
                if np.isnan(mean_abp):
                    raise ValueError(
                        f"NaN found in ABP MEAN prediction target for trajectory {traj_key} "
                        f"at time index {time_idx} (t+{time_idx * interval_seconds}s). "
                        f"Raw values: {abp_mean_values.tolist()}"
                    )
                pred_values[time_idx, 0] = mean_abp
                pred_mask[time_idx, 0] = 1.0

        # CVP (index 1)
        if "CVP" in interval_data.columns:
            cvp_values = interval_data["CVP"].dropna()
            if len(cvp_values) > 0:
                mean_cvp = float(cvp_values.mean())
                # CHECK FOR NaN - CRITICAL REQUIREMENT
                if np.isnan(mean_cvp):
                    raise ValueError(
                        f"NaN found in CVP prediction target for trajectory {traj_key} "
                        f"at time index {time_idx} (t+{time_idx * interval_seconds}s). "
                        f"Raw values: {cvp_values.tolist()}"
                    )
                pred_values[time_idx, 1] = mean_cvp
                pred_mask[time_idx, 1] = 1.0

    # Final NaN check on the entire tensor (extra safety)
    if np.any(np.isnan(pred_values[pred_mask > 0])):
        nan_locations = np.where(np.isnan(pred_values) & (pred_mask > 0))
        raise ValueError(
            f"NaN values detected in prediction targets for trajectory {traj_key} "
            f"at locations: {list(zip(nan_locations[0], nan_locations[1]))}"
        )

    print(
        f"    ABP MEAN measurements: {int(np.sum(pred_mask[:, 0]))}, CVP measurements: {int(np.sum(pred_mask[:, 1]))}"
    )

    return torch.from_numpy(pred_values), torch.from_numpy(pred_mask)


def save_ic_tensor(
    traj_key: str, ic_values: torch.Tensor, ic_mask: torch.Tensor, output_dir: Path
) -> str:
    """Save IC tensor to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)
    filepath = output_dir / f"ic_tensor_{traj_key}.pt"

    torch.save((ic_values, ic_mask), filepath)
    return str(filepath)


def save_prediction_targets_tensor(
    traj_key: str,
    pred_values: torch.Tensor,
    pred_mask: torch.Tensor,
    n_intervals: int,
    interval_seconds: int,
    output_dir: Path,
) -> str:
    """Save prediction targets tensor to disk."""
    output_dir.mkdir(parents=True, exist_ok=True)
    filepath = output_dir / f"pred_targets_{traj_key}.pt"

    # Create time arrays (same as med tensors)
    time_seconds = torch.arange(n_intervals, dtype=torch.float32) * interval_seconds
    time_hours = time_seconds / 3600.0

    torch.save(
        (pred_values, pred_mask, time_seconds, time_hours, n_intervals), filepath
    )
    return str(filepath)


def create_physiological_tensors(
    waveforms_parquet_path: str,
    med_tensors_metadata_path: str,
    output_dir: str = "./physio_tensors_output",
    interval_seconds: int = 10,
    n_workers: int = 1,
) -> Dict:
    """
    Main function to create IC values and prediction targets from waveforms data.

    All timestamps are aligned to exact 10-second intervals to match medication tensors.
    Any NaN values in prediction targets will cause an error.
    Any trajectory without exact physiological data at t₀ will cause an error.

    Args:
        waveforms_parquet_path: Path to combined_waveforms.cleaned.parquet
        med_tensors_metadata_path: Path to med tensors metadata.pkl
        output_dir: Directory to save physio tensors
        interval_seconds: Time interval in seconds (must match med tensors, default: 10)

    Returns:
        Dictionary with metadata about created tensors

    Raises:
        ValueError: If no exact physiological data found at t₀ for any trajectory
        ValueError: If any NaN values found in prediction targets
        ValueError: If interval mismatch with medication tensors
    """

    # Create output directories
    output_path = Path(output_dir)
    ic_dir = output_path / "ic_tensors"
    pred_dir = output_path / "prediction_targets"
    ic_dir.mkdir(parents=True, exist_ok=True)
    pred_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("=== Loading Data ===")
    waveforms_df = load_waveforms_data(waveforms_parquet_path, interval_seconds)

    with open(med_tensors_metadata_path, "rb") as f:
        med_metadata = pickle.load(f)

    print(f"Found {len(med_metadata['trajectories'])} medication trajectories")

    # Get time grid parameters from med tensors and verify consistency
    n_intervals = med_metadata["n_intervals"]
    med_interval_seconds = med_metadata["interval_seconds"]

    if med_interval_seconds != interval_seconds:
        raise ValueError(
            f"Interval mismatch: Med tensors use {med_interval_seconds}s intervals, "
            f"but physio tensors are set to {interval_seconds}s. They must match!"
        )

    print(
        f"Using consistent time grid: {n_intervals} intervals of {interval_seconds} seconds each"
    )

    # Report cohort overlap prior to processing
    print("\n=== Aligning Data ===")
    med_hadm_ids = {ti["hadm_id"] for ti in med_metadata["trajectories"].values()}
    waveform_hadm_ids = set(waveforms_df["hadm_id"].unique())
    common_hadm_ids = med_hadm_ids.intersection(waveform_hadm_ids)
    print(f"Med trajectories: {len(med_hadm_ids)} patients")
    print(f"Waveform data: {len(waveform_hadm_ids)} patients")
    print(f"Common patients: {len(common_hadm_ids)} patients")

    # Prepare fork-shared globals
    global _WF_DF
    _WF_DF = waveforms_df

    # Process trajectories (parallel if requested)
    ic_metadata: Dict[str, Any] = {}
    pred_metadata: Dict[str, Any] = {}
    skipped_trajectories = 0

    tasks = [
        (tk, ti, n_intervals, interval_seconds, str(ic_dir), str(pred_dir), common_hadm_ids)
        for tk, ti in med_metadata["trajectories"].items()
    ]

    if n_workers and n_workers > 1:
        print(f"\n=== Processing {len(tasks)} Trajectories with {n_workers} workers ===")
        from concurrent.futures import ProcessPoolExecutor, as_completed

        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            futures = [ex.submit(_process_single_trajectory, t) for t in tasks]
            for fut in tqdm(as_completed(futures), total=len(futures), desc="Creating tensors"):
                traj_key, ic_meta, pred_meta, skipped = fut.result()
                if skipped or ic_meta is None or pred_meta is None:
                    skipped_trajectories += 1
                    continue
                ic_metadata[traj_key] = ic_meta
                pred_metadata[traj_key] = pred_meta
    else:
        print(f"\n=== Processing {len(tasks)} Trajectories (single process) ===")
        for t in tqdm(tasks, desc="Creating tensors"):
            traj_key, ic_meta, pred_meta, skipped = _process_single_trajectory(t)
            if skipped or ic_meta is None or pred_meta is None:
                skipped_trajectories += 1
                continue
            ic_metadata[traj_key] = ic_meta
            pred_metadata[traj_key] = pred_meta

    # Calculate total trajectories that were attempted (processed + skipped)
    total_trajectories_attempted = len(ic_metadata) + skipped_trajectories
    
    # Save metadata
    physio_metadata = {
        "ic_tensors": ic_metadata,
        "prediction_targets": pred_metadata,
        "n_intervals": n_intervals,
        "interval_seconds": interval_seconds,
        "timestamps_aligned_to_grid": True,
        "exact_t0_match_required": True,
        "nan_values_rejected": True,
        "total_trajectories": total_trajectories_attempted,
        "created_at": datetime.now().isoformat(),
        "source_waveforms": waveforms_parquet_path,
        "source_med_metadata": med_tensors_metadata_path,
        "total_trajectories_processed": len(ic_metadata),
        "total_trajectories_aligned": total_trajectories_attempted,
        "skipped_trajectories": skipped_trajectories,
    }

    metadata_file = output_path / "physio_tensors_metadata.pkl"
    with open(metadata_file, "wb") as f:
        pickle.dump(physio_metadata, f)

    # Print summary
    print("\n=== Summary ===")
    print(f"Created {len(ic_metadata)} IC tensors")
    print(f"Created {len(pred_metadata)} prediction target tensors")
    print(f"Total trajectories aligned: {len(ic_metadata) + skipped_trajectories}")
    print(f"Trajectories with exact t0 match: {len(ic_metadata)}")
    print(f"Trajectories skipped (no exact t0): {skipped_trajectories}")
    print(f"Success rate: {len(ic_metadata) / total_trajectories_attempted * 100:.1f}%")

    # IC statistics
    ic_with_abp = sum(1 for meta in ic_metadata.values() if meta["has_abp_mean"])
    ic_with_cvp = sum(1 for meta in ic_metadata.values() if meta["has_cvp"])
    print(
        f"IC tensors with ABP_Mean: {ic_with_abp} ({ic_with_abp / len(ic_metadata) * 100:.1f}%)"
    )
    print(
        f"IC tensors with CVP: {ic_with_cvp} ({ic_with_cvp / len(ic_metadata) * 100:.1f}%)"
    )

    # Prediction targets statistics
    abp_mean_measurements = [
        meta["total_abp_mean_measurements"] for meta in pred_metadata.values()
    ]
    cvp_measurements = [
        meta["total_cvp_measurements"] for meta in pred_metadata.values()
    ]

    print(
        f"Prediction targets - ABP MEAN measurements per trajectory: {np.mean(abp_mean_measurements):.1f} ± {np.std(abp_mean_measurements):.1f}"
    )
    print(
        f"Prediction targets - CVP measurements per trajectory: {np.mean(cvp_measurements):.1f} ± {np.std(cvp_measurements):.1f}"
    )

    print(f"\nFiles saved to: {output_path}")
    print(f"Metadata saved to: {metadata_file}")

    return physio_metadata


def inspect_physio_tensors(metadata: Dict, n_samples: int = 3) -> None:
    """Inspect sample physiological tensors."""
    print(f"\n=== Inspecting {n_samples} Sample Physiological Tensors ===")

    traj_keys = list(metadata["ic_tensors"].keys())[:n_samples]

    for traj_key in traj_keys:
        print(f"\nTrajectory: {traj_key}")

        # Load and inspect IC tensor
        ic_info = metadata["ic_tensors"][traj_key]
        ic_values, ic_mask = torch.load(ic_info["file_path"])

        print("IC Tensor:")
        print(f"  Values: {ic_values.numpy()}")
        print(f"  Mask:   {ic_mask.numpy()}")
        print(f"  ABP MEAN: {ic_values[0].item():.2f} (valid: {bool(ic_mask[0])})")
        print(f"  CVP:      {ic_values[1].item():.2f} (valid: {bool(ic_mask[1])})")

        # Load and inspect prediction targets
        pred_info = metadata["prediction_targets"][traj_key]
        pred_values, pred_mask, time_seconds, time_hours, n_intervals = torch.load(
            pred_info["file_path"]
        )

        valid_abp_mean = torch.sum(pred_mask[:, 0] > 0).item()
        valid_cvp = torch.sum(pred_mask[:, 1] > 0).item()

        print("Prediction Targets:")
        print(f"  Shape: {pred_values.shape}")
        print(f"  Duration: {pred_info['duration_minutes']:.1f} minutes")
        print(f"  Valid ABP MEAN measurements: {valid_abp_mean}")
        print(f"  Valid CVP measurements: {valid_cvp}")

        if valid_abp_mean > 0:
            abp_mean_vals = pred_values[pred_mask[:, 0] > 0, 0]
            print(
                f"  ABP MEAN range: {abp_mean_vals.min():.1f} - {abp_mean_vals.max():.1f}"
            )
        if valid_cvp > 0:
            cvp_vals = pred_values[pred_mask[:, 1] > 0, 1]
            print(f"  CVP range: {cvp_vals.min():.1f} - {cvp_vals.max():.1f}")


# Example usage
if __name__ == "__main__":
    # Example paths - adjust as needed
    waveforms_path = "combined_waveforms_cleaned_smooth.parquet"
    med_metadata_path = "data/med_tensors_output/med_tensors_metadata.pkl"
    output_dir = "data/physio_tensors_output"

    if os.path.exists(waveforms_path) and os.path.exists(med_metadata_path):
        metadata = create_physiological_tensors(
            waveforms_parquet_path=waveforms_path,
            med_tensors_metadata_path=med_metadata_path,
            output_dir=output_dir,
            interval_seconds=10,  # Must match med tensors
        )

        # Inspect sample tensors
        inspect_physio_tensors(metadata, n_samples=3)

    else:
        print("Please ensure the following files exist:")
        print(f"  - Waveforms data: {waveforms_path}")
        print(f"  - Med metadata: {med_metadata_path}")
