import os
import pickle
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# Module-level globals for fork-based shared memory
_WF_DF: Optional[pd.DataFrame] = None
_MED_DF: Optional[pd.DataFrame] = None
_ITEM_LABELS: Optional[List[str]] = None


def _process_single_context(
    traj_key: str,
    traj_info: Dict,
    n_context_intervals: int,
    context_interval_minutes: int,
    rd_time_hours_from_t0_list: List[float],
    raindrop_context_dir: str,
) -> Tuple[str, Dict]:
    # Use preloaded globals (available via fork copy-on-write)
    physio_values, physio_mask = create_physio_context_tensor(
        traj_key, traj_info, _WF_DF, n_context_intervals, context_interval_minutes
    )

    meds_values, meds_mask = create_meds_context_tensor(
        traj_key,
        traj_info,
        _MED_DF,
        n_context_intervals,
        context_interval_minutes,
        _ITEM_LABELS or [],
    )

    # Raindrop-ready: concatenate values and invert presence mask
    rd_values = torch.cat(
        [physio_values.to(torch.float32), meds_values.to(torch.float32)], dim=1
    )
    present_mask = torch.cat(
        [physio_mask.to(torch.float32), meds_mask.to(torch.float32)], dim=1
    )
    rd_missing_mask = 1.0 - torch.clamp(present_mask, 0.0, 1.0)
    rd_src = torch.cat([rd_values, rd_missing_mask], dim=1)
    rd_length = int(n_context_intervals)
    rd_filepath = Path(raindrop_context_dir) / f"rd_context_{traj_key}.pt"
    torch.save(
        (
            rd_src,
            torch.tensor(rd_time_hours_from_t0_list, dtype=torch.float32),
            rd_length,
        ),
        rd_filepath,
    )
    rd_meta = {
        "hadm_id": traj_info["hadm_id"],
        "action_cluster_id": traj_info["action_cluster_id"],
        "t0_time": traj_info["t0_time"],
        "file_path": str(rd_filepath),
        "n_intervals": n_context_intervals,
        "interval_minutes": context_interval_minutes,
        "d_inp": int(rd_values.shape[1]),
    }
    return traj_key, rd_meta


def create_context_tensors(
    waveforms_parquet_path: str,
    med_tensors_metadata_path: str,
    med_data_parquet_path: str,
    output_dir: str = "./context_tensors_output",
    context_duration_minutes: int = 60,
    context_interval_minutes: int = 10,
    n_workers: int = 1,
) -> Dict:
    """
    Create context tensors for the hour before each t₀.
    Modified to always save tensors, even if they're all zeros.
    """

    # Calculate context parameters
    n_context_intervals = int(context_duration_minutes / context_interval_minutes)

    print("=== Creating Context Tensors ===")
    print(f"Context window: {context_duration_minutes} minutes before t₀")
    print(
        f"Context intervals: {n_context_intervals} intervals of {context_interval_minutes} minutes each"
    )

    # Create output directories
    output_path = Path(output_dir)
    raindrop_context_dir = output_path / "raindrop_context"
    # Legacy dirs removed; only Raindrop-ready context is persisted
    raindrop_context_dir.mkdir(parents=True, exist_ok=True)

    # Load existing trajectory metadata
    print("Loading med tensor metadata...")
    with open(med_tensors_metadata_path, "rb") as f:
        med_metadata = pickle.load(f)

    trajectories = med_metadata["trajectories"]
    print(f"Found {len(trajectories)} trajectories to create context for")

    # Load waveforms data
    print("Loading waveforms data...")
    waveforms_df = load_waveforms_data(waveforms_parquet_path)

    # Load medication data
    print("Loading medication data...")
    med_df = load_medication_data(med_data_parquet_path)

    # Create context tensors for each trajectory, optionally in parallel
    raindrop_context_metadata: Dict[str, Dict] = {}

    # Precompute time vectors once
    rd_time_minutes_from_t0 = torch.arange(
        n_context_intervals, dtype=torch.float32
    ) * context_interval_minutes - (n_context_intervals * context_interval_minutes)
    rd_time_hours_from_t0 = rd_time_minutes_from_t0 / 60.0

    # Prepare module-level globals for forked workers (copy-on-write)
    global _WF_DF, _MED_DF, _ITEM_LABELS
    _WF_DF = waveforms_df
    _MED_DF = med_df
    _ITEM_LABELS = med_metadata["item_labels"]

    if n_workers and n_workers > 1:
        print(f"Using {n_workers} workers for context generation...")
        with ProcessPoolExecutor(max_workers=n_workers) as ex:
            futures = [
                ex.submit(
                    _process_single_context,
                    tk,
                    ti,
                    n_context_intervals,
                    context_interval_minutes,
                    rd_time_hours_from_t0.tolist(),
                    str(raindrop_context_dir),
                )
                for tk, ti in trajectories.items()
            ]
            for fut in tqdm(
                as_completed(futures),
                total=len(futures),
                desc="Creating context tensors",
            ):
                traj_key, rd_meta = fut.result()
                raindrop_context_metadata[traj_key] = rd_meta
    else:
        for traj_key, traj_info in tqdm(
            trajectories.items(), desc="Creating context tensors"
        ):
            tk, rd_meta = _process_single_context(
                traj_key,
                traj_info,
                n_context_intervals,
                context_interval_minutes,
                rd_time_hours_from_t0.tolist(),
                str(raindrop_context_dir),
            )
            raindrop_context_metadata[tk] = rd_meta

    # Save metadata
    context_metadata = {
        "raindrop_context": raindrop_context_metadata,
        "n_context_intervals": n_context_intervals,
        "context_interval_minutes": context_interval_minutes,
        "context_duration_minutes": context_duration_minutes,
        "total_trajectories": len(trajectories),
        "raindrop_tensors_created": len(raindrop_context_metadata),
        "created_at": datetime.now().isoformat(),
        "source_trajectories": med_tensors_metadata_path,
    }

    metadata_file = output_path / "context_tensors_metadata.pkl"
    with open(metadata_file, "wb") as f:
        pickle.dump(context_metadata, f)

    # Print summary
    print("\n=== Context Tensors Summary ===")
    print(f"Total trajectories: {len(trajectories)}")
    print(f"Raindrop context tensors created: {len(raindrop_context_metadata)} (100%)")

    print(f"\nFiles saved to: {output_path}")
    print(f"Metadata saved to: {metadata_file}")

    return context_metadata


def load_waveforms_data(parquet_path: str) -> pd.DataFrame:
    """
    Load waveforms data for context tensor creation.
    """
    print(f"Loading waveforms data from {parquet_path}...")
    df = pd.read_parquet(parquet_path)

    # Ensure absolute_timestamp is datetime and tz-naive (UTC)
    if not pd.api.types.is_datetime64_any_dtype(df["absolute_timestamp"]):
        df["absolute_timestamp"] = pd.to_datetime(
            df["absolute_timestamp"], errors="coerce"
        )
    # If tz-aware, convert to UTC then drop tz
    if pd.api.types.is_datetime64tz_dtype(df["absolute_timestamp"]):
        try:
            df["absolute_timestamp"] = (
                df["absolute_timestamp"].dt.tz_convert("UTC").dt.tz_localize(None)
            )
        except Exception:
            # If tz_convert fails (ambiguous/non-existent), just drop tz
            df["absolute_timestamp"] = df["absolute_timestamp"].dt.tz_localize(None)

    # Check for required physio columns
    required_physio = ["ABP MEAN_z", "CVP_z", "HR_z", "RESP_z"]
    available_physio = [col for col in required_physio if col in df.columns]
    missing_physio = [col for col in required_physio if col not in df.columns]

    print(f"Available physio measurements: {available_physio}")
    if missing_physio:
        print(f"Missing physio measurements: {missing_physio}")

    return df


def load_medication_data(parquet_path: str) -> pd.DataFrame:
    """
    Load medication data for context tensor creation.

    Attempts parquet first (regardless of file extension), then falls back to
    pandas pickle, raw pickle, and a safe torch.load probe. Normalizes schema
    to include a 'rate_norm' column by mapping from common
    alternatives when needed.
    """
    print(f"Loading medication data from {parquet_path}...")

    df: Optional[pd.DataFrame] = None
    loaders_tried: List[str] = []

    # Always try parquet first; many .bin files are parquet saved with a custom extension
    try:
        df = pd.read_parquet(parquet_path)
    except Exception as e_parquet:
        loaders_tried.append(f"parquet:{e_parquet}")
        # Try pandas' pickle reader
        try:
            df = pd.read_pickle(parquet_path)
        except Exception as e_pd_pickle:
            loaders_tried.append(f"pd.read_pickle:{e_pd_pickle}")
            # Try raw pickle.load
            try:
                with open(parquet_path, "rb") as f:
                    obj = pickle.load(f)
                if isinstance(obj, pd.DataFrame):
                    df = obj
                else:
                    raise TypeError("pickled object was not a pandas DataFrame")
            except Exception as e_raw_pickle:
                loaders_tried.append(f"pickle.load:{e_raw_pickle}")
                # Try a lightweight torch.load probe (in case the file was saved via torch)
                try:
                    import torch as _torch

                    obj = _torch.load(parquet_path, map_location="cpu")
                    if isinstance(obj, pd.DataFrame):
                        df = obj
                    elif (
                        isinstance(obj, dict)
                        and "data" in obj
                        and isinstance(obj["data"], pd.DataFrame)
                    ):
                        df = obj["data"]
                    else:
                        raise TypeError("torch file did not contain a pandas DataFrame")
                except Exception as e_torch:
                    loaders_tried.append(f"torch.load:{e_torch}")

    if df is None:
        raise ValueError(
            f"Failed to load medication data from {parquet_path}. Tried: {loaders_tried}"
        )

    # Normalize column names where possible (map common variants to canonical)
    cols_lower_map = {c.lower(): c for c in df.columns}

    def _ensure_column(
        df_in: pd.DataFrame, canonical: str, candidates: List[str]
    ) -> None:
        for cand in candidates:
            cand_lower = cand.lower()
            if cand_lower in cols_lower_map:
                src = cols_lower_map[cand_lower]
                if src != canonical and canonical not in df_in.columns:
                    df_in.rename(columns={src: canonical}, inplace=True)
                return

    _ensure_column(df, "hadm_id", ["hadm_id", "HADM_ID"])
    _ensure_column(df, "start_time", ["start_time", "START_TIME", "STARTTIME"])
    _ensure_column(df, "end_time", ["end_time", "END_TIME", "ENDTIME"])
    _ensure_column(df, "item_label", ["item_label", "ITEM_LABEL"])

    # Normalize rate column -> create 'rate_norm' if missing
    rate_canonical = "rate_norm"
    if rate_canonical not in df.columns:
        # Candidates: common variants and fuzzy contains('rate') & contains('weight')
        candidate_order = [
            "rate_weight_normalized",
            "rate_per_kg",
            "rate_per_weight",
            "rate/weight",
            "rate",
            "RATE",
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
            fuzzy = [
                c for c in df.columns if ("rate" in c.lower() and "weight" in c.lower())
            ]
            if len(fuzzy) > 0:
                picked_src = fuzzy[0]

        if picked_src is not None:
            df[rate_canonical] = pd.to_numeric(df[picked_src], errors="coerce")
        else:
            # As a last resort, if a plain 'rate' exists, use it
            if "rate" in df.columns:
                df[rate_canonical] = pd.to_numeric(df["rate"], errors="coerce")

    # Validate required columns
    required_cols = ["hadm_id", "start_time", "end_time", "item_label", rate_canonical]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(
            "Missing required medication columns: {}. Available: {}".format(
                missing_cols, list(df.columns)
            )
        )

    # Ensure datetime types and make tz-naive (UTC)
    if not pd.api.types.is_datetime64_any_dtype(df["start_time"]):
        df["start_time"] = pd.to_datetime(df["start_time"], errors="coerce")
    if not pd.api.types.is_datetime64_any_dtype(df["end_time"]):
        df["end_time"] = pd.to_datetime(df["end_time"], errors="coerce")
    if pd.api.types.is_datetime64tz_dtype(df["start_time"]):
        try:
            df["start_time"] = (
                df["start_time"].dt.tz_convert("UTC").dt.tz_localize(None)
            )
        except Exception:
            df["start_time"] = df["start_time"].dt.tz_localize(None)
    if pd.api.types.is_datetime64tz_dtype(df["end_time"]):
        try:
            df["end_time"] = df["end_time"].dt.tz_convert("UTC").dt.tz_localize(None)
        except Exception:
            df["end_time"] = df["end_time"].dt.tz_localize(None)

    print(f"Loaded {len(df)} medication events")
   
    return df


def create_physio_context_tensor(
    traj_key: str,
    traj_info: Dict,
    waveforms_df: pd.DataFrame,
    n_intervals: int,
    interval_minutes: int,
) -> Tuple[torch.Tensor, torch.Tensor]:  # Changed: Always returns tensors, never None
    """
    Create physiological context tensor for the hour before t₀.
    Always returns tensors, even if all zeros.

    Returns:
        tuple: (physio_values, physio_mask) - always returns, may be all zeros
    """
    hadm_id = traj_info["hadm_id"]
    t0_time = pd.to_datetime(traj_info["t0_time"])
    # Ensure t0_time is tz-naive
    if getattr(t0_time, "tz", None) is not None:
        try:
            t0_time = t0_time.tz_convert("UTC").tz_localize(None)
        except Exception:
            t0_time = t0_time.tz_localize(None)

    # Define context window: 1 hour before t₀
    context_start_time = t0_time - pd.Timedelta(minutes=interval_minutes * n_intervals)
    context_end_time = t0_time

    # Initialize arrays with STRICTLY NORMALIZED physio signals (z-scores)
    # Default set based on availability in input parquet
    physio_measurements = ["ABP MEAN_z", "CVP_z", "HR_z", "RESP_z"]
    n_measurements = len(physio_measurements)

    # Get patient waveforms data in context window
    patient_waveforms = waveforms_df[
        (waveforms_df["hadm_id"] == hadm_id)
        & (waveforms_df["absolute_timestamp"] >= context_start_time)
        & (waveforms_df["absolute_timestamp"] < context_end_time)
    ].copy()

    if len(patient_waveforms) == 0:
        print(
            f"    No physio data in context window for {traj_key} - saving zero tensor"
        )
        return (
            torch.zeros((n_intervals, n_measurements), dtype=torch.float32),
            torch.zeros((n_intervals, n_measurements), dtype=torch.float32),
        )

    # Calculate bin indices for context window
    secs_from_start = (
        patient_waveforms["absolute_timestamp"] - context_start_time
    ).dt.total_seconds()
    patient_waveforms["context_interval_idx"] = (
        secs_from_start // (interval_minutes * 60)
    ).astype(int)

    # Restrict to available columns
    available_cols = [c for c in physio_measurements if c in patient_waveforms.columns]
    if not available_cols:
        return (
            torch.zeros((n_intervals, n_measurements), dtype=torch.float32),
            torch.zeros((n_intervals, n_measurements), dtype=torch.float32),
        )

    # Compute per-bin means and counts vectorized
    means = (
        patient_waveforms.groupby("context_interval_idx")[available_cols]
        .mean()
        .reindex(range(n_intervals), fill_value=np.nan)
    )
    counts = (
        patient_waveforms.groupby("context_interval_idx")[available_cols]
        .count()
        .reindex(range(n_intervals), fill_value=0)
    )

    # Build full arrays with fixed column order
    values = np.zeros((n_intervals, n_measurements), dtype=np.float32)
    mask = np.zeros((n_intervals, n_measurements), dtype=np.float32)
    for j, name in enumerate(physio_measurements):
        if name in available_cols:
            col_mean = means[name].to_numpy()
            col_cnt = counts[name].to_numpy()
            valid = col_cnt > 0
            values[valid, j] = col_mean[valid].astype(np.float32)
            mask[valid, j] = 1.0

    total_measurements = int(mask.sum())
    if total_measurements == 0:
        print(f"    No valid physio measurements for {traj_key} - saving zero tensor")
    else:
        print(
            f"    Physio context: {total_measurements} total measurements across {n_measurements} types"
        )

    return torch.from_numpy(values), torch.from_numpy(mask)


def create_meds_context_tensor(
    traj_key: str,
    traj_info: Dict,
    med_df: pd.DataFrame,
    n_intervals: int,
    interval_minutes: int,
    item_labels: List[str],
) -> Tuple[torch.Tensor, torch.Tensor]:  # Changed: Always returns tensors, never None
    """
    Create medication context tensor for the hour before t₀.
    Always returns tensors, even if all zeros.

    Returns:
        tuple: (meds_values, meds_mask) - always returns, may be all zeros
    """
    hadm_id = traj_info["hadm_id"]
    t0_time = pd.to_datetime(traj_info["t0_time"])
    # Ensure t0_time is tz-naive
    if getattr(t0_time, "tz", None) is not None:
        try:
            t0_time = t0_time.tz_convert("UTC").tz_localize(None)
        except Exception:
            t0_time = t0_time.tz_localize(None)

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
    patient_meds = med_df[med_df["hadm_id"] == hadm_id].copy()

    if len(patient_meds) == 0:
        print(f"    No medication data for patient {hadm_id} - saving zero tensor")
        return torch.from_numpy(values_array), torch.from_numpy(
            mask_array
        )  # Return zeros instead of None

    # Filter to infusions that overlap with context window
    relevant_infusions = patient_meds[
        (patient_meds["end_time"] > context_start_time)
        & (patient_meds["start_time"] < context_end_time)
    ].copy()

    if len(relevant_infusions) == 0:
        print(
            f"    No medication infusions in context window for {traj_key} - saving zero tensor"
        )
        return torch.from_numpy(values_array), torch.from_numpy(
            mask_array
        )  # Return zeros instead of None

    # Process each medication type separately, vectorized per item
    for item_label in item_labels:
        item_idx = item_to_idx.get(item_label)
        if item_idx is None:
            continue

        item_infusions = relevant_infusions[
            relevant_infusions["item_label"] == item_label
        ].sort_values("start_time")

        if len(item_infusions) == 0:
            continue

        # Compute effective start/end clipped to context window
        cs = np.array(context_start_time).astype("datetime64[s]")
        ce = np.array(context_end_time).astype("datetime64[s]")
        start_s = item_infusions["start_time"].to_numpy(dtype="datetime64[s]")
        end_s = item_infusions["end_time"].to_numpy(dtype="datetime64[s]")

        eff_start = np.maximum(start_s, cs)
        eff_end = np.minimum(end_s, ce)

        # Convert to seconds offset from context start
        eff_start_sec = (eff_start - cs).astype("timedelta64[s]").astype(np.int64)
        eff_end_sec = (eff_end - cs).astype("timedelta64[s]").astype(np.int64)

        # Interval indices
        bin_sec = interval_minutes * 60
        start_idx = np.floor_divide(
            np.clip(eff_start_sec, 0, bin_sec * n_intervals - 1), bin_sec
        ).astype(int)
        end_idx = np.ceil(
            np.clip(eff_end_sec, 0, bin_sec * n_intervals) / bin_sec
        ).astype(int)
        end_idx = np.clip(end_idx, 0, n_intervals)

        # Build coverage matrix [num_rows, n_intervals]
        num_rows = start_idx.shape[0]
        if num_rows == 0:
            continue
        intervals = np.arange(n_intervals)[None, :]
        coverage = (intervals >= start_idx[:, None]) & (intervals < end_idx[:, None])

        # Last-overlap wins: take index of last True per interval
        row_indices = np.arange(num_rows)[:, None]
        masked_idx = np.where(coverage, row_indices, -1)
        last_idx_per_interval = masked_idx.max(axis=0)

        rates = (
            pd.to_numeric(item_infusions["rate_norm"], errors="coerce")
            .fillna(0.0)
            .to_numpy(dtype=np.float32)
        )
        valid = last_idx_per_interval >= 0
        if valid.any():
            values_array[valid, item_idx] = rates[last_idx_per_interval[valid]]
            mask_array[valid, item_idx] = 1.0

    # Always return tensors (even if all zeros)
    total_medications = int(np.sum(mask_array))
    if total_medications == 0:
        print(
            f"    No valid medication data in context window for {traj_key} - saving zero tensor"
        )
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
    tensor_type: str,
) -> str:
    """
    Save context tensor to disk in same format as existing tensors.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create time arrays (relative to t₀, negative values since this is before t₀)
    time_minutes_from_t0 = torch.arange(
        n_intervals, dtype=torch.float32
    ) * interval_minutes - (n_intervals * interval_minutes)
    time_hours_from_t0 = (
        time_minutes_from_t0 / 60.0
    )  # [-1.0, -0.83, -0.67, -0.5, -0.33, -0.17]
    time_seconds_from_t0 = time_minutes_from_t0 * 60

    filepath = output_dir / f"{tensor_type}_{traj_key}.pt"

    # Save in same format as existing tensors: (values, mask, time_seconds, time_hours, length)
    torch.save(
        (
            values_tensor,
            mask_tensor,
            time_seconds_from_t0,
            time_hours_from_t0,
            n_intervals,
        ),
        filepath,
    )

    return str(filepath)


def inspect_context_tensors(metadata: Dict, n_samples: int = 3) -> None:
    """
    Inspect sample context tensors.
    """
    print(f"\n=== Inspecting {n_samples} Sample Context Tensors ===")

    # Get trajectories that have both physio and meds context
    physio_keys = set(metadata["physio_context"].keys())
    meds_keys = set(metadata["meds_context"].keys())
    common_keys = list(physio_keys.intersection(meds_keys))[:n_samples]

    for traj_key in common_keys:
        print(f"\nTrajectory: {traj_key}")

        # Load physio context
        physio_info = metadata["physio_context"][traj_key]
        physio_values, physio_mask, time_sec, time_hr, n_intervals = torch.load(
            physio_info["file_path"]
        )

        print("Physio Context:")
        print(f"  Shape: {physio_values.shape}")
        print(f"  Time range: {time_hr[0]:.2f} to {time_hr[-1]:.2f} hours from t₀")
        print(f"  Total measurements: {torch.sum(physio_mask > 0).item()}")

        # Load meds context
        meds_info = metadata["meds_context"][traj_key]
        meds_values, meds_mask, time_sec, time_hr, n_intervals = torch.load(
            meds_info["file_path"]
        )

        print("Meds Context:")
        print(f"  Shape: {meds_values.shape}")
        print(f"  Time range: {time_hr[0]:.2f} to {time_hr[-1]:.2f} hours from t₀")
        print(f"  Total medication intervals: {torch.sum(meds_mask > 0).item()}")


def preprocess_baseline_values(
    df,
    id_col="hadm_id",
    categorical_features=None,
    continuous_features=None,
    binary_features=None,
    drop_first=True,
):
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
        dummies = pd.get_dummies(
            processed_df[categorical_features], drop_first=drop_first, dummy_na=False
        ).astype(int)
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
        binary = (
            processed_df[binary_features]
            .apply(pd.to_numeric, errors="coerce")
            .fillna(0)
            .astype(int)
        )
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
    with open(trajectory_metadata_path, "rb") as f:
        trajectory_data = pickle.load(f)
    valid_hadm_ids = list(trajectory_data["trajectories"].keys())
    valid_hadm_ids = [
        int(s.split("_")[0]) if "_" in s else int(s) for s in valid_hadm_ids
    ]
    print(f"Found {len(valid_hadm_ids)} patients with trajectories.")

    # Load raw MIMIC data tables from the 'hosp' module
    patients_path = os.path.join(input_dir, "PATIENTS.csv")
    admission_path = os.path.join(input_dir, "ADMISSIONS.csv")
    transfers_path = os.path.join(input_dir, "TRANSFERS.csv")

    try:
        patients_df = pd.read_csv(patients_path)
        admission_df = pd.read_csv(admission_path)
        transfers_df = pd.read_csv(transfers_path)
    except FileNotFoundError as e:
        print(
            "Error: Could not find a required MIMIC 'hosp' file. Make sure your input directory is correct."
        )
        print(f"File not found: {e.filename}")
        return

    # Merge admissions with patients to get age and gender
    merged_df = pd.merge(admission_df, patients_df, on="SUBJECT_ID", how="left")

    # Calculate patient age at the time of admission
    merged_df["ADMITTIME"] = pd.to_datetime(merged_df["ADMITTIME"])
    merged_df["DOB"] = pd.to_datetime(merged_df["DOB"], errors="coerce")
    prov_age = merged_df["ADMITTIME"].dt.year - merged_df["DOB"].dt.year
    merged_df["AGE"] = prov_age.where(prov_age < 300, 89)

    # Merge with transfers to get discharge time
    merged_df = pd.merge(
        merged_df, transfers_df, on=["HADM_ID", "SUBJECT_ID"], how="left"
    )

    # Calculate admission duration in days
    merged_df["DISCHTIME"] = pd.to_datetime(merged_df["DISCHTIME"])
    merged_df["ADMIT_DURATION"] = (
        merged_df["DISCHTIME"] - merged_df["ADMITTIME"]
    ).dt.total_seconds() / (3600 * 24)

    # Filter by the valid hadm_ids that have trajectories
    merged_df_final_filtered = merged_df[merged_df["HADM_ID"].isin(valid_hadm_ids)]
    # print(sorted(list(merged_df['HADM_ID'].unique())))
    # print(merged_df_final_filtered[merged_df_final_filtered['HADM_ID'] == 100477])

    # Apply additional filters as requested
    # merged_with_disch_df_final_filtered = merged_df_final_filtered[merged_df_final_filtered['ADMIT_DURATION'] <= 10]
    # merged_with_disch_df_final_filtered = merged_with_disch_df_final_filtered[merged_with_disch_df_final_filtered['ADMIT_DURATION'] >= 2]
    merged_with_disch_df_final_filtered = merged_df_final_filtered

    print(f"Found {len(merged_with_disch_df_final_filtered)} patients.")
    print(
        "Available columns in merged data:", merged_df_final_filtered.columns.tolist()
    )

    # Define feature lists for preprocessing
    categorical_features = [
        "GENDER",
        "MARITAL_STATUS",
        "INSURANCE",
        "ADMISSION_LOCATION",
        "ADMISSION_TYPE",
    ]
    continuous_features = ["AGE", "ADMIT_DURATION"]
    binary_features = []  # None specified in example

    # Preprocess the baseline data
    processed_baseline_df = preprocess_baseline_values(
        merged_with_disch_df_final_filtered,
        id_col="HADM_ID",
        categorical_features=categorical_features,
        continuous_features=continuous_features,
        binary_features=binary_features,
        drop_first=True,
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
        "feature_names": list(processed_baseline_df.columns),
        "feature_dim": len(processed_baseline_df.columns),
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
    med_data_path = (
        "../../../data/mimic_3_data/processed_data/mv_filtered_10min.parquet"
    )
    output_dir = "../../../data/mimic_3_data/processed_data/context_tensors_output"

    if all(
        os.path.exists(p) for p in [waveforms_path, med_metadata_path, med_data_path]
    ):
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

        create_baseline_tensors(
            input_dir="../../../data/mimic_3_data/input_data",
            output_dir=output_dir,
            trajectory_metadata_path="../../../data/mimic_3_data/processed_data/med_tensors_output/med_tensors_metadata.pkl",
        )

    else:
        for p in [waveforms_path, med_metadata_path, med_data_path]:
            if not os.path.exists(p):
                print(p)
        print("Please ensure the following files exist:")
        print(f"  - Waveforms: {waveforms_path}")
        print(f"  - Med metadata: {med_metadata_path}")
        print(f"  - Med data: {med_data_path}")
