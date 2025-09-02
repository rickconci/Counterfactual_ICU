import os
import pickle
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple
import typing as t
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import multiprocessing as mp


@dataclass
class MedFeatureConfig:
    """
    Configuration for medication context features.

    Attributes:
        default_half_life_hours: Fallback half-life in hours if not provided per med.
        half_life_hours_by_label: Mapping of medication label to half-life hours.
        log_duration: Apply log1p to duration features.
        missing_stop_fill_seconds: Large positive value to use when no last stop exists pre-t0.
        final_clip_bounds: Optional global clamp (min, max) applied to final features.
    """

    default_half_life_hours: float = 0.5
    half_life_hours_by_label: Dict[str, float] = field(default_factory=dict)
    log_duration: bool = True
    missing_stop_fill_seconds: float = 1e9
    final_clip_bounds: t.Optional[Tuple[float, float]] = (-10.0, 10.0)


@dataclass
class TrajectoryGrid:
    """
    Time grid specification for discretization.

    Attributes:
        n_intervals: Number of time steps in the grid.
        interval_seconds: Duration of each step in seconds.
    """

    n_intervals: int
    interval_seconds: int


@dataclass
class MedPreContext:
    """Pre-t0 medication context derived from full patient history."""

    pre_on_hours: np.ndarray  # [M]
    pre_cum_hours: np.ndarray  # [M]
    trigger_flags: np.ndarray  # [M]
    dt_since_last_stop_at_t0_seconds: np.ndarray  # [M] (np.nan if none)


def compute_med_context_features(
    values: np.ndarray,
    mask: np.ndarray,
    grid: TrajectoryGrid,
    pre_on_hours: np.ndarray,
    pre_cum_hours: np.ndarray,
    trigger_flags: np.ndarray,
    tau_seconds_per_med: np.ndarray,
    dt_since_last_stop_at_t0_seconds: np.ndarray,
    config: t.Optional[MedFeatureConfig] = None,
) -> Tuple[np.ndarray, List[str]]:
    """
    Compute causal, per-medication context features for each time step.

    Per-med feature block: [active, current_rate, start_flag, stop_flag,
    on_duration_hours_log1p, off_duration_hours_log1p, exp_tau_1..K].

    Args:
        values: [T, M] infusion rate values (float32), zero when inactive.
        mask: [T, M] binary active indicator.
        grid: TrajectoryGrid definition.
        config: MedFeatureConfig; defaults used if None.

    Returns:
        features: [T, M * F] flattened feature matrix (float32).
        feature_names: list of per-med feature names in order.
    """
    if config is None:
        config = MedFeatureConfig()

    values = values.astype(np.float32)
    active = (mask > 0).astype(np.float32)
    rates = (values * active).astype(np.float32)
    T, M = rates.shape

    # Feature 1: rate (already masked to 0 when inactive)
    rate_feat = rates  # [T, M]

    # Feature 2: how long med was on before t0 (constant along T)
    pre_on = pre_on_hours.astype(np.float32)
    if config.log_duration:
        pre_on = np.log1p(pre_on)
    pre_on_feat = np.broadcast_to(pre_on[None, :], (T, M))

    # Feature 3: cumulative amount since start up to current t
    interval_hours = float(grid.interval_seconds) / 3600.0
    cum_within = np.cumsum(rates, axis=0) * interval_hours
    total_cum = cum_within + pre_cum_hours[None, :].astype(np.float32)
    total_cum = np.clip(total_cum, a_min=0.0, a_max=None)
    cum_feat = np.log1p(total_cum)

    # Feature 4: post-stop effect decay based on half-life
    effect = np.zeros((T, M), dtype=np.float32)
    dt_since_stop = dt_since_last_stop_at_t0_seconds.astype(np.float32).copy()

    for m in range(M):
        tau = float(tau_seconds_per_med[m]) if float(tau_seconds_per_med[m]) > 0 else 1.0
        # t=0
        if active[0, m] == 1.0:
            effect[0, m] = 1.0
            dt_since_stop[m] = np.nan
        else:
            base_dt = max(0.0, float(dt_since_stop[m]))
            # If 'missing large' was used, exp(-large/tau) ~ 0
            effect[0, m] = float(np.exp(-base_dt / tau))
            dt_since_stop[m] = base_dt + grid.interval_seconds
        # t>0
        for t_idx in range(1, T):
            if active[t_idx, m] == 1.0:
                effect[t_idx, m] = 1.0
                dt_since_stop[m] = np.nan
            else:
                if active[t_idx - 1, m] == 1.0:
                    # just stopped
                    effect[t_idx, m] = 1.0
                    dt_since_stop[m] = grid.interval_seconds
                else:
                    if not np.isfinite(dt_since_stop[m]):
                        dt_since_stop[m] = grid.interval_seconds
                    else:
                        dt_since_stop[m] += grid.interval_seconds
                    effect[t_idx, m] = float(np.exp(-dt_since_stop[m] / tau))

    # Feature 5: trigger flag per med, broadcast across time
    trigger_feat = np.broadcast_to(trigger_flags.astype(np.float32)[None, :], (T, M))

    per_med = np.stack(
        [rate_feat, pre_on_feat, cum_feat, effect, trigger_feat], axis=-1
    )  # [T, M, F]
    features = per_med.reshape(T, -1).astype(np.float32)

    # Sanitize and clamp to avoid extreme values in MLP inputs
    features = np.nan_to_num(features, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
    if config.final_clip_bounds is not None:
        lo, hi = config.final_clip_bounds
        features = np.clip(features, lo, hi, out=features)

    feature_names: List[str] = [
        "rate_hr_weight_norm",
        "pre_on_hours_log1p",
        "cumulative_since_start_hours_log1p",
        "post_stop_effect",
        "trigger_flag",
    ]

    return features, feature_names


def build_values_mask_from_events(
    events: pd.DataFrame,
    item_labels: List[str],
    t0_time: pd.Timestamp,
    trajectory_end_time: pd.Timestamp,
    grid: TrajectoryGrid,
    rate_column: str = "rate/weight_normalized",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build [T, M] values and mask arrays from raw medication event rows.

    Args:
        events: DataFrame with at least ['start_time','end_time','item_label', rate_column].
        item_labels: Master ordered list of medication labels (columns M).
        t0_time: Start of the trajectory window.
        trajectory_end_time: End of the trajectory window.
        grid: Trajectory grid spec.
        rate_column: Column name for rate/weight normalized values.

    Returns:
        values_array: [T, M] float32 rates aggregated per slice.
        mask_array: [T, M] float32 activity mask (0/1).
    """
    n_intervals = grid.n_intervals
    interval_seconds = grid.interval_seconds
    n_medications = len(item_labels)
    values_array = np.zeros((n_intervals, n_medications), dtype=np.float32)
    mask_array = np.zeros((n_intervals, n_medications), dtype=np.float32)

    item_to_idx = {item: idx for idx, item in enumerate(item_labels)}

    trajectory_duration_seconds = (trajectory_end_time - t0_time).total_seconds()
    max_interval_for_trajectory = min(
        n_intervals, int(np.ceil(trajectory_duration_seconds / interval_seconds))
    )

    for item_label in item_labels:
        if item_label not in item_to_idx:
            continue
        item_idx = item_to_idx[item_label]
        item_infusions = events[events["item_label"] == item_label].sort_values(
            "start_time"
        )

        for _, row in item_infusions.iterrows():
            rate_value = float(row.get(rate_column, 0.0) or 0.0)
            if not np.isfinite(rate_value) or rate_value == 0.0:
                continue

            effective_start = max(t0_time, row["start_time"])
            effective_end = min(trajectory_end_time, row["end_time"])

            start_seconds = (effective_start - t0_time).total_seconds()
            end_seconds = (effective_end - t0_time).total_seconds()

            start_idx = int(start_seconds // interval_seconds)
            end_idx = int(np.ceil(end_seconds / interval_seconds))

            for time_idx in range(start_idx, end_idx):
                if 0 <= time_idx < max_interval_for_trajectory:
                    values_array[time_idx, item_idx] += rate_value
                    mask_array[time_idx, item_idx] = 1.0

    return values_array, mask_array


def build_precontext_from_events(
    events_full_patient: pd.DataFrame,
    item_labels: List[str],
    t0_time: pd.Timestamp,
    rate_column: str,
) -> MedPreContext:
    """
    Build pre-t0 context from full patient history (not window-truncated).
    """
    M = len(item_labels)
    pre_on_hours = np.zeros((M,), dtype=np.float32)
    pre_cum_hours = np.zeros((M,), dtype=np.float32)
    trigger_flags = np.zeros((M,), dtype=np.float32)
    dt_since_last_stop_at_t0_seconds = np.full((M,), np.nan, dtype=np.float32)

    item_to_idx = {item: idx for idx, item in enumerate(item_labels)}

    for item_label in item_labels:
        m = item_to_idx[item_label]
        df_m = events_full_patient[events_full_patient["item_label"] == item_label]
        if df_m.empty:
            continue

        # Trigger flag: row with trigger True at exactly t0
        if "trigger" in df_m.columns:
            trig_rows = df_m[(df_m["trigger"] == True) & (df_m["start_time"] == t0_time)]  # noqa: E712
            if len(trig_rows) > 0:
                trigger_flags[m] = 1.0

        # Active at t0?
        overlapping = df_m[(df_m["start_time"] < t0_time) & (df_m["end_time"] > t0_time)]
        if len(overlapping) > 0:
            # Use the interval that covers t0 (there should be at most one non-overlapping)
            row = overlapping.iloc[0]
            pre_on_hours[m] = max(
                0.0, (t0_time - row["start_time"]).total_seconds() / 3600.0
            )
        else:
            pre_on_hours[m] = 0.0

        # Cumulative since first start up to t0
        cum_hours = 0.0
        for _, r in df_m.iterrows():
            if r["start_time"] >= t0_time:
                continue
            overlap_start = r["start_time"]
            overlap_end = r["end_time"] if r["end_time"] <= t0_time else t0_time
            if overlap_end <= overlap_start:
                continue
            dur_h = (overlap_end - overlap_start).total_seconds() / 3600.0
            rate_val = float(r.get(rate_column, 0.0) or 0.0)
            if np.isfinite(rate_val) and rate_val > 0.0:
                cum_hours += dur_h * rate_val
        pre_cum_hours[m] = float(cum_hours)

        # Time since last stop at t0
        ended_before = df_m[df_m["end_time"] <= t0_time]
        if len(ended_before) > 0:
            last_end = ended_before["end_time"].max()
            dt_sec = (t0_time - last_end).total_seconds()
            if dt_sec >= 0:
                dt_since_last_stop_at_t0_seconds[m] = float(dt_sec)

    # Replace NaNs: if active at t0, set 0 (no decay); if truly missing, set large constant
    active_at_t0 = np.zeros((M,), dtype=np.float32)
    for item_label in item_labels:
        m = item_to_idx[item_label]
        df_m = events_full_patient[events_full_patient["item_label"] == item_label]
        if len(df_m[(df_m["start_time"] < t0_time) & (df_m["end_time"] > t0_time)]) > 0:
            active_at_t0[m] = 1.0
    fill_large = MedFeatureConfig().missing_stop_fill_seconds
    # where active -> 0, else large constant
    nan_mask = ~np.isfinite(dt_since_last_stop_at_t0_seconds)
    dt_since_last_stop_at_t0_seconds = np.where(
        nan_mask & (active_at_t0 > 0), 0.0, dt_since_last_stop_at_t0_seconds
    )
    dt_since_last_stop_at_t0_seconds = np.where(
        ~np.isfinite(dt_since_last_stop_at_t0_seconds), fill_large, dt_since_last_stop_at_t0_seconds
    ).astype(np.float32)

    return MedPreContext(
        pre_on_hours=pre_on_hours,
        pre_cum_hours=pre_cum_hours,
        trigger_flags=trigger_flags,
        dt_since_last_stop_at_t0_seconds=dt_since_last_stop_at_t0_seconds,
    )


def build_tau_seconds_per_med(
    item_labels: List[str], config: MedFeatureConfig
) -> np.ndarray:
    tau_seconds = np.zeros((len(item_labels),), dtype=np.float32)
    for i, label in enumerate(item_labels):
        hl_h = config.half_life_hours_by_label.get(label, config.default_half_life_hours)
        hl_h = max(1e-6, float(hl_h))
        tau_seconds[i] = float((hl_h * 3600.0) / np.log(2.0))
    return tau_seconds



def load_medication_data(parquet_path: str) -> pd.DataFrame:
    """
    Load medication data for context tensor creation.

    Attempts parquet first (regardless of file extension), then falls back to
    pandas pickle, raw pickle, and a safe torch.load probe. Normalizes schema
    to include a 'rate/weight_normalized' column by mapping from common
    alternatives when needed.
    """
    print(f"Loading medication data from {parquet_path}...")

    df: t.Optional[pd.DataFrame] = None
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

    # Normalize rate column -> create 'rate/weight_normalized' if missing
    rate_canonical = "rate/weight_normalized"
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
        picked_src: t.Optional[str] = None

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
    # Clean meds: remove Esmolol; cap each medication at mean + 4*std (per item)
    if "item_label" in df.columns:
        # Drop Esmolol (case-insensitive)
        mask_esmo = df["item_label"].str.lower() == "esmolol"
        if mask_esmo.any():
            df = df.loc[~mask_esmo].copy()
            print(f"Removed {int(mask_esmo.sum())} Esmolol rows")

        # Ensure numeric
        df[rate_canonical] = pd.to_numeric(df[rate_canonical], errors="coerce")

        # Compute per-medication caps at mean + 4*std
        stats = df.groupby("item_label")[rate_canonical].agg(["mean", "std"])
        stats["cap"] = stats["mean"] + 4.0 * stats["std"]

        # Merge caps and clip
        df = df.merge(
            stats["cap"].rename("___cap"),
            left_on="item_label",
            right_index=True,
            how="left",
        )
        before = df[rate_canonical]
        df[rate_canonical] = before.clip(lower=0.0, upper=df["___cap"])
        num_clipped = int((before > df["___cap"]).sum())
        df.drop(columns=["___cap"], inplace=True)
        if num_clipped > 0:
            print(f"Capped {num_clipped} medication rows at mean+4*std per item_label")

    df_triggers = df.dropna(subset=["action_cluster_id"])
    return df, df_triggers



def identify_trajectories(
    df_full: pd.DataFrame,
    df_triggers: pd.DataFrame,
    trajectory_duration_minutes: int = 20,
) -> Dict:
    """
    Group by hadm_id and action_cluster_id to identify trajectories.
    Each trajectory lasts trajectory_duration_minutes OR until the start of the next action_cluster_id.
    """
    print("Identifying trajectories and calculating trajectory windows...")

    trajectories = {}

    # First, get all action_cluster_ids per hadm_id with their t0 times
    action_starts = (
        df_triggers.groupby(["hadm_id", "action_cluster_id"])["start_time"]
        .min()
        .reset_index()
    )
    action_starts.columns = ["hadm_id", "action_cluster_id", "t0_time"]

    # Sort by hadm_id and t0_time to find next action starts
    action_starts = action_starts.sort_values(["hadm_id", "t0_time"])

    print(f"Found {len(action_starts)} unique action_cluster_ids across all patients")

    # For each hadm_id, determine trajectory end times
    for hadm_id in tqdm(action_starts["hadm_id"].unique(), desc="Processing patients"):
        patient_actions = action_starts[action_starts["hadm_id"] == hadm_id].copy()

        for idx, row in patient_actions.iterrows():
            action_cluster_id = row["action_cluster_id"]
            t0_time = row["t0_time"]

            # Calculate trajectory end time
            # Option 1: configurable minutes after t0
            end_time_20min = t0_time + pd.Timedelta(minutes=trajectory_duration_minutes)

            # Option 2: Start of next action_cluster_id
            next_actions = patient_actions[patient_actions["t0_time"] > t0_time]
            if len(next_actions) > 0:
                next_action_start = next_actions["t0_time"].min()
                trajectory_end_time = min(end_time_20min, next_action_start)
            else:
                trajectory_end_time = end_time_20min

            # Get all data for this patient within the trajectory window
            patient_data = df_full[df_full["hadm_id"] == hadm_id].copy()
            trajectory_data = patient_data[
                (patient_data["end_time"] > t0_time)  # Still ongoing at/after t0
                & (
                    patient_data["start_time"] < trajectory_end_time
                )  # Starts before trajectory ends
            ]

            # Store trajectory info
            traj_key = f"{hadm_id}_{int(action_cluster_id)}"
            trajectories[traj_key] = {
                "hadm_id": hadm_id,
                "action_cluster_id": action_cluster_id,
                "t0_time": t0_time,
                "trajectory_end_time": trajectory_end_time,
                "duration_minutes": (trajectory_end_time - t0_time).total_seconds()
                / 60,
                "data": trajectory_data,
            }

    print(f"Created {len(trajectories)} trajectories")

    # Print some statistics about trajectory durations
    durations = [t["duration_minutes"] for t in trajectories.values()]
    print("Trajectory duration statistics:")
    print(f"  Mean: {np.mean(durations):.2f} minutes")
    print(f"  Std: {np.std(durations):.2f} minutes")
    print(f"  Min: {np.min(durations):.2f} minutes")
    print(f"  Max: {np.max(durations):.2f} minutes")
    print(
        f"  Full {trajectory_duration_minutes}min trajectories: {sum(1 for d in durations if d >= (trajectory_duration_minutes - 0.1))}"
    )
    print(
        f"  Truncated trajectories: {sum(1 for d in durations if d < (trajectory_duration_minutes - 0.1))}"
    )

    return trajectories


def create_time_grid(
    trajectories: Dict,
    interval_seconds: int = 10,
    trajectory_duration_minutes: int = 20,
) -> Dict:
    """
    Create time grid parameters using the configured maximum forward duration.
    """
    print("Creating time grid parameters...")

    # Maximum trajectory duration is configurable (default 20 minutes)
    max_duration_seconds = trajectory_duration_minutes * 60
    n_intervals = int(np.ceil(max_duration_seconds / interval_seconds))

    # Verify this covers all trajectories
    actual_max_duration = (
        max(traj["duration_minutes"] for traj in trajectories.values()) * 60
    )

    print(
        f"Fixed trajectory duration: {max_duration_seconds} seconds ({trajectory_duration_minutes} minutes)"
    )
    print(f"Actual max trajectory duration: {actual_max_duration:.1f} seconds")
    print(
        f"Time grid will have {n_intervals} intervals of {interval_seconds} seconds each"
    )

    return {
        "n_intervals": n_intervals,
        "interval_seconds": interval_seconds,
        "max_duration": max_duration_seconds,
    }


def process_single_trajectory(
    traj_key: str,
    traj_info: Dict,
    item_labels: List[str],
    n_intervals: int,
    interval_seconds: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Process a single trajectory and create medication tensor.
    Only includes data within the trajectory window (t0 to trajectory_end_time).

    Returns:
        values_array: (n_intervals, n_medications) array of medication rates
        mask_array: (n_intervals, n_medications) array indicating data presence
    """
    data = traj_info["data"]
    t0_time = traj_info["t0_time"]
    trajectory_end_time = traj_info["trajectory_end_time"]

    grid = TrajectoryGrid(n_intervals=n_intervals, interval_seconds=interval_seconds)
    values_array, mask_array = build_values_mask_from_events(
        events=data,
        item_labels=item_labels,
        t0_time=t0_time,
        trajectory_end_time=trajectory_end_time,
        grid=grid,
        rate_column="rate/weight_normalized",
    )
    return values_array, mask_array


def save_trajectory_tensor(
    traj_key: str,
    values_array: np.ndarray,
    mask_array: np.ndarray,
    n_intervals: int,
    interval_seconds: int,
    output_dir: Path,
    med_context: np.ndarray,
    med_feature_names: List[str],
) -> Tuple[str, List[str]]:
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
    med_context_tensor = torch.from_numpy(med_context).float()

    # Backward compatible: append med_context as 6th item
    torch.save(
        (
            values_tensor,
            mask_tensor,
            time_seconds_tensor,
            time_hours_tensor,
            n_intervals,
            med_context_tensor,
        ),
        filepath,
    )

    return str(filepath), med_feature_names


def create_med_tensors_from_parquet(
    parquet_path: str,
    output_dir: str = "./med_tensors_output",
    interval_seconds: int = 10,
    trajectory_duration_minutes: int = 20,
    n_workers: int = 1,
    force_rerun: bool = False,
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
    df_full, df_triggers = load_medication_data(parquet_path)

    # Step 2: Identify trajectories using trigger data, but get all meds from full data
    trajectories = identify_trajectories(
        df_full,
        df_triggers,
        trajectory_duration_minutes=trajectory_duration_minutes,
    )

    # Step 3: Get unique item labels from FULL dataset (all medications)
    unique_item_labels = sorted(df_full["item_label"].unique().tolist())
    print(f"Found {len(unique_item_labels)} unique item labels:")
    for i, label in enumerate(unique_item_labels):
        print(f"  {i}: {label}")

    # Step 4: Create time grid parameters respecting configurable forward duration
    grid_params = create_time_grid(
        trajectories,
        interval_seconds,
        trajectory_duration_minutes=trajectory_duration_minutes,
    )
    n_intervals = grid_params["n_intervals"]

    # Step 5: Determine which trajectories still need processing (skip existing unless force)
    existing_metadata: Dict = {}
    metadata_file = Path(output_path) / "med_tensors_metadata.pkl"
    if metadata_file.exists() and not force_rerun:
        try:
            with open(metadata_file, "rb") as f:
                existing_metadata = pickle.load(f)
        except Exception:
            existing_metadata = {}

    existing_files: Dict[str, str] = {}
    if isinstance(existing_metadata, dict) and "trajectories" in existing_metadata:
        for k, v in existing_metadata["trajectories"].items():
            path_str = v.get("file_path") if isinstance(v, dict) else None
            if path_str and os.path.exists(path_str):
                existing_files[k] = path_str

    all_keys = list(trajectories.keys())
    pending_keys = [k for k in all_keys if force_rerun or k not in existing_files]
    print(f"\nTotal trajectories: {len(all_keys)} | Pending: {len(pending_keys)} | Skipped: {len(all_keys) - len(pending_keys)}")

    # Start with existing metadata if present
    trajectory_metadata = existing_metadata.get("trajectories", {}) if isinstance(existing_metadata, dict) else {}
    saved_files = [v for v in existing_files.values()]

    med_feature_names: t.Optional[List[str]] = None

    def _run_sequential() -> None:
        nonlocal med_feature_names
        for traj_key in tqdm(pending_keys, desc="Creating tensors"):
            traj_info = trajectories[traj_key]
            values_array, mask_array = process_single_trajectory(
                traj_key=traj_key,
                traj_info=traj_info,
                item_labels=unique_item_labels,
                n_intervals=n_intervals,
                interval_seconds=interval_seconds,
            )

            hadm_id = traj_info["hadm_id"]
            t0_time = traj_info["t0_time"]
            patient_full = df_full[df_full["hadm_id"] == hadm_id]
            pre_ctx = build_precontext_from_events(
                events_full_patient=patient_full,
                item_labels=unique_item_labels,
                t0_time=t0_time,
                rate_column="rate/weight_normalized",
            )

            cfg = MedFeatureConfig()
            tau_seconds_per_med = build_tau_seconds_per_med(unique_item_labels, cfg)

            grid = TrajectoryGrid(n_intervals=n_intervals, interval_seconds=interval_seconds)
            med_context, feature_names = compute_med_context_features(
                values=values_array,
                mask=mask_array,
                grid=grid,
                pre_on_hours=pre_ctx.pre_on_hours,
                pre_cum_hours=pre_ctx.pre_cum_hours,
                trigger_flags=pre_ctx.trigger_flags,
                tau_seconds_per_med=tau_seconds_per_med,
                dt_since_last_stop_at_t0_seconds=pre_ctx.dt_since_last_stop_at_t0_seconds,
                config=cfg,
            )

            filepath, feature_names = save_trajectory_tensor(
                traj_key=traj_key,
                values_array=values_array,
                mask_array=mask_array,
                n_intervals=n_intervals,
                interval_seconds=interval_seconds,
                output_dir=output_path,
                med_context=med_context,
                med_feature_names=feature_names,
            )
            if med_feature_names is None:
                med_feature_names = feature_names

            trajectory_metadata[traj_key] = {
                "hadm_id": traj_info["hadm_id"],
                "action_cluster_id": traj_info["action_cluster_id"],
                "t0_time": traj_info["t0_time"],
                "n_intervals": n_intervals,
                "interval_seconds": interval_seconds,
                "trajectory_end_time": traj_info["trajectory_end_time"],
                "duration_minutes": traj_info["duration_minutes"],
                "n_medications": len(unique_item_labels),
                "file_path": filepath,
                "has_data": np.any(mask_array > 0),
                "total_nonzero_values": int(np.sum(mask_array)),
            }

            saved_files.append(filepath)

    # Globals for worker processes (leverages Linux fork COW to share df_full)
    _GLOBAL: Dict[str, t.Any] = {}

    def _worker_init(df_full_: pd.DataFrame, item_labels_: List[str], n_intervals_: int, interval_seconds_: int, output_dir_str_: str, cfg_: MedFeatureConfig) -> None:
        _GLOBAL["DF_FULL"] = df_full_
        _GLOBAL["ITEM_LABELS"] = item_labels_
        _GLOBAL["N_INTERVALS"] = n_intervals_
        _GLOBAL["INTERVAL_SECONDS"] = interval_seconds_
        _GLOBAL["OUTPUT_DIR_STR"] = output_dir_str_
        _GLOBAL["CFG"] = cfg_

    def _process_trajectory_task(task: Tuple[str, Dict]) -> Tuple[str, Dict, List[str]]:
        traj_key, traj_info = task
        item_labels_loc = _GLOBAL["ITEM_LABELS"]
        n_intervals_loc = _GLOBAL["N_INTERVALS"]
        interval_seconds_loc = _GLOBAL["INTERVAL_SECONDS"]
        output_dir_loc = Path(_GLOBAL["OUTPUT_DIR_STR"])
        cfg_loc: MedFeatureConfig = _GLOBAL["CFG"]
        df_full_loc: pd.DataFrame = _GLOBAL["DF_FULL"]

        values_array, mask_array = process_single_trajectory(
            traj_key=traj_key,
            traj_info=traj_info,
            item_labels=item_labels_loc,
            n_intervals=n_intervals_loc,
            interval_seconds=interval_seconds_loc,
        )

        hadm_id = traj_info["hadm_id"]
        t0_time = traj_info["t0_time"]
        patient_full = df_full_loc[df_full_loc["hadm_id"] == hadm_id]
        pre_ctx = build_precontext_from_events(
            events_full_patient=patient_full,
            item_labels=item_labels_loc,
            t0_time=t0_time,
            rate_column="rate/weight_normalized",
        )

        tau_seconds_per_med = build_tau_seconds_per_med(item_labels_loc, cfg_loc)
        grid = TrajectoryGrid(n_intervals=n_intervals_loc, interval_seconds=interval_seconds_loc)
        med_context, feature_names = compute_med_context_features(
            values=values_array,
            mask=mask_array,
            grid=grid,
            pre_on_hours=pre_ctx.pre_on_hours,
            pre_cum_hours=pre_ctx.pre_cum_hours,
            trigger_flags=pre_ctx.trigger_flags,
            tau_seconds_per_med=tau_seconds_per_med,
            dt_since_last_stop_at_t0_seconds=pre_ctx.dt_since_last_stop_at_t0_seconds,
            config=cfg_loc,
        )

        filepath, _ = save_trajectory_tensor(
            traj_key=traj_key,
            values_array=values_array,
            mask_array=mask_array,
            n_intervals=n_intervals_loc,
            interval_seconds=interval_seconds_loc,
            output_dir=output_dir_loc,
            med_context=med_context,
            med_feature_names=feature_names,
        )

        meta = {
            "hadm_id": traj_info["hadm_id"],
            "action_cluster_id": traj_info["action_cluster_id"],
            "t0_time": traj_info["t0_time"],
            "n_intervals": n_intervals_loc,
            "interval_seconds": interval_seconds_loc,
            "trajectory_end_time": traj_info["trajectory_end_time"],
            "duration_minutes": traj_info["duration_minutes"],
            "n_medications": len(item_labels_loc),
            "file_path": filepath,
            "has_data": bool(np.any(mask_array > 0)),
            "total_nonzero_values": int(np.sum(mask_array)),
        }
        return traj_key, meta, feature_names

    if len(pending_keys) == 0:
        print("No pending trajectories to process.")
    elif n_workers is None or n_workers <= 1:
        _run_sequential()
    else:
        ctx = mp.get_context("fork")
        cfg = MedFeatureConfig()
        with ctx.Pool(
            processes=n_workers,
            initializer=_worker_init,
            initargs=(df_full, unique_item_labels, n_intervals, interval_seconds, str(output_path), cfg),
        ) as pool:
            tasks_iter = [(k, trajectories[k]) for k in pending_keys]
            for traj_key, meta, feature_names in tqdm(
                pool.imap_unordered(_process_trajectory_task, tasks_iter),
                total=len(tasks_iter),
                desc="Creating tensors",
            ):
                trajectory_metadata[traj_key] = meta
                saved_files.append(meta["file_path"])
                if med_feature_names is None:
                    med_feature_names = feature_names

    # Step 6: Save metadata
    metadata = {
        "trajectories": trajectory_metadata,
        "item_labels": unique_item_labels,
        "n_intervals": n_intervals,
        "interval_seconds": interval_seconds,
        "total_trajectories": len(trajectories),
        "created_at": datetime.now().isoformat(),
        "source_file": parquet_path,
        "med_feature_names": med_feature_names,
    }

    # Step 6: Save metadata (merge with existing)
    final_metadata = {
        "trajectories": trajectory_metadata,
        "item_labels": existing_metadata.get("item_labels", unique_item_labels) if isinstance(existing_metadata, dict) else unique_item_labels,
        "n_intervals": n_intervals,
        "interval_seconds": interval_seconds,
        "total_trajectories": len(trajectories),
        "created_at": datetime.now().isoformat(),
        "source_file": parquet_path,
        "med_feature_names": med_feature_names if med_feature_names is not None else existing_metadata.get("med_feature_names") if isinstance(existing_metadata, dict) else None,
    }

    with open(metadata_file, "wb") as f:
        pickle.dump(final_metadata, f)

    # Step 7: Save summary statistics
    summary_stats = {
        "total_trajectories": len(trajectories),
        "total_tensor_files": len(saved_files),
        "unique_hadm_ids": df_triggers["hadm_id"].nunique(),
        "unique_action_cluster_ids": df_triggers["action_cluster_id"].nunique(),
        "unique_item_labels": len(unique_item_labels),
        "time_grid_intervals": n_intervals,
        "interval_seconds": interval_seconds,
        "max_duration_hours": grid_params["max_duration"] / 3600.0,
    }

    print("\n=== Summary ===")
    for key, value in summary_stats.items():
        print(f"{key}: {value}")

    print(f"\nTensor files saved to: {output_path}")
    print(f"Metadata saved to: {metadata_file}")

    return final_metadata


def load_and_inspect_tensor(tensor_path: str) -> None:
    """
    Load and inspect a saved tensor file.
    """
    print(f"Loading tensor from: {tensor_path}")

    loaded = torch.load(tensor_path)
    if isinstance(loaded, tuple) and len(loaded) >= 5:
        # New tuple optionally includes med_context at position 5
        values_tensor = loaded[0]
        mask_tensor = loaded[1]
        time_seconds_tensor = loaded[2]
        time_hours_tensor = loaded[3]
        n_intervals = loaded[4]
        med_context_tensor = loaded[5] if len(loaded) > 5 else None
    else:
        raise ValueError("Unexpected tensor file format")

    print(f"Values tensor shape: {values_tensor.shape}")
    print(f"Mask tensor shape: {mask_tensor.shape}")
    print(f"Time intervals: {n_intervals}")
    print(f"Duration: {time_hours_tensor[-1]:.2f} hours")
    print(f"Non-zero values: {torch.sum(mask_tensor).item()}")
    print(f"Medications with data: {torch.sum(torch.any(mask_tensor > 0, dim=0)).item()}")
    if med_context_tensor is not None:
        print(f"Med context shape: {tuple(med_context_tensor.shape)}")


# Example usage
if __name__ == "__main__":
    # Example usage - adjust paths as needed
    parquet_path = "mv_filtered_10min.parquet"
    output_dir = "data/med_tensors_output"

    if os.path.exists(parquet_path):
        metadata = create_med_tensors_from_parquet(
            parquet_path=parquet_path, output_dir=output_dir, interval_seconds=10
        )

        # Inspect a sample tensor
        if metadata["trajectories"]:
            sample_traj_key = list(metadata["trajectories"].keys())[0]
            sample_file = metadata["trajectories"][sample_traj_key]["file_path"]
            print("\n=== Sample Tensor Inspection ===")
            load_and_inspect_tensor(sample_file)
    else:
        print(f"Please ensure {parquet_path} exists in the current directory")
        print("You can modify the parquet_path variable to point to your file location")
