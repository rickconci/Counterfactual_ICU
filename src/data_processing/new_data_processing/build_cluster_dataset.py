
import argparse
import json
import math
import re
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Any

import numpy as np
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


HADM_RE = re.compile(r"full_waveform_(\d+)(?:_session_\d+)?\.parquet$", re.IGNORECASE)


# ------------------------ Time helpers ------------------------

def to_utc(s: pd.Series) -> pd.Series:
    """Coerce any datetime-like to timezone-aware UTC (tz-naive -> assume UTC)."""
    return pd.to_datetime(s, errors="coerce", utc=True)


def time_grid(start: pd.Timestamp, minutes: int, dt_sec: int) -> List[pd.Timestamp]:
    steps = int((minutes * 60) // dt_sec)
    return [start + pd.Timedelta(seconds=i * dt_sec) for i in range(steps)]


def bins_before_t0(t0: pd.Timestamp, minutes: int = 60, bins: int = 6) -> List[Tuple[pd.Timestamp, pd.Timestamp]]:
    """Return bins of equal width that partition [t0-minutes, t0) in chronological order."""
    bin_len = minutes // bins
    out = []
    for i in range(bins):
        start = t0 - pd.Timedelta(minutes=minutes - i * bin_len)
        end = t0 - pd.Timedelta(minutes=minutes - (i + 1) * bin_len)
        out.append((start, end))
    return out


# ------------------------ Normalization ------------------------

def build_drug_means(triggers: pd.DataFrame, dose_col: str = "rate/weight", label_col: str = "item_label") -> Dict[str, float]:
    g = triggers[[label_col, dose_col]].copy()
    g[label_col] = g[label_col].astype(str).str.strip()
    g[dose_col] = pd.to_numeric(g[dose_col], errors="coerce")
    g = g.dropna(subset=[label_col, dose_col])
    means = g.groupby(label_col)[dose_col].mean().to_dict()
    return means  # drug -> mean dose


# ------------------------ Waveform QC + smoothing ------------------------

def _smooth_1d_nanaware(arr: np.ndarray, neighbors: int) -> np.ndarray:
    """Centered moving average with window 2K+1, NaN-aware (does not bridge gaps)."""
    if neighbors <= 0 or arr.size == 0:
        return arr.copy()
    win_len = 2*neighbors + 1
    kernel = np.ones(win_len, dtype=float)
    vals = arr.astype(float)
    valid = np.isfinite(vals).astype(float)
    # replace NaN with 0 for numerator
    num = np.convolve(np.nan_to_num(vals, nan=0.0), kernel, mode="same")
    den = np.convolve(valid, kernel, mode="same")
    out = np.divide(num, den, out=np.full_like(num, np.nan), where=den > 0)
    return out


def qc_and_smooth_waveform(
    wf: pd.DataFrame,
    vars_of_interest: Sequence[str],
    cvp_min: float = 0.0,
    cvp_max: float = 40.0,
    smooth_neighbors: int = 1,
) -> pd.DataFrame:
    """
    - Coerce absolute_timestamp to UTC and numeric types for variables.
    - Apply CVP QC: values <cvp_min or >cvp_max -> NaN (mask downstream will mark missing as 1).
    - Smooth each variable column with centered mean over (2*neighbors+1) points, NaN-aware.
    """
    if wf.empty:
        return wf
    wf = wf.copy()
    wf["absolute_timestamp"] = to_utc(wf["absolute_timestamp"])
    wf = wf.sort_values("absolute_timestamp")

    # numeric coercion
    for v in vars_of_interest:
        if v in wf.columns:
            wf[v] = pd.to_numeric(wf[v], errors="coerce")

    # CVP QC
    if "CVP" in wf.columns:
        bad = (wf["CVP"] < cvp_min) | (wf["CVP"] > cvp_max)
        wf.loc[bad, "CVP"] = np.nan

    # Smoothing
    if smooth_neighbors > 0:
        for v in vars_of_interest:
            if v in wf.columns:
                wf[v] = _smooth_1d_nanaware(wf[v].to_numpy(), neighbors=smooth_neighbors)

    return wf


# ------------------------ Waveform I/O ------------------------

def waveform_bounds(path: Path) -> Optional[Tuple[pd.Timestamp, pd.Timestamp]]:
    try:
        df = pd.read_parquet(path, columns=["record_start_time", "time_seconds"])
    except Exception:
        df = pd.read_parquet(path)
        if "record_start_time" not in df or "time_seconds" not in df:
            return None
    rst = to_utc(df["record_start_time"])
    tsecs = pd.to_numeric(df["time_seconds"], errors="coerce")
    ts = rst + pd.to_timedelta(tsecs, unit="s")
    if len(ts) == 0:
        return None
    return ts.min(), ts.max()


def read_waveform_window(path: Path, start: pd.Timestamp, end: pd.Timestamp, vars_of_interest: Sequence[str]) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df["record_start_time"] = to_utc(df["record_start_time"])
    df["time_seconds"] = pd.to_numeric(df["time_seconds"], errors="coerce")
    df["absolute_timestamp"] = df["record_start_time"] + pd.to_timedelta(df["time_seconds"], unit="s")

    keep = (df["absolute_timestamp"] >= start) & (df["absolute_timestamp"] < end)

    # Keep any of the requested columns that are present
    present = [v for v in vars_of_interest if v in df.columns]
    dfw = df.loc[keep, ["absolute_timestamp"] + present].copy()

    # Add any missing requested columns as NaN
    for v in vars_of_interest:
        if v not in dfw.columns:
            dfw[v] = np.nan

    # Reorder to fixed schema and sort by time
    dfw = dfw[["absolute_timestamp"] + list(vars_of_interest)].sort_values("absolute_timestamp")
    return dfw


def pick_best_waveform_file(wf_dir: Path, hadm_id: str, window: Tuple[pd.Timestamp, pd.Timestamp]) -> Optional[Path]:
    files = sorted(wf_dir.glob(f"full_waveform_{hadm_id}.parquet"))
    files += sorted(wf_dir.glob(f"full_waveform_{hadm_id}_session_*.parquet"))
    if not files:
        return None
    wstart, wend = window
    best = None
    best_overlap = -1.0
    for p in files:
        b = waveform_bounds(p)
        if b is None:
            continue
        s, e = b
        inter_start = max(s, wstart)
        inter_end = min(e, wend)
        overlap = (inter_end - inter_start).total_seconds() if inter_end > inter_start else 0.0
        if overlap > best_overlap:
            best = p
            best_overlap = overlap
    return best


# ------------------------ Feature builders ------------------------

def initial_conditions_at_t0(wf: pd.DataFrame, t0: pd.Timestamp, vars_of_interest: Sequence[str], require_exact: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    vals = []
    mask = []
    wf_by_time = wf.set_index("absolute_timestamp")
    for v in vars_of_interest:
        if v not in wf_by_time.columns:
            vals.append(0.0); mask.append(1); continue
        if require_exact:
            # exact timestamp only
            if t0 in wf_by_time.index:
                val = wf_by_time.at[t0, v]
                vals.append(float(val) if pd.notna(val) else 0.0)
                mask.append(0 if pd.notna(val) else 1)
            else:
                vals.append(0.0); mask.append(1)
        else:
            # nearest within a 3-minute window
            start_time = t0 - pd.Timedelta(minutes=3)
            end_time = t0 + pd.Timedelta(minutes=3)
            subset = wf_by_time.loc[start_time:end_time, v].dropna()
            if not subset.empty:
                closest_time = subset.index[np.abs(subset.index - t0).argmin()]
                val = subset.loc[closest_time]
                if isinstance(val, pd.Series):
                    val = val.iloc[0]
                vals.append(float(val) if pd.notna(val) else 0.0)
                mask.append(0 if pd.notna(val) else 1)
            else:
                vals.append(0.0); mask.append(1)
    return np.array(vals, dtype="float32"), np.array(mask, dtype="int8")


def avg_previous_input(wf: pd.DataFrame, t0: pd.Timestamp, vars_of_interest: Sequence[str], minutes: int = 60, bins: int = 6) -> Tuple[np.ndarray, np.ndarray]:
    wf_by_time = wf.set_index("absolute_timestamp")
    out = np.zeros((len(vars_of_interest), bins), dtype="float32")
    mask = np.ones((len(vars_of_interest), bins), dtype="int8")
    intervals = bins_before_t0(t0, minutes=minutes, bins=bins)
    for ci, v in enumerate(vars_of_interest):
        if v not in wf_by_time.columns:
            continue
        for bi, (s, e) in enumerate(intervals):
            seg = wf_by_time.loc[(wf_by_time.index >= s) & (wf_by_time.index < e), v]
            if len(seg) > 0:
                val = float(seg.mean())
                out[ci, bi] = val
                mask[ci, bi] = 0
    return out, mask


def response_future(wf: pd.DataFrame, t0: pd.Timestamp, vars_of_interest: Sequence[str], minutes: int, dt_sec: int) -> Tuple[np.ndarray, np.ndarray]:
    times = time_grid(t0, minutes=minutes, dt_sec=dt_sec)
    T = len(times)
    out = np.zeros((len(vars_of_interest), T), dtype="float32")
    mask = np.ones((len(vars_of_interest), T), dtype="int8")

    wf_by_time = wf.set_index("absolute_timestamp").sort_index()
    tol = pd.Timedelta(seconds=dt_sec)  # nearest within ± dt_sec

    # Build aligned frame with nearest reindex per variable
    aligned = pd.DataFrame(index=pd.Index(times, name="absolute_timestamp"))
    for v in vars_of_interest:
        if v in wf_by_time.columns:
            s_aln = wf_by_time[v].reindex(aligned.index, method="nearest", tolerance=tol)
            aligned[v] = s_aln
        else:
            aligned[v] = np.nan

    for ci, v in enumerate(vars_of_interest):
        vals = aligned[v].to_numpy()
        out[ci, :] = np.nan_to_num(vals, nan=0.0).astype("float32")
        mask[ci, :] = np.isnan(vals).astype("int8")

    return out, mask


def treatments_grid(
    cluster_rows: pd.DataFrame,
    t0: pd.Timestamp,
    minutes: int,
    dt_sec: int,
    k_max: int,
    drug_means: Dict[str, float],
    label_col: str = "item_label",
    dose_col: str = "rate/weight",
) -> Tuple[np.ndarray, List[Optional[str]]]:
    """
    Build a (k_max, T) array of zero-mean normalized doses sampled every dt_sec for 'minutes' after t0.
    """
    t_end = t0 + pd.Timedelta(minutes=minutes)
    g = cluster_rows.copy()
    g["start_time"] = to_utc(g["start_time"])
    g["end_time"] = to_utc(g["end_time"])
    g[label_col] = g[label_col].astype(str).str.strip()
    g[dose_col] = pd.to_numeric(g[dose_col], errors="coerce")

    # Compute overlap duration per drug
    overlap_sec: Dict[str, float] = {}
    for _, r in g.iterrows():
        lab = r[label_col]
        s = max(r["start_time"], t0)
        e = min(r["end_time"], t_end)
        if pd.isna(s) or pd.isna(e) or e <= s or lab == "":
            continue
        sec = (e - s).total_seconds()
        overlap_sec[lab] = overlap_sec.get(lab, 0.0) + sec

    top_drugs = sorted(overlap_sec.items(), key=lambda x: (-x[1], x[0].lower()))
    top_drugs = [lab for lab, _ in top_drugs[:k_max]]
    legend: List[Optional[str]] = top_drugs + [None] * (k_max - len(top_drugs))

    times = time_grid(t0, minutes=minutes, dt_sec=dt_sec)
    T = len(times)
    out = np.zeros((k_max, T), dtype="float32")

    by_drug: Dict[str, pd.DataFrame] = {lab: g[g[label_col] == lab].sort_values("start_time") for lab in top_drugs}

    for row_idx, lab in enumerate(top_drugs):
        rows = by_drug[lab]
        mu = drug_means.get(lab, 0.0)
        j = 0
        for ti, t in enumerate(times):
            while j < len(rows) and rows.iloc[j]["end_time"] <= t:
                j += 1
            if j < len(rows):
                r = rows.iloc[j]
                if r["start_time"] <= t < r["end_time"] and pd.notna(r[dose_col]):
                    out[row_idx, ti] = float(r[dose_col]) - float(mu)
    return out, legend


def _create_debug_plot(
    sample_id: str,
    status: str,
    wf: pd.DataFrame,
    cluster_rows: pd.DataFrame,
    initial_conditions: np.ndarray,
    t0: pd.Timestamp,
    vars_of_interest: List[str],
    cfg: "Config",
    out_dir: Path,
):
    """Generate a multi-panel plot inspired by PanelStack for debugging a single sample."""
    n_vars = len(vars_of_interest)
    fig = plt.figure(figsize=(15, 4 * (n_vars + 1)))
    gs = fig.add_gridspec(n_vars + 1, 1, hspace=0.1, height_ratios=[1] * n_vars + [0.8])

    share_ax = None

    # --- Waveform Panels ---
    for i, var in enumerate(vars_of_interest):
        ax = fig.add_subplot(gs[i, 0], sharex=share_ax)
        if share_ax is None:
            share_ax = ax
        
        if not wf.empty:
            ax.plot(wf["absolute_timestamp"], wf[var], label=var, color='royalblue', alpha=0.9, linewidth=1.5)
            
            # Shade history and future windows
            hist_start = t0 - pd.Timedelta(minutes=cfg.history_minutes)
            future_end = t0 + pd.Timedelta(minutes=cfg.future_minutes)
            ax.axvspan(hist_start, t0, color='orange', alpha=0.1, label='History Window')
            ax.axvspan(t0, future_end, color='green', alpha=0.1, label='Future Window')

            ax.axvline(t0, color='red', linestyle='--', label='t0')
            
            # Plot the selected initial condition point
            ic_val = initial_conditions[i] if i < len(initial_conditions) else np.nan
            if pd.notna(ic_val) and ic_val != 0:
                # Find the actual time of the IC point to plot it accurately
                time_window_mask = (wf['absolute_timestamp'] >= t0 - pd.Timedelta(minutes=3)) & \
                                   (wf['absolute_timestamp'] <= t0 + pd.Timedelta(minutes=3))
                relevant_points = wf.loc[time_window_mask, ['absolute_timestamp', var]].dropna()
                if not relevant_points.empty:
                    time_of_ic = relevant_points.iloc[(relevant_points[var] - ic_val).abs().argmin()]['absolute_timestamp']
                    ax.scatter([time_of_ic], [ic_val], color='red', s=60, zorder=5, label='Initial Condition')

        ax.set_ylabel(var)
        ax.legend(loc="upper right")
        ax.grid(True, which='both', linestyle=':', linewidth=0.5)
        if i < n_vars -1:
            ax.tick_params(axis='x', labelbottom=False)

    # --- Treatment Timeline Panel ---
    ax = fig.add_subplot(gs[n_vars, 0], sharex=share_ax)
    if not cluster_rows.empty:
        unique_drugs = sorted(cluster_rows['item_label'].unique())
        drug_to_y = {drug: i for i, drug in enumerate(unique_drugs)}
        
        for _, row in cluster_rows.iterrows():
            y = drug_to_y[row['item_label']]
            ax.plot([row['start_time'], row['end_time']], [y, y], linewidth=5, solid_capstyle='butt')

        ax.set_yticks(range(len(unique_drugs)))
        ax.set_yticklabels(unique_drugs, fontsize=9)
        ax.set_ylim(-0.5, len(unique_drugs) - 0.5)

    ax.set_ylabel("Treatments")
    ax.grid(True, axis='x', which='both', linestyle=':', linewidth=0.5)

    # --- Formatting ---
    if share_ax:
        share_ax.set_xlim(
            t0 - pd.Timedelta(minutes=cfg.history_minutes + 5),
            t0 + pd.Timedelta(minutes=cfg.future_minutes + 5)
        )
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
    plt.xlabel("Time")
    fig.suptitle(f"Sample: {sample_id} | Status: {status}", fontsize=16, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.98])

    # --- Save ---
    plot_dir = out_dir / "accepted_plots" if status == "ok" else out_dir / "rejected_plots"
    fig.savefig(plot_dir / f"{sample_id}.png", dpi=120)
    plt.close(fig)


def _create_rejected_plot(
    sample_id: str,
    status: str,
    wf_path: Path,
    t0: pd.Timestamp,
    vars_of_interest: List[str],
    out_dir: Path,
):
    """Generate a plot of the full waveform trajectory for a rejected sample."""
    fig, ax = plt.subplots(figsize=(15, 6))
    
    try:
        wf = pd.read_parquet(wf_path)
        wf['absolute_timestamp'] = to_utc(wf.get('record_start_time')) + pd.to_timedelta(wf['time_seconds'], unit='s')

        for var in vars_of_interest:
            if var in wf.columns and pd.api.types.is_numeric_dtype(wf[var]):
                ax.plot(wf['absolute_timestamp'], wf[var], label=var, alpha=0.7)
        
        ax.axvline(t0, color='red', linestyle='--', linewidth=2, label='t0')
        ax.legend()
        ax.grid(True, linestyle=':')
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
        plt.setp(ax.get_xticklabels(), rotation=30, ha='right')

    except Exception as e:
        ax.text(0.5, 0.5, f"Could not load waveform data:\n{e}", ha='center', va='center')

    fig.suptitle(f"Rejected Sample: {sample_id} | Status: {status}", fontsize=16, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    
    plot_dir = out_dir / "rejected_plots"
    fig.savefig(plot_dir / f"{sample_id}.png", dpi=120)
    plt.close(fig)


# ------------------------ Main pipeline (parallel + resumable + discard masked) ------------------------

@dataclass
class Config:
    vars: List[str]
    k_max: int = 3
    dt_sec: int = 10
    history_minutes: int = 60
    future_minutes: int = 60
    cvp_min: float = 0.0
    cvp_max: float = 40.0
    smooth_neighbors: int = 1


def read_input(path: Path) -> pd.DataFrame:
    ext = path.suffix.lower()
    if ext in [".parquet", ".pq"]:
        return pd.read_parquet(path)
    elif ext in [".pkl", ".pickle"]:
        return pd.read_pickle(path)
    else:
        return pd.read_csv(path)


def _process_task(task: Dict[str, Any]) -> Dict[str, Any]:
    """Worker: process one cluster into an NPZ and manifest row. Discard all-masked initial/response."""
    sid = task["subject_id"]
    hid = task["hadm_id"]
    cid = task["action_cluster_id"]
    t0  = pd.Timestamp(task["t0"], tz="UTC")
    cfg_d = task["cfg"]
    cfg = Config(**cfg_d)

    print(f"[{hid}_{cid}] STARTING")

    wf_dir = Path(task["wf_dir"])
    out_dir = Path(task["out_dir"])
    samples_dir = out_dir / "samples"
    samples_dir.mkdir(exist_ok=True)

    vars_of_interest = task["vars"]
    drug_means = task["drug_means"]

    # recreate cluster rows DataFrame
    rows = pd.DataFrame.from_records(task["rows"])
    rows["start_time"] = to_utc(rows["start_time"])
    rows["end_time"] = to_utc(rows["end_time"])

    # required window
    req_start = t0 - pd.Timedelta(minutes=cfg.history_minutes)
    req_end   = t0 + pd.Timedelta(minutes=cfg.future_minutes)

    wf_path = pick_best_waveform_file(wf_dir, str(hid), (req_start, req_end))
    print(f"[{hid}_{cid}] Picked waveform file: {wf_path}")

    if wf_path is None:
        print(f"[{hid}_{cid}] No waveform file found, returning.")
        man = {
            "sample_id": f"{hid}_{cid}",
            "subject_id": sid, "hadm_id": hid, "action_cluster_id": cid,
            "t0": t0.isoformat(),
            "waveform_file": None,
            "status": "no_waveform_file"
        }
        if task["debug_plots"]:
            # Can't plot waveform if we couldn't find a file
            pass
        return man

    # Pre-check: Is t0 even within the bounds of the selected waveform file?
    bounds = waveform_bounds(wf_path)
    if bounds:
        wf_start, wf_end = bounds
        if not (wf_start <= t0 < wf_end):
            print(f"[{hid}_{cid}] t0 {t0} is outside waveform bounds [{wf_start}, {wf_end}], skipping.")
            man = {
                "sample_id": f"{hid}_{cid}",
                "subject_id": sid, "hadm_id": hid, "action_cluster_id": cid,
                "t0": t0.isoformat(),
                "waveform_file": wf_path.name,
                "status": "skipped_t0_outside_bounds"
            }
            if task["debug_plots"]:
                _create_rejected_plot(f"{hid}_{cid}", "skipped_t0_outside_bounds", wf_path, t0, vars_of_interest, out_dir)
            return man


    wf = read_waveform_window(wf_path, req_start, req_end, vars_of_interest)
    print(f"[{hid}_{cid}] Read waveform window, shape={wf.shape}")

    print(f"[{hid}_{cid}] Smoothing...")
    wf = qc_and_smooth_waveform(
        wf, vars_of_interest,
        cvp_min=cfg.cvp_min, cvp_max=cfg.cvp_max, smooth_neighbors=cfg.smooth_neighbors
    )
    print(f"[{hid}_{cid}] Smoothed.")

    # features
    print(f"[{hid}_{cid}] Building initial conditions...")
    init, init_mask = initial_conditions_at_t0(wf, t0, vars_of_interest, require_exact=False)
    # Discard if all initial conditions masked
    if int(init_mask.sum()) == int(init_mask.size):
        print(f"[{hid}_{cid}] All initial masked, skipping.")
        man = {
            "sample_id": f"{hid}_{cid}",
            "subject_id": sid, "hadm_id": hid, "action_cluster_id": cid,
            "t0": t0.isoformat(),
            "waveform_file": wf_path.name,
            "status": "skipped_all_initial_masked"
        }
        if task["debug_plots"]:
            _create_rejected_plot(f"{hid}_{cid}", "skipped_all_initial_masked", wf_path, t0, vars_of_interest, out_dir)
        return man

    print(f"[{hid}_{cid}] Building previous inputs...")
    prev_avg, prev_mask = avg_previous_input(wf, t0, vars_of_interest, minutes=cfg.history_minutes, bins=6)
    print(f"[{hid}_{cid}] Building future response...")
    resp, resp_mask = response_future(wf, t0, vars_of_interest, minutes=cfg.future_minutes, dt_sec=cfg.dt_sec)

    # Discard if both ABP MEAN and CVP are fully masked in response
    abp_fully_masked = "ABP MEAN" not in vars_of_interest
    if not abp_fully_masked:
        abp_idx = vars_of_interest.index("ABP MEAN")
        abp_fully_masked = resp_mask[abp_idx].all()

    cvp_fully_masked = "CVP" not in vars_of_interest
    if not cvp_fully_masked:
        cvp_idx = vars_of_interest.index("CVP")
        cvp_fully_masked = resp_mask[cvp_idx].all()

    if abp_fully_masked and cvp_fully_masked:
        print(f"[{hid}_{cid}] ABP and CVP fully masked, skipping.")
        man = {
            "sample_id": f"{hid}_{cid}",
            "subject_id": sid, "hadm_id": hid, "action_cluster_id": cid,
            "t0": t0.isoformat(),
            "waveform_file": wf_path.name,
            "status": "skipped_abp_cvp_response_masked"
        }
        if task["debug_plots"]:
             _create_rejected_plot(f"{hid}_{cid}", "skipped_abp_cvp_response_masked", wf_path, t0, vars_of_interest, out_dir)
        return man

    print(f"[{hid}_{cid}] Building treatments grid...")
    treats, legend = treatments_grid(
        cluster_rows=rows, t0=t0, minutes=cfg.future_minutes, dt_sec=cfg.dt_sec,
        k_max=cfg.k_max, drug_means=drug_means, label_col=task["label_col"], dose_col=task["dose_col"]
    )
    print(f"[{hid}_{cid}] Treatments grid built.")

    # Save NPZ
    sample_id = f"{hid}_{cid}"
    npz_path = samples_dir / f"{sample_id}.npz"
    print(f"[{hid}_{cid}] Saving to {npz_path}...")
    np.savez_compressed(
        npz_path,
        initial=init,
        initial_mask=init_mask,
        prev_avg=prev_avg,
        prev_avg_mask=prev_mask,
        response=resp,
        response_mask=resp_mask,
        treatments=treats,
        vars=np.array(vars_of_interest, dtype=str),
    )
    print(f"[{hid}_{cid}] Saved.")

    man = {
        "sample_id": sample_id,
        "subject_id": sid,
        "hadm_id": hid,
        "action_cluster_id": cid,
        "t0": t0.isoformat(),
        "waveform_file": wf_path.name,
        "vars": "|".join(vars_of_interest),
        "k_max": cfg.k_max,
        "dt_sec": cfg.dt_sec,
        "history_minutes": cfg.history_minutes,
        "future_minutes": cfg.future_minutes,
        "cvp_min": cfg.cvp_min,
        "cvp_max": cfg.cvp_max,
        "smooth_neighbors": cfg.smooth_neighbors,
        "status": "ok",
    }
    # Add legend_row0..legend_row{k_max-1}
    for i in range(cfg.k_max):
        man[f"legend_row{i}"] = legend[i] if i < len(legend) else None

    if task["debug_plots"]:
        _create_debug_plot(sample_id, "ok", wf, rows, init, t0, vars_of_interest, cfg, out_dir)

    print(f"[{hid}_{cid}] FINISHED OK")
    return man


def main():
    ap = argparse.ArgumentParser(description="Precompute cluster-centered dataset tensors (parallel + resumable). Includes CVP QC + smoothing.")
    ap.add_argument("--triggers", required=True, type=Path, help="Path to input_mv_triggers (CSV/Parquet/Pickle).")
    ap.add_argument("--waveforms", required=True, type=Path, help="Dir with full_trajectories_combined parquet files.")
    ap.add_argument("--out", required=True, type=Path, help="Output directory for dataset.")
    ap.add_argument("--vars", required=True, type=str, help="Comma-separated waveform variable names, e.g., 'HR,ABPMean,SpO2'.")
    ap.add_argument("--k-max", type=int, default=3, help="Max concurrent treatments rows.")
    ap.add_argument("--dt-sec", type=int, default=10, help="Sampling step in seconds for treatments/response.")
    ap.add_argument("--history-min", type=int, default=60, help="Minutes before t0 used for averages.")
    ap.add_argument("--future-min", type=int, default=60, help="Minutes after t0 for response/treatments.")
    ap.add_argument("--workers", type=int, default=4, help="Number of parallel workers.")
    ap.add_argument("--overwrite", action="store_true", help="Recompute and overwrite existing NPZ samples.")
    ap.add_argument("--cvp-min", type=float, default=0.0, help="CVP values below this are set to NaN (default: 0).")
    ap.add_argument("--cvp-max", type=float, default=40.0, help="CVP values above this are set to NaN (default: 40).")
    ap.add_argument("--smooth-neighbors", type=int, default=1, help="Centered moving-average neighbors (2K+1 window). 0 disables smoothing.")
    ap.add_argument("--debug-plots", action="store_true", help="Generate and save a debug plot for every sample.")
    args = ap.parse_args()

    # Load triggers
    def read_input(path: Path) -> pd.DataFrame:
        ext = path.suffix.lower()
        if ext in [".parquet", ".pq"]:
            return pd.read_parquet(path)
        elif ext in [".pkl", ".pickle"]:
            return pd.read_pickle(path)
        else:
            return pd.read_csv(path)

    triggers = read_input(args.triggers)
    # Ensure time columns are comparable
    for c in ("start_time", "end_time"):
        if c in triggers.columns:
            triggers[c] = to_utc(triggers[c])

    # Config & dirs
    cfg = Config(
        vars=[v.strip() for v in args.vars.split(",") if v.strip()],
        k_max=args.k_max,
        dt_sec=args.dt_sec,
        history_minutes=args.history_min,
        future_minutes=args.future_min,
        cvp_min=args.cvp_min,
        cvp_max=args.cvp_max,
        smooth_neighbors=args.smooth_neighbors,
    )
    out_dir = args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    samples_dir = out_dir / "samples"
    samples_dir.mkdir(exist_ok=True)
    if args.debug_plots:
        (out_dir / "accepted_plots").mkdir(exist_ok=True)
        (out_dir / "rejected_plots").mkdir(exist_ok=True)

    # Build drug means for zero-centering
    drug_means = build_drug_means(triggers, dose_col="rate/weight", label_col="item_label")
    with open(out_dir / "drug_means.json", "w") as fh:
        json.dump(drug_means, fh, indent=2)

    # Load existing manifest (to support resume and skip known-skipped)
    manifest_path = out_dir / "dataset_manifest.csv"
    if manifest_path.exists():
        old_manifest = pd.read_csv(manifest_path)
    else:
        legend_cols = [f"legend_row{i}" for i in range(cfg.k_max)]
        old_manifest = pd.DataFrame(columns=[
            "sample_id","subject_id","hadm_id","action_cluster_id","t0","waveform_file",
            "vars","k_max","dt_sec","history_minutes","future_minutes",
            *legend_cols,
            "cvp_min","cvp_max","smooth_neighbors","status"
        ])

    # Index existing statuses for resume logic
    existing_status = {}
    if not old_manifest.empty and "sample_id" in old_manifest.columns and "status" in old_manifest.columns:
        existing_status = dict(zip(old_manifest["sample_id"].astype(str), old_manifest["status"].astype(str)))

    # Group clusters
    if "action_cluster_id" not in triggers.columns:
        raise ValueError("Expected 'action_cluster_id' column in triggers dataframe.")
    gkeys = ["subject_id", "hadm_id", "action_cluster_id"]
    clusters = triggers[triggers["action_cluster_id"].notna()].groupby(gkeys, dropna=False)

    # Build task list (skip already done unless overwrite; also skip known-skipped in manifest unless overwrite)
    SKIP_STATUSES = {
        "ok", "no_waveform_file", "skipped_all_initial_masked",
        "skipped_abp_cvp_response_masked", "skipped_t0_outside_bounds"
    }
    tasks: List[Dict[str, Any]] = []
    for (sid, hid, cid), g in clusters:
        sample_id = f"{hid}_{cid}"
        npz_path = samples_dir / f"{sample_id}.npz"
        if not args.overwrite:
            if npz_path.exists():
                continue  # already built
            if existing_status.get(sample_id) in SKIP_STATUSES:
                # known built or skipped; avoid re-attempting unless --overwrite
                continue
        t0 = g["start_time"].min()
        rows = g[["start_time", "end_time", "item_label", "rate/weight"]].to_dict("records")
        task = {
            "subject_id": sid,
            "hadm_id": str(hid),
            "action_cluster_id": int(cid),
            "t0": t0.isoformat() if pd.notna(t0) else pd.Timestamp("1970-01-01", tz="UTC").isoformat(),
            "vars": cfg.vars,
            "cfg": asdict(cfg),
            "wf_dir": str(args.waveforms),
            "out_dir": str(out_dir),
            "drug_means": drug_means,
            "label_col": "item_label",
            "dose_col": "rate/weight",
            "rows": rows,
            "debug_plots": args.debug_plots,
        }
        tasks.append(task)

    # Parallel processing
    new_rows: List[Dict[str, Any]] = []
    if tasks:
        with ProcessPoolExecutor(max_workers=args.workers) as ex, tqdm(total=len(tasks), desc="Building samples", unit="cluster") as pbar:
            futs = [ex.submit(_process_task, t) for t in tasks]
            for fut in as_completed(futs):
                man = fut.result()
                new_rows.append(man)
                pbar.update(1)
    else:
        print("Nothing to do (all samples exist or are known-skipped). Use --overwrite to recompute.")

    # Merge manifests (prefer new rows on conflicts)
    if new_rows:
        new_manifest = pd.DataFrame(new_rows)
        combined = pd.concat([old_manifest, new_manifest], ignore_index=True)
        combined.sort_values("sample_id", inplace=True)
        combined = combined.drop_duplicates(subset=["sample_id"], keep="last")
    else:
        combined = old_manifest

    combined.to_csv(manifest_path, index=False)
    print(f"Wrote manifest to: {manifest_path} (rows={len(combined)})")

    # Summary
    counts = combined["status"].value_counts(dropna=False).to_dict() if "status" in combined.columns else {}
    print("--- Summary by status ---")
    for k, v in counts.items():
        print(f"{k}: {v}")
    print("-------------------------")

    print("Done.")


if __name__ == "__main__":
    main()
