import numpy as np
import pandas as pd
from typing import Dict, Tuple, Iterable

CANON_ABP_COL = "ABP_MEAN"
PHYSIO_COLS = [CANON_ABP_COL, "CVP", "HR", "RESP", "SpO2", "CO", "PULSE"]

def downsample_every_10s_mean(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    need = {"hadm_id","record_name","absolute_timestamp"}
    if not need.issubset(df.columns):
        missing = need - set(df.columns)
        raise KeyError(f"Missing required columns for resampling: {missing}")

    d = df.copy()
    d["absolute_timestamp"] = pd.to_datetime(d["absolute_timestamp"], errors="coerce", utc=True)
    # numeric cols to aggregate
    num_cols = [c for c in [CANON_ABP_COL,"CVP","HR","RESP","SpO2","CO","PULSE"] if c in d.columns]

    out_frames = []
    for (hadm, rec), g in d.groupby(["hadm_id","record_name"], sort=False, dropna=False):
        g = g.set_index("absolute_timestamp").sort_index()
        # resample
        agg = g[num_cols].resample("10s").mean()
        agg["hadm_id"] = hadm
        agg["record_name"] = rec
        # bring back simple meta if present (optional; you can drop these)
        if "record_start_time" in g.columns:
            agg["record_start_time"] = g["record_start_time"].iloc[0]
        if "record_end_time" in g.columns:
            agg["record_end_time"] = g["record_end_time"].iloc[-1]
        out_frames.append(agg.reset_index())

    ds = pd.concat(out_frames, ignore_index=True)
    ds = ds.sort_values(["hadm_id","record_name","absolute_timestamp"]).reset_index(drop=True)
    return ds

def clean_waveform_df(df: pd.DataFrame, drop_all_nan_physio: bool = True) -> pd.DataFrame:
    """
    Out-of-range -> NaN on canonical columns, then (optionally) drop rows where
    all physio columns are NaN.
    Bounds are easy to tweak below.
    """
    d = df.copy()

    # Coerce numeric where present
    for c in PHYSIO_COLS:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")

    # Basic bounds (edit if you prefer):
    if CANON_ABP_COL in d:
        d.loc[(d[CANON_ABP_COL] < 40) | (d[CANON_ABP_COL] > 190), CANON_ABP_COL] = np.nan
    if "CVP" in d:
        d.loc[(d["CVP"] < 0) | (d["CVP"] > 40), "CVP"] = np.nan
    if "HR" in d:
        d.loc[(d["HR"] < 40) | (d["HR"] > 180), "HR"] = np.nan
    if "RESP" in d:
        d.loc[(d["RESP"] < 8) | (d["RESP"] > 40), "RESP"] = np.nan
    if "SpO2" in d:
        d.loc[(d["SpO2"] < 0) | (d["SpO2"] > 100), "SpO2"] = np.nan
    if "CO" in d:
        d.loc[(d["CO"] < 0) | (d["CO"] > 20), "CO"] = np.nan
    if "PULSE" in d:
        d.loc[(d["PULSE"] < 30) | (d["PULSE"] > 240), "PULSE"] = np.nan

    if drop_all_nan_physio:
        keep_cols = [c for c in [CANON_ABP_COL, "CVP", "HR", "RESP", "SpO2", "CO", "PULSE"] if c in d.columns]
        if keep_cols:
            d = d.dropna(subset=keep_cols, how="all")

    return d

def _smooth_1d_nanaware(
    arr: np.ndarray, neighbors: int, keep_nan_center: bool = True, min_valid: int = 1
) -> np.ndarray:
    """Centered moving average that respects NaNs."""
    if neighbors <= 0 or arr.size == 0:
        return arr.astype(float, copy=True)
    win = 2 * neighbors + 1
    s = pd.Series(arr, dtype="float64")
    sm = s.rolling(win, center=True, min_periods=min_valid).mean()
    if keep_nan_center:
        sm[s.isna()] = np.nan
    return sm.to_numpy()


def _clip_inplace(g: pd.DataFrame):
    if CANON_ABP_COL in g:
        g[CANON_ABP_COL] = pd.to_numeric(g[CANON_ABP_COL], errors="coerce").clip(lower=40, upper=180)
    if "CVP" in g:
        g["CVP"] = pd.to_numeric(g["CVP"], errors="coerce").clip(lower=0, upper=40)
    return g


def _zero_center_cols(g: pd.DataFrame, cols, suffix="_zc"):
    for c in cols:
        if c in g:
            mu = g[c].mean(skipna=True)
            g[f"{c}{suffix}"] = g[c] - mu
    return g


def _zscore_cols(g: pd.DataFrame, cols, suffix="_zn"):
    for c in cols:
        if c in g:
            mu = g[c].mean(skipna=True)
            sd = g[c].std(skipna=True)
            g[f"{c}{suffix}"] = (g[c] - mu) / sd if (pd.notna(sd) and sd > 0) else np.nan
    return g


def _smooth_cols_multi(g: pd.DataFrame, cols, neighbors, source_suffixes, out_suffix):
    """
    For each c in cols and each source variant in source_suffixes
    ("" for raw, "_zc", "_zn"), create smoothed c{src}{out_suffix}.
    """
    for c in cols:
        for src in source_suffixes:
            base = f"{c}{src}" if src else c
            if base in g:
                g[f"{base}{out_suffix}"] = _smooth_1d_nanaware(
                    pd.to_numeric(g[base], errors="coerce").to_numpy(),
                    neighbors=neighbors,
                    keep_nan_center=True,
                    min_valid=1,
                )
    return g

def run_waveform_pipeline(
    df: pd.DataFrame,
    signals=(CANON_ABP_COL, "CVP", "HR", "RESP", "SpO2", "CO", "PULSE"),
    time_col="absolute_timestamp",
    group_cols=("hadm_id", "record_name"),
    *,
    do_zero_center: bool = True,
    do_zscore: bool = True,
    smooth_neighbors: int = 12,          # at 10s cadence, 12 neighbors ≈ 4-min window
    smooth_variants=("zc", "zn"),        # choose any subset of {"raw","zc","zn"}
    out_suffix: str = "_ma",
    flush_every_rows: int = 2_000_000,
):
    """
    Memory-friendly generator over (hadm_id, record_name):
      1) clip ABP_MEAN & CVP;
      2) add _zc and/or _zn per signal;
      3) smooth selected variants with centered moving average.

    Yields chunks; you can concat or stream-write.
    """
    need_time = time_col in df.columns
    out_frames, acc_rows = [], 0

    # Ensure numeric for signals up front
    d = df.copy()
    for c in signals:
        if c in d.columns:
            d[c] = pd.to_numeric(d[c], errors="coerce")

    sort_keys = list(group_cols) + ([time_col] if need_time else [])
    d = d.sort_values(sort_keys)

    var2suf = {"raw": "", "zc": "_zc", "zn": "_zn"}
    source_suffixes = [var2suf[v] for v in smooth_variants]

    for _, g in d.groupby(list(group_cols), sort=False, dropna=False):
        # Clip core pressure signals
        g = _clip_inplace(g)

        # Normalization variants (per group)
        if do_zero_center:
            g = _zero_center_cols(g, signals, suffix="_zc")
        if do_zscore:
            g = _zscore_cols(g, signals, suffix="_zn")

        # Smoothing
        if smooth_neighbors and smooth_neighbors > 0 and source_suffixes:
            g = _smooth_cols_multi(
                g, signals, neighbors=smooth_neighbors,
                source_suffixes=source_suffixes, out_suffix=out_suffix
            )

        out_frames.append(g)
        acc_rows += len(g)
        if acc_rows >= flush_every_rows:
            yield pd.concat(out_frames, ignore_index=True)
            out_frames.clear()
            acc_rows = 0

    if out_frames:
        yield pd.concat(out_frames, ignore_index=True)

def _hampel_mask(x: pd.Series, window: int = 6, n_sigma: float = 6.0) -> pd.Series:
    # window is in samples (e.g., 6 at 10s cadence ≈ 1 min)
    x = pd.to_numeric(x, errors="coerce")
    med = x.rolling(window*2+1, center=True, min_periods=3).median()
    diff = (x - med).abs()
    mad  = diff.rolling(window*2+1, center=True, min_periods=3).median()
    # consistent MAD scale to std: 1.4826
    thresh = n_sigma * 1.4826 * mad
    return (diff > thresh)

def _interp_small_gaps(s: pd.Series, max_gap_pts: int) -> pd.Series:
    if max_gap_pts <= 0:
        return s
    # mark long gaps to protect them from fill
    isnan = s.isna().to_numpy()
    if not isnan.any():
        return s
    # find contiguous NaN runs
    n = len(isnan)
    runs = []
    i = 0
    while i < n:
        if isnan[i]:
            j = i
            while j < n and isnan[j]:
                j += 1
            runs.append((i, j-1))
            i = j
        else:
            i += 1
    keep_nan = np.zeros(n, dtype=bool)
    for a, b in runs:
        if (b - a + 1) > max_gap_pts:
            keep_nan[a:b+1] = True

    out = s.copy()
    out[keep_nan] = np.nan  # protect long gaps
    out = out.interpolate(limit=max_gap_pts, limit_direction="both")
    return out

def despike_waveforms(
    df: pd.DataFrame,
    *,
    time_col: str = "absolute_timestamp",
    group_cols: Iterable[str] = ("hadm_id", "record_name"),
    # Per-signal configuration: step (per-sample), slope_per_min (using dt), and Hampel
    cfg: Dict[str, Dict[str, float]] = None,
    # What to do with detected artifacts
    action: str = "mask",             # {"mask","clip"}
    clip_back_to: str = "prev",       # when action=="clip": {"prev","median"}
    # Optional post-mask interpolation (only short gaps)
    interpolate_small_gaps_pts: int = 2,
) -> pd.DataFrame:
    """
    Remove/suppress measurement artifacts using step/slope caps and Hampel.
    Operates in-place on a copy; returns a new cleaned dataframe.
    """
    if cfg is None:
        # sensible defaults for 10s cadence
        cfg = {
            #    max per-sample step   max per-minute slope   hampel window (pts)   n_sigma
            "ABP_MEAN": {"step": 12.0, "slope_per_min": 40.0, "hwin": 6,  "nsig": 6.0},
            "HR":       {"step": 15.0, "slope_per_min": 50.0, "hwin": 6,  "nsig": 6.0},
            "CVP":      {"step":  5.0, "slope_per_min": 15.0, "hwin": 6,  "nsig": 6.0},
            "RESP":     {"step":  8.0, "slope_per_min": 25.0, "hwin": 6,  "nsig": 6.0},
        }

    out = df.copy()
    # ensure time is datetime
    if time_col in out.columns:
        out[time_col] = pd.to_datetime(out[time_col], errors="coerce", utc=True)

    # operate per encounter/file
    def _clean_group(g: pd.DataFrame) -> pd.DataFrame:
        if time_col in g.columns:
            # seconds between samples (handles small irregularities)
            dt = g[time_col].astype("int64").diff() / 1e9  # seconds
            # forward-fill the first dt with median of the group to avoid div-by-zero
            dt_med = np.nanmedian(dt.values) if np.isfinite(dt.values).any() else 10.0
            dt = dt.fillna(dt_med).replace(0.0, dt_med)
        else:
            # assume fixed 10s cadence
            dt = pd.Series(10.0, index=g.index, dtype="float64")

        for sig, params in cfg.items():
            if sig not in g.columns:
                continue
            s = pd.to_numeric(g[sig], errors="coerce")

            # 1) per-sample step cap
            step_thr = params.get("step", np.inf)
            step = s.diff().abs()
            bad_step = step > step_thr

            # 2) per-minute slope cap (|ds/dt| * 60)
            slope_thr = params.get("slope_per_min", np.inf)
            slope_per_min = (s.diff() / dt).abs() * 60.0
            bad_slope = slope_per_min > slope_thr

            # 3) Hampel spikes
            hwin = int(params.get("hwin", 6))
            nsig = float(params.get("nsig", 6.0))
            bad_hampel = _hampel_mask(s, window=hwin, n_sigma=nsig)

            # union of all artifact conditions
            bad = (bad_step | bad_slope | bad_hampel).reindex(g.index, fill_value=False)

            if action == "mask":
                s_clean = s.mask(bad)
                if interpolate_small_gaps_pts > 0:
                    s_clean = _interp_small_gaps(s_clean, max_gap_pts=interpolate_small_gaps_pts)
                g[sig] = s_clean

            elif action == "clip":
                if clip_back_to == "prev":
                    # clip spikes back to previous finite value
                    prev = s.shift(1)
                    s_clipped = s.where(~bad, prev)
                    g[sig] = s_clipped
                elif clip_back_to == "median":
                    med = s.rolling(hwin*2+1, center=True, min_periods=3).median()
                    s_clipped = s.where(~bad, med)
                    g[sig] = s_clipped
                else:
                    raise ValueError("clip_back_to must be 'prev' or 'median'")
            else:
                raise ValueError("action must be 'mask' or 'clip'")

        return g

    if group_cols:
        out = out.groupby(list(group_cols), group_keys=False, sort=False).apply(_clean_group)
    else:
        out = _clean_group(out)

    return out
