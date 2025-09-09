import os
from typing import Iterable, List, Tuple
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

CORE_TO_KEEP = ['ABP MEAN', 'ABP Mean','ABPMean', 'ART MEAN', 'ART Mean', 'CVP']
PLUS_TO_KEEP = ['HR', 'RESP', 'SpO2', 'CO', 'PULSE']
META_COLS    = ['hadm_id', 'record_name', 'record_start_time', 'record_end_time', 'time_seconds', 'absolute_timestamp']
CANON_ABP_COL = 'ABP_MEAN'   # canonical merged column we'll create

EXPECTED_COLS = [
    "hadm_id","record_name","absolute_timestamp","ABP_MEAN",
    "CVP","HR","RESP","SpO2","CO","PULSE",
    "record_start_time","record_end_time","time_seconds"
]

def save_aligned_waveform_filelist(
    aligned_cons: pd.DataFrame,
    out_csv_path: str,
    limit_per_segment: None = None,  # set None to keep all file_paths per segment
) -> pd.DataFrame:
    """
    Explode aligned consolidated segments into a per-file list and save to CSV.
    Returns the exploded dataframe.
    Expected aligned_cons columns: hadm_id, segment_id, file_paths (list), record_names (list, optional)
    """
    if aligned_cons is None or aligned_cons.empty:
        empty = pd.DataFrame(columns=["hadm_id", "segment_id", "file_path", "record_name"])
        empty.to_csv(out_csv_path, index=False)
        return empty

    df = aligned_cons.copy()
    if "file_paths" not in df.columns:
        raise KeyError("aligned_cons must include a 'file_paths' column (list of parquet paths per segment).")

    # explode file_paths
    exploded = df[["hadm_id", "segment_id", "file_paths", "record_names"]].copy()
    if "record_names" not in exploded:
        exploded["record_names"] = [[] for _ in range(len(exploded))]
    exploded = exploded.explode(["file_paths", "record_names"], ignore_index=True)
    exploded = exploded.rename(columns={"file_paths": "file_path", "record_names": "record_name"})

    # (optional) limit files per segment
    if limit_per_segment is not None:
        exploded["rnk"] = exploded.groupby(["hadm_id", "segment_id"]).cumcount()
        exploded = exploded.loc[exploded["rnk"] < limit_per_segment].drop(columns="rnk")

    # dedupe
    exploded = (
        exploded.dropna(subset=["file_path"])
                .drop_duplicates(subset=["hadm_id", "segment_id", "file_path"])
                .reset_index(drop=True)
    )

    os.makedirs(os.path.dirname(out_csv_path), exist_ok=True)
    exploded.to_csv(out_csv_path, index=False)
    return exploded

def _schema_lower_map(pf: pq.ParquetFile) -> dict:
    """Map lowercase->original column names from parquet schema."""
    m = {}
    for field in pf.schema_arrow:
        key = field.name.strip()
        lk = key.lower()
        if lk not in m:
            m[lk] = key
    return m

def _ensure_abs_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    if "absolute_timestamp" in df.columns and not df["absolute_timestamp"].isna().all():
        df["absolute_timestamp"] = pd.to_datetime(df["absolute_timestamp"], errors="coerce", utc=True)
        return df
    if {"record_start_time","time_seconds"}.issubset(df.columns):
        rstart = pd.to_datetime(df["record_start_time"], errors="coerce", utc=True)
        tsecs  = pd.to_numeric(df["time_seconds"], errors="coerce")
        df["absolute_timestamp"] = rstart + pd.to_timedelta(tsecs, unit="s")
    else:
        df["absolute_timestamp"] = pd.NaT
    return df

def _merge_abp_art_mean(df: pd.DataFrame, out_col: str = CANON_ABP_COL) -> pd.DataFrame:
    # Preferred order: ABP mean variants, then ART mean variants
    candidates = [c for c in ['ABP MEAN','ABP Mean','ABPMean','ART MEAN','ART Mean'] if c in df.columns]
    if not candidates:
        df[out_col] = np.nan
        return df
    # combine_first across candidates (row-wise). Start from first, fold in the rest.
    out = pd.to_numeric(df[candidates[0]], errors="coerce")
    for c in candidates[1:]:
        out = out.combine_first(pd.to_numeric(df[c], errors="coerce"))
    df[out_col] = out
    return df

def load_harmonized_waveforms_from_list(
    file_paths,
    core_to_keep=CORE_TO_KEEP,
    plus_to_keep=PLUS_TO_KEEP,
    verbose: bool = True,
) -> pd.DataFrame:
    keep_pretty = set(core_to_keep) | set(plus_to_keep) | set(META_COLS)

    frames = []
    for fp in file_paths:
        try:
            pf = pq.ParquetFile(fp)
        except Exception as e:
            if verbose: print(f"[skip] schema read error for {os.path.basename(fp)}: {e}")
            continue

        lowmap = _schema_lower_map(pf)
        cols_to_read = []
        for want in keep_pretty:
            lk = want.lower()
            if lk in lowmap:
                cols_to_read.append(lowmap[lk])

        try:
            table = pf.read(columns=cols_to_read)
        except Exception as e:
            if verbose: print(f"[skip] read error for {os.path.basename(fp)}: {e}")
            continue

        df = table.to_pandas()

        # Ensure timestamps
        for tcol in ("record_start_time","record_end_time"):
            if tcol in df.columns:
                df[tcol] = pd.to_datetime(df[tcol], errors="coerce", utc=True)
        if "time_seconds" in df.columns:
            df["time_seconds"] = pd.to_numeric(df["time_seconds"], errors="coerce")

        df = _ensure_abs_timestamp(df)
        df = _merge_abp_art_mean(df, out_col=CANON_ABP_COL)

        # enforce expected schema
        for c in EXPECTED_COLS:
            if c not in df.columns:
                df[c] = pd.NA
        df = df[EXPECTED_COLS]

        frames.append(df)

        if verbose:
            print(f"[ok ] {os.path.basename(fp)} -> rows {len(df)}")

    if not frames:
        return pd.DataFrame(columns=EXPECTED_COLS)

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values(["hadm_id","record_name","absolute_timestamp"], kind="mergesort").reset_index(drop=True)
    return combined
