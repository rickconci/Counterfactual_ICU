import os
import pandas as pd
import numpy as np
import pyarrow.parquet as pq
from typing import List, Iterable, Optional, Tuple
from collections import Counter

def analyze_waveform_columns(waveform_dir: Optional[str] = None) -> list[tuple[str, int]]:
    """Analyze columns across all waveform files to understand data structure.
    
    Args:
        waveform_dir: Directory containing waveform files. If None, uses default path.
        
    Returns:
        List of tuples (column_name, file_count) sorted by frequency.
        
    Raises:
        ValueError: If waveform directory is not provided and not set in environment.
    """
    if waveform_dir is None:
        waveform_dir = "/Users/riccardoconci/Local_documents/Counterfactual_ICU/data/mimic_3_data/processed_data/full_trajectories"
    
    if not os.path.exists(waveform_dir):
        raise ValueError(f"Waveform directory does not exist: {waveform_dir}")
    
    numerics_files = os.listdir(waveform_dir)
    print(f"Found {len(numerics_files)} files")
    
    col_counter = Counter()
    
    for fname in numerics_files:
        fpath = os.path.join(waveform_dir, fname)
        try:
            schema = pq.read_schema(fpath)
            col_counter.update(schema.names)
        except Exception as e:
            print(f"Could not read {fname}: {e}")
    
    # Rank columns by how many files they appear in
    sorted_cols = col_counter.most_common()
    return sorted_cols

def list_files_with_core_signals(
    waveform_dir: str,
    core_to_keep: Iterable[str],
    case_insensitive: bool = True,
    limit_print: int = 5,
) -> List[str]:
    """
    Return file paths of .parquet waveforms that contain at least one of the
    'core_to_keep' column names (numerics).
    
    Args:
        waveform_dir: Directory containing waveform parquet files
        core_to_keep: Iterable of core signal names to look for
        case_insensitive: Whether to perform case-insensitive matching
        limit_print: Maximum number of files to print in preview
        
    Returns:
        List of file paths containing at least one core signal
    """
    core = list(core_to_keep)
    if case_insensitive:
        core_lower = {c.lower() for c in core}

    files = [f for f in os.listdir(waveform_dir) if f.endswith(".parquet")]
    files_with_core: List[str] = []

    print(f"Found {len(files)} parquet files in {waveform_dir}")

    for fname in files:
        fpath = os.path.join(waveform_dir, fname)
        try:
            schema = pq.read_schema(fpath)
            names = schema.names
            if case_insensitive:
                cols = {n.lower() for n in names}
                hit = any(c in cols for c in core_lower)
            else:
                cols = set(names)
                hit = any(c in cols for c in core)
            if hit:
                files_with_core.append(fpath)
        except Exception as e:
            print(f"Could not read {fname}: {e}")

    print(
        f"\nFiles containing at least one core signal "
        f"({len(files_with_core)} / {len(files)}):"
    )
    for f in files_with_core[:limit_print]:
        print(os.path.basename(f))
    if len(files_with_core) > limit_print:
        print("...")

    return files_with_core

def build_waveform_metadata_from_files(file_paths: List[str]) -> pd.DataFrame:
    """
    Build metadata from selected waveform files.
    
    Args:
        file_paths: List of parquet file paths to process
        
    Returns:
        DataFrame with waveform metadata including hadm_id, record info, and file paths
    """
    rows = []
    for fpath in file_paths:
        try:
            tbl = pq.read_table(
                fpath,
                columns=["hadm_id", "record_name", "record_start_time", "record_end_time"],
            )
            df = tbl.to_pandas()
            df = df[["hadm_id", "record_name", "record_start_time", "record_end_time"]].drop_duplicates()
            df["file_path"] = fpath
            rows.append(df)
        except Exception as e:
            print(f"Skipping {os.path.basename(fpath)}: {e}")
    if not rows:
        return pd.DataFrame(columns=["hadm_id","record_name","record_start_time","record_end_time","file_path"])

    meta = pd.concat(rows, ignore_index=True)
    meta["record_start_time"] = pd.to_datetime(meta["record_start_time"], errors="coerce", utc=True)
    meta["record_end_time"]   = pd.to_datetime(meta["record_end_time"],   errors="coerce", utc=True)
    return meta

def dedup_file_bounds(waveform_meta: pd.DataFrame) -> pd.DataFrame:
    """
    Deduplicate per-file bounds if a file lists multiple rows.
    
    Args:
        waveform_meta: DataFrame with waveform metadata
        
    Returns:
        DataFrame with deduplicated file bounds
    """
    if waveform_meta.empty:
        return waveform_meta.copy()
    cols = ["hadm_id", "file_path"]
    agg = (waveform_meta
            .groupby(cols, as_index=False)
            .agg(record_name=("record_name","first"),
                    record_start_time=("record_start_time","min"),
                    record_end_time=("record_end_time","max")))
    return agg

def consolidate_waveforms(waveform_meta: pd.DataFrame, gap_hours: float = 2.0) -> pd.DataFrame:
    """
    Merge adjacent/overlapping files into a single continuous segment per hadm_id
    whenever the gap between consecutive files is <= gap_hours.
    
    Args:
        waveform_meta: DataFrame with waveform metadata
        gap_hours: Maximum gap in hours to consider files as contiguous
        
    Returns:
        DataFrame with consolidated segments containing:
        hadm_id, segment_id, seg_start_time, seg_end_time, component_count,
        record_names (list), file_paths (list), seg_duration_seconds
    """
    wm = dedup_file_bounds(waveform_meta)
    if wm.empty:
        return pd.DataFrame(columns=[
            "hadm_id","segment_id","seg_start_time","seg_end_time",
            "component_count","record_names","file_paths","seg_duration_seconds"
        ])

    wm = wm.sort_values(["hadm_id","record_start_time","record_end_time"]).reset_index(drop=True)
    gap = pd.to_timedelta(gap_hours, unit="h")

    seg_rows = []
    for hadm_id, g in wm.groupby("hadm_id", sort=False):
        current_start = None
        current_end   = None
        names, paths  = [], []
        seg_id = 0

        def flush():
            nonlocal seg_id, seg_rows, current_start, current_end, names, paths
            if current_start is None:
                return
            seg_id += 1
            seg_rows.append({
                "hadm_id": hadm_id,
                "segment_id": seg_id,
                "seg_start_time": current_start,
                "seg_end_time": current_end,
                "component_count": len(paths),
                "record_names": names.copy(),
                "file_paths": paths.copy(),
                "seg_duration_seconds": (current_end - current_start).total_seconds() if pd.notna(current_end) and pd.notna(current_start) else np.nan,
            })
            current_start = None
            current_end   = None
            names, paths  = [], []

        for _, row in g.iterrows():
            s = row["record_start_time"]
            e = row["record_end_time"]
            if current_start is None:
                # start new segment
                current_start, current_end = s, e
                names = [row["record_name"]]
                paths = [row["file_path"]]
            else:
                # if overlapping or gap <= threshold, extend; else flush and start new
                if s <= current_end + gap:
                    current_end = max(current_end, e)
                    names.append(row["record_name"])
                    paths.append(row["file_path"])
                else:
                    flush()
                    current_start, current_end = s, e
                    names = [row["record_name"]]
                    paths = [row["file_path"]]
        # flush last
        flush()

    consolidated = pd.DataFrame(seg_rows)
    consolidated = consolidated.sort_values(["hadm_id","seg_start_time"]).reset_index(drop=True)
    return consolidated
