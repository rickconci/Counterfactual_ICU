import argparse
from pathlib import Path
import pandas as pd
import pyarrow.parquet as pq
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import re
from typing import Dict, Tuple, Optional

# Regex to extract hadm_id from filenames like 'full_waveform_12345.parquet' or '..._session_1.parquet'
HADM_RE = re.compile(r"full_waveform_(\d+)(?:_session_\d+)?\.parquet$", re.IGNORECASE)

def to_utc(s: pd.Series) -> pd.Series:
    return pd.to_datetime(s, errors="coerce", utc=True)

def get_waveform_bounds(path: Path) -> Optional[Tuple[str, pd.Timestamp, pd.Timestamp]]:
    """Reads a parquet file's time columns to find its start and end times."""
    match = HADM_RE.search(path.name)
    if not match:
        return None
    hadm_id = match.group(1)
    
    try:
        df = pd.read_parquet(path, columns=["record_start_time", "time_seconds"])
        if df.empty:
            return None
        
        rst = to_utc(df["record_start_time"])
        tsecs = pd.to_numeric(df["time_seconds"], errors="coerce")
        ts = rst + pd.to_timedelta(tsecs, unit="s")
        ts = ts.dropna()
        
        if len(ts) == 0:
            return None
        
        return hadm_id, ts.min(), ts.max()
    except Exception:
        return None

def main():
    ap = argparse.ArgumentParser(
        description="Prefilters a triggers file to keep only events that fall within the bounds of available waveform data."
    )
    ap.add_argument("--waveforms", required=True, type=Path, help="Directory with full_waveform parquet files.")
    ap.add_argument("--triggers", required=True, type=Path, help="Path to the input triggers file (e.g., input_mv_triggers.parquet).")
    ap.add_argument("--out", required=True, type=Path, help="Path to save the new, filtered triggers parquet file.")
    ap.add_argument("--workers", type=int, default=4, help="Number of parallel workers.")
    args = ap.parse_args()

    # --- Step 1: Scan all waveform files to get their time bounds ---
    print("Scanning waveform files to determine time bounds...")
    all_files = list(args.waveforms.glob("*.parquet"))
    bounds_map: Dict[str, Tuple[pd.Timestamp, pd.Timestamp]] = {}

    with ProcessPoolExecutor(max_workers=args.workers) as executor, tqdm(total=len(all_files), desc="Scanning Waveforms") as pbar:
        futs = [executor.submit(get_waveform_bounds, f) for f in all_files]
        for fut in as_completed(futs):
            result = fut.result()
            if result:
                hadm_id, start, end = result
                if hadm_id not in bounds_map:
                    bounds_map[hadm_id] = (start, end)
                else:
                    # Extend the bounds if there are multiple session files
                    old_start, old_end = bounds_map[hadm_id]
                    bounds_map[hadm_id] = (min(old_start, start), max(old_end, end))
            pbar.update(1)
    
    print(f"Found time bounds for {len(bounds_map)} unique hospital admissions.")

    # --- Step 2: Load triggers and filter based on the bounds map ---
    print(f"Loading triggers from {args.triggers}...")
    triggers = pd.read_parquet(args.triggers)
    original_count = len(triggers)
    print(f"Loaded {original_count} total triggers.")

    # Ensure t0 (cluster start time) is in UTC for correct comparison
    triggers['t0'] = triggers.groupby('action_cluster_id')['start_time'].transform('min')
    triggers['t0'] = to_utc(triggers['t0'])
    
    # Define the filtering function
    def is_within_bounds(row: pd.Series) -> bool:
        hadm_id = str(row["hadm_id"])
        t0 = row["t0"]
        if pd.isna(t0) or hadm_id not in bounds_map:
            return False
        wf_start, wf_end = bounds_map[hadm_id]
        return wf_start <= t0 < wf_end

    # Apply the filter
    print("Filtering triggers...")
    mask = triggers.apply(is_within_bounds, axis=1)
    filtered_triggers = triggers[mask].copy()
    
    # We don't need the temporary 't0' column in the final output
    filtered_triggers = filtered_triggers.drop(columns=['t0'])

    # --- Step 3: Save the new file and report summary ---
    print(f"Saving {len(filtered_triggers)} valid triggers to {args.out}...")
    filtered_triggers.to_parquet(args.out, index=False)

    print("\n--- Prefiltering Complete ---")
    print(f"Original triggers: {original_count}")
    print(f"Valid triggers saved: {len(filtered_triggers)}")
    print(f"Discarded: {original_count - len(filtered_triggers)}")
    print("-----------------------------")

if __name__ == "__main__":
    main()
