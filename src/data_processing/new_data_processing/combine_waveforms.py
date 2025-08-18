
import argparse
import json
import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Tuple, Optional, Any

import numpy as np
import pandas as pd
from tqdm import tqdm


FNAME_RE = re.compile(r"^full_waveform_(\d+)_")

def _extract_hadm_id_from_name(name: str) -> Optional[str]:
    m = FNAME_RE.match(name)
    return m.group(1) if m else None


def discover_hadm_ids(input_dir: Path) -> List[str]:
    ids = set()
    for p in input_dir.glob("full_waveform_*_*.parquet"):
        hid = _extract_hadm_id_from_name(p.name)
        if hid is not None:
            ids.add(hid)
    return sorted(ids, key=lambda x: int(x))


def _sanitize_value(v: Any) -> Any:
    # Convert numpy types to Python scalars
    if isinstance(v, (np.generic, )):
        return v.item()
    # Convert numpy arrays to Python lists
    if isinstance(v, np.ndarray):
        try:
            return v.tolist()
        except Exception:
            return None
    return v


def sanitize_for_parquet(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    obj_cols = [c for c in df.columns if df[c].dtype == 'object']
    for c in obj_cols:
        df[c] = df[c].map(_sanitize_value)
    return df


def read_concat_compute_timestamps(files: List[Path]) -> pd.DataFrame:
    dfs = []
    for f in files:
        df = pd.read_parquet(f)
        dfs.append(df)
    if not dfs:
        return pd.DataFrame()
    full_df = pd.concat(dfs, ignore_index=True)

    full_df["record_start_time"] = pd.to_datetime(full_df["record_start_time"], errors="coerce")
    full_df["time_seconds"] = pd.to_numeric(full_df["time_seconds"], errors="coerce")
    full_df["absolute_timestamp"] = full_df["record_start_time"] + pd.to_timedelta(full_df["time_seconds"], unit="s")

    # Drop rows where timestamp couldn't be computed
    full_df = full_df.dropna(subset=["absolute_timestamp"])

    # Stable sort
    full_df = full_df.sort_values("absolute_timestamp", kind="mergesort").reset_index(drop=True)
    return full_df


def group_into_sessions(df: pd.DataFrame, gap_threshold_hours: int) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.sort_values("absolute_timestamp", kind="mergesort").reset_index(drop=True)
    time_diff = df["absolute_timestamp"].diff()
    session_starts = time_diff > pd.Timedelta(hours=gap_threshold_hours)
    df = df.copy()
    df["session_id"] = session_starts.cumsum()
    return df


def _atomic_write_parquet(df: pd.DataFrame, out_path: Path) -> None:
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    if tmp.exists():
        tmp.unlink()
    df.to_parquet(tmp, index=False)
    # On POSIX, rename is atomic
    os.replace(tmp, out_path)


def write_sessions(
    df: pd.DataFrame,
    hadm_id: str,
    out_dir: Path,
) -> List[Tuple[str, int]]:
    """
    Writes either a single file (if one session) or multiple files with session suffixes.
    Returns a list of (output_file_name, num_rows).
    """
    rows_written: List[Tuple[str, int]] = []
    out_dir.mkdir(parents=True, exist_ok=True)

    if df.empty:
        return rows_written

    df = sanitize_for_parquet(df)

    num_sessions = int(df["session_id"].nunique())
    if num_sessions <= 1:
        out_path = out_dir / f"full_waveform_{hadm_id}.parquet"
        _atomic_write_parquet(df, out_path)
        rows_written.append((out_path.name, len(df)))
    else:
        for sess_id, g in df.groupby("session_id", sort=True):
            sess_idx = int(sess_id) + 1  # 1-indexed in filename
            out_path = out_dir / f"full_waveform_{hadm_id}_session_{sess_idx}.parquet"
            g = sanitize_for_parquet(g)
            _atomic_write_parquet(g, out_path)
            rows_written.append((out_path.name, len(g)))
    return rows_written


def sentinel_path(out_dir: Path, hadm_id: str) -> Path:
    return out_dir / f".done_{hadm_id}"


def mark_done(out_dir: Path, hadm_id: str, meta: dict) -> None:
    sp = sentinel_path(out_dir, hadm_id)
    meta = dict(meta)
    meta["hadm_id"] = hadm_id
    meta["completed_at"] = datetime.now(timezone.utc).isoformat()
    tmp = sp.with_suffix(sp.suffix + ".tmp")
    with open(tmp, "w") as fh:
        json.dump(meta, fh)
    os.replace(tmp, sp)


def is_done(out_dir: Path, hadm_id: str) -> bool:
    return sentinel_path(out_dir, hadm_id).exists()


def process_one_hadm(
    hadm_id: str,
    input_dir: Path,
    out_dir: Path,
    gap_threshold_hours: int,
    resume: bool = True,
    skip_if_outputs_exist: bool = False,
    force: bool = False,
) -> str:
    try:
        if resume and not force and is_done(out_dir, hadm_id):
            return f"HADM_ID {hadm_id}: resume-skip (.done exists)."

        # Optional skip if any outputs already present (heuristic)
        if skip_if_outputs_exist and not force:
            if (out_dir / f"full_waveform_{hadm_id}.parquet").exists():
                return f"HADM_ID {hadm_id}: skip (combined already exists)."
            for _ in out_dir.glob(f"full_waveform_{hadm_id}_session_*.parquet"):
                return f"HADM_ID {hadm_id}: skip (some session files exist)."

        files = sorted(input_dir.glob(f"full_waveform_{hadm_id}_*.parquet"))
        if not files:
            return f"HADM_ID {hadm_id}: no files found."

        df = read_concat_compute_timestamps(files)
        if df.empty:
            return f"HADM_ID {hadm_id}: empty after read."

        df = group_into_sessions(df, gap_threshold_hours)
        written = write_sessions(df, hadm_id, out_dir)
        if not written:
            return f"HADM_ID {hadm_id}: nothing written."

        total_rows = int(sum(n for _, n in written))
        mark_done(out_dir, hadm_id, meta={
            "sessions": len(written),
            "rows": total_rows,
            "gap_threshold_hours": int(gap_threshold_hours),
        })

        if len(written) == 1:
            name, n = written[0]
            return f"HADM_ID {hadm_id}: wrote {name} with {n} rows."
        else:
            parts = ", ".join([f"{nm} ({n})" for nm, n in written])
            return f"HADM_ID {hadm_id}: wrote {len(written)} sessions -> {parts}"
    except Exception as e:
        return f"HADM_ID {hadm_id}: ERROR {type(e).__name__}: {e}"


def main():
    parser = argparse.ArgumentParser(description="Combine per-HADM waveform shards into chronologically sorted sessions.")
    parser.add_argument("--input", required=True, type=Path, help="Path to full_trajectories directory with parquet shards.")
    parser.add_argument("--output", required=False, type=Path, default=None, help="Output directory (default: sibling 'full_trajectories_combined').")
    parser.add_argument("--gap-hours", type=int, default=24, help="Gap threshold in hours to start a new session (default: 24).")
    parser.add_argument("--workers", type=int, default=os.cpu_count() or 4, help="Parallel workers (default: CPU count).")
    parser.add_argument("--force", action="store_true", help="Reprocess and overwrite outputs even if they already exist and ignore .done sentinels.")
    parser.add_argument("--no-resume", action="store_true", help="Ignore .done sentinels; process everything (unless --force is also set).")
    parser.add_argument("--skip-existing-outputs", action="store_true", help="Skip HADM_IDs if any output files already exist (heuristic).")
    args = parser.parse_args()

    in_dir: Path = args.input
    if not in_dir.exists():
        raise SystemExit(f"Input directory does not exist: {in_dir}")

    out_dir: Path = args.output or (in_dir.parent / "full_trajectories_combined")
    out_dir.mkdir(parents=True, exist_ok=True)

    hadm_ids = discover_hadm_ids(in_dir)
    if not hadm_ids:
        raise SystemExit("No HADM_IDs discovered. Check your input directory and filename patterns.")

    print(f"Discovered {len(hadm_ids)} HADM_ID(s). Processing with {args.workers} worker(s).")

    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex, tqdm(total=len(hadm_ids), desc="Combining", unit="hadm") as pbar:
        futures = {
            ex.submit(
                process_one_hadm,
                hid,
                in_dir,
                out_dir,
                args.gap_hours,
                resume=(not args.no_resume),
                skip_if_outputs_exist=args.skip_existing_outputs,
                force=args.force,
            ): hid
            for hid in hadm_ids
        }
        for fut in as_completed(futures):
            msg = fut.result()
            tqdm.write(msg)
            results.append(msg)
            pbar.update(1)

    # Append a manifest (jsonl) for the run
    manifest = out_dir / "manifest.jsonl"
    with open(manifest, "a") as fh:
        for line in results:
            rec = {"message": line, "ts": datetime.now(timezone.utc).isoformat()}
            fh.write(json.dumps(rec) + "\n")
    print(f"Done. Manifest appended at: {manifest}")


if __name__ == "__main__":
    main()
