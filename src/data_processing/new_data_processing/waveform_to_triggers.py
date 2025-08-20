
import argparse
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any

import pandas as pd
from tqdm import tqdm


FNAME_HADM_RE = re.compile(r"full_waveform_(\d+)(?:_session_(\d+))?\.parquet$", re.IGNORECASE)


@dataclass
class Bounds:
    start: pd.Timestamp
    end: pd.Timestamp

    def expand(self, before: pd.Timedelta, after: pd.Timedelta) -> "Bounds":
        return Bounds(self.start - before, self.end + after)

    def intersect(self, other: "Bounds") -> Optional["Bounds"]:
        s = max(self.start, other.start)
        e = min(self.end, other.end)
        if pd.isna(s) or pd.isna(e) or s > e:
            return None
        return Bounds(s, e)

    def duration_seconds(self) -> float:
        return float((self.end - self.start).total_seconds())


def to_utc(s: pd.Series) -> pd.Series:
    """
    Coerce any datetime-like to timezone-aware UTC.
    - tz-naive -> localize as UTC
    - tz-aware -> convert to UTC
    """
    dt = pd.to_datetime(s, errors="coerce", utc=True)
    return dt


def read_triggers(path: Path) -> pd.DataFrame:
    ext = path.suffix.lower()
    if ext in [".parquet", ".pq"]:
        df = pd.read_parquet(path)
    elif ext in [".pkl", ".pickle"]:
        df = pd.read_pickle(path)
    else:
        df = pd.read_csv(path)
    if "hadm_id" not in df.columns:
        raise ValueError("Triggers file must include 'hadm_id' column.")
    for c in ("start_time", "end_time"):
        if c not in df.columns:
            raise ValueError(f"Triggers file must include '{c}' column.")
        df[c] = to_utc(df[c])
    df["hadm_id"] = df["hadm_id"].astype(str)
    return df


def waveform_bounds(path: Path) -> Optional[Bounds]:
    try:
        df = pd.read_parquet(path, columns=["record_start_time", "time_seconds"])
    except Exception:
        df = pd.read_parquet(path)
        if "record_start_time" not in df or "time_seconds" not in df:
            return None
    if df.empty:
        return None
    rst = to_utc(df["record_start_time"])
    tsecs = pd.to_numeric(df["time_seconds"], errors="coerce")
    abs_ts = rst + pd.to_timedelta(tsecs, unit="s")
    return Bounds(abs_ts.min(), abs_ts.max())


def crop_waveform(path: Path, out_path: Path, window: Bounds) -> int:
    df = pd.read_parquet(path)
    df["record_start_time"] = to_utc(df["record_start_time"])
    df["time_seconds"] = pd.to_numeric(df["time_seconds"], errors="coerce")
    df["absolute_timestamp"] = df["record_start_time"] + pd.to_timedelta(df["time_seconds"], unit="s")
    mask = (df["absolute_timestamp"] >= window.start) & (df["absolute_timestamp"] <= window.end)
    out = df.loc[mask].copy()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)
    return len(out)


def extract_ids_from_name(path: Path) -> Tuple[Optional[str], Optional[int]]:
    m = FNAME_HADM_RE.search(path.name)
    if not m:
        return None, None
    hadm = m.group(1)
    sess = m.group(2)
    return hadm, (int(sess) if sess is not None else None)


def process_one_file(
    wf_path: Path,
    hadm_triggers: pd.DataFrame,
    margin: pd.Timedelta,
    out_dir: Path,
) -> Tuple[str, Optional[pd.DataFrame], Dict[str, Any]]:
    hadm_id, session_id = extract_ids_from_name(wf_path)
    manifest = {
        "file": wf_path.name,
        "hadm_id": hadm_id,
        "session_id": session_id,
        "status": "skipped",
        "wf_rows_out": 0,
        "triggers_before": 0,
        "triggers_after": 0,
        "wf_start": pd.NaT,
        "wf_end": pd.NaT,
        "desired_start": pd.NaT,
        "desired_end": pd.NaT,
        "final_start": pd.NaT,
        "final_end": pd.NaT,
        "reason": "",
    }

    if hadm_id is None:
        manifest["reason"] = "no hadm_id in filename"
        return (f"{wf_path.name}: skip (no hadm_id)", None, manifest)

    wf_b = waveform_bounds(wf_path)
    if wf_b is None:
        manifest["reason"] = "cannot read waveform bounds"
        return (f"{wf_path.name}: skip (cannot read bounds)", None, manifest)

    manifest["wf_start"] = wf_b.start
    manifest["wf_end"] = wf_b.end

    g = hadm_triggers
    manifest["triggers_before"] = len(g)
    if g.empty:
        manifest["reason"] = "no triggers for hadm_id"
        return (f"{wf_path.name}: no triggers for HADM_ID {hadm_id}", None, manifest)

    # Ensure triggers are UTC (safety if caller didn't normalize)
    g = g.copy()
    g["start_time"] = to_utc(g["start_time"])
    g["end_time"] = to_utc(g["end_time"])

    # Overlap triggers with waveform bounds
    g = g[(g["end_time"] >= wf_b.start) & (g["start_time"] <= wf_b.end)].copy()
    if g.empty:
        manifest["reason"] = "no overlapping triggers"
        return (f"{wf_path.name}: no overlapping triggers", None, manifest)

    # Desired window = trigger span ± margin
    t0 = g["start_time"].min()
    t1 = g["end_time"].max()
    desired = Bounds(t0, t1).expand(margin, margin)
    manifest["desired_start"] = desired.start
    manifest["desired_end"] = desired.end

    # Final window = intersection(desired, waveform)
    final = desired.intersect(wf_b)
    if final is None or final.duration_seconds() <= 0:
        manifest["reason"] = "no intersection after margin"
        return (f"{wf_path.name}: no intersection after margin", None, manifest)

    manifest["final_start"] = final.start
    manifest["final_end"] = final.end

    # Crop waveform
    out_path = out_dir / wf_path.name
    n = crop_waveform(wf_path, out_path, final)
    manifest["wf_rows_out"] = n
    manifest["status"] = "written"

    # Crop triggers to final window (any overlap with window)
    trig_cropped = g[(g["start_time"] <= final.end) & (g["end_time"] >= final.start)].copy()
    trig_cropped["aligned_window_start"] = final.start
    trig_cropped["aligned_window_end"] = final.end
    manifest["triggers_after"] = len(trig_cropped)

    return (f"{wf_path.name}: wrote {n} rows; triggers {len(g)} -> {len(trig_cropped)}", trig_cropped, manifest)


def main():
    ap = argparse.ArgumentParser(description="Align/crop waveforms to medication trigger windows, per session. All times coerced to UTC.")
    ap.add_argument("--triggers", required=True, type=Path)
    ap.add_argument("--waveforms", required=True, type=Path)
    ap.add_argument("--out", required=False, type=Path, default=None)
    ap.add_argument("--margin-minutes", type=int, default=120)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--triggers-pickle", type=Path, default=None)
    ap.add_argument("--manifest-csv", type=Path, default=None)
    args = ap.parse_args()

    trig_df = read_triggers(args.triggers)

    wf_dir = args.waveforms
    if not wf_dir.exists():
        raise SystemExit(f"Waveforms dir not found: {wf_dir}")

    out_dir = args.out or (wf_dir / "aligned_full_trajectories_mv")
    out_dir.mkdir(parents=True, exist_ok=True)

    triggers_pickle_path = args.triggers_pickle or (out_dir / "aligned_triggers.pkl")
    manifest_csv_path = args.manifest_csv or (out_dir / "alignment_manifest.csv")

    # hadm -> triggers slice (UTC-normalized)
    by_hadm: Dict[str, pd.DataFrame] = {hid: g.copy() for hid, g in trig_df.groupby(trig_df["hadm_id"])}

    wf_files = sorted(wf_dir.glob("full_waveform_*.parquet"))
    if not wf_files:
        raise SystemExit(f"No waveform parquet files found in: {wf_dir}")

    margin = pd.Timedelta(minutes=args.margin_minutes)

    results_trig: List[pd.DataFrame] = []
    manifest_rows: List[Dict[str, Any]] = []

    with ProcessPoolExecutor(max_workers=args.workers) as ex, tqdm(total=len(wf_files), desc="Aligning", unit="file") as pbar:
        futs = []
        for wf in wf_files:
            hid, _ = extract_ids_from_name(wf)
            hadm_triggers = by_hadm.get(hid, pd.DataFrame(columns=trig_df.columns))
            futs.append(ex.submit(process_one_file, wf, hadm_triggers, margin, out_dir))

        for fut in as_completed(futs):
            msg, part, man = fut.result()
            tqdm.write(msg)
            if man:
                manifest_rows.append(man)
            if part is not None and not part.empty:
                results_trig.append(part)
            pbar.update(1)

    if results_trig:
        all_trig = pd.concat(results_trig, ignore_index=True)
        all_trig.to_pickle(triggers_pickle_path)
        print(f"Wrote cropped triggers pickle: {triggers_pickle_path} ({len(all_trig)} rows)")
    else:
        print("No cropped triggers to write.")

    manifest_df = pd.DataFrame(manifest_rows)
    manifest_df.sort_values(["hadm_id", "session_id", "file"], inplace=True, na_position="last")
    manifest_df.to_csv(manifest_csv_path, index=False)
    print(f"Wrote manifest CSV: {manifest_csv_path} ({len(manifest_df)} rows)")

    total_written = int(manifest_df.loc[manifest_df["status"] == "written", "wf_rows_out"].sum())
    files_written = int((manifest_df["status"] == "written").sum())
    print("--- Sanity summary ---")
    print(f"Files processed: {len(manifest_df)}")
    print(f"Files written:   {files_written}")
    print(f"Total rows out:  {total_written}")
    if results_trig:
        print(f"Cropped triggers rows total: {len(all_trig)}")
    print("----------------------")


if __name__ == "__main__":
    main()
