import argparse
import math
from pathlib import Path
from typing import Iterable, List, Tuple, Optional, Any

import numpy as np
import pandas as pd


def _ensure_cols_from_index(df: pd.DataFrame, names=("subject_id", "hadm_id")) -> pd.DataFrame:
    """If keys are index levels, bring them out as columns only if not already columns; if already columns, drop the index level."""
    df2 = df
    def _has_level(idx, name):
        if isinstance(idx, pd.MultiIndex):
            return name in idx.names
        else:
            return idx.name == name

    for n in names:
        if not n:
            continue
        in_cols = n in df2.columns
        in_index = _has_level(df2.index, n)

        if in_index and in_cols:
            # Drop the index level; keep existing column
            if isinstance(df2.index, pd.MultiIndex):
                df2 = df2.reset_index(level=n, drop=True)
            else:
                df2 = df2.reset_index(drop=True)
        elif in_index and not in_cols:
            # Materialize as column
            if isinstance(df2.index, pd.MultiIndex):
                df2 = df2.reset_index(level=n)
            else:
                df2 = df2.reset_index()  # single index named n
        elif (not in_index) and (not in_cols):
            df2[n] = pd.NA
    return df2


def round_sig(x: Any, sig: int = 2) -> Optional[float]:
    """Round to N significant figures. Returns None for NaN/None/non-finite."""
    try:
        if x is None:
            return None
        xv = float(x)
        if not math.isfinite(xv):
            return None
        if xv == 0.0:
            return 0.0
        # significant figures rounding
        return float(f"{xv:.{sig}g}")
    except Exception:
        return None


def canonical_cluster_signature(labels: Iterable[str], doses: Iterable[Any]) -> Tuple[Tuple[str, Optional[float]], ...]:
    """Create a canonical, sortable signature as a tuple of (label, rounded_dose).
    - labels: medication names (item_label)
    - doses: numeric values; we take rate/weight and round to 2 sig figs
    Output is sorted by label then dose to ensure canonical equality across orderings.
    """
    pairs: List[Tuple[str, Optional[float]]] = []
    for lab, d in zip(labels, doses):
        lab_str = "" if lab is None else str(lab).strip()
        d_rounded = round_sig(d, sig=2)
        pairs.append((lab_str, d_rounded))
    # Remove empty labels if any slipped in
    pairs = [(l, d) for (l, d) in pairs if l != ""]
    # Sort canonical: by label, then dose (None sorts last)
    pairs.sort(key=lambda x: (x[0].lower(), (float('inf') if x[1] is None else x[1])))
    return tuple(pairs)


def signature_to_string(sig: Tuple[Tuple[str, Optional[float]], ...]) -> str:
    """Render signature as 'Drug=dose; Drug2=dose2'. Use 'NA' for missing dose."""
    parts = []
    for lab, d in sig:
        if d is None:
            parts.append(f"{lab}=NA")
        else:
            # keep as plain number; caller can add units if desired
            parts.append(f"{lab}={d}")
    return "; ".join(parts)


def build_cluster_vocab(
    df: pd.DataFrame,
    dose_col: str = "rate/weight",
    label_col: str = "item_label",
    time_col: str = "start_time",
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Return (clusters_with_signatures_df, signature_vocab_df)"""
    # hygiene
    df = _ensure_cols_from_index(df, names=("subject_id", "hadm_id")).copy()
    for c in (time_col,):
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    # We only consider rows that belong to a cluster id (not NaN)
    if "action_cluster_id" not in df.columns:
        raise ValueError("Expected 'action_cluster_id' column in df.")
    work = df[df["action_cluster_id"].notna()].copy()
    if work.empty:
        # Return empty frames with expected columns
        empty_clusters = pd.DataFrame(columns=[
            "subject_id","hadm_id","action_cluster_id","cluster_size","cluster_start","cluster_end",
            "signature_tuple","signature_str"
        ])
        empty_vocab = pd.DataFrame(columns=["signature_str","signature_tuple","frequency","mean_cluster_size"])
        return empty_clusters, empty_vocab

    # Build per-cluster signature
    group_keys = ["subject_id","hadm_id","action_cluster_id"]
    def build_one(g: pd.DataFrame) -> pd.Series:
        sig = canonical_cluster_signature(g[label_col], g[dose_col])
        sig_str = signature_to_string(sig)
        return pd.Series({
            "cluster_size": int(len(g)),
            "cluster_start": g[time_col].min(),
            "cluster_end": g[time_col].max(),
            "signature_tuple": sig,
            "signature_str": sig_str,
        })

    clusters = work.groupby(group_keys, dropna=False).apply(build_one).reset_index()

    # Build vocabulary: count identical signatures across all clusters
    vocab = (
        clusters.groupby(["signature_str"], dropna=False)
                .agg(frequency=("signature_str","size"),
                     mean_cluster_size=("cluster_size","mean"))
                .reset_index()
                .sort_values(["frequency","signature_str"], ascending=[False, True])
    )
    # Keep tuple for exact matching if needed
    # Note: tuples are not JSON/CSV friendly, so we keep only string in CSV; but return the tuples in memory
    return clusters, vocab


def main():
    ap = argparse.ArgumentParser(description="Build a vocabulary of action-cluster signatures (drug + 2-sig-fig dose from rate/weight).")
    ap.add_argument("--input", required=True, type=Path, help="Triggers dataframe with action_cluster_id (CSV/Parquet/Pickle).")
    ap.add_argument("--outdir", required=True, type=Path, help="Output directory for CSVs.")
    ap.add_argument("--dose-col", default="rate/weight")
    ap.add_argument("--label-col", default="item_label")
    ap.add_argument("--time-col", default="start_time")
    args = ap.parse_args()

    # Load
    p = args.input
    ext = p.suffix.lower()
    if ext in [".parquet", ".pq"]:
        df = pd.read_parquet(p)
    elif ext in [".pkl", ".pickle"]:
        df = pd.read_pickle(p)
    else:
        df = pd.read_csv(p)

    clusters, vocab = build_cluster_vocab(df, dose_col=args.dose_col, label_col=args.label_col, time_col=args.time_col)

    args.outdir.mkdir(parents=True, exist_ok=True)
    clusters_out = args.outdir / "clusters_with_signatures.csv"
    vocab_out = args.outdir / "cluster_vocabulary.csv"

    # Write (tuples are not CSV-friendly; drop them in CSV outputs, keep string)
    clusters_to_write = clusters.drop(columns=["signature_tuple"], errors="ignore")
    clusters_to_write.to_csv(clusters_out, index=False)
    vocab.to_csv(vocab_out, index=False)

    print(f"Wrote clusters to: {clusters_out} ({len(clusters_to_write)} rows)")
    print(f"Wrote vocabulary to: {vocab_out} ({len(vocab)} rows)")


if __name__ == "__main__":
    main()
