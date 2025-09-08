#!/usr/bin/env python3
"""
Consolidate trajectory metadata from med tensors and physio tensors.

This script combines metadata from:
1. Med tensors metadata (medications at t0, trigger info)
2. Physio tensors metadata (response paths for ABP/CVP)

Into a single consolidated metadata file with all trajectory information.
"""

import argparse
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import os
import pandas as pd
from collections import Counter, defaultdict
import numpy as np


def load_metadata(metadata_path: str) -> Dict[str, Any]:
    """Load metadata from pickle file."""
    with open(metadata_path, "rb") as f:
        return pickle.load(f)


def extract_trigger_medications(
    med_data_path: str, 
    med_metadata: Dict[str, Any]
) -> Dict[str, List[str]]:
    """
    Extract which medications triggered each trajectory.
    
    Args:
        med_data_path: Path to the original medication data file (e.g., mv_filtered.bin)
        med_metadata: Med tensors metadata
        
    Returns:
        Dictionary mapping trajectory keys to list of triggering medication labels
    """
    print(f"Loading original medication data from {med_data_path} to extract trigger information...")
    
    # Load the original medication data - use the same loader as create_med_tensors
    from create_med_tensors import load_medication_data
    df_full, _ = load_medication_data(med_data_path)
    
    # Ensure we have the trigger column
    if "trigger" not in df_full.columns:
        print("Warning: No 'trigger' column found in medication data. Trigger info will be empty.")
        return {traj_key: [] for traj_key in med_metadata["trajectories"].keys()}
    
    trigger_medications = {}
    
    for traj_key, traj_info in med_metadata["trajectories"].items():
        hadm_id = traj_info["hadm_id"]
        action_cluster_id = traj_info["action_cluster_id"]
        
        # Find the specific trigger event(s) for this trajectory by filtering
        # on both the patient ID and the specific action cluster ID. This is
        # the most robust way to isolate the trigger.
        trigger_meds = df_full[
            (df_full["hadm_id"] == hadm_id) & 
            (df_full["action_cluster_id"] == action_cluster_id) &
            (df_full["trigger"] == True)
        ]
        
        # Get the medication labels for these triggers
        trigger_labels = trigger_meds["item_label"].unique().tolist()
        trigger_medications[traj_key] = trigger_labels
    
    return trigger_medications


# --- helpers
def _to_utc(s): 
    return pd.to_datetime(s, errors="coerce", utc=True)

def _norm_label(x: str) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)): 
        return ""
    return str(x).strip()

# Treat IV bolus of NaCl/LR as distinct tokens
_BOLUS_NAME = "03-IV Fluid Bolus"
_LR_ALIASES = {"LR"}
_NACL_ALIASES = {"NaCl 0.9%"}  # extend as needed

def _med_token(row: pd.Series) -> str:
    base = _norm_label(row.get("item_label"))
    iname = str(row.get("input_name", "")).strip()
    if iname == _BOLUS_NAME:
        if base in _NACL_ALIASES:
            return f"{base} [Bolus]"
        if base in _LR_ALIASES:
            return f"{base} [Bolus]"
    return base

def rank_action_window_combinations(
    df: pd.DataFrame,
    window_minutes: int = 20,
    drop_empty: bool = True,
):
    """
    For each (hadm_id, action_cluster_id):
      • t0 = earliest trigger start_time; fallback to earliest start if no trigger in cluster
      • action window = [t0, t0 + window_minutes]
      • include any med rows that overlap the window:
          start_time <= window_end  AND  (end_time is NA OR end_time >= t0)
      • build a unique, sorted tuple of med tokens for that cluster
        (NaCl/LR IV bolus gets a '[Bolus]' tag separate from drips)

    Returns:
      combo_df: DataFrame with columns ['combo','count','percent','example_pairs']
      cluster_combo_map: DataFrame mapping each (hadm_id, action_cluster_id) to its 'combo'
    """
    m = df.copy()
    # keep only clusters
    m = m[m["action_cluster_id"].notna()]
    if m.empty:
        return (
            pd.DataFrame(columns=["combo","count","percent","example_pairs"]),
            pd.DataFrame(columns=["hadm_id","action_cluster_id","combo"])
        )

    # times
    m["start_time"] = _to_utc(m["start_time"])
    m["end_time"]   = _to_utc(m.get("end_time"))

    gkeys = ["hadm_id", "action_cluster_id"]

    # t0 = earliest trigger; fallback to earliest start
    trig_mask = m["trigger"].fillna(False)
    t0_trigger  = m.loc[trig_mask].groupby(gkeys)["start_time"].min()
    t0_fallback = m.groupby(gkeys)["start_time"].min()
    t0_series   = t0_trigger.combine_first(t0_fallback)

    delta = pd.Timedelta(minutes=window_minutes)

    cluster_pairs   = []
    cluster_combos  = []
    for (hadm, cid), t0 in t0_series.items():
        sub = m[(m["hadm_id"] == hadm) & (m["action_cluster_id"] == cid)]
        if pd.isna(t0):
            continue
        window_end = t0 + delta

        # overlap with action window
        st = sub["start_time"]
        et = sub["end_time"]
        overlap = (st <= window_end) & (et.isna() | (et >= t0))
        win = sub.loc[overlap].copy()
        if win.empty and drop_empty:
            continue

        # make tokens (bolus NaCl/LR distinct)
        win["med_token"] = win.apply(_med_token, axis=1)
        meds = tuple(sorted(t for t in win["med_token"].dropna().unique() if t and t.strip()))
        if not meds and drop_empty:
            continue

        cluster_pairs.append((hadm, cid))
        cluster_combos.append(meds)

    if not cluster_combos:
        return (
            pd.DataFrame(columns=["combo","count","percent","example_pairs"]),
            pd.DataFrame(columns=["hadm_id","action_cluster_id","combo"])
        )

    # Count combos across all clusters
    counter = Counter(cluster_combos)
    total   = sum(counter.values())

    # Ranked table
    rows = []
    # collect up to a few example pairs per combo
    examples = defaultdict(list)
    for pair, combo in zip(cluster_pairs, cluster_combos):
        if len(examples[combo]) < 5:
            examples[combo].append(pair)

    for combo, cnt in counter.most_common():
        rows.append({
            "combo": combo,
            "count": cnt,
            "percent": cnt / total if total else np.nan,
            "example_pairs": examples[combo],  # list of (hadm_id, action_cluster_id)
        })
    combo_df = pd.DataFrame(rows)

    # Mapping each cluster -> its combo
    cluster_combo_map = pd.DataFrame(cluster_pairs, columns=["hadm_id","action_cluster_id"])
    cluster_combo_map["combo"] = cluster_combos

    return combo_df, cluster_combo_map


def consolidate_trajectory_metadata(
    med_metadata_path: str,
    physio_metadata_path: str,
    output_path: str,
    med_data_path: Optional[str] = None,
    chartevents_metadata_path: Optional[str] = None,
):
    """
    Consolidates trajectory metadata from medication and physiological tensors.
    
    Args:
        med_metadata_path: Path to med tensors metadata.pkl
        physio_metadata_path: Path to physio tensors metadata.pkl
        output_path: Path to save consolidated metadata
        med_data_path: Path to original medication data (for trigger extraction)
        chartevents_metadata_path: Path to chartevents context metadata.pkl
        
    Returns:
        Consolidated metadata dictionary
    """
    print("=== Loading Metadata ===")
    
    # Load both metadata files
    med_metadata = load_metadata(med_metadata_path)
    physio_metadata = load_metadata(physio_metadata_path)
    
    # Load chartevents context metadata if provided
    chartevents_tensors_meta = {}
    if chartevents_metadata_path and os.path.exists(chartevents_metadata_path):
        print(f"Loading chartevents context metadata from {chartevents_metadata_path}")
        chartevents_metadata = load_metadata(chartevents_metadata_path)
        chartevents_tensors_meta = chartevents_metadata.get("tensors", {})
        print(f"Loaded chartevents metadata with {len(chartevents_tensors_meta)} tensors")
    else:
        print("No chartevents metadata path provided. Context paths will be empty.")

    print(f"Loaded med metadata with {len(med_metadata['trajectories'])} trajectories")
    print(f"Loaded physio metadata with {len(physio_metadata['ic_tensors'])} IC tensors")
    
    # Extract trigger medications and rank action combos if medication data is provided
    trigger_medications = {}
    combo_df = pd.DataFrame()
    cluster_combo_map = pd.DataFrame()
    
    if med_data_path and os.path.exists(med_data_path):
        from create_med_tensors import load_medication_data
        df_full, _ = load_medication_data(med_data_path)
        trigger_medications = extract_trigger_medications(med_data_path, med_metadata)
        print("Ranking medication combinations...")
        combo_df, cluster_combo_map = rank_action_window_combinations(df_full)
        # Create a lookup map from (hadm_id, action_cluster_id) to combo
        combo_lookup = cluster_combo_map.set_index(['hadm_id', 'action_cluster_id'])['combo'].to_dict()
        # Create a mapping from combo tuple to a unique integer ID
        # Stable, deterministic IDs: sort by frequency(desc), then lexicographically by token tuple
        combos_sorted = (
            combo_df.sort_values(['count','combo'], ascending=[False, True])['combo'].tolist()
        )
        combo_to_id = {combo: i for i, combo in enumerate(combos_sorted)}
    else:
        print("No medication data path provided. Trigger and combo information will be empty.")
        trigger_medications = {traj_key: [] for traj_key in med_metadata["trajectories"].keys()}
        combo_lookup = {}
        combo_to_id = {}
    
    # Consolidate trajectory information
    print("\n=== Consolidating Trajectory Metadata ===")
    
    consolidated_trajectories = {}
    
    for traj_key, med_traj_info in med_metadata["trajectories"].items():
        # Get corresponding physio info if it exists
        ic_info = physio_metadata["ic_tensors"].get(traj_key)
        pred_info = physio_metadata["prediction_targets"].get(traj_key)
        
        # Get medication combination for this trajectory
        hadm_id = med_traj_info["hadm_id"]
        action_cluster_id = med_traj_info["action_cluster_id"]
        combo = combo_lookup.get((hadm_id, action_cluster_id), tuple())
        combo_id = combo_to_id.get(combo, -1) # -1 for combos not found (e.g., empty)

        # Get chartevents context path
        chartevents_context_info = chartevents_tensors_meta.get(traj_key)

        # Build consolidated trajectory info
        consolidated_traj = {
            # Basic trajectory info
            "hadm_id": med_traj_info["hadm_id"],
            "traj_id": traj_key,  # This is the trajectory key (hadm_id_action_cluster_id)
            "action_cluster_id": med_traj_info["action_cluster_id"],
            "t0_time": med_traj_info["t0_time"],
            "trajectory_end_time": med_traj_info["trajectory_end_time"],
            "duration_minutes": med_traj_info["duration_minutes"],
            
            # Medication combination info
            "med_combo": combo,
            "med_combo_id": combo_id,

            # Medication information at t0
            "medications_at_t0": med_traj_info.get("medications_at_t0", {}),
            "n_medications_at_t0": med_traj_info.get("n_medications_at_t0", 0),
            
            # Trigger information
            "trigger_medications": trigger_medications.get(traj_key, []),
            "n_trigger_medications": len(trigger_medications.get(traj_key, [])),
            
            # Response paths (if available)
            "ic_tensor_path": ic_info["file_path"] if ic_info else None,
            "prediction_targets_path": pred_info["file_path"] if pred_info else None,
            
            # Physiological data availability
            "has_abp_mean": ic_info.get("has_abp_mean", False) if ic_info else False,
            "has_cvp": ic_info.get("has_cvp", False) if ic_info else False,
            "abp_mean_value": ic_info.get("abp_mean_value") if ic_info else None,
            "cvp_value": ic_info.get("cvp_value") if ic_info else None,
            
            # Prediction target statistics
            "total_abp_mean_measurements": pred_info.get("total_abp_mean_measurements", 0) if pred_info else 0,
            "total_cvp_measurements": pred_info.get("total_cvp_measurements", 0) if pred_info else 0,
            
            # Data availability flags
            "has_medication_data": True,  # Always true if in med metadata
            "has_physiological_data": ic_info is not None,
            "has_complete_data": ic_info is not None and pred_info is not None,
            
            # New: Path to chartevents context tensor
            "chartevents_context_path": chartevents_context_info.get("file_path") if chartevents_context_info else None,
        }
        
        consolidated_trajectories[traj_key] = consolidated_traj
    
    # Create summary statistics
    total_trajectories = len(consolidated_trajectories)
    trajectories_with_physio = sum(1 for t in consolidated_trajectories.values() if t["has_physiological_data"])
    trajectories_with_triggers = sum(1 for t in consolidated_trajectories.values() if t["n_trigger_medications"] > 0)
    trajectories_with_meds_at_t0 = sum(1 for t in consolidated_trajectories.values() if t["n_medications_at_t0"] > 0)
    
    # Get unique medication labels
    all_meds_at_t0 = set()
    all_trigger_meds = set()
    for traj in consolidated_trajectories.values():
        all_meds_at_t0.update(traj["medications_at_t0"].keys())
        all_trigger_meds.update(traj["trigger_medications"])
    
    # Create consolidated metadata
    consolidated_metadata = {
        "source_metadata": {
            "med_metadata_path": med_metadata_path,
            "physio_metadata_path": physio_metadata_path,
            "chartevents_metadata_path": chartevents_metadata_path,
            "med_data_path": med_data_path,
        },
        "trajectories": consolidated_trajectories,
        "med_combos": {
            "combo_df": combo_df,
            "combo_to_id": combo_to_id,
            "id_to_combo": {i: combo for combo, i in combo_to_id.items()}
        },
        "summary_stats": {
            "total_trajectories": total_trajectories,
            "trajectories_with_physiological_data": trajectories_with_physio,
            "trajectories_with_trigger_medications": trajectories_with_triggers,
            "trajectories_with_medications_at_t0": trajectories_with_meds_at_t0,
            "unique_medications_at_t0": len(all_meds_at_t0),
            "unique_trigger_medications": len(all_trigger_meds),
            "medications_at_t0_list": sorted(list(all_meds_at_t0)),
            "trigger_medications_list": sorted(list(all_trigger_meds)),
        },
        "medication_mappings": {
            "item_to_idx": med_metadata.get("item_to_idx", {}),
            "idx_to_item": med_metadata.get("idx_to_item", {}),
            "item_labels": med_metadata.get("item_labels", []),
        },
        "time_grid": {
            "n_intervals": med_metadata.get("n_intervals"),
            "interval_seconds": med_metadata.get("interval_seconds"),
        },
    }
    
    # Save consolidated metadata
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "wb") as f:
            pickle.dump(consolidated_metadata, f)
        
        print(f"\nConsolidated metadata saved to: {output_path}")
    
    # Print summary
    print("\n=== Consolidation Summary ===")
    print(f"Total trajectories: {total_trajectories}")
    print(f"Trajectories with physiological data: {trajectories_with_physio} ({trajectories_with_physio/total_trajectories*100:.1f}%)")
    print(f"Trajectories with trigger medications: {trajectories_with_triggers} ({trajectories_with_triggers/total_trajectories*100:.1f}%)")
    print(f"Trajectories with medications at t0: {trajectories_with_meds_at_t0} ({trajectories_with_meds_at_t0/total_trajectories*100:.1f}%)")
    print(f"Unique medications at t0: {len(all_meds_at_t0)}")
    print(f"Unique trigger medications: {len(all_trigger_meds)}")
    
    return consolidated_metadata


def inspect_consolidated_metadata(metadata: Dict[str, Any], n_samples: int = 3) -> None:
    """Inspect sample trajectories from consolidated metadata."""
    print(f"\n=== Inspecting {n_samples} Sample Trajectories ===")
    
    traj_keys = list(metadata["trajectories"].keys())[:n_samples]
    
    for traj_key in traj_keys:
        traj = metadata["trajectories"][traj_key]
        print(f"\nTrajectory: {traj_key}")
        print(f"  HADM ID: {traj['hadm_id']}")
        print(f"  Action Cluster ID: {traj['action_cluster_id']}")
        print(f"  T0 Time: {traj['t0_time']}")
        print(f"  Duration: {traj['duration_minutes']:.1f} minutes")
        print(f"  Medications at t0: {traj['n_medications_at_t0']}")
        if traj['medications_at_t0']:
            for med, rate in list(traj['medications_at_t0'].items())[:3]:  # Show first 3
                print(f"    {med}: {rate:.3f}")
        print(f"  Trigger medications: {traj['trigger_medications']}")
        print(f"  Has ABP: {traj['has_abp_mean']} (value: {traj['abp_mean_value']})")
        print(f"  Has CVP: {traj['has_cvp']} (value: {traj['cvp_value']})")
        print(f"  IC tensor path: {traj['ic_tensor_path']}")
        print(f"  Prediction targets path: {traj['prediction_targets_path']}")


def main():
    parser = argparse.ArgumentParser(
        description="Consolidate trajectory metadata from med and physio tensors"
    )
    
    parser.add_argument(
        "--med-metadata",
        required=True,
        help="Path to med tensors metadata.pkl"
    )
    parser.add_argument(
        "--physio-metadata", 
        required=True,
        help="Path to physio tensors metadata.pkl"
    )
    parser.add_argument(
        "--med-data",
        help="Path to original medication data file (for trigger extraction)"
    )
    parser.add_argument(
        "--output",
        help="Path to save consolidated metadata (default: consolidated_trajectory_metadata.pkl)"
    )
    parser.add_argument(
        "--inspect",
        action="store_true",
        help="Inspect sample trajectories after consolidation"
    )
    
    args = parser.parse_args()
    
    # Set default output path
    if not args.output:
        med_metadata_dir = Path(args.med_metadata).parent
        args.output = med_metadata_dir / "consolidated_trajectory_metadata.pkl"
    
    # Consolidate metadata
    consolidated_metadata = consolidate_trajectory_metadata(
        med_metadata_path=args.med_metadata,
        physio_metadata_path=args.physio_metadata,
        output_path=args.output,
        med_data_path=args.med_data,
    )
    
    # Inspect if requested
    if args.inspect:
        inspect_consolidated_metadata(consolidated_metadata)


if __name__ == "__main__":
    import os
    main()
