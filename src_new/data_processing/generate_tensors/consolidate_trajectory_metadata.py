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

import pandas as pd


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
        t0_time = pd.to_datetime(traj_info["t0_time"])
        
        # Find medications that triggered this trajectory
        patient_data = df_full[df_full["hadm_id"] == hadm_id]
        trigger_meds = patient_data[
            (patient_data["trigger"] == True) & 
            (patient_data["start_time"] == t0_time)
        ]
        
        # Get the medication labels for these triggers
        trigger_labels = trigger_meds["item_label"].unique().tolist()
        trigger_medications[traj_key] = trigger_labels
    
    return trigger_medications


def consolidate_trajectory_metadata(
    med_metadata_path: str,
    physio_metadata_path: str,
    med_data_path: Optional[str] = None,
    output_path: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Consolidate trajectory metadata from med and physio tensors.
    
    Args:
        med_metadata_path: Path to med tensors metadata.pkl
        physio_metadata_path: Path to physio tensors metadata.pkl
        med_data_path: Path to original medication data (for trigger extraction)
        output_path: Path to save consolidated metadata
        
    Returns:
        Consolidated metadata dictionary
    """
    print("=== Loading Metadata ===")
    
    # Load both metadata files
    med_metadata = load_metadata(med_metadata_path)
    physio_metadata = load_metadata(physio_metadata_path)
    
    print(f"Loaded med metadata with {len(med_metadata['trajectories'])} trajectories")
    print(f"Loaded physio metadata with {len(physio_metadata['ic_tensors'])} IC tensors")
    
    # Extract trigger medications if medication data is provided
    trigger_medications = {}
    if med_data_path and os.path.exists(med_data_path):
        trigger_medications = extract_trigger_medications(med_data_path, med_metadata)
    else:
        print("No medication data path provided. Trigger information will be empty.")
        trigger_medications = {traj_key: [] for traj_key in med_metadata["trajectories"].keys()}
    
    # Consolidate trajectory information
    print("\n=== Consolidating Trajectory Metadata ===")
    
    consolidated_trajectories = {}
    
    for traj_key, med_traj_info in med_metadata["trajectories"].items():
        # Get corresponding physio info if it exists
        ic_info = physio_metadata["ic_tensors"].get(traj_key)
        pred_info = physio_metadata["prediction_targets"].get(traj_key)
        
        # Build consolidated trajectory info
        consolidated_traj = {
            # Basic trajectory info
            "hadm_id": med_traj_info["hadm_id"],
            "traj_id": traj_key,  # This is the trajectory key (hadm_id_action_cluster_id)
            "action_cluster_id": med_traj_info["action_cluster_id"],
            "t0_time": med_traj_info["t0_time"],
            "trajectory_end_time": med_traj_info["trajectory_end_time"],
            "duration_minutes": med_traj_info["duration_minutes"],
            
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
            "med_data_path": med_data_path,
        },
        "trajectories": consolidated_trajectories,
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
        med_data_path=args.med_data,
        output_path=args.output,
    )
    
    # Inspect if requested
    if args.inspect:
        inspect_consolidated_metadata(consolidated_metadata)


if __name__ == "__main__":
    import os
    main()
