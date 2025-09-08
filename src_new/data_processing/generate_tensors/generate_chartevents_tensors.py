#!/usr/bin/env python3
"""
Generate chartevents context tensors for each trajectory.

This script creates tensors representing the 24-hour context window of 
normalized chart events data prior to the t0 of each intervention trajectory.
The output format is compatible with the Raindrop model encoder.
"""

import argparse
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# Module-level globals for fork-based shared memory to improve performance
_CHART_EVENTS_DF: Optional[pd.DataFrame] = None
_ITEM_LABELS: Optional[List[str]] = None
_CONTEXT_DURATION_HOURS: Optional[int] = None
_CONTEXT_INTERVAL_HOURS: Optional[int] = None


def _process_single_trajectory(
    traj_key: str, traj_info: Dict[str, Any], output_dir: Path
) -> Optional[Tuple[str, Dict[str, Any]]]:
    """
    Processes a single trajectory to generate and save its chartevents context tensor.

    For a given trajectory, it filters the global chartevents DataFrame for the
    correct patient and 24-hour time window, creates a time-binned representation,
    and saves it as a PyTorch tensor.

    Args:
        traj_key: The unique identifier for the trajectory (e.g., "hadm_id_cluster_id").
        traj_info: A dictionary containing metadata for the trajectory, including
                   'hadm_id' and 't0_time'.
        output_dir: The directory where the resulting tensor file should be saved.

    Returns:
        A tuple containing the trajectory key and a metadata dictionary for the
        generated tensor, or None if processing fails.
    """
    try:
        hadm_id = traj_info["hadm_id"]
        t0_time = pd.to_datetime(traj_info["t0_time"])

        if getattr(t0_time, "tz", None) is not None:
            t0_time = t0_time.tz_convert("UTC").tz_localize(None)

        context_start = t0_time - pd.Timedelta(hours=_CONTEXT_DURATION_HOURS)
        context_end = t0_time

        patient_df = _CHART_EVENTS_DF[
            (_CHART_EVENTS_DF["hadm_id"] == hadm_id)
            & (_CHART_EVENTS_DF["charttime"] >= context_start)
            & (_CHART_EVENTS_DF["charttime"] < context_end)
        ]

        n_intervals = int(_CONTEXT_DURATION_HOURS / _CONTEXT_INTERVAL_HOURS)
        d_inp = len(_ITEM_LABELS)

        # Create a time grid that aligns with the floored timestamps of the data
        time_grid_start = pd.to_datetime(context_start).floor(f"{_CONTEXT_INTERVAL_HOURS}h")
        time_grid = pd.date_range(
            start=time_grid_start,
            periods=n_intervals,
            freq=f"{_CONTEXT_INTERVAL_HOURS}h",
        )

        if patient_df.empty:
            values = torch.zeros((n_intervals, d_inp), dtype=torch.float32)
            mask = torch.zeros((n_intervals, d_inp), dtype=torch.float32)
        else:
            patient_df_copy = patient_df.copy()
            patient_df_copy["time_bin"] = patient_df_copy["charttime"].dt.floor(
                f"{_CONTEXT_INTERVAL_HOURS}h"
            )

            pivot = patient_df_copy.pivot_table(
                index="time_bin", columns="item_label", values="value", aggfunc="mean"
            )

            pivot = pivot.reindex(columns=_ITEM_LABELS)
            pivot = pivot.reindex(index=time_grid)

            values_np = pivot.fillna(0).to_numpy(dtype=np.float32)
            mask_np = (~pivot.isnull()).to_numpy(dtype=np.float32)

            values = torch.from_numpy(values_np)
            mask = torch.from_numpy(mask_np)

        rd_src = torch.cat([values, 1.0 - mask], dim=1)
        rd_times = torch.tensor(
            (time_grid - t0_time).total_seconds() / 3600, dtype=torch.float32
        )
        rd_length = torch.tensor(n_intervals, dtype=torch.long)

        filepath = output_dir / f"chartevents_context_{traj_key}.pt"
        torch.save((rd_src, rd_times, rd_length), filepath)

        meta = {
            "traj_key": traj_key,
            "hadm_id": hadm_id,
            "file_path": str(filepath),
            "d_inp": d_inp,
            "n_intervals": n_intervals,
            "non_empty_bins": int((mask.sum(dim=1) > 0).sum()),
        }
        return traj_key, meta
    except Exception as e:
        print(f"Error processing trajectory {traj_key}: {e}")
        return None


class ChartEventsTensors:
    def __init__(
        self, chart_events_df: pd.DataFrame, med_metadata: Dict[str, Any]
    ):
        self.chart_events_df = chart_events_df
        self.med_metadata = med_metadata
        self.trajectories = med_metadata["trajectories"]

    def generate_tensors(
        self,
        output_dir: str,
        n_top_features: int = 100,
        context_duration_hours: int = 24,
        context_interval_hours: int = 1,
        n_workers: int = 8,
    ):
        """
        Generates and saves context tensors for all trajectories in parallel.
        """
        output_path = Path(output_dir)
        tensors_dir = output_path / "chartevents_context"
        tensors_dir.mkdir(parents=True, exist_ok=True)
        print(f"Saving generated tensors to: {tensors_dir}")

        print(f"Determining top {n_top_features} features from chartevents...")
        top_items = (
            self.chart_events_df["item_label"]
            .value_counts()
            .nlargest(n_top_features)
            .index.tolist()
        )

        print("Preprocessing chartevents DataFrame...")
        self.chart_events_df["charttime"] = pd.to_datetime(
            self.chart_events_df["charttime"], errors="coerce"
        )
        self.chart_events_df.dropna(subset=['charttime', 'hadm_id'], inplace=True)
        self.chart_events_df['hadm_id'] = self.chart_events_df['hadm_id'].astype(int)

        filtered_df = self.chart_events_df[
            self.chart_events_df["item_label"].isin(top_items)
        ].copy()

        global _CHART_EVENTS_DF, _ITEM_LABELS, _CONTEXT_DURATION_HOURS, _CONTEXT_INTERVAL_HOURS
        _CHART_EVENTS_DF = filtered_df
        _ITEM_LABELS = top_items
        _CONTEXT_DURATION_HOURS = context_duration_hours
        _CONTEXT_INTERVAL_HOURS = context_interval_hours

        print(f"Processing {len(self.trajectories)} trajectories using {n_workers} workers...")
        generated_metadata = {}
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {
                executor.submit(_process_single_trajectory, key, info, tensors_dir): key
                for key, info in self.trajectories.items()
            }

            for future in tqdm(as_completed(futures), total=len(futures), desc="Generating Tensors"):
                result = future.result()
                if result:
                    key, meta = result
                    generated_metadata[key] = meta

        
        final_metadata = {
            "source_med_metadata_path": self.med_metadata.get("source_metadata_path"),
            "generation_params": {
                "n_top_features": n_top_features,
                "context_duration_hours": context_duration_hours,
                "context_interval_hours": context_interval_hours,
                "generated_at": datetime.now().isoformat(),
            },
            "feature_labels": top_items,
            "tensors": generated_metadata,
            "summary_stats": {
                "total_trajectories_processed": len(self.trajectories),
                "tensors_successfully_created": len(generated_metadata),
            }
        }

        metadata_path = output_path / "chartevents_context_metadata.pkl"
        with open(metadata_path, "wb") as f:
            pickle.dump(final_metadata, f)
        
        print("\n=== Generation Summary ===")
        print(f"Successfully created tensors for {len(generated_metadata)} trajectories.")
        print(f"Saved metadata to: {metadata_path}")
        return final_metadata


def create_chartevents_tensors(
    chart_events_path: str,
    med_metadata_path: str,
    output_dir: str,
    n_features: int = 100,
    n_workers: int = 8,
) -> Dict[str, Any]:
    """
    Wrapper function to generate chartevents context tensors.

    This function loads the necessary data and metadata, then orchestrates the
    parallel generation of 24-hour chartevents context tensors for each
    trajectory defined in the medication metadata.

    Args:
        chart_events_path: Path to the normalized chartevents DataFrame (parquet).
        med_metadata_path: Path to the med_tensors_metadata.pkl file.
        output_dir: Directory to save the generated tensors and metadata.
        n_features: Number of most frequent chartevents items to use as features.
        n_workers: Number of parallel workers to use for tensor generation.

    Returns:
        A dictionary containing the metadata for the generated tensors.
    """
    print("\n--- Generating Chartevents Tensors ---")
    chart_events_df = pd.read_parquet(chart_events_path)
    with open(med_metadata_path, "rb") as f:
        med_metadata = pickle.load(f)

    # Add the source path to the metadata for traceability
    med_metadata["source_metadata_path"] = med_metadata_path

    generator = ChartEventsTensors(chart_events_df, med_metadata)
    metadata = generator.generate_tensors(
        output_dir=output_dir,
        n_top_features=n_features,
        n_workers=n_workers,
    )
    return metadata


def main():
    parser = argparse.ArgumentParser(
        description="Generate chartevents context tensors for intervention trajectories."
    )
    parser.add_argument(
        "--chart-events-path",
        required=True,
        help="Path to the normalized chartevents DataFrame (parquet).",
    )
    parser.add_argument(
        "--metadata-path",
        required=True,
        help="Path to the med_tensors_metadata.pkl file.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to save the generated tensors and metadata.",
    )
    parser.add_argument(
        "--n-features",
        type=int,
        default=100,
        help="Number of most frequent chartevents items to use as features.",
    )
    parser.add_argument(
        "--n-workers",
        type=int,
        default=8,
        help="Number of parallel workers to use for tensor generation.",
    )
    args = parser.parse_args()

    # Use the wrapper function
    create_chartevents_tensors(
        chart_events_path=args.chart_events_path,
        med_metadata_path=args.metadata_path,
        output_dir=args.output_dir,
        n_features=args.n_features,
        n_workers=args.n_workers,
    )

if __name__ == "__main__":
    main()
