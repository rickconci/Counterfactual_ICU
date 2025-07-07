import pandas as pd
import matplotlib.pyplot as plt
import os
import polars as pl
import gc
from tqdm import tqdm
from typing import Dict, Any, List, Tuple
from scipy.spatial.distance import cdist
import numpy as np
import json
import torch
import pickle
import concurrent.futures
from functools import partial
from pathlib import Path
from datetime import datetime, timedelta
from scipy.interpolate import interp1d
import argparse


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Process MIMIC-IV data for ICU patient trajectories')

    # Time interval parameters
    parser.add_argument('--interval-minutes', type=int, default=5,
                        help='Time interval in minutes between observations (default: 5)')

    # Trajectory window parameters
    parser.add_argument('--trajectory-before-minutes', type=int, default=None,
                        help='Minutes before t0 to include in trajectory (default: from ICU admission)')
    parser.add_argument('--trajectory-after-minutes', type=int, default=0,
                        help='Minutes after t0 to include in trajectory (default: 0, ends at t0)')

    # CO/R_TPR parameters
    parser.add_argument('--max-co-age-minutes', type=int, default=10,
                        help='Maximum age of CO measurement for R_TPR calculation (default: 10)')
    parser.add_argument('--co-guess', type=float, default=4.0,
                        help='Default CO value if no recent measurement (default: 4.0)')

    # Processing parameters
    parser.add_argument('--n-workers', type=int, default=4,
                        help='Number of parallel workers (default: 4)')
    parser.add_argument('--debug-patient-id', type=int, default=20214994,
                        help='Patient ID to save debug CSV files for (default: 32128372)')

    # Data limit parameter (for testing)
    parser.add_argument('--data-limit', type=int, default=None,
                        help='Limit on number of rows to read from CSV files (default: None for unlimited)')

    # Output directories
    parser.add_argument('--output-dir', type=str, default='../../data/processed_data',
                        help='Base output directory (default: ../../data/processed_data)')
    parser.add_argument('--input-dir', type=str, default='../../data/input_data',
                        help='Base input directory (default: ../../input_data)')

    return parser.parse_args()

def find_relevant_patients(measurements, MAP_id = 220052, load_path_events = "../../data/input_data/chartevents.csv",
                          load_path_stays = "../../data/input_data/icustays.csv",  save_path = "../../data/processed_data/treated_patients_chartevents.parquet",
                          data_limit=None):
    """
    Finds all potentially relevant patients by filtering on those that have had a blood pressure event and
    that have stayed in the ICU for over 24h.
    Args:
        measurements: List of measurement IDs to extract
        MAP_id: Measurement ID for mean arterial pressure
        load_path_events: path to original dataset containing all events
        load_path_stays: path to dataset containing meta-information on ICU stay
        save_path: path to save relevant patients
        data_limit: Maximum number of rows to read from CSV

    Returns:
            dataset containing all occurrences of the treatment to be used to filter patients
    """
    if not os.path.exists(save_path):
        long_stays_query = pl.scan_csv(load_path_stays)
        treated_patients_query = pl.scan_csv(load_path_events)
        if data_limit is not None:
            long_stays_query = long_stays_query.limit(data_limit)
            treated_patients_query = treated_patients_query.limit(data_limit)

        long_stays = (long_stays_query.filter(pl.col("los") > 1).collect())
        long_stays_id = long_stays["hadm_id"].unique().to_list()
        treated_patients = (treated_patients_query.filter(pl.col("hadm_id").is_in(long_stays_id))
                                .filter(pl.col("itemid") == MAP_id)
                                .filter(pl.col("value").cast(pl.Float64, strict=False) < 70)
                                .collect())

        treated_patients_all_values = read_large_csv_with_polars(load_path_events, treated_patients, measurements, data_limit=data_limit)
        treated_patients_all_values.write_parquet(save_path)
        print(f"Saved new dataset of patient values to {save_path}")
    else:
        print(f"Loading dataset from {save_path}")
        treated_patients_all_values = pl.read_parquet(save_path)

    return treated_patients_all_values

def read_large_csv_with_polars(load_path, ids_df, measurements, id_column='hadm_id', item_column = 'itemid', data_limit=None):
    """
    Function to get all measurements from patients that have had the treatment
    Args:
        load_path: The path to the dataset of all patient measurements
        ids_df: df containing the ID's of patients with the treatment
        measurements: All IDs of measurements necessary for modelling
        id_column: The column to merge the dataset
        item_column: The column containing measurement IDs
        data_limit: Maximum number of rows to read

    Returns: df with the treated patient's events

    """

    valid_ids = ids_df[id_column].unique().to_list()
    query = pl.scan_csv(load_path)
    if data_limit is not None:
        query = query.limit(data_limit)
    result = (
        query.filter(pl.col(id_column).is_in(valid_ids))
        .filter(pl.col(item_column).is_in(measurements))
        .collect()
    )

    return result


def find_min_max_heartrates(all_patients_path, save_path, metadata_path, hr_ID = 220045, patient_id = "subject_id", item_column = "itemid", value_column = "value"):
    """
    Function to find the minimum and maximum heart rates and save them as parquet
    Args:
        all_patients_path: path to all patient measurement data
        metadata_path: path to the metadata for each patient

    Returns: dataframe hr_params with min and max HR for each patient
    """
    all_patients = pl.scan_parquet(all_patients_path).collect()
    patients = all_patients[patient_id].unique().to_list()
    patient_metadata = pl.scan_csv(metadata_path).collect()
    hr_params = pd.DataFrame({"subject_id":[], "min_hr":[], "max_hr":[]})
    for patient in patients:
        patient = int(patient)
        patient_data = (all_patients
                        .filter(pl.col(patient_id) == patient)
                        .filter(pl.col(item_column) == hr_ID))

        # TODO is 50 reasonable
        patient_min = min(50, patient_data[value_column].cast(pl.Float64).min())
        current_patient_metadata = patient_metadata.filter(pl.col(patient_id) == patient)
        patient_age = 0 if len(current_patient_metadata["anchor_age"]) == 0 else current_patient_metadata["anchor_age"].item()
        patient_max = 220 - patient_age
        hr_params_patients = pd.DataFrame({"subject_id":[patient], "min_hr":[patient_min], "max_hr":[patient_max]})
        hr_params = pd.concat((hr_params, hr_params_patients))
    hr_params.to_csv(save_path)
    return hr_params


def find_relevant_inputevents(all_patients_chartevents, save_path, events, inputevents_path = "../../data/input_data/inputevents.csv",
                             patient_id = "hadm_id", item_column = "itemid", data_limit=None):
    if not os.path.exists(save_path):
        patients = all_patients_chartevents[patient_id].unique().to_list()
        schema_overrides = {
            'totalamount': pl.Float64  # or pl.Float32 if you want less precision
        }
        query = pl.scan_csv(inputevents_path, schema_overrides=schema_overrides)
        if data_limit is not None:
            query = query.limit(data_limit)
        all_patients_inputevents = (query.filter(pl.col(patient_id).is_in(patients))
                                        .filter(pl.col(item_column).is_in(events))
                                        .collect())

        all_patients_inputevents.write_parquet(save_path)
        print(f"Saved new dataset to {save_path}")
    else:
        print(f"Loading existing dataset from {save_path}")
        all_patients_inputevents = pl.read_parquet(save_path)

    return all_patients_inputevents


def analyze_icu_stays(icustays_path='../../data/icustays.csv'):
    """
    Analyze the distribution of ICU stay lengths
    """
    icustays_df = pd.read_csv(icustays_path)

    print("ICU Length of Stay Statistics:")
    print(f"Mean: {icustays_df['los'].mean():.2f} days")
    print(f"Median: {icustays_df['los'].median():.2f} days")
    print(f"Max: {icustays_df['los'].max():.2f} days")
    print(f"95th percentile: {icustays_df['los'].quantile(0.95):.2f} days")
    print(f"99th percentile: {icustays_df['los'].quantile(0.99):.2f} days")

    # Show distribution of very long stays
    long_stays = icustays_df[icustays_df['los'] > 30]
    print(f"\nStays longer than 30 days: {len(long_stays)} ({len(long_stays) / len(icustays_df) * 100:.1f}%)")

    return icustays_df['los'].describe()


def debug_specific_patient(parquet_path='../../data/processed_data/treated_patients_inputevents.parquet',
                            itemid=225158,
                            hadm_id=20214994,
                            output_file='../../data/processed_data/debug_inputevents.csv',
                            chartevents_path = '../../data/processed_data/treated_patients_chartevents.parquet'):
    """
    Extract and examine all records for a specific itemid and hadm_id
    """
    df = pd.read_parquet(parquet_path)

    # Filter for the specific itemid and hadm_id
    filtered = df[(df['itemid'] == itemid) & (df['hadm_id'] == hadm_id)].copy()

    # Sort by starttime to see the sequence
    filtered = filtered.sort_values('starttime')

    # Print summary
    print(f"Found {len(filtered)} records for itemid={itemid}, hadm_id={hadm_id}")
    print("\nColumns in dataset:")
    print(filtered.columns.tolist())

    # Display the records
    print("\nRecords (sorted by starttime):")
    print(filtered[['starttime', 'endtime', 'rate', 'amount', 'statusdescription']].to_string())

    # Save to CSV
    filtered.to_csv(output_file, index=False)
    print(f"\nSaved {len(filtered)} records to {output_file}")

    # Show rate changes
    if len(filtered) > 1:
        print("\nRate progression:")
        for idx, row in filtered.iterrows():
            print(f"  {row['starttime']} -> {row['endtime']}: {row['rate']} mL/hr ({row['statusdescription']})")

    chartevents = pl.scan_parquet(chartevents_path).filter(pl.col("hadm_id") == hadm_id).collect()
    chartevents.write_csv("../../data/processed_data/debug_chartevents.csv")

    return filtered


def process_single_patient_physio(
        hadm_id,
        chartevents_path,  # Path to parquet file instead of DataFrame
        physio_params,
        co_itemids,  # Need CO itemids for r_tpr calculation
        n_intervals,
        interval_minutes,
        icu_admission_time,
        trajectory_boundaries,
        cache_dir,
        save_prediction_targets=True,
        prediction_target_dir="../../data/processed_data/prediction_targets"
):
    """
    Process physiological measurements for a single patient and save as trajectory tensors.

    Now optionally saves prediction target tensors (MAP and CVP only) at the same time
    to avoid re-reading p_out tensors later.

    Uses the same trajectory boundaries as medication tensors (based on t0 triggers).

    Mask logic:
    - mask=1 if measurement exists in that interval, mask=0 if no measurement
    - For r_tpr: mask=1 only if MAP, CVP, and CO all exist in that interval

    Returns:
        Tuple of (hadm_id, trajectory_info) where trajectory_info is a list of
        dictionaries containing trajectory metadata
    """
    # Read only this patient's data from parquet
    chart_df_patient = pl.read_parquet(chartevents_path).filter(
        pl.col('hadm_id') == hadm_id
    )

    # Initialize arrays for all physiological parameters
    n_params = len(physio_params)
    values_array = np.zeros((n_intervals, n_params), dtype=np.float32)
    mask_array = np.zeros((n_intervals, n_params), dtype=np.float32)
    count_array = np.zeros((n_intervals, n_params), dtype=np.int32)  # Track number of measurements

    # Create parameter index mapping
    param_idx_map = {}
    map_idx = None
    cvp_idx = None
    r_tpr_idx = None

    for idx, param in enumerate(physio_params):
        if param['param_type'] == 'MAP':
            map_idx = idx
        elif param['param_type'] == 'CVP':
            cvp_idx = idx
        elif param['param_type'] == 'R_TPR':
            r_tpr_idx = idx
        else:
            param_idx_map[param['itemid']] = idx

    # Also need to track CO values for r_tpr calculation
    co_values_array = np.zeros((n_intervals,), dtype=np.float32)
    co_mask_array = np.zeros((n_intervals,), dtype=np.float32)
    co_count_array = np.zeros((n_intervals,), dtype=np.int32)

    # Process each physiological parameter (except r_tpr which is calculated)
    for idx, param_info in enumerate(physio_params):
        if param_info['param_type'] == 'R_TPR':
            continue  # Skip r_tpr as it's calculated

        itemid = param_info['itemid']

        # Get measurements for this parameter
        param_events = chart_df_patient.filter(pl.col('itemid') == itemid)

        if param_events.height > 0:
            # Group by time interval and average
            for row in param_events.iter_rows(named=True):
                # Calculate which interval this measurement belongs to
                time_idx = row['time_idx']
                if 0 <= time_idx < n_intervals:
                    # If first measurement in this interval, set it
                    if mask_array[time_idx, idx] == 0:
                        values_array[time_idx, idx] = row['value']
                        mask_array[time_idx, idx] = 1.0
                        count_array[time_idx, idx] = 1
                    else:
                        # Calculate running average
                        current_count = count_array[time_idx, idx]
                        current_sum = values_array[time_idx, idx] * current_count
                        new_count = current_count + 1
                        values_array[time_idx, idx] = (current_sum + row['value']) / new_count
                        count_array[time_idx, idx] = new_count

    # Process CO measurements separately for r_tpr calculation
    for co_itemid in co_itemids:
        co_events = chart_df_patient.filter(pl.col('itemid') == co_itemid)

        if co_events.height > 0:
            for row in co_events.iter_rows(named=True):
                time_idx = row['time_idx']
                if 0 <= time_idx < n_intervals:
                    if co_mask_array[time_idx] == 0:
                        co_values_array[time_idx] = row['value']
                        co_mask_array[time_idx] = 1.0
                        co_count_array[time_idx] = 1
                    else:
                        # Calculate running average
                        current_count = co_count_array[time_idx]
                        current_sum = co_values_array[time_idx] * current_count
                        new_count = current_count + 1
                        co_values_array[time_idx] = (current_sum + row['value']) / new_count
                        co_count_array[time_idx] = new_count

    # Calculate r_tpr for each time interval
    if r_tpr_idx is not None and map_idx is not None and cvp_idx is not None:
        for t in range(n_intervals):
            # Check if all three measurements exist
            if (mask_array[t, map_idx] == 1 and
                    mask_array[t, cvp_idx] == 1 and
                    co_mask_array[t] == 1 and
                    co_values_array[t] > 0):  # Avoid division by zero

                map_value = values_array[t, map_idx]
                cvp_value = values_array[t, cvp_idx]
                co_value = co_values_array[t]

                # Calculate r_tpr = (MAP - CVP) / CO
                values_array[t, r_tpr_idx] = (map_value - cvp_value) / co_value
                mask_array[t, r_tpr_idx] = 1.0
            else:
                # Missing data - set to 0 with mask 0
                values_array[t, r_tpr_idx] = 0.0
                mask_array[t, r_tpr_idx] = 0.0

    # Calculate time arrays (same as medication tensors)
    rel_time_array = np.arange(n_intervals) * interval_minutes / 60.0
    abs_time_array = rel_time_array.copy()

    # Save each trajectory using the provided boundaries
    trajectory_info = []
    prediction_target_info = []  # NEW: Track prediction target info

    # Loop up to the second to last boundary to define p_out
    for traj_num in range(len(trajectory_boundaries) - 1):
        # Current trajectory defines p_in
        start_idx_in, end_idx_in = trajectory_boundaries[traj_num]

        # Next trajectory defines the end of p_out
        _, end_idx_out = trajectory_boundaries[traj_num + 1]

        # --- P_IN ---
        p_in_values = values_array[start_idx_in:end_idx_in, :]
        p_in_mask = mask_array[start_idx_in:end_idx_in, :]
        p_in_abs_time = abs_time_array[start_idx_in:end_idx_in]
        p_in_rel_time = rel_time_array[start_idx_in:end_idx_in]
        p_in_len = end_idx_in - start_idx_in

        # Convert to tensors
        p_in_values_tensor = torch.from_numpy(p_in_values).float()
        p_in_mask_tensor = torch.from_numpy(p_in_mask).float()
        p_in_abs_time_tensor = torch.from_numpy(p_in_abs_time).float()
        p_in_rel_time_tensor = torch.from_numpy(p_in_rel_time).float()

        # Save p_in tensor
        file_path_in = os.path.join(cache_dir, f"p_tensor_in_{int(hadm_id)}_traj_{traj_num:03d}.pt")
        torch.save(
            (p_in_values_tensor, p_in_mask_tensor, p_in_abs_time_tensor, p_in_rel_time_tensor, p_in_len),
            file_path_in
        )

        # --- P_OUT ---
        # p_out is the segment from the end of p_in to the end of the next full trajectory
        p_out_values = values_array[end_idx_in:end_idx_out, :]
        p_out_mask = mask_array[end_idx_in:end_idx_out, :]
        p_out_abs_time = abs_time_array[end_idx_in:end_idx_out]
        p_out_rel_time = rel_time_array[end_idx_in:end_idx_out]
        p_out_len = end_idx_out - end_idx_in

        # Convert to tensors
        p_out_values_tensor = torch.from_numpy(p_out_values).float()
        p_out_mask_tensor = torch.from_numpy(p_out_mask).float()
        p_out_abs_time_tensor = torch.from_numpy(p_out_abs_time).float()
        p_out_rel_time_tensor = torch.from_numpy(p_out_rel_time).float()

        # NEW: Save prediction target tensor if requested
        if save_prediction_targets and prediction_target_dir is not None:
            if map_idx is not None and cvp_idx is not None:
                # Extract only MAP and CVP from p_out
                pred_values = torch.stack([
                    p_out_values_tensor[:, map_idx],  # MAP
                    p_out_values_tensor[:, cvp_idx]  # CVP
                ], dim=1)

                pred_mask = torch.stack([
                    p_out_mask_tensor[:, map_idx],  # MAP mask
                    p_out_mask_tensor[:, cvp_idx]  # CVP mask
                ], dim=1)

                # Save prediction target tensor
                pred_target_path = os.path.join(
                    prediction_target_dir,
                    f"prediction_target_{int(hadm_id)}_traj_{traj_num:03d}.pt"
                )
                torch.save(
                    (pred_values, pred_mask, p_out_abs_time_tensor, p_out_rel_time_tensor, p_out_len),
                    pred_target_path
                )

                # Store metadata
                prediction_target_info.append({
                    'hadm_id': hadm_id,
                    'trajectory_num': traj_num,
                    'length': p_out_len,
                    'has_map_data': torch.any(pred_mask[:, 0] > 0).item(),
                    'has_cvp_data': torch.any(pred_mask[:, 1] > 0).item(),
                    'file_path': pred_target_path
                })

        # Calculate trajectory metadata
        has_any_data_in = np.any(p_in_mask > 0)
        has_any_data_out = np.any(p_out_mask > 0)

        trajectory_info.append({
            'hadm_id': hadm_id,
            'trajectory_num': traj_num,
            'p_in_start_idx': start_idx_in,
            'p_in_end_idx': end_idx_in,
            'p_in_length': p_in_len,
            'p_out_start_idx': end_idx_in,
            'p_out_end_idx': end_idx_out,
            'p_out_length': p_out_len,
            'start_time_hours': 0.0,
            'end_time_hours': rel_time_array[end_idx_out - 1] if end_idx_out > 0 else 0,
            'has_physio_data_in': has_any_data_in,
            'has_physio_data_out': has_any_data_out,
            'file_path_in': file_path_in
        })

    # NEW: Return prediction target info if created
    if save_prediction_targets:
        return (hadm_id, trajectory_info, prediction_target_info)
    else:
        return (hadm_id, trajectory_info)

def save_patient_physio_as_csv_with_rtpr(
        hadm_id,
        chartevents_path,
        physio_params,
        co_itemids,  # Need CO itemids for r_tpr calculation
        n_intervals,
        interval_minutes,
        icu_admission_time,
        trajectory_boundaries,
        output_dir
):
    """
    Save patient physiological data including r_tpr calculations to CSV for inspection.
    """
    # Create output directory
    patient_dir = Path(output_dir) / f"patient_{hadm_id}_physio_inspection"
    patient_dir.mkdir(parents=True, exist_ok=True)

    # Read only this patient's data
    chart_df_patient = pl.read_parquet(chartevents_path).filter(
        pl.col('hadm_id') == hadm_id
    )

    # Initialize arrays for all physiological parameters
    n_params = len(physio_params)
    values_array = np.zeros((n_intervals, n_params), dtype=np.float32)
    mask_array = np.zeros((n_intervals, n_params), dtype=np.float32)
    count_array = np.zeros((n_intervals, n_params), dtype=np.int32)

    # Find indices for MAP, CVP, and R_TPR
    map_idx = None
    cvp_idx = None
    r_tpr_idx = None

    for idx, param in enumerate(physio_params):
        if param['param_type'] == 'MAP':
            map_idx = idx
        elif param['param_type'] == 'CVP':
            cvp_idx = idx
        elif param['param_type'] == 'R_TPR':
            r_tpr_idx = idx

    # Track CO values for r_tpr calculation
    co_values_array = np.zeros((n_intervals,), dtype=np.float32)
    co_mask_array = np.zeros((n_intervals,), dtype=np.float32)
    co_count_array = np.zeros((n_intervals,), dtype=np.int32)

    # Process each physiological parameter (except r_tpr)
    for idx, param_info in enumerate(physio_params):
        if param_info['param_type'] == 'R_TPR':
            continue

        itemid = param_info['itemid']
        param_events = chart_df_patient.filter(pl.col('itemid') == itemid)

        if param_events.height > 0:
            for row in param_events.iter_rows(named=True):
                time_idx = row['time_idx']
                if 0 <= time_idx < n_intervals:
                    if mask_array[time_idx, idx] == 0:
                        values_array[time_idx, idx] = row['value']
                        mask_array[time_idx, idx] = 1.0
                        count_array[time_idx, idx] = 1
                    else:
                        current_count = count_array[time_idx, idx]
                        current_sum = values_array[time_idx, idx] * current_count
                        new_count = current_count + 1
                        values_array[time_idx, idx] = (current_sum + row['value']) / new_count
                        count_array[time_idx, idx] = new_count

    # Process CO measurements
    for co_itemid in co_itemids:
        co_events = chart_df_patient.filter(pl.col('itemid') == co_itemid)

        if co_events.height > 0:
            for row in co_events.iter_rows(named=True):
                time_idx = row['time_idx']
                if 0 <= time_idx < n_intervals:
                    if co_mask_array[time_idx] == 0:
                        co_values_array[time_idx] = row['value']
                        co_mask_array[time_idx] = 1.0
                        co_count_array[time_idx] = 1  # ADD THIS!
                    else:
                        # FIXED: Use proper running average
                        current_count = co_count_array[time_idx]
                        current_sum = co_values_array[time_idx] * current_count
                        new_count = current_count + 1
                        co_values_array[time_idx] = (current_sum + row['value']) / new_count
                        co_count_array[time_idx] = new_count

    # Calculate r_tpr
    if r_tpr_idx is not None and map_idx is not None and cvp_idx is not None:
        for t in range(n_intervals):
            if (mask_array[t, map_idx] == 1 and
                    mask_array[t, cvp_idx] == 1 and
                    co_mask_array[t] == 1 and
                    co_values_array[t] > 0):

                map_value = values_array[t, map_idx]
                cvp_value = values_array[t, cvp_idx]
                co_value = co_values_array[t]

                values_array[t, r_tpr_idx] = (map_value - cvp_value) / co_value
                mask_array[t, r_tpr_idx] = 1.0
            else:
                values_array[t, r_tpr_idx] = 0.0
                mask_array[t, r_tpr_idx] = 0.0

    # Find actual data length
    has_data = np.any(mask_array > 0, axis=1)
    if np.any(has_data):
        actual_length = np.where(has_data)[0][-1] + 1
    else:
        actual_length = 0

    # Truncate arrays to actual length for CSV
    if actual_length > 0:
        values_to_save = values_array[:actual_length]
        mask_to_save = mask_array[:actual_length]
        co_values_to_save = co_values_array[:actual_length]
        co_mask_to_save = co_mask_array[:actual_length]
    else:
        values_to_save = values_array[:1]
        mask_to_save = mask_array[:1]
        co_values_to_save = co_values_array[:1]
        co_mask_to_save = co_mask_array[:1]

    # Create column names
    col_names = [param['param_name'] for param in physio_params]

    # Save values CSV
    values_df = pd.DataFrame(values_to_save, columns=col_names)
    values_df.insert(0, 'time_hours', np.arange(len(values_to_save)) * interval_minutes / 60)
    values_df.insert(1, 'time_minutes', np.arange(len(values_to_save)) * interval_minutes)
    values_df.to_csv(patient_dir / 'physio_values.csv', index=False)

    # Save mask CSV
    mask_df = pd.DataFrame(mask_to_save, columns=col_names)
    mask_df.insert(0, 'time_hours', np.arange(len(mask_to_save)) * interval_minutes / 60)
    mask_df.insert(1, 'time_minutes', np.arange(len(mask_to_save)) * interval_minutes)
    mask_df.to_csv(patient_dir / 'physio_mask.csv', index=False)

    # Save CO values separately for debugging
    co_df = pd.DataFrame({
        'time_hours': np.arange(len(co_values_to_save)) * interval_minutes / 60,
        'time_minutes': np.arange(len(co_values_to_save)) * interval_minutes,
        'co_value': co_values_to_save,
        'co_mask': co_mask_to_save
    })
    co_df.to_csv(patient_dir / 'co_values_debug.csv', index=False)

    # Save raw measurements for debugging
    if chart_df_patient.height > 0:
        raw_data = chart_df_patient.select([
            'itemid', 'value', 'charttime', 'time_idx', 'minutes_from_admission'
        ]).sort(['itemid', 'charttime'])
        raw_data.write_csv(patient_dir / 'raw_measurements.csv')

        # Create r_tpr calculation debug file
        r_tpr_debug = []
        if r_tpr_idx is not None and map_idx is not None and cvp_idx is not None:
            for t in range(min(actual_length, n_intervals)):
                r_tpr_debug.append({
                    'time_idx': t,
                    'time_hours': t * interval_minutes / 60,
                    'MAP': values_array[t, map_idx] if mask_array[t, map_idx] else np.nan,
                    'MAP_mask': mask_array[t, map_idx],
                    'CVP': values_array[t, cvp_idx] if mask_array[t, cvp_idx] else np.nan,
                    'CVP_mask': mask_array[t, cvp_idx],
                    'CO': co_values_array[t] if co_mask_array[t] else np.nan,
                    'CO_mask': co_mask_array[t],
                    'R_TPR': values_array[t, r_tpr_idx] if mask_array[t, r_tpr_idx] else np.nan,
                    'R_TPR_mask': mask_array[t, r_tpr_idx],
                    'calculated_r_tpr': ((values_array[t, map_idx] - values_array[t, cvp_idx]) / co_values_array[t]
                                         if (mask_array[t, map_idx] and mask_array[t, cvp_idx] and co_mask_array[t] and
                                             co_values_array[t] > 0)
                                         else np.nan)
                })

        if r_tpr_debug:
            r_tpr_debug_df = pd.DataFrame(r_tpr_debug)
            r_tpr_debug_df.to_csv(patient_dir / 'r_tpr_calculation_debug.csv', index=False)

    print(f"\nSaved physio inspection CSVs for patient {hadm_id} to: {patient_dir}")
    print(f"Files created:")
    print(f"  - physio_values.csv: Physiological measurements over time")
    print(f"  - physio_mask.csv: Measurement mask over time")
    print(f"  - co_values_debug.csv: CO values used for r_tpr calculation")
    print(f"  - r_tpr_calculation_debug.csv: Detailed r_tpr calculation debug info")
    print(f"  - raw_measurements.csv: Raw measurement data")

    return patient_dir


def create_physio_tensors(
        chartevents_path,
        hr_itemids,
        map_itemids,
        cvp_itemids,
        sv_itemids,
        co_itemids,
        trajectory_metadata_path,
        time_interval_minutes=5,
        icustays_path='../../data/input_data/icustays.csv',
        los_data_path='../../data/input_data/icustays.csv',
        cache_dir='../../data/processed_data/p_tensors',
        prediction_target_dir='../../data/processed_data/prediction_targets',  # NEW
        n_workers=4,
        max_co_age_minutes=10,
        co_guess=4.0,
        debug_patient_id=20214994  # NEW: Add debug patient parameter
):
    """
    Create physiological measurement tensors aligned with medication trajectories.
    Now also creates prediction target tensors (MAP and CVP) at the same time.
    """
    # Create cache directories
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    Path(prediction_target_dir).mkdir(parents=True, exist_ok=True)  # NEW

    # Load trajectory metadata from medication processing
    print("Loading medication trajectory metadata...")
    with open(trajectory_metadata_path, 'rb') as f:
        med_trajectory_data = pickle.load(f)

    all_med_trajectories = med_trajectory_data['all_trajectories']
    med_n_intervals = med_trajectory_data['n_intervals']

    # Load length of stay data
    print("Loading length of stay data...")
    los_df = pl.read_csv(los_data_path)
    max_los_days = los_df['los'].max()
    max_minutes = max_los_days * 24 * 60
    max_intervals = int(np.ceil(max_minutes / time_interval_minutes))

    print(f"Maximum intervals: {max_intervals} (based on max LOS of {max_los_days:.2f} days)")

    # Verify interval settings match
    if max_intervals != med_n_intervals:
        print(f"Warning: Physio max_intervals ({max_intervals}) differs from medication ({med_n_intervals})")
        n_intervals = max(max_intervals, med_n_intervals)
    else:
        n_intervals = max_intervals

    # Load ICU stays data
    print("Loading ICU stays data...")
    icustays_df = pl.read_csv(icustays_path)
    if icustays_df.schema['intime'] != pl.Datetime:
        icustays_df = icustays_df.with_columns(
            pl.col('intime').str.to_datetime()
        )

    # Create physiological parameter metadata
    physio_params = []

    # Add heart rate parameters
    for i, itemid in enumerate(hr_itemids):
        physio_params.append({
            'itemid': itemid,
            'param_type': 'HR',
            'param_name': f'HR_{i}' if len(hr_itemids) > 1 else 'HR'
        })

    # Add MAP parameters
    for i, itemid in enumerate(map_itemids):
        physio_params.append({
            'itemid': itemid,
            'param_type': 'MAP',
            'param_name': f'MAP_{i}' if len(map_itemids) > 1 else 'MAP'
        })

    # Add CVP parameters
    for i, itemid in enumerate(cvp_itemids):
        physio_params.append({
            'itemid': itemid,
            'param_type': 'CVP',
            'param_name': f'CVP_{i}' if len(cvp_itemids) > 1 else 'CVP'
        })

    # Add SV parameters
    for i, itemid in enumerate(sv_itemids):
        physio_params.append({
            'itemid': itemid,
            'param_type': 'SV',
            'param_name': f'SV_{i}' if len(sv_itemids) > 1 else 'SV'
        })

    # Add R_TPR as a calculated parameter (no itemid)
    physio_params.append({
        'itemid': None,  # This is calculated, not directly measured
        'param_type': 'R_TPR',
        'param_name': 'R_TPR'
    })

    # Get all item IDs (including CO for calculation, but not as separate column)
    all_itemids = hr_itemids + map_itemids + cvp_itemids + sv_itemids + co_itemids

    # Get stay IDs that have medication trajectories
    hadm_ids_with_trajectories = list(all_med_trajectories.keys())
    print(f"Processing {len(hadm_ids_with_trajectories)} patients with medication trajectories")

    # Filter chart events
    print("Filtering chart events...")
    chartevents_df = pl.scan_parquet(chartevents_path)

    # Ensure datetime parsing
    chart_df = chartevents_df.filter(
        pl.col('itemid').is_in(all_itemids) &
        pl.col('value').is_not_null() &
        pl.col('hadm_id').is_in(hadm_ids_with_trajectories)
    )

    chart_df = chart_df.collect()

    # Cast value to float, filtering out non-numeric values
    chart_df = chart_df.with_columns([
        pl.col('value').cast(pl.Float64, strict=False).alias('value')
    ]).filter(
        pl.col('value').is_not_null()
    )

    # Join with ICU admission times
    chart_df = chart_df.join(
        icustays_df.select(['hadm_id', 'intime']),
        on='hadm_id',
        how='inner'
    )

    # Parse charttime string to datetime
    chart_df = chart_df.with_columns([
        pl.col('charttime').str.to_datetime()
    ])

    # Calculate time indices
    chart_df = chart_df.with_columns([
        ((pl.col('charttime') - pl.col('intime')).dt.total_seconds() / 60).alias('minutes_from_admission'),
        ((pl.col('charttime') - pl.col('intime')).dt.total_seconds() / 60 / time_interval_minutes)
        .floor().cast(pl.Int32).alias('time_idx')
    ])

    # Filter out pre-admission measurements
    chart_df = chart_df.filter(pl.col('minutes_from_admission') >= 0)

    # Save prepared data to temporary parquet for worker processes
    temp_prepared_path = Path(cache_dir) / "temp_prepared_chartevents.parquet"
    chart_df.write_parquet(temp_prepared_path)
    print(f"Saved prepared chart events to {temp_prepared_path}")

    stay_admission_map = {}
    for row in icustays_df.select(['hadm_id', 'intime']).iter_rows(named=True):
        stay_admission_map[row['hadm_id']] = row['intime']

    # Process each patient
    all_physio_trajectory_info = {}
    all_prediction_targets = {}  # NEW: Track prediction targets

    # Update the process_func to include prediction target saving
    process_func = partial(
        process_single_patient_physio,
        chartevents_path=str(temp_prepared_path),
        physio_params=physio_params,
        co_itemids=co_itemids,
        n_intervals=n_intervals,
        interval_minutes=time_interval_minutes,
        cache_dir=cache_dir,
        save_prediction_targets=True,  # NEW
        prediction_target_dir=prediction_target_dir  # NEW
    )

    # In create_physio_tensors, for BOTH parallel and sequential processing:

    if n_workers > 1:
        # Parallel processing
        with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {}

            for hadm_id in hadm_ids_with_trajectories:
                # Get medication trajectory boundaries for this patient
                med_trajectories = all_med_trajectories.get(hadm_id, [])
                if not med_trajectories:
                    continue

                # Extract trajectory boundaries
                trajectory_boundaries = [(traj['start_idx'], traj['end_idx'])
                                         for traj in med_trajectories]

                admission_time = stay_admission_map.get(hadm_id)

                if admission_time is not None:
                    future = executor.submit(
                        process_func,
                        hadm_id=hadm_id,
                        trajectory_boundaries=trajectory_boundaries,
                        icu_admission_time=admission_time
                    )

                    futures[future] = hadm_id

            # Collect results
            for future in tqdm(concurrent.futures.as_completed(futures),
                               total=len(futures), desc="Processing patients"):
                hadm_id = futures[future]
                try:
                    result = future.result()
                    if len(result) == 3:
                        all_physio_trajectory_info[result[0]] = result[1]
                        all_prediction_targets[result[0]] = result[2]
                    else:
                        all_physio_trajectory_info[result[0]] = result[1]

                    # Save physio inspection CSV for debug patient
                    if int(hadm_id) == debug_patient_id:
                        # Need to retrieve the trajectory boundaries again
                        med_trajectories = all_med_trajectories.get(hadm_id, [])
                        trajectory_boundaries = [(traj['start_idx'], traj['end_idx'])
                                                 for traj in med_trajectories]
                        admission_time = stay_admission_map.get(hadm_id)

                        if admission_time is not None:
                            save_patient_physio_as_csv_with_rtpr(
                                hadm_id=hadm_id,
                                chartevents_path=str(temp_prepared_path),
                                physio_params=physio_params,
                                co_itemids=co_itemids,
                                n_intervals=n_intervals,
                                interval_minutes=time_interval_minutes,
                                icu_admission_time=admission_time,
                                trajectory_boundaries=trajectory_boundaries,
                                output_dir=cache_dir
                            )

                    # Save prediction target debug CSV for specific patient
                    if int(hadm_id) == debug_patient_id and hadm_id in all_prediction_targets:
                        save_prediction_target_debug_csv(
                            hadm_id,
                            all_prediction_targets[hadm_id],
                            all_med_trajectories.get(hadm_id, []),
                            prediction_target_dir
                        )
                except Exception as exc:
                    print(f'Hadm ID {hadm_id} generated an exception: {exc}')
                    import traceback
                    traceback.print_exc()

    else:
        # Sequential processing
        for hadm_id in tqdm(hadm_ids_with_trajectories, desc="Processing patients"):
            # Get medication trajectory boundaries
            med_trajectories = all_med_trajectories.get(hadm_id, [])
            if not med_trajectories:
                continue

            trajectory_boundaries = [(traj['start_idx'], traj['end_idx'])
                                     for traj in med_trajectories]

            admission_time = stay_admission_map.get(hadm_id)

            if admission_time is not None:
                result = process_func(
                    hadm_id=hadm_id,
                    trajectory_boundaries=trajectory_boundaries,
                    icu_admission_time=admission_time
                )
                if len(result) == 3:
                    all_physio_trajectory_info[result[0]] = result[1]
                    all_prediction_targets[result[0]] = result[2]
                else:
                    all_physio_trajectory_info[result[0]] = result[1]

                # Save physio inspection CSV for debug patient
                if int(hadm_id) == debug_patient_id:
                    save_patient_physio_as_csv_with_rtpr(
                        hadm_id=hadm_id,
                        chartevents_path=str(temp_prepared_path),
                        physio_params=physio_params,
                        co_itemids=co_itemids,
                        n_intervals=n_intervals,
                        interval_minutes=time_interval_minutes,
                        icu_admission_time=admission_time,
                        trajectory_boundaries=trajectory_boundaries,
                        output_dir=cache_dir
                    )

                # Save prediction target debug CSV for specific patient
                if int(hadm_id) == debug_patient_id and hadm_id in all_prediction_targets:
                    save_prediction_target_debug_csv(
                        hadm_id,
                        all_prediction_targets[hadm_id],
                        med_trajectories,
                        prediction_target_dir
                    )

    # After processing, add a note about what debug files were saved
    if debug_patient_id in all_physio_trajectory_info:
        print(
            f"\nPhysio inspection CSVs saved for patient {debug_patient_id} in {cache_dir}/patient_{debug_patient_id}_physio_inspection/")
    if debug_patient_id in all_prediction_targets:
        print(
            f"Prediction target CSVs saved for patient {debug_patient_id} in {prediction_target_dir}/patient_{debug_patient_id}_prediction_targets/")
    # Calculate summary statistics
    total_trajectories = sum(len(traj_list) for traj_list in all_physio_trajectory_info.values())
    total_prediction_targets = sum(len(traj_list) for traj_list in all_prediction_targets.values())  # NEW

    print(f"\nProcessing complete:")
    print(f"Total patients processed: {len(all_physio_trajectory_info)}")
    print(f"Total physio trajectories created: {total_trajectories}")
    print(f"Total prediction targets created: {total_prediction_targets}")  # NEW

    temp_prepared_path.unlink(missing_ok=True)

    # Save physiological trajectory metadata
    physio_metadata = {
        'all_trajectories': all_physio_trajectory_info,
        'physio_params': physio_params,
        'n_intervals': n_intervals,
        'interval_minutes': time_interval_minutes,
        'total_trajectories': total_trajectories,
        'aligned_with_med_trajectories': trajectory_metadata_path,
        'max_co_age_minutes': max_co_age_minutes,
        'co_guess': co_guess
    }

    metadata_file = Path(cache_dir) / "physio_trajectory_metadata.pkl"
    with open(metadata_file, "wb") as f:
        pickle.dump(physio_metadata, f)

    # NEW: Save prediction target metadata
    # Find MAP and CVP indices for metadata
    map_idx = None
    cvp_idx = None
    for idx, param in enumerate(physio_params):
        if param['param_type'] == 'MAP':
            map_idx = idx
        elif param['param_type'] == 'CVP':
            cvp_idx = idx

    prediction_metadata = {
        'all_prediction_targets': all_prediction_targets,
        'map_idx': map_idx,
        'cvp_idx': cvp_idx,
        'total_targets': total_prediction_targets,
        'source': 'created_with_physio_tensors'
    }

    pred_metadata_file = Path(prediction_target_dir) / "prediction_target_metadata.pkl"
    with open(pred_metadata_file, "wb") as f:
        pickle.dump(prediction_metadata, f)

    print(f"\nSaved physio trajectory metadata to {metadata_file}")
    print(f"Saved prediction target metadata to {pred_metadata_file}")
    print(f"Saved {total_trajectories} physio trajectory tensors to {cache_dir}")
    print(f"Saved {total_prediction_targets} prediction target tensors to {prediction_target_dir}")

    return all_physio_trajectory_info


def extract_trajectories_from_patient(
        values_array, mask_array, abs_time_array, rel_time_array, t0_trigger_idx,
        trajectory_before_minutes=None, trajectory_after_minutes=0, interval_minutes=5
):
    """
    Split patient data into trajectories based on t0 triggers.

    If trajectory_before_minutes is None (default), trajectories start from ICU admission.
    Otherwise, trajectories are windowed around t0 events.

    Parameters:
    - values_array: Patient values array
    - mask_array: Patient mask array
    - abs_time_array: Absolute time array
    - rel_time_array: Relative time array
    - t0_trigger_idx: Index of t0 trigger column
    - trajectory_before_minutes: Minutes before t0 to include (None = from ICU admission)
    - trajectory_after_minutes: Minutes after t0 to include
    - interval_minutes: Minutes per interval

    Returns:
        List of tuples: [(start_idx, end_idx), ...] where each tuple represents
        a trajectory from start_idx (inclusive) to end_idx (exclusive)
    """
    t0_triggers = values_array[:, t0_trigger_idx]
    t0_indices = np.where(t0_triggers == 1)[0]

    trajectories = []

    if len(t0_indices) == 0:
        # No t0 triggers - return entire sequence if there's any data
        if np.any(values_array != 0):
            last_nonzero = np.where(np.any(values_array != 0, axis=1))[0]
            if len(last_nonzero) > 0:
                trajectories.append((0, last_nonzero[-1] + 1))
    else:
        # Create trajectories around each t0
        for t0_idx in t0_indices:
            if trajectory_before_minutes is None:
                # Original behavior: start from ICU admission
                start_idx = 0
            else:
                # New behavior: windowed trajectory
                before_intervals = int(trajectory_before_minutes / interval_minutes)
                start_idx = max(0, t0_idx - before_intervals)

            # Calculate end index
            after_intervals = int(trajectory_after_minutes / interval_minutes)
            end_idx = min(len(values_array), t0_idx + 1 + after_intervals)

            trajectories.append((start_idx, end_idx))

    return trajectories


def save_trajectory_tensor(
        values_array, mask_array, abs_time_array, rel_time_array,
        start_idx, end_idx, hadm_id, traj_num, cache_dir
):
    """
    Save a single trajectory as a tensor file.
    """
    # Extract trajectory slice
    traj_values = values_array[start_idx:end_idx, :]
    traj_mask = mask_array[start_idx:end_idx, :]
    traj_abs_time = abs_time_array[start_idx:end_idx]
    traj_rel_time = rel_time_array[start_idx:end_idx]

    # Length is the number of time points in this trajectory
    length = end_idx - start_idx

    # Convert to tensors
    values_tensor = torch.from_numpy(traj_values).float()
    mask_tensor = torch.from_numpy(traj_mask).float()
    abs_time_tensor = torch.from_numpy(traj_abs_time).float()
    rel_time_tensor = torch.from_numpy(traj_rel_time).float()

    # Save tensor with matching trajectory number
    file_path = os.path.join(cache_dir, f"med_tensor_{int(hadm_id)}_traj_{traj_num:03d}.pt")
    torch.save(
        (values_tensor, mask_tensor, abs_time_tensor, rel_time_tensor, length),
        file_path
    )

    return file_path


def process_single_patient_medications(
        hadm_id,
        inputevents_path,  # Path to parquet file instead of DataFrame
        medication_info,
        n_intervals,
        interval_minutes,
        icu_admission_time,
        cache_dir,
        trajectory_before_minutes=None,
        trajectory_after_minutes=0
):
    """
    Process medications for a single patient and save as trajectory tensors.

    Parameters include trajectory window configuration.
    """
    # Read only this patient's data from parquet
    med_df_patient = pl.read_parquet(inputevents_path).filter(
        pl.col('hadm_id') == hadm_id
    )

    # Initialize arrays for all medications
    n_medications = len(medication_info) + 2  # +2 for crystalloid_sum and t0_trigger
    values_array = np.zeros((n_intervals, n_medications), dtype=np.float32)
    mask_array = np.zeros((n_intervals, n_medications), dtype=np.float32)

    # Create medication index mapping
    med_idx_map = {med['itemid']: idx for idx, med in enumerate(medication_info)}
    crystalloid_sum_idx = len(medication_info)
    t0_trigger_idx = len(medication_info) + 1

    # Track arrays for sum calculations
    crystalloid_arrays = {}
    vasopressor_arrays = {}

    # Process each medication
    for idx, med_info in enumerate(medication_info):
        itemid = med_info['itemid']
        rate_array = np.zeros(n_intervals)

        # Get events for this medication
        med_events = med_df_patient.filter(pl.col('itemid') == itemid)

        # Fill in rates for actual medication periods
        for row in med_events.iter_rows(named=True):
            start_idx = max(0, min(row['start_idx'], n_intervals - 1))
            end_idx = max(0, min(row['end_idx'], n_intervals))

            if start_idx < end_idx:
                rate_array[start_idx:end_idx] = row['rate']

        # Round crystalloid rates
        if med_info['medication_type'] == 'crystalloid':
            rate_array = np.round(rate_array).astype(np.float32)
            crystalloid_arrays[itemid] = rate_array
        else:  # vasopressor
            vasopressor_arrays[itemid] = rate_array

        values_array[:, idx] = rate_array
        # Mask is 1 where values are non-zero, 0 where values are zero
        mask_array[:, idx] = (rate_array != 0).astype(np.float32)

    # Calculate crystalloid sum
    crystalloid_sum = np.zeros(n_intervals, dtype=np.float32)
    for array in crystalloid_arrays.values():
        crystalloid_sum += array

    values_array[:, crystalloid_sum_idx] = crystalloid_sum
    # Mask is 1 where crystalloid sum is non-zero
    mask_array[:, crystalloid_sum_idx] = (crystalloid_sum != 0).astype(np.float32)

    # Calculate t0 triggers
    t0_array = np.zeros(n_intervals, dtype=np.float32)
    for i in range(1, n_intervals):
        trigger = 0

        # Check vasopressor increase
        for vaso_array in vasopressor_arrays.values():
            if vaso_array[i] > vaso_array[i - 1]:
                trigger = 1
                break

        # Check crystalloid condition
        if trigger == 0:
            crystalloid_change = abs(crystalloid_sum[i] - crystalloid_sum[i - 1])
            if crystalloid_change > 20 and crystalloid_sum[i] > 50:
                trigger = 1

        t0_array[i] = trigger

    values_array[:, t0_trigger_idx] = t0_array
    # t0 trigger is always considered measured (mask = 1 for all time points)
    mask_array[:, t0_trigger_idx] = 1.0

    # Calculate time tensors
    # Relative time: hours from ICU admission
    rel_time_array = np.arange(n_intervals) * interval_minutes / 60.0

    # For compatibility with the tensor format, we include abs_time_array
    # but set it equal to rel_time_array (both measure hours from ICU admission)
    abs_time_array = rel_time_array.copy()

    # Extract trajectories based on t0 triggers with window parameters
    trajectories = extract_trajectories_from_patient(
        values_array, mask_array, abs_time_array, rel_time_array, t0_trigger_idx,
        trajectory_before_minutes, trajectory_after_minutes, interval_minutes
    )

    # Save each trajectory
    trajectory_info = []
    for traj_num, (start_idx, end_idx) in enumerate(trajectories):
        file_path = save_trajectory_tensor(
            values_array, mask_array, abs_time_array, rel_time_array,
            start_idx, end_idx, hadm_id, traj_num, cache_dir
        )

        # Calculate trajectory metadata
        traj_length = end_idx - start_idx
        has_t0_at_end = (end_idx > 0 and
                         end_idx <= n_intervals and
                         values_array[end_idx - 1, t0_trigger_idx] == 1)

        trajectory_info.append({
            'hadm_id': hadm_id,
            'trajectory_num': traj_num,
            'start_idx': start_idx,
            'end_idx': end_idx,
            'length': traj_length,
            'start_time_hours': rel_time_array[start_idx] if start_idx < n_intervals else 0,
            'end_time_hours': rel_time_array[end_idx - 1] if end_idx > 0 else 0,
            'has_t0_trigger': has_t0_at_end,
            'file_path': file_path
        })

    return (hadm_id, trajectory_info)


def save_patient_data_as_csv(
        hadm_id,
        inputevents_path,  # Path to parquet file
        medication_info,
        n_intervals,
        interval_minutes,
        icu_admission_time,
        output_dir
):
    """
    Save all data for a single patient as CSV files for inspection.

    Creates the following CSV files:
    - values.csv: Medication rates over time
    - mask.csv: Mask values over time
    - time.csv: Time information
    - trajectories.csv: Trajectory metadata
    """
    import pandas as pd
    from pathlib import Path

    # Create output directory
    patient_dir = Path(output_dir) / f"patient_{hadm_id}_inspection"
    patient_dir.mkdir(parents=True, exist_ok=True)

    # Read only this patient's data (from prepared parquet that has start_idx/end_idx)
    try:
        med_df_patient = pl.read_parquet(inputevents_path).filter(
            pl.col('hadm_id') == hadm_id
        )

        print(f"Debug CSV save: Patient {hadm_id} data shape: {med_df_patient.shape}")
        print(f"Debug CSV save: Columns available: {med_df_patient.columns}")

        # Check if we have the required columns
        if med_df_patient.height == 0:
            print(f"No medication data found for patient {hadm_id}")
            # Create empty CSVs
            time_array = np.arange(n_intervals) * interval_minutes / 60.0

            # Create column names
            col_names = []
            for med_info in medication_info:
                col_names.append(f"{med_info['medication_name']}_{med_info['itemid']}")
            col_names.append('crystalloid_sum')
            col_names.append('t0_trigger')

            # Save empty data
            empty_values = np.zeros((1, len(col_names)))
            values_df = pd.DataFrame(empty_values, columns=col_names)
            values_df.insert(0, 'time_hours', [0.0])
            values_df.insert(1, 'time_minutes', [0.0])
            values_df.to_csv(patient_dir / 'values.csv', index=False)

            print(f"Saved empty CSV files for patient {hadm_id} (no medication data)")
            return patient_dir

        # Check for required columns
        required_cols = ['itemid', 'rate', 'start_idx', 'end_idx']
        missing_cols = [col for col in required_cols if col not in med_df_patient.columns]
        if missing_cols:
            print(f"Warning: Missing columns {missing_cols} for patient {hadm_id}")
            print(f"Cannot create detailed CSV files without time indices")
            return patient_dir

    except Exception as e:
        print(f"Error reading data for CSV save for patient {hadm_id}: {e}")
        return patient_dir

    # Verify required columns exist
    required_cols = ['itemid', 'rate', 'start_idx', 'end_idx']
    missing_cols = [col for col in required_cols if col not in med_df_patient.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}. Available columns: {med_df_patient.columns}")

    # Process patient data (same as process_single_patient_medications)
    # Initialize arrays
    n_medications = len(medication_info) + 2
    values_array = np.zeros((n_intervals, n_medications), dtype=np.float32)
    mask_array = np.zeros((n_intervals, n_medications), dtype=np.float32)

    # Create medication index mapping
    crystalloid_sum_idx = len(medication_info)
    t0_trigger_idx = len(medication_info) + 1

    # Track arrays for sum calculations
    crystalloid_arrays = {}
    vasopressor_arrays = {}

    # Process each medication
    for idx, med_info in enumerate(medication_info):
        itemid = med_info['itemid']
        rate_array = np.zeros(n_intervals)

        # Get events for this medication
        med_events = med_df_patient.filter(pl.col('itemid') == itemid)

        # Fill in rates
        for row in med_events.iter_rows(named=True):
            start_idx = max(0, min(row['start_idx'], n_intervals - 1))
            end_idx = max(0, min(row['end_idx'], n_intervals))

            if start_idx < end_idx:
                rate_array[start_idx:end_idx] = row['rate']

        # Round crystalloid rates
        if med_info['medication_type'] == 'crystalloid':
            rate_array = np.round(rate_array).astype(np.float32)
            crystalloid_arrays[itemid] = rate_array
        else:
            vasopressor_arrays[itemid] = rate_array

        values_array[:, idx] = rate_array
        mask_array[:, idx] = (rate_array != 0).astype(np.float32)

    # Calculate crystalloid sum
    crystalloid_sum = np.zeros(n_intervals, dtype=np.float32)
    for array in crystalloid_arrays.values():
        crystalloid_sum += array

    values_array[:, crystalloid_sum_idx] = crystalloid_sum
    mask_array[:, crystalloid_sum_idx] = (crystalloid_sum != 0).astype(np.float32)

    # Calculate t0 triggers
    t0_array = np.zeros(n_intervals, dtype=np.float32)
    for i in range(1, n_intervals):
        trigger = 0

        # Check vasopressor increase
        for vaso_array in vasopressor_arrays.values():
            if vaso_array[i] > vaso_array[i - 1]:
                trigger = 1
                break

        # Check crystalloid condition
        if trigger == 0:
            crystalloid_change = abs(crystalloid_sum[i] - crystalloid_sum[i - 1])
            if crystalloid_change > 20 and crystalloid_sum[i] > 50:
                trigger = 1

        t0_array[i] = trigger

    values_array[:, t0_trigger_idx] = t0_array
    mask_array[:, t0_trigger_idx] = 1.0

    # Calculate time arrays
    rel_time_array = np.arange(n_intervals) * interval_minutes / 60.0
    abs_time_array = rel_time_array.copy()

    # Find actual data length
    has_data = np.any(values_array > 0, axis=1)
    if np.any(has_data):
        actual_length = np.where(has_data)[0][-1] + 1
    else:
        actual_length = 0

    # Truncate arrays to actual length for CSV
    if actual_length > 0:
        values_to_save = values_array[:actual_length]
        mask_to_save = mask_array[:actual_length]
        time_to_save = rel_time_array[:actual_length]
    else:
        values_to_save = values_array[:1]  # At least one row
        mask_to_save = mask_array[:1]
        time_to_save = rel_time_array[:1]

    # Create column names
    col_names = []
    for med_info in medication_info:
        col_names.append(f"{med_info['medication_name']}_{med_info['itemid']}")
    col_names.append('crystalloid_sum')
    col_names.append('t0_trigger')

    # Save values CSV
    values_df = pd.DataFrame(values_to_save, columns=col_names)
    values_df.insert(0, 'time_hours', time_to_save)
    values_df.insert(1, 'time_minutes', time_to_save * 60)
    values_df.to_csv(patient_dir / 'values.csv', index=False)

    # Save mask CSV
    mask_df = pd.DataFrame(mask_to_save, columns=col_names)
    mask_df.insert(0, 'time_hours', time_to_save)
    mask_df.insert(1, 'time_minutes', time_to_save * 60)
    mask_df.to_csv(patient_dir / 'mask.csv', index=False)

    # Save time CSV
    time_df = pd.DataFrame({
        'interval_index': np.arange(len(time_to_save)),
        'time_hours': time_to_save,
        'time_minutes': time_to_save * 60,
        'icu_admission_time': icu_admission_time
    })
    time_df.to_csv(patient_dir / 'time.csv', index=False)

    # Extract and save trajectory information
    trajectories = extract_trajectories_from_patient(
        values_array, mask_array, abs_time_array, rel_time_array, t0_trigger_idx
    )

    traj_data = []
    for traj_num, (start_idx, end_idx) in enumerate(trajectories):
        has_t0_at_end = (end_idx > 0 and
                         end_idx <= n_intervals and
                         values_array[end_idx - 1, t0_trigger_idx] == 1)

        traj_data.append({
            'trajectory_num': traj_num,
            'start_idx': start_idx,
            'end_idx': end_idx,
            'length': end_idx - start_idx,
            'start_time_hours': 0.0,
            'end_time_hours': rel_time_array[end_idx - 1] if end_idx > 0 else 0,
            'has_t0_trigger': has_t0_at_end
        })

    traj_df = pd.DataFrame(traj_data)
    traj_df.to_csv(patient_dir / 'trajectories.csv', index=False)

    # Save summary information
    summary_data = {
        'hadm_id': [hadm_id],
        'icu_admission_time': [icu_admission_time],
        'total_intervals': [n_intervals],
        'interval_minutes': [interval_minutes],
        'actual_data_length': [actual_length],
        'num_trajectories': [len(trajectories)],
        'num_t0_triggers': [int(np.sum(t0_array))],
        'total_crystalloid_given': [np.sum(crystalloid_sum)],
        'num_medications': [len(medication_info)]
    }
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(patient_dir / 'summary.csv', index=False)

    print(f"\nSaved inspection CSVs for patient {hadm_id} to: {patient_dir}")
    print(f"Files created:")
    print(f"  - values.csv: Medication rates over time")
    print(f"  - mask.csv: Measurement mask over time")
    print(f"  - time.csv: Time information")
    print(f"  - trajectories.csv: Trajectory definitions")
    print(f"  - summary.csv: Patient summary statistics")

    return patient_dir


def create_medication_tensors(
        inputevents_path,
        crystalloid_itemids,
        vasopressor_itemids,
        time_interval_minutes=5,
        trajectory_before_minutes=None,
        trajectory_after_minutes=0,
        icustays_path='../../data/input_data/icustays.csv',
        los_data_path='../../data/input_data/icustays.csv',
        cache_dir='../../data/processed_data/med_tensors',
        n_workers=4,
        debug_patient_id=20214994):
    """
    Create medication trajectory tensors for each patient with configurable trajectory windows.
    """
    # Create cache directory
    Path(cache_dir).mkdir(parents=True, exist_ok=True)

    # Load length of stay data to determine max intervals
    print("Loading length of stay data...")
    los_df = pl.read_csv(los_data_path)

    # Verify 'los' column exists
    if 'los' not in los_df.columns:
        raise ValueError(f"Column 'los' not found in {los_data_path}. Available columns: {los_df.columns}")

    # Get maximum length of stay in days
    max_los_days = los_df['los'].max()
    print(f"Maximum length of stay: {max_los_days:.2f} days")

    # Convert to minutes and calculate required intervals
    max_minutes = max_los_days * 24 * 60  # Convert days to minutes
    max_intervals = int(np.ceil(max_minutes / time_interval_minutes))

    print(f"Maximum intervals needed: {max_intervals} "
          f"(based on max LOS of {max_los_days:.2f} days with {time_interval_minutes}-min intervals)")

    # Print trajectory window configuration
    if trajectory_before_minutes is None:
        print("Trajectory window: from ICU admission to t0")
    else:
        print(
            f"Trajectory window: {trajectory_before_minutes} minutes before t0 to {trajectory_after_minutes} minutes after t0")

    # Use the time interval parameter directly
    interval_minutes = time_interval_minutes
    n_intervals = max_intervals
    total_hours = (n_intervals * interval_minutes) / 60
    total_days = total_hours / 24

    print(f"Using {n_intervals} intervals of {interval_minutes} minutes each")
    print(f"Total time span: {total_hours:.1f} hours ({total_days:.1f} days)")

    # Load ICU stays data
    print("\nLoading ICU stays data...")
    icustays_df = pl.read_csv(icustays_path)

    # Parse datetime columns
    if icustays_df.schema['intime'] != pl.Datetime:
        icustays_df = icustays_df.with_columns(
            pl.col('intime').str.to_datetime()
        )

    # Get all unique hadm_ids efficiently using scan
    print("Reading unique stay IDs from input events...")
    unique_stays = pl.scan_parquet(inputevents_path).select('hadm_id').unique().collect()
    all_hadm_ids = unique_stays['hadm_id'].to_list()
    print(f"Total patients: {len(all_hadm_ids)}")

    # Create medication metadata
    medication_info = []
    all_meds = crystalloid_itemids + vasopressor_itemids

    for itemid in crystalloid_itemids:
        medication_info.append({
            'itemid': itemid,
            'medication_type': 'crystalloid',
            'medication_name': f'crystalloid_{itemid}'
        })

    for itemid in vasopressor_itemids:
        medication_info.append({
            'itemid': itemid,
            'medication_type': 'vasopressor',
            'medication_name': f'vasopressor_{itemid}'
        })

    # Prepare medication data with time indices
    print("Preparing medication events...")

    # Read and filter medication data
    med_df = pl.read_parquet(inputevents_path)

    # Ensure datetime columns are parsed
    if med_df.schema.get('starttime') != pl.Datetime:
        med_df = med_df.with_columns([
            pl.col('starttime').str.to_datetime(),
            pl.col('endtime').str.to_datetime()
        ])

    # Filter for relevant medications
    med_df = med_df.filter(
        pl.col('itemid').is_in(all_meds) &
        pl.col('rate').is_not_null() &
        pl.col('hadm_id').is_in(all_hadm_ids)
    )

    # Check for missing endtimes
    missing_endtimes = med_df.filter(pl.col('endtime').is_null()).height
    if missing_endtimes > 0:
        raise ValueError(f"Found {missing_endtimes} infusion events with missing endtimes.")

    # Join with ICU admission times
    med_df = med_df.join(
        icustays_df.select(['hadm_id', 'intime']),
        on='hadm_id',
        how='inner'
    )

    # Calculate time indices
    med_df = med_df.with_columns([
        ((pl.col('starttime') - pl.col('intime')).dt.total_seconds() / 60).alias('start_minutes'),
        ((pl.col('starttime') - pl.col('intime')).dt.total_seconds() / 60 / interval_minutes)
        .floor().cast(pl.Int32).alias('start_idx'),
        ((pl.col('endtime') - pl.col('intime')).dt.total_seconds() / 60 / interval_minutes)
        .ceil().cast(pl.Int32).alias('end_idx')
    ])

    # Filter out pre-admission events
    med_df = med_df.filter(pl.col('start_minutes') >= 0)

    # Save prepared data to temporary parquet for worker processes
    temp_prepared_path = Path(cache_dir) / "temp_prepared_inputevents.parquet"
    med_df.write_parquet(temp_prepared_path)
    print(f"Saved prepared medication events to {temp_prepared_path}")

    # Create hadm_id to admission time mapping
    stay_admission_map = {}
    for row in icustays_df.select(['hadm_id', 'intime']).iter_rows(named=True):
        stay_admission_map[row['hadm_id']] = row['intime']

    # Prepare for parallel processing
    process_func = partial(
        process_single_patient_medications,
        inputevents_path=str(temp_prepared_path),
        medication_info=medication_info,
        n_intervals=n_intervals,
        interval_minutes=interval_minutes,
        cache_dir=cache_dir,
        trajectory_before_minutes=trajectory_before_minutes,
        trajectory_after_minutes=trajectory_after_minutes
    )

    all_trajectory_info = {}

    # Process each patient
    if n_workers > 1:
        # Parallel processing
        with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {}

            for hadm_id in all_hadm_ids:
                admission_time = stay_admission_map.get(hadm_id)

                if admission_time is not None:
                    future = executor.submit(
                        process_func,
                        hadm_id=hadm_id,
                        icu_admission_time=admission_time
                    )
                    futures[future] = hadm_id

            # Collect results
            for future in tqdm(concurrent.futures.as_completed(futures),
                               total=len(futures), desc="Processing patients"):
                hadm_id = futures[future]
                try:
                    result = future.result()
                    all_trajectory_info[result[0]] = result[1]

                    # Save inspection CSVs for specific patient
                    if int(hadm_id) == debug_patient_id:
                        admission_time = stay_admission_map.get(hadm_id)
                        if admission_time is not None:
                            save_patient_data_as_csv(
                                hadm_id=hadm_id,
                                inputevents_path=str(temp_prepared_path),
                                medication_info=medication_info,
                                n_intervals=n_intervals,
                                interval_minutes=interval_minutes,
                                icu_admission_time=admission_time,
                                output_dir=cache_dir
                            )
                except Exception as exc:
                    print(f'Stay ID {hadm_id} generated an exception: {exc}')
    else:
        # Sequential processing
        for hadm_id in tqdm(all_hadm_ids, desc="Processing patients"):
            admission_time = stay_admission_map.get(hadm_id)

            if admission_time is not None:
                result = process_func(
                    hadm_id=hadm_id,
                    icu_admission_time=admission_time
                )
                all_trajectory_info[result[0]] = result[1]

                # Save inspection CSVs for specific patient
                if int(hadm_id) == debug_patient_id:
                    if admission_time is not None:
                        save_patient_data_as_csv(
                            hadm_id=hadm_id,
                            inputevents_path=str(temp_prepared_path),
                            medication_info=medication_info,
                            n_intervals=n_intervals,
                            interval_minutes=interval_minutes,
                            icu_admission_time=admission_time,
                            output_dir=cache_dir
                        )

    # Calculate summary statistics
    total_trajectories = sum(len(traj_list) for traj_list in all_trajectory_info.values())
    trajectories_per_patient = [len(traj_list) for traj_list in all_trajectory_info.values()]

    print(f"\nProcessing complete:")
    print(f"Total patients processed: {len(all_trajectory_info)}")
    print(f"Total trajectories created: {total_trajectories}")
    print(f"Average trajectories per patient: {np.mean(trajectories_per_patient):.2f}")
    print(f"Max trajectories for a patient: {np.max(trajectories_per_patient) if trajectories_per_patient else 0}")

    # Clean up temporary file
    temp_prepared_path.unlink(missing_ok=True)
    print(f"Cleaned up temporary file: {temp_prepared_path}")

    # Check if inspection patient was found
    inspection_patient_found = any(int(hadm_id) == debug_patient_id for hadm_id in all_trajectory_info.keys())
    if inspection_patient_found:
        print(
            f"\nInspection CSVs saved for patient {debug_patient_id} in {cache_dir}/patient_{debug_patient_id}_inspection/")
    else:
        print(f"\nNote: Patient {debug_patient_id} not found in the dataset")

    # Save trajectory metadata
    trajectory_metadata = {
        'all_trajectories': all_trajectory_info,
        'medication_info': medication_info,
        'n_intervals': n_intervals,
        'max_intervals': max_intervals,
        'interval_minutes': interval_minutes,
        'max_los_days': max_los_days,
        'trajectory_before_minutes': trajectory_before_minutes,
        'trajectory_after_minutes': trajectory_after_minutes,
        'total_trajectories': total_trajectories,
        'summary_stats': {
            'total_patients': len(all_trajectory_info),
            'total_trajectories': total_trajectories,
            'mean_trajectories_per_patient': np.mean(trajectories_per_patient) if trajectories_per_patient else 0,
            'max_trajectories_per_patient': np.max(trajectories_per_patient) if trajectories_per_patient else 0,
            'min_trajectories_per_patient': np.min(trajectories_per_patient) if trajectories_per_patient else 0
        }
    }

    metadata_file = Path(cache_dir) / "trajectory_metadata.pkl"
    with open(metadata_file, "wb") as f:
        pickle.dump(trajectory_metadata, f)

    print(f"\nSaved trajectory metadata to {metadata_file}")
    print(f"Saved {total_trajectories} trajectory tensors to {cache_dir}")

    return all_trajectory_info


def debug_time_aggregation(
        chartevents_path='../../data/treated_patients_chartevents.parquet',
        hadm_id=32128372,
        itemid=220052,  # MAP
        time_interval_minutes=5
):
    """Debug the time aggregation for a specific patient and measurement"""

    # Read the prepared data
    chart_df = pl.read_parquet(chartevents_path).filter(
        (pl.col('hadm_id') == hadm_id) &
        (pl.col('itemid') == itemid)
    ).sort('charttime')

    print(f"\nDebugging aggregation for patient {hadm_id}, item {itemid}")
    print(f"Total measurements: {chart_df.height}")

    if chart_df.height > 0:
        print("\nFirst 10 measurements:")
        print(chart_df.select(['charttime', 'value', 'time_idx', 'minutes_from_admission']).head(10))

        # Show aggregation by time_idx
        print("\nAggregation by time interval:")
        grouped = chart_df.group_by('time_idx').agg([
            pl.col('value').count().alias('count'),
            pl.col('value').alias('all_values'),
            pl.col('value').mean().alias('mean'),
            pl.col('value').min().alias('min'),
            pl.col('value').max().alias('max'),
            pl.col('minutes_from_admission').min().alias('interval_start'),
            pl.col('minutes_from_admission').max().alias('interval_end')
        ]).sort('time_idx')

        for row in grouped.head(10).iter_rows(named=True):
            print(f"\nTime idx {row['time_idx']} ({row['interval_start']:.1f}-{row['interval_end']:.1f} min):")
            print(f"  Values: {row['all_values']}")
            print(f"  Count: {row['count']}, Mean: {row['mean']:.2f}, Min: {row['min']}, Max: {row['max']}")


def extract_initial_conditions_for_patient(
        hadm_id,
        chartevents_path,  # Path to raw chartevents parquet
        inputevents_path,  # Path to raw inputevents parquet
        physio_params,
        medication_info,  # List of medication metadata
        co_itemids,
        trajectory_boundaries,
        interval_minutes,
        icu_admission_time,
        max_co_age_minutes=10,
        co_guess=4.0,
        cache_dir='../../data/initial_conditions'
):
    """
    Extract initial condition tensors for each trajectory of a patient.

    Initial conditions include both physiological values AND medication rates at t0.
    Uses linear interpolation from RAW data for maximum accuracy.

    Parameters:
    - hadm_id: Patient ID
    - chartevents_path: Path to raw chart events parquet
    - inputevents_path: Path to raw input events parquet
    - physio_params: List of physiological parameters
    - medication_info: List of medication metadata (itemid, type, name)
    - co_itemids: List of CO item IDs
    - trajectory_boundaries: List of (start_idx, end_idx) tuples
    - interval_minutes: Minutes per time interval
    - icu_admission_time: ICU admission datetime
    - max_co_age_minutes: Maximum age of CO measurement to use for R_TPR (default 10)
    - co_guess: Default CO value if no recent measurement (default 4.0)
    - cache_dir: Directory to save initial condition tensors

    Returns:
    - List of dictionaries with trajectory initial condition info
    """
    # Part 1: Process physiological data
    # Get all relevant itemids for physio measurements
    all_physio_itemids = []
    for param in physio_params:
        if param['itemid'] is not None:
            all_physio_itemids.append(param['itemid'])
    all_physio_itemids.extend(co_itemids)

    # Read raw patient chart data
    chart_df_patient = pl.scan_parquet(chartevents_path).filter(
        (pl.col('hadm_id') == hadm_id) &
        (pl.col('itemid').is_in(all_physio_itemids)) &
        (pl.col('value').is_not_null())
    ).collect()

    # Parse charttime if needed
    if chart_df_patient.height > 0:
        if chart_df_patient.schema.get('charttime') != pl.Datetime:
            chart_df_patient = chart_df_patient.with_columns(
                pl.col('charttime').str.to_datetime()
            )

        # Cast value to float
        chart_df_patient = chart_df_patient.with_columns(
            pl.col('value').cast(pl.Float64, strict=False)
        ).filter(pl.col('value').is_not_null())

        # Calculate minutes from admission
        chart_df_patient = chart_df_patient.with_columns(
            ((pl.col('charttime') - icu_admission_time).dt.total_seconds() / 60).alias('minutes_from_admission')
        ).filter(pl.col('minutes_from_admission') >= 0)

    # Part 2: Process medication data
    # Get all medication itemids
    all_med_itemids = [med['itemid'] for med in medication_info]

    # Read raw patient medication data
    med_df_patient = pl.scan_parquet(inputevents_path).filter(
        (pl.col('hadm_id') == hadm_id) &
        (pl.col('itemid').is_in(all_med_itemids)) &
        (pl.col('rate').is_not_null())
    ).collect()

    # Parse datetime columns if needed
    if med_df_patient.height > 0:
        if med_df_patient.schema.get('starttime') != pl.Datetime:
            med_df_patient = med_df_patient.with_columns([
                pl.col('starttime').str.to_datetime(),
                pl.col('endtime').str.to_datetime()
            ])

        # Calculate minutes from admission
        med_df_patient = med_df_patient.with_columns([
            ((pl.col('starttime') - icu_admission_time).dt.total_seconds() / 60).alias('start_minutes'),
            ((pl.col('endtime') - icu_admission_time).dt.total_seconds() / 60).alias('end_minutes')
        ]).filter(pl.col('start_minutes') >= 0)  # Only post-admission

    # Initialize storage for interpolators
    physio_interpolators = {}
    physio_raw_data = {}
    med_interpolators = {}
    med_raw_data = {}

    # Find indices for special parameters
    map_idx = None
    cvp_idx = None
    r_tpr_idx = None

    for idx, param in enumerate(physio_params):
        if param['param_type'] == 'MAP':
            map_idx = idx
        elif param['param_type'] == 'CVP':
            cvp_idx = idx
        elif param['param_type'] == 'R_TPR':
            r_tpr_idx = idx

    # Build interpolators for physiological parameters
    for idx, param_info in enumerate(physio_params):
        if param_info['param_type'] == 'R_TPR':
            continue  # Skip R_TPR as it's calculated

        itemid = param_info['itemid']
        param_events = chart_df_patient.filter(pl.col('itemid') == itemid).sort('minutes_from_admission')

        if param_events.height > 0:
            times = param_events['minutes_from_admission'].to_numpy()
            values = param_events['value'].to_numpy()

            physio_raw_data[idx] = {
                'times': times,
                'values': values,
                'charttimes': param_events['charttime'].to_list()
            }

            if len(times) >= 2:
                # Remove duplicates by averaging
                unique_times, unique_indices = np.unique(times, return_inverse=True)
                unique_values = np.zeros(len(unique_times))

                for i, t in enumerate(unique_times):
                    mask = unique_indices == i
                    unique_values[i] = np.mean(values[mask])

                physio_interpolators[idx] = interp1d(
                    unique_times, unique_values,
                    kind='linear',
                    bounds_error=False,
                    fill_value=(unique_values[0], unique_values[-1])
                )
            elif len(times) == 1:
                physio_interpolators[idx] = lambda x, v=values[0]: v

    # Build interpolators for CO (needed for R_TPR)
    co_data = {'times': [], 'values': []}

    for co_itemid in co_itemids:
        co_events = chart_df_patient.filter(pl.col('itemid') == co_itemid).sort('minutes_from_admission')

        if co_events.height > 0:
            co_data['times'].extend(co_events['minutes_from_admission'].to_list())
            co_data['values'].extend(co_events['value'].to_list())

    co_interpolator = None
    co_times = np.array([])

    if co_data['times']:
        sorted_indices = np.argsort(co_data['times'])
        co_times = np.array(co_data['times'])[sorted_indices]
        co_values = np.array(co_data['values'])[sorted_indices]

        if len(co_times) >= 2:
            unique_times, unique_indices = np.unique(co_times, return_inverse=True)
            unique_values = np.zeros(len(unique_times))

            for i, t in enumerate(unique_times):
                mask = unique_indices == i
                unique_values[i] = np.mean(co_values[mask])

            co_interpolator = interp1d(
                unique_times, unique_values,
                kind='linear',
                bounds_error=False,
                fill_value=(unique_values[0], unique_values[-1])
            )
        elif len(co_times) == 1:
            co_interpolator = lambda x, v=co_values[0]: v

    # Build interpolators for medications
    # For medications, we need step-wise interpolation since rates are constant during infusion
    for idx, med_info in enumerate(medication_info):
        itemid = med_info['itemid']
        med_events = med_df_patient.filter(pl.col('itemid') == itemid).sort('start_minutes')

        if med_events.height > 0:
            # Create time-rate pairs for step function
            time_points = []
            rate_values = []

            for row in med_events.iter_rows(named=True):
                start_min = row['start_minutes']
                end_min = row['end_minutes']
                rate = row['rate']

                # Add start and end points
                time_points.extend([start_min, end_min])
                rate_values.extend([rate, rate])

            # Add zero rate before first infusion and after last
            if time_points:
                # Add zero at time 0 if first infusion doesn't start at 0
                if time_points[0] > 0:
                    time_points.insert(0, 0)
                    rate_values.insert(0, 0)

                # Sort by time
                sorted_indices = np.argsort(time_points)
                sorted_times = np.array(time_points)[sorted_indices]
                sorted_rates = np.array(rate_values)[sorted_indices]

                # Create step interpolator (using 'previous' to maintain rate until next change)
                med_interpolators[idx] = interp1d(
                    sorted_times, sorted_rates,
                    kind='previous',
                    bounds_error=False,
                    fill_value=(0, 0)  # Zero rate outside infusion periods
                )

                med_raw_data[idx] = {
                    'times': sorted_times,
                    'rates': sorted_rates,
                    'events': med_events.to_dicts()
                }

    # Extract initial conditions for each trajectory
    initial_conditions = []

    for traj_num, (start_idx, end_idx) in enumerate(trajectory_boundaries):
        # t0 is at end_idx - 1 (last point of trajectory)
        t0_idx = end_idx - 1
        t0_minutes = t0_idx * interval_minutes

        # Initialize initial condition vector
        n_physio = len(physio_params)
        n_meds = len(medication_info) + 1  # +1 for crystalloid sum
        n_total = n_physio + n_meds
        ic_values = np.zeros(n_total, dtype=np.float32)
        ic_mask = np.zeros(n_total, dtype=np.float32)

        # Store debug info
        debug_info = {
            't0_minutes': t0_minutes,
            't0_datetime': icu_admission_time + timedelta(minutes=int(t0_minutes))
        }

        # Part 1: Interpolate physiological values
        for idx in range(n_physio):
            if idx == r_tpr_idx:
                continue  # Handle R_TPR separately

            if idx in physio_interpolators:
                ic_values[idx] = float(physio_interpolators[idx](t0_minutes))
                ic_mask[idx] = 1.0

                # Add interpolation details for debugging
                if idx in physio_raw_data:
                    times = physio_raw_data[idx]['times']
                    before_mask = times <= t0_minutes
                    after_mask = times >= t0_minutes

                    if np.any(before_mask):
                        nearest_before_idx = np.where(before_mask)[0][-1]
                        debug_info[f'physio_{idx}_before'] = {
                            'time': times[nearest_before_idx],
                            'value': physio_raw_data[idx]['values'][nearest_before_idx]
                        }

                    if np.any(after_mask):
                        nearest_after_idx = np.where(after_mask)[0][0]
                        debug_info[f'physio_{idx}_after'] = {
                            'time': times[nearest_after_idx],
                            'value': physio_raw_data[idx]['values'][nearest_after_idx]
                        }
            else:
                ic_values[idx] = 0.0
                ic_mask[idx] = 0.0

        # Calculate R_TPR
        if r_tpr_idx is not None and map_idx is not None and cvp_idx is not None:
            if ic_mask[map_idx] == 1 and ic_mask[cvp_idx] == 1:
                map_value = ic_values[map_idx]
                cvp_value = ic_values[cvp_idx]

                co_value = co_guess
                co_used = 'guess'
                co_age = None

                if co_interpolator is not None and len(co_times) > 0:
                    recent_co_mask = co_times <= t0_minutes

                    if np.any(recent_co_mask):
                        most_recent_co_idx = np.where(recent_co_mask)[0][-1]
                        most_recent_co_time = co_times[most_recent_co_idx]
                        co_age_minutes = t0_minutes - most_recent_co_time

                        if co_age_minutes <= max_co_age_minutes:
                            co_value = float(co_interpolator(t0_minutes))
                            co_used = 'interpolated'
                            co_age = co_age_minutes

                if co_value > 0:
                    ic_values[r_tpr_idx] = (map_value - cvp_value) / co_value
                    ic_mask[r_tpr_idx] = 1.0

                    debug_info['r_tpr_calculation'] = {
                        'map': map_value,
                        'cvp': cvp_value,
                        'co': co_value,
                        'co_source': co_used,
                        'co_age_minutes': co_age,
                        'r_tpr': ic_values[r_tpr_idx]
                    }
                else:
                    ic_values[r_tpr_idx] = 0.0
                    ic_mask[r_tpr_idx] = 0.0
            else:
                ic_values[r_tpr_idx] = 0.0
                ic_mask[r_tpr_idx] = 0.0

        # Part 2: Interpolate medication rates
        crystalloid_rates = []

        for idx, med_info in enumerate(medication_info):
            med_idx = n_physio + idx

            if idx in med_interpolators:
                rate = float(med_interpolators[idx](t0_minutes))
                ic_values[med_idx] = rate
                ic_mask[med_idx] = 1.0

                # Track crystalloid rates for sum
                if med_info['medication_type'] == 'crystalloid':
                    crystalloid_rates.append(rate)

                # Add medication state to debug info
                if idx in med_raw_data:
                    times = med_raw_data[idx]['times']
                    rates = med_raw_data[idx]['rates']

                    # Find active rate at t0
                    active_idx = np.searchsorted(times, t0_minutes, side='right') - 1
                    if 0 <= active_idx < len(rates):
                        debug_info[f'med_{med_info["medication_name"]}_rate'] = rates[active_idx]
                    else:
                        debug_info[f'med_{med_info["medication_name"]}_rate'] = 0
            else:
                ic_values[med_idx] = 0.0
                ic_mask[med_idx] = 1.0  # Medication not given is still valid info

        # Calculate crystalloid sum
        crystalloid_sum_idx = n_physio + len(medication_info)
        ic_values[crystalloid_sum_idx] = sum(crystalloid_rates)
        ic_mask[crystalloid_sum_idx] = 1.0

        # Convert to tensor and save
        ic_tensor = torch.from_numpy(ic_values).float()
        ic_mask_tensor = torch.from_numpy(ic_mask).float()

        # Save initial condition tensor
        ic_file_path = Path(cache_dir) / f"ic_tensor_{int(hadm_id)}_traj_{traj_num:03d}.pt"
        torch.save((ic_tensor, ic_mask_tensor), ic_file_path)

        initial_conditions.append({
            'hadm_id': hadm_id,
            'trajectory_num': traj_num,
            't0_time_minutes': t0_minutes,
            't0_time_hours': t0_minutes / 60.0,
            'file_path': str(ic_file_path),
            'has_complete_physio': bool(np.all(ic_mask[:n_physio] > 0)),
            'has_complete_data': bool(np.all(ic_mask > 0)),
            'n_physio': n_physio,
            'n_meds': n_meds,
            'debug_info': debug_info
        })

    return initial_conditions


def save_initial_conditions_debug_csv(
        hadm_id,
        initial_conditions_info,
        physio_params,
        medication_info,
        output_dir
):
    """
    Save initial conditions for a patient to CSV for debugging.
    Includes both physiological and medication values.
    """
    if not initial_conditions_info:
        print(f"No initial conditions found for patient {hadm_id}")
        return

    # Create output directory
    debug_dir = Path(output_dir) / f"patient_{hadm_id}_initial_conditions"
    debug_dir.mkdir(parents=True, exist_ok=True)

    # Collect data for all trajectories
    ic_data = []
    interpolation_details = []

    for ic_info in initial_conditions_info:
        # Load the tensor
        ic_tensor, ic_mask_tensor = torch.load(ic_info['file_path'])

        n_physio = ic_info['n_physio']
        n_meds = ic_info['n_meds']

        row_data = {
            'trajectory_num': ic_info['trajectory_num'],
            't0_time_hours': ic_info['t0_time_hours'],
            't0_time_minutes': ic_info['t0_time_minutes'],
            't0_datetime': ic_info['debug_info']['t0_datetime']
        }

        # Add physiological values and masks
        for idx, param in enumerate(physio_params):
            param_name = param['param_name']
            row_data[f'{param_name}_value'] = float(ic_tensor[idx])
            row_data[f'{param_name}_mask'] = float(ic_mask_tensor[idx])

        # Add medication values and masks
        for idx, med in enumerate(medication_info):
            med_name = med['medication_name']
            row_data[f'{med_name}_value'] = float(ic_tensor[n_physio + idx])
            row_data[f'{med_name}_mask'] = float(ic_mask_tensor[n_physio + idx])

        # Add crystalloid sum
        row_data['crystalloid_sum_value'] = float(ic_tensor[n_physio + len(medication_info)])
        row_data['crystalloid_sum_mask'] = float(ic_mask_tensor[n_physio + len(medication_info)])

        row_data['has_complete_physio'] = ic_info['has_complete_physio']
        row_data['has_complete_data'] = ic_info['has_complete_data']
        ic_data.append(row_data)

        # Collect interpolation details
        debug = ic_info['debug_info']
        interp_row = {
            'trajectory_num': ic_info['trajectory_num'],
            't0_time_minutes': debug['t0_minutes']
        }

        # Add physiological interpolation details
        for idx, param in enumerate(physio_params):
            if param['param_type'] == 'R_TPR':
                continue

            param_name = param['param_name']
            before_key = f'physio_{idx}_before'
            after_key = f'physio_{idx}_after'

            if before_key in debug:
                before = debug[before_key]
                interp_row[f'{param_name}_before_time'] = before['time']
                interp_row[f'{param_name}_before_value'] = before['value']
                interp_row[f'{param_name}_before_age_min'] = debug['t0_minutes'] - before['time']

            if after_key in debug:
                after = debug[after_key]
                interp_row[f'{param_name}_after_time'] = after['time']
                interp_row[f'{param_name}_after_value'] = after['value']
                interp_row[f'{param_name}_after_age_min'] = after['time'] - debug['t0_minutes']

        # Add R_TPR calculation details
        if 'r_tpr_calculation' in debug:
            r_tpr = debug['r_tpr_calculation']
            interp_row['R_TPR_MAP'] = r_tpr['map']
            interp_row['R_TPR_CVP'] = r_tpr['cvp']
            interp_row['R_TPR_CO'] = r_tpr['co']
            interp_row['R_TPR_CO_source'] = r_tpr['co_source']
            interp_row['R_TPR_CO_age_min'] = r_tpr['co_age_minutes']
            interp_row['R_TPR_calculated'] = r_tpr['r_tpr']

        # Add medication rates from debug info
        for med in medication_info:
            med_key = f'med_{med["medication_name"]}_rate'
            if med_key in debug:
                interp_row[f'{med["medication_name"]}_rate_at_t0'] = debug[med_key]

        interpolation_details.append(interp_row)

    # Create DataFrames and save
    ic_df = pd.DataFrame(ic_data)
    ic_df.to_csv(debug_dir / 'initial_conditions.csv', index=False)

    interp_df = pd.DataFrame(interpolation_details)
    interp_df.to_csv(debug_dir / 'interpolation_details.csv', index=False)

    # Save combined parameter metadata
    all_params = []

    # Add physiological parameters
    for idx, param in enumerate(physio_params):
        all_params.append({
            'index': idx,
            'category': 'physiological',
            'itemid': param.get('itemid'),
            'param_type': param['param_type'],
            'param_name': param['param_name']
        })

    # Add medication parameters
    for idx, med in enumerate(medication_info):
        all_params.append({
            'index': len(physio_params) + idx,
            'category': 'medication',
            'itemid': med['itemid'],
            'param_type': med['medication_type'],
            'param_name': med['medication_name']
        })

    # Add crystalloid sum
    all_params.append({
        'index': len(physio_params) + len(medication_info),
        'category': 'medication',
        'itemid': -1,
        'param_type': 'crystalloid_sum',
        'param_name': 'crystalloid_sum'
    })

    param_df = pd.DataFrame(all_params)
    param_df.to_csv(debug_dir / 'parameter_info.csv', index=False)

    print(f"Saved initial conditions debug CSV for patient {hadm_id} to {debug_dir}")
    print(f"Files created:")
    print(f"  - initial_conditions.csv: IC values and masks for each trajectory")
    print(f"  - interpolation_details.csv: Details about interpolation from raw data")
    print(f"  - parameter_info.csv: Complete parameter metadata (physio + meds)")

    return debug_dir


def create_initial_condition_tensors(
        chartevents_path,
        inputevents_path,
        trajectory_metadata_path,
        physio_params,
        medication_info,
        co_itemids,
        icustays_path='../../data/input_data/icustays.csv',
        max_co_age_minutes=10,
        co_guess=4.0,
        cache_dir='../../data/processed_data/initial_conditions',
        n_workers=4,
        debug_patient_id=20214994):
    """
    Create initial condition tensors for all patients with trajectories.
    Now accepts max_co_age_minutes, co_guess, and debug_patient_id as parameters.
    """
    # Create cache directory
    Path(cache_dir).mkdir(parents=True, exist_ok=True)

    # Load trajectory metadata
    print("Loading trajectory metadata...")
    with open(trajectory_metadata_path, 'rb') as f:
        trajectory_data = pickle.load(f)

    all_trajectories = trajectory_data['all_trajectories']
    interval_minutes = trajectory_data['interval_minutes']

    # Load ICU stays to get admission times
    print("Loading ICU admission times...")
    icustays_df = pl.read_csv(icustays_path)
    if icustays_df.schema['intime'] != pl.Datetime:
        icustays_df = icustays_df.with_columns(
            pl.col('intime').str.to_datetime()
        )

    # Create hadm_id to admission time mapping
    stay_admission_map = {}
    for row in icustays_df.select(['hadm_id', 'intime']).iter_rows(named=True):
        stay_admission_map[row['hadm_id']] = row['intime']

    print(f"Processing initial conditions for {len(all_trajectories)} patients")
    print(f"Initial condition will include:")
    print(f"  - {len(physio_params)} physiological parameters (interpolated from chartevents)")
    print(f"  - {len(medication_info)} medications (interpolated from inputevents)")
    print(f"  - 1 crystalloid sum")
    print(f"  Total: {len(physio_params) + len(medication_info) + 1} features")

    # Prepare processing function
    process_func = partial(
        extract_initial_conditions_for_patient,
        chartevents_path=chartevents_path,
        inputevents_path=inputevents_path,
        physio_params=physio_params,
        medication_info=medication_info,
        co_itemids=co_itemids,
        interval_minutes=interval_minutes,
        max_co_age_minutes=max_co_age_minutes,
        co_guess=co_guess,
        cache_dir=cache_dir
    )

    all_initial_conditions = {}

    if n_workers > 1:
        # Parallel processing
        with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {}

            for hadm_id, trajectories in all_trajectories.items():
                if hadm_id not in stay_admission_map:
                    print(f"Warning: No admission time found for hadm_id {hadm_id}")
                    continue

                # Extract trajectory boundaries
                boundaries = [(traj['start_idx'], traj['end_idx'])
                              for traj in trajectories]

                future = executor.submit(
                    process_func,
                    hadm_id=hadm_id,
                    trajectory_boundaries=boundaries,
                    icu_admission_time=stay_admission_map[hadm_id]
                )
                futures[future] = hadm_id

            # Collect results
            for future in tqdm(concurrent.futures.as_completed(futures),
                               total=len(futures), desc="Processing initial conditions"):
                hadm_id = futures[future]
                try:
                    result = future.result()
                    if result:
                        all_initial_conditions[hadm_id] = result

                        # Save debug CSV for specific patient
                        if int(hadm_id) == debug_patient_id:
                            save_initial_conditions_debug_csv(
                                hadm_id,
                                result,
                                physio_params,
                                medication_info,
                                cache_dir
                            )
                except Exception as exc:
                    print(f'Stay ID {hadm_id} generated an exception: {exc}')
                    import traceback
                    traceback.print_exc()
    else:
        # Sequential processing
        for hadm_id, trajectories in tqdm(all_trajectories.items(),
                                          desc="Processing initial conditions"):
            if hadm_id not in stay_admission_map:
                print(f"Warning: No admission time found for hadm_id {hadm_id}")
                continue

            boundaries = [(traj['start_idx'], traj['end_idx'])
                          for traj in trajectories]

            result = process_func(
                hadm_id=hadm_id,
                trajectory_boundaries=boundaries,
                icu_admission_time=stay_admission_map[hadm_id]
            )

            if result:
                all_initial_conditions[hadm_id] = result

                # Save debug CSV for specific patient
                if int(hadm_id) == debug_patient_id:
                    save_initial_conditions_debug_csv(
                        hadm_id,
                        result,
                        physio_params,
                        medication_info,
                        cache_dir
                    )

    # Calculate statistics
    total_ics = sum(len(ics) for ics in all_initial_conditions.values())
    complete_physio = sum(
        sum(1 for ic in ics if ic['has_complete_physio'])
        for ics in all_initial_conditions.values()
    )
    complete_all = sum(
        sum(1 for ic in ics if ic['has_complete_data'])
        for ics in all_initial_conditions.values()
    )

    print(f"\nInitial condition extraction complete:")
    print(f"Total patients processed: {len(all_initial_conditions)}")
    print(f"Total initial conditions created: {total_ics}")
    print(f"ICs with complete physiological data: {complete_physio} ({complete_physio / total_ics * 100:.1f}%)")
    print(f"ICs with complete data (physio + meds): {complete_all} ({complete_all / total_ics * 100:.1f}%)")

    # Save metadata
    ic_metadata = {
        'all_initial_conditions': all_initial_conditions,
        'physio_params': physio_params,
        'medication_info': medication_info,
        'max_co_age_minutes': max_co_age_minutes,
        'co_guess': co_guess,
        'total_ics': total_ics,
        'complete_physio': complete_physio,
        'complete_all': complete_all,
        'interval_minutes': interval_minutes,
        'n_physio_features': len(physio_params),
        'n_med_features': len(medication_info) + 1  # +1 for crystalloid sum
    }

    metadata_file = Path(cache_dir) / "initial_conditions_metadata.pkl"
    with open(metadata_file, "wb") as f:
        pickle.dump(ic_metadata, f)

    print(f"\nSaved initial conditions metadata to {metadata_file}")
    print(f"Saved {total_ics} initial condition tensors to {cache_dir}")

    # Check if debug patient was found
    if debug_patient_id in all_initial_conditions:
        print(
            f"\nDebug CSV saved for patient {debug_patient_id} in {cache_dir}/patient_{debug_patient_id}_initial_conditions/")

    return all_initial_conditions

def save_prediction_target_debug_csv(hadm_id, patient_targets, trajectories, output_dir):
    """
    Save debug CSV files for a specific patient's prediction targets.
    """
    debug_dir = Path(output_dir) / f"patient_{hadm_id}_prediction_targets"
    debug_dir.mkdir(parents=True, exist_ok=True)

    # Save summary of all prediction windows
    summary_data = []

    for target_info in patient_targets:
        traj_num = target_info['trajectory_num']

        # Load the tensor to get more details
        pred_values, pred_mask, _, _, pred_len = torch.load(target_info['file_path'])

        summary_data.append({
            'trajectory_num': traj_num,
            'length': pred_len,
            'has_map_data': target_info['has_map_data'],
            'has_cvp_data': target_info['has_cvp_data'],
            'map_measurements': torch.sum(pred_mask[:, 0]).item(),
            'cvp_measurements': torch.sum(pred_mask[:, 1]).item(),
            'map_mean': torch.mean(pred_values[pred_mask[:, 0] > 0, 0]).item() if target_info['has_map_data'] else None,
            'cvp_mean': torch.mean(pred_values[pred_mask[:, 1] > 0, 1]).item() if target_info['has_cvp_data'] else None
        })

        # Save individual prediction window
        window_df = pd.DataFrame({
            'time_idx': range(pred_len),
            'MAP_value': pred_values[:, 0].numpy(),
            'MAP_mask': pred_mask[:, 0].numpy(),
            'CVP_value': pred_values[:, 1].numpy(),
            'CVP_mask': pred_mask[:, 1].numpy()
        })
        window_df.to_csv(debug_dir / f'prediction_window_traj_{traj_num:03d}.csv', index=False)

    # Save summary
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(debug_dir / 'prediction_windows_summary.csv', index=False)

    print(f"Saved debug CSV files for patient {hadm_id} to {debug_dir}")

def main():
    # Heart Rate: 220045
    # MAP: 220052
    # CVP: 220074
    # CO-related params: 220088, 224842, 228369, 229897
    # SV: 228375

    # Things that act on / increase I_control

        #Crystalloids
        #225158, NaCl 0.9 %, NaCl 0.9 %, inputevents, Fluids / Intake, mL, Solution,,
        #225159, NaCl 0.45 %, NaCl 0.45 %, inputevents, Fluids / Intake, mL, Solution,,
        #225161, NaCl 3 % (Hypertonic Saline), NaCl 3 % (Hypertonic Saline), inputevents, Fluids / Intake, mL, Solution,,

    # Ways to increase R_TPR

        # Vasopressors
        # 221906, Norepinephrine, Norepinephrine, inputevents, Medications, mg, Solution,,

        # 229630, Phenylephrine(50 / 250), Phenylephrine(50 / 250), inputevents, Medications, mg, Solution,,
        # 229631, Phenylephrine(200 / 250)_OLD_1, Phenylephrine(200 / 250)_OLD_1, inputevents, Medications, mg, Solution,,
        # 229632, Phenylephrine(200 / 250), Phenylephrine(200 / 250), inputevents, Medications, mg, Solution,,

        # 221662, Dopamine, Dopamine, inputevents, Medications, mg, Solution,,

    # Things that increase cardiac contractility - (and therefore SV)

        # 221653, Dobutamine, Dobutamine, inputevents, Medications, mg, Solution,,

        # 221986, Milrinone, Milrinone, inputevents, Medications, mg, Solution,,
    # Parse command line arguments
    args = parse_args()

    # Create output directories
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    input_dir = Path(args.input_dir)
    input_dir.mkdir(parents=True, exist_ok=True)

    # Define paths based on output directory
    relevant_patients_chartevents_path = output_dir / "treated_patients_chartevents.parquet"
    patient_metadata_path = input_dir / "patients.csv"
    hr_params_path = output_dir / "hr_params.csv"
    relevant_patients_inputevents_path = output_dir / "treated_patients_inputevents.parquet"
    icustays_path = input_dir / "icustays.csv"

    # Define the lists (these should match what was used in process_mimic_data)
    crystalloids_list = [225158, 225159, 225161]  # NaCl 0.9%, 0.45%, 3%
    vasopressors_list = [221906, 229630, 229631, 229632, 221662]
    SV_list = [228375]

    print(f"Configuration:")
    print(f"  Interval: {args.interval_minutes} minutes")
    print(f"  Trajectory window: ", end="")
    if args.trajectory_before_minutes is None:
        print("from ICU admission to t0")
    else:
        print(f"{args.trajectory_before_minutes} min before to {args.trajectory_after_minutes} min after t0")
    print(f"  Max CO age: {args.max_co_age_minutes} minutes")
    print(f"  CO guess: {args.co_guess}")
    print(f"  Workers: {args.n_workers}")
    print(f"  Debug patient: {args.debug_patient_id}")
    print(f"  Data limit: {args.data_limit} rows")
    print(f"  Output directory: {args.output_dir}\n")

    all_patients_chartevents = find_relevant_patients(measurements=[220045, 220052, 220074, 220088, 224842, 228369, 229897, 228375],
                                                      MAP_id=220052,
                                                      save_path=str(relevant_patients_chartevents_path),
                                                      data_limit=args.data_limit)

    hr_params = find_min_max_heartrates(
        str(relevant_patients_chartevents_path),
        metadata_path=str(patient_metadata_path),
        save_path=str(hr_params_path)
    )

    inputevents = find_relevant_inputevents(
        all_patients_chartevents=all_patients_chartevents,
        save_path=str(relevant_patients_inputevents_path),
        events=[225158, 225159, 225161, 221906, 229630, 229631, 229632, 221662, 221653, 221986],
        data_limit=args.data_limit
    )

    # Create medication tensors with CLI parameters
    meds_matrix = create_medication_tensors(
        inputevents_path=str(relevant_patients_inputevents_path),
        crystalloid_itemids=crystalloids_list,
        vasopressor_itemids=vasopressors_list,
        time_interval_minutes=args.interval_minutes,
        trajectory_before_minutes=args.trajectory_before_minutes,
        trajectory_after_minutes=args.trajectory_after_minutes,
        cache_dir=str(output_dir / "med_tensors"),
        n_workers=args.n_workers,
        debug_patient_id=args.debug_patient_id
    )

    physio_matrix = create_physio_tensors(
        chartevents_path=str(relevant_patients_chartevents_path),
        hr_itemids=[220045],
        map_itemids=[220052],
        cvp_itemids=[220074],
        co_itemids=[220088, 224842, 228369, 229897],
        sv_itemids=[228375],
        trajectory_metadata_path=str(output_dir / "med_tensors" / "trajectory_metadata.pkl"),
        time_interval_minutes=args.interval_minutes,
        cache_dir=str(output_dir / "p_tensors"),
        prediction_target_dir=str(output_dir / "prediction_targets"),  # NEW: Add this
        n_workers=args.n_workers,
        max_co_age_minutes=args.max_co_age_minutes,
        co_guess=args.co_guess,
        debug_patient_id=args.debug_patient_id  # NEW: Add this
    )

    #summarize_medication_matrix(meds_matrix)


    #results_df = pd.read_csv('../../data/mimics_stays.csv')



    # First, analyze data completeness
    #analyze_data_completeness(results_df, crystalloids_list, vasopressors_list, SV_list)

    # Detect t0 points and create training data
    #training_data = detect_t0(results_df, crystalloids_list, vasopressors_list)
    df = debug_specific_patient()
    with open(output_dir / "med_tensors" / "trajectory_metadata.pkl", 'rb') as f:
        med_metadata = pickle.load(f)
    medication_info = med_metadata['medication_info']

    physio_params = [
        {'itemid': 220045, 'param_type': 'HR', 'param_name': 'HR'},
        {'itemid': 220052, 'param_type': 'MAP', 'param_name': 'MAP'},
        {'itemid': 220074, 'param_type': 'CVP', 'param_name': 'CVP'},
        {'itemid': 228375, 'param_type': 'SV', 'param_name': 'SV'},
        {'itemid': None, 'param_type': 'R_TPR', 'param_name': 'R_TPR'}
    ]

    # Create initial conditions with CLI parameters
    initial_conditions = create_initial_condition_tensors(
        chartevents_path=str(relevant_patients_chartevents_path),
        inputevents_path=str(relevant_patients_inputevents_path),
        trajectory_metadata_path=str(output_dir / "med_tensors" / "trajectory_metadata.pkl"),
        physio_params=physio_params,
        medication_info=medication_info,
        co_itemids=[220088, 224842, 228369, 229897],
        icustays_path=str(icustays_path),
        max_co_age_minutes=args.max_co_age_minutes,
        co_guess=args.co_guess,
        cache_dir=str(output_dir / "initial_conditions"),
        n_workers=args.n_workers,
        debug_patient_id=args.debug_patient_id
    )

    print("\nProcessing complete!")


if __name__ == "__main__":
    main()






