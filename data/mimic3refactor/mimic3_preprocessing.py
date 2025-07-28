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
    """Parse command line arguments for MIMIC-III."""
    parser = argparse.ArgumentParser(description='Process MIMIC-III data for ICU patient trajectories')

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
    parser.add_argument('--debug-patient-id', type=int, default=None,
                        help='Patient ID to save debug CSV files for (set automatically if not provided)')

    # Data limit parameter (for testing)
    parser.add_argument('--data-limit', type=int, default=None,
                        help='Limit on number of rows to read from CSV files (default: None for unlimited)')

    # Output directories
    parser.add_argument('--output-dir', type=str, default='../../data/processed_data',
                        help='Base output directory (default: ../../data/processed_data)')
    parser.add_argument('--input-dir', type=str, default='../../data/mimic3',
                        help='Base input directory (default: ../../data/mimic3)')

    # MIMIC-III specific parameters
    parser.add_argument('--hadm-filter-file', type=str, default='results_with_hadm_id.csv',
                        help='CSV file containing hadm_id column for filtering (default: results_with_hadm_id.csv)')

    return parser.parse_args()


def load_hadm_id_filter(filter_file_path):
    """
    Load hadm_ids from the specified CSV file for filtering.

    Args:
        filter_file_path: Path to CSV file containing hadm_id column

    Returns:
        List of hadm_ids to include in processing
    """
    try:
        filter_df = pd.read_csv(filter_file_path)
        if 'hadm_id' not in filter_df.columns:
            raise ValueError(
                f"Column 'hadm_id' not found in {filter_file_path}. Available columns: {filter_df.columns.tolist()}")

        hadm_ids = filter_df['hadm_id'].dropna().unique().tolist()
        print(f"Loaded {len(hadm_ids)} unique hadm_ids from {filter_file_path}")
        return hadm_ids

    except Exception as e:
        print(f"Error loading hadm_id filter file: {e}")
        raise


def get_mimic3_item_ids():
    """
    Get MIMIC-III item IDs for medications and physiological measurements.

    Returns:
        Dictionary containing item ID mappings for MIMIC-III
    """
    return {
        # Physiological measurements (CHARTEVENTS)
        'hr': [211, 220045],  # Heart Rate
        'map': [52, 6702, 443, 224, 52, 6702, 443, 224],  # Mean Arterial Pressure
        'cvp': [113],  # Central Venous Pressure
        'sv': [198],  # Stroke Volume (if available)
        'co': [198, 228],  # Cardiac Output related

        # Medications (INPUTEVENTS_MV and INPUTEVENTS_CV)
        'crystalloids': [
            30018,  # NaCl 0.9%
            30020,  # NaCl 0.45%
            30021, # Ringers Lactate
            30143,  # NaCl 3%
            30161, # .3% normal saline
            30159,  # D5 Ringers Lactate
            30160,  # D5 normal saline
        ],
        'vasopressors': [
            221906,  # Norepinephrine (Levophed)
            222315, #Vasopressin
            1136, # Vasopressin
            1222, #Vasopressin
            1327, # Vasopressin unit/ml
            2248, # Vasopressin unit/ml
            2234, # Vasopressin unit/hr
            2445, # Vasopressin
            2561, # Vasopressin
            2765, #Vasopressin
            6255, #Vasopressin
            7341, # Vasopressin
            30051, # Vasopressin
            42273, # Vasopressin
            42802, # Vasopressin

            30306, # Dopamine
            30307,  # Dopamine Drip
            221662, # Dopamine drip
            4501, # Dopamine drip
            5805, # Dopamine
            5329, # Dopamine
            30043, # Dopamine
            30307 # Dopamine
        ]
    }


def find_relevant_patients_mimic3(measurements, MAP_id, hadm_id_filter,
                                  load_path_events="../../data/mimic3/CHARTEVENTS.csv",
                                  load_path_stays="../../data/mimic3/ICUSTAYS.csv",
                                  save_path="../../data/processed_data/treated_patients_chartevents.parquet",
                                  data_limit=None):
    """
    Finds relevant patients for MIMIC-III by filtering on:
    1. Patients in the provided hadm_id filter list
    2. Patients with blood pressure events
    3. Patients with ICU stays over 24h

    Args:
        measurements: List of measurement IDs to extract
        MAP_id: Measurement ID for mean arterial pressure
        hadm_id_filter: List of hadm_ids to include
        load_path_events: path to CHARTEVENTS.csv
        load_path_stays: path to ICUSTAYS.csv
        save_path: path to save relevant patients
        data_limit: Maximum number of rows to read from CSV

    Returns:
        dataset containing all occurrences of measurements for filtered patients
    """
    if not os.path.exists(save_path):
        print("Loading ICU stays data...")
        long_stays_query = pl.scan_csv(load_path_stays)
        if data_limit is not None:
            long_stays_query = long_stays_query.limit(data_limit)

        # Filter by length of stay AND hadm_id filter
        long_stays = (long_stays_query
                      .filter(pl.col("LOS") > 1)  # MIMIC-III uses "LOS" not "los"
                      .filter(pl.col("HADM_ID").is_in(hadm_id_filter))  # Apply hadm_id filter
                      .collect())

        valid_hadm_ids = long_stays["HADM_ID"].unique().to_list()
        print(f"Found {len(valid_hadm_ids)} patients with LOS > 1 day in hadm_id filter")

        print("Loading chart events data...")
        treated_patients_query = pl.scan_csv(load_path_events)
        if data_limit is not None:
            treated_patients_query = treated_patients_query.limit(data_limit)

        # Find patients with hypotensive episodes (MAP < 70)
        treated_patients = (treated_patients_query
                            .filter(pl.col("HADM_ID").is_in(valid_hadm_ids))
                            .filter(pl.col("ITEMID") == MAP_id)
                            .filter(pl.col("VALUE").cast(pl.Float64, strict=False) < 70)
                            .collect())

        final_hadm_ids = treated_patients["HADM_ID"].unique().to_list()
        print(f"Found {len(final_hadm_ids)} patients with hypotensive episodes")

        # Get all measurements for these patients
        print("Loading all chart events for selected patients...")
        treated_patients_all_values = read_large_csv_with_polars_mimic3(
            load_path_events, final_hadm_ids, measurements, data_limit=data_limit)

        treated_patients_all_values.write_parquet(save_path)
        print(f"Saved new dataset of patient values to {save_path}")
    else:
        print(f"Loading existing dataset from {save_path}")
        treated_patients_all_values = pl.read_parquet(save_path)

    return treated_patients_all_values


def read_large_csv_with_polars_mimic3(load_path, hadm_ids, measurements, data_limit=None):
    """
    Function to get all measurements from patients in MIMIC-III format

    Args:
        load_path: The path to CHARTEVENTS.csv
        hadm_ids: List of HADM_IDs to include
        measurements: All IDs of measurements necessary for modelling
        data_limit: Maximum number of rows to read

    Returns: df with the treated patient's events
    """
    query = pl.scan_csv(load_path)
    if data_limit is not None:
        query = query.limit(data_limit)

    result = (
        query.filter(pl.col("HADM_ID").is_in(hadm_ids))
        .filter(pl.col("ITEMID").is_in(measurements))
        .collect()
    )

    return result


def find_relevant_inputevents_mimic3(hadm_ids, save_path, events,
                                     inputevents_mv_path="../../data/mimic3/INPUTEVENTS_MV.csv",
                                     inputevents_cv_path="../../data/mimic3/INPUTEVENTS_CV.csv",
                                     data_limit=None):
    """
    Load relevant input events from both INPUTEVENTS_MV and INPUTEVENTS_CV for MIMIC-III

    Args:
        hadm_ids: List of HADM_IDs to include
        save_path: Path to save combined input events
        events: List of item IDs for medications
        inputevents_mv_path: Path to INPUTEVENTS_MV.csv
        inputevents_cv_path: Path to INPUTEVENTS_CV.csv
        data_limit: Maximum number of rows to read from each file
    """
    if not os.path.exists(save_path):
        print("Loading INPUTEVENTS_MV...")
        # Load MV events
        mv_query = pl.scan_csv(inputevents_mv_path)
        if data_limit is not None:
            mv_query = mv_query.limit(data_limit)

        mv_events = (mv_query
                     .filter(pl.col("HADM_ID").is_in(hadm_ids))
                     .filter(pl.col("ITEMID").is_in(events))
                     .collect())

        print("Loading INPUTEVENTS_CV...")
        # Load CV events
        cv_query = pl.scan_csv(inputevents_cv_path)
        if data_limit is not None:
            cv_query = cv_query.limit(data_limit)

        cv_events = (cv_query
                     .filter(pl.col("HADM_ID").is_in(hadm_ids))
                     .filter(pl.col("ITEMID").is_in(events))
                     .collect())

        print(f"Found {mv_events.height} MV events and {cv_events.height} CV events")

        # Combine and standardize columns
        # MV has STARTTIME, ENDTIME, RATE, AMOUNT
        # CV has CHARTTIME, AMOUNT, RATE (but rate calculation might be different)

        # Standardize MV events
        mv_standardized = mv_events.select([
            pl.col("HADM_ID").alias("hadm_id"),
            pl.col("ITEMID").alias("itemid"),
            pl.col("STARTTIME").alias("starttime"),
            pl.col("ENDTIME").alias("endtime"),
            pl.col("RATE").alias("rate"),
            pl.col("AMOUNT").alias("amount"),
            pl.lit("MV").alias("source")
        ])

        # Standardize CV events (approximate endtime if not available)
        cv_standardized = cv_events.select([
            pl.col("HADM_ID").alias("hadm_id"),
            pl.col("ITEMID").alias("itemid"),
            pl.col("CHARTTIME").alias("starttime"),
            pl.col("CHARTTIME").alias("endtime"),  # Will need to handle this differently
            pl.col("RATE").alias("rate"),
            pl.col("AMOUNT").alias("amount"),
            pl.lit("CV").alias("source")
        ])

        # Combine both datasets
        all_events = pl.concat([mv_standardized, cv_standardized])

        all_events.write_parquet(save_path)
        print(f"Saved combined input events to {save_path}")
    else:
        print(f"Loading existing input events from {save_path}")
        all_events = pl.read_parquet(save_path)

    return all_events


def debug_specific_patient_mimic3(parquet_path='../../data/processed_data/treated_patients_inputevents.parquet',
                                  itemid=30047,  # Norepinephrine in MIMIC-III
                                  hadm_id=None,  # Will be set automatically
                                  output_file='../../data/processed_data/debug_inputevents.csv',
                                  chartevents_path='../../data/processed_data/treated_patients_chartevents.parquet'):
    """
    Extract and examine all records for a specific itemid and hadm_id in MIMIC-III format
    """
    df = pd.read_parquet(parquet_path)

    # If no hadm_id provided, use the first one with this itemid
    if hadm_id is None:
        available_hadm_ids = df[df['itemid'] == itemid]['hadm_id'].unique()
        if len(available_hadm_ids) > 0:
            hadm_id = available_hadm_ids[0]
            print(f"Auto-selected hadm_id: {hadm_id}")
        else:
            print(f"No data found for itemid {itemid}")
            return None

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
    available_cols = ['starttime', 'endtime', 'rate', 'amount']
    display_cols = [col for col in available_cols if col in filtered.columns]
    print(filtered[display_cols].to_string())

    # Save to CSV
    filtered.to_csv(output_file, index=False)
    print(f"\nSaved {len(filtered)} records to {output_file}")

    # Show rate changes if available
    if len(filtered) > 1 and 'rate' in filtered.columns:
        print("\nRate progression:")
        for idx, row in filtered.iterrows():
            print(
                f"  {row['starttime']} -> {row.get('endtime', 'N/A')}: {row.get('rate', 'N/A')} ({'source: ' + str(row.get('source', 'unknown'))})")

    # Save chart events for this patient
    try:
        chartevents = pl.scan_parquet(chartevents_path).filter(pl.col("hadm_id") == hadm_id).collect()
        chartevents.write_csv("../../data/processed_data/debug_chartevents.csv")
        print(f"Saved chart events for patient {hadm_id}")
    except Exception as e:
        print(f"Could not save chart events: {e}")

    return filtered


def create_medication_tensors_mimic3(
        inputevents_path,
        crystalloid_itemids,
        vasopressor_itemids,
        time_interval_minutes=5,
        trajectory_before_minutes=None,
        trajectory_after_minutes=0,
        icustays_path='../../data/mimic3/ICUSTAYS.csv',
        cache_dir='../../data/processed_data/med_tensors',
        n_workers=4,
        debug_patient_id=None):
    """
    Create medication trajectory tensors for MIMIC-III patients with configurable trajectory windows.
    """
    # Create cache directory
    Path(cache_dir).mkdir(parents=True, exist_ok=True)

    interval_minutes = time_interval_minutes

    # Calculate n_intervals based on trajectory configuration
    if trajectory_before_minutes is not None:
        # Fixed window trajectories
        total_window_minutes = trajectory_before_minutes + trajectory_after_minutes
        n_intervals = int(np.ceil(total_window_minutes / interval_minutes))
        print(f"Using fixed trajectory windows: {total_window_minutes} minutes total")
    else:
        # Variable length trajectories - use max LOS
        print("Loading MIMIC-III ICU stays...")
        los_df = pl.read_csv(icustays_path)
        max_los_days = los_df['LOS'].max()  # MIMIC-III uses 'LOS'
        max_minutes = max_los_days * 24 * 60
        n_intervals = int(np.ceil(max_minutes / interval_minutes))
        print(f"Maximum LOS: {max_los_days:.2f} days, max intervals: {n_intervals}")

    # Load ICU stays data
    print("Loading ICU stays data...")
    icustays_df = pl.read_csv(icustays_path)

    # Parse datetime columns for MIMIC-III
    if icustays_df.schema['INTIME'] != pl.Datetime:
        icustays_df = icustays_df.with_columns(
            pl.col('INTIME').str.to_datetime()
        )

    # Get unique hadm_ids from input events
    print("Reading unique hadm_ids from input events...")
    unique_stays = pl.scan_parquet(inputevents_path).select('hadm_id').unique().collect()
    all_hadm_ids = unique_stays['hadm_id'].to_list()
    print(f"Total patients: {len(all_hadm_ids)}")

    # Set debug patient if not provided
    if debug_patient_id is None and len(all_hadm_ids) > 0:
        debug_patient_id = all_hadm_ids[0]
        print(f"Auto-selected debug patient: {debug_patient_id}")

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

    # Prepare medication data
    print("Preparing MIMIC-III medication events...")
    med_df = pl.read_parquet(inputevents_path)

    # Handle datetime parsing for MIMIC-III
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

    # Handle missing endtimes (especially for CV events)
    missing_endtimes = med_df.filter(pl.col('endtime').is_null()).height
    if missing_endtimes > 0:
        print(f"Warning: Found {missing_endtimes} events with missing endtimes, will estimate...")
        # For missing endtimes, estimate as starttime + 1 hour
        med_df = med_df.with_columns(
            pl.when(pl.col('endtime').is_null())
            .then(pl.col('starttime') + pl.duration(hours=1))
            .otherwise(pl.col('endtime'))
            .alias('endtime')
        )

    # Join with ICU admission times (using MIMIC-III column names)
    med_df = med_df.join(
        icustays_df.select(['HADM_ID', 'INTIME']).rename({'HADM_ID': 'hadm_id', 'INTIME': 'intime'}),
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

    # Save prepared data
    temp_prepared_path = Path(cache_dir) / "temp_prepared_inputevents.parquet"
    med_df.write_parquet(temp_prepared_path)
    print(f"Saved prepared medication events to {temp_prepared_path}")

    # Create admission time mapping
    stay_admission_map = {}
    for row in icustays_df.select(['HADM_ID', 'INTIME']).iter_rows(named=True):
        stay_admission_map[row['HADM_ID']] = row['INTIME']

    # Use the existing processing functions (they should work with the standardized data)
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

    # Process patients (sequential for debugging)
    for hadm_id in tqdm(all_hadm_ids[:10], desc="Processing patients"):  # Limit for initial testing
        admission_time = stay_admission_map.get(hadm_id)

        if admission_time is not None:
            try:
                result = process_func(
                    hadm_id=hadm_id,
                    icu_admission_time=admission_time
                )
                all_trajectory_info[result[0]] = result[1]

                # Save debug CSV for specific patient
                if int(hadm_id) == debug_patient_id:
                    save_patient_data_as_csv(
                        hadm_id=hadm_id,
                        inputevents_path=str(temp_prepared_path),
                        medication_info=medication_info,
                        n_intervals=n_intervals,
                        interval_minutes=interval_minutes,
                        icu_admission_time=admission_time,
                        output_dir=cache_dir
                    )
            except Exception as e:
                print(f"Error processing patient {hadm_id}: {e}")
                continue

    # Clean up and save metadata
    temp_prepared_path.unlink(missing_ok=True)

    # Calculate summary statistics
    total_trajectories = sum(len(traj_list) for traj_list in all_trajectory_info.values())

    print(f"\nProcessing complete:")
    print(f"Total patients processed: {len(all_trajectory_info)}")
    print(f"Total trajectories created: {total_trajectories}")

    # Save trajectory metadata
    trajectory_metadata = {
        'all_trajectories': all_trajectory_info,
        'medication_info': medication_info,
        'n_intervals': n_intervals,
        'interval_minutes': interval_minutes,
        'trajectory_before_minutes': trajectory_before_minutes,
        'trajectory_after_minutes': trajectory_after_minutes,
        'total_trajectories': total_trajectories,
        'mimic_version': 'MIMIC-III'
    }

    metadata_file = Path(cache_dir) / "trajectory_metadata.pkl"
    with open(metadata_file, "wb") as f:
        pickle.dump(trajectory_metadata, f)

    print(f"Saved trajectory metadata to {metadata_file}")
    return all_trajectory_info


# Copy over the helper functions from the original code that don't need modification
def extract_trajectories_from_patient(
        values_array, mask_array, abs_time_array, rel_time_array, t0_trigger_idx,
        trajectory_before_minutes=None, trajectory_after_minutes=0, interval_minutes=5
):
    """
    Split patient data into trajectories based on t0 triggers.
    (Same as original - no changes needed)
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
    (Same as original - no changes needed)
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
    (Same as original - works with standardized data format)
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
    (Same logic as original, just with MIMIC-III data)
    """
    # Create output directory
    patient_dir = Path(output_dir) / f"patient_{hadm_id}_inspection"
    patient_dir.mkdir(parents=True, exist_ok=True)

    # Read only this patient's data
    try:
        med_df_patient = pl.read_parquet(inputevents_path).filter(
            pl.col('hadm_id') == hadm_id
        )

        print(f"Debug CSV save: Patient {hadm_id} data shape: {med_df_patient.shape}")

        if med_df_patient.height == 0:
            print(f"No medication data found for patient {hadm_id}")
            return patient_dir

        # Check for required columns
        required_cols = ['itemid', 'rate', 'start_idx', 'end_idx']
        missing_cols = [col for col in required_cols if col not in med_df_patient.columns]
        if missing_cols:
            print(f"Warning: Missing columns {missing_cols} for patient {hadm_id}")
            return patient_dir

    except Exception as e:
        print(f"Error reading data for CSV save for patient {hadm_id}: {e}")
        return patient_dir

    # Process patient data (same logic as original)
    n_medications = len(medication_info) + 2
    values_array = np.zeros((n_intervals, n_medications), dtype=np.float32)
    mask_array = np.zeros((n_intervals, n_medications), dtype=np.float32)

    crystalloid_sum_idx = len(medication_info)
    t0_trigger_idx = len(medication_info) + 1

    crystalloid_arrays = {}
    vasopressor_arrays = {}

    # Process each medication (same logic as process_single_patient_medications)
    for idx, med_info in enumerate(medication_info):
        itemid = med_info['itemid']
        rate_array = np.zeros(n_intervals)

        med_events = med_df_patient.filter(pl.col('itemid') == itemid)

        for row in med_events.iter_rows(named=True):
            start_idx = max(0, min(row['start_idx'], n_intervals - 1))
            end_idx = max(0, min(row['end_idx'], n_intervals))

            if start_idx < end_idx:
                rate_array[start_idx:end_idx] = row['rate']

        if med_info['medication_type'] == 'crystalloid':
            rate_array = np.round(rate_array).astype(np.float32)
            crystalloid_arrays[itemid] = rate_array
        else:
            vasopressor_arrays[itemid] = rate_array

        values_array[:, idx] = rate_array
        mask_array[:, idx] = (rate_array != 0).astype(np.float32)

    # Calculate crystalloid sum and t0 triggers (same logic)
    crystalloid_sum = np.zeros(n_intervals, dtype=np.float32)
    for array in crystalloid_arrays.values():
        crystalloid_sum += array

    values_array[:, crystalloid_sum_idx] = crystalloid_sum
    mask_array[:, crystalloid_sum_idx] = (crystalloid_sum != 0).astype(np.float32)

    t0_array = np.zeros(n_intervals, dtype=np.float32)
    for i in range(1, n_intervals):
        trigger = 0

        for vaso_array in vasopressor_arrays.values():
            if vaso_array[i] > vaso_array[i - 1]:
                trigger = 1
                break

        if trigger == 0:
            crystalloid_change = abs(crystalloid_sum[i] - crystalloid_sum[i - 1])
            if crystalloid_change > 20 and crystalloid_sum[i] > 50:
                trigger = 1

        t0_array[i] = trigger

    values_array[:, t0_trigger_idx] = t0_array
    mask_array[:, t0_trigger_idx] = 1.0

    # Save CSV files (same logic as original)
    rel_time_array = np.arange(n_intervals) * interval_minutes / 60.0

    has_data = np.any(values_array > 0, axis=1)
    actual_length = np.where(has_data)[0][-1] + 1 if np.any(has_data) else 1

    values_to_save = values_array[:actual_length]
    mask_to_save = mask_array[:actual_length]
    time_to_save = rel_time_array[:actual_length]

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

    print(f"\nSaved inspection CSVs for patient {hadm_id} to: {patient_dir}")

    return patient_dir


def preprocess_baseline_values_mimic3(df,
                                      id_col='HADM_ID',
                                      categorical_features=None,
                                      continuous_features=None,
                                      binary_features=None,
                                      drop_first=True):
    """
    Preprocess baseline values for MIMIC-III deep learning input while preserving a unique identifier.
    """
    # Create a copy of the input dataframe and set the identifier as the index.
    processed_df = df.copy()
    if id_col in processed_df.columns:
        processed_df.set_index(id_col, inplace=True)

    # Process categorical features with one-hot encoding and ensure 0/1 integer values.
    if categorical_features is not None:
        # Handle missing values before one-hot encoding
        for col in categorical_features:
            if col in processed_df.columns:
                processed_df[col] = processed_df[col].fillna('UNKNOWN')

        dummies = pd.get_dummies(processed_df[categorical_features], drop_first=drop_first, dummy_na=False).astype(int)
    else:
        dummies = pd.DataFrame(index=processed_df.index)

    # Process continuous features ensuring they are numeric and scale them.
    if continuous_features is not None:
        continuous = processed_df[continuous_features].astype(float)
        # Normalize continuous features
        for col in continuous.columns:
            mean = continuous[col].mean()
            std = continuous[col].std()
            if std > 0:  # Avoid division by zero
                continuous[col] = (continuous[col] - mean) / std
            else:
                continuous[col] = 0  # If std is 0, set all values to 0
    else:
        continuous = pd.DataFrame(index=processed_df.index)

    # Process binary features and convert them to integer 0 or 1.
    if binary_features is not None:
        binary = processed_df[binary_features].apply(pd.to_numeric, errors='coerce').fillna(0).astype(int)
    else:
        binary = pd.DataFrame(index=processed_df.index)

    # Combine the processed parts into one DataFrame.
    final_df = pd.concat([continuous, binary, dummies], axis=1)

    return final_df


def create_baseline_tensors_mimic3(input_dir, output_dir, trajectory_metadata_path):
    """
    Load, merge, preprocess, and save static baseline features for each MIMIC-III patient.
    """
    print("\nCreating baseline feature tensors for MIMIC-III...")

    # Load trajectory metadata to get the list of hadm_id's we actually have trajectories for.
    with open(trajectory_metadata_path, 'rb') as f:
        trajectory_data = pickle.load(f)
    valid_hadm_ids = list(trajectory_data['all_trajectories'].keys())
    print(f"Found {len(valid_hadm_ids)} patients with trajectories.")

    # Load raw MIMIC-III data tables
    patients_path = os.path.join(input_dir, 'PATIENTS.csv')
    admissions_path = os.path.join(input_dir, 'ADMISSIONS.csv')

    try:
        print("Loading MIMIC-III PATIENTS.csv...")
        patients_df = pd.read_csv(patients_path)
        print("Loading MIMIC-III ADMISSIONS.csv...")
        admissions_df = pd.read_csv(admissions_path)
    except FileNotFoundError as e:
        print(f"Error: Could not find a required MIMIC-III file. Make sure your input directory is correct.")
        print(f"File not found: {e.filename}")
        return

    print(f"Loaded {len(patients_df)} patients and {len(admissions_df)} admissions")

    # Check required columns
    required_patient_cols = ['SUBJECT_ID', 'GENDER', 'DOB']
    required_admission_cols = ['SUBJECT_ID', 'HADM_ID', 'ADMITTIME', 'DISCHTIME', 'ADMISSION_TYPE',
                               'ADMISSION_LOCATION']

    missing_patient_cols = [col for col in required_patient_cols if col not in patients_df.columns]
    missing_admission_cols = [col for col in required_admission_cols if col not in admissions_df.columns]

    if missing_patient_cols:
        print(f"Warning: Missing patient columns: {missing_patient_cols}")
    if missing_admission_cols:
        print(f"Warning: Missing admission columns: {missing_admission_cols}")

    print("Available PATIENTS columns:", patients_df.columns.tolist())
    print("Available ADMISSIONS columns:", admissions_df.columns.tolist())

    # Merge admissions with patients to get age and gender
    merged_df = pd.merge(admissions_df, patients_df, on='SUBJECT_ID', how='left')
    print(f"Merged dataset has {len(merged_df)} records")

    # Calculate patient age at the time of admission (MIMIC-III has actual DOB)
    merged_df['ADMITTIME'] = pd.to_datetime(merged_df['ADMITTIME'])
    merged_df['DOB'] = pd.to_datetime(merged_df['DOB'])
    merged_df['age'] = (merged_df['ADMITTIME'] - merged_df['DOB']).dt.days / 365.25

    # Handle age > 89 (MIMIC-III masks ages > 89 by setting DOB to 300 years ago)
    # Patients with age > 200 actually have age > 89
    merged_df.loc[merged_df['age'] > 200, 'age'] = 91.4  # Set to median of >89 age group

    # Calculate admission duration in days
    merged_df['DISCHTIME'] = pd.to_datetime(merged_df['DISCHTIME'])
    merged_df['admit_duration'] = (merged_df['DISCHTIME'] - merged_df['ADMITTIME']).dt.total_seconds() / (3600 * 24)

    print(
        f"Age statistics: min={merged_df['age'].min():.1f}, max={merged_df['age'].max():.1f}, mean={merged_df['age'].mean():.1f}")
    print(
        f"Duration statistics: min={merged_df['admit_duration'].min():.1f}, max={merged_df['admit_duration'].max():.1f}, mean={merged_df['admit_duration'].mean():.1f}")

    # Filter by the valid hadm_ids that have trajectories
    merged_df_filtered = merged_df[merged_df['HADM_ID'].isin(valid_hadm_ids)]
    print(f"After filtering by valid hadm_ids: {len(merged_df_filtered)} records")

    # Apply additional filters as requested (2-10 days)
    merged_df_final = merged_df_filtered[
        (merged_df_filtered['admit_duration'] <= 10) &
        (merged_df_filtered['admit_duration'] >= 2)
        ]
    print(f"After duration filtering (2-10 days): {len(merged_df_final)} records")

    if len(merged_df_final) == 0:
        print("Warning: No patients remain after filtering!")
        return

    # Define feature lists for preprocessing (adapt to MIMIC-III column names)
    categorical_features = []
    continuous_features = ['age', 'admit_duration']
    binary_features = []

    # Add categorical features if they exist
    potential_categorical = ['GENDER', 'ETHNICITY', 'MARITAL_STATUS', 'INSURANCE',
                             'LANGUAGE', 'RELIGION', 'ADMISSION_TYPE', 'ADMISSION_LOCATION']

    for col in potential_categorical:
        if col in merged_df_final.columns:
            categorical_features.append(col)
            print(f"Added categorical feature: {col}")
            # Show unique values
            unique_vals = merged_df_final[col].value_counts()
            print(f"  Unique values: {unique_vals.to_dict()}")

    print(f"\nFinal feature configuration:")
    print(f"  Categorical features: {categorical_features}")
    print(f"  Continuous features: {continuous_features}")
    print(f"  Binary features: {binary_features}")

    # Preprocess the baseline data
    processed_baseline_df = preprocess_baseline_values_mimic3(
        merged_df_final,
        id_col='HADM_ID',
        categorical_features=categorical_features,
        continuous_features=continuous_features,
        binary_features=binary_features,
        drop_first=True
    )

    print(f"Processed baseline data shape: {processed_baseline_df.shape}")
    print(f"Feature names: {list(processed_baseline_df.columns)}")

    # Save a tensor for each patient
    baseline_dir = Path(output_dir) / "baseline_tensors"
    baseline_dir.mkdir(parents=True, exist_ok=True)

    saved_count = 0
    for hadm_id, row in processed_baseline_df.iterrows():
        if not np.any(np.isnan(row.values)):  # Only save if no NaN values
            tensor = torch.tensor(row.values, dtype=torch.float32)
            torch.save(tensor, baseline_dir / f"baseline_{hadm_id}.pt")
            saved_count += 1
        else:
            print(f"Skipping patient {hadm_id} due to NaN values")

    print(f"Saved {saved_count} baseline tensors to {baseline_dir}")

    # Save metadata for the dataloader and model
    baseline_metadata = {
        'feature_names': list(processed_baseline_df.columns),
        'feature_dim': len(processed_baseline_df.columns),
        'mimic_version': 'MIMIC-III',
        'categorical_features_used': categorical_features,
        'continuous_features_used': continuous_features,
        'binary_features_used': binary_features,
        'patients_processed': saved_count
    }
    metadata_file = baseline_dir / "baseline_metadata.pkl"
    with open(metadata_file, "wb") as f:
        pickle.dump(baseline_metadata, f)

    print(f"Saved baseline metadata to {metadata_file}")
    print(f"Baseline feature dimension: {baseline_metadata['feature_dim']}")

    # Save a sample of the processed data for inspection
    sample_df = processed_baseline_df.head(10)
    sample_df.to_csv(baseline_dir / "baseline_sample.csv")
    print(f"Saved sample baseline data to {baseline_dir / 'baseline_sample.csv'}")


def main():
    """Main function for MIMIC-III processing"""
    args = parse_args()

    # Create output directories
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    input_dir = Path(args.input_dir)

    print(f"MIMIC-III Processing Configuration:")
    print(f"  Input directory: {args.input_dir}")
    print(f"  Output directory: {args.output_dir}")
    print(f"  Filter file: {args.hadm_filter_file}")
    print(f"  Interval: {args.interval_minutes} minutes")
    print(f"  Workers: {args.n_workers}")
    print(f"  Data limit: {args.data_limit}")

    # Load the hadm_id filter
    filter_file_path = input_dir / args.hadm_filter_file
    hadm_id_filter = load_hadm_id_filter(filter_file_path)

    # Get MIMIC-III item IDs
    mimic3_ids = get_mimic3_item_ids()

    print(f"\nMIMIC-III Item ID Mappings:")
    for category, ids in mimic3_ids.items():
        print(f"  {category}: {ids}")

    # Define paths
    relevant_patients_chartevents_path = output_dir / "treated_patients_chartevents.parquet"
    relevant_patients_inputevents_path = output_dir / "treated_patients_inputevents.parquet"

    # Process chart events
    print("\n=== Processing Chart Events ===")
    all_patients_chartevents = find_relevant_patients_mimic3(
        measurements=mimic3_ids['hr'] + mimic3_ids['map'] + mimic3_ids['cvp'] + mimic3_ids['sv'] + mimic3_ids['co'],
        MAP_id=mimic3_ids['map'][0],  # Use first MAP item ID
        hadm_id_filter=hadm_id_filter,
        load_path_events=str(input_dir / "CHARTEVENTS.csv"),
        load_path_stays=str(input_dir / "ICUSTAYS.csv"),
        save_path=str(relevant_patients_chartevents_path),
        data_limit=args.data_limit
    )

    # Get hadm_ids that passed chart events filtering
    chart_hadm_ids = all_patients_chartevents['HADM_ID'].unique().to_list()

    # Process input events
    print("\n=== Processing Input Events ===")
    all_patients_inputevents = find_relevant_inputevents_mimic3(
        hadm_ids=chart_hadm_ids,
        save_path=str(relevant_patients_inputevents_path),
        events=mimic3_ids['crystalloids'] + mimic3_ids['vasopressors'] + mimic3_ids['inotropes'],
        inputevents_mv_path=str(input_dir / "INPUTEVENTS_MV.csv"),
        inputevents_cv_path=str(input_dir / "INPUTEVENTS_CV.csv"),
        data_limit=args.data_limit
    )

    # Debug a specific patient
    print("\n=== Debug Analysis ===")
    debug_result = debug_specific_patient_mimic3(
        parquet_path=str(relevant_patients_inputevents_path),
        itemid=mimic3_ids['vasopressors'][0],  # Use first vasopressor
        chartevents_path=str(relevant_patients_chartevents_path)
    )

    # Create medication tensors
    print("\n=== Creating Medication Tensors ===")
    medication_trajectories = create_medication_tensors_mimic3(
        inputevents_path=str(relevant_patients_inputevents_path),
        crystalloid_itemids=mimic3_ids['crystalloids'],
        vasopressor_itemids=mimic3_ids['vasopressors'],
        time_interval_minutes=args.interval_minutes,
        trajectory_before_minutes=args.trajectory_before_minutes,
        trajectory_after_minutes=args.trajectory_after_minutes,
        icustays_path=str(input_dir / "ICUSTAYS.csv"),
        cache_dir=str(output_dir / "med_tensors"),
        n_workers=args.n_workers,
        debug_patient_id=args.debug_patient_id
    )

    # Create baseline tensors
    print("\n=== Creating Baseline Tensors ===")
    med_trajectory_metadata_path = str(output_dir / "med_tensors" / "trajectory_metadata.pkl")

    if Path(med_trajectory_metadata_path).exists():
        create_baseline_tensors_mimic3(
            input_dir=str(input_dir),
            output_dir=str(output_dir),
            trajectory_metadata_path=med_trajectory_metadata_path
        )
    else:
        print(f"Warning: Could not find trajectory metadata at {med_trajectory_metadata_path}")
        print("Skipping baseline tensor creation.")

    print(f"\n=== Processing Complete ===")
    print(f"Total patients with trajectories: {len(medication_trajectories)}")
    print(f"Output saved to: {output_dir}")


if __name__ == "__main__":
    main()