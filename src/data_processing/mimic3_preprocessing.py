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
import wfdb


def parse_args():
    """Parse command line arguments for MIMIC-III."""
    parser = argparse.ArgumentParser(description='Process MIMIC-III data for ICU patient trajectories')

    # Time interval parameters
    parser.add_argument('--interval-minutes', type=int, default=1,
                        help='Time interval in minutes between observations (default: 1)')

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
    parser.add_argument('--output-dir', type=str, default='../../data/mimic3refactor/processed_data/',
                        help='Base output directory (default: ../../data/processed_data/mimic3refactor/)')
    parser.add_argument('--input-dir', type=str, default='../../data/mimic3refactor/input_data/',
                        help='Base input directory (default: ../../data/mimic3refactor)')

    # MIMIC-III specific parameters
    parser.add_argument('--hadm-filter-file', type=str, default='results_with_hadm_id.csv',
                        help='CSV file containing hadm_id column for filtering (default: results_with_hadm_id.csv)')

    return parser.parse_args()


def find_relevant_patients(MAP_id, load_path_events = "../../data/mimic3refactor/input_data/CHARTEVENTS.csv",
                          load_path_stays = "../../data/mimic3refactor/input_data/ICUSTAYS.csv",  save_path = "../../data/mimic3refactor/processed_data/relevant_patient_ids.csv",
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
        chartevents_schema_overrides = {
            'VALUE': pl.Utf8,  # Read as string first, then filter/convert
            'VALUENUM': pl.Float64  # If this column exists
        }
        treated_patients_query = pl.scan_csv(load_path_events, schema_overrides=chartevents_schema_overrides)
        if data_limit is not None:
            long_stays_query = long_stays_query.limit(data_limit)
            treated_patients_query = treated_patients_query.limit(data_limit)

        long_stays = (long_stays_query.filter(pl.col("LOS") > 1).collect())
        long_stays_id = long_stays["HADM_ID"].unique().to_list()
        treated_patients = (treated_patients_query.filter(pl.col("HADM_ID").is_in(long_stays_id))
                            .filter(pl.col("ITEMID").is_in(MAP_id))
                            .filter(pl.col("VALUE") != "Not Given")  # Filter out non-numeric values
                            .filter(pl.col("VALUE").cast(pl.Float64, strict=False) < 70)
                            .collect())

        # Extract and save the patient IDs
        patient_ids = treated_patients.select(['HADM_ID', 'SUBJECT_ID']).unique()
        patient_ids.write_csv(save_path)
        print(f"Saved new dataset of patient values to {save_path}")
    else:
        print(f"Loading dataset from {save_path}")
        patient_ids = pl.read_csv(save_path)

    hadm_ids = patient_ids['HADM_ID'].unique().to_list()

    return hadm_ids


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

def find_relevant_inputevents(hadm_ids, save_path, events, inputevents_mv_path = "../../data/mimic3refactor/input_data/INPUTEVENTS_MV.csv",
inputevents_cv_path = "../../data/mimic3refactor/input_data/INPUTEVENTS_CV.csv", data_limit=None):
    if not os.path.exists(save_path):
        print("Loading INPUTEVENTS_MV...")

        # Define schema to handle data type issues - THIS FIXES THE PARSING ERROR
        mv_schema_overrides = {
            'TOTALAMOUNT': pl.Float64,  # This was causing the parsing error!
            'RATE': pl.Float64,
            'AMOUNT': pl.Float64
        }

        # Load MV events with proper schema
        mv_query = pl.scan_csv(inputevents_mv_path, schema_overrides=mv_schema_overrides)
        if data_limit is not None:
            mv_query = mv_query.limit(data_limit)

        mv_events = (mv_query
                     .filter(pl.col("HADM_ID").is_in(hadm_ids))
                     .filter(pl.col("ITEMID").is_in(events))
                     .collect())

        # CV events schema
        cv_schema_overrides = {
            'AMOUNT': pl.Float64,
            'RATE': pl.Float64
        }


        cv_query = pl.scan_csv(inputevents_cv_path, schema_overrides=cv_schema_overrides)
        if data_limit is not None:
            cv_query = cv_query.limit(data_limit)

        cv_events = (cv_query
                     .filter(pl.col("HADM_ID").is_in(hadm_ids))
                     .filter(pl.col("ITEMID").is_in(events))
                     .collect())

        print(f"Found {mv_events.height} MV events and {cv_events.height} CV events")

        # Combine and standardize columns
        if mv_events.height > 0:
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
        else:
            # Create empty standardized MV dataframe
            mv_standardized = pl.DataFrame({
                "HADM_ID": [], "itemid": [], "starttime": [], "endtime": [],
                "rate": [], "amount": [], "source": []
            })

        if cv_events.height > 0:
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
        else:
            # Create empty standardized CV dataframe
            cv_standardized = pl.DataFrame({
                "hadm_id": [], "itemid": [], "starttime": [], "endtime": [],
                "rate": [], "amount": [], "source": []
            })

        # Combine both datasets
        if mv_standardized.height > 0 or cv_standardized.height > 0:
            all_patients_inputevents = pl.concat([mv_standardized, cv_standardized])
        else:
            # Create completely empty dataframe
            all_patients_inputevents = pl.DataFrame({
                "HADM_ID": [], "itemid": [], "starttime": [], "endtime": [],
                "rate": [], "amount": [], "source": []
            })

        all_patients_inputevents.write_parquet(save_path)
        print(f"Saved combined input events to {save_path}")


    else:
        print(f"Loading existing dataset from {save_path}")
        all_patients_inputevents = pl.read_parquet(save_path)

    return all_patients_inputevents

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
    """

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
            # Create empty CSVs with minimal data
            time_array = np.arange(1) * interval_minutes / 60.0  # Just one row

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

    # Process patient data (same logic as process_single_patient_medications)
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
        actual_length = 1  # At least one row

    # Truncate arrays to actual length for CSV
    values_to_save = values_array[:actual_length]
    mask_to_save = mask_array[:actual_length]
    time_to_save = rel_time_array[:actual_length]

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
        'icu_admission_time': str(icu_admission_time)
    })
    time_df.to_csv(patient_dir / 'time.csv', index=False)

    # Extract and save trajectory information
    trajectories = extract_trajectories_from_patient(
        values_array, mask_array, abs_time_array, rel_time_array, t0_trigger_idx,
        trajectory_before_minutes=None,  # Use default for CSV output
        trajectory_after_minutes=0,
        interval_minutes=interval_minutes
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
            'start_time_hours': rel_time_array[start_idx] if start_idx < len(rel_time_array) else 0,
            'end_time_hours': rel_time_array[end_idx - 1] if end_idx > 0 and end_idx <= len(rel_time_array) else 0,
            'has_t0_trigger': has_t0_at_end
        })

    if traj_data:
        traj_df = pd.DataFrame(traj_data)
        traj_df.to_csv(patient_dir / 'trajectories.csv', index=False)

    # Save summary information
    summary_data = {
        'hadm_id': [hadm_id],
        'icu_admission_time': [str(icu_admission_time)],
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
        time_interval_minutes=1,
        trajectory_before_minutes=None,
        trajectory_after_minutes=0,
        icustays_path='../../data/mimic3refactor/input_data/icustays.csv',
        los_data_path='../../data/mimic3refactor/input_data/icustays.csv',
        cache_dir='../../data/mimic3refactor/processed_data/med_tensors',
        n_workers=4,
        debug_patient_id=20214994):
    """
    Create medication trajectory tensors for each patient with configurable trajectory windows.
    """
    # Create cache directory
    Path(cache_dir).mkdir(parents=True, exist_ok=True)

    interval_minutes = time_interval_minutes

    # Calculate n_intervals based on trajectory configuration
    if trajectory_before_minutes is not None:
        # Fixed window trajectories - calculate intervals from window size
        total_window_minutes = trajectory_before_minutes + trajectory_after_minutes
        n_intervals = int(np.ceil(total_window_minutes / interval_minutes))

        print(f"Using fixed trajectory windows:")
        print(f"  {trajectory_before_minutes} minutes before t0")
        print(f"  {trajectory_after_minutes} minutes after t0")
        print(f"  Total window: {total_window_minutes} minutes")
        print(f"  Number of intervals: {n_intervals} (at {interval_minutes} minutes each)")

        # Don't need max_los_days for fixed windows
        max_los_days = None
    else:
        # Variable length trajectories - need to check maximum LOS
        print("Using variable trajectory windows (ICU admission to t0)")
        print("Loading length of stay data...")

        los_df = pl.read_csv(los_data_path)
        if 'LOS' not in los_df.columns:
            raise ValueError(f"Column 'los' not found in {los_data_path}. Available columns: {los_df.columns}")

        max_los_days = los_df['LOS'].max()
        max_minutes = max_los_days * 24 * 60
        n_intervals = int(np.ceil(max_minutes / interval_minutes))

        print(f"Maximum length of stay: {max_los_days:.2f} days")
        print(f"Maximum intervals needed: {n_intervals} "
              f"(based on max LOS with {interval_minutes}-min intervals)")

    # Load ICU stays data
    print("\nLoading ICU stays data...")
    icustays_df = pl.read_csv(icustays_path)

    # Parse datetime columns
    if icustays_df.schema['INTIME'] != pl.Datetime:
        icustays_df = icustays_df.with_columns(
            pl.col('INTIME').str.to_datetime()
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
        icustays_df.select(['HADM_ID', 'INTIME']).rename({'HADM_ID': 'hadm_id'}),
        on='hadm_id',
        how='inner'
    )

    # Calculate time indices
    med_df = med_df.with_columns([
        ((pl.col('starttime') - pl.col('INTIME')).dt.total_seconds() / 60).alias('start_minutes'),
        ((pl.col('starttime') - pl.col('INTIME')).dt.total_seconds() / 60 / interval_minutes)
        .floor().cast(pl.Int32).alias('start_idx'),
        ((pl.col('endtime') - pl.col('INTIME')).dt.total_seconds() / 60 / interval_minutes)
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
    for row in icustays_df.select(['HADM_ID', 'INTIME']).iter_rows(named=True):
        stay_admission_map[row['HADM_ID']] = row['INTIME']

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


def preprocess_baseline_values(df,
                               id_col='hadm_id',
                               categorical_features=None,
                               continuous_features=None,
                               binary_features=None,
                               drop_first=True):
    """
    Preprocess baseline values for deep learning input while preserving a unique identifier.
    """
    # Create a copy of the input dataframe and set the identifier as the index.
    processed_df = df.copy()
    if id_col in processed_df.columns:
        processed_df.set_index(id_col, inplace=True)
    print(f"Categorical features: {categorical_features}")
    # Process categorical features with one-hot encoding and ensure 0/1 integer values.
    if categorical_features is not None:
        dummies = pd.get_dummies(processed_df[categorical_features], drop_first=drop_first, dummy_na=False).astype(int)
    else:
        dummies = pd.DataFrame(index=processed_df.index)

    print(f"Dummies shape: {dummies.shape}")
    print(f"Dummies columns: {list(dummies.columns)}")

    # Process continuous features ensuring they are numeric and scale them.
    if continuous_features is not None:
        continuous = processed_df[continuous_features].astype(float)
        # Normalize continuous features
        for col in continuous.columns:
            mean = continuous[col].mean()
            std = continuous[col].std()
            continuous[col] = (continuous[col] - mean) / std
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


def create_baseline_tensors(input_dir, output_dir, trajectory_metadata_path):
    """
    Load, merge, preprocess, and save static baseline features for each patient.
    """
    print("\nCreating baseline feature tensors...")

    # Load trajectory metadata to get the list of hadm_id's we actually have trajectories for.
    with open(trajectory_metadata_path, 'rb') as f:
        trajectory_data = pickle.load(f)
    valid_hadm_ids = list(trajectory_data['all_trajectories'].keys())
    print(f"Found {len(valid_hadm_ids)} patients with trajectories.")

    # Load raw MIMIC data tables from the 'hosp' module
    patients_path = os.path.join(input_dir, 'PATIENTS.csv')
    admission_path = os.path.join(input_dir, 'ADMISSIONS.csv')
    transfers_path = os.path.join(input_dir, 'TRANSFERS.csv')

    try:
        patients_df = pd.read_csv(patients_path)
        admission_df = pd.read_csv(admission_path)
        transfers_df = pd.read_csv(transfers_path)
    except FileNotFoundError as e:
        print(f"Error: Could not find a required MIMIC 'hosp' file. Make sure your input directory is correct.")
        print(f"File not found: {e.filename}")
        return

    # Merge admissions with patients to get age and gender
    merged_df = pd.merge(admission_df, patients_df, on='SUBJECT_ID', how='left')

    # Calculate patient age at the time of admission
    merged_df['ADMITTIME'] = pd.to_datetime(merged_df['ADMITTIME'])
    merged_df['DOB'] = pd.to_datetime(merged_df['DOB'], errors='coerce')
    prov_age =  merged_df['ADMITTIME'].dt.year - merged_df['DOB'].dt.year
    merged_df['AGE'] = prov_age.where(prov_age < 300, 89)

    # Merge with transfers to get discharge time
    merged_df = pd.merge(merged_df, transfers_df, on=['HADM_ID', 'SUBJECT_ID'], how='left')

    # Calculate admission duration in days
    merged_df['DISCHTIME'] = pd.to_datetime(merged_df['DISCHTIME'])
    merged_df['ADMIT_DURATION'] = (merged_df['DISCHTIME'] - merged_df['ADMITTIME']).dt.total_seconds() / (3600 * 24)

    # Filter by the valid hadm_ids that have trajectories
    merged_df_final_filtered = merged_df[merged_df['HADM_ID'].isin(valid_hadm_ids)]

    # Apply additional filters as requested
    merged_with_disch_df_final_filtered = merged_df_final_filtered[merged_df_final_filtered['ADMIT_DURATION'] <= 10]
    merged_with_disch_df_final_filtered = merged_with_disch_df_final_filtered[
        merged_with_disch_df_final_filtered['ADMIT_DURATION'] >= 2]

    print(f"Found {len(merged_with_disch_df_final_filtered)} patients after duration filtering (2-10 days).")
    print("Available columns in merged data:", merged_df_final_filtered.columns.tolist())

    # Define feature lists for preprocessing
    categorical_features = ['GENDER', 'ETHNICITY', 'MARITAL_STATUS',
                            'INSURANCE', 'LANGUAGE', 'ADMISSION_LOCATION', 'ADMISSION_TYPE']
    continuous_features = ['AGE', 'ADMIT_DURATION']
    binary_features = []  # None specified in example

    # Preprocess the baseline data
    processed_baseline_df = preprocess_baseline_values(
        merged_with_disch_df_final_filtered,
        id_col='HADM_ID',
        categorical_features=categorical_features,
        continuous_features=continuous_features,
        binary_features=binary_features,
        drop_first=True
    )

    # Save a tensor for each patient
    baseline_dir = Path(output_dir) / "baseline_tensors"
    baseline_dir.mkdir(parents=True, exist_ok=True)

    saved_count = 0
    for hadm_id, row in processed_baseline_df.iterrows():
        tensor = torch.tensor(row.values, dtype=torch.float32)
        torch.save(tensor, baseline_dir / f"baseline_{hadm_id}.pt")
        saved_count += 1

    print(f"Saved {saved_count} baseline tensors to {baseline_dir}")

    # Save metadata for the dataloader and model
    baseline_metadata = {
        'feature_names': list(processed_baseline_df.columns),
        'feature_dim': len(processed_baseline_df.columns)
    }
    metadata_file = baseline_dir / "baseline_metadata.pkl"
    with open(metadata_file, "wb") as f:
        pickle.dump(baseline_metadata, f)

    print(f"Saved baseline metadata to {metadata_file}")
    print(f"Baseline feature dimension: {baseline_metadata['feature_dim']}")


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
    hadm_ids = find_relevant_patients(MAP_id=[225312, 52, 6702], data_limit=args.data_limit)

    # Get MIMIC-III item IDs
    mimic3_ids = get_mimic3_item_ids()

    print(f"\nMIMIC-III Item ID Mappings:")
    for category, ids in mimic3_ids.items():
        print(f"  {category}: {ids}")

    # Define paths
    relevant_patients_chartevents_path = output_dir / "treated_patients_chartevents.parquet"
    relevant_patients_inputevents_path = output_dir / "treated_patients_inputevents.parquet"


    # Process input events
    print("\n=== Processing Input Events ===")
    # find all those inputs that are relevant to us and done on the relevant patients
    all_patients_inputevents = find_relevant_inputevents(
        hadm_ids=hadm_ids,
        save_path=str(relevant_patients_inputevents_path),
        events=mimic3_ids['crystalloids'] + mimic3_ids['vasopressors'],
        inputevents_mv_path=str(input_dir / "INPUTEVENTS_MV.csv"),
        inputevents_cv_path=str(input_dir / "INPUTEVENTS_CV.csv"),
        data_limit=args.data_limit
    )

    # Debug a specific patient
    print("\n=== Debug Analysis ===")
    #debug_result = debug_specific_patient_mimic3(
    #    parquet_path=str(relevant_patients_inputevents_path),
     #   itemid=mimic3_ids['vasopressors'][0],  # Use first vasopressor
      #  chartevents_path=str(relevant_patients_chartevents_path)
    #)

    # Create medication tensors
    print("\n=== Creating Medication Tensors ===")
    medication_trajectories = create_medication_tensors(
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
        create_baseline_tensors(
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