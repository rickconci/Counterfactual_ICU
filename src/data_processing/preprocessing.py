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

def find_relevant_patients(measurements, MAP_id = 220052, load_path_events = "../../data/chartevents.csv", load_path_stays = "../../data/icustays.csv",  save_path = "../../data/treated_patients_chartevents.parquet"):
    """
    Finds all potentially relevant patients by filtering on those that have had a blood pressure event and
    that have stayed in the ICU for over 24h.
    Args:
        MAP_id: Measurement ID for mean arterial pressure
        load_path_events: path to original dataset containing all events
        load_path_stays: path to dataset containing meta-information on ICU stay
        save_path: path to save relevant patients

    Returns:
            dataset containing all occurrences of the treatment to be used to filter patients
    """
    if not os.path.exists(save_path):
        # TODO there is a limit here
        long_stays = (pl.scan_csv(load_path_stays).limit(1000000)
                      .filter(pl.col("los") > 1)
                      .collect())
        long_stays_id = long_stays["stay_id"].unique().to_list()
        # Find patients with a blood pressure event
        # TODO there is a limit here
        treated_patients = (pl.scan_csv(load_path_events).limit(1000000)
                       .filter(pl.col("stay_id").is_in(long_stays_id))
                       .filter(pl.col("itemid") == MAP_id)
                       .filter(pl.col("value").cast(pl.Float64, strict=False) < 70)
                       .collect())

        treated_patients_all_values = read_large_csv_with_polars(load_path_events, treated_patients, measurements)
        treated_patients_all_values.write_parquet(save_path)
        print(f"Saved new dataset of patient values to {save_path}")
    else:
        print(f"Loading dataset from {save_path}")
        treated_patients_all_values = pl.read_parquet(save_path)

    return treated_patients_all_values

def read_large_csv_with_polars(load_path, ids_df, measurements, id_column='stay_id', item_column = 'itemid'):
    """
    Function to get all measurements from patients that have had the treatment
    Args:
        load_path: The path to the dataset of all patient measurements
        ids_df: df containing the ID's of patients with the treatment
        measurements: All IDs of measurements necessary for modelling
        id_column: The column to merge the dataset
        item_column: The column containing measurement IDs

    Returns: df with the treated patient's events

    """

    valid_ids = ids_df[id_column].unique().to_list()
    ids_df.write_parquet("../../data/temp.parquet")
    # Polars handles large files much better
    result = (
        # TODO there is a limit here
        pl.scan_csv(load_path).limit(1000000)
        .filter(pl.col(id_column).is_in(valid_ids))
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

def find_relevant_inputevents(all_patients_chartevents, save_path, events, inputevents_path = "../../data/inputevents.csv", patient_id = "stay_id", item_column = "itemid"):
    if not os.path.exists(save_path):
        patients = all_patients_chartevents[patient_id].unique().to_list()
        schema_overrides = {
            'totalamount': pl.Float64  # or pl.Float32 if you want less precision
        }

        all_patients_inputevents = (pl.scan_csv(inputevents_path, schema_overrides=schema_overrides).limit(1000000)
                                    .filter(pl.col(patient_id).is_in(patients))
                                    .filter(pl.col(item_column).is_in(events))
                                    .collect())
        all_patients_inputevents.write_parquet(save_path)
        print(f"Saved new dataset to {save_path}")
    else:
        print(f"Loading existing dataset from {save_path}")
        all_patients_inputevents = pl.read_parquet(save_path)

    return all_patients_inputevents


def create_medication_rate_matrix(
        inputevents_df,  # Polars DataFrame
        crystalloid_itemids,
        vasopressor_itemids,
        time_interval='5T',  # 5 minutes
        icustays_path='../../data/icustays.csv'
):
    """
    Create a matrix where each row is a patient-medication combination and
    columns are time intervals showing medication rates.
    Creates rows for ALL medication-patient combinations, even if never given.

    Parameters:
    - inputevents_df: Polars DataFrame with inputevents data
    - crystalloid_itemids: list of itemids for crystalloids
    - vasopressor_itemids: list of itemids for vasopressors
    - time_interval: pandas-like frequency string (default '5T' = 5 minutes)
    - icustays_path: path to icustays.csv file

    Returns:
    - Polars DataFrame with rows for each patient-medication and columns for time intervals
    """

    # Step 0: Load ICU stays data
    print("Loading ICU stays data...")
    icustays_df = pl.read_csv(icustays_path)

    # Parse datetime columns if needed
    if icustays_df.schema['intime'] != pl.Datetime:
        icustays_df = icustays_df.with_columns(
            pl.col('intime').str.to_datetime()
        )

    # Calculate max intervals from maximum LOS (length of stay)
    max_los_days = icustays_df['los'].max()
    print(f"Maximum ICU length of stay: {max_los_days:.2f} days")

    # Extract interval minutes from string like '5T'
    interval_minutes = int(time_interval.replace('T', '').replace('min', ''))
    total_minutes = max_los_days * 24 * 60
    n_intervals = int(np.ceil(total_minutes / interval_minutes))

    print(f"Creating {n_intervals} intervals of {interval_minutes} minutes each")

    # Get all unique stay_ids from inputevents
    all_stay_ids = inputevents_df['stay_id'].unique().to_list()
    print(f"Total patients in ICU stays: {len(all_stay_ids)}")

    # Create medication metadata
    all_meds = crystalloid_itemids + vasopressor_itemids
    medication_info = []
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

    # Step 1: Filter inputevents for medications of interest
    print("Processing medication events...")

    # Ensure datetime columns are parsed
    if inputevents_df.schema.get('starttime') != pl.Datetime:
        inputevents_df = inputevents_df.with_columns([
            pl.col('starttime').str.to_datetime(),
            pl.col('endtime').str.to_datetime()
        ])

    # Filter for relevant medications and stay_ids
    med_df = inputevents_df.filter(
        pl.col('itemid').is_in(all_meds) &
        pl.col('rate').is_not_null() &
        pl.col('stay_id').is_in(all_stay_ids)
    )

    # Check for missing endtimes
    missing_endtimes_count = med_df.filter(pl.col('endtime').is_null()).height
    if missing_endtimes_count > 0:
        print(f"\nERROR: Found {missing_endtimes_count} events with missing endtimes")
        raise ValueError(f"Found {missing_endtimes_count} infusion events with missing endtimes.")

    # Join with ICU admission times
    med_df = med_df.join(
        icustays_df.select(['stay_id', 'intime']),
        on='stay_id',
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

    # Create a dictionary to store actual medication data
    print("Building medication event dictionary...")
    medication_data = {}

    for row in tqdm(med_df.iter_rows(named=True), total=med_df.height, desc="Processing events"):
        key = (row['stay_id'], row['itemid'])
        if key not in medication_data:
            medication_data[key] = []
        medication_data[key].append((row['start_idx'], row['end_idx'], row['rate']))

    # Create comprehensive result data
    print("\nCreating comprehensive matrix...")
    result_data = []

    # Lists to collect rate changes for statistics
    all_vasopressor_changes = []
    all_crystalloid_rate_sum_changes = []

    # Counter for t0 rows
    t0_rows_created = 0

    # Calculate total rows for progress bar (+2 now for sum row and t0 row)
    total_rows = len(all_stay_ids) * (len(medication_info) + 2)  # +2 for sum row and t0 row

    # Create a row for EVERY stay_id × medication combination
    with tqdm(total=total_rows, desc="Building matrix rows") as pbar:
        for stay_id in all_stay_ids:
            # Process all medications for this patient
            crystalloid_arrays = {}  # Store for sum calculation
            vasopressor_arrays = {}  # Store for t0 calculation

            for med_info in medication_info:
                itemid = med_info['itemid']

                # Initialize rate array with zeros
                rate_array = np.zeros(n_intervals)

                # Fill in actual rates if this patient received this medication
                key = (stay_id, itemid)
                if key in medication_data:
                    for start_idx, end_idx, rate in medication_data[key]:
                        # Clip indices to valid range
                        start_idx = max(0, min(start_idx, n_intervals - 1))
                        end_idx = max(0, min(end_idx, n_intervals))

                        if start_idx < end_idx:
                            rate_array[start_idx:end_idx] = rate

                # Round crystalloid rates and store for sum
                if med_info['medication_type'] == 'crystalloid':
                    rate_array = np.round(rate_array).astype(int)
                    crystalloid_arrays[itemid] = rate_array
                else:  # vasopressor
                    vasopressor_arrays[itemid] = rate_array

                # Create row dictionary
                row_dict = {
                    'stay_id': stay_id,
                    'itemid': itemid,
                    'medication_type': med_info['medication_type'],
                    'medication_name': med_info['medication_name']
                }

                # Add time columns
                for i in range(n_intervals):
                    col_name = f"T_{i * interval_minutes:04d}"
                    row_dict[col_name] = rate_array[i]

                result_data.append(row_dict)
                pbar.update(1)

            # Add crystalloid sum row for this patient
            sum_row = {
                'stay_id': stay_id,
                'itemid': -1,
                'medication_type': 'crystalloid_rate_sum',
                'medication_name': 'TOTAL_CRYSTALLOIDS'
            }

            # Calculate sum using stored arrays
            sum_array = np.zeros(n_intervals, dtype=int)
            for itemid, array in crystalloid_arrays.items():
                sum_array += array

            # Add time columns for sum
            for i in range(n_intervals):
                col_name = f"T_{i * interval_minutes:04d}"
                sum_row[col_name] = sum_array[i]

            result_data.append(sum_row)
            pbar.update(1)

            # Collect crystalloid sum rate changes for statistics
            for i in range(1, n_intervals):
                if sum_array[i - 1] > 0 or sum_array[i] > 0:  # Only count if there's activity
                    change = sum_array[i] - sum_array[i - 1]
                    if change != 0:
                        all_crystalloid_rate_sum_changes.append(change)

            # Collect vasopressor rate changes for statistics
            for vaso_itemid, vaso_array in vasopressor_arrays.items():
                for i in range(1, n_intervals):
                    if vaso_array[i - 1] > 0 or vaso_array[i] > 0:  # Only count if there's activity
                        change = vaso_array[i] - vaso_array[i - 1]
                        if change != 0:
                            all_vasopressor_changes.append(change)

            # Add t0 trigger row for this patient
            t0_row = {
                'stay_id': stay_id,
                'itemid': -2,  # Special identifier for t0 rows
                'medication_type': 'trigger',
                'medication_name': 't0_trigger'  # Changed from just 't0' for clarity
            }

            # Calculate t0 triggers
            t0_array = np.zeros(n_intervals, dtype=int)

            # Check each time interval (starting from index 1 to compare with previous)
            for i in range(1, n_intervals):
                trigger = 0

                # Check vasopressor condition: any vasopressor rate increased
                for vaso_array in vasopressor_arrays.values():
                    if vaso_array[i] > vaso_array[i - 1]:
                        trigger = 1
                        break

                # Check TOTAL crystalloid condition: total changed by >20ml/h AND is >50ml/h
                if trigger == 0:  # Only check if not already triggered
                    crystalloid_change = abs(sum_array[i] - sum_array[i - 1])
                    if crystalloid_change > 20 and sum_array[i] > 50:
                        trigger = 1

                t0_array[i] = trigger

            # Add time columns for t0
            for i in range(n_intervals):
                col_name = f"T_{i * interval_minutes:04d}"
                t0_row[col_name] = t0_array[i]

            result_data.append(t0_row)
            t0_rows_created += 1
            pbar.update(1)

    # Create DataFrame from results
    print("Converting to DataFrame...")
    result_df = pl.DataFrame(result_data)

    # Sort final result - make sure t0 rows appear after crystalloid_rate_sum
    result_df = result_df.sort(['stay_id', 'medication_type', 'itemid'])

    # Verify t0 rows are present
    t0_count = result_df.filter(pl.col('medication_name') == 't0_trigger').height
    print(f"\nVerification: Created {t0_rows_created} t0 rows, found {t0_count} in final DataFrame")

    # Print summary statistics
    print(f"\nCreated comprehensive matrix:")
    print(f"Total rows: {result_df.height}")
    print(f"Expected rows: {len(all_stay_ids) * (len(all_meds) + 2)}")  # +2 for sum and t0
    print(f"Unique patients: {result_df['stay_id'].n_unique()}")
    print(f"Medications tracked: {len(crystalloid_itemids)} crystalloids, {len(vasopressor_itemids)} vasopressors")

    # Show row type breakdown
    print("\nRow type breakdown:")
    row_types = result_df.group_by('medication_type').agg(pl.count().alias('count')).sort('medication_type')
    print(row_types)

    # Check how many patients actually received each medication
    print("\nCalculating medication usage statistics...")
    actual_usage = {}
    for med_info in medication_info:
        itemid = med_info['itemid']
        patients_with_med = len([k for k in medication_data.keys() if k[1] == itemid])
        actual_usage[med_info['medication_name']] = patients_with_med

    print("\nMedication usage:")
    for med_name, count in sorted(actual_usage.items()):
        percentage = (count / len(all_stay_ids)) * 100
        print(f"  {med_name}: {count}/{len(all_stay_ids)} patients ({percentage:.1f}%)")

    # Calculate t0 trigger statistics
    t0_rows = result_df.filter(pl.col('medication_name') == 't0_trigger')
    if t0_rows.height > 0:
        time_cols = [col for col in result_df.columns if col.startswith('T_')]
        total_triggers = 0
        for col in time_cols:
            total_triggers += t0_rows[col].sum()

        print(f"\nt0 trigger statistics:")
        print(f"  Total trigger events: {total_triggers}")
        print(f"  Average triggers per patient: {total_triggers / len(all_stay_ids):.1f}")

        # Show sample of t0 rows
        print("\nSample t0 rows (first 3):")
        print(t0_rows.select(['stay_id', 'itemid', 'medication_type', 'medication_name']).head(3))
    else:
        print("\nWARNING: No t0 rows found in the final DataFrame!")

    # Print rate change distribution statistics
    print("\n" + "=" * 60)
    print("RATE CHANGE DISTRIBUTION ANALYSIS")
    print("=" * 60)

    # Vasopressor rate changes
    if all_vasopressor_changes:
        vaso_changes = np.array(all_vasopressor_changes)
        print("\nVasopressor rate changes (non-zero):")
        print(f"  Total changes: {len(vaso_changes)}")
        print(f"  Increases: {np.sum(vaso_changes > 0)} ({np.sum(vaso_changes > 0) / len(vaso_changes) * 100:.1f}%)")
        print(f"  Decreases: {np.sum(vaso_changes < 0)} ({np.sum(vaso_changes < 0) / len(vaso_changes) * 100:.1f}%)")
        print(f"  Mean change: {np.mean(vaso_changes):.3f}")
        print(f"  Median change: {np.median(vaso_changes):.3f}")
        print(f"  Std dev: {np.std(vaso_changes):.3f}")
        print(f"  Min/Max: {np.min(vaso_changes):.3f} / {np.max(vaso_changes):.3f}")

        # Percentiles
        print("\n  Percentiles:")
        for p in [5, 25, 50, 75, 95]:
            print(f"    {p}th: {np.percentile(vaso_changes, p):.3f}")
    else:
        print("\nNo vasopressor rate changes found.")

    # Total crystalloid rate changes
    if all_crystalloid_rate_sum_changes:
        cryst_changes = np.array(all_crystalloid_rate_sum_changes)
        print("\nTotal crystalloid rate changes (non-zero):")
        print(f"  Total changes: {len(cryst_changes)}")
        print(f"  Increases: {np.sum(cryst_changes > 0)} ({np.sum(cryst_changes > 0) / len(cryst_changes) * 100:.1f}%)")
        print(f"  Decreases: {np.sum(cryst_changes < 0)} ({np.sum(cryst_changes < 0) / len(cryst_changes) * 100:.1f}%)")
        print(f"  Mean change: {np.mean(cryst_changes):.1f} mL/hr")
        print(f"  Median change: {np.median(cryst_changes):.1f} mL/hr")
        print(f"  Std dev: {np.std(cryst_changes):.1f} mL/hr")
        print(f"  Min/Max: {np.min(cryst_changes):.1f} / {np.max(cryst_changes):.1f} mL/hr")

        # Percentiles
        print("\n  Percentiles:")
        for p in [5, 25, 50, 75, 95]:
            print(f"    {p}th: {np.percentile(cryst_changes, p):.1f} mL/hr")

        # Changes > 20 mL/hr (relevant for t0 trigger)
        large_changes = np.sum(np.abs(cryst_changes) > 20)
        print(f"\n  Changes > 20 mL/hr: {large_changes} ({large_changes / len(cryst_changes) * 100:.1f}%)")
    else:
        print("\nNo crystalloid rate changes found.")

    result_df.write_parquet("../../data/meds_matrix.parquet")

    return result_df


# Helper function to validate the output
def summarize_medication_matrix(matrix_df):
    """
    Provide summary statistics for the medication matrix
    """
    print("\nMatrix Summary:")
    print(f"Total rows: {matrix_df.height}")
    print(f"Columns: {matrix_df.width}")

    # Count by medication type
    type_counts = matrix_df.group_by('medication_type').agg(
        pl.count().alias('count')
    ).sort('medication_type')
    print("\nRows by medication type:")
    print(type_counts)

    # Find time columns
    time_cols = [col for col in matrix_df.columns if col.startswith('T_')]
    print(f"\nTime intervals: {len(time_cols)}")

    # Check for any non-zero values
    if time_cols:
        # Sample a few time columns to check activity
        sample_cols = time_cols[::len(time_cols) // 10] if len(time_cols) > 10 else time_cols
        for col in sample_cols[:3]:
            non_zero = matrix_df.filter(pl.col(col) > 0).height
            print(f"  {col}: {non_zero} rows with active infusions")

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


def debug_specific_infusion(meds_matrix,
                            parquet_path='../../data/treated_patients_inputevents.parquet',
                            itemid=225158,
                            stay_id=32128372,
                            output_file='debug.csv'):
    """
    Extract and examine all records for a specific itemid and stay_id
    """
    meds_matrix = meds_matrix.filter(pl.col("stay_id") == stay_id)
    meds_matrix.write_csv("debug_meds_matrix.csv")
    # Read the parquet file
    df = pd.read_parquet(parquet_path)

    # Filter for the specific itemid and stay_id
    filtered = df[(df['itemid'] == itemid) & (df['stay_id'] == stay_id)].copy()

    # Sort by starttime to see the sequence
    filtered = filtered.sort_values('starttime')

    # Print summary
    print(f"Found {len(filtered)} records for itemid={itemid}, stay_id={stay_id}")
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

    return filtered


def extract_trajectories_from_patient(
        values_array, mask_array, abs_time_array, rel_time_array, t0_trigger_idx
):
    """
    Split patient data into trajectories based on t0 triggers.
    ALL trajectories start from ICU admission (index 0).
    Each trajectory ends at a t0 trigger.

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
        # Create trajectories from start (0) to each t0
        for t0_idx in t0_indices:
            trajectories.append((0, t0_idx + 1))

    return trajectories


def save_trajectory_tensor(
        values_array, mask_array, abs_time_array, rel_time_array,
        start_idx, end_idx, stay_id, traj_num, cache_dir
):
    """
    Save a single trajectory as a tensor file.

    Note: All trajectories start from ICU admission (time 0), so no time adjustment needed.
    """
    # Extract trajectory slice
    traj_values = values_array[start_idx:end_idx, :]
    traj_mask = mask_array[start_idx:end_idx, :]
    traj_abs_time = abs_time_array[start_idx:end_idx]
    traj_rel_time = rel_time_array[start_idx:end_idx]

    # No time adjustment needed - all trajectories start from ICU admission

    # Length is the number of time points in this trajectory
    length = end_idx - start_idx

    # Convert to tensors
    values_tensor = torch.from_numpy(traj_values).float()
    mask_tensor = torch.from_numpy(traj_mask).float()
    abs_time_tensor = torch.from_numpy(traj_abs_time).float()
    rel_time_tensor = torch.from_numpy(traj_rel_time).float()

    # Save tensor
    file_path = os.path.join(cache_dir, f"med_tensor_{int(stay_id)}_traj_{traj_num:03d}.pt")
    torch.save(
        (values_tensor, mask_tensor, abs_time_tensor, rel_time_tensor, length),
        file_path
    )

    return file_path


def process_single_patient_medications(
        stay_id,
        med_df_patient,
        medication_info,
        n_intervals,
        interval_minutes,
        icu_admission_time,
        cache_dir
):
    """
    Process medications for a single patient and save as trajectory tensors.

    Each trajectory starts from ICU admission and extends to a t0 trigger.
    This creates cumulative trajectories with increasing lengths.

    Mask logic:
    - For medications and crystalloid sum: mask=1 if value!=0, mask=0 if value==0
    - For t0_trigger: mask=1 always (all time points are considered measured)

    Time arrays:
    - Both abs_time_tensor and rel_time_tensor represent hours from ICU admission
    - They are identical (included for compatibility with tensor format)

    Returns:
        Tuple of (stay_id, trajectory_info) where trajectory_info is a list of
        dictionaries containing trajectory metadata
    """
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

    # Extract trajectories based on t0 triggers
    trajectories = extract_trajectories_from_patient(
        values_array, mask_array, abs_time_array, rel_time_array, t0_trigger_idx
    )

    # Save each trajectory
    trajectory_info = []
    for traj_num, (start_idx, end_idx) in enumerate(trajectories):
        file_path = save_trajectory_tensor(
            values_array, mask_array, abs_time_array, rel_time_array,
            start_idx, end_idx, stay_id, traj_num, cache_dir
        )

        # Calculate trajectory metadata
        traj_length = end_idx - start_idx
        has_t0_at_end = (end_idx > 0 and
                         end_idx <= n_intervals and
                         values_array[end_idx - 1, t0_trigger_idx] == 1)

        trajectory_info.append({
            'stay_id': stay_id,
            'trajectory_num': traj_num,
            'start_idx': start_idx,
            'end_idx': end_idx,
            'length': traj_length,
            'start_time_hours': 0.0,  # Always starts at ICU admission
            'end_time_hours': rel_time_array[end_idx - 1] if end_idx > 0 else 0,
            'has_t0_trigger': has_t0_at_end,
            'file_path': file_path
        })

    return (stay_id, trajectory_info)


def save_patient_data_as_csv(
        stay_id,
        med_df_patient,
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
    patient_dir = Path(output_dir) / f"patient_{stay_id}_inspection"
    patient_dir.mkdir(parents=True, exist_ok=True)

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
        'stay_id': [stay_id],
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

    print(f"\nSaved inspection CSVs for patient {stay_id} to: {patient_dir}")
    print(f"Files created:")
    print(f"  - values.csv: Medication rates over time")
    print(f"  - mask.csv: Measurement mask over time")
    print(f"  - time.csv: Time information")
    print(f"  - trajectories.csv: Trajectory definitions")
    print(f"  - summary.csv: Patient summary statistics")

    return patient_dir


def create_medication_tensors(
        inputevents_df,
        crystalloid_itemids,
        vasopressor_itemids,
        time_interval_minutes=5,  # Time interval in minutes (default 5)
        icustays_path='../../data/icustays.csv',
        los_data_path='../../data/icustays.csv',  # Path to CSV with length of stay data
        cache_dir='../../data/med_tensors',
        n_workers=4
):
    """
    Create medication trajectory tensors for each patient.

    Trajectories are defined as cumulative segments from ICU admission to each t0:
    - First trajectory: from ICU admission to first t0
    - Second trajectory: from ICU admission to second t0
    - Third trajectory: from ICU admission to third t0
    - etc.

    ALL trajectories start at time 0 (ICU admission) and include the full history.

    Special feature: If patient with stay_id 32128372 is found, detailed CSV files
    will be saved for inspection in {cache_dir}/patient_32128372_inspection/

    Parameters:
    - inputevents_df: Polars DataFrame with medication events
    - crystalloid_itemids: List of item IDs for crystalloid medications
    - vasopressor_itemids: List of item IDs for vasopressor medications
    - time_interval_minutes: Time interval in minutes between observations (default 5)
    - icustays_path: Path to ICU stays CSV file
    - los_data_path: Path to CSV file containing 'los' column with length of stay in days (e.g., 3.5 = 3.5 days)
    - cache_dir: Directory to save tensor files
    - n_workers: Number of parallel workers (1 for sequential processing)

    Returns:
    - Dictionary containing trajectory information for all patients
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

    # Load ICU stays data
    print("\nLoading ICU stays data...")
    icustays_df = pl.read_csv(icustays_path)

    # Parse datetime columns
    if icustays_df.schema['intime'] != pl.Datetime:
        icustays_df = icustays_df.with_columns(
            pl.col('intime').str.to_datetime()
        )

    # Use the time interval parameter directly
    interval_minutes = time_interval_minutes
    n_intervals = max_intervals
    total_hours = (n_intervals * interval_minutes) / 60
    total_days = total_hours / 24

    print(f"Using {n_intervals} intervals of {interval_minutes} minutes each")
    print(f"Total time span: {total_hours:.1f} hours ({total_days:.1f} days)")

    # Get all unique stay_ids
    all_stay_ids = inputevents_df['stay_id'].unique().to_list()
    print(f"Total patients: {len(all_stay_ids)}")

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

    # Filter and prepare medication data
    print("Processing medication events...")

    # Ensure datetime columns are parsed
    if inputevents_df.schema.get('starttime') != pl.Datetime:
        inputevents_df = inputevents_df.with_columns([
            pl.col('starttime').str.to_datetime(),
            pl.col('endtime').str.to_datetime()
        ])

    # Filter for relevant medications
    med_df = inputevents_df.filter(
        pl.col('itemid').is_in(all_meds) &
        pl.col('rate').is_not_null() &
        pl.col('stay_id').is_in(all_stay_ids)
    )

    # Check for missing endtimes
    missing_endtimes = med_df.filter(pl.col('endtime').is_null()).height
    if missing_endtimes > 0:
        raise ValueError(f"Found {missing_endtimes} infusion events with missing endtimes.")

    # Join with ICU admission times
    med_df = med_df.join(
        icustays_df.select(['stay_id', 'intime']),
        on='stay_id',
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

    # Create stay_id to admission time mapping
    stay_admission_map = {}
    for row in icustays_df.select(['stay_id', 'intime']).iter_rows(named=True):
        stay_admission_map[row['stay_id']] = row['intime']

    # Prepare for parallel processing
    process_func = partial(
        process_single_patient_medications,
        medication_info=medication_info,
        n_intervals=n_intervals,
        interval_minutes=interval_minutes,
        cache_dir=cache_dir
    )

    all_trajectory_info = {}

    # Process each patient
    if n_workers > 1:
        # Parallel processing
        with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {}

            for stay_id in all_stay_ids:
                # Get data for this patient
                patient_med_df = med_df.filter(pl.col('stay_id') == stay_id)
                admission_time = stay_admission_map.get(stay_id)

                if admission_time is not None:
                    future = executor.submit(
                        process_func,
                        stay_id,
                        patient_med_df,
                        icu_admission_time=admission_time
                    )
                    futures[future] = stay_id

            # Collect results
            for future in tqdm(concurrent.futures.as_completed(futures),
                               total=len(futures), desc="Processing patients"):
                stay_id = futures[future]
                try:
                    result = future.result()
                    all_trajectory_info[result[0]] = result[1]

                    # Save inspection CSVs for specific patient
                    if int(stay_id) == 32128372:
                        # Need to re-get the patient data since we're in the main thread now
                        patient_med_df = med_df.filter(pl.col('stay_id') == stay_id)
                        admission_time = stay_admission_map.get(stay_id)
                        if admission_time is not None:
                            save_patient_data_as_csv(
                                stay_id=stay_id,
                                med_df_patient=patient_med_df,
                                medication_info=medication_info,
                                n_intervals=n_intervals,
                                interval_minutes=interval_minutes,
                                icu_admission_time=admission_time,
                                output_dir=cache_dir
                            )
                except Exception as exc:
                    print(f'Stay ID {stay_id} generated an exception: {exc}')
    else:
        # Sequential processing
        for stay_id in tqdm(all_stay_ids, desc="Processing patients"):
            patient_med_df = med_df.filter(pl.col('stay_id') == stay_id)
            admission_time = stay_admission_map.get(stay_id)

            if admission_time is not None:
                result = process_func(
                    stay_id,
                    patient_med_df,
                    icu_admission_time=admission_time
                )
                all_trajectory_info[result[0]] = result[1]

    # Calculate summary statistics
    total_trajectories = sum(len(traj_list) for traj_list in all_trajectory_info.values())
    trajectories_per_patient = [len(traj_list) for traj_list in all_trajectory_info.values()]

    print(f"\nProcessing complete:")
    print(f"Total patients processed: {len(all_trajectory_info)}")
    print(f"Total trajectories created: {total_trajectories}")
    print(f"Average trajectories per patient: {np.mean(trajectories_per_patient):.2f}")
    print(f"Max trajectories for a patient: {np.max(trajectories_per_patient) if trajectories_per_patient else 0}")

    # Check if inspection patient was found
    inspection_patient_found = any(int(stay_id) == 32128372 for stay_id in all_trajectory_info.keys())
    if inspection_patient_found:
        print(f"\nInspection CSVs saved for patient 32128372 in {cache_dir}/patient_32128372_inspection/")
    else:
        print(f"\nNote: Patient 32128372 not found in the dataset")

    # Save trajectory metadata
    trajectory_metadata = {
        'all_trajectories': all_trajectory_info,
        'medication_info': medication_info,
        'n_intervals': n_intervals,
        'max_intervals': max_intervals,
        'interval_minutes': interval_minutes,
        'max_los_days': max_los_days,
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
    all_patients_chartevents_path = "../../data/treated_patients_chartevents.parquet"
    patient_metadata_path = "../../data/patients.csv"
    hr_params_path = "../../data/hr_params.csv"
    inputevents_path = "../../data/treated_patients_inputevents.parquet"

    # Define the lists (these should match what was used in process_mimic_data)
    crystalloids_list = [225158, 225159, 225161]  # NaCl 0.9%, 0.45%, 3%
    vasopressors_list = [221906, 229630, 229631, 229632, 221662]  # Add your vasopressor IDs here
    SV_list = [228375]  # Add your SV measurement IDs here

    all_patients_chartevents = find_relevant_patients(measurements=[220045, 220052, 220074, 220088, 224842, 228369, 229897, 228375], MAP_id=220052)
    hr_params = find_min_max_heartrates(all_patients_chartevents_path, metadata_path=patient_metadata_path, save_path=hr_params_path)

    inputevents = find_relevant_inputevents(all_patients_chartevents=all_patients_chartevents,
                                            save_path=inputevents_path,
                                            events=[225158,225159,225161,221906,229630,229631,229632,221662,221653,221986])


    meds_matrix = create_medication_tensors(inputevents_df=inputevents,
                                          crystalloid_itemids =crystalloids_list,
                                          vasopressor_itemids=vasopressors_list)

    #summarize_medication_matrix(meds_matrix)


    #results_df = pd.read_csv('../../data/mimics_stays.csv')



    # First, analyze data completeness
    #analyze_data_completeness(results_df, crystalloids_list, vasopressors_list, SV_list)

    # Detect t0 points and create training data
    #training_data = detect_t0(results_df, crystalloids_list, vasopressors_list)
    #df = debug_specific_infusion(meds_matrix)


if __name__ == "__main__":
    main()






