#!/usr/bin/env python3
"""
Simplified MIMIC-III Initial Condition Tensor Creation from Waveform Numerics
COMPLETELY REMOVED ALL QUALITY CHECKS
"""

import pandas as pd
import numpy as np
import torch
import pickle
import wfdb
import os
import requests
from bs4 import BeautifulSoup
from pathlib import Path
from datetime import datetime, timedelta
from tqdm import tqdm
import json
import time
from functools import lru_cache
import hashlib
import concurrent.futures

# --- Debug Setup ---
DEBUG = os.environ.get('DEBUG', 'False').lower() in ('true', '1', 't')

def debug_print(s):
    if DEBUG:
        print(s)
# --- End Debug Setup ---



def setup_wfdb_credentials():
    """Set up WFDB credentials for PhysioNet access"""
    if 'WFDB_USERNAME' not in os.environ or 'WFDB_PASSWORD' not in os.environ:
        print("⚠️  PhysioNet credentials not found!")
        print("Please set environment variables:")
        print("export WFDB_USERNAME='your_username'")
        print("export WFDB_PASSWORD='your_password'")
        return False

    wfdb.set_db_index_url('https://physionet.org/files/')
    print("✓ PhysioNet credentials configured")
    return True


def load_hadm_to_patient_mapping(relevant_patient_ids_path):
    """
    Load mapping from HADM_ID to patient ID

    Args:
        relevant_patient_ids_path (str): Path to relevant_patient_ids.csv

    Returns:
        dict: {hadm_id: subject_id}
    """
    df = pd.read_csv(relevant_patient_ids_path)
    mapping = {}

    for _, row in df.iterrows():
        mapping[row['HADM_ID']] = row['SUBJECT_ID']

    print(f"Loaded mapping for {len(mapping)} patients")
    return mapping


def calculate_t0_timestamp(trajectory_info, icu_admission_time, interval_minutes):
    """
    Calculate exact t0 timestamp from trajectory metadata

    Args:
        trajectory_info: Trajectory metadata dict
        icu_admission_time: ICU admission datetime
        interval_minutes: Minutes per interval

    Returns:
        datetime: Exact t0 timestamp
    """
    # t0 is at the end of the trajectory (end_idx - 1)
    t0_interval_idx = trajectory_info['end_idx'] - 1

    # Convert interval index to minutes from admission
    t0_minutes_from_admission = t0_interval_idx * interval_minutes

    # Convert to regular Python int/float for timedelta
    t0_minutes_from_admission = float(t0_minutes_from_admission)

    # Calculate exact t0 timestamp
    t0_timestamp = icu_admission_time + timedelta(minutes=t0_minutes_from_admission)

    return t0_timestamp


def find_waveform_file_for_t0_local(patient_id, t0_timestamp, waveform_base_dir):
    """
    Find the appropriate waveform file for a given t0 timestamp using local files

    Strategy:
    1. Find all files ending with "n.hea" (numerics header files) in patient directory
    2. For each header file, read the metadata to get:
       - Start time (base_date + base_time)
       - Duration (number of samples / sampling frequency)
    3. Calculate end time = start time + duration
    4. Check if t_0 falls within [start_time, end_time]
    5. Return the record that contains t_0

    Args:
        patient_id: MIMIC patient ID (SUBJECT_ID)
        t0_timestamp: Target timestamp
        waveform_base_dir: Base directory containing waveform files

    Returns:
        tuple: (patient_path, record_name) or (None, None)
    """
    # Convert patient_id to waveform patient path format
    patient_id_str = f"p{patient_id:06d}"
    # Top-level directory uses first two digits of the patient record (including leading zeros)
    patient_subdir = f"p{patient_id_str[1:3]}"  # First 2 digits after 'p'
    patient_path = f"{patient_subdir}/{patient_id_str}"

    # Look for numerics header files in the patient directory
    patient_dir = Path(waveform_base_dir) / patient_subdir / patient_id_str

    if not patient_dir.exists():
        print(f"  Patient directory not found: {patient_dir}")
        return None, None

    # Find all numerics header files (ending with "n.hea")
    numerics_headers = list(patient_dir.glob("*n.hea"))

    if not numerics_headers:
        print(f"  No numerics header files (*n.hea) found in {patient_dir}")
        return None, None

    print(f"  Found {len(numerics_headers)} numerics files, checking which contains t_0...")

    for header_file in numerics_headers:
        try:
            # Read header to get metadata
            record_name = header_file.stem
            record = wfdb.rdheader(str(header_file.with_suffix('')))

            # Extract start time from header
            if hasattr(record, 'base_date') and record.base_date is not None:
                if hasattr(record, 'base_time') and record.base_time is not None:
                    record_start = datetime.combine(record.base_date, record.base_time)
                else:
                    record_start = datetime.combine(record.base_date, datetime.min.time())
            else:
                print(f"    {record_name}: No datetime info in header")
                continue

            # Calculate record duration and end time
            if record.fs > 0 and record.sig_len > 0:
                duration_seconds = record.sig_len / record.fs
                record_end = record_start + timedelta(seconds=duration_seconds)
            else:
                print(f"    {record_name}: Invalid sampling info (fs={record.fs}, len={record.sig_len})")
                continue

            print(f"    {record_name}: {record_start} to {record_end} (duration: {duration_seconds / 3600:.1f}h)")

            # Check if t_0 falls within this record's time interval
            if record_start <= t0_timestamp <= record_end:
                print(f"    ✓ t_0 ({t0_timestamp}) falls within {record_name}")
                return patient_path, record_name
            else:
                print(f"    ✗ t_0 outside this record")

        except Exception as e:
            print(f"    Error reading header {header_file}: {e}")
            continue

    print(f"  No numerics record found that contains t_0: {t0_timestamp}")
    return None, None


def find_waveform_file_for_t0_remote_with_discovery(patient_id, t0_timestamp, database='mimic3wdb-matched'):
    """
    Find the appropriate waveform file for t0 using remote access

    Strategy:
    1. Access patient directory URL directly
    2. Look for all files ending with "n.hea" (numerics files)
    3. For each numerics file, read header and check if t_0 falls within time interval
    4. Return the record containing t_0

    Args:
        patient_id: MIMIC patient ID (SUBJECT_ID)
        t0_timestamp: Target timestamp
        database: PhysioNet database name

    Returns:
        tuple: (patient_path, record_name) or (None, None)
    """
    patient_id_str = f"p{patient_id:06d}"
    # Top-level directory uses first two digits of the patient record (including leading zeros)
    patient_subdir = f"p{patient_id_str[1:3]}"  # First 2 digits after 'p'
    patient_path = f"{patient_subdir}/{patient_id_str}"

    # Construct the patient directory URL
    patient_url = f"https://physionet.org/files/{database}/1.0/{patient_path}/"

    print(f"  Accessing patient directory: {patient_url}")

    try:
        # Get the directory listing
        response = requests.get(patient_url)

        # Check for 404 - patient directory doesn't exist
        if response.status_code == 404:
            print(f"  ✗ Patient directory not found (404) - no waveform data for patient {patient_id}")
            return None, None

        # Raise for other HTTP errors
        response.raise_for_status()

        # Parse the HTML to find files ending with "n.hea"
        soup = BeautifulSoup(response.content, 'html.parser')

        # Find all links that end with "n.hea"
        numerics_files = []
        for link in soup.find_all('a'):
            href = link.get('href', '')
            if href.endswith('n.hea'):
                numerics_files.append(href)

        print(f"  Found {len(numerics_files)} numerics files: {numerics_files}")

        if not numerics_files:
            print(f"  No numerics files (*n.hea) found in directory")
            return None, None

        # Check each numerics file to see if it contains t_0
        full_pn_dir = f"{database}/1.0/{patient_path}"

        for hea_file in numerics_files:
            record_name = hea_file.replace('.hea', '')  # Remove .hea extension

            try:
                # Read header to get timing info
                print(f"    Checking {record_name}...")
                record = wfdb.rdheader(record_name, pn_dir=full_pn_dir)

                # Extract start time from header
                if hasattr(record, 'base_date') and record.base_date is not None:
                    if hasattr(record, 'base_time') and record.base_time is not None:
                        record_start = datetime.combine(record.base_date, record.base_time)
                    else:
                        record_start = datetime.combine(record.base_date, datetime.min.time())
                else:
                    print(f"      No datetime info in header")
                    continue

                # Calculate record duration and end time
                if record.fs > 0 and record.sig_len > 0:
                    duration_seconds = record.sig_len / record.fs
                    record_end = record_start + timedelta(seconds=duration_seconds)
                else:
                    print(f"      Invalid sampling info (fs={record.fs}, len={record.sig_len})")
                    continue

                print(f"      {record_start} to {record_end} (duration: {duration_seconds / 3600:.1f}h)")

                # Check if t_0 falls within this record's time interval
                if record_start <= t0_timestamp <= record_end:
                    print(f"      ✓ t_0 ({t0_timestamp}) falls within {record_name}")
                    return patient_path, record_name
                else:
                    print(f"      ✗ t_0 outside this record")

            except Exception as e:
                print(f"      Error reading record {record_name}: {e}")
                continue

        print(f"  No numerics record found that contains t_0: {t0_timestamp}")
        return None, None

    except requests.exceptions.RequestException as e:
        if hasattr(e, 'response') and e.response is not None and e.response.status_code == 404:
            print(f"  ✗ Patient directory not found (404) - no waveform data for patient {patient_id}")
        else:
            print(f"  Error accessing patient directory {patient_url}: {e}")
        return None, None
    except Exception as e:
        print(f"  Error processing patient directory {patient_url}: {e}")
        return None, None


def extract_signal_by_name(record, signal_name):
    """Extract a specific signal by name from the record"""
    if record is None or record.p_signal is None:
        return None, None, None

    try:
        signal_index = record.sig_name.index(signal_name)
        signal_data = record.p_signal[:, signal_index]
        unit = record.units[signal_index]
        return signal_data, unit, record.fs
    except (ValueError, IndexError):
        return None, None, None


def extract_ic_from_waveform_file(patient_path, record_name, t0_timestamp,
                                  extraction_window_seconds=30, database='mimic3wdb-matched'):
    """
    Extract initial conditions from a specific waveform file - ZERO QUALITY CHECKS

    Strategy:
    1. Find t_0 sample index
    2. Read ±30 seconds around t_0
    3. If signal exists and has ANY finite values within window, use closest to t_0
    4. NO quality assessment whatsoever
    """
    required_signals = ['HR', 'ABP Mean', 'CVP']
    optional_signals = ['SV', 'CO']  # Add this line
    all_signals = required_signals + optional_signals

    try:
        # Read the waveform record from PhysioNet
        full_pn_dir = f"{database}/1.0/{patient_path}"
        record = wfdb.rdrecord(record_name, pn_dir=full_pn_dir)

        # Get record start time
        if hasattr(record, 'base_date') and record.base_date is not None:
            if hasattr(record, 'base_time') and record.base_time is not None:
                record_start = datetime.combine(record.base_date, record.base_time)
            else:
                record_start = datetime.combine(record.base_date, datetime.min.time())
        else:
            print(f"No timestamp info in record {record_name}")
            return None, None

        # Calculate time offset from record start to t0
        time_offset_seconds = (t0_timestamp - record_start).total_seconds()

        # Check if t0 is within the record
        record_duration = record.sig_len / record.fs if record.fs > 0 else 0
        if time_offset_seconds < 0 or time_offset_seconds > record_duration:
            print(f"t0 outside record range")
            return None, None

        # Calculate sample index for t0
        t0_sample_idx = int(time_offset_seconds * record.fs)

        # Define search window (±30 seconds)
        window_samples = int(extraction_window_seconds * record.fs)
        start_sample = max(0, t0_sample_idx - window_samples)
        end_sample = min(record.sig_len, t0_sample_idx + window_samples)

        samples_to_read = end_sample - start_sample
        print(f"  Reading {samples_to_read} samples around t_0")

        ic_values = {}
        ic_mask = {}

        # Extract each required signal
        for signal_name in all_signals:
            signal_data, unit, fs = extract_signal_by_name(record, signal_name)

            if signal_data is not None:
                # Extract window around t0
                segment = signal_data[start_sample:end_sample]

                # Find any finite values - NO QUALITY ASSESSMENT AT ALL
                valid_mask = np.isfinite(segment)
                valid_indices = np.where(valid_mask)[0]

                if len(valid_indices) > 0:
                    # Find closest valid sample to t_0
                    t0_relative_idx = t0_sample_idx - start_sample
                    closest_idx = valid_indices[np.argmin(np.abs(valid_indices - t0_relative_idx))]
                    closest_value = segment[closest_idx]

                    # Distance from t_0
                    distance_samples = abs(closest_idx - t0_relative_idx)
                    distance_seconds = distance_samples / record.fs if record.fs > 0 else 0

                    ic_values[signal_name] = float(closest_value)
                    ic_mask[signal_name] = 1.0

                    print(f"  ✓ {signal_name}: {closest_value:.2f} ({distance_seconds:.1f}s from t_0)")
                else:
                    ic_values[signal_name] = 0.0
                    ic_mask[signal_name] = 0.0
                    print(f"  ✗ {signal_name}: No data within ±{extraction_window_seconds}s")
            else:
                ic_values[signal_name] = 0.0
                ic_mask[signal_name] = 0.0
                print(f"  ✗ {signal_name}: Signal not found")

        # Check if all required signals have data
        missing_signals = [sig for sig in required_signals if ic_mask.get(sig, 0) == 0]

        if missing_signals:
            print(f"  Missing: {missing_signals}")
            return None, None

        if ic_mask.get('HR', 0) > 0 and ic_values.get('HR', 0) > 0:
            hr_value = ic_values['HR']

            # If we have CO but missing SV: calculate SV = CO / HR
            if (ic_mask.get('CO', 0) > 0 and ic_mask.get('SV', 0) == 0 and
                    ic_values.get('CO', 0) > 0):
                ic_values['SV'] = ic_values['CO'] / hr_value
                ic_mask['SV'] = 1.0
                print(f"  ✓ SV: {ic_values['SV']:.2f} (calculated from CO/HR)")

            # If we have SV but missing CO: calculate CO = SV * HR
            elif (ic_mask.get('SV', 0) > 0 and ic_mask.get('CO', 0) == 0 and
                  ic_values.get('SV', 0) > 0):
                ic_values['CO'] = ic_values['SV'] * hr_value
                ic_mask['CO'] = 1.0
                print(f"  ✓ CO: {ic_values['CO']:.2f} (calculated from SV*HR)")

        if (ic_mask.get('ABP Mean', 0) > 0 and ic_mask.get('CVP', 0) > 0 and
                ic_mask.get('CO', 0) > 0 and ic_values.get('CO', 0) > 0):
            ic_values['R_TPR'] = (ic_values['ABP Mean'] - ic_values['CVP']) / ic_values['CO']
            ic_mask['R_TPR'] = 1.0
            print(f"  ✓ R_TPR: {ic_values['R_TPR']:.2f} (calculated)")
        else:
            ic_values['R_TPR'] = 0.0
            ic_mask['R_TPR'] = 0.0
            print(f"  ✗ R_TPR: Cannot calculate (missing MAP/CVP/CO)")

        print(f"  ✓ All signals found!")
        return ic_values, ic_mask

    except Exception as e:
        print(f"Error reading record {record_name}: {e}")
        return None, None


def process_patient_simplified(
    hadm_id,
    trajectories,
    patient_mapping,
    icu_admission_map,
    interval_minutes,
    waveform_base_dir,
    database,
    cache_dir
):
    debug_print(f"[Worker PID: {os.getpid()}] Processing patient HADM_ID: {hadm_id}")
    if hadm_id not in patient_mapping or hadm_id not in icu_admission_map:
        debug_print(f"  [Worker PID: {os.getpid()}] Patient {hadm_id} missing from mapping or ICU stay info. Skipping.")
        return hadm_id, []

    patient_id = patient_mapping[hadm_id]
    icu_admission_time = icu_admission_map[hadm_id]
    patient_ics = []
    debug_print(f"  [Worker PID: {os.getpid()}] Found SUBJECT_ID: {patient_id}, ICU Admission: {icu_admission_time}")

    for i, traj in enumerate(trajectories):
        traj_num = traj['trajectory_num']
        debug_print(f"  [Worker PID: {os.getpid()}]  Processing trajectory {i+1}/{len(trajectories)} (traj_num: {traj_num})")
        
        t0_timestamp = calculate_t0_timestamp(traj, icu_admission_time, interval_minutes)
        debug_print(f"    [Worker PID: {os.getpid()}] Calculated t0 timestamp: {t0_timestamp}")

        if waveform_base_dir:
            debug_print(f"    [Worker PID: {os.getpid()}] Searching for local waveform file...")
            patient_path, record_name = find_waveform_file_for_t0_local(
                patient_id, t0_timestamp, waveform_base_dir
            )
        else:
            debug_print(f"    [Worker PID: {os.getpid()}] Searching for remote waveform file...")
            patient_path, record_name = find_waveform_file_for_t0_remote_with_discovery(
                patient_id, t0_timestamp, database
            )

        if patient_path is None or record_name is None:
            debug_print(f"    [Worker PID: {os.getpid()}] No suitable waveform file found. Skipping trajectory.")
            continue
        debug_print(f"    [Worker PID: {os.getpid()}] Found record: {patient_path}/{record_name}")

        debug_print(f"    [Worker PID: {os.getpid()}] Extracting ICs from waveform file...")
        ic_values, ic_mask = extract_ic_from_waveform_file(
            patient_path, record_name, t0_timestamp, database=database
        )

        if ic_values is None:
            debug_print(f"    [Worker PID: {os.getpid()}] IC extraction failed. Skipping trajectory.")
            continue
        debug_print(f"    [Worker PID: {os.getpid()}] Successfully extracted ICs: {ic_values}")

        physio_values = [
            ic_values['HR'], ic_values['ABP Mean'], ic_values['CVP'],
            ic_values.get('SV', 0.0), ic_values.get('R_TPR', 0.0)
        ]
        physio_masks = [
            ic_mask['HR'], ic_mask['ABP Mean'], ic_mask['CVP'],
            ic_mask.get('SV', 0.0), ic_mask.get('R_TPR', 0.0)
        ]
        ic_tensor = torch.tensor(physio_values, dtype=torch.float32)
        ic_mask_tensor = torch.tensor(physio_masks, dtype=torch.float32)

        ic_file_path = Path(cache_dir) / f"ic_tensor_{int(hadm_id)}_traj_{traj_num:03d}.pt"
        debug_print(f"    [Worker PID: {os.getpid()}] Saving tensor to {ic_file_path}")
        torch.save((ic_tensor, ic_mask_tensor), ic_file_path)

        patient_ics.append({
            'hadm_id': hadm_id, 'trajectory_num': traj_num, 't0_timestamp': t0_timestamp,
            'file_path': str(ic_file_path), 'ic_values': ic_values, 'record_used': f"{patient_path}/{record_name}"
        })

    debug_print(f"  [Worker PID: {os.getpid()}] Finished processing patient {hadm_id}. Found {len(patient_ics)} valid trajectories.")
    return hadm_id, patient_ics



def create_ic_tensors_simplified(
        medication_trajectory_metadata_path,
        relevant_patient_ids_path,
        icustays_path,
        waveform_base_dir=None,
        cache_dir='initial_conditions_waveform',
        database='mimic3wdb-matched',
        skip_existing=True,
        n_workers=4
):
    """
    Simplified IC tensor creation from MIMIC-III waveform data WITH CACHING & PARALLEL PROCESSING
    """
    debug_print("--- Starting Simplified IC Tensor Creation ---")
    if not setup_wfdb_credentials():
        return None

    Path(cache_dir).mkdir(parents=True, exist_ok=True)

    debug_print(f"Loading trajectory metadata from: {medication_trajectory_metadata_path}")
    with open(medication_trajectory_metadata_path, 'rb') as f:
        med_metadata = pickle.load(f)
    all_trajectories = med_metadata['all_trajectories']
    interval_minutes = med_metadata['interval_minutes']
    debug_print(f"Loaded metadata for {len(all_trajectories)} patients.")

    debug_print(f"Loading patient mapping from: {relevant_patient_ids_path}")
    patient_mapping = load_hadm_to_patient_mapping(relevant_patient_ids_path)

    debug_print(f"Loading ICU stays from: {icustays_path}")
    import polars as pl
    icustays_df = pl.read_csv(icustays_path)
    if icustays_df.schema['INTIME'] != pl.Datetime:
        icustays_df = icustays_df.with_columns(
            pl.col('INTIME').str.to_datetime()
        )
    icu_admission_map = {row['HADM_ID']: row['INTIME'] for row in icustays_df.select(['HADM_ID', 'INTIME']).iter_rows(named=True)}

    all_initial_conditions = {}
    tasks = []
    skipped_patients = 0

    debug_print("Preparing tasks for parallel processing...")
    for hadm_id, trajectories in all_trajectories.items():
        if skip_existing:
            all_files_exist = True
            for traj in trajectories:
                traj_num = traj['trajectory_num']
                ic_file_path = Path(cache_dir) / f"ic_tensor_{int(hadm_id)}_traj_{traj_num:03d}.pt"
                if not ic_file_path.exists():
                    all_files_exist = False
                    break
            if all_files_exist:
                debug_print(f"Skipping patient {hadm_id} - all output files already exist.")
                skipped_patients += 1
                continue
        
        tasks.append(
            (hadm_id, trajectories, patient_mapping, icu_admission_map, interval_minutes, waveform_base_dir, database, cache_dir)
        )
    
    debug_print(f"Submitting {len(tasks)} tasks to {n_workers} workers.")
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
        future_to_hadm = {executor.submit(process_patient_simplified, *task): task[0] for task in tasks}

        for future in tqdm(concurrent.futures.as_completed(future_to_hadm), total=len(tasks), desc="Processing patients"):
            hadm_id, patient_ics = future.result()
            if patient_ics:
                all_initial_conditions[hadm_id] = patient_ics

    total_ics_created = sum(len(ics) for ics in all_initial_conditions.values())
    
    metadata_file = Path(cache_dir) / "ic_metadata.pkl"
    debug_print(f"Saving final metadata to {metadata_file}...")
    ic_metadata = {
        'all_initial_conditions': all_initial_conditions,
        'total_ics': total_ics_created,
        'parameters': ['HR', 'ABP Mean', 'CVP', 'SV', 'CO'],
        'extraction_method': 'waveform_numerics_simplified_parallel',
        'patients_processed': len(all_initial_conditions),
        'patients_skipped': skipped_patients
    }

    with open(metadata_file, "wb") as f:
        pickle.dump(ic_metadata, f)

    print(f"\n✅ FINAL RESULTS:")
    print(f"  Patients with new ICs created: {len(all_initial_conditions)}")
    print(f"  Patients skipped (already processed): {skipped_patients}")
    print(f"  Total IC tensors: {total_ics_created}")
    print(f"  Metadata saved to: {metadata_file}")

    return all_initial_conditions


# Example usage
if __name__ == "__main__":
    # Set up paths
    trajectory_metadata_path = "../../data/mimic3refactor/processed_data/med_tensors/trajectory_metadata.pkl"
    relevant_patient_ids_path = "../../data/mimic3refactor/processed_data/relevant_patient_ids.csv"
    icustays_path = "../../data/mimic3refactor/input_data/ICUSTAYS.csv"

    # For local files: set path to your local waveform directory
    # For remote access: set waveform_base_dir=None (will browse PhysioNet URLs)
    waveform_base_dir = None  # Use remote access - browse patient directories for *n.hea files

    # Create IC tensors
    ic_results = create_ic_tensors_simplified(
        medication_trajectory_metadata_path=trajectory_metadata_path,
        relevant_patient_ids_path=relevant_patient_ids_path,
        icustays_path=icustays_path,
        waveform_base_dir=waveform_base_dir,  # None = remote access
        cache_dir="../../data/mimic3refactor/processed_data/initial_conditions",
        n_workers=8  # Set the number of parallel workers
    )