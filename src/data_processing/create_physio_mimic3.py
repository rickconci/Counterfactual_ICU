#!/usr/bin/env python3
"""
Efficient MIMIC-III Waveform Tensor Creation with Minimized Remote Access
Process all trajectories per patient in one batch to minimize network calls
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
from scipy.interpolate import interp1d
import warnings
from collections import defaultdict
import concurrent.futures

# --- Debug Setup ---
DEBUG = os.environ.get('DEBUG', 'False').lower() in ('true', '1', 't')

def debug_print(s):
    if DEBUG:
        print(s)
# --- End Debug Setup ---


warnings.filterwarnings('ignore')


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


def load_trajectory_metadata(trajectory_metadata_path, ic_metadata_path):
    """Load trajectory and initial condition metadata"""
    print("Loading trajectory metadata...")
    with open(trajectory_metadata_path, 'rb') as f:
        traj_metadata = pickle.load(f)

    print("Loading initial condition metadata...")
    with open(ic_metadata_path, 'rb') as f:
        ic_metadata = pickle.load(f)

    return traj_metadata, ic_metadata


def get_patient_mapping(relevant_patient_ids_path):
    """Load mapping from HADM_ID to SUBJECT_ID"""
    df = pd.read_csv(relevant_patient_ids_path)
    mapping = {}
    for _, row in df.iterrows():
        mapping[row['HADM_ID']] = row['SUBJECT_ID']
    return mapping


def get_icu_admission_times(icustays_path):
    """Load ICU admission times mapping"""
    import polars as pl
    icustays_df = pl.read_csv(icustays_path)
    if icustays_df.schema['INTIME'] != pl.Datetime:
        icustays_df = icustays_df.with_columns(
            pl.col('INTIME').str.to_datetime()
        )

    icu_admission_map = {}
    for row in icustays_df.select(['HADM_ID', 'INTIME']).iter_rows(named=True):
        icu_admission_map[row['HADM_ID']] = row['INTIME']

    return icu_admission_map


class WaveformRecordCache:
    """Cache waveform record metadata to minimize remote access"""

    def __init__(self):
        self.patient_records = {}  # {patient_id: [record_info, ...]}
        self.record_data = {}  # {(patient_path, record_name): record_data}
        self.header_cache = {}  # {(patient_path, record_name): wfdb_record_header}

    def discover_patient_records(self, patient_id, waveform_base_dir=None, database='mimic3wdb-matched'):
        """Discover all waveform records for a patient (ONE network call per patient)"""
        if patient_id in self.patient_records:
            return self.patient_records[patient_id]

        # Convert patient_id to waveform patient path format
        patient_id_str = f"p{patient_id:06d}"
        patient_subdir = f"p{patient_id_str[1:3]}"
        patient_path = f"{patient_subdir}/{patient_id_str}"

        records = []

        if waveform_base_dir:
            # Local access
            patient_dir = Path(waveform_base_dir) / patient_subdir / patient_id_str
            if not patient_dir.exists():
                self.patient_records[patient_id] = []
                return []

            numerics_headers = list(patient_dir.glob("*n.hea"))

            for header_file in numerics_headers:
                try:
                    record_name = header_file.stem
                    cache_key = (patient_path, record_name)

                    # Check cache first
                    if cache_key in self.header_cache:
                        record = self.header_cache[cache_key]
                    else:
                        record = wfdb.rdheader(str(header_file.with_suffix('')))
                        self.header_cache[cache_key] = record

                    if record is not None:
                        record_info = self._extract_record_info(record, patient_path, record_name)
                        if record_info:
                            records.append(record_info)

                except Exception as e:
                    print(f"    Warning: Could not read header {header_file}: {e}")
                    continue
        else:
            # Remote access - SINGLE network call to get directory listing
            patient_url = f"https://physionet.org/files/{database}/1.0/{patient_path}/"

            try:
                response = requests.get(patient_url)
                if response.status_code == 404:
                    self.patient_records[patient_id] = []
                    return []

                response.raise_for_status()
                soup = BeautifulSoup(response.content, 'html.parser')
                numerics_files = [link.get('href', '') for link in soup.find_all('a')
                                  if link.get('href', '').endswith('n.hea')]

                # Read headers for all numerics files (cached)
                full_pn_dir = f"{database}/1.0/{patient_path}"

                for hea_file in numerics_files:
                    record_name = hea_file.replace('.hea', '')

                    try:
                        record = self._get_cached_header(record_name, full_pn_dir, patient_path)
                        if record is not None:
                            record_info = self._extract_record_info(record, patient_path, record_name)
                            if record_info:
                                records.append(record_info)

                    except Exception as e:
                        print(f"    Warning: Could not read header {record_name}: {e}")
                        continue

            except requests.exceptions.RequestException as e:
                print(f"    Error accessing patient directory {patient_url}: {e}")
                self.patient_records[patient_id] = []
                return []

        # Sort records by start time
        records.sort(key=lambda x: x['start_time'])
        self.patient_records[patient_id] = records

        print(f"    Discovered {len(records)} waveform records for patient {patient_id}")
        return records

    def _get_cached_header(self, record_name, full_pn_dir, patient_path):
        """Get header with caching to avoid redundant network calls"""
        cache_key = (patient_path, record_name)

        if cache_key in self.header_cache:
            return self.header_cache[cache_key]

        try:
            record = wfdb.rdheader(record_name, pn_dir=full_pn_dir)
            self.header_cache[cache_key] = record
            return record
        except Exception:
            self.header_cache[cache_key] = None
            return None

    def _extract_record_info(self, record, patient_path, record_name):
        """Extract timing and signal info from a record header"""
        try:
            if hasattr(record, 'base_date') and record.base_date is not None:
                if hasattr(record, 'base_time') and record.base_time is not None:
                    start_time = datetime.combine(record.base_date, record.base_time)
                else:
                    start_time = datetime.combine(record.base_date, datetime.min.time())
            else:
                return None

            if record.fs > 0 and record.sig_len > 0:
                duration_seconds = record.sig_len / record.fs
                end_time = start_time + timedelta(seconds=duration_seconds)
            else:
                return None

            return {
                'patient_path': patient_path,
                'record_name': record_name,
                'start_time': start_time,
                'end_time': end_time,
                'duration_seconds': duration_seconds,
                'sampling_rate': record.fs,
                'signal_names': record.sig_name,
                'n_samples': record.sig_len
            }

        except Exception:
            return None

    def find_records_for_timerange(self, patient_id, start_time, end_time):
        """Find all records that overlap with the given time range"""
        records = self.patient_records.get(patient_id, [])
        overlapping = []

        for record_info in records:
            # Check for overlap: record_start <= end_time AND record_end >= start_time
            if (record_info['start_time'] <= end_time and
                    record_info['end_time'] >= start_time):
                overlapping.append(record_info)

        return overlapping

    def load_record_data(self, record_info, database='mimic3wdb-matched'):
        """Load actual waveform data for a record (cached)"""
        cache_key = (record_info['patient_path'], record_info['record_name'])

        if cache_key in self.record_data:
            return self.record_data[cache_key]

        try:
            full_pn_dir = f"{database}/1.0/{record_info['patient_path']}"
            record = wfdb.rdrecord(record_info['record_name'], pn_dir=full_pn_dir)

            self.record_data[cache_key] = {
                'record': record,
                'info': record_info
            }

            print(f"      Loaded record {record_info['record_name']} "
                  f"({record_info['duration_seconds'] / 3600:.1f}h, {record.fs}Hz)")

            return self.record_data[cache_key]

        except Exception as e:
            print(f"      Error loading record {record_info['record_name']}: {e}")
            return None


def calculate_patient_time_requirements(hadm_id, patient_trajectories, patient_ics,
                                        icu_admission_time, interval_minutes, prediction_minutes):
    """Calculate all time ranges needed for a patient's trajectories"""
    time_requirements = []

    for ic_info in patient_ics:
        traj_num = ic_info['trajectory_num']

        # Find corresponding trajectory info
        trajectory_info = None
        for traj in patient_trajectories:
            if traj['trajectory_num'] == traj_num:
                trajectory_info = traj
                break

        if trajectory_info is None:
            continue

        # Calculate trajectory time range (for p_tensor)
        traj_start_minutes = trajectory_info['start_idx'] * interval_minutes
        traj_end_minutes = trajectory_info['end_idx'] * interval_minutes
        traj_start_time = icu_admission_time + timedelta(minutes=traj_start_minutes)
        traj_end_time = icu_admission_time + timedelta(minutes=traj_end_minutes)

        # Calculate t_0 and prediction time range
        t0_minutes = (trajectory_info['end_idx'] - 1) * interval_minutes
        t0_time = icu_admission_time + timedelta(minutes=t0_minutes)
        prediction_end_time = t0_time + timedelta(minutes=prediction_minutes)

        time_requirements.append({
            'trajectory_num': traj_num,
            'trajectory_info': trajectory_info,
            'p_tensor_range': (traj_start_time, traj_end_time),
            'prediction_range': (t0_time, prediction_end_time),
            't0_time': t0_time
        })

    return time_requirements


def extract_signals_from_record(record_data, start_time, end_time, target_signals):
    """Extract specific signals from a loaded record for given time range"""
    if record_data is None:
        return None

    record = record_data['record']
    record_info = record_data['info']
    record_start = record_info['start_time']

    # Calculate sample indices
    start_offset = max(0, (start_time - record_start).total_seconds())
    end_offset = min(record_info['duration_seconds'], (end_time - record_start).total_seconds())

    if start_offset >= end_offset:
        return None

    start_sample = max(0, int(start_offset * record.fs))
    end_sample = min(record.sig_len, int(end_offset * record.fs))

    if start_sample >= end_sample:
        return None

    # Extract signals
    signals_data = {}
    base_time_seconds = start_offset  # Time of first sample relative to record start

    for signal_name in target_signals:
        try:
            signal_index = record.sig_name.index(signal_name)
            signal_values = record.p_signal[start_sample:end_sample, signal_index]

            # Create time array for this signal
            n_samples = len(signal_values)
            signal_times = base_time_seconds + np.arange(n_samples) / record.fs

            # Remove invalid values
            valid_mask = np.isfinite(signal_values)
            if np.any(valid_mask):
                signals_data[signal_name] = {
                    'times': signal_times[valid_mask],
                    'values': signal_values[valid_mask]
                }

        except (ValueError, IndexError):
            continue

    return signals_data


def interpolate_signals_to_grid(signals_data, target_times_seconds):
    """Interpolate signals to target time points"""
    interpolated = {}

    for signal_name, signal_info in signals_data.items():
        times = signal_info['times']
        values = signal_info['values']

        if len(times) < 2:
            continue

        try:
            # Remove duplicates and sort
            unique_indices = np.unique(times, return_index=True)[1]
            unique_times = times[unique_indices]
            unique_values = values[unique_indices]

            if len(unique_times) < 2:
                continue

            # Create interpolator
            interpolator = interp1d(unique_times, unique_values, kind='linear',
                                    bounds_error=False, fill_value=np.nan)

            # Interpolate
            interpolated_values = interpolator(target_times_seconds)
            interpolated[signal_name] = interpolated_values

        except Exception:
            continue

    return interpolated


def process_patient_waveforms(hadm_id, patient_id, patient_trajectories, patient_ics,
                              icu_admission_time, interval_minutes, prediction_minutes,
                              target_interval_seconds, max_interval_seconds,
                              waveform_cache, database='mimic3wdb-matched'):
    """
    FIXED: Process trajectories, but SKIP those that don't fit entirely in a single record
    """

    print(f"\nProcessing patient {patient_id} (HADM {hadm_id}) - {len(patient_ics)} trajectories")

    # Step 1: Discover all waveform records for this patient (ONE network call)
    patient_records = waveform_cache.discover_patient_records(patient_id, database=database)

    if not patient_records:
        print(f"  No waveform records found for patient {patient_id}")
        return [], []

    # Step 2: Calculate time requirements for all trajectories
    time_requirements = calculate_patient_time_requirements(
        hadm_id, patient_trajectories, patient_ics, icu_admission_time,
        interval_minutes, prediction_minutes
    )

    print(f"  Processing {len(time_requirements)} trajectories with {len(patient_records)} available records")

    patient_p_tensors = []
    patient_predictions = []
    skipped_multi_record = 0
    skipped_no_record = 0

    # Step 3: Process each trajectory, but SKIP if it spans multiple records
    for req in time_requirements:
        traj_num = req['trajectory_num']
        trajectory_info = req['trajectory_info']

        try:
            # Find all records that overlap with this trajectory's p_tensor period
            p_records = waveform_cache.find_records_for_timerange(
                patient_id, req['p_tensor_range'][0], req['p_tensor_range'][1]
            )

            # Find all records that overlap with this trajectory's prediction period
            pred_records = waveform_cache.find_records_for_timerange(
                patient_id, req['prediction_range'][0], req['prediction_range'][1]
            )

            # SKIP if trajectory spans multiple records for p_tensor
            if len(p_records) > 1:
                print(f"    ✗ Skipping trajectory {traj_num} - spans {len(p_records)} records for p_tensor")
                skipped_multi_record += 1
                continue
            elif len(p_records) == 0:
                print(f"    ✗ Skipping trajectory {traj_num} - no records cover p_tensor period")
                skipped_no_record += 1
                continue

            # SKIP if trajectory spans multiple records for predictions
            valid_pred_records = []
            for record_info in pred_records:
                if record_info['sampling_rate'] > 0:
                    actual_interval = 1.0 / record_info['sampling_rate']
                    if actual_interval <= max_interval_seconds:
                        valid_pred_records.append(record_info)

            if len(valid_pred_records) > 1:
                print(f"    ✗ Skipping trajectory {traj_num} - spans {len(valid_pred_records)} records for predictions")
                skipped_multi_record += 1
                continue
            elif len(valid_pred_records) == 0:
                print(f"    ✗ Skipping trajectory {traj_num} - no valid records for predictions")
                skipped_no_record += 1
                continue

            # CHECK: Ensure the single p_record FULLY contains the trajectory
            p_record = p_records[0]
            if not (p_record['start_time'] <= req['p_tensor_range'][0] and
                    p_record['end_time'] >= req['p_tensor_range'][1]):
                print(f"    ✗ Skipping trajectory {traj_num} - record doesn't fully contain trajectory")
                print(f"      Record: {p_record['start_time']} to {p_record['end_time']}")
                print(f"      Trajectory: {req['p_tensor_range'][0]} to {req['p_tensor_range'][1]}")
                skipped_multi_record += 1
                continue

            # CHECK: Ensure the single pred_record FULLY contains the prediction period
            pred_record = valid_pred_records[0]
            if not (pred_record['start_time'] <= req['prediction_range'][0] and
                    pred_record['end_time'] >= req['prediction_range'][1]):
                print(f"    ✗ Skipping trajectory {traj_num} - record doesn't fully contain prediction period")
                skipped_multi_record += 1
                continue

            print(f"    ✓ Processing trajectory {traj_num} - fits in single records")

            # Create p_tensor from the single overlapping record
            record_data = waveform_cache.load_record_data(p_record, database)
            if record_data is None:
                print(f"    ✗ Failed to load record data for trajectory {traj_num}")
                continue

            p_tensor_result = create_p_tensor_from_record(
                hadm_id, traj_num, trajectory_info, req['p_tensor_range'],
                record_data, icu_admission_time, interval_minutes
            )

            if p_tensor_result:
                patient_p_tensors.append(p_tensor_result)

            # Create prediction targets from the single overlapping record
            pred_record_data = waveform_cache.load_record_data(pred_record, database)
            if pred_record_data is None:
                print(f"    ✗ Failed to load prediction record data for trajectory {traj_num}")
                continue

            pred_result = create_prediction_targets_from_record(
                hadm_id, traj_num, req['prediction_range'], req['t0_time'],
                pred_record_data, prediction_minutes, target_interval_seconds
            )

            if pred_result:
                patient_predictions.append(pred_result)

        except Exception as e:
            print(f"    Error processing trajectory {traj_num}: {e}")
            continue

    print(f"  Created {len(patient_p_tensors)} p_tensors and {len(patient_predictions)} prediction targets")
    print(f"  Skipped {skipped_multi_record} multi-record trajectories, {skipped_no_record} with no coverage")
    return patient_p_tensors, patient_predictions

def process_patient_waveforms_and_save(
    hadm_id, patient_id, patient_trajectories, patient_ics, icu_admission_time, interval_minutes, 
    prediction_minutes, target_interval_seconds, max_interval_seconds, waveform_cache, database, 
    p_tensor_cache_dir, prediction_target_cache_dir
):
    debug_print(f"[Worker PID: {os.getpid()}] Processing patient HADM_ID: {hadm_id}")
    patient_p_tensors, patient_predictions = process_patient_waveforms(
        hadm_id, patient_id, patient_trajectories, patient_ics,
        icu_admission_time, interval_minutes, prediction_minutes,
        target_interval_seconds, max_interval_seconds,
        waveform_cache, database
    )

    p_tensor_info = []
    for p_tensor in patient_p_tensors:
        traj_num = p_tensor['trajectory_num']
        debug_print(f"  [Worker PID: {os.getpid()}] Saving p_tensor for traj {traj_num}")
        p_file = Path(p_tensor_cache_dir) / f"p_tensor_{int(hadm_id)}_traj_{traj_num:03d}.pt"
        torch.save(p_tensor['tensor_data'], p_file)
        p_tensor_info.append({
            'hadm_id': hadm_id, 'trajectory_num': traj_num, 'file_path': str(p_file), 'metadata': p_tensor['metadata']
        })

    prediction_info = []
    for prediction in patient_predictions:
        traj_num = prediction['trajectory_num']
        debug_print(f"  [Worker PID: {os.getpid()}] Saving prediction_target for traj {traj_num}")
        pred_file = Path(prediction_target_cache_dir) / f"prediction_target_{int(hadm_id)}_traj_{traj_num:03d}.pt"
        torch.save(prediction['tensor_data'], pred_file)
        prediction_info.append({
            'hadm_id': hadm_id, 'trajectory_num': traj_num, 'file_path': str(pred_file), 'metadata': prediction['metadata']
        })
    
    debug_print(f"[Worker PID: {os.getpid()}] Finished saving for patient {hadm_id}. Returning metadata.")
    return hadm_id, p_tensor_info, prediction_info



def create_p_tensor_from_record(hadm_id, traj_num, trajectory_info, time_range,
                                record_data, icu_admission_time, interval_minutes):
    """Create p_tensor from loaded record data"""

    start_time, end_time = time_range

    # Extract signals from record
    signals_data = extract_signals_from_record(
        record_data, start_time, end_time, ['HR', 'ABP Mean', 'CVP', 'SV', 'CO']
    )

    if not signals_data:
        return None

    # Create target time grid aligned with med_tensor intervals
    n_intervals = trajectory_info['end_idx'] - trajectory_info['start_idx']
    target_times_seconds = []

    for i in range(n_intervals):
        interval_minutes_from_admission = (trajectory_info['start_idx'] + i) * interval_minutes
        target_time = icu_admission_time + timedelta(minutes=interval_minutes_from_admission + interval_minutes / 2)
        time_from_start = (target_time - start_time).total_seconds()
        target_times_seconds.append(time_from_start)

    # Interpolate signals to target grid
    interpolated = interpolate_signals_to_grid(signals_data, np.array(target_times_seconds))

    if not interpolated:
        return None

    # QUALITY CHECK: Require HR, ABP Mean, and CVP to be present
    required_for_p_tensor = ['HR', 'ABP Mean', 'CVP']
    missing_required = [sig for sig in required_for_p_tensor if sig not in interpolated]

    if missing_required:
        print(f"      ✗ Skipping p_tensor for trajectory {traj_num} - missing required signals: {missing_required}")
        return None

    # Additional check: Make sure these signals have some valid finite values
    for signal_name in required_for_p_tensor:
        signal_values = interpolated[signal_name]
        if not np.any(np.isfinite(signal_values)):
            print(f"      ✗ Skipping p_tensor for trajectory {traj_num} - {signal_name} has no valid values")
            return None

    print(f"      ✓ All required signals present for p_tensor trajectory {traj_num}")

    # Create tensor arrays
    # Create tensor arrays - 5 signals: HR, ABP Mean, CVP, SV, R_TPR (NO CO in final tensor)
    final_signal_names = ['HR', 'ABP Mean', 'CVP', 'SV', 'R_TPR']
    n_final_signals = len(final_signal_names)

    values_array = np.zeros((n_intervals, n_final_signals), dtype=np.float32)
    mask_array = np.zeros((n_intervals, n_final_signals), dtype=np.float32)

    # First, process the directly measured signals (HR, ABP Mean, CVP, SV)
    direct_signals = ['HR', 'ABP Mean', 'CVP', 'SV']
    for i, signal_name in enumerate(direct_signals):
        if signal_name in interpolated:
            signal_values = interpolated[signal_name]
            valid_mask = np.isfinite(signal_values)

            values_array[:, i] = np.where(valid_mask, signal_values, 0)
            mask_array[:, i] = valid_mask.astype(np.float32)
        else:
            # Signal not found - set to zero
            values_array[:, i] = 0
            mask_array[:, i] = 0

    # Handle CO/SV relationship (same as your IC code)
    hr_idx = 0  # HR
    sv_idx = 3  # SV

    # Create temporary arrays for CO (not included in final tensor)
    co_values = np.zeros(n_intervals, dtype=np.float32)
    co_mask = np.zeros(n_intervals, dtype=np.float32)

    # Get CO values if available
    if 'CO' in interpolated:
        co_signal_values = interpolated['CO']
        co_valid_mask = np.isfinite(co_signal_values)
        co_values = np.where(co_valid_mask, co_signal_values, 0)
        co_mask = co_valid_mask.astype(np.float32)

    # Calculate missing SV or CO using HR (like your IC code)
    for t in range(n_intervals):
        if mask_array[t, hr_idx] > 0 and values_array[t, hr_idx] > 0:
            hr_value = values_array[t, hr_idx]

            # If we have CO but missing SV: calculate SV = CO / HR
            if (co_mask[t] > 0 and mask_array[t, sv_idx] == 0 and co_values[t] > 0):
                values_array[t, sv_idx] = co_values[t] / hr_value
                mask_array[t, sv_idx] = 1.0

            # If we have SV but missing CO: calculate CO = SV * HR
            elif (mask_array[t, sv_idx] > 0 and co_mask[t] == 0 and values_array[t, sv_idx] > 0):
                co_values[t] = values_array[t, sv_idx] * hr_value
                co_mask[t] = 1.0

    # Calculate R_TPR for each time point (exactly like your IC code)
    map_idx = 1  # ABP Mean
    cvp_idx = 2  # CVP
    r_tpr_idx = 4  # R_TPR (last position)

    for t in range(n_intervals):
        # Check if MAP, CVP, and CO all exist (like your IC code)
        if (mask_array[t, map_idx] > 0 and mask_array[t, cvp_idx] > 0 and
                co_mask[t] > 0 and co_values[t] > 0):

            map_value = values_array[t, map_idx]
            cvp_value = values_array[t, cvp_idx]
            co_value = co_values[t]

            # Calculate r_tpr = (MAP - CVP) / CO
            values_array[t, r_tpr_idx] = (map_value - cvp_value) / co_value
            mask_array[t, r_tpr_idx] = 1.0
        else:
            # Missing data - set to 0 with mask 0
            values_array[t, r_tpr_idx] = 0.0
            mask_array[t, r_tpr_idx] = 0.0

    # Convert to tensors
    values_tensor = torch.from_numpy(values_array).float()
    mask_tensor = torch.from_numpy(mask_array).float()

    # Time tensor (hours from ICU admission)
    time_hours = np.array([(trajectory_info['start_idx'] + i) * interval_minutes / 60.0
                           for i in range(n_intervals)])
    time_tensor = torch.from_numpy(time_hours).float()

    return {
        'hadm_id': hadm_id,
        'trajectory_num': traj_num,
        'tensor_data': (values_tensor, mask_tensor, time_tensor, n_intervals),
        'metadata': {
            'record_used': f"{record_data['info']['patient_path']}/{record_data['info']['record_name']}",
            'sampling_rate': record_data['info']['sampling_rate'],
            'signals_extracted': list(interpolated.keys()),
            'time_range': time_range
        }
    }


def create_prediction_targets_from_record(hadm_id, traj_num, time_range, t0_time,
                                          record_data, prediction_minutes, target_interval_seconds):
    """Create prediction targets from loaded record data"""

    start_time, end_time = time_range

    # Extract signals from record (MAP and CVP only for predictions)
    signals_data = extract_signals_from_record(
        record_data, start_time, end_time, ['ABP Mean', 'CVP']
    )

    if not signals_data:
        return None

    # Create target time grid at specified intervals
    n_prediction_points = int((prediction_minutes * 60) / target_interval_seconds)
    target_times_seconds = []

    for i in range(n_prediction_points):
        target_time = t0_time + timedelta(seconds=i * target_interval_seconds)
        time_from_start = (target_time - start_time).total_seconds()
        target_times_seconds.append(time_from_start)

    # Interpolate signals
    interpolated = interpolate_signals_to_grid(signals_data, np.array(target_times_seconds))

    if not interpolated:
        return None

    # QUALITY CHECK: Require both ABP Mean and CVP for prediction targets
    required_for_prediction = ['ABP Mean', 'CVP']
    missing_required = [sig for sig in required_for_prediction if sig not in interpolated]

    if missing_required:
        print(
            f"      ✗ Skipping prediction targets for trajectory {traj_num} - missing required signals: {missing_required}")
        return None

    # Additional check: Make sure these signals have some valid finite values
    for signal_name in required_for_prediction:
        signal_values = interpolated[signal_name]
        if not np.any(np.isfinite(signal_values)):
            print(f"      ✗ Skipping prediction targets for trajectory {traj_num} - {signal_name} has no valid values")
            return None

    print(f"      ✓ All required signals present for prediction targets trajectory {traj_num}")

    # Create tensor arrays
    signal_names = ['ABP Mean', 'CVP']
    n_signals = len(signal_names)

    values_array = np.zeros((n_prediction_points, n_signals), dtype=np.float32)
    mask_array = np.zeros((n_prediction_points, n_signals), dtype=np.float32)

    for i, signal_name in enumerate(signal_names):
        if signal_name in interpolated:
            signal_values = interpolated[signal_name]
            valid_mask = np.isfinite(signal_values)

            values_array[:, i] = np.where(valid_mask, signal_values, 0)
            mask_array[:, i] = valid_mask.astype(np.float32)

    # Convert to tensors
    values_tensor = torch.from_numpy(values_array).float()
    mask_tensor = torch.from_numpy(mask_array).float()

    # Time tensor (seconds from t_0)
    time_seconds = np.array([i * target_interval_seconds for i in range(n_prediction_points)])
    time_tensor = torch.from_numpy(time_seconds).float()

    return {
        'hadm_id': hadm_id,
        'trajectory_num': traj_num,
        'tensor_data': (values_tensor, mask_tensor, time_tensor, n_prediction_points),
        'metadata': {
            'record_used': f"{record_data['info']['patient_path']}/{record_data['info']['record_name']}",
            'sampling_rate': record_data['info']['sampling_rate'],
            'actual_interval_seconds': 1.0 / record_data['info']['sampling_rate'],
            'signals_extracted': list(interpolated.keys()),
            'prediction_window': time_range
        }
    }


def create_waveform_tensors_efficient(
        trajectory_metadata_path,
        ic_metadata_path,
        relevant_patient_ids_path,
        icustays_path,
        waveform_base_dir=None,
        p_tensor_cache_dir='waveform_p_tensors',
        prediction_target_cache_dir='waveform_prediction_targets',
        database='mimic3wdb-matched',
        prediction_minutes=25,
        target_interval_seconds=10,
        max_interval_seconds=10,
        skip_existing=True,
        n_workers=4
):
    debug_print("--- Starting Efficient Waveform Tensor Creation ---")
    if not setup_wfdb_credentials():
        return None, None

    Path(p_tensor_cache_dir).mkdir(parents=True, exist_ok=True)
    Path(prediction_target_cache_dir).mkdir(parents=True, exist_ok=True)

    debug_print("Loading metadata...")
    traj_metadata, ic_metadata = load_trajectory_metadata(trajectory_metadata_path, ic_metadata_path)
    patient_mapping = get_patient_mapping(relevant_patient_ids_path)
    icu_admission_map = get_icu_admission_times(icustays_path)
    all_trajectories = traj_metadata['all_trajectories']
    interval_minutes = traj_metadata['interval_minutes']
    all_initial_conditions = ic_metadata['all_initial_conditions']
    debug_print(f"Loaded data for {len(all_initial_conditions)} patients.")

    all_p_tensors = {}
    all_prediction_targets = {}
    tasks = []
    skipped_patients = 0
    waveform_cache = WaveformRecordCache()

    debug_print("Preparing tasks for parallel processing...")
    for hadm_id in all_initial_conditions.keys():
        if hadm_id not in patient_mapping or hadm_id not in icu_admission_map:
            continue

        if skip_existing:
            all_files_exist = True
            for ic_info in all_initial_conditions[hadm_id]:
                traj_num = ic_info['trajectory_num']
                p_file = Path(p_tensor_cache_dir) / f"p_tensor_{int(hadm_id)}_traj_{traj_num:03d}.pt"
                pred_file = Path(prediction_target_cache_dir) / f"prediction_target_{int(hadm_id)}_traj_{traj_num:03d}.pt"
                if not (p_file.exists() and pred_file.exists()):
                    all_files_exist = False
                    break
            if all_files_exist:
                debug_print(f"Skipping patient {hadm_id} - all output files exist.")
                skipped_patients += 1
                continue

        patient_id = patient_mapping[hadm_id]
        icu_admission_time = icu_admission_map[hadm_id]
        patient_trajectories = all_trajectories.get(hadm_id, [])
        patient_ics = all_initial_conditions[hadm_id]
        tasks.append(
            (hadm_id, patient_id, patient_trajectories, patient_ics, icu_admission_time, interval_minutes, 
             prediction_minutes, target_interval_seconds, max_interval_seconds, waveform_cache, database, 
             p_tensor_cache_dir, prediction_target_cache_dir)
        )

    debug_print(f"Submitting {len(tasks)} tasks to {n_workers} workers.")
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
        future_to_hadm = {executor.submit(process_patient_waveforms_and_save, *task): task[0] for task in tasks}

        for future in tqdm(concurrent.futures.as_completed(future_to_hadm), total=len(tasks), desc="Processing patients"):
            hadm_id, p_tensor_info, prediction_info = future.result()
            if p_tensor_info:
                all_p_tensors[hadm_id] = p_tensor_info
            if prediction_info:
                all_prediction_targets[hadm_id] = prediction_info

    debug_print("Aggregating final metadata...")
    p_tensor_metadata = {
        'all_p_tensors': all_p_tensors,
        'signal_names': ['HR', 'ABP Mean', 'CVP', 'SV', 'R_TPR'],
        'n_signals': 5,
        'interval_minutes': interval_minutes,
        'total_trajectories': sum(len(p) for p in all_p_tensors.values()),
        'aligned_with_med_trajectories': trajectory_metadata_path,
        'source': 'waveform_numerics_efficient_parallel'
    }
    prediction_metadata = {
        'all_prediction_targets': all_prediction_targets,
        'signal_names': ['ABP Mean', 'CVP'],
        'n_signals': 2,
        'prediction_minutes': prediction_minutes,
        'target_interval_seconds': target_interval_seconds,
        'max_interval_seconds': max_interval_seconds,
        'total_trajectories': sum(len(preds) for preds in all_prediction_targets.values()),
        'source': 'waveform_numerics_efficient_parallel'
    }

    p_metadata_file = Path(p_tensor_cache_dir) / "p_tensor_metadata.pkl"
    with open(p_metadata_file, "wb") as f:
        pickle.dump(p_tensor_metadata, f)

    pred_metadata_file = Path(prediction_target_cache_dir) / "prediction_target_metadata.pkl"
    with open(pred_metadata_file, "wb") as f:
        pickle.dump(prediction_metadata, f)

    print(f"\n✅ EFFICIENT PROCESSING COMPLETE:")
    print(f"  New patients processed: {len(all_p_tensors)}")
    print(f"  Patients skipped: {skipped_patients}")
    print(f"  P-tensors created: {sum(len(pts) for pts in all_p_tensors.values())}")
    print(f"  Prediction targets created: {sum(len(pts) for pts in all_prediction_targets.values())}")
    print(f"  P-tensor metadata saved to: {p_metadata_file}")
    print(f"  Prediction metadata saved to: {pred_metadata_file}")

    return all_p_tensors, all_prediction_targets


# Example usage
if __name__ == "__main__":
    # Set up paths
    trajectory_metadata_path = "../../data/mimic3refactor/processed_data/med_tensors/trajectory_metadata.pkl"
    ic_metadata_path = "../../data/mimic3refactor/processed_data/initial_conditions/initial_conditions_metadata.pkl"
    relevant_patient_ids_path = "../../data/mimic3refactor/processed_data/relevant_patient_ids.csv"
    icustays_path = "../../data/mimic3refactor/input_data/ICUSTAYS.csv"

    # Create waveform tensors efficiently
    p_tensors, prediction_targets = create_waveform_tensors_efficient(
        trajectory_metadata_path=trajectory_metadata_path,
        ic_metadata_path=ic_metadata_path,
        relevant_patient_ids_path=relevant_patient_ids_path,
        icustays_path=icustays_path,
        waveform_base_dir=None,  # Remote access
        p_tensor_cache_dir="../../data/mimic3refactor/processed_data/waveform_p_tensors",
        prediction_target_cache_dir="../../data/mimic3refactor/processed_data/waveform_prediction_targets",
        prediction_minutes=25,
        target_interval_seconds=5,
        max_interval_seconds=10,
        n_workers=8
    )