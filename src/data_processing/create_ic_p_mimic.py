#!/usr/bin/env python3
"""
Integrated MIMIC-III Waveform Processing: IC Tensors + Prediction Targets
Creates both initial condition tensors at t_0 and prediction targets for 25min ahead
Uses efficient patient-batch processing to minimize network calls
Now with parallel processing support for faster execution
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
from functools import partial

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


def load_trajectory_metadata(trajectory_metadata_path, ic_metadata_path=None):
    """Load trajectory metadata and optional IC metadata"""
    print("Loading trajectory metadata...")
    with open(trajectory_metadata_path, 'rb') as f:
        traj_metadata = pickle.load(f)

    ic_metadata = None
    if ic_metadata_path and Path(ic_metadata_path).exists():
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


def calculate_t0_timestamp(trajectory_info, icu_admission_time, interval_minutes):
    """Calculate exact t0 timestamp from trajectory metadata"""
    # t0 is at the end of the trajectory (end_idx - 1)
    t0_interval_idx = trajectory_info['end_idx'] - 1

    # Convert interval index to minutes from admission
    t0_minutes_from_admission = t0_interval_idx * interval_minutes
    t0_minutes_from_admission = float(t0_minutes_from_admission)

    # Calculate exact t0 timestamp
    t0_timestamp = icu_admission_time + timedelta(minutes=t0_minutes_from_admission)

    return t0_timestamp


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


def extract_ic_from_loaded_record(hadm_id, traj_num, t0_timestamp, record_data,
                                  extraction_window_seconds=30):
    """Extract initial conditions from already-loaded record data"""

    record = record_data['record']
    record_info = record_data['info']
    record_start = record_info['start_time']

    # Calculate time offset and sample index
    time_offset_seconds = (t0_timestamp - record_start).total_seconds()

    # Check if t0 is within the record
    record_duration = record_info['duration_seconds']
    if time_offset_seconds < 0 or time_offset_seconds > record_duration:
        print(f"      t0 outside record range for trajectory {traj_num}")
        return None

    t0_sample_idx = int(time_offset_seconds * record.fs)

    # Define search window (±30 seconds)
    window_samples = int(extraction_window_seconds * record.fs)
    start_sample = max(0, t0_sample_idx - window_samples)
    end_sample = min(record.sig_len, t0_sample_idx + window_samples)

    samples_to_read = end_sample - start_sample
    print(f"      Reading {samples_to_read} samples around t_0 for trajectory {traj_num}")

    required_signals = ['HR', 'ABP Mean', 'CVP']
    optional_signals = ['SV', 'CO']
    all_signals = required_signals + optional_signals

    ic_values = {}
    ic_mask = {}

    # Extract each signal
    for signal_name in all_signals:
        try:
            signal_index = record.sig_name.index(signal_name)
            signal_data = record.p_signal[:, signal_index]

            # Extract window around t0
            segment = signal_data[start_sample:end_sample]

            # Find any finite values
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

                print(f"      ✓ {signal_name}: {closest_value:.2f} ({distance_seconds:.1f}s from t_0)")
            else:
                ic_values[signal_name] = 0.0
                ic_mask[signal_name] = 0.0
                print(f"      ✗ {signal_name}: No data within ±{extraction_window_seconds}s")

        except (ValueError, IndexError):
            ic_values[signal_name] = 0.0
            ic_mask[signal_name] = 0.0
            print(f"      ✗ {signal_name}: Signal not found")

    # Check if all required signals have data
    missing_signals = [sig for sig in required_signals if ic_mask.get(sig, 0) == 0]
    if missing_signals:
        print(f"      Missing required signals: {missing_signals}")
        return None

    # Calculate derived values
    if ic_mask.get('HR', 0) > 0 and ic_values.get('HR', 0) > 0:
        hr_value = ic_values['HR']

        # If we have CO but missing SV: calculate SV = CO / HR
        if (ic_mask.get('CO', 0) > 0 and ic_mask.get('SV', 0) == 0 and
                ic_values.get('CO', 0) > 0):
            ic_values['SV'] = ic_values['CO'] / hr_value
            ic_mask['SV'] = 1.0
            print(f"      ✓ SV: {ic_values['SV']:.2f} (calculated from CO/HR)")

        # If we have SV but missing CO: calculate CO = SV * HR
        elif (ic_mask.get('SV', 0) > 0 and ic_mask.get('CO', 0) == 0 and
              ic_values.get('SV', 0) > 0):
            ic_values['CO'] = ic_values['SV'] * hr_value
            ic_mask['CO'] = 1.0
            print(f"      ✓ CO: {ic_values['CO']:.2f} (calculated from SV*HR)")

    # Calculate R_TPR
    if (ic_mask.get('ABP Mean', 0) > 0 and ic_mask.get('CVP', 0) > 0 and
            ic_mask.get('CO', 0) > 0 and ic_values.get('CO', 0) > 0):
        ic_values['R_TPR'] = (ic_values['ABP Mean'] - ic_values['CVP']) / ic_values['CO']
        ic_mask['R_TPR'] = 1.0
        print(f"      ✓ R_TPR: {ic_values['R_TPR']:.2f} (calculated)")
    else:
        ic_values['R_TPR'] = 0.0
        ic_mask['R_TPR'] = 0.0
        print(f"      ✗ R_TPR: Cannot calculate (missing MAP/CVP/CO)")

    print(f"      ✓ All required signals found for trajectory {traj_num}!")

    return {
        'hadm_id': hadm_id,
        'trajectory_num': traj_num,
        't0_timestamp': t0_timestamp,
        'ic_values': ic_values,
        'ic_mask': ic_mask,
        'record_used': f"{record_info['patient_path']}/{record_info['record_name']}"
    }


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


def create_prediction_targets_with_ic_anchor(hadm_id, traj_num, t0_timestamp, record_data,
                                             ic_values, prediction_minutes, target_interval_seconds,
                                             min_points_per_minute=12):
    """
    Create prediction targets using IC as anchor point for interpolation

    New strategy:
    1. Use IC values at t_0 as anchor point (time 0)
    2. Extract prediction window data from t_0 to t_0+25min
    3. Check data density: require min_points_per_minute for EACH individual minute
       - Example: 25-minute window = check minutes [0-1, 1-2, 2-3, ..., 24-25]
       - Each minute must have >= min_points_per_minute data points
       - Rejects if ANY minute fails (no averaging allowed)
    4. Interpolate using IC + prediction data (IC not included in final targets)
    5. Reject if any NaNs remain after interpolation
    """

    # Calculate full time range (including t_0 for interpolation anchor)
    start_time = t0_timestamp  # Include t_0 in data extraction
    end_time = t0_timestamp + timedelta(minutes=prediction_minutes)

    # Extract signals from record (MAP and CVP only for predictions)
    signals_data = extract_signals_from_record(
        record_data, start_time, end_time, ['ABP Mean', 'CVP']
    )

    if not signals_data:
        print(f"      ✗ No waveform data found in prediction window")
        return None

    # Add IC values as anchor points at t_0
    record_start = record_data['info']['start_time']
    t0_offset_seconds = (t0_timestamp - record_start).total_seconds()

    required_signals = ['ABP Mean', 'CVP']
    for signal_name in required_signals:
        if signal_name in ic_values and signal_name in signals_data:
            # Add IC value at t_0 to the signal data
            ic_value = ic_values[signal_name]

            # Insert IC point at the beginning (sorted by time)
            signals_data[signal_name]['times'] = np.concatenate([
                [t0_offset_seconds], signals_data[signal_name]['times']
            ])
            signals_data[signal_name]['values'] = np.concatenate([
                [ic_value], signals_data[signal_name]['values']
            ])

            print(f"      ✓ Added IC anchor: {signal_name} = {ic_value:.1f} at t_0")
        elif signal_name in ic_values:
            # Create new signal data with just IC value
            signals_data[signal_name] = {
                'times': np.array([t0_offset_seconds]),
                'values': np.array([ic_values[signal_name]])
            }
            print(f"      ✓ Created IC anchor: {signal_name} = {ic_values[signal_name]:.1f} at t_0")

    # DATA DENSITY CHECK: Ensure sufficient data points PER MINUTE (not average)
    total_minutes = prediction_minutes
    for signal_name in required_signals:
        if signal_name not in signals_data:
            print(f"      ✗ Missing required signal: {signal_name}")
            return None

        signal_times = signals_data[signal_name]['times']
        t0_offset_seconds = (t0_timestamp - record_start).total_seconds()

        # Check each individual minute in the prediction window
        minutes_failed = []
        for minute_idx in range(total_minutes):
            minute_start = t0_offset_seconds + (minute_idx * 60)  # Start of this minute
            minute_end = t0_offset_seconds + ((minute_idx + 1) * 60)  # End of this minute

            # Count data points in this specific minute
            points_in_minute = np.sum((signal_times >= minute_start) & (signal_times < minute_end))

            if points_in_minute < min_points_per_minute:
                minutes_failed.append({
                    'minute': minute_idx,
                    'points': points_in_minute,
                    'required': min_points_per_minute
                })

        if minutes_failed:
            print(f"      ✗ {signal_name}: {len(minutes_failed)}/{total_minutes} minutes fail density requirement")
            for failure in minutes_failed[:3]:  # Show first 3 failures
                print(
                    f"        Minute {failure['minute']}: {failure['points']} points < {failure['required']} required")
            if len(minutes_failed) > 3:
                print(f"        ... and {len(minutes_failed) - 3} more minutes")
            return None
        else:
            total_points = len(signal_times)
            avg_density = total_points / total_minutes
            print(
                f"      ✓ {signal_name}: All {total_minutes} minutes pass density check (avg: {avg_density:.1f} points/min)")

    # Create target time grid for prediction (EXCLUDING t_0)
    n_prediction_points = int((prediction_minutes * 60) / target_interval_seconds)
    target_times_seconds = np.array([
        t0_offset_seconds + (i + 1) * target_interval_seconds
        for i in range(n_prediction_points)
    ])

    # Interpolate using IC anchor + prediction window data
    interpolated = {}
    for signal_name in required_signals:
        if signal_name not in signals_data:
            continue

        times = signals_data[signal_name]['times']
        values = signals_data[signal_name]['values']

        if len(times) < 2:
            print(f"      ✗ {signal_name}: Need at least 2 points for interpolation")
            return None

        try:
            # Remove duplicates and sort
            unique_indices = np.unique(times, return_index=True)[1]
            unique_times = times[unique_indices]
            unique_values = values[unique_indices]

            if len(unique_times) < 2:
                print(f"      ✗ {signal_name}: Less than 2 unique time points after deduplication")
                return None

            # Create interpolator
            interpolator = interp1d(unique_times, unique_values, kind='linear',
                                    bounds_error=False, fill_value=np.nan)

            # Interpolate to target times (excluding t_0)
            interpolated_values = interpolator(target_times_seconds)

            # STRICT CHECK: No NaNs allowed after interpolation
            if np.any(np.isnan(interpolated_values)):
                n_nans = np.sum(np.isnan(interpolated_values))
                print(f"      ✗ {signal_name}: {n_nans}/{len(interpolated_values)} NaN values after interpolation")
                return None

            interpolated[signal_name] = interpolated_values
            print(f"      ✓ {signal_name}: Perfect interpolation, no NaN values")

        except Exception as e:
            print(f"      ✗ {signal_name}: Interpolation failed: {e}")
            return None

    if len(interpolated) != len(required_signals):
        print(f"      ✗ Only {len(interpolated)}/{len(required_signals)} signals successfully interpolated")
        return None

    print(f"      ✓ All signals successfully interpolated with IC anchor")

    # Create tensor arrays
    signal_names = list(interpolated.keys())
    n_signals = len(signal_names)

    values_array = np.zeros((n_prediction_points, n_signals), dtype=np.float32)
    mask_array = np.ones((n_prediction_points, n_signals), dtype=np.float32)  # All 1s since no NaNs

    for i, signal_name in enumerate(signal_names):
        values_array[:, i] = interpolated[signal_name]

    # Convert to tensors
    values_tensor = torch.from_numpy(values_array).float()
    mask_tensor = torch.from_numpy(mask_array).float()

    # Time tensor (seconds from t_0) - starts at target_interval_seconds
    time_seconds_from_t0 = np.array([(i + 1) * target_interval_seconds for i in range(n_prediction_points)])
    time_tensor = torch.from_numpy(time_seconds_from_t0).float()

    return {
        'hadm_id': hadm_id,
        'trajectory_num': traj_num,
        'tensor_data': (values_tensor, mask_tensor, time_tensor, n_prediction_points),
        'metadata': {
            'record_used': f"{record_data['info']['patient_path']}/{record_data['info']['record_name']}",
            'sampling_rate': record_data['info']['sampling_rate'],
            'actual_interval_seconds': 1.0 / record_data['info']['sampling_rate'],
            'signals_used': signal_names,
            'ic_anchor_used': True,
            'min_points_per_minute': min_points_per_minute,
            'density_check_method': 'per_minute_strict',
            'data_density_per_signal': {
                sig: len(signals_data[sig]['times']) / prediction_minutes
                for sig in signal_names if sig in signals_data
            },
            'prediction_window': (start_time, end_time),
            't0_timestamp': t0_timestamp,
            'interpolation_method': 'ic_anchored_linear'
        }
    }


def process_patient_integrated(hadm_id, patient_id, patient_trajectories, icu_admission_time,
                               interval_minutes, prediction_minutes, target_interval_seconds,
                               waveform_cache, ic_cache_dir, prediction_cache_dir,
                               min_points_per_minute=12, database='mimic3wdb-matched'):
    """
    Integrated processing: Create both IC tensors and prediction targets for each trajectory

    Logic:
    1. Extract IC at t_0 from the record containing t_0
    2. Create prediction targets from t_0 to min(t_0+25min, next_t_0) using the SAME record
    3. Skip trajectory if the record doesn't extend long enough for full prediction window

    This ensures IC and prediction come from the same continuous recording session.
    """

    print(f"\nProcessing patient {patient_id} (HADM {hadm_id}) - {len(patient_trajectories)} trajectories")

    # Step 1: Discover all waveform records for this patient (ONE network call)
    patient_records = waveform_cache.discover_patient_records(patient_id, database=database)

    if not patient_records:
        print(f"  No waveform records found for patient {patient_id}")
        return [], []

    print(f"  Processing {len(patient_trajectories)} trajectories with {len(patient_records)} available records")

    # Sort trajectories by trajectory number to ensure proper ordering for finding next t_0
    sorted_trajectories = sorted(patient_trajectories, key=lambda x: x['trajectory_num'])

    patient_ics = []
    patient_predictions = []
    processed_count = 0
    skipped_no_records = 0
    skipped_insufficient_duration = 0
    skipped_extraction_failed = 0

    # Step 2: Process each trajectory in sorted order
    for traj_idx, traj in enumerate(sorted_trajectories):
        traj_num = traj['trajectory_num']

        try:
            # Calculate t_0 timestamp
            t0_timestamp = calculate_t0_timestamp(traj, icu_admission_time, interval_minutes)

            # Find records that contain t_0 (±30 seconds window for IC extraction)
            ic_window_start = t0_timestamp - timedelta(seconds=30)
            ic_window_end = t0_timestamp + timedelta(seconds=30)

            ic_records = waveform_cache.find_records_for_timerange(
                patient_id, ic_window_start, ic_window_end
            )

            # SKIP if no records contain t_0
            if not ic_records:
                print(f"    ✗ Skipping trajectory {traj_num} - no records contain t_0")
                skipped_no_records += 1
                continue

            # Use the first record that contains t_0 for both IC extraction and prediction
            record = ic_records[0]
            record_start = record['start_time']
            record_end = record['end_time']

            # Calculate when this trajectory should end (25min after t_0 OR next t_0, whichever is earlier)
            max_prediction_end = t0_timestamp + timedelta(minutes=prediction_minutes)

            # Find next t_0 if there are more trajectories for this patient
            next_t0 = None
            if traj_idx + 1 < len(sorted_trajectories):
                next_traj = sorted_trajectories[traj_idx + 1]
                next_t0 = calculate_t0_timestamp(next_traj, icu_admission_time, interval_minutes)

            # Determine actual prediction end time
            if next_t0 is not None and next_t0 < max_prediction_end:
                actual_prediction_end = next_t0
                prediction_duration_min = (actual_prediction_end - t0_timestamp).total_seconds() / 60
                print(f"      Prediction window limited by next t_0: {prediction_duration_min:.1f} min")
            else:
                actual_prediction_end = max_prediction_end
                prediction_duration_min = prediction_minutes
                print(f"      Full prediction window: {prediction_duration_min:.1f} min")

            # CHECK: Does the record extend at least to our prediction end time?
            if record_end < actual_prediction_end:
                time_shortage = (actual_prediction_end - record_end).total_seconds() / 60
                print(
                    f"    ✗ Skipping trajectory {traj_num} - record ends {time_shortage:.1f} min before prediction end")
                print(f"      Record ends: {record_end}")
                print(f"      Need until: {actual_prediction_end}")
                skipped_insufficient_duration += 1
                continue

            print(f"    ✓ Processing trajectory {traj_num} at t_0={t0_timestamp}")
            print(f"      Using record: {record['record_name']} ({record_start} to {record_end})")
            print(f"      Prediction: t_0 to t_0+{prediction_duration_min:.1f}min")

            # Load record data (same record for both IC and prediction)
            record_data = waveform_cache.load_record_data(record, database)
            if record_data is None:
                print(f"    ✗ Failed to load record data for trajectory {traj_num}")
                skipped_extraction_failed += 1
                continue

            # Extract IC at t_0
            ic_result = extract_ic_from_loaded_record(
                hadm_id, traj_num, t0_timestamp, record_data
            )

            if not ic_result:
                print(f"      ✗ IC extraction failed for trajectory {traj_num}")
                skipped_extraction_failed += 1
                continue

            # Create prediction targets using IC as anchor point
            # Note: using actual prediction duration (may be < 25min if limited by next t_0)
            pred_result = create_prediction_targets_with_ic_anchor(
                hadm_id, traj_num, t0_timestamp, record_data,
                ic_result['ic_values'],  # Pass IC values as anchor
                prediction_duration_min, target_interval_seconds,
                min_points_per_minute
            )

            if not pred_result:
                print(f"      ✗ Prediction target creation failed for trajectory {traj_num}")
                continue

            # SAVE BOTH TENSORS IMMEDIATELY
            # Save IC tensor
            ic_file = Path(ic_cache_dir) / f"ic_tensor_{int(hadm_id)}_traj_{traj_num:03d}.pt"
            physio_values = [
                ic_result['ic_values']['HR'],
                ic_result['ic_values']['ABP Mean'],
                ic_result['ic_values']['CVP'],
                ic_result['ic_values'].get('SV', 0.0),
                ic_result['ic_values'].get('R_TPR', 0.0)
            ]
            physio_masks = [
                ic_result['ic_mask']['HR'],
                ic_result['ic_mask']['ABP Mean'],
                ic_result['ic_mask']['CVP'],
                ic_result['ic_mask'].get('SV', 0.0),
                ic_result['ic_mask'].get('R_TPR', 0.0)
            ]
            ic_tensor = torch.tensor(physio_values, dtype=torch.float32)
            ic_mask_tensor = torch.tensor(physio_masks, dtype=torch.float32)
            torch.save((ic_tensor, ic_mask_tensor), ic_file)

            # Save prediction tensor
            pred_file = Path(prediction_cache_dir) / f"prediction_target_{int(hadm_id)}_traj_{traj_num:03d}.pt"
            torch.save(pred_result['tensor_data'], pred_file)

            # Store only metadata
            patient_ics.append({
                'hadm_id': hadm_id,
                'trajectory_num': traj_num,
                't0_timestamp': ic_result['t0_timestamp'],
                'file_path': str(ic_file),
                'ic_values': ic_result['ic_values'],
                'record_used': ic_result['record_used']
            })

            patient_predictions.append({
                'hadm_id': hadm_id,
                'trajectory_num': traj_num,
                'file_path': str(pred_file),
                'metadata': pred_result['metadata']
            })

            print(f"      ✓ Both IC and prediction tensors saved for trajectory {traj_num}")
            processed_count += 1

        except Exception as e:
            print(f"    Error processing trajectory {traj_num}: {e}")
            skipped_extraction_failed += 1
            continue

    total_skipped = skipped_no_records + skipped_insufficient_duration + skipped_extraction_failed
    print(f"  Created {len(patient_ics)} IC tensors and {len(patient_predictions)} prediction targets")
    print(f"  Processed: {processed_count}, Total Skipped: {total_skipped}")
    print(f"    - No records contain t_0: {skipped_no_records}")
    print(f"    - Insufficient record duration: {skipped_insufficient_duration}")
    print(f"    - Extraction failed: {skipped_extraction_failed}")
    return patient_ics, patient_predictions


def process_patient_integrated_standalone(
        hadm_id, patient_id, patient_trajectories, icu_admission_time,
        interval_minutes, prediction_minutes, target_interval_seconds,
        ic_cache_dir, prediction_cache_dir, min_points_per_minute=12, database='mimic3wdb-matched'
):
    """
    Standalone version for parallel processing - creates its own WaveformRecordCache
    """
    # Create a new cache instance for this worker process
    waveform_cache = WaveformRecordCache()

    return process_patient_integrated(
        hadm_id, patient_id, patient_trajectories, icu_admission_time,
        interval_minutes, prediction_minutes, target_interval_seconds,
        waveform_cache, ic_cache_dir, prediction_cache_dir, min_points_per_minute, database
    )


def create_integrated_waveform_tensors(
        trajectory_metadata_path,
        relevant_patient_ids_path,
        icustays_path,
        waveform_base_dir=None,
        ic_cache_dir='initial_conditions_waveform',
        prediction_cache_dir='waveform_prediction_targets',
        database='mimic3wdb-matched',
        prediction_minutes=25,
        target_interval_seconds=10,
        min_points_per_minute=12,
        n_workers=4,
        skip_existing=True
):
    """
    Integrated tensor creation: Both IC tensors at t_0 and prediction targets for 25min ahead
    Uses efficient patient-batch processing to minimize network calls
    Now with parallel processing support for faster execution

    NEW STRATEGY:
    - IC extracted at t_0 from record containing t_0
    - IC used as anchor point (time 0) for prediction interpolation
    - Require min_points_per_minute data density FOR EACH MINUTE (not average)
    - Prediction targets start after t_0 (IC not included in targets)
    - Reject trajectory if any NaN values remain after interpolation
    - Ensures perfect consistency between IC and prediction data

    This ensures IC and predictions come from the same continuous recording.
    """

    # Setup
    if not setup_wfdb_credentials():
        return None, None

    # Create output directories
    Path(ic_cache_dir).mkdir(parents=True, exist_ok=True)
    Path(prediction_cache_dir).mkdir(parents=True, exist_ok=True)

    # Load metadata
    traj_metadata, _ = load_trajectory_metadata(trajectory_metadata_path)
    patient_mapping = get_patient_mapping(relevant_patient_ids_path)
    icu_admission_map = get_icu_admission_times(icustays_path)

    all_trajectories = traj_metadata['all_trajectories']
    interval_minutes = traj_metadata['interval_minutes']

    print(f"Processing {len(all_trajectories)} patients")
    print(f"Prediction window: {prediction_minutes} minutes after t_0")
    print(f"Target sampling: {target_interval_seconds}s intervals")
    print(f"Data quality: min {min_points_per_minute} points EVERY minute (per-minute check)")
    print(f"Strategy: IC-anchored interpolation with strict NaN rejection")
    print(f"Integrated processing: IC + Prediction targets in one pass")
    print(f"Parallel workers: {n_workers}")

    # Initialize results storage
    all_ics = {}
    all_predictions = {}
    processed_trajectories = 0
    skipped_trajectories = 0

    # Prepare processing function
    process_func = partial(
        process_patient_integrated_standalone,
        interval_minutes=interval_minutes,
        prediction_minutes=prediction_minutes,
        target_interval_seconds=target_interval_seconds,
        min_points_per_minute=min_points_per_minute,
        ic_cache_dir=ic_cache_dir,
        prediction_cache_dir=prediction_cache_dir,
        database=database
    )

    # Filter patients that need processing
    patients_to_process = []
    for hadm_id in all_trajectories.keys():
        if hadm_id not in patient_mapping or hadm_id not in icu_admission_map:
            continue

        patient_id = patient_mapping[hadm_id]
        icu_admission_time = icu_admission_map[hadm_id]
        patient_trajectories = all_trajectories[hadm_id]

        # Check if files already exist
        if skip_existing:
            all_exist = True
            for traj in patient_trajectories:
                traj_num = traj['trajectory_num']
                ic_file = Path(ic_cache_dir) / f"ic_tensor_{int(hadm_id)}_traj_{traj_num:03d}.pt"
                pred_file = Path(prediction_cache_dir) / f"prediction_target_{int(hadm_id)}_traj_{traj_num:03d}.pt"

                if not (ic_file.exists() and pred_file.exists()):
                    all_exist = False
                    break

            if all_exist:
                print(f"Skipping patient {patient_id} - all files exist")
                skipped_trajectories += len(patient_trajectories)
                continue

        patients_to_process.append({
            'hadm_id': hadm_id,
            'patient_id': patient_id,
            'patient_trajectories': patient_trajectories,
            'icu_admission_time': icu_admission_time
        })

    print(f"Processing {len(patients_to_process)} patients...")

    # Process patients
    if n_workers > 1:
        # Parallel processing
        print(f"Using {n_workers} parallel workers")

        with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = {}

            for patient_info in patients_to_process:
                future = executor.submit(
                    process_func,
                    hadm_id=patient_info['hadm_id'],
                    patient_id=patient_info['patient_id'],
                    patient_trajectories=patient_info['patient_trajectories'],
                    icu_admission_time=patient_info['icu_admission_time']
                )
                futures[future] = patient_info

            # Collect results
            for future in tqdm(concurrent.futures.as_completed(futures),
                               total=len(futures), desc="Processing patients in parallel"):
                patient_info = futures[future]
                hadm_id = patient_info['hadm_id']
                patient_id = patient_info['patient_id']

                try:
                    patient_ics, patient_predictions = future.result()

                    # Store in results (tensors already saved by worker)
                    if patient_ics:
                        all_ics[hadm_id] = patient_ics
                        processed_trajectories += len(patient_ics)
                    if patient_predictions:
                        all_predictions[hadm_id] = patient_predictions

                except Exception as exc:
                    print(f'Patient {patient_id} (HADM {hadm_id}) generated an exception: {exc}')
                    skipped_trajectories += len(patient_info['patient_trajectories'])
                    import traceback
                    traceback.print_exc()

    else:
        # Sequential processing
        print("Using sequential processing")
        waveform_cache = WaveformRecordCache()

        for patient_info in tqdm(patients_to_process, desc="Processing patients sequentially"):
            hadm_id = patient_info['hadm_id']
            patient_id = patient_info['patient_id']

            try:
                patient_ics, patient_predictions = process_patient_integrated(
                    hadm_id, patient_id, patient_info['patient_trajectories'],
                    patient_info['icu_admission_time'], interval_minutes,
                    prediction_minutes, target_interval_seconds,
                    waveform_cache, ic_cache_dir, prediction_cache_dir,
                    min_points_per_minute, database
                )

                # Store in results
                if patient_ics:
                    all_ics[hadm_id] = patient_ics
                    processed_trajectories += len(patient_ics)
                if patient_predictions:
                    all_predictions[hadm_id] = patient_predictions

            except Exception as e:
                print(f"Error processing patient {patient_id}: {e}")
                skipped_trajectories += len(patient_info['patient_trajectories'])
                continue

    # Save IC metadata
    ic_metadata = {
        'all_initial_conditions': all_ics,
        'total_ics': sum(len(ics) for ics in all_ics.values()),
        'parameters': ['HR', 'ABP Mean', 'CVP', 'SV', 'R_TPR'],
        'extraction_method': 'waveform_numerics_ic_anchored_parallel',
        'prediction_strategy': 'ic_anchored_interpolation',
        'min_points_per_minute': min_points_per_minute,
        'interval_minutes': interval_minutes,
        'n_workers': n_workers
    }

    ic_metadata_file = Path(ic_cache_dir) / "ic_metadata.pkl"
    with open(ic_metadata_file, "wb") as f:
        pickle.dump(ic_metadata, f)

    # Save prediction metadata
    prediction_metadata = {
        'all_prediction_targets': all_predictions,
        'signal_names': ['ABP Mean', 'CVP'],
        'n_signals': 2,
        'prediction_minutes': prediction_minutes,
        'target_interval_seconds': target_interval_seconds,
        'min_points_per_minute': min_points_per_minute,
        'total_trajectories': sum(len(preds) for preds in all_predictions.values()),
        'source': 'waveform_numerics_ic_anchored_parallel',
        'interpolation_method': 'ic_anchored_linear_strict',
        'quality_control': 'per_minute_density_plus_strict_no_nans',
        'n_workers': n_workers
    }

    pred_metadata_file = Path(prediction_cache_dir) / "prediction_target_metadata.pkl"
    with open(pred_metadata_file, "wb") as f:
        pickle.dump(prediction_metadata, f)

    print(f"\n✅ INTEGRATED IC-ANCHORED PROCESSING COMPLETE:")
    print(f"  Trajectories processed: {processed_trajectories}")
    print(f"  Trajectories skipped: {skipped_trajectories}")
    print(f"  IC tensors created: {sum(len(ics) for ics in all_ics.values())}")
    print(f"  Prediction targets created: {sum(len(preds) for preds in all_predictions.values())}")
    print(f"  Workers used: {n_workers}")
    print(f"  Data quality: {min_points_per_minute} points required EVERY minute")
    print(f"  Interpolation: IC-anchored linear with strict NaN rejection")
    print(f"  Network efficiency: ~1-3 calls per patient vs 20+ in naive approach")
    print(f"  Data consistency: IC and predictions perfectly aligned")
    print(f"  IC metadata saved to: {ic_metadata_file}")
    print(f"  Prediction metadata saved to: {pred_metadata_file}")

    return all_ics, all_predictions


# Example usage
if __name__ == "__main__":
    # Set up paths
    trajectory_metadata_path = "../../data/mimic3refactor/processed_data/med_tensors/trajectory_metadata.pkl"
    relevant_patient_ids_path = "../../data/mimic3refactor/processed_data/relevant_patient_ids.csv"
    icustays_path = "../../data/mimic3refactor/input_data/ICUSTAYS.csv"

    # Create integrated waveform tensors with parallel processing and IC anchoring
    ic_results, prediction_results = create_integrated_waveform_tensors(
        trajectory_metadata_path=trajectory_metadata_path,
        relevant_patient_ids_path=relevant_patient_ids_path,
        icustays_path=icustays_path,
        waveform_base_dir=None,  # Remote access
        ic_cache_dir="../../data/mimic3refactor/processed_data/initial_conditions",
        prediction_cache_dir="../../data/mimic3refactor/processed_data/waveform_prediction_targets",
        prediction_minutes=25,
        target_interval_seconds=5,
        min_points_per_minute=12,  # Require 12 data points in EVERY minute
        n_workers=4  # Use 4 parallel workers
    )