#!/usr/bin/env python3
"""
Integrated MIMIC-III Waveform Processing: IC Tensors + Prediction Targets
Creates both initial condition tensors at t_0 and prediction targets for 25min ahead
Uses efficient patient-batch processing to minimize network calls
Updated with IC-anchored validation and strict per-minute density requirements
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

# --- Project Root and Data Directories ---
# Establish a reliable project root assuming a standard src layout
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Define key data directories using the project root
# This makes path handling robust, regardless of where the script is executed
DATA_DIR = PROJECT_ROOT / "data"
MIMIC_DIR = DATA_DIR / "mimic_3_data"
MIMIC_PROCESSED_DIR = MIMIC_DIR / "processed_data"
PHYSIONET_INPUT_DIR = PROJECT_ROOT / "physionet.org" / "files" / "mimiciii" / "1.4"
DEFAULT_OUTPUT_DIR = DATA_DIR / "mimic_3_data"

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
    # t0 is at the end of the trajectory (end_idx - 1), we take 1min before because that is when our p_tensors end
    t0_interval_idx = trajectory_info['end_idx'] - 2

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


def find_signal_index(record_signal_names, target_signal):
    """Find signal index with flexible name matching"""
    # Create mapping for common variations
    signal_variations = {
        'ABP Mean': ['ABP Mean', 'ABPMean', 'ABP_Mean', 'ABP MEAN', 'ABPMEAN'],
        'HR': ['HR', 'Heart Rate', 'HEART_RATE', 'HeartRate'],
        'CVP': ['CVP', 'Central Venous Pressure', 'CVP_Mean'],
        'SV': ['SV', 'Stroke Volume', 'STROKE_VOLUME'],
        'CO': ['CO', 'Cardiac Output', 'CARDIAC_OUTPUT']
    }

    # Get all possible names for this target signal
    possible_names = signal_variations.get(target_signal, [target_signal])

    # Try to find any of the variations
    for name in possible_names:
        try:
            return record_signal_names.index(name)
        except ValueError:
            continue

    # If no exact match, try partial matching
    for i, record_signal in enumerate(record_signal_names):
        for possible_name in possible_names:
            if possible_name.lower().replace(' ', '').replace('_', '') in record_signal.lower().replace(' ','').replace('_',''):
                return i

    raise ValueError(f"Signal {target_signal} not found")



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
            signal_index = find_signal_index(record.sig_name, signal_name)
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


def extract_ic_and_create_predictions_with_clean_separation(hadm_id, traj_num, t0_timestamp, record_data,
                                                            prediction_minutes, target_interval_seconds,
                                                            min_points_per_minute=12):
    """
    Extract IC and create prediction targets with clean temporal separation

    Strategy:
    1. Extract waveform data from t₀-5min to t₀+prediction_minutes
    2. Find data points closest to t₀ for each signal → IC values (HR, ABP Mean, CVP, SV, CO)
    3. Use only data points AFTER the IC timepoints → prediction targets (ABP Mean, CVP only)
    4. No overlap, clean temporal separation
    5. Calculate derived IC values (R_TPR, missing SV/CO)
    """

    # Extract waveform data from a wider window (buffer before t₀ + full prediction window)
    # Check if data exists within ±30s of t₀ first
    window_seconds = 30
    check_start = t0_timestamp - timedelta(seconds=window_seconds)
    check_end = t0_timestamp + timedelta(seconds=window_seconds)

    # Extract signals from record (all IC signals + prediction signals)
    all_signals = ['HR', 'ABP Mean', 'CVP', 'SV', 'CO']  # All potential IC signals
    prediction_signals = ['ABP Mean', 'CVP']  # Only these for prediction targets

    # First check: Do we have data within ±30s of t₀?
    signals_data = extract_signals_from_record(
        record_data, check_start, check_end, all_signals
    )

    if not signals_data:
        print(f"      ✗ No waveform data found within ±{window_seconds}s of t₀ - REJECTING trajectory")
        return None

    # Check that we have the required prediction signals within ±30s
    missing_prediction_signals = [sig for sig in prediction_signals if sig not in signals_data]
    if missing_prediction_signals:
        print(
            f"      ✗ Missing prediction signals within ±{window_seconds}s: {missing_prediction_signals} - REJECTING trajectory")
        return None

    print(f"      ✓ Found required data within ±{window_seconds}s of t₀")

    # Convert t₀ to record-relative time
    record_start = record_data['info']['start_time']
    t0_offset_seconds = (t0_timestamp - record_start).total_seconds()

    # Define signal categories
    required_ic_signals = ['ABP Mean', 'CVP']
    optional_ic_signals = ['SV', 'CO']

    ic_values = {}
    ic_mask = {}
    ic_timestamps = {}  # Track when each IC was actually measured

    # Step 1: Find the closest waveform values to t₀ for each IC signal
    for signal_name in required_ic_signals + optional_ic_signals:
        if signal_name not in signals_data:
            if signal_name in required_ic_signals:
                print(f"      ✗ Missing required IC signal: {signal_name}")
                return None
            else:
                # Optional signal missing
                ic_values[signal_name] = 0.0
                ic_mask[signal_name] = 0.0
                print(f"      ✗ Optional IC signal {signal_name}: Not found")
                continue

        signal_times = signals_data[signal_name]['times']
        signal_values = signals_data[signal_name]['values']

        # Find the data point closest to t₀
        time_differences = np.abs(signal_times - t0_offset_seconds)
        closest_idx = np.argmin(time_differences)

        closest_time = signal_times[closest_idx]
        closest_value = signal_values[closest_idx]
        time_diff_seconds = abs(closest_time - t0_offset_seconds)

        # Store IC value and its actual timestamp
        ic_values[signal_name] = float(closest_value)
        ic_mask[signal_name] = 1.0
        ic_timestamps[signal_name] = closest_time

        print(f"      ✓ IC {signal_name}: {closest_value:.1f} at {time_diff_seconds:.1f}s from t₀")

    # Check if all required IC signals have data
    missing_required = [sig for sig in required_ic_signals if ic_mask.get(sig, 0) == 0]
    if missing_required:
        print(f"      ✗ Missing required IC signals: {missing_required}")
        return None

    # Calculate derived IC values (same logic as original)
    if ic_mask.get('HR', 0) > 0 and ic_values.get('HR', 0) > 0:
        hr_value = ic_values['HR']

        # If we have CO but missing SV: calculate SV = CO / HR * 1000
        if (ic_mask.get('CO', 0) > 0 and ic_mask.get('SV', 0) == 0 and
                ic_values.get('CO', 0) > 0):
            ic_values['SV'] = (ic_values['CO'] / hr_value) * 1000
            ic_mask['SV'] = 1.0
            print(f"      ✓ IC SV: {ic_values['SV']:.2f} (calculated from CO/HR)")

        # If we have SV but missing CO: calculate CO = SV * HR / 1000
        elif (ic_mask.get('SV', 0) > 0 and ic_mask.get('CO', 0) == 0 and
              ic_values.get('SV', 0) > 0):
            ic_values['CO'] = (ic_values['SV'] * hr_value) / 1000
            ic_mask['CO'] = 1.0
            print(f"      ✓ IC CO: {ic_values['CO']:.2f} (calculated from SV*HR)")

    # Calculate R_TPR
    if (ic_mask.get('ABP Mean', 0) > 0 and ic_mask.get('CVP', 0) > 0 and
            ic_mask.get('CO', 0) > 0 and ic_values.get('CO', 0) > 0):
        rtpr_estimate = (ic_values['ABP Mean'] - ic_values['CVP']) / ic_values['CO']
        if rtpr_estimate > 0:
            ic_values['R_TPR'] = rtpr_estimate
            ic_mask['R_TPR'] = 1.0
        else:
            ic_values['R_TPR'] = 0
            ic_mask['R_TPR'] = 0.0
        print(f"      ✓ IC R_TPR: {ic_values['R_TPR']:.2f} (calculated)")
    else:
        ic_values['R_TPR'] = 0.0
        ic_mask['R_TPR'] = 0.0
        print(f"      ✗ IC R_TPR: Cannot calculate (missing MAP/CVP/CO)")

    # Step 2: Find the latest IC timestamp from PREDICTION signals only (this becomes our separation point)
    # Step 2: Use the closest point to t₀ as our reference timepoint for prediction start
    # Find the closest point to t₀ from prediction signals
    closest_times = []
    for signal_name in prediction_signals:
        signal_times = signals_data[signal_name]['times']
        time_differences = np.abs(signal_times - t0_offset_seconds)
        closest_idx = np.argmin(time_differences)
        closest_times.append(signal_times[closest_idx])

    # Use the latest of the closest points as our reference time (prediction start)
    reference_time = max(closest_times)
    reference_timestamp = record_start + timedelta(seconds=reference_time)

    print(f"      ✓ Reference timepoint: {reference_time - t0_offset_seconds:.1f}s from t₀")
    print(f"      ✓ Predictions will start from this timepoint")

    # Step 3: Extract prediction data starting from reference timepoint
    prediction_end = reference_timestamp + timedelta(minutes=prediction_minutes)
    prediction_signals_data = extract_signals_from_record(
        record_data, reference_timestamp, prediction_end, prediction_signals
    )

    if not prediction_signals_data:
        print(f"      ✗ No prediction data available from reference timepoint")
        return None

    print(f"      ✓ Extracted prediction data from reference timepoint")

    # Step 4: Check data density per minute for prediction window (full duration expected)
    actual_prediction_duration = prediction_minutes
    prediction_start_time = reference_time
    print(f"      ✓ Prediction duration: {actual_prediction_duration} minutes from reference timepoint")

    # STRICT DATA DENSITY CHECK: Require min_points_per_minute in EVERY minute of prediction window
    total_minutes = int(actual_prediction_duration)
    for signal_name in prediction_signals:
        if signal_name not in prediction_signals_data:
            continue

        signal_times = np.unique(prediction_signals_data[signal_name]['times'])

        # Check each individual minute in the prediction window
        minutes_failed = []
        for minute_idx in range(total_minutes):
            minute_start = prediction_start_time + (minute_idx * 60)
            minute_end = prediction_start_time + ((minute_idx + 1) * 60)

            # Count data points in this specific minute
            points_in_minute = np.sum((signal_times >= minute_start) & (signal_times < minute_end))

            if points_in_minute < min_points_per_minute:
                minutes_failed.append({
                    'minute': minute_idx,
                    'points': points_in_minute,
                    'required': min_points_per_minute
                })

        if minutes_failed:
            print(f"      ✗ REJECTING trajectory {traj_num}: {signal_name} fails strict per-minute requirement")
            print(f"        {len(minutes_failed)} out of {total_minutes} minutes have insufficient data")
            for failure in minutes_failed[:3]:
                print(
                    f"        Minute {failure['minute']}: {failure['points']} points < {failure['required']} required")
            if len(minutes_failed) > 3:
                print(f"        ... and {len(minutes_failed) - 3} more minutes fail")
            print(f"        STRICT POLICY: ALL {total_minutes} minutes must have >= {min_points_per_minute} points")
            return None
        else:
            total_points = len(signal_times)
            avg_density = total_points / total_minutes if total_minutes > 0 else 0
            print(
                f"      ✓ {signal_name}: All {total_minutes} minutes pass density check (avg: {avg_density:.1f} points/min)")

    # Step 5: Create target time grid for prediction starting from reference time + one interval (do not include ic in prediction)
    n_prediction_points = int((actual_prediction_duration * 60) / target_interval_seconds) - 1
    target_times_seconds = np.array([
        reference_time + target_interval_seconds + (i * target_interval_seconds)
        for i in range(n_prediction_points)
    ])

    # Step 6: Interpolate prediction signals to target grid
    interpolated = {}
    for signal_name in prediction_signals:
        if signal_name not in prediction_signals_data:
            continue

        times = prediction_signals_data[signal_name]['times']
        values = prediction_signals_data[signal_name]['values']

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

            # Interpolate to target times
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

    if len(interpolated) != len(prediction_signals):
        print(
            f"      ✗ Only {len(interpolated)}/{len(prediction_signals)} prediction signals successfully interpolated")
        return None

    print(
        f"      ✓ Clean temporal separation: IC at t₀ → prediction starts {reference_time - t0_offset_seconds:.1f}s later")

    # Step 7: Create tensor arrays
    signal_names = list(interpolated.keys())
    n_signals = len(signal_names)

    values_array = np.zeros((n_prediction_points, n_signals), dtype=np.float32)
    mask_array = np.ones((n_prediction_points, n_signals), dtype=np.float32)  # All 1s since no NaNs

    for i, signal_name in enumerate(signal_names):
        values_array[:, i] = interpolated[signal_name]

    # Convert to tensors
    values_tensor = torch.from_numpy(values_array).float()
    mask_tensor = torch.from_numpy(mask_array).float()

    # Time tensor (seconds from actual prediction start, not from t₀)
    time_seconds_from_prediction_start = np.array([(i+1) * target_interval_seconds for i in range(n_prediction_points)])
    time_tensor = torch.from_numpy(time_seconds_from_prediction_start).float()

    return {
        'hadm_id': hadm_id,
        'trajectory_num': traj_num,
        # IC data
        'ic_values': ic_values,
        'ic_mask': ic_mask,
        't0_timestamp': t0_timestamp,
        'record_used': f"{record_data['info']['patient_path']}/{record_data['info']['record_name']}",
        # Prediction data
        'tensor_data': (values_tensor, mask_tensor, time_tensor, n_prediction_points),
        'metadata': {
            'record_used': f"{record_data['info']['patient_path']}/{record_data['info']['record_name']}",
            'sampling_rate': record_data['info']['sampling_rate'],
            'signals_used': signal_names,
            'temporal_separation_method': 'clean_ic_prediction_split',
            'ic_timestamps': ic_timestamps,
            'separation_time_offset': reference_time - t0_offset_seconds,
            'actual_prediction_duration_minutes': actual_prediction_duration,
            'min_points_per_minute': min_points_per_minute,
            'density_check_method': 'per_minute_strict_post_separation',
            't0_timestamp': t0_timestamp,
            'prediction_start_timestamp': reference_timestamp,
            'reference_time_offset_from_t0': reference_time - t0_offset_seconds,
            'interpolation_method': 'linear_clean_separation'
        }
    }


def save_full_trajectory_data(
        hadm_id, record_data, all_patient_t0s, icu_admission_time, full_trajectory_dir,
        resample_interval_seconds=10
):
    """
    Saves the complete, resampled waveform data for a single waveform record,
    annotating it with all relevant t0 timestamps.

    Args:
        hadm_id (int): The hospital admission ID.
        record_data (dict): The loaded waveform record data.
        all_patient_t0s (list): A list of all t0 datetime objects for this patient.
        icu_admission_time (datetime): The patient's ICU admission time.
        full_trajectory_dir (str or Path): Directory to save the full trajectory file.
        resample_interval_seconds (int): The interval in seconds for resampling.
    """
    if record_data is None:
        return

    record = record_data['record']
    record_info = record_data['info']
    all_signals = record.sig_name
    record_start = record_info['start_time']
    record_end = record_info['end_time']

    # Extract all signals from the entire record
    signals_data = extract_signals_from_record(
        record_data, record_start, record_end, all_signals
    )

    if not signals_data:
        print(f"      [Full Traj] No signals data found for record {record_info['record_name']}")
        return

    # Create a unified time grid for the entire record duration
    duration_seconds = record_info['duration_seconds']
    n_points = int(duration_seconds / resample_interval_seconds)
    target_times_seconds = np.arange(n_points) * resample_interval_seconds

    # Interpolate all signals to this grid
    interpolated_signals = interpolate_signals_to_grid(signals_data, target_times_seconds)

    if not interpolated_signals:
        print(f"      [Full Traj] Interpolation failed for record {record_info['record_name']}")
        return

    # Create a DataFrame
    df_data = {'time_seconds': target_times_seconds}
    for signal_name, values in interpolated_signals.items():
        df_data[signal_name] = values

    df = pd.DataFrame(df_data)

    # Find which of the patient's t0s fall within this record's timeframe
    t0s_in_record = [t0 for t0 in all_patient_t0s if record_start <= t0 <= record_end]

    # Add rich metadata to the DataFrame for context
    df['record_name'] = record_info['record_name']
    df['record_start_time'] = record_info['start_time']
    df['record_end_time'] = record_info['end_time']
    df['hadm_id'] = hadm_id
    df['icu_admission_time'] = icu_admission_time
    # Store the list of t0s in the column. Each cell will contain the same list.
    df['t0_timestamps'] = [t0s_in_record] * len(df)

    # Save to parquet
    output_file = Path(full_trajectory_dir) / f"full_waveform_{int(hadm_id)}_{record_info['record_name']}.parquet"
    df.to_parquet(output_file, index=False)
    print(f"      [Full Traj] Saved full waveform record to {output_file} with {len(t0s_in_record)} t0s.")


def process_patient_integrated_clean_separation(hadm_id, patient_id, patient_trajectories, icu_admission_time,
                                                interval_minutes, prediction_minutes, target_interval_seconds,
                                                waveform_cache, database='mimic3wdb-matched', min_points_per_minute=12,
                                                save_full_trajectory=False, full_trajectory_dir=None):
    """
    Processes a patient's data.
    1. If enabled, saves ALL discovered waveform records for the patient.
    2. Generates t0-centric initial conditions and prediction targets.
    """
    print(f"\nProcessing patient {patient_id} (HADM {hadm_id})")

    # Step 1: Discover all waveform records for this patient
    patient_records = waveform_cache.discover_patient_records(patient_id, database=database)
    if not patient_records:
        print(f"  No waveform records found for patient {patient_id}")
        return [], []

    # Step 2: Save all full waveform trajectories if requested
    if save_full_trajectory and full_trajectory_dir:
        print(f"  Saving all {len(patient_records)} discovered waveform records...")
        
        all_t0_timestamps = [
            calculate_t0_timestamp(t, icu_admission_time, interval_minutes)
            for t in patient_trajectories
        ]

        for record_info in patient_records:
            output_file = Path(full_trajectory_dir) / f"full_waveform_{int(hadm_id)}_{record_info['record_name']}.parquet"
            if not output_file.exists():
                print(f"    [Full Traj] Saving new record {record_info['record_name']}.")
                record_data = waveform_cache.load_record_data(record_info, database)
                if record_data:
                    save_full_trajectory_data(
                        hadm_id=hadm_id,
                        record_data=record_data,
                        all_patient_t0s=all_t0_timestamps,
                        icu_admission_time=icu_admission_time,
                        full_trajectory_dir=full_trajectory_dir,
                        resample_interval_seconds=target_interval_seconds
                    )
            else:
                debug_print(f"    [Full Traj] Record {record_info['record_name']} already exists. Skipping.")

    # Step 3: Generate IC and Prediction Targets (t0-centric)
    print(f"  Generating ICs and Prediction Targets for {len(patient_trajectories)} t0-based trajectories...")
    patient_ics = []
    patient_predictions = []
    processed_count = 0
    skipped_count = 0

    for traj in patient_trajectories:
        traj_num = traj['trajectory_num']
        try:
            t0_timestamp = calculate_t0_timestamp(traj, icu_admission_time, interval_minutes)
            
            window_start = t0_timestamp - timedelta(seconds=30)
            window_end = t0_timestamp + timedelta(minutes=prediction_minutes)

            overlapping_records = waveform_cache.find_records_for_timerange(
                patient_id, window_start, window_end
            )

            if not overlapping_records:
                print(f"    ✗ Skipping t0-trajectory {traj_num} - no records overlap with window")
                skipped_count += 1
                continue

            if len(overlapping_records) > 1:
                print(f"    ✗ Skipping t0-trajectory {traj_num} - window spans {len(overlapping_records)} records")
                skipped_count += 1
                continue

            record = overlapping_records[0]
            record_data = waveform_cache.load_record_data(record, database)
            if record_data is None:
                print(f"    ✗ Failed to load record data for t0-trajectory {traj_num}")
                skipped_count += 1
                continue
            
            result = extract_ic_and_create_predictions_with_clean_separation(
                hadm_id, traj_num, t0_timestamp, record_data,
                prediction_minutes, target_interval_seconds,
                min_points_per_minute
            )

            if result:
                ic_result = {
                    'hadm_id': result['hadm_id'],
                    'trajectory_num': result['trajectory_num'],
                    't0_timestamp': result['t0_timestamp'],
                    'ic_values': result['ic_values'],
                    'ic_mask': result['ic_mask'],
                    'record_used': result['record_used']
                }
                prediction_result = {
                    'hadm_id': result['hadm_id'],
                    'trajectory_num': result['trajectory_num'],
                    'tensor_data': result['tensor_data'],
                    'metadata': result['metadata']
                }
                patient_ics.append(ic_result)
                patient_predictions.append(prediction_result)
                processed_count += 1
            else:
                skipped_count += 1

        except Exception as e:
            print(f"    Error processing t0-trajectory {traj_num}: {e}")
            skipped_count += 1
            continue

    print(f"  Created {len(patient_ics)} IC tensors and {len(patient_predictions)} prediction targets")
    return patient_ics, patient_predictions


def process_patient_and_save(
        hadm_id, patient_id, patient_trajectories, icu_admission_time, interval_minutes,
        prediction_minutes, target_interval_seconds, waveform_cache, database, ic_cache_dir, prediction_cache_dir,
        min_points_per_minute=12, save_full_trajectory=False, full_trajectory_dir=None, skip_existing=True
):
    debug_print(f"[Worker PID: {os.getpid()}] Processing patient HADM_ID: {hadm_id}")
    patient_ics, patient_predictions = process_patient_integrated_clean_separation(
        hadm_id, patient_id, patient_trajectories, icu_admission_time,
        interval_minutes, prediction_minutes, target_interval_seconds,
        waveform_cache, database, min_points_per_minute,
        save_full_trajectory=save_full_trajectory,
        full_trajectory_dir=full_trajectory_dir
    )

    ic_info = []
    for ic_result in patient_ics:
        traj_num = ic_result['trajectory_num']
        ic_file = Path(ic_cache_dir) / f"ic_tensor_{int(hadm_id)}_traj_{traj_num:03d}.pt"

        # Granular skipping: only write file if it doesn't exist
        if skip_existing and ic_file.exists():
            debug_print(f"  [Worker PID: {os.getpid()}] Skipping existing IC tensor for traj {traj_num}")
            continue

        debug_print(f"  [Worker PID: {os.getpid()}] Saving IC tensor for traj {traj_num}")
        physio_values = [
            ic_result['ic_values']['ABP Mean'],  # p_a
            ic_result['ic_values']['CVP'],  # p_v
            0.5,  # s_reflex (baroreflex sensitivity - normal baseline)
            ic_result['ic_values'].get('SV', 70.0),  # sv (default ~70ml if missing)
            0.0  # r_tpr_mod (TPR modifier starts at 0)
        ]

        physio_masks = [
            ic_result['ic_mask']['ABP Mean'],  # p_a mask
            ic_result['ic_mask']['CVP'],  # p_v mask
            0,  # s_reflex mask (never available)
            ic_result['ic_mask'].get('SV', 0.0),  # sv mask
            0  # r_tpr_mod mask (never available)
        ]
        ic_tensor = torch.tensor(physio_values, dtype=torch.float32)
        ic_mask_tensor = torch.tensor(physio_masks, dtype=torch.float32)
        torch.save((ic_tensor, ic_mask_tensor), ic_file)
        ic_info.append({
            'hadm_id': hadm_id, 'trajectory_num': traj_num, 't0_timestamp': ic_result['t0_timestamp'],
            'file_path': str(ic_file), 'ic_values': ic_result['ic_values'], 'record_used': ic_result['record_used']
        })

    prediction_info = []
    for prediction in patient_predictions:
        traj_num = prediction['trajectory_num']
        pred_file = Path(prediction_cache_dir) / f"prediction_target_{int(hadm_id)}_traj_{traj_num:03d}.pt"

        # Granular skipping: only write file if it doesn't exist
        if skip_existing and pred_file.exists():
            debug_print(f"  [Worker PID: {os.getpid()}] Skipping existing prediction tensor for traj {traj_num}")
            continue

        debug_print(f"  [Worker PID: {os.getpid()}] Saving prediction tensor for traj {traj_num}")
        torch.save(prediction['tensor_data'], pred_file)
        prediction_info.append({
            'hadm_id': hadm_id, 'trajectory_num': traj_num, 'file_path': str(pred_file),
            'metadata': prediction['metadata']
        })

    debug_print(f"[Worker PID: {os.getpid()}] Finished saving for patient {hadm_id}. Returning metadata.")
    return hadm_id, ic_info, prediction_info


def create_integrated_waveform_tensors(
        trajectory_metadata_path,
        relevant_patient_ids_path,
        icustays_path,
        waveform_base_dir=None,
        ic_cache_dir='initial_conditions_waveform',
        prediction_cache_dir='waveform_prediction_targets',
        full_trajectory_dir='full_trajectories',
        database='mimic3wdb-matched',
        prediction_minutes=25,
        target_interval_seconds=10,
        min_points_per_minute=12,
        skip_existing=True,
        n_workers=4,
        save_full_trajectory=False,
):
    debug_print("--- Starting Integrated Waveform Tensor Creation with IC-Anchored Validation ---")
    if not setup_wfdb_credentials():
        return None, None

    Path(ic_cache_dir).mkdir(parents=True, exist_ok=True)
    Path(prediction_cache_dir).mkdir(parents=True, exist_ok=True)
    if save_full_trajectory:
        Path(full_trajectory_dir).mkdir(parents=True, exist_ok=True)

    debug_print(f"Loading trajectory metadata from: {trajectory_metadata_path}")
    traj_metadata, _ = load_trajectory_metadata(trajectory_metadata_path)
    patient_mapping = get_patient_mapping(relevant_patient_ids_path)
    icu_admission_map = get_icu_admission_times(icustays_path)
    all_trajectories = traj_metadata['all_trajectories']
    interval_minutes = traj_metadata['interval_minutes']
    debug_print(f"Loaded metadata for {len(all_trajectories)} patients.")

    print(f"Data quality requirements:")
    print(f"- Min {min_points_per_minute} data points required in EVERY minute")
    print(f"- ANY minute with insufficient data = trajectory rejected")
    print(f"- IC values used as anchor points for interpolation")
    print(f"- Zero tolerance for NaN values after interpolation")

    all_ics = {}
    all_predictions = {}
    tasks = []
    skipped_patients = 0
    waveform_cache = WaveformRecordCache()

    debug_print("Preparing tasks for parallel processing...")
    for hadm_id in list(all_trajectories.keys())[:200]:  # TODO: remove this limit
        if hadm_id not in patient_mapping or hadm_id not in icu_admission_map:
            continue

        # If not saving full trajectories, we can use the faster patient-level skip.
        if skip_existing and not save_full_trajectory:
            all_files_exist = True
            for traj in all_trajectories[hadm_id]:
                traj_num = traj['trajectory_num']
                ic_file = Path(ic_cache_dir) / f"ic_tensor_{int(hadm_id)}_traj_{traj_num:03d}.pt"
                pred_file = Path(prediction_cache_dir) / f"prediction_target_{int(hadm_id)}_traj_{traj_num:03d}.pt"
                if not (ic_file.exists() and pred_file.exists()):
                    all_files_exist = False
                    break
            if all_files_exist:
                debug_print(f"Skipping patient {hadm_id} - all IC/prediction files exist (fast path).")
                skipped_patients += 1
                continue

        # When saving full trajectories, we delegate skipping to the worker to handle
        # the more complex, file-level de-duplication logic.
        patient_id = patient_mapping[hadm_id]
        icu_admission_time = icu_admission_map[hadm_id]
        patient_trajectories = all_trajectories[hadm_id]
        tasks.append(
            (hadm_id, patient_id, patient_trajectories, icu_admission_time, interval_minutes,
             prediction_minutes, target_interval_seconds, waveform_cache, database, ic_cache_dir, prediction_cache_dir,
             min_points_per_minute, save_full_trajectory, full_trajectory_dir, skip_existing)
        )

    debug_print(f"Submitting {len(tasks)} tasks to {n_workers} workers.")
    with concurrent.futures.ProcessPoolExecutor(max_workers=n_workers) as executor:
        future_to_hadm = {executor.submit(process_patient_and_save, *task): task[0] for task in tasks}

        for future in tqdm(concurrent.futures.as_completed(future_to_hadm), total=len(tasks),
                           desc="Processing patients"):
            hadm_id, ic_info, prediction_info = future.result()
            if ic_info:
                all_ics[hadm_id] = ic_info
            if prediction_info:
                all_predictions[hadm_id] = prediction_info

    debug_print("Aggregating final metadata...")
    ic_metadata = {
        'all_initial_conditions': all_ics,
        'total_ics': sum(len(ics) for ics in all_ics.values()),
        'parameters': ['HR', 'ABP Mean', 'CVP', 'SV', 'R_TPR'],
        'extraction_method': 'waveform_clean_temporal_separation',
        'separation_strategy': 'closest_to_t0_then_chronological_split',
        'min_points_per_minute': min_points_per_minute,
        'interval_minutes': interval_minutes
    }
    ic_metadata_file = Path(ic_cache_dir) / "ic_metadata.pkl"
    with open(ic_metadata_file, "wb") as f:
        pickle.dump(ic_metadata, f)

    prediction_metadata = {
        'all_prediction_targets': all_predictions,
        'signal_names': ['ABP Mean', 'CVP'],
        'n_signals': 2,
        'prediction_minutes': prediction_minutes,
        'target_interval_seconds': target_interval_seconds,
        'min_points_per_minute': min_points_per_minute,
        'total_trajectories': sum(len(preds) for preds in all_predictions.values()),
        'source': 'waveform_clean_temporal_separation',
        'interpolation_method': 'linear_post_separation',
        'quality_control': 'per_minute_density_strict_no_overlap',
        'separation_strategy': 'ic_closest_to_t0_predictions_after'
    }
    pred_metadata_file = Path(prediction_cache_dir) / "prediction_target_metadata.pkl"
    with open(pred_metadata_file, "wb") as f:
        pickle.dump(prediction_metadata, f)

    print(f"\n✅ IC-ANCHORED INTEGRATED PROCESSING COMPLETE:")
    print(f"  New patients processed: {len(all_ics)}")
    print(f"  Patients skipped: {skipped_patients}")
    print(f"  IC tensors created: {sum(len(ics) for ics in all_ics.values())}")
    print(f"  Strategy: IC closest to t₀ → predictions chronologically after")
    print(f"  Quality: ZERO overlap, strict per-minute density, no NaN tolerance")
    print(f"  Validation method: Clean separation with {min_points_per_minute} points/min")
    print(f"  Validation method: IC-anchored interpolation with {min_points_per_minute} points/min")
    print(f"  IC metadata saved to: {ic_metadata_file}")
    print(f"  Prediction metadata saved to: {pred_metadata_file}")

    return all_ics, all_predictions


# Example usage
if __name__ == "__main__":
    # Set up paths using the robust constants defined at the top of the script
    trajectory_metadata_path = DEFAULT_OUTPUT_DIR / "processed_data"/"med_tensors" / "trajectory_metadata.pkl"
    relevant_patient_ids_path = DEFAULT_OUTPUT_DIR / "processed_data"/"relevant_patient_ids.csv"
    icustays_path = DATA_DIR /"mimic_3_data"/"input_data" / "ICUSTAYS.csv"
    ic_cache_dir = DEFAULT_OUTPUT_DIR / "processed_data"/ "initial_conditions"
    prediction_cache_dir = DEFAULT_OUTPUT_DIR / "processed_data"/"prediction_targets"
    full_trajectory_dir = DEFAULT_OUTPUT_DIR / "processed_data" / "full_trajectories"

    # Create integrated waveform tensors with IC-anchored validation
    ic_results, prediction_results = create_integrated_waveform_tensors(
        trajectory_metadata_path=trajectory_metadata_path,
        relevant_patient_ids_path=relevant_patient_ids_path,
        icustays_path=icustays_path,
        waveform_base_dir=None,  # Remote access
        ic_cache_dir=ic_cache_dir,
        prediction_cache_dir=prediction_cache_dir,
        full_trajectory_dir=full_trajectory_dir,
        prediction_minutes=25,
        target_interval_seconds=10,
        min_points_per_minute=6,
        n_workers=6,
        save_full_trajectory=True
    )