import argparse
import concurrent.futures
import gc
import json
import os
import pickle
from datetime import datetime, timedelta
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import torch
import wfdb
from scipy.interpolate import interp1d
from scipy.spatial.distance import cdist
from tqdm import tqdm


# --- Project Root and Data Directories ---
# Establish a reliable project root assuming a standard src layout
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

# Define key data directories using the project root
# This makes path handling robust, regardless of where the script is executed
DATA_DIR = PROJECT_ROOT / "data"
MIMIC_DIR = DATA_DIR / "mimic_3_data"
MIMIC_INPUT_DIR = MIMIC_DIR / "input_data"
MIMIC_PROCESSED_DIR = MIMIC_DIR / "processed_data"
PHYSIONET_INPUT_DIR = PROJECT_ROOT / "physionet.org" / "files" / "mimiciii" / "1.4"
#MIMIC_INPUT_DIR = PHYSIONET_INPUT_DIR
DEFAULT_OUTPUT_DIR = DATA_DIR / "mimic_3_data" / "processed_data"

# Ensure processed data directories exist
(DEFAULT_OUTPUT_DIR / "med_tensors").mkdir(parents=True, exist_ok=True)
(DEFAULT_OUTPUT_DIR / "p_tensors").mkdir(parents=True, exist_ok=True)
(DEFAULT_OUTPUT_DIR / "prediction_targets").mkdir(parents=True, exist_ok=True)
(DEFAULT_OUTPUT_DIR / "baseline_tensors").mkdir(parents=True, exist_ok=True)
MIMIC_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# --- Debug Setup ---
DEBUG = os.environ.get('DEBUG', 'False').lower() in ('true', '1', 't')

def debug_print(s):
    if DEBUG:
        print(s)
# --- End Debug Setup ---



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
    parser.add_argument('--debug-patient-id', type=int, default=20214994,
                        help='Patient ID to save debug CSV files for (set automatically if not provided)')

    # Data limit parameter (for testing)
    parser.add_argument('--data-limit', type=int, default=None,
                        help='Limit on number of rows to read from CSV files (default: None for unlimited)')
    parser.add_argument('--patient-limit', type=int, default=None,
                        help='Limit number of patients to process for faster testing (default: None)')

    # Output directories
    parser.add_argument('--output-dir', type=str, default=str(DEFAULT_OUTPUT_DIR),
                        help=f'Base output directory (default: {DEFAULT_OUTPUT_DIR})')
    parser.add_argument('--input-dir', type=str, default=str(PHYSIONET_INPUT_DIR),
                        help=f'Base input directory (default: {PHYSIONET_INPUT_DIR})')

    # MIMIC-III specific parameters
    parser.add_argument('--hadm-filter-file', type=str, default='results_with_hadm_id.csv',
                        help='CSV file containing hadm_id column for filtering (default: results_with_hadm_id.csv)')

    return parser.parse_args()


def find_relevant_patients(
    waveform_patient_ids,
    measurements,
    MAP_id,
    load_path_events=MIMIC_INPUT_DIR / "CHARTEVENTS.csv",
    load_path_stays=MIMIC_INPUT_DIR / "ICUSTAYS.csv",
    save_path=MIMIC_PROCESSED_DIR / "treated_patients_chartevents.parquet",
    data_limit=None
):
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

        long_stays_query_filtered = (
            long_stays_query.filter(pl.col("LOS") > 1)
            .filter(pl.col("SUBJECT_ID").is_in(waveform_patient_ids))
            .select("HADM_ID")
            .unique()
        )

        # 2. Join the lazy queries and then apply the rest of the filters.
        treated_patients = (
            treated_patients_query.join(
                long_stays_query_filtered, on="HADM_ID", how="inner"
            )
            .filter(
                (pl.col("ITEMID").is_in(MAP_id))
                & (pl.col("VALUE") != "Not Given")
                & (pl.col("VALUE").cast(pl.Float64, strict=False) < 70)
            )
            .collect()
        )

        treated_patients_all_values = read_large_csv_with_polars(load_path_events, treated_patients, measurements,
                                                                 data_limit=data_limit,
                                                                 schema_overrides=chartevents_schema_overrides)
        treated_patients_all_values.write_parquet(save_path)
        print(f"Saved new dataset of patient values to {save_path}")
    else:
        print(f"Loading dataset from {save_path}")
        treated_patients_all_values = pl.read_parquet(save_path)

    return treated_patients_all_values

def read_large_csv_with_polars(load_path, ids_df, measurements, id_column='HADM_ID', item_column='ITEMID',
                                   data_limit=None, schema_overrides=None):
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

        query = pl.scan_csv(load_path, schema_overrides=schema_overrides)

        if data_limit is not None:
            query = query.limit(data_limit)
        result = (
            query.filter(pl.col(id_column).is_in(valid_ids))
            .filter(pl.col(item_column).is_in(measurements))
            .collect()
        )

        return result



def get_mimic3_item_ids():
    """
    Get MIMIC-III item IDs for medications and physiological measurements.

    Returns:
        Dictionary containing item ID mappings for MIMIC-III
    """
    return {
        # Physiological measurements (CHARTEVENTS)
        'hr': [211, 220045],  # Heart Rate
        'map': [52, 6702, 443, 6926],  # Mean Arterial Pressure
        'cvp': [113, 220074, 1103],  # Central Venous Pressure
        'sv': [662, 228374],  # Stroke Volume (if available)
        'co': [224842, 44920, 44970, 41946, 40909,228369, 220088],  # Cardiac Output related

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
def process_single_patient_physio(
        hadm_id,
        physio_df, # Pass the entire DataFrame
        physio_params,
        co_itemids,  # Need CO itemids for r_tpr calculation
        n_intervals,
        interval_minutes,
        icu_admission_time,
        trajectory_boundaries,
        cache_dir,
        save_prediction_targets=False,
        prediction_target_dir=DEFAULT_OUTPUT_DIR / "prediction_targets"
):
    """
    Process physiological measurements for a single patient and save as trajectory tensors.
    Vectorized version for high performance.
    """
    debug_print(f"  [Physio] Processing patient {hadm_id} with vectorized logic.")

    # 1. Filter the in-memory DataFrame for the specific patient
    chart_df_patient = physio_df.filter(pl.col('HADM_ID') == hadm_id)
    if chart_df_patient.height == 0:
        debug_print(f"  [Physio] No chart events found for patient {hadm_id}. Skipping.")
        return hadm_id, []

    debug_print(f"  [Physio] Found {chart_df_patient.height} chart events for patient {hadm_id}.")

    # 2. Group by time interval and itemid, calculating the mean value at once.
    # This is the core vectorization step, replacing the slow iter_rows loop.
    debug_print(f"  [Physio] Performing grouped aggregation on chart events...")
    aggregated_df = chart_df_patient.group_by(['time_idx', 'ITEMID']).agg(
        pl.mean('VALUE').alias('mean_value')
    )
    debug_print(f"  [Physio] Aggregation complete. Result shape: {aggregated_df.shape}")

    # 3. Pivot the data to a "wide" format.
    # Rows are time intervals, columns are itemids, values are the mean measurements.
    debug_print(f"  [Physio] Pivoting data to wide format...")
    pivoted_df = aggregated_df.pivot(
        index='time_idx',
        columns='ITEMID',
        values='mean_value'
    )
    debug_print(f"  [Physio] Pivoting complete. Pivoted shape: {pivoted_df.shape}")

    # Initialize arrays for all physiological parameters
    n_params = len(physio_params)
    values_array = np.zeros((n_intervals, n_params), dtype=np.float32)
    mask_array = np.zeros((n_intervals, n_params), dtype=np.float32)

    # 4. Efficiently populate the numpy arrays from the pivoted DataFrame.
    debug_print(f"  [Physio] Populating NumPy arrays from pivoted data...")
    all_item_ids_in_pivot = pivoted_df.columns
    time_indices_in_pivot = pivoted_df['time_idx'].to_numpy()

    hr_idx, sv_idx, map_idx, cvp_idx, r_tpr_idx = -1, -1, -1, -1, -1
    for i, param in enumerate(physio_params):
        param_type = param['param_type']
        if param_type == 'HR': hr_idx = i
        elif param_type == 'SV': sv_idx = i
        elif param_type == 'MAP': map_idx = i
        elif param_type == 'CVP': cvp_idx = i
        elif param_type == 'R_TPR': r_tpr_idx = i

        itemid_str = str(param['itemid'])
        if itemid_str in all_item_ids_in_pivot:
            # Get values and identify where they are not null
            col_values = pivoted_df[itemid_str].to_numpy()
            valid_mask = ~np.isnan(col_values)

            # Use the boolean mask to get the indices and values
            valid_indices = time_indices_in_pivot[valid_mask]
            valid_values = col_values[valid_mask]

            # Place the valid values and masks into the final arrays at the correct time indices
            values_array[valid_indices, i] = valid_values
            mask_array[valid_indices, i] = 1.0
            debug_print(f"    - Populated '{param_type}' ({param['param_name']}) with {len(valid_values)} values.")

    # Also need to track CO values for r_tpr calculation
    co_values_array = np.zeros(n_intervals, dtype=np.float32)
    co_mask_array = np.zeros(n_intervals, dtype=np.float32)

    for co_id in co_itemids:
        co_id_str = str(co_id)
        if co_id_str in all_item_ids_in_pivot:
            col_values = pivoted_df[co_id_str].to_numpy()
            valid_mask = ~np.isnan(col_values)
            valid_indices = time_indices_in_pivot[valid_mask]
            valid_values = col_values[valid_mask]

            # In case of multiple CO types, we just take the first one found for simplicity
            co_values_array[valid_indices] = valid_values
            co_mask_array[valid_indices] = 1.0
    debug_print(f"    - Populated Cardiac Output with {int(co_mask_array.sum())} values.")

    # 5. Vectorized calculation of derived values
    debug_print(f"  [Physio] Performing vectorized calculations for derived parameters (SV/CO, R_TPR)...")
    if hr_idx != -1 and sv_idx != -1:
        hr_present_mask = (mask_array[:, hr_idx] > 0) & (values_array[:, hr_idx] > 0)

        # Vectorized calculation for SV from CO
        sv_calc_mask = hr_present_mask & (co_mask_array > 0) & (mask_array[:, sv_idx] == 0) & (co_values_array > 0)
        if np.any(sv_calc_mask):
            values_array[sv_calc_mask, sv_idx] = (co_values_array[sv_calc_mask] / values_array[sv_calc_mask, hr_idx])*1000
            mask_array[sv_calc_mask, sv_idx] = 1.0
            debug_print(f"    - Calculated {np.sum(sv_calc_mask)} SV values from CO/HR.")

        # Vectorized calculation for CO from SV
        co_calc_mask = hr_present_mask & (mask_array[:, sv_idx] > 0) & (co_mask_array == 0) & (values_array[:, sv_idx] > 0)
        if np.any(co_calc_mask):
            co_values_array[co_calc_mask] = (values_array[co_calc_mask, sv_idx] * values_array[co_calc_mask, hr_idx])/1000
            co_mask_array[co_calc_mask] = 1.0
            debug_print(f"    - Calculated {np.sum(co_calc_mask)} CO values from SV*HR.")

    if r_tpr_idx != -1 and map_idx != -1 and cvp_idx != -1:
        # Vectorized calculation for R_TPR
        rtpr_calc_mask = (mask_array[:, map_idx] == 1) & (mask_array[:, cvp_idx] == 1) & (co_mask_array == 1) & (co_values_array > 0)
        if np.any(rtpr_calc_mask):
            map_vals = values_array[rtpr_calc_mask, map_idx]
            cvp_vals = values_array[rtpr_calc_mask, cvp_idx]
            co_vals = co_values_array[rtpr_calc_mask]

            values_array[rtpr_calc_mask, r_tpr_idx] = (map_vals - cvp_vals) / co_vals
            mask_array[rtpr_calc_mask, r_tpr_idx] = 1.0
            debug_print(f"    - Calculated {np.sum(rtpr_calc_mask)} R_TPR values.")

    # The rest of the function (trajectory slicing and saving) remains largely the same
    # as it operates on NumPy arrays which are already prepared.
    debug_print(f"  [Physio] Slicing arrays into trajectories and saving tensors...")
    rel_time_array = np.arange(n_intervals) * interval_minutes / 60.0
    abs_time_array = rel_time_array.copy()
    trajectory_info = []
    prediction_target_info = []

    for traj_num in range(len(trajectory_boundaries) - 1):
        start_idx_in, end_idx_in = trajectory_boundaries[traj_num]
        _, end_idx_out = trajectory_boundaries[traj_num + 1]

        # P_IN
        p_in_values = values_array[start_idx_in:end_idx_in, :]
        p_in_mask = mask_array[start_idx_in:end_idx_in, :]
        p_in_abs_time = abs_time_array[start_idx_in:end_idx_in]
        p_in_rel_time = rel_time_array[start_idx_in:end_idx_in]
        p_in_len = end_idx_in - start_idx_in

        p_in_values_tensor = torch.from_numpy(p_in_values).float()
        p_in_mask_tensor = torch.from_numpy(p_in_mask).float()
        p_in_abs_time_tensor = torch.from_numpy(p_in_abs_time).float()
        p_in_rel_time_tensor = torch.from_numpy(p_in_rel_time).float()

        file_path_in = os.path.join(cache_dir, f"p_tensor_in_{int(hadm_id)}_traj_{traj_num:03d}.pt")
        torch.save(
            (p_in_values_tensor, p_in_mask_tensor, p_in_abs_time_tensor, p_in_rel_time_tensor, p_in_len),
            file_path_in
        )

        # P_OUT
        p_out_values = values_array[end_idx_in:end_idx_out, :]
        p_out_mask = mask_array[end_idx_in:end_idx_out, :]
        p_out_abs_time = abs_time_array[end_idx_in:end_idx_out]
        p_out_rel_time = rel_time_array[end_idx_in:end_idx_out]
        p_out_len = end_idx_out - end_idx_in

        p_out_values_tensor = torch.from_numpy(p_out_values).float()
        p_out_mask_tensor = torch.from_numpy(p_out_mask).float()
        p_out_abs_time_tensor = torch.from_numpy(p_out_abs_time).float()
        p_out_rel_time_tensor = torch.from_numpy(p_out_rel_time).float()

        if save_prediction_targets and prediction_target_dir is not None:
            if map_idx != -1 and cvp_idx != -1:
                pred_values = torch.stack([p_out_values_tensor[:, map_idx], p_out_values_tensor[:, cvp_idx]], dim=1)
                pred_mask = torch.stack([p_out_mask_tensor[:, map_idx], p_out_mask_tensor[:, cvp_idx]], dim=1)
                pred_target_path = os.path.join(prediction_target_dir, f"prediction_target_{int(hadm_id)}_traj_{traj_num:03d}.pt")
                torch.save((pred_values, pred_mask, p_out_abs_time_tensor, p_out_rel_time_tensor, p_out_len), pred_target_path)
                prediction_target_info.append({
                    'hadm_id': hadm_id, 'trajectory_num': traj_num, 'length': p_out_len,
                    'has_map_data': torch.any(pred_mask[:, 0] > 0).item(),
                    'has_cvp_data': torch.any(pred_mask[:, 1] > 0).item(),
                    'file_path': pred_target_path
                })

        has_any_data_in = np.any(p_in_mask > 0)
        has_any_data_out = np.any(p_out_mask > 0)

        trajectory_info.append({
            'hadm_id': hadm_id, 'trajectory_num': traj_num,
            'p_in_start_idx': start_idx_in, 'p_in_end_idx': end_idx_in, 'p_in_length': p_in_len,
            'p_out_start_idx': end_idx_in, 'p_out_end_idx': end_idx_out, 'p_out_length': p_out_len,
            'start_time_hours': 0.0, 'end_time_hours': rel_time_array[end_idx_out - 1] if end_idx_out > 0 else 0,
            'has_physio_data_in': has_any_data_in, 'has_physio_data_out': has_any_data_out,
            'file_path_in': file_path_in
        })
    debug_print(f"  [Physio] Saved {len(trajectory_info)} trajectories for patient {hadm_id}.")

    if save_prediction_targets:
        return (hadm_id, trajectory_info, prediction_target_info)
    else:
        return (hadm_id, trajectory_info)


def create_physio_tensors(
        physio_df,
        hr_itemids,
        map_itemids,
        cvp_itemids,
        sv_itemids,
        co_itemids,
        trajectory_metadata_path,
        relevant_patients_chartevents_path,
        time_interval_minutes=1,
        icustays_path=PHYSIONET_INPUT_DIR / 'icustays.csv',
        cache_dir=DEFAULT_OUTPUT_DIR / 'p_tensors',
        prediction_target_dir=DEFAULT_OUTPUT_DIR / 'prediction_targets',
        n_workers=4,
        max_co_age_minutes=10,
        co_guess=4.0,
        debug_patient_id=20214994
):
    """
    Create physiological measurement tensors aligned with medication trajectories.
    Now also creates prediction target tensors (MAP and CVP) at the same time.
    """
    debug_print(f"Creating physiological tensors. Cache dir: {cache_dir}, Workers: {n_workers}")
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    Path(prediction_target_dir).mkdir(parents=True, exist_ok=True)

    debug_print(f"Loading medication trajectory metadata from: {trajectory_metadata_path}")
    with open(trajectory_metadata_path, 'rb') as f:
        med_trajectory_data = pickle.load(f)
    all_med_trajectories = med_trajectory_data['all_trajectories']
    trajectory_before_minutes = med_trajectory_data.get('trajectory_before_minutes')
    trajectory_after_minutes = med_trajectory_data.get('trajectory_after_minutes', 0)

    if trajectory_before_minutes is not None:
        # Fixed window trajectories - all have the same length
        total_minutes = trajectory_before_minutes + trajectory_after_minutes
        n_intervals = int(np.ceil(total_minutes / time_interval_minutes))
        debug_print(
            f"Using fixed trajectory windows: {trajectory_before_minutes} min before + {trajectory_after_minutes} min after t0")
        debug_print(f"All trajectories will have {n_intervals} intervals ({total_minutes} minutes)")
    else:
        # Variable length trajectories (from admission to t0) - need max based on actual data
        # Use the n_intervals from medication metadata which was already calculated correctly
        n_intervals = med_trajectory_data['n_intervals']
        debug_print(f"Using variable trajectory windows up to {n_intervals} intervals.")

    debug_print("Loading ICU stays data...")
    icustays_df = pl.read_csv(icustays_path)
    if icustays_df.schema['INTIME'] != pl.Datetime:
        icustays_df = icustays_df.with_columns(pl.col('INTIME').str.to_datetime())

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
    chartevents_df = pl.scan_parquet(relevant_patients_chartevents_path)

    # Ensure datetime parsing
    chart_df = chartevents_df.filter(
        pl.col('ITEMID').is_in(all_itemids) &
        pl.col('VALUE').is_not_null() &
        pl.col('HADM_ID').is_in(hadm_ids_with_trajectories)
    )

    chart_df = chart_df.collect()

    # Cast value to float, filtering out non-numeric values
    chart_df = chart_df.with_columns([
        pl.col('VALUE').cast(pl.Float64, strict=False).alias('VALUE')
    ]).filter(
        pl.col('VALUE').is_not_null()
    )

    chart_df = chart_df.join(
        icustays_df.select(['HADM_ID', 'INTIME']),
        on='HADM_ID',  # ← Fixed: both tables have HADM_ID
        how='inner'
    )

    # Parse charttime string to datetime
    chart_df = chart_df.with_columns([
        pl.col('CHARTTIME').str.to_datetime()
    ])

    # Calculate time indices
    chart_df = chart_df.with_columns([
        ((pl.col('CHARTTIME') - pl.col('INTIME')).dt.total_seconds() / 60).alias('minutes_from_admission'),
        # ← Fixed: use INTIME
        ((pl.col('CHARTTIME') - pl.col(
            'INTIME')).dt.total_seconds() / 60 / time_interval_minutes)  # ← Fixed: use INTIME
        .floor().cast(pl.Int32).alias('time_idx')
    ])

    # Filter out pre-admission measurements
    chart_df = chart_df.filter(pl.col('minutes_from_admission') >= 0)

    stay_admission_map = {}
    for row in icustays_df.select(['HADM_ID', 'INTIME']).iter_rows(named=True):
        stay_admission_map[row['HADM_ID']] = row['INTIME']

    # Process each patient
    all_physio_trajectory_info = {}
    all_prediction_targets = {}  # NEW: Track prediction targets

    # Update the process_func to include prediction target saving
    process_func = partial(
        process_single_patient_physio,
        physio_df=chart_df,
        physio_params=physio_params,
        co_itemids=co_itemids,
        n_intervals=n_intervals,
        interval_minutes=time_interval_minutes,
        cache_dir=cache_dir,
        save_prediction_targets=False,  # NEW
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
                                chartevents_path=str(relevant_patients_chartevents_path),
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

def find_relevant_inputevents(
    hadm_ids,
    save_path,
    events,
    inputevents_mv_path=MIMIC_INPUT_DIR / "INPUTEVENTS_MV.csv",
    inputevents_cv_path=MIMIC_INPUT_DIR / "INPUTEVENTS_CV.csv",
    data_limit=None
):
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
        med_df,
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
    Vectorized version using a 'delta and cumulative sum' approach for high performance.
    """
    debug_print(f"  [Meds] Processing patient {hadm_id} with vectorized logic.")
    med_df_patient = med_df.filter(pl.col('hadm_id') == hadm_id)
    if med_df_patient.height == 0:
        debug_print(f"  [Meds] No medication events found for patient {hadm_id}. Skipping.")
        return hadm_id, []

    debug_print(f"  [Meds] Found {med_df_patient.height} medication events for patient {hadm_id}.")

    # Deconstruct infusions into "delta events"
    starts = med_df_patient.select(pl.col('start_idx').alias('time_idx'), pl.col('itemid'), pl.col('rate').alias('rate_delta'))
    ends = med_df_patient.select(pl.col('end_idx').alias('time_idx'), pl.col('itemid'), (-pl.col('rate')).alias('rate_delta'))
    delta_df = pl.concat([starts, ends])
    debug_print(f"  [Meds] Created {delta_df.height} delta events (starts and ends).")

    # Aggregate deltas and pivot
    aggregated_deltas = delta_df.group_by(['time_idx', 'itemid']).agg(pl.sum('rate_delta').alias('net_delta'))
    pivoted_deltas = aggregated_deltas.pivot(index='time_idx', columns='itemid', values='net_delta').sort('time_idx')
    debug_print(f"  [Meds] Pivoted deltas. Shape: {pivoted_deltas.shape}")

    # Create a full time range and join to ensure all intervals are present
    full_time_range = pl.DataFrame({'time_idx': range(n_intervals)})
    pivoted_deltas = full_time_range.join(pivoted_deltas, on='time_idx', how='left').fill_null(0)

    # Calculate active rates with cumulative sum
    item_id_cols = [col for col in pivoted_deltas.columns if col != 'time_idx']
    active_rates_df = pivoted_deltas.select(
        [pl.col('time_idx')] + [pl.col(c).cum_sum().alias(c) for c in item_id_cols]
    )
    debug_print(f"  [Meds] Calculated active rates via cumsum. Shape: {active_rates_df.shape}")


    # Prepare final numpy arrays
    n_medications = len(medication_info) + 2  # +2 for crystalloid_sum and t0_trigger
    values_array = np.zeros((n_intervals, n_medications), dtype=np.float32)
    crystalloid_sum_idx = len(medication_info)
    t0_trigger_idx = len(medication_info) + 1

    # Populate arrays from the active_rates_df
    med_idx_map = {info['itemid']: i for i, info in enumerate(medication_info)}
    for itemid_str in active_rates_df.columns:
        if itemid_str == 'time_idx': continue
        itemid = int(itemid_str)
        if itemid in med_idx_map:
            idx = med_idx_map[itemid]
            values_array[:, idx] = active_rates_df[itemid_str].to_numpy()

    # Calculate crystalloid sum
    crystalloid_indices = [med_idx_map[m['itemid']] for m in medication_info if m['medication_type'] == 'crystalloid' and m['itemid'] in med_idx_map]
    if crystalloid_indices:
        values_array[:, crystalloid_sum_idx] = np.round(np.sum(values_array[:, crystalloid_indices], axis=1))

    # Vectorized t0 trigger calculation
    vasopressor_indices = [med_idx_map[m['itemid']] for m in medication_info if m['medication_type'] == 'vasopressor' and m['itemid'] in med_idx_map]
    if vasopressor_indices:
        vaso_rates = values_array[:, vasopressor_indices]
        vaso_start_mask = (vaso_rates[1:] > 0) & (vaso_rates[:-1] == 0)
        vasopressor_triggers = np.any(vaso_start_mask, axis=1)
    else:
        vasopressor_triggers = np.zeros(n_intervals -1, dtype=bool)

    crystalloid_sum = values_array[:, crystalloid_sum_idx]
    crystalloid_triggers = (np.abs(crystalloid_sum[1:] - crystalloid_sum[:-1]) > 20) & (crystalloid_sum[1:] > 50)

    t0_array = np.zeros(n_intervals, dtype=np.float32)
    t0_array[1:] = np.where(vasopressor_triggers, 1, np.where(crystalloid_triggers, 1, 0))
    values_array[:, t0_trigger_idx] = t0_array

    mask_array = (values_array > 1e-6).astype(np.float32)
    mask_array[:, t0_trigger_idx] = 1.0 # t0 is always "measured"

    debug_print(f"  [Meds] Vectorized calculations complete. Found {int(np.sum(t0_array))} t0 triggers.")

    # Slicing and saving trajectories
    rel_time_array = np.arange(n_intervals) * interval_minutes / 60.0
    abs_time_array = rel_time_array.copy()
    trajectories = extract_trajectories_from_patient(values_array, mask_array, abs_time_array, rel_time_array, t0_trigger_idx, trajectory_before_minutes, trajectory_after_minutes, interval_minutes)

    trajectory_info = []
    for traj_num, (start_idx, end_idx) in enumerate(trajectories):
        file_path = save_trajectory_tensor(values_array, mask_array, abs_time_array, rel_time_array, start_idx, end_idx, hadm_id, traj_num, cache_dir)
        has_t0_at_end = (end_idx > 0 and end_idx <= n_intervals and values_array[end_idx - 1, t0_trigger_idx] == 1)
        trajectory_info.append({
            'hadm_id': hadm_id, 'trajectory_num': traj_num, 'start_idx': start_idx, 'end_idx': end_idx,
            'length': end_idx - start_idx,
            'start_time_hours': rel_time_array[start_idx] if start_idx < n_intervals else 0,
            'end_time_hours': rel_time_array[end_idx - 1] if end_idx > 0 else 0,
            'has_t0_trigger': has_t0_at_end, 'file_path': file_path
        })

    debug_print(f"  [Meds] Saved {len(trajectory_info)} trajectories for patient {hadm_id}.")
    return hadm_id, trajectory_info


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


def save_patient_physio_as_csv_with_rtpr(
        hadm_id,
        chartevents_path,
        physio_params,
        co_itemids,
        n_intervals,
        interval_minutes,
        icu_admission_time,
        trajectory_boundaries,
        output_dir
):
    """
    Save all physiological data for a single patient as CSV files for inspection,
    including the calculated R_TPR value.
    """
    patient_dir = Path(output_dir) / f"patient_{hadm_id}_physio_inspection"
    patient_dir.mkdir(parents=True, exist_ok=True)

    physio_df_patient = pl.read_parquet(chartevents_path).filter(pl.col('hadm_id') == hadm_id)
    if physio_df_patient.height == 0:
        print(f"No physio data for debug patient {hadm_id}")
        return

    # This logic mirrors process_single_patient_physio
    n_params = len(physio_params)
    values_array = np.zeros((n_intervals, n_params), dtype=np.float32)
    mask_array = np.zeros((n_intervals, n_params), dtype=np.float32)
    count_array = np.zeros((n_intervals, n_params), dtype=np.int32)

    map_idx = cvp_idx = r_tpr_idx = hr_idx = sv_idx = None
    for idx, p in enumerate(physio_params):
        if p['param_type'] == 'MAP': map_idx = idx
        elif p['param_type'] == 'CVP': cvp_idx = idx
        elif p['param_type'] == 'R_TPR': r_tpr_idx = idx
        elif p['param_type'] == 'HR': hr_idx = idx
        elif p['param_type'] == 'SV': sv_idx = idx

    co_values_array = np.zeros(n_intervals, dtype=np.float32)
    co_mask_array = np.zeros(n_intervals, dtype=np.float32)
    co_count_array = np.zeros(n_intervals, dtype=np.int32)

    for idx, param_info in enumerate(physio_params):
        if param_info['param_type'] == 'R_TPR': continue
        param_events = physio_df_patient.filter(pl.col('itemid') == param_info['itemid'])
        for row in param_events.iter_rows(named=True):
            time_idx = row['time_idx']
            if 0 <= time_idx < n_intervals:
                if mask_array[time_idx, idx] == 0:
                    values_array[time_idx, idx] = row['value']
                    mask_array[time_idx, idx] = 1
                    count_array[time_idx, idx] = 1
                else:
                    current_sum = values_array[time_idx, idx] * count_array[time_idx, idx]
                    count_array[time_idx, idx] += 1
                    values_array[time_idx, idx] = (current_sum + row['value']) / count_array[time_idx, idx]

    for co_itemid in co_itemids:
        co_events = physio_df_patient.filter(pl.col('itemid') == co_itemid)
        for row in co_events.iter_rows(named=True):
            time_idx = row['time_idx']
            if 0 <= time_idx < n_intervals:
                if co_mask_array[time_idx] == 0:
                    co_values_array[time_idx] = row['value']
                    co_mask_array[time_idx] = 1
                    co_count_array[time_idx] = 1
                else:
                    current_sum = co_values_array[time_idx] * co_count_array[time_idx]
                    co_count_array[time_idx] += 1
                    co_values_array[time_idx] = (current_sum + row['value']) / co_count_array[time_idx]

    if hr_idx is not None and sv_idx is not None:
        for t in range(n_intervals):
            if mask_array[t, hr_idx] > 0 and values_array[t, hr_idx] > 0:
                hr_value = values_array[t, hr_idx]
                if co_mask_array[t] > 0 and mask_array[t, sv_idx] == 0 and co_values_array[t] > 0:
                    values_array[t, sv_idx] = co_values_array[t] / hr_value
                    mask_array[t, sv_idx] = 1.0
                elif mask_array[t, sv_idx] > 0 and co_mask_array[t] == 0 and values_array[t, sv_idx] > 0:
                    co_values_array[t] = values_array[t, sv_idx] * hr_value
                    co_mask_array[t] = 1.0

    if all([r_tpr_idx, map_idx, cvp_idx]):
        for t in range(n_intervals):
            if mask_array[t, map_idx] == 1 and mask_array[t, cvp_idx] == 1 and co_mask_array[t] == 1 and co_values_array[t] > 0:
                values_array[t, r_tpr_idx] = (values_array[t, map_idx] - values_array[t, cvp_idx]) / co_values_array[t]
                mask_array[t, r_tpr_idx] = 1.0

    col_names = [p['param_name'] for p in physio_params]
    values_df = pd.DataFrame(values_array, columns=col_names)
    time_hours = np.arange(n_intervals) * interval_minutes / 60.0
    values_df.insert(0, 'time_hours', time_hours)
    values_df.to_csv(patient_dir / 'physio_values.csv', index=False)

    mask_df = pd.DataFrame(mask_array, columns=col_names)
    mask_df.insert(0, 'time_hours', time_hours)
    mask_df.to_csv(patient_dir / 'physio_mask.csv', index=False)

    co_df = pd.DataFrame({'co_values': co_values_array, 'co_mask': co_mask_array})
    co_df.insert(0, 'time_hours', time_hours)
    co_df.to_csv(patient_dir / 'co_values.csv', index=False)

    print(f"Saved physio inspection CSVs for patient {hadm_id} to {patient_dir}")


def save_prediction_target_debug_csv(hadm_id, prediction_target_info, med_trajectories, output_dir):
    """Saves prediction target data to a CSV for easy debugging."""
    patient_dir = Path(output_dir) / f"patient_{hadm_id}_prediction_targets"
    patient_dir.mkdir(parents=True, exist_ok=True)

    all_traj_df = []
    for traj_info, med_traj in zip(prediction_target_info, med_trajectories):
        traj_num = traj_info['trajectory_num']
        p_out_len = traj_info['length']
        t0_time = med_traj['end_time_hours'] * 60 # minutes

        file_path = traj_info['file_path']
        pred_values, pred_mask, abs_time, _, _ = torch.load(file_path)

        df = pd.DataFrame({
            'trajectory_num': traj_num,
            'time_from_t0_mins': (abs_time - abs_time[0]) / 60,
            'map_value': pred_values[:, 0].numpy(),
            'map_mask': pred_mask[:, 0].numpy(),
            'cvp_value': pred_values[:, 1].numpy(),
            'cvp_mask': pred_mask[:, 1].numpy(),
            't0_time_mins_from_admission': t0_time,
        })
        all_traj_df.append(df)

    if all_traj_df:
        final_df = pd.concat(all_traj_df)
        final_df.to_csv(patient_dir / f'prediction_targets_debug_{hadm_id}.csv', index=False)
        print(f"Saved prediction target debug CSV for patient {hadm_id} to {patient_dir}")

def create_medication_tensors(
        med_df,
        crystalloid_itemids,
        vasopressor_itemids,
        relevant_patients_inputevents_path,
        time_interval_minutes=1,
        trajectory_before_minutes=None,
        trajectory_after_minutes=0,
        icustays_path=PHYSIONET_INPUT_DIR / 'icustays.csv',
        cache_dir=DEFAULT_OUTPUT_DIR / 'med_tensors',
        n_workers=4,
        debug_patient_id=20214994):
    """
    Create medication trajectory tensors for each patient with configurable trajectory windows.
    """
    debug_print(f"Creating medication tensors. Cache dir: {cache_dir}, Workers: {n_workers}")
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    interval_minutes = time_interval_minutes

    debug_print("Loading ICU stays data...")
    los_df = pl.read_csv(icustays_path)
    if 'LOS' not in los_df.columns:
        raise ValueError(f"Column 'LOS' not found in {icustays_path}. Available columns: {los_df.columns}")

    if trajectory_before_minutes is not None:
        total_window_minutes = trajectory_before_minutes + trajectory_after_minutes
        n_intervals = int(np.ceil(total_window_minutes / interval_minutes))
        debug_print(f"Using fixed trajectory windows: {total_window_minutes} mins, with {trajectory_before_minutes} minutes before t0 and {trajectory_after_minutes} minutes after t0 -> {n_intervals} intervals at {interval_minutes} minutes each")
        max_los_days = None
    else:
        debug_print("Using variable trajectory windows (ICU admission to t0).")
        max_los_days = los_df['LOS'].max()
        max_minutes = max_los_days * 24 * 60
        n_intervals = int(np.ceil(max_minutes / interval_minutes))
        debug_print(f"Max LOS: {max_los_days:.2f} days -> {n_intervals} intervals at {interval_minutes} minutes each")

    icustays_df = los_df
    if icustays_df.schema['INTIME'] != pl.Datetime:
        icustays_df = icustays_df.with_columns(pl.col('INTIME').str.to_datetime())

    all_hadm_ids = med_df['hadm_id'].unique().to_list()
    debug_print(f"Processing {len(all_hadm_ids)} unique patients from pre-loaded medication data.")

    # Create medication metadata
    medication_info = []
    all_meds = crystalloid_itemids + vasopressor_itemids

    # Create medication metadata using faster list comprehensions
    medication_info = [
        {
            'itemid': itemid,
            'medication_type': 'crystalloid',
            'medication_name': f'crystalloid_{itemid}'
        }
        for itemid in crystalloid_itemids
    ] + [
        {
            'itemid': itemid,
            'medication_type': 'vasopressor',
            'medication_name': f'vasopressor_{itemid}'
        }
        for itemid in vasopressor_itemids
    ]

    # Create hadm_id to admission time mapping using a vectorized approach
    # This is significantly faster than iterating over rows.
    stay_admission_map = dict(zip(
        icustays_df['HADM_ID'].to_list(),
        icustays_df['INTIME'].to_list()
    ))

    # Prepare for parallel processing
    process_func = partial(
        process_single_patient_medications,
        med_df=med_df,
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
                                inputevents_path=str(relevant_patients_inputevents_path),
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
                            inputevents_path=str(relevant_patients_inputevents_path),
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
    # Use pathlib for robust path handling and ensure the directory exists.
    # This is crucial in a multiprocessing context to avoid race conditions where
    # a child process might try to save a file before the OS has made the
    # directory (created by the parent process) available.
    cache_dir_path = Path(cache_dir)
    cache_dir_path.mkdir(parents=True, exist_ok=True)

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

    # Construct the final file path using the robust pathlib object
    file_path = cache_dir_path / f"med_tensor_{int(hadm_id)}_traj_{traj_num:03d}.pt"

    torch.save(
        (values_tensor, mask_tensor, abs_time_tensor, rel_time_tensor, length),
        file_path
    )

    return file_path


def prepare_medication_data(med_df, icustays_path, all_med_ids, all_hadm_ids, interval_minutes):
    """
    Prepare medication data by joining with ICU stays and calculating time indices.
    """
    debug_print("Preparing medication events...")

    # Ensure datetime columns are parsed
    if med_df.schema.get('starttime') != pl.Datetime:
        med_df = med_df.with_columns([
            pl.col('starttime').str.to_datetime(),
            pl.col('endtime').str.to_datetime()
        ])

    # Filter for relevant medications
    med_df = med_df.filter(
        pl.col('itemid').is_in(all_med_ids) &
        pl.col('rate').is_not_null() &
        pl.col('hadm_id').is_in(all_hadm_ids)
    )

    # Check for missing endtimes
    missing_endtimes = med_df.filter(pl.col('endtime').is_null()).height
    if missing_endtimes > 0:
        raise ValueError(f"Found {missing_endtimes} infusion events with missing endtimes.")

    # Load and join with ICU admission times
    icustays_df = pl.read_csv(icustays_path)
    if icustays_df.schema['INTIME'] != pl.Datetime:
        icustays_df = icustays_df.with_columns(pl.col('INTIME').str.to_datetime())

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

    debug_print("Medication data preparation complete.")
    return med_df




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
    input_dir = Path(args.input_dir)

    debug_print(f"MIMIC-III Processing Configuration: {args}")

    with open('../../data/mimic_3_data/input_data/RECORDS-numerics.txt', 'r') as f:
        numbers = [int(line.strip().split('/')[1].lstrip('p'))
                   for line in f
                   if line.strip() and len(line.strip().split('/')) >= 2
                   and line.strip().split('/')[1].lstrip('p').isdigit()]

    print(f"\nMIMIC-III Item ID Mappings:")

    mimic3_ids = get_mimic3_item_ids()
    debug_print(f"MIMIC-III Item ID Mappings: {mimic3_ids}")

    relevant_patients_chartevents_path = MIMIC_PROCESSED_DIR / "treated_patients_chartevents.parquet"
    relevant_patients_inputevents_path = MIMIC_PROCESSED_DIR / "treated_patients_inputevents.parquet"

    nested_list = [mimic3_ids['hr'], mimic3_ids['map'], mimic3_ids['cvp'], mimic3_ids['co'], mimic3_ids['sv']]
    measurements = [item for sublist in nested_list for item in sublist]

    hadm_ids_df = find_relevant_patients(
        waveform_patient_ids=numbers,
        measurements=measurements,
        MAP_id=[225312, 52, 6702],
        save_path=relevant_patients_chartevents_path,
        data_limit=args.data_limit
    )
    unique_hadm_ids = hadm_ids_df['HADM_ID'].unique().to_list()
    print(f"UNIQUE HADM_IDS: {len(unique_hadm_ids)}")

    # Apply patient limit for testing if specified
    if args.patient_limit is not None and args.patient_limit > 0:
        debug_print(f"Limiting processing to {args.patient_limit} patients for testing.")
        unique_hadm_ids = unique_hadm_ids[:args.patient_limit]

    relevant_patients_df = pd.DataFrame({'HADM_ID': unique_hadm_ids})
    admissions_df = pd.read_csv(input_dir / 'ADMISSIONS.csv')
    relevant_patients_df = relevant_patients_df.merge(admissions_df[['HADM_ID', 'SUBJECT_ID']], on='HADM_ID', how='left')
    relevant_patients_df.to_csv(output_dir / 'relevant_patient_ids.csv', index=False)
    debug_print(f"Saved {len(relevant_patients_df)} relevant patient IDs.")

    debug_print("\n=== Loading and Preparing Data into Memory ===")
    physio_df = pl.read_parquet(relevant_patients_chartevents_path)
    med_df = find_relevant_inputevents(
        hadm_ids=unique_hadm_ids,
        save_path=relevant_patients_inputevents_path,
        events=mimic3_ids['crystalloids'] + mimic3_ids['vasopressors'],
        inputevents_mv_path=input_dir / "INPUTEVENTS_MV.csv",
        inputevents_cv_path=input_dir / "INPUTEVENTS_CV.csv",
        data_limit=args.data_limit
    )
    med_df_prepared = prepare_medication_data(
        med_df=med_df,
        icustays_path=input_dir / "ICUSTAYS.csv",
        all_med_ids=mimic3_ids['crystalloids'] + mimic3_ids['vasopressors'],
        all_hadm_ids=unique_hadm_ids,
        interval_minutes=args.interval_minutes
    )
    debug_print(f"Medication data prepared with shape: {med_df_prepared.shape}")

    debug_print("\n=== Creating Medication Tensors ===")
    medication_trajectories = create_medication_tensors(
        med_df=med_df_prepared,
        crystalloid_itemids=mimic3_ids['crystalloids'],
        vasopressor_itemids=mimic3_ids['vasopressors'],
        relevant_patients_inputevents_path=relevant_patients_inputevents_path,
        time_interval_minutes=args.interval_minutes,
        trajectory_before_minutes=args.trajectory_before_minutes,
        trajectory_after_minutes=args.trajectory_after_minutes,
        icustays_path=input_dir / "ICUSTAYS.csv",
        cache_dir=output_dir / "med_tensors",
        n_workers=args.n_workers,
        debug_patient_id=args.debug_patient_id
    )
    med_trajectory_metadata_path = output_dir / "med_tensors" / "trajectory_metadata.pkl"

    debug_print("\n=== Creating Physio Tensors ===")
    create_physio_tensors(
        physio_df=physio_df, # physio_df is already prepared
        hr_itemids=mimic3_ids['hr'],
        map_itemids=mimic3_ids['map'],
        cvp_itemids=mimic3_ids['cvp'],
        co_itemids=mimic3_ids['co'],
        sv_itemids=mimic3_ids['sv'],
        trajectory_metadata_path=med_trajectory_metadata_path,
        relevant_patients_chartevents_path=relevant_patients_chartevents_path,
        time_interval_minutes=args.interval_minutes,
        icustays_path=input_dir / "ICUSTAYS.csv",
        cache_dir=output_dir / "p_tensors",
        n_workers=args.n_workers,
        max_co_age_minutes=args.max_co_age_minutes,
        co_guess=args.co_guess,
        debug_patient_id=args.debug_patient_id
    )

    debug_print("\n=== Creating Baseline Tensors ===")
    if med_trajectory_metadata_path.exists():
        create_baseline_tensors(
            input_dir=str(input_dir),
            output_dir=str(output_dir),
            trajectory_metadata_path=med_trajectory_metadata_path
        )
    else:
        print(f"Warning: Could not find trajectory metadata at {med_trajectory_metadata_path}")
        print("Skipping baseline tensor creation.")

    print(f"\n=== Processing Complete ===")
   # print(f"Total patients with trajectories: {len(medication_trajectories)}")
    print(f"Output saved to: {output_dir}")


if __name__ == "__main__":
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Data directory: {DATA_DIR}")
    print(f"MIMIC directory: {MIMIC_DIR}")
    print(f"MIMIC input directory: {MIMIC_INPUT_DIR}")
    print(f"MIMIC processed directory: {MIMIC_PROCESSED_DIR}")
    print(f"Physionet input directory: {PHYSIONET_INPUT_DIR}")
    print(f"Default output directory: {DEFAULT_OUTPUT_DIR}")
    main()