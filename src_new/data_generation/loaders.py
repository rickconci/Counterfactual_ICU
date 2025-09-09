import os
from pathlib import Path
import pandas as pd
from typing import Dict, Any, Set, Optional, List
import numpy as np

def load_initial_data(
    full_numerics_dir: str,
    records_numerics_path: str,
    icu_stays_path: str,
    items_path: str,
    all_trigger_meds_path: str
) -> Dict[str, Any]:
    """Load and preprocess initial data files."""
    data = {}

    # Load numerics files
    data['numerics_files'] = os.listdir(full_numerics_dir)
    print(f"Found {len(data['numerics_files'])} numerics files")
    
    # Load and process records
    records = pd.read_csv(
        records_numerics_path,
        sep="\t", 
        header=None
    )
    data['records'] = records
    
    # Split the path into columns
    split_cols = records[0].str.split("/", expand=True)
    split_cols.columns = ["dir1", "dir2", "filename"]
    
    # Clean dir2: strip leading "p", then cast to int to drop leading zeros
    split_cols["dir2_clean"] = split_cols["dir2"].str.lstrip("p").astype(int)
    data['split_cols'] = split_cols
    
    print("Split columns preview:")
    print(split_cols.head())
    
    # Load ICU stays data
    icu_stays = pd.read_csv(icu_stays_path)
    data['icu_stays'] = icu_stays
    print("\nICU stays preview:")
    print(icu_stays.head())
    
    # Process ICU stays for ICD9 codes
    icu_stays_split = icu_stays[['ICUSTAY_ID', 'ICUSTAY_ICD9_CODES']].copy()
    data['icu_stays_split'] = icu_stays_split
    print("\nICU stays split preview:")
    print(icu_stays_split.head())
    
    # Split ICD9 codes
    icu_stays_split['ICUSTAY_ICD9_CODES'] = (
        icu_stays_split['ICUSTAY_ICD9_CODES'].str.split(';')
    )
    print("\nICU stays with split ICD9 codes:")
    print(icu_stays_split.head())
    
    # Load items data
    items_df = pd.read_csv(items_path)
    data['items_df'] = items_df
    print("\nItems dataframe preview:")
    print(items_df.head())
    
    # Find subjects with waveforms and filter ICU stays
    subject_ids_with_waveforms = (
        set(icu_stays['SUBJECT_ID'].unique()) & 
        set(split_cols['dir2_clean'])
    )
    data['subject_ids_with_waveforms'] = subject_ids_with_waveforms
    
    icu_stays_w_waveforms = icu_stays[
        icu_stays['SUBJECT_ID'].isin(subject_ids_with_waveforms)
    ]
    data['icu_stays_w_waveforms'] = icu_stays_w_waveforms
    
    hadm_ids = icu_stays_w_waveforms['HADM_ID'].unique()
    data['hadm_ids'] = hadm_ids

    all_trigger_meds = pd.read_csv(all_trigger_meds_path)
    data['all_trigger_meds'] = all_trigger_meds
    
    print(f"\nFound {len(subject_ids_with_waveforms)} subjects with waveforms")
    print(f"Found {len(hadm_ids)} unique hospital admissions")
    
    return data
