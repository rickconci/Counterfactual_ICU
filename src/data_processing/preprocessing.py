import pandas as pd
import matplotlib.pyplot as plt
import os
import polars as pl
import gc
from tqdm import tqdm
from typing import Dict, Any, List, Tuple
from scipy.spatial.distance import cdist
import numpy as np



def find_relevant_patients(measurements, MAP_id = 220052, load_path_events = "chartevents.csv", load_path_stays = "icustays.csv",  save_path = "treated_patients_all_values.parquet"):
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
        long_stays = (pl.scan_csv(load_path_stays).limit(100000)
                      .filter(pl.col("los") > 1)
                      .collect())
        long_stays_id = long_stays["stay_id"].unique().to_list()
        # Find patients with a blood pressure event
        treated_patients = (pl.scan_csv(load_path_events)
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
    ids_df.write_parquet("temp.parquet")
    # Polars handles large files much better
    result = (
        pl.scan_csv(load_path).limit(100000)
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
    all_patients_hr = (pl.scan_parquet(all_patients_path)
                       .filter(pl.col(item_column) == hr_ID)
                       .collect())
    patients = all_patients_hr[patient_id].unique().to_list()
    metadata = pl.scan_csv(metadata_path).collect()
    hr_params = pd.DataFrame({"stay_id":[], "min_hr":[], "max_hr":[]})
    for patient in patients:
        patient_data = all_patients_hr.filter(pl.col(patient_id) == patient)
        patient_min = patient_data[value_column].min()
        patient_metadata = patient_metadata.filter(pl.col(patient_id == patient))
        patient_age = patient_metadata["anchor_age"].item()
        patient_max = 220 - patient_age
        hr_params_patients = pd.DataFrame({"stay_id":[patient], "min_hr":[patient_min], "max_hr":[patient_max]})
        hr_params = pd.concat((hr_params, hr_params_patients))
    hr_params.to_csv(save_path)
    return hr_params

def main():
    # Heart Rate: 220045
    # MAP: 220052
    # CVP: 220074
    # CO-related params: 220088, 224842, 228369, 229897
    patients = find_relevant_patients(measurements=[220045, 220052, 220074, 220088, 224842, 228369, 229897], MAP_id=220052)
    hr_params = find_min_max_heartrates("treated_patients_all_values.parquet", metadata_path="patients.csv", save_path="hr_params.csv")


if __name__ == "__main__":
    main()






