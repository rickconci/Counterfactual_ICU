# Data Generation Pipeline for Counterfactual ICU Analysis

This document provides instructions on how to run the data generation pipeline for the Counterfactual ICU Analysis project. The pipeline processes raw MIMIC-III data, including clinical events, numerical data, and waveforms, to generate a cleaned and aligned dataset ready for downstream analysis.

## Overview

The pipeline performs the following key steps:

1.  **Load Initial Data**: Loads patient records, ICU stays, item definitions, and trigger medications.
2.  **Process Clinical Events**: Processes `CHARTEVENTS` by filtering, normalizing, and attaching item labels.
3.  **Identify Triggers and Clusters**: Identifies medication uptitration events (triggers) and groups them into action clusters.
4.  **Process Waveforms**: Scans for relevant physiological waveform data and consolidates discontinuous segments.
5.  **Align Data**: Aligns the action clusters with the corresponding waveform segments.
6.  **Harmonize Waveforms**: Loads and saves the raw waveform data for the aligned segments.
7.  **Clean and Smooth Waveforms**: Downsamples, despikes, and smoothes the harmonized waveforms.

## Prerequisites

Before running the pipeline, ensure you have the following:

- Python 3.10+
- Required Python packages installed (e.g., `pandas`, `pyarrow`, `dotenv`). You can create a `requirements.txt` and install them via `pip install -r requirements.txt`.
- Access to the MIMIC-III dataset, with the necessary files downloaded.

## Configuration

The pipeline is configured using a `.env` file. Create a file named `.env` in the root directory of the project and add the following environment variables, pointing to the correct paths on your system.

```env
# Input Data Paths
FULL_NUMERICS_DIR="/path/to/your/numerics_data"
RECORDS_NUMERICS_PATH="/path/to/your/RECORDS-numerics.csv"
ICU_STAYS_PATH="/path/to/your/ICUSTAYS.csv"
ITEMS_PATH="/path/to/your/D_ITEMS.csv"
INPUT_DATA_DIR="/path/to/your/input_data" # Should contain INPUTEVENTS_MV.csv
ALL_TRIGGER_MEDS_PATH="/path/to/your/all_trigger_meds.csv"
WAVEFORM_DIR="/path/to/your/waveform_data"
CHARTEVENTS_PATH="/path/to/your/CHARTEVENTS.csv"

# Output Data Paths
ALIGNED_WAVEFORM_FILES_CSV_PATH="/path/to/your/output/aligned_waveform_files.csv"
HARMONIZED_WAVEFORMS_PARQUET_PATH="/path/to/your/output/harmonized_waveforms.parquet"
CHARTEVENTS_NORMALIZED_PARQUET_PATH="/path/to/your/output/chartevents_normalized.parquet"
CLEANED_WAVEFORMS_PARQUET_PATH="/path/to/your/output/cleaned_waveforms.parquet"
```

### Configuration Details

-   `FULL_NUMERICS_DIR`: Directory containing the full numerical data from MIMIC-III.
-   `RECORDS_NUMERICS_PATH`: Path to the `RECORDS-numerics.csv` file, which contains metadata about the waveform records.
-   `ICU_STAYS_PATH`: Path to the `ICUSTAYS.csv` file.
-   `ITEMS_PATH`: Path to the `D_ITEMS.csv` file, which contains definitions for item IDs.
-   `INPUT_DATA_DIR`: Directory containing `INPUTEVENTS_MV.csv`.
-   `ALL_TRIGGER_MEDS_PATH`: Path to a custom CSV file that lists the medications to be considered as triggers.
-   `WAVEFORM_DIR`: The root directory where the raw MIMIC-III waveform data is stored.
-   `CHARTEVENTS_PATH`: Path to the `CHARTEVENTS.csv` file.
-   `ALIGNED_WAVEFORM_FILES_CSV_PATH`: Output path for a CSV file listing the waveform files that have been aligned with action clusters.
-   `HARMONIZED_WAVEFORMS_PARQUET_PATH`: Output path for the harmonized waveform data in Parquet format.
-   `CHARTEVENTS_NORMALIZED_PARQUET_PATH`: Output path for the processed and normalized `CHARTEVENTS` data.
-   `CLEANED_WAVEFORMS_PARQUET_PATH`: Output path for the final cleaned, downsampled, and smoothed waveform data.

## How to Run

To run the data generation pipeline, navigate to the `src_new` directory and execute the `main.py` script:

```bash
cd /path/to/Counterfactual_ICU/src_new
python -m data_generation.main
```

The script will print progress messages to the console as it executes each step of the pipeline.

## Output

The pipeline generates several intermediate and final data files, as specified in the `.env` configuration. The main outputs are:

-   **`trig_with_waveform.parquet`**: A Parquet file saved in the location specified in `main.py` containing the trigger events that have associated waveform data.
-   **Harmonized Waveforms**: A Parquet file containing the raw waveform data corresponding to the aligned segments.
-   **Cleaned and Smoothed Waveforms**: The final processed waveform data, ready for analysis.
-   **Normalized Chartevents**: Processed clinical event data.

The script will also print previews of the generated dataframes to the console upon completion.
