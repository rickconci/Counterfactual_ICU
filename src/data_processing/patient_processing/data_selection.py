import os
import pandas as pd
import numpy as np
import sys
import json
import pickle
from typing import List, Dict, Any, Set, Optional
import yaml

from LLM_utils import run_LLM
from src.data_processing.lab_name_aligner import LabNameAligner

class DataSelector:
    """
    A class to select, filter, and process MIMIC-IV data to create a curated dataset.

    This class encapsulates the entire data processing pipeline, from loading raw
    MIMIC-IV CSV files to generating a final, cleaned, and normalized dataframe.
    The pipeline is configurable through a dictionary passed during instantiation.

    Attributes:
        config (Dict[str, Any]): Configuration dictionary.
        base_path (str): Path to the MIMIC-IV data directory.
        temp_path (str): Path to store intermediate dataframes.
        d_chartitems_df (pd.DataFrame): Dataframe for chart item definitions.
        d_labitems_df (pd.DataFrame): Dataframe for lab item definitions.
        merged_df (Optional[pd.DataFrame]): Dataframe of merged admission data.
        hadm_ids (Optional[Set[int]]): Set of hospital admission IDs to be processed.
        chartevents_df (Optional[pd.DataFrame]): Processed chartevents data.
        outputevents_df (Optional[pd.DataFrame]): Processed outputevents data.
        labevents_df (Optional[pd.DataFrame]): Processed labevents data.
        micro_events_df (Optional[pd.DataFrame]): Processed microbiology events data.
        combined_physio_df (Optional[pd.DataFrame]): Combined physiological data.
        final_df (Optional[pd.DataFrame]): The final, processed dataframe.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the DataSelector with a configuration dictionary.

        Args:
            config (Dict[str, Any]): A dictionary containing configuration parameters.
                Expected keys:
                - base_path (str): Path to the MIMIC-IV data directory.
                - temp_path (str): Path to store intermediate dataframes.
                - min_admit_duration (int): Minimum admission duration in days.
                - max_admit_duration (int): Maximum admission duration in days.
                - top_n_physio_features (int): Number of top physiological features to keep.
                - physio_categories_to_keep (List[str]): List of chartevents categories to keep.
                - chunk_size (int): Chunk size for reading large CSVs.
                - use_cache (bool): Whether to use cached intermediate dataframes.
        """
        self.config = config
        self.base_path = self.config['base_path']
        self.temp_path = self.config['temp_path']
        os.makedirs(self.temp_path, exist_ok=True)

        self._define_paths()

        self.d_chartitems_df = pd.read_csv(self.d_chartitems_path)
        self.d_labitems_df = pd.read_csv(self.d_labitems_path)
        
        self.hadm_ids: Optional[Set[int]] = None
        self.chartevents_df: Optional[pd.DataFrame] = None
        self.outputevents_df: Optional[pd.DataFrame] = None
        self.labevents_df: Optional[pd.DataFrame] = None
        self.micro_events_df: Optional[pd.DataFrame] = None
        self.inputevents_df: Optional[pd.DataFrame] = None
        self.emar_df: Optional[pd.DataFrame] = None
        self.combined_tx_df: Optional[pd.DataFrame] = None
        self.combined_physio_df: Optional[pd.DataFrame] = None
        self.final_df: Optional[pd.DataFrame] = None

    def _define_paths(self):
        """Defines paths to all required CSV files."""
        hosp_path = os.path.join(self.base_path, 'hosp')
        icu_path = os.path.join(self.base_path, 'icu')
        note_path = os.path.join(self.base_path, 'note')

        self.patients_path = os.path.join(hosp_path, 'patients.csv')
        self.admission_path = os.path.join(hosp_path, 'admissions.csv')
        self.transfers_path = os.path.join(hosp_path, 'transfers.csv')
        self.labevents_path = os.path.join(hosp_path, 'labevents.csv')
        self.micro_events_path = os.path.join(hosp_path, 'microbiologyevents.csv')
        self.emar_path = os.path.join(hosp_path, 'emar.csv')
        self.d_labitems_path = os.path.join(hosp_path, 'd_labitems.csv')

        self.icustays_path = os.path.join(icu_path, 'icustays.csv')
        self.chartevents_path = os.path.join(icu_path, 'chartevents.csv')
        self.outputevents_path = os.path.join(icu_path, 'outputevents.csv')
        self.inputevents_path = os.path.join(icu_path, 'inputevents.csv')
        self.d_chartitems_path = os.path.join(icu_path, 'd_items.csv')
        
        self.discharge_path = os.path.join(note_path, 'discharge.csv')

    def run(self) -> pd.DataFrame:
        """
        Executes the full data selection and processing pipeline.

        Returns:
            pd.DataFrame: The final processed dataframe.
        """
        initial_hadm_ids = self._get_initial_admissions()
        self.hadm_ids = self._filter_patients_by_icu_and_map(initial_hadm_ids)
        
        self._load_and_process_events()
        self._process_and_combine_treatments()
        self._align_lab_names()
        self._combine_and_finalize_physio()
        self._normalize_physio_data()
        
        print("Data processing pipeline finished.")
        if self.final_df is not None:
            output_path = os.path.join(self.temp_path, 'final_processed_data.pkl')
            self.final_df.to_pickle(output_path)
            print(f"Final dataframe saved to {output_path}")

        return self.final_df

    def _get_initial_admissions(self) -> Set[int]:
        """Loads admission data and applies initial filters based on admission duration and discharge notes."""
        print("Loading and applying initial admission filters...")
        patients_df = pd.read_csv(self.patients_path)
        admission_df = pd.read_csv(self.admission_path)
        transfers_df = pd.read_csv(self.transfers_path)
        discharge_df = pd.read_csv(self.discharge_path)[['hadm_id']].drop_duplicates()

        merged_df = pd.merge(admission_df, patients_df, on='subject_id', how='left')
        merged_df = pd.merge(merged_df, transfers_df, on=['hadm_id', 'subject_id'], how='left')
        
        merged_df['admittime'] = pd.to_datetime(merged_df['admittime'])
        merged_df['dischtime'] = pd.to_datetime(merged_df['dischtime'])
        merged_df['admit_duration'] = (merged_df['dischtime'] - merged_df['admittime']).dt.total_seconds() / (3600 * 24)

        merged_df = merged_df[merged_df['hadm_id'].isin(discharge_df['hadm_id'])]

        merged_df = merged_df[
            (merged_df['admit_duration'] >= self.config['min_admit_duration']) &
            (merged_df['admit_duration'] <= self.config['max_admit_duration'])
        ]
        
        hadm_ids = set(merged_df['hadm_id'])
        print(f"Found {len(hadm_ids)} admissions after initial filtering.")
        return hadm_ids

    def _filter_patients_by_icu_and_map(self, hadm_ids: Set[int]) -> Set[int]:
        """Filters patients based on ICU length of stay and hypotension events."""
        print("Filtering patients by ICU stay and MAP < 70...")
        
        # Filter for ICU stays > min_icu_los_days
        icustays_df = pd.read_csv(self.icustays_path)
        long_stays_df = icustays_df[icustays_df['los'] > self.config['min_icu_los_days']]
        long_stay_hadm_ids = set(long_stays_df['hadm_id'])
        
        # Find patients who had a MAP < 70 event
        map_itemid = self.config['MAP_itemid']
        hypotensive_hadm_ids = set()
        
        for chunk in pd.read_csv(self.chartevents_path, chunksize=self.config.get('chunk_size', 1_000_000), usecols=['hadm_id', 'itemid', 'valuenum']):
            hypotensive_chunk = chunk[
                (chunk['itemid'] == map_itemid) &
                (chunk['valuenum'] < 70) &
                (chunk['hadm_id'].isin(hadm_ids))
            ]
            hypotensive_hadm_ids.update(hypotensive_chunk['hadm_id'])

        final_hadm_ids = hadm_ids.intersection(long_stay_hadm_ids).intersection(hypotensive_hadm_ids)
        
        print(f"Found {len(final_hadm_ids)} patients after ICU stay and MAP filtering.")
        return final_hadm_ids

    def _load_and_process_chunked(self, file_path: str, usecols: List[str], source: str) -> pd.DataFrame:
        """
        Loads a CSV file in chunks, filtering for rows with hadm_id in self.hadm_ids,
        and applies source-specific processing. Caches results as pickles.
        """
        pickle_file = os.path.join(self.temp_path, f"{os.path.basename(file_path).replace('.csv', '')}_{source}.pkl")

        if self.config.get('use_cache', True) and os.path.exists(pickle_file):
            print(f"Loading cached file: {pickle_file}")
            return pd.read_pickle(pickle_file)

        chunks = []
        try:
            for i, chunk in enumerate(pd.read_csv(file_path, chunksize=self.config.get('chunk_size', 1_000_000), usecols=usecols, on_bad_lines='warn')):
                if 'hadm_id' not in chunk.columns or chunk['hadm_id'].isnull().all():
                    continue

                filtered_chunk = chunk.dropna(subset=['hadm_id'])
                filtered_chunk = filtered_chunk[filtered_chunk['hadm_id'].isin(self.hadm_ids)]
                if filtered_chunk.empty:
                    continue
                
                # Apply source-specific renaming
                rename_map = {
                    'chartevents': {'valuenum': 'value'},
                    'inputevents': {'amount': 'value', 'starttime': 'charttime'},
                    'labevents': {'valuenum': 'value'},
                    'emar': {'medication': 'name', 'event_txt': 'value', 'emar_id': 'itemid'},
                    'micro_events': {'micro_specimen_id': 'itemid', 'test_name': 'name', 'comments': 'value'}
                }
                if source in rename_map:
                    filtered_chunk = filtered_chunk.rename(columns=rename_map[source])
                
                if 'value' in filtered_chunk.columns:
                    filtered_chunk = filtered_chunk.dropna(subset=['value'])
                
                if 'charttime' in filtered_chunk.columns:
                    filtered_chunk['charttime'] = pd.to_datetime(filtered_chunk['charttime'], errors='coerce')
                
                chunks.append(filtered_chunk)
                print(f"Chunk {i} from {source} processed, shape after filter: {filtered_chunk.shape}")
        except FileNotFoundError:
            print(f"Warning: File not found at {file_path}. Skipping.")
            return pd.DataFrame()

        result = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
        
        print(f"Saving result for {source} to pickle file: {pickle_file}")
        result.to_pickle(pickle_file)
        
        return result

    def _load_and_process_events(self):
        """Loads and processes all event data."""
        print("Loading and processing event data...")
        physio_cols_to_keep = {
            'chartevents': ['hadm_id', 'itemid', 'charttime', 'valuenum'],
            'outputevents': ['hadm_id', 'itemid', 'charttime', 'value'],
            'labevents': ['hadm_id', 'itemid', 'charttime', 'valuenum'],
            'micro_events': ['hadm_id', 'micro_specimen_id', 'charttime', 'test_name', 'comments'],
        }
        treatment_cols_to_keep = {
            'inputevents': ['hadm_id', 'itemid', 'starttime', 'endtime', 'amount', 'amountuom', 'patientweight'],
            'emar': ['hadm_id', 'emar_id', 'charttime', 'medication', 'event_txt']
        }
        
        self.chartevents_df = self._load_and_process_chunked(self.chartevents_path, physio_cols_to_keep['chartevents'], 'chartevents')
        self.outputevents_df = self._load_and_process_chunked(self.outputevents_path, physio_cols_to_keep['outputevents'], 'outputevents')
        self.labevents_df = self._load_and_process_chunked(self.labevents_path, physio_cols_to_keep['labevents'], 'labevents')
        self.micro_events_df = self._load_and_process_chunked(self.micro_events_path, physio_cols_to_keep['micro_events'], 'micro_events')
        
        self.inputevents_df = self._load_and_process_chunked(self.inputevents_path, treatment_cols_to_keep['inputevents'], 'inputevents')
        self.emar_df = self._load_and_process_chunked(self.emar_path, treatment_cols_to_keep['emar'], 'emar')

        self._process_physio_data()

    def _process_physio_data(self):
        """Merges physiological data with dictionaries to add names and applies filters."""
        print("Adding names and categories to physiological data...")
        
        if self.chartevents_df is not None and not self.chartevents_df.empty:
            self.chartevents_df = pd.merge(self.chartevents_df, self.d_chartitems_df[['itemid', 'label', 'category']], on='itemid', how='left')
            self.chartevents_df.rename(columns={'label': 'name'}, inplace=True)
            self.chartevents_df = self.chartevents_df[self.chartevents_df['category'].isin(self.config['physio_categories_to_keep'])]

        if self.labevents_df is not None and not self.labevents_df.empty:
            self.labevents_df = pd.merge(self.labevents_df, self.d_labitems_df[['itemid', 'label', 'category']], on='itemid', how='left')
            self.labevents_df.rename(columns={'label': 'name'}, inplace=True)

        if self.outputevents_df is not None and not self.outputevents_df.empty:
            self.outputevents_df = pd.merge(self.outputevents_df, self.d_chartitems_df[['itemid', 'label', 'category']], on='itemid', how='left')
            self.outputevents_df.rename(columns={'label': 'name'}, inplace=True)
        
        if self.micro_events_df is not None and not self.micro_events_df.empty:
            self.micro_events_df['value_text'] = self.micro_events_df['value']
            self.micro_events_df['value'] = (~self.micro_events_df['value_text'].str.contains('no', case=False, na=False)).astype(int)

    def _process_and_combine_treatments(self):
        """Processes and combines treatment data from inputevents and emar."""
        print("Processing and combining treatment data...")
        
        if self.emar_df is None or self.inputevents_df is None:
            print("Skipping treatment data processing: emar or inputevents data is missing.")
            return

        # Filter emar for 'Administered' events
        emar_filtered = self.emar_df[self.emar_df['value'] == 'Administered'].copy()

        # Add item names to inputevents from d_chartitems
        inputevents_named = pd.merge(
            self.inputevents_df,
            self.d_chartitems_df[['itemid', 'label', 'category']],
            on='itemid',
            how='left'
        )
        inputevents_named.rename(columns={'label': 'name'}, inplace=True)
        
        # Combine the two dataframes
        self.combined_tx_df = pd.concat([emar_filtered, inputevents_named], ignore_index=True)
        
        # Save the combined dataframe
        output_path = os.path.join(self.temp_path, 'combined_treatment_data.pkl')
        self.combined_tx_df.to_pickle(output_path)
        print(f"Combined treatment dataframe saved to {output_path} with {len(self.combined_tx_df)} records.")

    def _align_lab_names(self):
        """
        Aligns lab names from chartevents with labevents using the LabNameAligner utility.
        """
        aligner = LabNameAligner(
            chartevents_df=self.chartevents_df,
            labevents_df=self.labevents_df,
            temp_path=self.temp_path,
            use_cache=self.config.get('use_cache', True)
        )
        self.chartevents_df = aligner.align()

    def _combine_and_finalize_physio(self):
        """Combines all physiological dataframes and filters for top N features."""
        print("Combining and finalizing physiological data...")
        dfs_to_concat = [
            df for df in [self.chartevents_df, self.labevents_df, self.outputevents_df, self.micro_events_df] 
            if df is not None and not df.empty
        ]
        
        if not dfs_to_concat:
            print("No physiological data to combine.")
            self.combined_physio_df = pd.DataFrame()
            return

        self.combined_physio_df = pd.concat(dfs_to_concat, ignore_index=True)
        
        top_names = self.combined_physio_df['name'].value_counts().nlargest(self.config['top_n_physio_features']).index
        self.combined_physio_df = self.combined_physio_df[self.combined_physio_df['name'].isin(top_names)]
        
        self.combined_physio_df['charttime'] = pd.to_datetime(self.combined_physio_df['charttime'])
        self.combined_physio_df = self.combined_physio_df.sort_values(by=['hadm_id', 'charttime']).reset_index(drop=True)
        print(f"Combined physiological data contains {len(self.combined_physio_df)} records after filtering for top {self.config['top_n_physio_features']} features.")

    def _normalize_physio_data(self):
        """Normalizes the physiological data values."""
        if self.combined_physio_df is None or self.combined_physio_df.empty:
            print("Skipping normalization, no data available.")
            self.final_df = pd.DataFrame()
            return
        
        print("Normalizing physiological data...")
        df_numeric = self.combined_physio_df.copy()
        df_numeric['value'] = pd.to_numeric(df_numeric['value'], errors='coerce')
        df_numeric.dropna(subset=['value'], inplace=True)

        means = df_numeric.groupby('name')['value'].transform('mean')
        stds = df_numeric.groupby('name')['value'].transform('std')
        
        stds[stds == 0] = 1 # Avoid division by zero

        normalized = (df_numeric['value'] - means) / stds
        
        df_numeric['value'] = normalized.clip(lower=-3, upper=3)
        
        self.final_df = df_numeric
        print("Normalization complete.")


if __name__ == '__main__':
    # Load configuration from YAML file
    with open('src/data_processing/config.yaml', 'r') as f:
        config = yaml.safe_load(f)

    # Instantiate and run the pipeline
    data_selector = DataSelector(config)
    final_dataframe = data_selector.run()

    # Display information about the final dataframe
    if final_dataframe is not None and not final_dataframe.empty:
        print("\n--- Final Dataframe Info ---")
        print(final_dataframe.info())
        print("\n--- Final Dataframe Head ---")
        print(final_dataframe.head())
        print(f"\nNumber of unique patients (hadm_id): {final_dataframe['hadm_id'].nunique()}")
        print(f"Number of unique features (name): {final_dataframe['name'].nunique()}")
        print("\nValue counts for top 10 features:")
        print(final_dataframe['name'].value_counts().head(10))
    else:
        print("Pipeline did not produce a final dataframe.")

