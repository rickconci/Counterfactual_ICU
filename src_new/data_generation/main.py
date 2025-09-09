import os
import dotenv
from pathlib import Path
from data_generation.core import DataGenerator

def main():
    """Main function to run the data generation pipeline."""
    dotenv.load_dotenv()

    # Configuration dictionary
    config = {
        'full_numerics_dir': os.getenv('FULL_NUMERICS_DIR'),
        'records_numerics_path': os.getenv('RECORDS_NUMERICS_PATH'),
        'icu_stays_path': os.getenv('ICU_STAYS_PATH'),
        'items_path': os.getenv('ITEMS_PATH'),
        'input_data_dir': Path(os.getenv('INPUT_DATA_DIR')),
        'all_trigger_meds_path': os.getenv('ALL_TRIGGER_MEDS_PATH'),
        'waveform_dir': os.getenv('WAVEFORM_DIR'),
        'core_signals_to_keep': ['ABP MEAN', 'ABP Mean', 'ABPMean', 'ART MEAN', 'ART Mean', 'CVP'],
        'uptitration_rel_threshold': 0.25,
        'min_abs_change': 0.02,
        'cluster_window_minutes': 20,
        'min_triggers_in_cluster': 1,
        'waveform_gap_hours': 2,
        'alignment_window_minutes': 10,
        'aligned_waveform_files_csv_path': os.getenv('ALIGNED_WAVEFORM_FILES_CSV_PATH'),
        'harmonized_waveforms_parquet_path': os.getenv('HARMONIZED_WAVEFORMS_PARQUET_PATH'),
    }

    # Initialize and run the data generator
    data_generator = DataGenerator(config)
    processed_data = data_generator.run_pipeline()

    # You can now access all the processed dataframes from the `processed_data` dictionary
    print("Data generation pipeline completed successfully.")
    print("Available data keys:", processed_data.keys())

    # Example of accessing a processed dataframe
    if 'best_segments' in processed_data:
        print("\nPreview of best segments:")
        print(processed_data['best_segments'].head())

    # Sanity check on aligned data
    if 'aligned_consolidated' in processed_data:
        print("\nSanity check on aligned consolidated data:")
        aligned_cons = processed_data['aligned_consolidated']
        print(aligned_cons.groupby(["hadm_id", "action_cluster_id"]).size().describe())

        # Merge with trigger data to get trig_with_waveform
        trig_clustered = processed_data['trig_clustered']
        clusters_with_waveform = aligned_cons[["hadm_id", "action_cluster_id"]].drop_duplicates()
        trig_with_waveform = trig_clustered.merge(
            clusters_with_waveform,
            on=["hadm_id", "action_cluster_id"],
            how="inner"
        )
        
        # Save the resulting dataframe
        out_path = "/n/netscratch/mzitnik_lab/Lab/rconci/BIOMM/numerics/trig_with_waveform.parquet"
        print(f"\nSaving trig_with_waveform to {out_path}...")
        trig_with_waveform.to_parquet(out_path, index=False, engine="pyarrow")
        print("Save complete.")
    
    if 'harmonized_waveforms' in processed_data:
        print("\nPreview of harmonized waveforms:")
        print(processed_data['harmonized_waveforms'].head())

if __name__ == "__main__":
    main()
