import os
from pathlib import Path
import pandas as pd
from typing import Optional, List, Any, Dict, Set

from . import loaders
from . import inputs
from . import triggers
from . import clustering
from . import waveforms
from . import alignment
from . import harmonization
from . import cleaning
from . import clinical_events

class DataGenerator:
    """Data generator for ICU counterfactual analysis.
    
    This class handles loading and preprocessing of ICU data including
    numerics, waveforms, and patient records for counterfactual analysis.
    """
    
    def __init__(self, config: Dict[str, Any]) -> None:
        """Initialize the DataGenerator with a configuration dictionary."""
        self.config = config
        self.data: Dict[str, Any] = {}

    def process_chartevents(self) -> None:
        """Loads and processes CHARTEVENTS data."""
        if 'hadm_ids' not in self.data:
            raise ValueError("HADM IDs not available. Run load_data() first.")

        print("\nProcessing CHARTEVENTS...")
        
        chartevents_path = self.config['chartevents_path']
        hadm_ids = self.data['hadm_ids']
        
        chartevents_cols_to_keep = [col.upper() for col in clinical_events.physio_cols_to_keep['chartevents']]
        
        chartevents_filtered = clinical_events.load_and_process_csv_for_hadm_ids(
            chartevents_path, 
            hadm_ids, 
            usecols=chartevents_cols_to_keep, 
            source='chartevents'
        )
        
        chart_events_filtered = chartevents_filtered[chartevents_filtered["value"] > 0]
        chart_events_filtered.rename(columns={'itemid': 'item_id'}, inplace=True)

        items_df = self.data['items_df']
        
        chart_events_filtered = inputs.attach_item_labels(chart_events_filtered, items_df)

        top_items = chart_events_filtered['item_id'].value_counts()[:100]
        chart_events_filtered_top_items = chart_events_filtered[chart_events_filtered['item_id'].isin(top_items.index)]
        
        chart_events_filtered_top_items_normalized = clinical_events.normalize_value(chart_events_filtered_top_items)
        
        self.data['chartevents_normalized'] = chart_events_filtered_top_items_normalized
        
        out_path = self.config['chartevents_normalized_parquet_path']
        print(f"\nSaving chartevents_normalized to {out_path}...")
        self.data['chartevents_normalized'].to_parquet(out_path, index=False, engine="pyarrow")
        print("Save complete.")

    def load_data(self) -> None:
        """Loads and preprocesses all initial data."""
        self.data = loaders.load_initial_data(
            full_numerics_dir=self.config['full_numerics_dir'],
            records_numerics_path=self.config['records_numerics_path'],
            icu_stays_path=self.config['icu_stays_path'],
            items_path=self.config['items_path'],
            all_trigger_meds_path=self.config['all_trigger_meds_path']
        )
        
        self.data['inputevents_mv'] = inputs.load_mv_data(
            input_data_dir=self.config['input_data_dir'],
            items_df=self.data['items_df'],
            hadm_ids=self.data['hadm_ids']
        )
        
        trigger_itemids = triggers.get_trigger_itemids(self.data['all_trigger_meds'])
        
        self.data['inputevents_mv_trigger_filtered'] = inputs.filter_mv_hypo_meds(
            self.data['inputevents_mv'],
            trigger_itemids
        )

    def process_triggers_and_clusters(self) -> None:
        """Computes triggers and action clusters."""
        if 'inputevents_mv_trigger_filtered' not in self.data:
            raise ValueError("MV trigger filtered data not loaded. Run load_data() first.")

        self.data['trig_clustered'] = clustering.get_trigger_clusters(
            self.data['inputevents_mv_trigger_filtered'],
            uptitration_rel_threshold=self.config.get('uptitration_rel_threshold', 0.25),
            min_abs_change=self.config.get('min_abs_change', 0.02),
            window_minutes=self.config.get('cluster_window_minutes', 20),
            min_triggers_in_cluster=self.config.get('min_triggers_in_cluster', 1)
        )

    def process_waveforms(self) -> None:
        """Processes waveform data."""
        files_with_core = waveforms.list_files_with_core_signals(
            self.config['waveform_dir'], 
            self.config['core_signals_to_keep']
        )
        
        waveform_meta = waveforms.build_waveform_metadata_from_files(files_with_core)
        
        self.data['consolidated_waveforms'] = waveforms.consolidate_waveforms(
            waveform_meta, 
            gap_hours=self.config.get('waveform_gap_hours', 2)
        )

    def align_data(self) -> None:
        """Aligns actions to consolidated waveform segments."""
        if 'trig_clustered' not in self.data or 'consolidated_waveforms' not in self.data:
            raise ValueError("Trigger clusters or consolidated waveforms not generated.")

        self.data['aligned_consolidated'] = alignment.align_actions_to_consolidated_segments(
            self.data['trig_clustered'], 
            self.data['consolidated_waveforms'], 
            window_minutes=self.config.get('alignment_window_minutes', 10)
        )
        
        self.data['best_segments'] = alignment.pick_best_segment_per_action(
            self.data['aligned_consolidated']
        )

    def harmonize_waveforms(self) -> None:
        """Harmonizes raw waveform data based on aligned segments."""
        if 'aligned_consolidated' not in self.data:
            raise ValueError("Aligned consolidated data not available. Run align_data() first.")

        files_df = harmonization.save_aligned_waveform_filelist(
            self.data['aligned_consolidated'],
            self.config['aligned_waveform_files_csv_path']
        )

        file_list = files_df["file_path"].dropna().unique().tolist()
        self.data['harmonized_waveforms'] = harmonization.load_harmonized_waveforms_from_list(file_list)

        # Save the harmonized waveforms
        out_path = self.config['harmonized_waveforms_parquet_path']
        print(f"\nSaving harmonized_waveforms to {out_path}...")
        self.data['harmonized_waveforms'].to_parquet(out_path, index=False, engine="pyarrow")
        print("Save complete.")

    def clean_and_smooth_waveforms(self) -> None:
        """Downsamples, cleans, despikes, and smoothes the harmonized waveforms."""
        if 'harmonized_waveforms' not in self.data:
            raise ValueError("Harmonized waveforms not available. Run harmonize_waveforms() first.")

        print("\nDownsampling waveforms to 10s mean...")
        dsr = cleaning.downsample_every_10s_mean(self.data['harmonized_waveforms'])
        
        print("Cleaning downsampled waveforms...")
        wf_10s_clean = cleaning.clean_waveform_df(dsr, drop_all_nan_physio=True)
        
        print("Despiking cleaned waveforms...")
        wf_10s_despiked = cleaning.despike_waveforms(
            wf_10s_clean,
            time_col="absolute_timestamp",
            group_cols=("hadm_id", "record_name"),
            action="mask",
            interpolate_small_gaps_pts=2,
        )

        print("Smoothing despiked waveforms...")
        smoothed_frames = []
        for chunk in cleaning.run_waveform_pipeline(
            wf_10s_despiked,
            do_zero_center=False,
            do_zscore=False,
            smooth_neighbors=4,
            smooth_variants=("raw",),
            out_suffix="_smooth4",
        ):
            smoothed_frames.append(chunk)
        
        wf_10s_smoothed_despiked = pd.concat(smoothed_frames, ignore_index=True)
        self.data['cleaned_smoothed_waveforms'] = wf_10s_smoothed_despiked

        out_path = self.config['cleaned_waveforms_parquet_path']
        print(f"\nSaving cleaned_smoothed_waveforms to {out_path}...")
        self.data['cleaned_smoothed_waveforms'].to_parquet(out_path, index=False, engine="pyarrow")
        print("Save complete.")


    def run_pipeline(self) -> Dict[str, Any]:
        """Runs the full data generation pipeline."""
        self.load_data()
        self.process_chartevents()
        self.process_triggers_and_clusters()
        self.process_waveforms()
        self.align_data()
        self.harmonize_waveforms()
        self.clean_and_smooth_waveforms()
        return self.data
