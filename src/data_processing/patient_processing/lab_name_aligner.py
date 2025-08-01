import os
import json
import pickle
import pandas as pd
from typing import List, Dict

from LLM_utils import run_LLM

class LabNameAligner:
    """
    A utility class to align lab names from chartevents with labevents using an LLM.

    This class encapsulates the logic for generating prompts, interacting with the LLM,
    caching the results, and applying the alignment to the chartevents dataframe.
    """

    def __init__(self, chartevents_df: pd.DataFrame, labevents_df: pd.DataFrame, temp_path: str, use_cache: bool = True):
        """
        Initializes the LabNameAligner.

        Args:
            chartevents_df (pd.DataFrame): Dataframe containing chartevents data, specifically with a 'category' and 'name' column.
            labevents_df (pd.DataFrame): Dataframe containing labevents data, with a 'name' and 'itemid' column.
            temp_path (str): Path to the directory for storing cached alignment files.
            use_cache (bool): Whether to use a cached alignment map if available.
        """
        self.chartevents_df = chartevents_df
        self.labevents_df = labevents_df
        self.temp_path = temp_path
        self.use_cache = use_cache
        self.alignment_path = os.path.join(self.temp_path, "lab_name_alignment.pkl")

    def align(self) -> pd.DataFrame:
        """
        Performs the lab name alignment.

        It gets the alignment map (from cache or by calling the LLM) and then
        applies this map to the chartevents dataframe.

        Returns:
            pd.DataFrame: The chartevents dataframe with lab names and itemids aligned.
        """
        if self.chartevents_df is None or self.chartevents_df.empty or self.labevents_df is None or self.labevents_df.empty:
            print("Skipping lab name alignment due to empty dataframes.")
            return self.chartevents_df.copy()

        print("Aligning lab names...")
        alignment_map = self._get_alignment_map()

        # Apply the alignment
        reference_name_to_itemid = self.labevents_df.drop_duplicates(subset='name').set_index('name')['itemid'].to_dict()
        
        aligned_chartevents_df = self.chartevents_df.copy()
        labs_mask = aligned_chartevents_df['category'] == 'Labs'
        labs_df = aligned_chartevents_df.loc[labs_mask].copy()
        
        labs_df['ref_name'] = labs_df['name'].map(alignment_map)
        labs_df['name'] = labs_df['ref_name'].combine_first(labs_df['name'])
        labs_df['itemid'] = labs_df['ref_name'].map(reference_name_to_itemid).combine_first(labs_df['itemid'])
        
        aligned_chartevents_df.loc[labs_mask, ['name', 'itemid']] = labs_df[['name', 'itemid']]
        
        print("Lab name alignment applied.")
        return aligned_chartevents_df

    def _get_alignment_map(self) -> Dict[str, str]:
        """
        Retrieves the lab name alignment map, either from cache or by computing it via an LLM.
        """
        if self.use_cache and os.path.exists(self.alignment_path):
            print("Loading lab name alignment from cache...")
            with open(self.alignment_path, "rb") as f:
                return pickle.load(f)
        
        print("Computing lab name alignment via LLM...")
        labs_in_charts = self.chartevents_df[self.chartevents_df['category'] == 'Labs']['name'].unique()
        reference_lab_names = self.labevents_df['name'].unique()
        
        system_prompt = (
            "You are a helpful matching assistant. Map each new lab name to the best matching reference name from the list provided. "
            "Return only valid JSON (no markdown, no triple backticks)."
        )
        
        matched_results = {}
        chunk_size = 10
        for i in range(0, len(labs_in_charts), chunk_size):
            chunk = labs_in_charts[i:i + chunk_size]
            prompt = self._get_align_labs_prompt(list(chunk), list(reference_lab_names))
            try:
                raw_results = run_LLM(system_prompt, prompt, iterations=1, model="gpt-4o")
                json_results = json.loads(raw_results)
                matched_results.update(json_results)
            except Exception as e:
                print(f"Error during LLM call or JSON parsing: {e}")
        
        with open(self.alignment_path, "wb") as f:
            pickle.dump(matched_results, f)

        return matched_results

    def _get_align_labs_prompt(self, chartname_list: List[str], labname_big_list: List[str]) -> str:
        """Generates the prompt for the LLM to align lab names."""
        return f'''
        Your task is to map a set of new lab names (list A) to their best matching names from a reference list (list B).
        For each name in list A, find the most appropriate corresponding name in list B.

        IMPORTANT:
        - Return only a valid JSON object without any additional text or markdown formatting.
        - Do NOT include triple backticks or any markdown markers.
        - The JSON object should be a flat dictionary mapping names from A to names from B.

        Output format:
        {{"new_name_from_A": "reference_name_from_B", "new_name2_from_A": "reference_name2_from_B", ...}}

        A) New lab names: 
        {chartname_list}

        B) Reference list: 
        {labname_big_list}
        ''' 