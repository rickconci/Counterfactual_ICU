import os
import pandas as pd
from typing import Dict

physio_cols_to_keep = {
    "chartevents": [
        "subject_id",
        "hadm_id",
        "itemid",
        "charttime",
        "valuenum",
        "valueuom",
    ],
    "outputevents": [
        "subject_id",
        "hadm_id",
        "itemid",
        "charttime",
        "value",
        "valueuom",
    ],
    "labevents": [
        "subject_id",
        "hadm_id",
        "itemid",
        "charttime",
        "valuenum",
        "valueuom",
    ],
    "micro_events": [
        "subject_id",
        "hadm_id",
        "charttime",
        "SPEC_TYPE_DESC",
        "ORG_NAME",
        "ab_name",
        "INTERPRETATION",
    ],
}

def load_and_process_csv_for_hadm_ids(
    file_path, hadm_ids, chunk_size=1000000, usecols=None, source=None
):
    """
    Loads a CSV file in chunks, filtering for rows with hadm_id in hadm_ids,
    and applies source-specific processing.
    If the pickle already exists in the temp_dfs folder, load and return it.
    """
    # Ensure the directory exists
    temp_dir = "src_new/temp_dfs"
    os.makedirs(temp_dir, exist_ok=True)

    # Define a unique pickle filename based on the CSV file name and source.
    base_name = os.path.basename(file_path).replace(".csv", "")
    pickle_file = os.path.join(temp_dir, f"{base_name}_{source}.pkl")

    # If the pickle file exists, load and return it.
    if os.path.exists(pickle_file):
        print(f"Loading existing pickle file: {pickle_file}")
        return pd.read_pickle(pickle_file)

    chunks = []
    for i, chunk in enumerate(
        pd.read_csv(file_path, chunksize=chunk_size, usecols=usecols)
    ):
        chunk.columns = chunk.columns.str.lower()
        # Filter the chunk for the given hadm_ids.
        filtered_chunk = chunk[chunk["hadm_id"].isin(hadm_ids)]

        # Apply source-specific renaming:
        if source == "chartevents":
            filtered_chunk = filtered_chunk.rename(columns={"valuenum": "value"})
        elif source == "inputevents":
            filtered_chunk = filtered_chunk.rename(
                columns={"amount": "value", "starttime": "charttime"}
            )
        elif source == "labevents":
            filtered_chunk = filtered_chunk.rename(columns={"valuenum": "value"})
        elif source == "emar":
            filtered_chunk = filtered_chunk.rename(
                columns={
                    "medication": "name",
                    "event_txt": "value",
                    "emar_id": "itemid",
                }
            )
        elif source == "micro_events":
            filtered_chunk = filtered_chunk.rename(
                columns={
                    "micro_specimen_id": "itemid",
                    "test_name": "name",
                    "comments": "value",
                }
            )

        # Drop rows with missing 'value'
        filtered_chunk = filtered_chunk.dropna(subset=["value"])

        # Convert 'charttime' to datetime if it exists in the columns.
        if "charttime" in filtered_chunk.columns:
            filtered_chunk["charttime"] = pd.to_datetime(
                filtered_chunk["charttime"], errors="coerce"
            )

        chunks.append(filtered_chunk)
        print(f"Chunk {i} processed, shape after filter: {filtered_chunk.shape}")

    # Concatenate all chunks
    result = pd.concat(chunks, ignore_index=True)

    # Save the resulting DataFrame as a pickle file.
    print(f"Saving result to pickle file: {pickle_file}")
    result.to_pickle(pickle_file)

    return result

def normalize_value(df, value_col='value', group_col='item_id'):
    """
    Normalizes the values for each itemid in the DataFrame based on their own mean and std.
    Any standardized values greater than 3 or less than -3 (i.e. outliers) are clipped to ±3.
    """
    # Compute the mean and std for each group (each itemid)
    means = df.groupby(group_col)[value_col].transform('mean')
    stds = df.groupby(group_col)[value_col].transform('std')
    
    # Standardize the values: (value - mean) / std
    normalized = (df[value_col] - means) / stds
    
    # Clip the standardized values to the range [-3, 3]
    normalized_clipped = normalized.clip(lower=-3, upper=3)
    
    # Create a copy of the DataFrame with the normalized values
    df_normalized = df.copy()
    df_normalized[value_col] = normalized_clipped
    
    return df_normalized
