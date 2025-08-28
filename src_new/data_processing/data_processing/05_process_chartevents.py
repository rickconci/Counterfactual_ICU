import os

import pandas as pd
from config import HELPFUL_DF_DIR, MIMIC_DATA_DIR, physio_cols_to_keep


def load_and_process_csv_for_hadm_ids(
    file_path, hadm_ids, chunk_size=1000000, usecols=None, source=None
):
    """
    Loads a CSV file in chunks, filtering for rows with hadm_id in hadm_ids,
    and applies source-specific processing:
      - For 'chartevents': renames 'valuenum' to 'value'
      - For 'inputevents': renames 'amount' -> 'value' and 'starttime' -> 'charttime'
      - For 'labevents': renames 'valuenum' to 'value'
      - For 'emar': renames 'medication' -> 'name', 'event_txt' -> 'value', 'emar_id' -> 'itemid'
      - For 'micro_events': renames 'micro_specimen_id' -> 'itemid', 'test_name' -> 'name', 'comments' -> 'value'
    Drops rows with NA in 'value' and converts 'charttime' (or starttime) to datetime.

    If the pickle already exists in the temp_dfs folder, load and return it.
    """
    # Ensure the directory exists
    temp_dir = "../temp_dfs"
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
        print(chunk.columns)
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


if __name__ == "__main__":
    relevant_pt_id_path = HELPFUL_DF_DIR / "relevant_patient_ids.csv"
    relevant_patient_ids = pd.read_csv(relevant_pt_id_path)
    hadm_ids = relevant_patient_ids["HADM_ID"]

    chartevents_path = MIMIC_DATA_DIR / "CHARTEVENTS.csv"
    chartevents_filtered = load_and_process_csv_for_hadm_ids(
        chartevents_path,
        hadm_ids,
        usecols=physio_cols_to_keep["chartevents"],
        source="chartevents",
    )
