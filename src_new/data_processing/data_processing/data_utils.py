import pandas as pd


def _enforce_pandas_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "hadm_id" in df:
        df["hadm_id"] = pd.to_numeric(df["hadm_id"], errors="coerce").astype("Int64")
    if "time_seconds" in df:
        df["time_seconds"] = pd.to_numeric(df["time_seconds"], errors="coerce").astype(
            "Int64"
        )
    for c in ["ABP MEAN", "NBP MEAN", "CVP", "HR", "RESP"]:
        if c in df:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")
    if "record_name" in df:
        df["record_name"] = df["record_name"].astype("string")
    if "absolute_timestamp" in df:
        df["absolute_timestamp"] = pd.to_datetime(
            df["absolute_timestamp"], errors="coerce", utc=True
        )
    for c in ["record_start_time", "record_end_time", "icu_admission_time"]:
        if c in df:
            df[c] = pd.to_datetime(df[c], errors="coerce")  # tz-naive for us timestamps
    return df


def _ensure_hadm_column(df: pd.DataFrame) -> pd.DataFrame:
    idx_names = list(df.index.names or [])
    if "hadm_id" in idx_names:
        if "hadm_id" in df.columns:
            # already a column → just drop the index to avoid duplicate
            df = df.reset_index(drop=True)
        else:
            # move index to columns
            df = df.reset_index()
    return df


def _to_utc(s: pd.Series) -> pd.Series:
    # parse and force timezone-aware UTC for safe comparisons
    return pd.to_datetime(s, errors="coerce", utc=True)


def attach_item_labels(inputs_df: pd.DataFrame, items_df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a string label for each item_id using D_ITEMS.
    Creates a new column 'item_label'. If no match, falls back to str(item_id).
    """
    # tolerate case differences (MIMIC-III/IV)
    items = items_df.rename(columns=str.lower)
    if not {"itemid", "label"}.issubset(items.columns):
        raise ValueError("items_df must have columns ITEMID and LABEL (any case).")

    items_slim = items[["itemid", "label"]].drop_duplicates()

    df = inputs_df.merge(items_slim, left_on="item_id", right_on="itemid", how="left")
    df = df.drop(columns=["itemid"])
    df["item_label"] = df["label"].fillna(df["item_id"].astype(str))
    df = df.drop(columns=["label"])  # keep things tidy (label now lives in item_label)
    return df
