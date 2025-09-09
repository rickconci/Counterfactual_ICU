import pandas as pd
import numpy as np
from typing import Optional

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

def build_inputs_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean IV/medication input events and return a tidy dataframe with:
    subject_id, hadm_id, item_id, input_name, input_class,
    start_time, end_time, rate, rate_uom,
    patientweight,
    rate/weight,           # weight-normalized rate
    rate/weight_capped,    # per-item_id cap at 95th percentile
    rate/weight_norm       # per-item_id 0–1 normalization (divide by 95th pct)
    Rules:
    - Remove cancelled/rejected rows.
    - If rate_uom already encodes per-kg (e.g., 'mcg/kg/min', 'per kg'), then 'rate/weight' == 'rate'.
    - Otherwise, 'rate/weight' = rate / patientweight.
    - patientweight values of 1 or NaN are replaced with the cohort mean of valid weights.
    - For each item_id, cap 'rate/weight' at the 95th percentile.
    - Normalize per item_id by dividing by the 95th percentile.
    - Sorted by hadm_id, item_id, start_time.
    """
    if df.empty:
        return pd.DataFrame()

    # ---- Column resolver
    lower = {c.lower(): c for c in df.columns}

    def pick(*cands: str) -> Optional[str]:
        for c in cands:
            if c.lower() in lower:
                return lower[c.lower()]
        return None

    col_subject = pick("subject_id", "subjectid")
    col_hadm = pick("hadm_id", "hadmid")
    col_item = pick("item_id", "itemid")
    col_name = pick("ordercategoryname")
    col_class = pick("ordercategorydescription")
    col_start = pick("start_time", "starttime", "startdate", "charttime")
    col_end = pick("end_time", "endtime", "enddate", "stoptime")
    col_rate = pick("rate")
    col_rate_u = pick("rate_uom", "rateuom")
    col_weight = pick("patientweight", "weight", "actualbedweight")

    needed = [col_subject, col_hadm, col_item, col_start, col_end, col_rate]
    if any(c is None for c in needed):
        raise ValueError(f"Missing required columns; found: {needed}")

    # ---- Remove cancelled/rejected
    cancel_cols = [
        c
        for c in [
            pick("cancelreason"),
            pick("iscanceled"),
            pick("is_canceled"),
            pick("iscancelled"),
            pick("is_cancelled"),
            pick("canceled"),
            pick("cancelled"),
        ]
        if c is not None
    ]
    status_col = pick("statusdescription", "status", "orderstatus")
    comments_cancel_col = pick("comments_cancelledby", "comments_canceledby")

    keep = pd.Series(True, index=df.index)
    for c in cancel_cols:
        s = df[c]
        keep &= (
            s.isna()
            | (s == 0)
            | (s is False)
            | (s.astype(str).str.strip().isin(["0", "False", "false"]))
        )
    if status_col is not None:
        keep &= ~df[status_col].astype(str).str.contains(
            r"cancel|reject|discontinu", case=False, na=False
        )
    if comments_cancel_col is not None:
        keep &= df[comments_cancel_col].isna()

    dfc = df.loc[keep].copy()

    # ---- Select & normalize
    out_cols = [
        col_subject,
        col_hadm,
        col_item,
        col_name,
        col_class,
        col_start,
        col_end,
        col_rate,
    ]
    if col_rate_u is not None:
        out_cols.append(col_rate_u)
    if col_weight is not None:
        out_cols.append(col_weight)

    out = dfc[out_cols].copy()

    # Convert datetimes
    out[col_start] = pd.to_datetime(out[col_start], errors="coerce", utc=True)
    out[col_end] = pd.to_datetime(out[col_end], errors="coerce", utc=True)

    # Numeric conversions
    out[col_rate] = pd.to_numeric(out[col_rate], errors="coerce")
    if col_weight is not None:
        out[col_weight] = pd.to_numeric(out[col_weight], errors="coerce")

    # ---- Rename to desired schema (normalize names)
    rename_map = {
        col_subject: "subject_id",
        col_hadm: "hadm_id",
        col_item: "item_id",
        col_name: "input_name",
        col_class: "input_class",
        col_start: "start_time",
        col_end: "end_time",
        col_rate: "rate",
    }
    if col_rate_u is not None:
        rename_map[col_rate_u] = "rate_uom"
    if col_weight is not None:
        rename_map[col_weight] = "patientweight"

    out = out.rename(columns=rename_map)

    # Ensure expected columns exist
    for must_col in ["input_name", "input_class", "rate_uom", "patientweight"]:
        if must_col not in out.columns:
            out[must_col] = pd.NA

    # ---- Clean patientweight: replace 1 or NaN with mean of valid weights
    pw = pd.to_numeric(out["patientweight"], errors="coerce")
    valid = pw[(~pw.isna()) & (pw != 1)]
    if not valid.empty:
        mean_wt = valid.mean()
        pw = pw.mask((pw.isna()) | (pw == 1), mean_wt)
    out["patientweight"] = pw

    # ---- Compute rate/weight with per-kg guard
    ru = out["rate_uom"].astype(str)
    is_perkg = ru.str.contains(r"/\s*kg\b", case=False, na=False) | ru.str.contains(
        r"\bper\s*kg\b", case=False, na=False
    )

    rw = pd.Series(np.nan, index=out.index, dtype="float64")
    rw.loc[is_perkg] = out.loc[is_perkg, "rate"]
    denom = out["patientweight"].replace({0: np.nan})
    rw.loc[~is_perkg] = out.loc[~is_perkg, "rate"] / denom.loc[~is_perkg]
    out["rate/weight"] = rw

    # ---- Per-item_id capping at 95th percentile
    p95 = out.groupby("item_id")["rate/weight"].transform(
        lambda x: x.quantile(0.95)
    )
    out["rate/weight_capped"] = out["rate/weight"].clip(upper=p95)

    # ---- Per-item_id normalization by 95th percentile
    safe_p95 = p95.where((p95 > 0) & (~p95.isna()))
    norm = out["rate/weight_capped"] / safe_p95
    out["rate/weight_norm"] = norm.clip(lower=0, upper=1)

    if "input_name" in out.columns:
        mask_bolus = out["input_name"].astype(str).str.strip() == "03-IV Fluid Bolus"
        out.loc[mask_bolus, "rate/weight_norm"] = 1.0

    # ---- Sort
    out = out.sort_values(
        ["hadm_id", "item_id", "start_time"], kind="mergesort", na_position="last"
    )

    return out[
        [
            "subject_id",
            "hadm_id",
            "item_id",
            "input_name",
            "input_class",
            "start_time",
            "end_time",
            "rate",
            "rate_uom",
            "patientweight",
            "rate/weight",
            "rate/weight_capped",
            "rate/weight_norm",
        ]
    ].reset_index(drop=True)

def load_mv_data(
    input_data_dir: str, 
    items_df: pd.DataFrame, 
    hadm_ids: np.ndarray
) -> pd.DataFrame:
    '''
    Loads the MV data and preprocesses it.
    '''
    inputevents_mv_df = pd.read_csv(f"{input_data_dir}/INPUTEVENTS_MV.csv")
    inputevents_mv_df = build_inputs_df(inputevents_mv_df)
    inputevents_mv_df = attach_item_labels(inputevents_mv_df, items_df)
    inputevents_mv_df = inputevents_mv_df[inputevents_mv_df["hadm_id"].isin(hadm_ids)]
    inputevents_mv_df = inputevents_mv_df.sort_values(
        ["hadm_id", "start_time"], ascending=[True, True]
    ).set_index("hadm_id", drop=False)
    return inputevents_mv_df

def filter_mv_hypo_meds(
    inputevents_mv_df: pd.DataFrame, 
    trigger_itemids: list
) -> pd.DataFrame:
    inputevents_mv_df_trigger_filtered = inputevents_mv_df[
        inputevents_mv_df["item_id"].isin(trigger_itemids)
    ]
    inputevents_mv_df_trigger_filtered = inputevents_mv_df_trigger_filtered[
        ~inputevents_mv_df_trigger_filtered['input_name'].isin([
            '08-Antibiotics (IV)', 
            '10-Prophylaxis (IV)'
        ])
    ]
    
    return inputevents_mv_df_trigger_filtered
