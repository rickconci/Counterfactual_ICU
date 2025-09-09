import pandas as pd
import numpy as np

def get_trigger_itemids(all_trigger_meds: pd.DataFrame) -> list[int]:
    """
    Extracts and returns a list of trigger item IDs from the trigger meds dataframe.
    """
    trigger_itemids = (
        all_trigger_meds["item_ids"]
        .str.split(",")                 # split into lists of strings
        .explode()                      # flatten into one long Series
        .str.strip()                    # remove spaces
        .astype(int)                    # convert to ints
        .tolist()                       # back to Python list
    )
    return trigger_itemids

def compute_triggers_simple(
    df: pd.DataFrame,
    uptitration_rel_threshold: float = 0.25,  # ≥25% relative increase
    min_abs_change: float = 0.02,             # ignore micro-jitters smaller than this
) -> pd.DataFrame:
    """
    Triggers (on normalized rate basis):
    1) Bolus: input_name contains 'bolus'
    2) Uptitration: (curr - prev) > 0, (curr - prev)/prev > rel_threshold, and abs(curr - prev) >= min_abs_change
    3) New medication: prev == 0 and curr > 0
        • Special case: if item_label is 'NaCl 0.9%' or 'LR', require curr > 0.5 (normalized) to count as a start
    """
    # --- safe copy / index handling
    df = df.copy()
    if isinstance(df.index, pd.MultiIndex):
        idx_names = [n for n in df.index.names if n is not None]
        if any(n in df.columns for n in idx_names):
            df = df.reset_index(drop=True)
        else:
            df = df.reset_index()
    else:
        idx_name = df.index.name
        if idx_name is not None and idx_name in df.columns:
            df = df.reset_index(drop=True)
        else:
            df = df.reset_index()

    # --- coerce times
    for c in ("start_time", "end_time"):
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")

    # --- unified rate (prefer normalized)
    df["rate_norm"] = (
        pd.to_numeric(df.get("rate/weight_norm"), errors="coerce")
        .fillna(pd.to_numeric(df.get("rate/weight"), errors="coerce"))
        .fillna(pd.to_numeric(df.get("rate"), errors="coerce"))
        .fillna(0.0)
    ).clip(0, 1)

    # --- grouping keys
    item_key = "item_label" if "item_label" in df.columns else ("item_id" if "item_id" in df.columns else None)
    group_keys = [k for k in ["subject_id", "hadm_id", item_key] if k and k in df.columns]
    if not group_keys or (item_key is None):
        raise KeyError("Need subject_id, hadm_id, and one of ['item_label','item_id'] to group titrations.")

    # --- order and previous rate
    sort_keys = group_keys + (["start_time"] if "start_time" in df.columns else [])
    df = df.sort_values(sort_keys, kind="mergesort").copy()
    df["prev_rate"] = df.groupby(group_keys, dropna=False)["rate_norm"].shift(1).fillna(0.0)

    # (1) bolus?
    is_bolus = (
        df["input_name"].astype(str).str.contains("bolus", case=False, na=False)
        if "input_name" in df.columns else pd.Series(False, index=df.index)
    )

    # (2) uptitration (relative + absolute guard)
    prev = df["prev_rate"]
    curr = df["rate_norm"]
    delta = curr - prev
    with np.errstate(divide="ignore", invalid="ignore"):
        rel_increase = np.where(prev > 0, delta / prev, np.nan)
    uptitrate = (delta > 0) & (np.nan_to_num(rel_increase, nan=0.0) > uptitration_rel_threshold) & (delta >= min_abs_change)

    # (3) new medication start with saline/LR gate
    first_start_base = (prev == 0) & (curr > 0)
    saline_or_lr = df.get("item_label", pd.Series("", index=df.index)).isin(["NaCl 0.9%", "LR"])
    first_start = first_start_base & ~(saline_or_lr & (curr <= 0.5))

    # Compose trigger + reason
    trigger = is_bolus | uptitrate | first_start
    reason = np.where(
        is_bolus, "bolus",
        np.where(
            uptitrate, f"uptitrate_gt{int(uptitration_rel_threshold*100)}pct_abs>={min_abs_change}",
            np.where(first_start, "start_new_med", "")
        )
    )

    df["trigger"] = trigger
    df["trigger_reason"] = reason
    return df
