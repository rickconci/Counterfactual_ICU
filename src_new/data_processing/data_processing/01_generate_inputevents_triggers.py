from typing import Optional

import numpy as np
import pandas as pd
from config import (
    HELPFUL_DF_DIR,
    MIMIC_DATA_DIR,
    PROCESSED_DATA_DIR,
    treatment_cols_to_keep,
)
from data_utils import attach_item_labels


def clean_inputevents_mv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean IV/medication input events and return a tidy dataframe with:
       subject_id, hadm_id, item_id, input_name, input_class,
       start_time, end_time, rate, rate_uom, rate/weight
    Removes cancelled/rejected rows. Sorted by hadm_id, item_id, start_time.
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

    # Compute rate/weight
    if col_weight is not None:
        rw = out[col_rate] / out[col_weight].replace({0: np.nan})
    else:
        rw = np.nan

    # Rename to desired schema
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
    out = out.rename(columns=rename_map)

    # Ensure rate_uom exists
    if "rate_uom" not in out.columns:
        out["rate_uom"] = pd.NA

    out["rate/weight"] = rw

    # Sort
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
            "rate/weight",
        ]
    ].reset_index(drop=True)


def build_inputevents_mv_df(
    items_df: pd.DataFrame, hadm_ids: list[int]
) -> pd.DataFrame:
    """
    Clean IV/medication input events and return a tidy dataframe with:
       subject_id, hadm_id, item_id, input_name, input_class,
       start_time, end_time, rate, rate_uom, rate/weight
    Removes cancelled/rejected rows. Sorted by hadm_id, item_id, start_time.
    """
    inputevents_mv_df = pd.read_csv(MIMIC_DATA_DIR / "INPUTEVENTS_MV.csv")
    inputevents_mv_df = clean_inputevents_mv(inputevents_mv_df)
    inputevents_mv_df = attach_item_labels(inputevents_mv_df, items_df)
    inputevents_mv_df = inputevents_mv_df[inputevents_mv_df["hadm_id"].isin(hadm_ids)]
    inputevents_mv_df = inputevents_mv_df.sort_values(
        ["hadm_id", "start_time"], ascending=[True, True]
    ).set_index("hadm_id", drop=False)
    inputevents_mv_df.head()


def build_inputevents_cv_df(
    df: pd.DataFrame, items_df: pd.DataFrame, hadm_ids: list[int]
) -> pd.DataFrame:
    """
    Clean IV/medication input events and return a tidy dataframe with:
       subject_id, hadm_id, item_id, input_name, input_class,
       start_time, end_time, rate, rate_uom, rate/weight
    Removes cancelled/rejected rows. Sorted by hadm_id, item_id, start_time.
    """
    input_event_cv_df = pd.read_csv(MIMIC_DATA_DIR / "INPUTEVENTS_CV.csv")
    input_event_cv_df.columns = input_event_cv_df.columns.str.lower()
    input_event_cv_df = input_event_cv_df[treatment_cols_to_keep["inputevents_cv"]]
    input_event_cv_df.rename(columns={"itemid": "item_id"}, inplace=True)
    input_event_cv_df = attach_item_labels(input_event_cv_df, items_df)
    input_event_cv_df.dropna(subset=["hadm_id"], inplace=True)
    input_event_cv_df = input_event_cv_df[input_event_cv_df["hadm_id"].isin(hadm_ids)]
    input_event_cv_df = input_event_cv_df.sort_values(
        ["hadm_id", "charttime"], ascending=[True, True]
    ).set_index("hadm_id", drop=False)
    return input_event_cv_df


import pandas as pd

# --- helpers ---------------------------------------------------------------


def _ensure_cols_from_index(
    df: pd.DataFrame, names=("subject_id", "hadm_id")
) -> pd.DataFrame:
    """
    If any of `names` are index levels, bring them out as columns *only if they
    don't already exist as columns*. If they already exist, drop that index level.
    Leaves other index levels untouched.
    """
    df2 = df

    def _has_level(idx, name):
        if isinstance(idx, pd.MultiIndex):
            return name in idx.names
        else:
            return idx.name == name

    for n in names:
        if not n:
            continue
        in_cols = n in df2.columns
        in_index = _has_level(df2.index, n)

        if in_index and in_cols:
            # Drop the index level; keep the existing column
            if isinstance(df2.index, pd.MultiIndex):
                df2 = df2.reset_index(level=n, drop=True)
            else:
                # single Index named n
                df2 = df2.reset_index(drop=True)
        elif in_index and not in_cols:
            # Materialize the index level as a new column
            if isinstance(df2.index, pd.MultiIndex):
                df2 = df2.reset_index(level=n)
            else:
                df2 = df2.reset_index()  # single Index named n -> becomes a column
        # else: neither in index nor in columns -> create empty column (rare)
        elif (not in_index) and (not in_cols):
            df2[n] = pd.NA

    return df2


def _resolve_encounter_keys(df: pd.DataFrame):
    keys = []
    if "subject_id" in df.columns:
        keys.append("subject_id")
    if "hadm_id" in df.columns:
        keys.append("hadm_id")
    if not keys:
        raise KeyError(
            "Need at least one of 'subject_id' or 'hadm_id' (in columns or index)."
        )
    return keys


def _resolve_med_key(df: pd.DataFrame):
    """Prefer a human-readable medication key."""
    for k in ("item_label", "input_name", "item_id"):
        if k in df.columns:
            return k
    raise KeyError(
        "Need one of 'item_label', 'input_name', or 'item_id' to group by medication."
    )


def compute_triggers(
    df: pd.DataFrame,
    relative_change: float = 0.30,  # ≥30% up or down = substantial
    abs_min_change: float = 0.02,  # ignore tiny jitter
    nacl_high_rate: float = 250.0,  # mL/h considered "much higher" for NaCl 0.9%
    nacl_jump_abs: float = 200.0,  # absolute jump for NaCl to count
    throttle: bool = False,  # still off by default
    window_max_trig: int = 2,
    window_minutes: int = 10,
) -> pd.DataFrame:
    df0 = df.copy()

    # bring encounter keys out of the index if needed
    df0 = _ensure_cols_from_index(df0, names=("subject_id", "hadm_id"))
    enc_keys = _resolve_encounter_keys(df0)
    med_key = _resolve_med_key(df0)

    # time columns
    for c in ("start_time", "end_time"):
        if c in df0.columns:
            df0[c] = pd.to_datetime(df0[c], errors="coerce")

    # normalize to a single numeric "rate_norm"
    # your upstream cleaning already makes per-kg units populate "rate/weight"
    has_rate_weight = "rate/weight" in df0.columns
    df0["rate_weight_norm"] = df0["rate/weight"] if has_rate_weight else np.nan
    df0["rate_norm"] = df0["rate_weight_norm"].where(
        ~df0["rate_weight_norm"].isna(), df0["rate"]
    )
    df0["rate_norm"] = pd.to_numeric(df0["rate_norm"], errors="coerce").fillna(0.0)

    # pushes/boluses: if present but lacking a numeric rate, mark as pulse=1
    if "input_class" in df0.columns:
        is_bolus = (
            df0["input_class"]
            .astype(str)
            .str.contains("bolus|push", case=False, na=False)
        )
        df0.loc[is_bolus & (df0["rate_norm"] == 0), "rate_norm"] = 1.0

    # order and previous rate within (encounter, medication)
    order_cols = enc_keys + [med_key, "start_time"]
    df0 = df0.sort_values(order_cols, kind="mergesort").copy()
    df0["prev_rate"] = df0.groupby(enc_keys + [med_key], group_keys=False)[
        "rate_norm"
    ].shift(1)

    # triggers
    first_admin = (df0["prev_rate"].fillna(0) == 0) & (df0["rate_norm"] > 0)
    delta = df0["rate_norm"] - df0["prev_rate"].fillna(0)
    abs_delta = delta.abs()
    with np.errstate(divide="ignore", invalid="ignore"):
        rel_delta = (
            (abs_delta / df0["prev_rate"])
            .replace([np.inf, -np.inf], np.nan)
            .fillna(1.0)
        )

    significant_increase = (
        (delta > 0) & (rel_delta >= relative_change) & (abs_delta >= abs_min_change)
    )
    significant_decrease = (
        (delta < 0) & (rel_delta >= relative_change) & (abs_delta >= abs_min_change)
    )

    # NaCl special case
    if "item_label" in df0.columns:
        name_series = df0["item_label"].astype(str).str.strip().str.lower()
    else:
        name_series = pd.Series("", index=df0.index)
    nacl_aliases = {
        "nacl 0.9%",
        "ns",
        "normal saline",
        "sodium chloride 0.9%",
        "0.9% sodium chloride",
    }
    is_nacl = name_series.isin(nacl_aliases)
    nacl_high = (df0["rate_norm"] >= nacl_high_rate) & (
        (df0["prev_rate"].fillna(0) < nacl_high_rate) | (abs_delta >= nacl_jump_abs)
    )

    # explicit fluid bolus (if present)
    if "input_name" in df0.columns:
        is_fluid_bolus = (
            df0["input_name"].astype(str).str.contains("bolus", case=False, na=False)
        )
    else:
        is_fluid_bolus = pd.Series(False, index=df0.index)

    base_trigger = (
        first_admin | significant_increase | significant_decrease | is_fluid_bolus
    )

    # restrict NaCl triggers
    trigger = base_trigger.copy()
    trigger.loc[is_nacl] = nacl_high.loc[is_nacl] | is_fluid_bolus.loc[is_nacl]

    # reasons
    reasons = np.where(first_admin, "start", "")
    reasons = np.where(significant_increase & (reasons == ""), "increase", reasons)
    reasons = np.where(significant_decrease & (reasons == ""), "decrease", reasons)
    reasons = np.where(is_fluid_bolus & (reasons == ""), "bolus", reasons)
    reasons = np.where(is_nacl & nacl_high, "NaCl_high_rate", reasons)
    reasons = np.where(trigger & (reasons == ""), "trigger", reasons)

    df0["trigger"] = trigger.astype(bool)
    df0["trigger_reason"] = reasons

    # optional throttle (defaults OFF)
    if throttle:

        def throttle_group(g):
            g2 = g.sort_values("start_time").copy()
            roll = (
                g2.set_index("start_time")["trigger"]
                .astype(int)
                .rolling(f"{window_minutes}min")
                .sum()
            )
            keep = roll <= window_max_trig
            kept = g2["trigger"].copy()
            kept.loc[keep.index] = g2.loc[keep.index, "trigger"] & keep.values
            return kept.sort_index()

        throttled = df0.groupby(enc_keys + [med_key], group_keys=False).apply(
            throttle_group
        )
        throttled = throttled.astype(bool).reindex(df0.index)
        df0["trigger_reason"] = np.where(
            df0["trigger"] & (~throttled),
            "throttled_rolling_window",
            df0["trigger_reason"],
        )
        df0["trigger"] = throttled.values

    return df0


def main_trigger_meds(save_dir=None):
    items_df = pd.read_csv(MIMIC_DATA_DIR / "D_ITEMS.csv")
    hadm_ids = pd.read_csv(HELPFUL_DF_DIR / "relevant_patient_ids.csv")[
        "hadm_id"
    ].tolist()
    trigger_meds_df = pd.read_csv(HELPFUL_DF_DIR / "all_trigger_meds.csv")

    inputevents_mv_df = build_inputevents_mv_df(items_df, hadm_ids)

    s = trigger_meds_df["item_id"].astype(str).str.strip()
    mask_num = s.str.fullmatch(r"\d+")
    trigger_ids = (
        pd.to_numeric(s.where(mask_num), errors="coerce").dropna().astype("int64")
    )

    inputevents_mv_df_filtered_trigger_meds = inputevents_mv_df[
        inputevents_mv_df["item_id"].isin(trigger_ids)
    ]

    input_mv_triggers = compute_triggers(inputevents_mv_df_filtered_trigger_meds)
    input_mv_triggers = input_mv_triggers.sort_values(
        ["hadm_id", "start_time"]
        if "subject_id" not in input_mv_triggers.columns
        else ["subject_id", "hadm_id", "start_time"]
    )
    input_mv_triggers = compute_action_clusters(
        input_mv_triggers, window_minutes=20, min_events_in_cluster=1
    )

    if save_dir is not None:
        input_mv_triggers.to_csv(save_dir / "input_mv_triggers.csv", index=False)


def compute_action_clusters(
    df_with_triggers: pd.DataFrame,
    window_minutes: int = 15,
    min_events_in_cluster: int = 2,
) -> pd.DataFrame:
    """
    Build action clusters WITHIN an encounter (per hadm_id, and subject_id if present),
    scanning triggers across ALL meds. A cluster starts at a trigger, includes subsequent
    triggers up to (start + window_minutes), then the next cluster starts at the first
    trigger strictly after that window.
    """
    # Make sure subject_id/hadm_id are columns (not only index) and not duplicated
    df = _ensure_cols_from_index(
        df_with_triggers, names=("subject_id", "hadm_id")
    ).copy()
    enc_keys = _resolve_encounter_keys(df)

    # Hygiene: time & trigger types
    df["start_time"] = pd.to_datetime(df["start_time"], errors="coerce")
    df["trigger"] = df.get("trigger", False)
    df["trigger"] = df["trigger"].fillna(False).astype(bool)

    # Init outputs
    df["action_cluster_id"] = np.nan
    df["action_cluster_size"] = np.nan
    df["action_cluster_rank"] = np.nan  # 1..N within cluster

    def cluster_one_enc(g):
        g = g.sort_values("start_time").copy()
        # ignore rows without a timestamp
        g = g[~g["start_time"].isna()]
        if g.empty:
            return g

        trig_idx = g.index[g["trigger"]].to_list()
        if not trig_idx:
            return g

        cluster_id = 0
        i = 0
        while i < len(trig_idx):
            start_idx = trig_idx[i]
            t0 = g.loc[start_idx, "start_time"]
            window_end = t0 + pd.Timedelta(minutes=window_minutes)

            # collect all triggers within [t0, window_end] (inclusive at end)
            members = [start_idx]
            j = i + 1
            while j < len(trig_idx) and g.loc[trig_idx[j], "start_time"] <= window_end:
                members.append(trig_idx[j])
                j += 1

            if len(members) >= min_events_in_cluster:
                cluster_id += 1
                size = len(members)
                for rank, idx in enumerate(members, start=1):
                    g.at[idx, "action_cluster_id"] = cluster_id
                    g.at[idx, "action_cluster_size"] = size
                    g.at[idx, "action_cluster_rank"] = rank
                # jump to the first trigger strictly AFTER the window
                i = j
            else:
                i += 1

            # next iteration starts at the next trigger (or first after window)

        return g

    df = df.groupby(enc_keys, group_keys=False, sort=False).apply(cluster_one_enc)

    # Optional: make IDs integers (keep NaN for non-clustered rows)
    if df["action_cluster_id"].notna().any():
        df.loc[df["action_cluster_id"].notna(), "action_cluster_id"] = df.loc[
            df["action_cluster_id"].notna(), "action_cluster_id"
        ].astype(int)
        df.loc[df["action_cluster_size"].notna(), "action_cluster_size"] = df.loc[
            df["action_cluster_size"].notna(), "action_cluster_size"
        ].astype(int)
        df.loc[df["action_cluster_rank"].notna(), "action_cluster_rank"] = df.loc[
            df["action_cluster_rank"].notna(), "action_cluster_rank"
        ].astype(int)

    return df


if __name__ == "__main__":
    main_trigger_meds(PROCESSED_DATA_DIR)
