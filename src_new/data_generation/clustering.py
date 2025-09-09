import pandas as pd
import numpy as np
from collections import Counter, defaultdict
from . import utils

def compute_action_clusters(
    df_with_triggers: pd.DataFrame,
    window_minutes: int = 20,        # cluster window length
    min_triggers_in_cluster: int = 1 # keep clusters with at least this many triggers
) -> pd.DataFrame:
    """
    Build non-overlapping action clusters per encounter:
    • pick t0 = first trigger start_time
    • include all triggers with start_time <= t0 + window
    • cluster window = [t0, t0 + window]
    • include *all rows* (trigger or not) that overlap the window:
        (start_time <= window_end) AND (end_time is NA OR end_time >= t0)
    • jump to the first trigger strictly after window_end and repeat.
    Notes:
    - action_cluster_id assigned to all members (triggers and non-triggers).
    - action_cluster_rank is assigned only to triggers (1..#triggers); non-triggers get <NA>.
    - action_cluster_size = number of triggers in the cluster.
    """
    df = df_with_triggers.copy()

    # ensure columns exist
    for col in ("action_cluster_id","action_cluster_size","action_cluster_rank"):
        if col not in df.columns:
            df[col] = pd.NA

    # times -> UTC
    if "start_time" in df.columns: df["start_time"] = utils.to_utc(df["start_time"])
    if "end_time"   in df.columns: df["end_time"]   = utils.to_utc(df["end_time"])

    enc_keys = utils.encounter_keys(df)

    def cluster_one_enc(g):
        g = g.sort_values("start_time", kind="mergesort").copy()
        trig_mask = g["trigger"].fillna(False)
        trig_idx = g.index[trig_mask].tolist()
        if not trig_idx:
            return g

        cluster_id = 0
        i = 0
        while i < len(trig_idx):
            # first trigger in this cluster
            start_idx = trig_idx[i]
            t0 = g.loc[start_idx, "start_time"]
            if pd.isna(t0):
                i += 1
                continue
            window_end = t0 + pd.Timedelta(minutes=window_minutes)

            # collect all trigger indices within [t0, window_end]
            trigger_members = [start_idx]
            j = i + 1
            while j < len(trig_idx):
                t_next = g.loc[trig_idx[j], "start_time"]
                if pd.isna(t_next) or t_next > window_end:
                    break
                trigger_members.append(trig_idx[j])
                j += 1

            # how many triggers define this cluster?
            n_triggers = len(trigger_members)
            if n_triggers >= min_triggers_in_cluster:
                cluster_id += 1

                # include ALL rows that overlap the window:
                #   (start_time <= window_end) AND (end_time >= t0 OR end_time is NA)
                st = g["start_time"]
                et = g["end_time"] if "end_time" in g.columns else pd.Series(pd.NaT, index=g.index)
                overlap_mask = (st <= window_end) & (et.isna() | (et >= t0))
                member_idx = g.index[overlap_mask].tolist()

                # assign cluster_id to ALL members (triggers and non-triggers)
                g.loc[member_idx, "action_cluster_id"] = cluster_id
                g.loc[member_idx, "action_cluster_size"] = n_triggers

                # ranks only for triggers, ordered by start_time
                trigger_members_sorted = (
                    g.loc[trigger_members]
                    .sort_values("start_time", kind="mergesort")
                    .index.tolist()
                )
                for rank, idx in enumerate(trigger_members_sorted, start=1):
                    g.at[idx, "action_cluster_rank"] = rank

                # jump to first trigger AFTER the window
                i = j
            else:
                i += 1

        return g

    df = df.groupby(enc_keys, group_keys=False).apply(cluster_one_enc)

    # tidy dtypes
    df["action_cluster_id"]   = df["action_cluster_id"].astype("Int64")
    df["action_cluster_size"] = df["action_cluster_size"].astype("Int64")
    df["action_cluster_rank"] = df["action_cluster_rank"].astype("Int64")
    return df

def get_trigger_clusters(
    inputevents_mv_df_trigger_filtered: pd.DataFrame,
    uptitration_rel_threshold: float,
    min_abs_change: float,
    window_minutes: int,
    min_triggers_in_cluster: int
) -> pd.DataFrame:
    from . import triggers
    trig = triggers.compute_triggers_simple(
        inputevents_mv_df_trigger_filtered, 
        uptitration_rel_threshold=uptitration_rel_threshold, 
        min_abs_change=min_abs_change
    )
    trig_clustered = compute_action_clusters(
        trig, 
        window_minutes=window_minutes, 
        min_triggers_in_cluster=min_triggers_in_cluster
    )
    trig_clustered = trig_clustered.sort_values(by=['hadm_id', 'start_time'])

    return trig_clustered

def rank_action_window_combinations(
    df: pd.DataFrame,
    window_minutes: int = 20,
    drop_empty: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    For each (hadm_id, action_cluster_id):
        • t0 = earliest trigger start_time; fallback to earliest start if no trigger in cluster
        • action window = [t0, t0 + window_minutes]
        • include any med rows that overlap the window:
            start_time <= window_end  AND  (end_time is NA OR end_time >= t0)
        • build a unique, sorted tuple of med tokens for that cluster
        (NaCl/LR IV bolus gets a '[Bolus]' tag separate from drips)
    Args:
        df: DataFrame with medication data including action clusters
        window_minutes: Duration of action window in minutes
        drop_empty: Whether to drop empty combinations
    Returns:
        tuple containing:
            - combo_df: DataFrame with columns ['combo','count','percent','example_pairs']
            - cluster_combo_map: DataFrame mapping each (hadm_id, action_cluster_id) to its 'combo'
    """
    m = df.copy()
    # keep only clusters
    m = m[m["action_cluster_id"].notna()]
    if m.empty:
        return (
            pd.DataFrame(columns=["combo","count","percent","example_pairs"]),
            pd.DataFrame(columns=["hadm_id","action_cluster_id","combo"])
        )

    # times
    m["start_time"] = utils.to_utc(m["start_time"])
    m["end_time"]   = utils.to_utc(m.get("end_time"))

    gkeys = ["hadm_id", "action_cluster_id"]

    # t0 = earliest trigger; fallback to earliest start
    trig_mask = m["trigger"].fillna(False)
    t0_trigger  = m.loc[trig_mask].groupby(gkeys)["start_time"].min()
    t0_fallback = m.groupby(gkeys)["start_time"].min()
    t0_series   = t0_trigger.combine_first(t0_fallback)

    delta = pd.Timedelta(minutes=window_minutes)

    cluster_pairs   = []
    cluster_combos  = []
    for (hadm, cid), t0 in t0_series.items():
        sub = m[(m["hadm_id"] == hadm) & (m["action_cluster_id"] == cid)]
        if pd.isna(t0):
            continue
        window_end = t0 + delta

        # overlap with action window
        st = sub["start_time"]
        et = sub["end_time"]
        overlap = (st <= window_end) & (et.isna() | (et >= t0))
        win = sub.loc[overlap].copy()
        if win.empty and drop_empty:
            continue

        # make tokens (bolus NaCl/LR distinct)
        win["med_token"] = win.apply(utils.med_token, axis=1)
        meds = tuple(sorted(t for t in win["med_token"].dropna().unique() if t and t.strip()))
        if not meds and drop_empty:
            continue

        cluster_pairs.append((hadm, cid))
        cluster_combos.append(meds)

    if not cluster_combos:
        return (
            pd.DataFrame(columns=["combo","count","percent","example_pairs"]),
            pd.DataFrame(columns=["hadm_id","action_cluster_id","combo"])
        )

    # Count combos across all clusters
    counter = Counter(cluster_combos)
    total   = sum(counter.values())

    # Ranked table
    rows = []
    # collect up to a few example pairs per combo
    examples = defaultdict(list)
    for pair, combo in zip(cluster_pairs, cluster_combos):
        if len(examples[combo]) < 5:
            examples[combo].append(pair)

    for combo, cnt in counter.most_common():
        rows.append({
            "combo": combo,
            "count": cnt,
            "percent": cnt / total if total else np.nan,
            "example_pairs": examples[combo],  # list of (hadm_id, action_cluster_id)
        })
    combo_df = pd.DataFrame(rows)

    # Mapping each cluster -> its combo
    cluster_combo_map = pd.DataFrame(cluster_pairs, columns=["hadm_id","action_cluster_id"])
    cluster_combo_map["combo"] = cluster_combos

    return combo_df, cluster_combo_map
