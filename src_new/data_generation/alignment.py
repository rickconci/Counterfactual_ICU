import pandas as pd
import numpy as np

def compute_action_times_first_trigger(trig_clustered: pd.DataFrame) -> pd.DataFrame:
    """
    Compute action times from first trigger in each cluster.
    
    Args:
        trig_clustered: DataFrame with trigger and clustering information
        
    Returns:
        DataFrame with first trigger times per action cluster
    """
    df = trig_clustered.copy()
    df = df[df["trigger"].fillna(False) & df["action_cluster_id"].notna()].copy()
    df["start_time"] = pd.to_datetime(df["start_time"], errors="coerce", utc=True)
    df = df.sort_values(["hadm_id","action_cluster_id","start_time"])
    firsts = (df.groupby(["hadm_id","action_cluster_id"], as_index=False)
                .agg(action_time=("start_time","first"),
                        subject_id=("subject_id","first"),
                        item_label=("item_label","first"),
                        action_cluster_size=("action_cluster_size","first"),
                        action_cluster_rank=("action_cluster_rank","min")))
    return firsts

def align_actions_to_consolidated_segments(
    trig_clustered: pd.DataFrame,
    consolidated_meta: pd.DataFrame,
    window_minutes: int = 10,
) -> pd.DataFrame:
    """
    Join actions to consolidated segments where
    seg_start_time <= window_end and seg_end_time >= window_start
    and compute overlap + offsets relative to the consolidated segment.
    
    Args:
        trig_clustered: DataFrame with trigger and clustering information
        consolidated_meta: DataFrame with consolidated waveform segments
        window_minutes: Time window around action in minutes
        
    Returns:
        DataFrame with actions aligned to consolidated segments
    """
    acts = compute_action_times_first_trigger(trig_clustered).copy()
    delta = pd.to_timedelta(window_minutes, unit="m")
    acts["window_start"] = acts["action_time"] - delta
    acts["window_end"]   = acts["action_time"] + delta

    if consolidated_meta.empty or acts.empty:
        return pd.DataFrame(columns=[
            "hadm_id","action_cluster_id","segment_id","action_time",
            "window_start","window_end","seg_start_time","seg_end_time",
            "overlap_start","overlap_end","offset_start_seconds","offset_end_seconds",
            "overlap_seconds","component_count","record_names","file_paths",
            "subject_id","item_label","action_cluster_size","action_cluster_rank"
        ])

    merged = acts.merge(consolidated_meta, on="hadm_id", how="inner")

    mask = (merged["seg_start_time"] <= merged["window_end"]) & \
            (merged["seg_end_time"]   >= merged["window_start"])
    hit = merged.loc[mask].copy()
    if hit.empty:
        return hit

    hit["overlap_start"] = hit[["seg_start_time","window_start"]].max(axis=1)
    hit["overlap_end"]   = hit[["seg_end_time","window_end"]].min(axis=1)
    hit = hit.loc[hit["overlap_end"] > hit["overlap_start"]].copy()

    s_seg = hit["seg_start_time"].view("int64") // 10**9
    o_st  = hit["overlap_start"].view("int64") // 10**9
    o_en  = hit["overlap_end"].view("int64") // 10**9

    hit["offset_start_seconds"] = (o_st - s_seg).clip(lower=0)
    hit["offset_end_seconds"]   = (o_en - s_seg).clip(lower=0)
    hit["overlap_seconds"]      = hit["offset_end_seconds"] - hit["offset_start_seconds"]

    cols = [
        "hadm_id","action_cluster_id","subject_id","item_label",
        "action_time","window_start","window_end",
        "segment_id","seg_start_time","seg_end_time",
        "overlap_start","overlap_end","offset_start_seconds","offset_end_seconds","overlap_seconds",
        "component_count","record_names","file_paths","action_cluster_size","action_cluster_rank",
    ]
    cols = [c for c in cols if c in hit.columns]
    hit = hit[cols].sort_values(["hadm_id","action_cluster_id","segment_id"]).reset_index(drop=True)
    return hit

def pick_best_segment_per_action(aligned_cons: pd.DataFrame) -> pd.DataFrame:
    """
    Pick the single best consolidated segment per action based on overlap duration.
    
    Args:
        aligned_cons: DataFrame with actions aligned to consolidated segments
        
    Returns:
        DataFrame with best segment per action (highest overlap_seconds)
    """
    if aligned_cons.empty:
        return aligned_cons
    best = (aligned_cons.sort_values(["hadm_id","action_cluster_id","overlap_seconds"],
                                        ascending=[True, True, False])
                        .groupby(["hadm_id","action_cluster_id"], as_index=False)
                        .head(1)
                        .reset_index(drop=True))
    return best
