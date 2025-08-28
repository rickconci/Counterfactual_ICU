import numpy as np
import pandas as pd
from config import PROCESSED_DATA_DIR
from data_utils import _ensure_hadm_column, _to_utc


def align_triggers_with_waveforms(
    combined_waveform_df, input_event_cv_df_trigger_meds, input_mv_triggers
):
    # waveform ranges per hadm_id
    wf = combined_waveform_df.copy()
    wf["absolute_timestamp"] = _to_utc(wf["absolute_timestamp"])

    wf_ranges = wf.groupby("hadm_id", as_index=False)["absolute_timestamp"].agg(
        first="min", last="max"
    )

    # CV
    cv = _ensure_hadm_column(input_event_cv_df_trigger_meds.copy())
    cv["charttime"] = _to_utc(cv["charttime"])

    cv_aligned = cv.merge(wf_ranges, on="hadm_id", how="inner")
    cv_aligned = cv_aligned[
        (cv_aligned["charttime"] >= cv_aligned["first"])
        & (cv_aligned["charttime"] <= cv_aligned["last"])
    ]

    # MV
    mv = _ensure_hadm_column(input_mv_triggers.copy())
    mv["start_time"] = _to_utc(mv["start_time"])

    mv_aligned = mv.merge(wf_ranges, on="hadm_id", how="inner")
    mv_aligned = mv_aligned[
        (mv_aligned["start_time"] >= mv_aligned["first"])
        & (mv_aligned["start_time"] <= mv_aligned["last"])
    ]

    summary = {
        "CV_before": int(cv_aligned.shape[0]),
        "CV_after": int(cv_aligned.shape[0]),
        "MV_before": int(mv_aligned.shape[0]),
        "MV_after": int(mv_aligned.shape[0]),
    }
    return cv_aligned, mv_aligned, summary


def coerce_hadm_id_to_int(df: pd.DataFrame, col="hadm_id") -> pd.DataFrame:
    """
    Make hadm_id reliably integer-typed:
    - parse to numeric
    - drop non-integer-like values (e.g., 123.45)
    - cast to pandas nullable Int64 (preserves NA)
    """
    df = df.copy()
    s = pd.to_numeric(df[col], errors="coerce")

    # keep only values that are whole numbers (e.g., 123.0 -> 123)
    is_whole = s.dropna().mod(1).eq(0)
    # set non-whole to NA
    s.loc[~s.index.isin(is_whole.index[is_whole])] = pd.NA

    df[col] = s.astype("Int64")  # nullable integer
    return df


import pandas as pd


def filter_mv_triggers_by_waveform_groupwise(
    mv_df: pd.DataFrame,
    wf_df: pd.DataFrame,
    hadm_col="hadm_id",
    mv_time_col="start_time",
    wf_time_col="absolute_timestamp",
    required_signals=("ABP MEAN", "CVP"),
    tolerance="10min",
):
    # copy + normalize dtypes
    mv = mv_df.copy()
    wf = wf_df.copy()

    mv[hadm_col] = pd.to_numeric(mv[hadm_col], errors="coerce").astype("Int64")
    wf[hadm_col] = pd.to_numeric(wf[hadm_col], errors="coerce").astype("Int64")

    mv[mv_time_col] = pd.to_datetime(mv[mv_time_col], errors="coerce", utc=True)
    wf[wf_time_col] = pd.to_datetime(wf[wf_time_col], errors="coerce", utc=True)

    # keep waveform rows where at least one of the required signals exists
    mask_valid = False
    for sig in required_signals:
        if sig in wf.columns:
            mask_valid = mask_valid | wf[sig].notna()
    wf_valid = wf.loc[mask_valid].copy()

    # drop rows missing keys
    mv = mv.dropna(subset=[hadm_col, mv_time_col])
    wf_valid = wf_valid.dropna(subset=[hadm_col, wf_time_col])

    tol = pd.Timedelta(tolerance)

    results = []
    # operate per-hadm to avoid global sorting pitfalls
    common_ids = pd.Index(mv[hadm_col].dropna().unique()).intersection(
        wf_valid[hadm_col].dropna().unique()
    )

    for hid in common_ids:
        mv_g = (
            mv.loc[mv[hadm_col] == hid].sort_values(mv_time_col).reset_index(drop=True)
        )
        wf_g = (
            wf_valid.loc[wf_valid[hadm_col] == hid]
            .sort_values(wf_time_col)
            .reset_index(drop=True)
        )

        if mv_g.empty or wf_g.empty:
            continue

        matched = pd.merge_asof(
            mv_g,
            wf_g[[wf_time_col]],
            left_on=mv_time_col,
            right_on=wf_time_col,
            direction="nearest",
            tolerance=tol,
            allow_exact_matches=True,
        )

        matched = matched[matched[wf_time_col].notna()].copy()
        matched["wf_time_delta_s"] = (
            matched[wf_time_col] - matched[mv_time_col]
        ).dt.total_seconds()
        results.append(matched)

    if results:
        return pd.concat(results, ignore_index=True)
    else:
        # return empty frame with same columns if no matches
        return mv.iloc[0:0].assign(
            **{wf_time_col: pd.NaT, "wf_time_delta_s": pd.Series(dtype="float")}
        )


# Calculate min and max for each group instead of mean and std


# Apply min-max normalization for each group
def normalize_by_group(row):
    item = row["item_label"]
    value = row["rate/weight"]

    if pd.isna(value) or item not in stats.index:
        return np.nan

    min_val = stats.loc[item, "min"]
    max_val = stats.loc[item, "max"]

    # Avoid division by zero (when min equals max)
    if max_val == min_val:
        return 0

    return (value - min_val) / (max_val - min_val)


def clean_smooth_waveforms(df_clean, save_dir=None):
    abp_valid = df_clean["ABP MEAN"].dropna()
    cvp_valid = df_clean["CVP"].dropna()


def main(
    df_clean,
    input_event_cv_df_trigger_meds,
    input_mv_triggers,
    args=None,
    save_dir=None,
):
    df_clean = coerce_hadm_id_to_int(df_clean, "hadm_id")
    input_event_cv_df_trigger_meds = coerce_hadm_id_to_int(
        input_event_cv_df_trigger_meds, "hadm_id"
    )
    input_mv_triggers = coerce_hadm_id_to_int(input_mv_triggers, "hadm_id")

    cv_aligned, mv_aligned, summary = align_triggers_with_waveforms(
        df_clean, input_event_cv_df_trigger_meds, input_mv_triggers
    )

    mv_filtered_10min = filter_mv_triggers_by_waveform_groupwise(
        mv_aligned,
        df_clean,
        required_signals=("ABP MEAN", "CVP"),
        tolerance=args.tolerance,
    )
    print(f"MV triggers before: {len(mv_aligned)}  after: {len(mv_filtered_10min)}")

    mv_filtered_10min["rate/weight_normalized"] = mv_filtered_10min.apply(
        normalize_by_group, axis=1
    )

    if save_dir is not None:
        mv_filtered_10min.to_parquet(save_dir / "mv_filtered_10min.parquet")

    hadm_trigger_counts = (
        mv_filtered_10min[mv_filtered_10min["trigger"] == True]
        .groupby("hadm_id")
        .size()
        .reset_index(name="n_triggers")
        .sort_values("n_triggers", ascending=False)
    )
    if save_dir is not None:
        hadm_trigger_counts.to_csv(save_dir / "hadm_trigger_counts.csv", index=False)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--tolerance", type=str, default="10min")
    args = parser.parse_args()

    df_clean = pd.read_parquet(PROCESSED_DATA_DIR / "combined_waveforms.parquet")
    input_event_cv_df_trigger_meds = pd.read_csv(
        PROCESSED_DATA_DIR / "input_event_cv_df_trigger_meds.csv"
    )
    input_mv_triggers = pd.read_csv(PROCESSED_DATA_DIR / "input_mv_triggers.csv")

    main(df_clean, input_event_cv_df_trigger_meds, input_mv_triggers, args)
