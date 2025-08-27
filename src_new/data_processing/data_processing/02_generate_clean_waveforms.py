import os
import argparse
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm.auto import tqdm
from data_config import WAVEFORM_DIR, PROCESSED_DATA_DIR, TARGET_COLUMNS, TARGET_SCHEMA
from data_utils import _enforce_pandas_dtypes



def build_combined_waveform_df_streaming(
    waveform_dir_path: str,
    prefix: str = "full_waveform_",
    suffix: str = ".parquet",
    cols_to_keep_pretty = (
        'hadm_id','record_name','absolute_timestamp',
        'ABP MEAN','NBP MEAN','CVP','HR','RESP',
        'record_start_time','record_end_time','icu_admission_time','time_seconds'
    ),
    require_both_abp_cvp: bool = True,
    output_path: str = None,          # if given, writes a fresh parquet
    return_dataframe: bool = True,
    max_rows_in_memory: int = 2_000_000,
    progress: bool = True,
    verbose: bool = True,
):
    files = sorted(
        f for f in os.listdir(waveform_dir_path)
        if f.startswith(prefix) and f.endswith(suffix)
    )

    cols_to_keep_lower = [c.lower() for c in cols_to_keep_pretty]
    lower_to_pretty = dict(zip(cols_to_keep_lower, cols_to_keep_pretty))

    need_abp = "abp mean"
    need_cvp = "cvp"

    # fresh output
    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        if os.path.exists(output_path):
            os.remove(output_path)

    def _lower_map_from_schema(pf: pq.ParquetFile):
        names = [col.name for col in pf.schema_arrow]
        m = {}
        for n in names:
            key = n.strip().lower()
            if key not in m:
                m[key] = n
        return m

    skipped = []
    filled_missing = {}
    frames = []
    frames_row_count = 0
    writer = None
    combined_df = None
    total_written_rows = 0

    it = tqdm(files, desc="Streaming waveforms") if progress else files

    for fname in it:
        fpath = os.path.join(waveform_dir_path, fname)
        try:
            pf = pq.ParquetFile(fpath)
        except Exception as e:
            skipped.append((fname, f"read_error(schema): {e}"))
            continue

        lowmap = _lower_map_from_schema(pf)
        has_abp = need_abp in lowmap
        has_cvp = need_cvp in lowmap

        if require_both_abp_cvp:
            if not (has_abp and has_cvp):
                skipped.append((fname, "requires both 'ABP MEAN' and 'CVP'"))
                continue
        else:
            if not (has_abp or has_cvp):
                skipped.append((fname, "missing both 'ABP MEAN' and 'CVP'"))
                continue

        columns_to_read, missing_lower = [], []
        for lc in cols_to_keep_lower:
            if lc in lowmap:
                columns_to_read.append(lowmap[lc])
            else:
                missing_lower.append(lc)

        rows_written_this_file = 0

        for rg in range(pf.num_row_groups):
            try:
                table = pf.read_row_group(rg, columns=columns_to_read)
            except Exception as e:
                skipped.append((fname, f"read_error(row_group={rg}): {e}"))
                continue

            df = table.to_pandas()
            df.columns = df.columns.str.strip().str.lower()

            if missing_lower:
                for c in missing_lower:
                    df[c] = pd.NA
                if fname not in filled_missing and missing_lower:
                    filled_missing[fname] = list(missing_lower)

            if ("absolute_timestamp" not in df.columns) or df["absolute_timestamp"].isna().all():
                if ("record_start_time" in df.columns) and ("time_seconds" in df.columns):
                    rstart = pd.to_datetime(df["record_start_time"], errors="coerce", utc=True)
                    tsecs  = pd.to_numeric(df["time_seconds"], errors="coerce")
                    df["absolute_timestamp"] = rstart + pd.to_timedelta(tsecs, unit="s")
                else:
                    df["absolute_timestamp"] = pd.NaT

            if "hadm_id" in df.columns:
                s = pd.to_numeric(df["hadm_id"], errors="coerce").astype("float64")
                s_np = s.to_numpy()
                is_whole = np.isfinite(s_np) & np.isclose(s_np, np.floor(s_np), rtol=0, atol=1e-9)
                s_np[~is_whole] = np.nan
                df["hadm_id"] = pd.Series(s_np, index=df.index).astype("Int64")

            df = df[cols_to_keep_lower].rename(columns=lower_to_pretty)

            if len(df) == 0:
                continue

            df = df[TARGET_COLUMNS]
            df = _enforce_pandas_dtypes(df)

            tbl = pa.Table.from_pandas(df, preserve_index=False)

            # >>> WRITE ONLY IF output_path IS PROVIDED
            if output_path:  # <<< add guard
                if writer is None:
                    writer = pq.ParquetWriter(output_path, TARGET_SCHEMA, compression="snappy")
                if not tbl.schema.equals(TARGET_SCHEMA):
                    tbl = tbl.cast(TARGET_SCHEMA, safe=False)
                writer.write_table(tbl)

            if return_dataframe:
                frames.append(df)
                frames_row_count += len(df)
                if frames_row_count >= max_rows_in_memory:
                    combined_df = (
                        pd.concat([combined_df, *frames], ignore_index=True)
                        if combined_df is not None else pd.concat(frames, ignore_index=True)
                    )
                    frames.clear()
                    frames_row_count = 0

            rows_written_this_file += len(df)

        total_written_rows += rows_written_this_file
        if verbose:
            print(f"{fname}: wrote {rows_written_this_file} rows")

    if writer is not None:
        writer.close()

    if return_dataframe:
        if frames:
            combined_df = (pd.concat([combined_df, *frames], ignore_index=True)
                           if combined_df is not None else pd.concat(frames, ignore_index=True))
        if combined_df is None:
            combined_df = pd.DataFrame(columns=cols_to_keep_pretty)

    if verbose and output_path:
        print(f"TOTAL rows written to {output_path}: {total_written_rows}")

    # (Optional) return a small stats dict too:
    # stats = {"total_files": len(files), "skipped": len(skipped), "rows_written": total_written_rows}
    # return combined_df, skipped, filled_missing, stats

    return combined_df, skipped, filled_missing


def clean_waveform_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Out-of-range -> NaN
    if "CVP" in df:
        df.loc[(df["CVP"] < 0) | (df["CVP"] > 40), "CVP"] = np.nan
    if "HR" in df:
        df.loc[(df["HR"] < 40) | (df["HR"] > 180), "HR"] = np.nan
    if "RESP" in df:
        df.loc[(df["RESP"] < 8) | (df["RESP"] > 40), "RESP"] = np.nan
    if "ABP MEAN" in df:
        df.loc[(df["ABP MEAN"] > 190 )| (df["ABP MEAN"] < 40) , "ABP MEAN"] = np.nan
    

    keep_cols = [c for c in ["ABP MEAN", "CVP"] if c in df.columns]
    if keep_cols:
        df = df.dropna(subset=keep_cols, how="all")
    return df

def clean_parquet_in_chunks(
    input_path: str,
    output_clean_path: str,
    chunk_rows: int = 10_000_000,
    progress: bool = True,
) -> dict:
    """
    Stream 'input_path' Parquet, clean in chunks (~chunk_rows), write to 'output_clean_path'.
    Returns a stats dict.
    """
    pf = pq.ParquetFile(input_path)

    # ensure fresh output
    os.makedirs(os.path.dirname(output_clean_path), exist_ok=True)
    if os.path.exists(output_clean_path):
        os.remove(output_clean_path)

    writer = None
    batch_frames, batch_rows = [], 0
    total_in, total_out = 0, 0

    rgs = range(pf.num_row_groups)
    if progress:
        rgs = tqdm(rgs, desc="Cleaning in chunks")

    for rg in rgs:
        table = pf.read_row_group(rg)  # read all columns; it's already column-pruned upstream
        df = table.to_pandas()
        total_in += len(df)

        # clean this RG
        df = clean_waveform_df(df)

        if df.empty:
            continue

        # enforce column order/dtypes, with safe fallback if some columns are missing
        # (align columns to TARGET_COLUMNS; any missing will be added as NA)
        for c in TARGET_COLUMNS:
            if c not in df.columns:
                df[c] = pd.NA
        df = df[TARGET_COLUMNS]
        df = _enforce_pandas_dtypes(df)

        batch_frames.append(df)
        batch_rows += len(df)

        # flush if batch large enough
        if batch_rows >= chunk_rows:
            chunk = pd.concat(batch_frames, ignore_index=True)
            batch_frames, batch_rows = [], 0

            tbl = pa.Table.from_pandas(chunk, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(output_clean_path, TARGET_SCHEMA, compression="snappy")
            if not tbl.schema.equals(TARGET_SCHEMA):
                tbl = tbl.cast(TARGET_SCHEMA, safe=False)
            writer.write_table(tbl)
            total_out += len(chunk)

    # flush any remainder
    if batch_frames:
        chunk = pd.concat(batch_frames, ignore_index=True)
        tbl = pa.Table.from_pandas(chunk, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(output_clean_path, TARGET_SCHEMA, compression="snappy")
        if not tbl.schema.equals(TARGET_SCHEMA):
            tbl = tbl.cast(TARGET_SCHEMA, safe=False)
        writer.write_table(tbl)
        total_out += len(chunk)

    if writer is not None:
        writer.close()

    return {
        "row_groups": pf.num_row_groups,
        "rows_in": total_in,
        "rows_out": total_out,
        "output_path": output_clean_path,
    }


## Smoothing functions



def _smooth_1d_nanaware(arr: np.ndarray, neighbors: int,
                        keep_nan_center: bool = True,
                        min_valid: int = 1) -> np.ndarray:
    if neighbors <= 0 or arr.size == 0:
        return arr.astype(float, copy=True)
    win = 2*neighbors + 1
    s = pd.Series(arr, dtype="float64")
    sm = s.rolling(win, center=True, min_periods=min_valid).mean()
    if keep_nan_center:
        sm[s.isna()] = np.nan
    return sm.to_numpy()

def _clip_inplace(g: pd.DataFrame):
    if "ABP MEAN" in g:
        g["ABP MEAN"] = pd.to_numeric(g["ABP MEAN"], errors="coerce").clip(lower=40, upper=180)
    if "CVP" in g:
        g["CVP"] = pd.to_numeric(g["CVP"], errors="coerce").clip(lower=0, upper=40)
    return g

def _zero_center_cols(g: pd.DataFrame, cols, suffix="_zc"):
    for c in cols:
        if c in g:
            mu = g[c].mean(skipna=True)
            g[f"{c}{suffix}"] = g[c] - mu
    return g

def _zscore_cols(g: pd.DataFrame, cols, suffix="_zn"):
    for c in cols:
        if c in g:
            mu = g[c].mean(skipna=True)
            sd = g[c].std(skipna=True)
            g[f"{c}{suffix}"] = (g[c] - mu) / sd if (pd.notna(sd) and sd > 0) else np.nan
    return g

def _smooth_cols_multi(g: pd.DataFrame, cols, neighbors, source_suffixes, out_suffix):
    """
    Smooth multiple source variants (e.g., raw/zc/zn) in one go.
    For each c in cols and each src in source_suffixes, create c{src}{out_suffix}.
    """
    for c in cols:
        for src in source_suffixes:
            base = f"{c}{src}" if src else c
            if base in g:
                g[f"{base}{out_suffix}"] = _smooth_1d_nanaware(
                    pd.to_numeric(g[base], errors="coerce").to_numpy(),
                    neighbors=neighbors, keep_nan_center=True, min_valid=1
                )
    return g

# goes thru wf database, choose z_score or z_center. Keep signal, abs timestamp, zero_center false, z-score false. 
def run_waveform_pipeline(
    df: pd.DataFrame,
    signals=("ABP MEAN","CVP","HR","RESP"),
    time_col="absolute_timestamp",
    group_cols=("hadm_id","record_name"),
    *,
    do_zero_center: bool = True,
    do_zscore: bool = True,
    smooth_neighbors: int = 120,
    smooth_variants=("zc","zn"),   # <- choose any of {"raw","zc","zn"}; e.g. ("zc","zn")
    out_suffix: str = "_ma120",
    flush_every_rows: int = 2_000_000,
):
    """
    Memory-friendly generator: per (hadm_id, record_name) it clips ABP/CVP,
    optionally creates _zc and/or _zn, then smooths any of the requested variants
    (e.g., zc and zn), yielding chunks so you can concat or write to disk.
    """
    need_time = time_col in df.columns
    out_frames, acc_rows = [], 0
    sort_keys = list(group_cols) + ([time_col] if need_time else [])
    df_sorted = df.sort_values(sort_keys)

    # map variant tokens to suffixes
    var2suf = {"raw": "", "zc": "_zc", "zn": "_zn"}
    src_suffixes = [var2suf[v] for v in smooth_variants]

    for _, g in df_sorted.groupby(list(group_cols), sort=False, dropna=False):
        # ensure numeric for signals
        for c in signals:
            if c in g:
                g[c] = pd.to_numeric(g[c], errors="coerce")

        # 1) clip only ABP MEAN and CVP
        g = _clip_inplace(g)

        # 2) normalization variants (per group)
        if do_zero_center:
            g = _zero_center_cols(g, signals, suffix="_zc")
        if do_zscore:
            g = _zscore_cols(g, signals, suffix="_zn")

        # 3) smoothing for any requested variants
        if smooth_neighbors and smooth_neighbors > 0 and src_suffixes:
            g = _smooth_cols_multi(g, signals, neighbors=smooth_neighbors,
                                   source_suffixes=src_suffixes, out_suffix=out_suffix)

        out_frames.append(g)
        acc_rows += len(g)
        if acc_rows >= flush_every_rows:
            yield pd.concat(out_frames, ignore_index=True)
            out_frames.clear()
            acc_rows = 0

    if out_frames:
        yield pd.concat(out_frames, ignore_index=True)






if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combine, clean, smooth ICU waveforms")
    parser.add_argument("--waveform_dir", type=str, default=str(WAVEFORM_DIR), help="Directory with full_waveform_*.parquet files")
    parser.add_argument("--neighbors", type=int, default=120, help="Smoothing neighbors each side (window=2*n+1)")
    parser.add_argument("--require_both", action="store_true", help="Require both ABP MEAN and CVP to include a file")
    args = parser.parse_args()

    waveform_root = args.waveform_dir
    require_both = bool(args.require_both)
    neighbors = int(args.neighbors)

    # 1) Combine waveforms that have ABP MEAN or CVP and add absolute_timestamp
    combined_path = PROCESSED_DATA_DIR / "combined_waveforms.parquet"
    _, _, _ = build_combined_waveform_df_streaming(
        waveform_root,
        require_both_abp_cvp=require_both,
        output_path=str(combined_path),
        return_dataframe=False,
        verbose=False,
    )
    print("Saved combined:", combined_path, "exists?", os.path.exists(combined_path))

    # 2) Clean out-of-range values -> NaN and drop rows with both ABP MEAN and CVP missing
    cleaned_path = PROCESSED_DATA_DIR / "combined_waveforms_cleaned.parquet"
    stats = clean_parquet_in_chunks(str(combined_path), str(cleaned_path), chunk_rows=10_000_000)
    print(stats)

    # 3) Smooth and create normalized variants (zc/zn), then write final parquet
    df_clean = pd.read_parquet(cleaned_path)
    final_path = PROCESSED_DATA_DIR / "combined_waveforms_cleaned_smoothed.parquet"
    if os.path.exists(final_path):
        os.remove(final_path)

    writer = None
    for chunk in run_waveform_pipeline(
        df_clean,
        signals=("ABP MEAN", "CVP", "HR", "RESP"),
        do_zero_center=True,
        do_zscore=True,
        smooth_neighbors=neighbors,
        smooth_variants=("zc", "zn"),
        out_suffix="_ma120",
        flush_every_rows=1_000_000,
    ):
        tbl = pa.Table.from_pandas(chunk, preserve_index=False)
        if writer is None:
            writer = pq.ParquetWriter(str(final_path), tbl.schema, compression="snappy")
        writer.write_table(tbl)
    if writer is not None:
        writer.close()

    print("Saved final:", final_path, "exists?", os.path.exists(final_path))


        



