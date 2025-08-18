import argparse
from pathlib import Path
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
import pyarrow.parquet as pq
from tqdm import tqdm
from typing import List, Set

def _check_and_move(
    source_path: Path, dest_dir: Path, required_cols: Set[str]
) -> str:
    """
    Checks if a Parquet file contains all required columns and moves it if it does.

    Args:
        source_path: The path to the source Parquet file.
        dest_dir: The directory to move the file to if it matches.
        required_cols: A set of column names that must be present.

    Returns:
        A status string: "moved" or "skipped".
    """
    try:
        schema = pq.read_schema(source_path)
        present_cols = set(name.upper() for name in schema.names)
        
        if required_cols.issubset(present_cols):
            dest_path = dest_dir / source_path.name
            shutil.move(source_path, dest_path)
            return "moved"
        else:
            return "skipped"
    except Exception as e:
        return f"error: {e}"


def main():
    ap = argparse.ArgumentParser(
        description="Filter Parquet waveform files by checking for required columns."
    )
    ap.add_argument(
        "--source-dir",
        type=Path,
        required=True,
        help="Directory containing the source Parquet files.",
    )
    ap.add_argument(
        "--dest-dir",
        type=Path,
        required=True,
        help="Directory to move the matching files to.",
    )
    ap.add_argument(
        "--required-cols",
        type=str,
        default="ABP MEAN,CVP",
        help="Comma-separated list of required column names.",
    )
    ap.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of parallel workers to use.",
    )
    args = ap.parse_args()

    # --- Setup ---
    source_dir = args.source_dir
    dest_dir = args.dest_dir
    dest_dir.mkdir(parents=True, exist_ok=True)
    
    required_cols = {col.strip().upper() for col in args.required_cols.split(",")}
    
    files_to_check = list(source_dir.glob("*.parquet"))
    
    if not files_to_check:
        print(f"No .parquet files found in {source_dir}.")
        return

    print(f"Found {len(files_to_check)} Parquet files to check in {source_dir}.")
    print(f"Required columns: {', '.join(sorted(list(required_cols)))}")
    print(f"Destination for matching files: {dest_dir}")

    # --- Parallel Processing ---
    moved_count = 0
    skipped_count = 0
    error_count = 0

    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futs = {
            executor.submit(_check_and_move, f, dest_dir, required_cols): f
            for f in files_to_check
        }
        
        with tqdm(total=len(futs), desc="Filtering waveforms") as pbar:
            for fut in as_completed(futs):
                status = fut.result()
                if status == "moved":
                    moved_count += 1
                elif status == "skipped":
                    skipped_count += 1
                else:
                    error_count += 1
                    print(f"Error processing {futs[fut].name}: {status}")
                pbar.update(1)

    # --- Summary ---
    print("\n--- Filtering Complete ---")
    print(f"Moved:   {moved_count}")
    print(f"Skipped: {skipped_count}")
    print(f"Errors:  {error_count}")
    print("--------------------------")


if __name__ == "__main__":
    main()
