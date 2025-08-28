import argparse
import logging
import os
import pickle
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

from create_context_tensors import create_baseline_tensors, create_context_tensors
from create_ic_targets import create_physiological_tensors

# Local imports (modules live in the same directory)
from create_med_tensors import create_med_tensors_from_parquet


def _configure_logging(verbosity: int) -> None:
    """Configure root logger.

    Args:
        verbosity: Verbosity level (0=WARNING, 1=INFO, 2=DEBUG).
    """
    level = logging.WARNING
    if verbosity == 1:
        level = logging.INFO
    elif verbosity >= 2:
        level = logging.DEBUG

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def generate_all_tensors(
    waveforms_parquet_path: str,
    med_parquet_path: str,
    output_dir: str,
    interval_seconds: int = 10,
    trajectory_duration_minutes: int = 20,
    context_duration_minutes: int = 60,
    context_interval_minutes: int = 10,
    hosp_input_dir: str | None = None,
    force_reload: bool = False,
    context_workers: int = 6,
) -> Dict[str, Any]:
    """Generate medication tensors, physiological IC/target tensors, and context tensors.

    This orchestrates the three existing generators so you can run a single command
    starting from the smoothed numerics parquet (waveforms) and the input_mv parquet
    (triggers/meds).

    Args:
        waveforms_parquet_path: Path to the smoothed numerics parquet with physio columns
            (e.g., ABP MEAN, CVP).
        med_parquet_path: Path to the input_mv parquet (e.g., mv_filtered_10min.parquet).
        output_dir: Base directory to write all outputs under.
        interval_seconds: Temporal resolution in seconds (must be consistent across steps).
        context_duration_minutes: Lookback window length before t0 for context tensors.
        context_interval_minutes: Bin size in minutes within the context window.

    Returns:
        Dictionary collecting the produced metadata and key output locations.

    Raises:
        FileNotFoundError: If any of the required input paths do not exist.
        ValueError: If downstream generators detect irrecoverable data issues.
    """
    # Validate inputs
    if not os.path.exists(waveforms_parquet_path):
        raise FileNotFoundError(
            f"Waveforms parquet not found: {waveforms_parquet_path}"
        )
    if not os.path.exists(med_parquet_path):
        raise FileNotFoundError(f"Medication parquet not found: {med_parquet_path}")

    base_output = Path(output_dir)
    med_out_dir = base_output / "med_tensors_output"
    physio_out_dir = base_output / "physio_tensors_output"
    context_out_dir = base_output / "context_tensors_output"
    baseline_out_dir = base_output / "baseline_tensors_output"

    med_out_dir.mkdir(parents=True, exist_ok=True)
    physio_out_dir.mkdir(parents=True, exist_ok=True)
    context_out_dir.mkdir(parents=True, exist_ok=True)
    baseline_out_dir.mkdir(parents=True, exist_ok=True)

    med_metadata_path = med_out_dir / "med_tensors_metadata.pkl"
    med_metadata: Dict[str, Any] | None = None
    if med_metadata_path.exists() and not force_reload:
        logging.info(
            "Step 1/4: Medication tensors already exist. Skipping (use --force-reload to regenerate)."
        )
        with open(med_metadata_path, "rb") as f:
            med_metadata = pickle.load(f)
    else:
        logging.info("Step 1/4: Generating medication tensors ...")
        med_metadata = create_med_tensors_from_parquet(
            parquet_path=med_parquet_path,
            output_dir=str(med_out_dir),
            interval_seconds=interval_seconds,
            trajectory_duration_minutes=trajectory_duration_minutes,
        )
        if not med_metadata_path.exists():
            # Fallback: persist metadata if the underlying function didn't for any reason
            with open(med_metadata_path, "wb") as f:
                pickle.dump(med_metadata, f)

    physio_metadata_path = physio_out_dir / "physio_tensors_metadata.pkl"
    physio_metadata: Dict[str, Any] | None = None
    if physio_metadata_path.exists() and not force_reload:
        logging.info(
            "Step 2/4: Physiological tensors already exist. Skipping (use --force-reload to regenerate)."
        )
        with open(physio_metadata_path, "rb") as f:
            physio_metadata = pickle.load(f)
    else:
        logging.info(
            "Step 2/4: Generating physiological IC and prediction target tensors ..."
        )
        physio_metadata = create_physiological_tensors(
            waveforms_parquet_path=str(waveforms_parquet_path),
            med_tensors_metadata_path=str(med_metadata_path),
            output_dir=str(physio_out_dir),
            interval_seconds=interval_seconds,
        )

    context_metadata_path = context_out_dir / "context_tensors_metadata.pkl"
    context_metadata: Dict[str, Any] | None = None
    if context_metadata_path.exists() and not force_reload:
        logging.info(
            "Step 3/4: Context tensors already exist. Skipping (use --force-reload to regenerate)."
        )
        with open(context_metadata_path, "rb") as f:
            context_metadata = pickle.load(f)
    else:
        logging.info("Step 3/4: Generating context tensors (physio + meds) ...")
        context_metadata = create_context_tensors(
            waveforms_parquet_path=str(waveforms_parquet_path),
            med_tensors_metadata_path=str(med_metadata_path),
            med_data_parquet_path=str(med_parquet_path),
            output_dir=str(context_out_dir),
            context_duration_minutes=context_duration_minutes,
            context_interval_minutes=context_interval_minutes,
            n_workers=context_workers,
        )

    # Optional Step 4/4: Baseline tensors
    baseline_metadata_obj: Dict[str, Any] | None = None
    baseline_metadata_file = (
        baseline_out_dir / "baseline_tensors" / "baseline_metadata.pkl"
    )
    if hosp_input_dir is None:
        logging.info(
            "Step 4/4: Baseline tensors skipped (no --hosp-input-dir provided)."
        )
    else:
        if baseline_metadata_file.exists() and not force_reload:
            logging.info(
                "Step 4/4: Baseline tensors already exist. Skipping (use --force-reload to regenerate)."
            )
        else:
            logging.info("Step 4/4: Generating baseline tensors ...")
            create_baseline_tensors(
                input_dir=hosp_input_dir,
                output_dir=str(baseline_out_dir),
                trajectory_metadata_path=str(med_metadata_path),
            )
        # Attempt to load baseline metadata if available
        if baseline_metadata_file.exists():
            try:
                with open(baseline_metadata_file, "rb") as f:
                    baseline_metadata_obj = pickle.load(f)
            except Exception:
                baseline_metadata_obj = None

    combined_metadata: Dict[str, Any] = {
        "created_at": datetime.now().isoformat(),
        "inputs": {
            "waveforms_parquet_path": str(waveforms_parquet_path),
            "med_parquet_path": str(med_parquet_path),
            "hosp_input_dir": str(hosp_input_dir)
            if hosp_input_dir is not None
            else None,
        },
        "params": {
            "interval_seconds": interval_seconds,
            "trajectory_duration_minutes": trajectory_duration_minutes,
            "context_duration_minutes": context_duration_minutes,
            "context_interval_minutes": context_interval_minutes,
            "force_reload": force_reload,
        },
        "outputs": {
            "base_output_dir": str(base_output),
            "med_tensors_output_dir": str(med_out_dir),
            "physio_tensors_output_dir": str(physio_out_dir),
            "context_tensors_output_dir": str(context_out_dir),
            "med_metadata_path": str(med_metadata_path),
            "baseline_output_dir": str(baseline_out_dir),
            "baseline_metadata_path": str(baseline_metadata_file),
        },
        "med_metadata": med_metadata,
        "physio_metadata": physio_metadata,
        "context_metadata": context_metadata,
        "baseline_metadata": baseline_metadata_obj,
    }

    # Persist a single top-level metadata artifact
    combined_metadata_path = base_output / "combined_tensors_metadata.pkl"
    with open(combined_metadata_path, "wb") as f:
        pickle.dump(combined_metadata, f)

    logging.info("All requested tensors processed.")
    logging.info("- Medication tensors: %s", med_out_dir)
    logging.info("- Physiological tensors: %s", physio_out_dir)
    logging.info("- Context tensors: %s", context_out_dir)
    logging.info("- Baseline tensors: %s", baseline_out_dir)
    logging.info("- Combined metadata: %s", combined_metadata_path)

    return combined_metadata


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate medication, physiological IC/targets, context, and optional baseline tensors "
            "from waveforms (smoothed numerics) and input_mv parquet files."
        )
    )

    parser.add_argument(
        "--waveforms-parquet",
        required=True,
        help="Path to smoothed numerics parquet containing physiologic signals.",
    )
    parser.add_argument(
        "--med-parquet",
        required=True,
        help="Path to input_mv (e.g., mv_filtered_10min.parquet).",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Base output directory where all tensors will be written.",
    )
    parser.add_argument(
        "--interval-seconds",
        type=int,
        default=10,
        help="Temporal resolution in seconds (default: 10).",
    )
    parser.add_argument(
        "--trajectory-duration-minutes",
        "--forward-minutes",
        dest="trajectory_duration_minutes",
        type=int,
        default=20,
        help="Forward window length in minutes for med/physio targets (default: 20).",
    )
    parser.add_argument(
        "--context-duration-minutes",
        type=int,
        default=60,
        help="Lookback duration in minutes for context tensors (default: 60).",
    )
    parser.add_argument(
        "--context-interval-minutes",
        type=int,
        default=10,
        help="Bin size in minutes for context tensors (default: 10).",
    )
    parser.add_argument(
        "--context-workers",
        type=int,
        default=6,
        help="Number of worker processes for context generation (default: 1).",
    )
    parser.add_argument(
        "--mimic-III-input-dir",
        type=str,
        default=None,
        help=(
            "Path to MIMIC-III input directory (contains PATIENTS.csv, ADMISSIONS.csv, TRANSFERS.csv). "
            "If provided, baseline tensors will also be generated."
        ),
    )
    parser.add_argument(
        "--force-reload",
        action="store_true",
        help="Regenerate tensors even if outputs already exist.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=1,
        help="Increase verbosity. Repeat for more detail (e.g., -vv).",
    )

    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    _configure_logging(args.verbose)

    generate_all_tensors(
        waveforms_parquet_path=args.waveforms_parquet,
        med_parquet_path=args.med_parquet,
        output_dir=args.output_dir,
        interval_seconds=args.interval_seconds,
        trajectory_duration_minutes=args.trajectory_duration_minutes,
        context_duration_minutes=args.context_duration_minutes,
        context_interval_minutes=args.context_interval_minutes,
        hosp_input_dir=args.mimic_III_input_dir,
        force_reload=args.force_reload,
        context_workers=args.context_workers,
    )


if __name__ == "__main__":
    main()
