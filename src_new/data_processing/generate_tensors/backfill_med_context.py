import argparse
import os
from pathlib import Path
from typing import Tuple

import numpy as np
import torch


def compute_med_context(values: torch.Tensor, mask: torch.Tensor, interval_seconds: int) -> torch.Tensor:
    """Compute [last_rate, recency] per time step for each medication.

    Args:
        values: [T, M] float tensor of medication rates
        mask: [T, M] float/bool tensor indicating presence
        interval_seconds: grid spacing in seconds

    Returns:
        med_context: [T, 2*M] tensor with interleaved [last_rate, recency]
    """
    T, M = values.shape

    # Use numpy for convenience; convert back to torch at end
    vals = values.cpu().numpy().astype(np.float32)
    msk = (mask.cpu().numpy() > 0).astype(np.int64)

    time_idx = np.arange(T, dtype=np.int64)
    valid_idxs = np.where(msk > 0, time_idx[None, :].T + 1, 0)  # [T, M]
    last_idx_plus1 = np.maximum.accumulate(valid_idxs, axis=0)
    has_valid = last_idx_plus1 > 0
    last_idx = np.clip(last_idx_plus1 - 1, 0, T - 1)

    rows = last_idx
    cols = np.tile(np.arange(M)[None, :], (T, 1))
    last_rates = vals[rows, cols]
    t_grid = (time_idx[:, None] * interval_seconds).astype(np.float32)
    last_times = (last_idx.astype(np.float32) * float(interval_seconds))
    time_since = t_grid - last_times
    recency = np.clip(time_since / 1200.0 - 1.0 / 1200.0, a_min=0.0, a_max=None)

    last_rates = np.where(has_valid, last_rates, 0.0).astype(np.float32)
    recency = np.where(has_valid, recency, 1.0).astype(np.float32)
    med_context = np.stack([last_rates, recency], axis=-1).reshape(T, 2 * M)
    return torch.from_numpy(med_context).float()


def backfill_directory(med_dir: Path) -> Tuple[int, int]:
    """Backfill med_context into existing med_tensor_*.pt files in-place.

    Returns:
        processed, skipped counts
    """
    processed = 0
    skipped = 0
    for file in sorted(med_dir.glob("med_tensor_*.pt")):
        loaded = torch.load(file)
        if len(loaded) == 6:
            skipped += 1
            continue
        values, mask, time_seconds, time_hours, n_intervals = loaded
        try:
            med_context = compute_med_context(values, mask, int(time_seconds[1].item() - time_seconds[0].item()))
            torch.save((values, mask, time_seconds, time_hours, n_intervals, med_context), file)
            processed += 1
        except Exception as e:
            print(f"[WARN] Failed to backfill {file}: {e}")
    return processed, skipped


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Backfill med_context into med_tensors")
    parser.add_argument("--med_tensors_dir", type=str, required=True, help="Path to med_tensors_output directory")
    args = parser.parse_args()
    med_dir = Path(args.med_tensors_dir)
    if not med_dir.exists():
        raise FileNotFoundError(f"Directory not found: {med_dir}")
    processed, skipped = backfill_directory(med_dir)
    print(f"Backfill complete. Updated: {processed}, Already up-to-date: {skipped}")


