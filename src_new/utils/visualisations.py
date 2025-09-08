#!/usr/bin/env python3
"""
Complete script to create medication rate distribution visualization
comparing flat vs upward vs downward trajectory trends using box plots.
Uses ALL data (no train/test/val splits).
"""

import numpy as np
import torch
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy import stats

# Import your dataloader classes
import sys

sys.path.append('..')
from models.dataloaders.MIMIC_data import MIMICDataset, MIMICDataModule

# Data root path
DATA_ROOT = "../../data/mimic_3_data/processed_data"
ICU_STAYS_PATH = ""  # Will be set automatically if needed


def classify_trajectory_trend(p_out_values, p_out_mask,
                              min_valid_points=5,
                              up_thresholds=[15.0, 5.0],
                              down_thresholds=[-15.0, -5.0]):
    """
    Classify trajectory as 'up', 'down', or 'flat'.

    Up: diff > 15 for channel 0 OR > 5 for channel 1
    Down: diff < -15 for channel 0 OR < -5 for channel 1
    Flat: everything else
    """
    for channel in range(p_out_values.shape[1]):
        channel_values = p_out_values[:, channel]
        channel_mask = p_out_mask[:, channel]

        valid_values = channel_values[channel_mask > 0]

        if len(valid_values) < min_valid_points:
            continue

        # Calculate difference between first and last valid values
        first_value = valid_values[0].item()
        last_value = valid_values[-1].item()
        diff = last_value - first_value

        # Check thresholds for this channel
        up_threshold = up_thresholds[min(channel, len(up_thresholds) - 1)]
        down_threshold = down_thresholds[min(channel, len(down_thresholds) - 1)]

        if diff > up_threshold:
            return 'up'
        elif diff < down_threshold:
            return 'down'

    return 'flat'


def get_all_data_loader(data_root, icu_stays_path, batch_size=16, max_samples=None):
    """
    Create a dataloader that uses ALL the data (no train/test/val split).
    """
    # Create a dataset with train_ratio=1.0 to get all data in "train" split
    dataset = MIMICDataset(
        data_root=data_root,
        icu_stays_path=icu_stays_path,
        split="train",
        train_ratio=1.0,  # Put all data in train split
        val_ratio=0.0,
        filter_flat_trajectories=False,  # We want both types
        max_samples=max_samples
    )

    # Create datamodule just to get the collate function - DON'T call setup()
    datamodule = MIMICDataModule(
        data_root=data_root,
        icu_stays_path=icu_stays_path,
        batch_size=batch_size,
        filter_flat_trajectories=False,
        num_workers=4
    )

    # Create dataloader manually
    from torch.utils.data import DataLoader

    kwargs = datamodule._loader_common_kwargs()
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,  # No need to shuffle for visualization
        collate_fn=datamodule.collate_fn,
        **{k: v for k, v in kwargs.items() if v is not None},
    )

    return loader, dataset


def plot_medication_distributions(data_root, icu_stays_path, max_batches=None,
                                  batch_size=16, max_samples=None, figsize=(14, 12)):
    """
    Create a box plot showing medication rate distributions for flat vs up vs down trajectories.
    Uses ALL available data.
    """

    print("Loading ALL data (no train/test/val splits)...")

    # Hardcoded medication mapping from metadata (excluding Hetastarch)
    medication_mapping = {
        'Albumin 25%': 0,
        'Albumin 5%': 1,
        'Dopamine': 2,
        'Epinephrine': 3,
        'LR': 5,
        'Milrinone': 6,
        'NaCl 0.9%': 7,
        'Norepinephrine': 8,
        'Packed Red Blood Cells': 9,
        'Phenylephrine': 10,
        'Vasopressin': 11
    }

    # Create ordered medication names list from the mapping
    idx_to_med = {v: k for k, v in medication_mapping.items()}
    medication_names = [idx_to_med[i] for i in sorted(idx_to_med.keys())]
    print(f"Using medication names: {medication_names}")

    # Get dataloader with all data
    loader, dataset = get_all_data_loader(
        data_root=data_root,
        icu_stays_path=icu_stays_path,
        batch_size=batch_size,
        max_samples=max_samples
    )

    print(f"Total dataset size: {len(dataset)} trajectories")

    # Debug: Check a sample to see data dimensions
    sample = dataset[0]
    print(f"Sample med_tensors shape: {sample['med_tensors'].shape}")
    print(f"Sample med_mask shape: {sample['med_mask'].shape}")
    print(f"Sample Y shape: {sample['Y'].shape}")

    # Determine how many batches to process
    total_batches = len(loader)
    if max_batches is None:
        max_batches = total_batches
    else:
        max_batches = min(max_batches, total_batches)

    print(f"Processing {max_batches}/{total_batches} batches...")

    flat_rates = []
    up_rates = []
    down_rates = []
    total_processed = 0
    total_valid_timepoints = 0
    debug_first_batch = True

    # Trajectory counters
    flat_count = 0
    up_count = 0
    down_count = 0

    for batch_idx, batch in enumerate(tqdm(loader, desc="Processing all data")):
        if batch_idx >= max_batches:
            break

        # Unpack batch according to collate_fn return order
        Y = batch[6]  # Target values [B, T_fwd, 2]
        Y_mask = batch[7]  # Target mask [B, T_fwd, 2]
        med_tensors = batch[12]  # Med tensors [B, T_fwd, 2*M]
        med_mask = batch[10]  # Med mask [B, T_fwd, M]

        batch_size_actual = Y.shape[0]

        # Debug first batch
        if debug_first_batch:
            print(f"\nDEBUG - First batch:")
            print(f"  Y shape: {Y.shape}")
            print(f"  Y_mask shape: {Y_mask.shape}")
            print(f"  med_tensors shape: {med_tensors.shape}")
            print(f"  med_mask shape: {med_mask.shape}")
            print(f"  Rate columns will be: {list(range(0, med_tensors.shape[-1], 5))}")
            print(f"  Number of medications: {len(list(range(0, med_tensors.shape[-1], 5)))}")
            debug_first_batch = False

        for i in range(batch_size_actual):
            total_processed += 1

            # Classify trajectory trend
            trend = classify_trajectory_trend(Y[i], Y_mask[i])

            # Get medication rates (every 5th column: 0, 5, 10, 15, ...)
            med_tensor = med_tensors[i]  # [T_fwd, 2*M]

            # Find valid time points
            valid_mask = torch.any(med_mask[i] > 0, dim=1)
            valid_timepoints_count = torch.sum(valid_mask).item()
            total_valid_timepoints += valid_timepoints_count

            if valid_timepoints_count == 0:
                continue

            # Extract rate columns
            rate_cols = list(range(0, med_tensor.shape[1], 5))
            all_med_rates = med_tensor[valid_mask][:, rate_cols]

            # Remove Hetastarch (index 4) and keep only our 11 medications
            keep_indices = [0, 1, 2, 3, 5, 6, 7, 8, 9, 10, 11]  # Skip index 4
            med_rates = all_med_rates[:, keep_indices]

            # Debug first few samples
            if total_processed <= 3:
                print(f"\nDEBUG - Sample {total_processed}:")
                print(f"  Trend: {trend}")
                print(f"  Med tensor shape: {med_tensor.shape}")
                print(f"  Valid timepoints: {valid_timepoints_count}/{med_tensor.shape[0]}")
                print(f"  Med rates shape after extraction: {med_rates.shape}")
                print(f"  Med rates sample values: {med_rates[:3, :3] if med_rates.shape[0] > 0 else 'No data'}")

            # Store rates by trend type
            if trend == 'flat':
                flat_rates.append(med_rates.numpy())
                flat_count += 1
            elif trend == 'up':
                up_rates.append(med_rates.numpy())
                up_count += 1
            else:  # trend == 'down'
                down_rates.append(med_rates.numpy())
                down_count += 1

    print(f"\nProcessing complete:")
    print(f"Total samples processed: {total_processed}")
    print(f"Total valid timepoints across all samples: {total_valid_timepoints}")
    print(
        f"Average valid timepoints per sample: {total_valid_timepoints / total_processed if total_processed > 0 else 0:.1f}")

    # Print trajectory counts and percentages
    print(f"\nTrajectory Classification Results:")
    print(f"Flat trajectories: {flat_count} ({flat_count / total_processed * 100:.1f}%)")
    print(f"Upward trajectories: {up_count} ({up_count / total_processed * 100:.1f}%)")
    print(f"Downward trajectories: {down_count} ({down_count / total_processed * 100:.1f}%)")

    # Combine all data by trajectory type
    if flat_rates:
        flat_data = np.vstack(flat_rates)
        print(f"Flat trajectory timepoints: {flat_data.shape[0]}")
    else:
        flat_data = np.empty((0, len(medication_names)))
        print("No flat trajectories found")

    if up_rates:
        up_data = np.vstack(up_rates)
        print(f"Up trajectory timepoints: {up_data.shape[0]}")
    else:
        up_data = np.empty((0, len(medication_names)))
        print("No up trajectories found")

    if down_rates:
        down_data = np.vstack(down_rates)
        print(f"Down trajectory timepoints: {down_data.shape[0]}")
    else:
        down_data = np.empty((0, len(medication_names)))
        print("No down trajectories found")

    # Determine number of medications
    n_meds = len(medication_names)

    if n_meds == 0:
        print("No medication data found!")
        return None, None

    print(f"Found {n_meds} medications to plot")

    # Use actual medication names
    med_labels = medication_names

    # Create the plot
    plt.style.use('default')  # Reset any previous styles
    fig, ax = plt.subplots(figsize=figsize, facecolor='white')

    # Three-category color scheme
    flat_color = '#808080'  # Gray for flat/stable
    up_color = '#2ca02c'  # Green for upward trend
    down_color = '#d62728'  # Red for downward trend

    # Plot each medication
    has_flat_plots = False
    has_up_plots = False
    has_down_plots = False

    for med_idx in range(n_meds):
        y_pos = n_meds - med_idx - 1  # Reverse order for plotting (top to bottom)

        # Get data for this medication by trajectory type
        flat_med = flat_data[:, med_idx] if flat_data.shape[0] > 0 else np.array([])
        up_med = up_data[:, med_idx] if up_data.shape[0] > 0 else np.array([])
        down_med = down_data[:, med_idx] if down_data.shape[0] > 0 else np.array([])

        # Clean data (remove NaN/inf and zeros, allow all positive values)
        flat_med = flat_med[np.isfinite(flat_med) & (flat_med > 0)]
        up_med = up_med[np.isfinite(up_med) & (up_med > 0)]
        down_med = down_med[np.isfinite(down_med) & (down_med > 0)]

        # Plot upward trajectories (green, at top)
        if len(up_med) > 10:
            bp1 = ax.boxplot(up_med, positions=[y_pos + 0.3], widths=0.2, vert=False,
                             patch_artist=True, showfliers=False,
                             boxprops=dict(facecolor=up_color, alpha=0.7),
                             medianprops=dict(color='black', linewidth=1.5))
            if not has_up_plots:
                bp1['boxes'][0].set_label('Upward')
                has_up_plots = True

        # Plot flat trajectories (gray, in middle)
        if len(flat_med) > 10:
            bp2 = ax.boxplot(flat_med, positions=[y_pos], widths=0.2, vert=False,
                             patch_artist=True, showfliers=False,
                             boxprops=dict(facecolor=flat_color, alpha=0.7),
                             medianprops=dict(color='black', linewidth=1.5))
            if not has_flat_plots:
                bp2['boxes'][0].set_label('Flat')
                has_flat_plots = True

        # Plot downward trajectories (red, at bottom)
        if len(down_med) > 10:
            bp3 = ax.boxplot(down_med, positions=[y_pos - 0.3], widths=0.2, vert=False,
                             patch_artist=True, showfliers=False,
                             boxprops=dict(facecolor=down_color, alpha=0.7),
                             medianprops=dict(color='black', linewidth=1.5))
            if not has_down_plots:
                bp3['boxes'][0].set_label('Downward')
                has_down_plots = True

        # Add subtle baseline line
        ax.axhline(y=y_pos, color='#cccccc', linewidth=0.8, alpha=0.7)

    # Customize the plot
    ax.set_yticks(range(n_meds))
    ax.set_yticklabels(med_labels[::-1], fontsize=11, fontfamily='sans-serif')  # Reverse to match plot order
    ax.set_xlabel('Medication Rate', fontsize=13, fontfamily='sans-serif', fontweight='normal')
    ax.set_ylabel('Medication', fontsize=13, fontfamily='sans-serif', fontweight='normal')
    ax.set_title('Medication Rate Distributions by Trajectory Trend\n(Upward vs Flat vs Downward)',
                 fontsize=15, fontweight='bold', fontfamily='sans-serif', pad=20)

    # Legend
    if has_flat_plots or has_up_plots or has_down_plots:
        legend = ax.legend(loc='upper right', frameon=True, fancybox=False,
                           fontsize=11, framealpha=0.95, edgecolor='black')
        legend.get_frame().set_facecolor('white')
        legend.get_frame().set_linewidth(0.5)

    # Clean grid and spines
    ax.grid(True, alpha=0.25, axis='x', linestyle='-', linewidth=0.5)
    ax.set_axisbelow(True)

    # Remove top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.0)
    ax.spines['bottom'].set_linewidth(1.0)
    ax.spines['left'].set_color('black')
    ax.spines['bottom'].set_color('black')

    # Set tick parameters
    ax.tick_params(axis='both', which='major', labelsize=10,
                   direction='out', length=4, width=1, colors='black')
    ax.tick_params(axis='both', which='minor', length=2, width=0.5, colors='black')

    # Adjust layout
    plt.tight_layout()
    plt.subplots_adjust(left=0.2)  # More space for medication names

    return fig, ax


def main():
    """Main function to run the visualization."""

    print("Creating medication distribution visualization using ALL data...")

    # Create plot using all available data
    fig, ax = plot_medication_distributions(
        data_root=DATA_ROOT,
        icu_stays_path=ICU_STAYS_PATH,
        max_batches=None,  # Use ALL batches (all data)
        batch_size=16,
        max_samples=None,  # Use ALL samples
        figsize=(14, 10)
    )

    if fig is not None:
        # Save plot
        plt.savefig("medications_distributions_3trajectories.png", dpi=300, bbox_inches='tight')
        print("Plot saved as medications_distributions_3trajectories.png")

        plt.show()
        print("Visualization completed!")
    else:
        print("Could not create plot - no data found")


if __name__ == "__main__":
    main()