import argparse
import os
import sys

import numpy as np
import torch


def _ensure_src_on_path() -> None:
    this_dir = os.path.dirname(os.path.abspath(__file__))
    src_root = os.path.abspath(os.path.join(this_dir, ".."))
    if src_root not in sys.path:
        sys.path.insert(0, src_root)


_ensure_src_on_path()

from models.ZenkerModel import ZenkerODE  # noqa: E402
from models.zenker_tracking_demo import make_pa_pv_trajectory_tracker  # noqa: E402
from models.dataloaders.MIMIC_data import MIMICDataModule  # noqa: E402


def build_argparser():
    p = argparse.ArgumentParser(description="Zenker trajectory tracking on MIMIC data")
    p.add_argument("--data_root", type=str, required=True, help="Path to processed data root (BIOMM/processed_data)")
    p.add_argument("--icu_stays_path", type=str, required=True, help="Path to ICUSTAYS.csv")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--max_samples", type=int, default=32, help="Limit total dataset for faster testing")
    p.add_argument("--num_plots", type=int, default=10, help="Number of trajectories to process and plot")
    p.add_argument("--out_dir", type=str, default="zenker_tracking_outputs", help="Directory to save plots")
    p.add_argument("--device", type=str, default="cpu")
    # Controller gains for tracker
    p.add_argument("--kv", type=float, default=0.12)
    p.add_argument("--ka1", type=float, default=0.025)
    p.add_argument("--ka2", type=float, default=-1.0, help="If <0, use 2*sqrt(ka1)")
    p.add_argument("--mode", type=str, default="pdff", choices=["pdff", "ff"])
    return p


def make_unified_plot(t, pa_ref, pv_ref, sol_aug, control_series, out_path):
    import matplotlib.pyplot as plt
    try:
        plt.switch_backend("Agg")
    except Exception:
        pass

    pa = sol_aug[:, 0]
    pv = sol_aug[:, 1]

    fig, axes = plt.subplots(4, 1, figsize=(9, 9), sharex=True)

    axes[0].plot(t, pa_ref, "k--", linewidth=1.5, label="p_a* (ref)")
    axes[0].plot(t, pa, "r-", linewidth=2.0, label="p_a (actual)")
    axes[0].set_ylabel("p_a (mmHg)")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(loc="upper right")

    axes[1].plot(t, pv_ref, "k--", linewidth=1.5, label="p_v* (ref)")
    axes[1].plot(t, pv, "b-", linewidth=2.0, label="p_v (actual)")
    axes[1].set_ylabel("p_v (mmHg)")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(loc="upper right")

    labels = ["u1: +dpv_dt", "u2: dsv_dt", "u3: dca/dt", "u4: d(r_tpr_mod)/dt"]
    # Derivatives
    dt_arr = np.maximum(np.diff(t), 1e-9)
    dU = np.vstack([np.zeros((1, control_series.shape[1])), np.diff(control_series, axis=0) / dt_arr[:, None]])
    # Integrated controls
    for i in range(control_series.shape[1]):
        axes[2].plot(t, control_series[:, i], linewidth=1.6, label=labels[i])
    axes[2].axhline(0.0, color="black", linestyle=":", alpha=0.4)
    axes[2].set_ylabel("controls")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(loc="upper right", ncol=2, fontsize=8)
    # Derivatives separate
    for i in range(control_series.shape[1]):
        axes[3].plot(t, dU[:, i], linewidth=1.2, label=f"d({labels[i]})/dt")
    axes[3].axhline(0.0, color="black", linestyle=":", alpha=0.4)
    axes[3].set_ylabel("d(controls)/dt")
    axes[3].set_xlabel("Time (s)")
    axes[3].grid(True, alpha=0.3)
    axes[3].legend(loc="upper right", ncol=2, fontsize=8)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()


def main():
    args = build_argparser().parse_args()
    device = torch.device(args.device)

    # Setup datamodule (use validation split by default)
    dm = MIMICDataModule(
        data_root=args.data_root,
        icu_stays_path=args.icu_stays_path,
        batch_size=args.batch_size,
        num_workers=0,
        random_state=42,
        max_samples=args.max_samples,
        use_raindrop_context=True,
        expert_latent_dim=14,
    )
    dm.setup()
    val_loader = dm.val_dataloader()

    zen = ZenkerODE()

    processed = 0
    sample_global_idx = 0
    for batch in val_loader:
        # Unpack collated batch per MIMICDataModule.collate_fn
        (
            rd_src,
            rd_times,
            rd_length,
            static,
            init_state,
            ic_mask,
            Y,
            Y_mask,
            t_Y,
            med_values,
            med_mask,
            med_time,
        ) = batch

        B, T, C = Y.shape
        assert C == 2, "Expected Y to have 2 channels (p_a, p_v)"

        # Filter samples with full mask (Pa and Pv observed entire trajectory)
        full_mask = (Y_mask[..., 0].sum(dim=1) == T) & (Y_mask[..., 1].sum(dim=1) == T)
        idxs = torch.where(full_mask)[0].tolist()
        for bi in idxs:
            if processed >= args.num_plots:
                break

            t = t_Y[bi].detach().cpu().numpy()
            pa_ref = Y[bi, :, 0].detach().cpu().numpy()
            pv_ref = Y[bi, :, 1].detach().cpu().numpy()

            # Build tracker
            ka2 = None if args.ka2 < 0 else args.ka2
            u1, u2, u3, u4 = make_pa_pv_trajectory_tracker(
                zen,
                t_grid=t,
                pa_ref=pa_ref,
                pv_ref=pv_ref,
                mode=args.mode,
                kv=args.kv,
                ka1=args.ka1,
                ka2=ka2,
                w=(1.0, 5.0, 5.0),
            )

            # Simulate with feedback controls
            t_out, sol_aug, U = zen.simulate_with_controls(
                t_span=float(t[-1]),
                dt=max(float(np.diff(t).min()), 0.1),
                controls={"u1_dpv": u1, "u2_dsv": u2, "u3_dca": u3, "u4_drtpr": u4},
                t_grid=t,
                clamp_states=True,
            )

            # Plot unified figure
            out_name = f"sample_{sample_global_idx:05d}.png"
            out_path = os.path.join(args.out_dir, out_name)
            make_unified_plot(t_out, pa_ref, pv_ref, sol_aug, U, out_path)
            print(f"[PLOT] Saved: {out_path}")

            processed += 1
            sample_global_idx += 1

        if processed >= args.num_plots:
            break

    if processed == 0:
        print("No fully-observed Pa/Pv trajectories found in validation split.")
    else:
        print(f"Done. Generated {processed} tracking plots in '{args.out_dir}'.")


if __name__ == "__main__":
    main()


