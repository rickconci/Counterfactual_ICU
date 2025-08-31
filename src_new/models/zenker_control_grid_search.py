import os
import sys
from typing import Dict, List, Tuple

import numpy as np


def _ensure_src_on_path() -> None:
    this_dir = os.path.dirname(os.path.abspath(__file__))
    src_root = os.path.abspath(os.path.join(this_dir, ".."))
    if src_root not in sys.path:
        sys.path.insert(0, src_root)


_ensure_src_on_path()

from models.ZenkerModel import ZenkerODE  # noqa: E402


def build_time_grid(total_seconds: float = 20 * 60, dt: float = 1.0) -> np.ndarray:
    return np.arange(0.0, total_seconds + dt, dt, dtype=float)


def make_profiles(t: np.ndarray, amplitude: float) -> Dict[str, np.ndarray]:
    T = t[-1] if t.size > 0 else 1.0
    profiles: Dict[str, np.ndarray] = {}

    # Constant over full horizon
    profiles["constant"] = np.full_like(t, amplitude, dtype=float)

    # Pulse: middle third
    start_p = 0.33 * T
    end_p = 0.66 * T
    pulse = np.zeros_like(t, dtype=float)
    pulse[(t >= start_p) & (t <= end_p)] = amplitude
    profiles["pulse_mid"] = pulse

    # Ramp up from 0 -> amplitude over full horizon
    ramp = amplitude * (t / max(T, 1e-6))
    profiles["ramp_up"] = ramp

    # Single-period sine over horizon
    sine = amplitude * np.sin(2.0 * np.pi * t / max(T, 1e-6))
    profiles["sine_1T"] = sine

    return profiles


def simulate_and_score(model: ZenkerODE, t: np.ndarray, control_key: str, profile: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    controls = {control_key: profile}
    t_out, sol_aug, control_series = model.simulate_with_controls(t_grid=t, controls=controls)

    # sol_aug columns: [p_a, p_v, s_reflex, sv, ca, r_tpr_mod]
    p_a = sol_aug[:, 0]
    p_v = sol_aug[:, 1]

    metrics = {
        "pa_mean": float(np.mean(p_a)),
        "pv_mean": float(np.mean(p_v)),
        "pa_end": float(p_a[-1]),
        "pv_end": float(p_v[-1]),
        "pa_min": float(np.min(p_a)),
        "pa_max": float(np.max(p_a)),
        "pv_min": float(np.min(p_v)),
        "pv_max": float(np.max(p_v)),
    }
    return t_out, sol_aug, control_series, metrics


def run_grid_search(out_dir: str) -> None:
    import csv
    import matplotlib
    matplotlib.use("Agg")

    os.makedirs(out_dir, exist_ok=True)

    # Fixed initial conditions across all runs
    model = ZenkerODE()  # use defaults aligned with Hybrid_SDE midpoints/means

    t = build_time_grid(total_seconds=20 * 60, dt=1.0)

    # Baseline (no control) for deltas
    t_base, sol_base, ctrl_base = model.simulate_with_controls(t_grid=t, controls={})
    base_pa_mean = float(np.mean(sol_base[:, 0]))
    base_pv_mean = float(np.mean(sol_base[:, 1]))

    # Control keys and sensible base scales (per-second) for 20 min horizon
    control_specs = [
        ("u1_dpv", 0.10),   # mmHg/s
        ("u2_dsv", 0.002),  # ml/s
        ("u3_dca", 0.0005), # compliance units/s
        ("u4_drtpr", 0.0005), # resistance-mod units/s
    ]

    multipliers = [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]
    profile_names = ["constant", "pulse_mid", "ramp_up", "sine_1T"]

    summary_rows: List[Dict[str, object]] = []

    for control_key, base_scale in control_specs:
        ctrl_dir = os.path.join(out_dir, control_key)
        os.makedirs(ctrl_dir, exist_ok=True)

        csv_path = os.path.join(ctrl_dir, f"summary_{control_key}.csv")
        with open(csv_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "control",
                "multiplier",
                "amplitude",
                "profile",
                "pa_mean",
                "pv_mean",
                "pa_mean_delta",
                "pv_mean_delta",
                "pa_end",
                "pv_end",
                "pa_min",
                "pa_max",
                "pv_min",
                "pv_max",
                "plot_path",
            ])

            for mult in multipliers:
                amplitude = mult * base_scale
                profiles = make_profiles(t, amplitude)
                for pname in profile_names:
                    profile = profiles[pname]

                    # Fresh model per run to avoid side-effects
                    model_i = ZenkerODE()

                    t_out, sol_aug, control_series, metrics = simulate_and_score(
                        model_i, t, control_key, profile
                    )

                    # Save plot
                    plot_name = f"{control_key}_{pname}_mult{mult:+.2f}.png".replace("+", "p").replace("-", "m").replace(".", "p")
                    plot_path = os.path.join(ctrl_dir, plot_name)
                    try:
                        model_i.plot_control_impact(t_out, sol_aug, control_series, save_path=plot_path)
                    except Exception as e:
                        plot_path = f"ERROR: {e}"

                    pa_delta = metrics["pa_mean"] - base_pa_mean
                    pv_delta = metrics["pv_mean"] - base_pv_mean

                    writer.writerow([
                        control_key,
                        f"{mult:.2f}",
                        f"{amplitude:.6f}",
                        pname,
                        f"{metrics['pa_mean']:.4f}",
                        f"{metrics['pv_mean']:.4f}",
                        f"{pa_delta:.4f}",
                        f"{pv_delta:.4f}",
                        f"{metrics['pa_end']:.4f}",
                        f"{metrics['pv_end']:.4f}",
                        f"{metrics['pa_min']:.4f}",
                        f"{metrics['pa_max']:.4f}",
                        f"{metrics['pv_min']:.4f}",
                        f"{metrics['pv_max']:.4f}",
                        plot_path,
                    ])

                    summary_rows.append(
                        {
                            "control": control_key,
                            "multiplier": mult,
                            "amplitude": amplitude,
                            "profile": pname,
                            **metrics,
                            "pa_mean_delta": pa_delta,
                            "pv_mean_delta": pv_delta,
                            "plot_path": plot_path,
                        }
                    )

    # Combined summary CSV
    comb_csv = os.path.join(out_dir, "summary_all_controls.csv")
    import csv as _csv
    with open(comb_csv, "w", newline="") as f:
        if summary_rows:
            fieldnames = list(summary_rows[0].keys())
        else:
            fieldnames = [
                "control",
                "multiplier",
                "amplitude",
                "profile",
                "pa_mean",
                "pv_mean",
                "pa_mean_delta",
                "pv_mean_delta",
                "pa_end",
                "pv_end",
                "pa_min",
                "pa_max",
                "pv_min",
                "pv_max",
                "plot_path",
            ]
        writer = _csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(row)

    print(f"Saved per-control summaries and plots to: {out_dir}")


def main():
    out_root = os.environ.get(
        "ZK_GRID_OUT",
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..",
            "..",
            "zenker_grid_outputs",
        ),
    )
    out_root = os.path.abspath(out_root)
    run_grid_search(out_root)


if __name__ == "__main__":
    main()


