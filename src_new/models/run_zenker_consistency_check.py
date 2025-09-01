import os
import sys
from typing import Dict, Tuple

import numpy as np
import torch


def add_project_to_syspath() -> None:
    # Add .../Counterfactual_ICU/src_new to sys.path
    src_new_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    if src_new_root not in sys.path:
        sys.path.insert(0, src_new_root)


add_project_to_syspath()

# Imports that depend on sys.path including the project root
from utils.train_utils import zenker_derivatives  # noqa: E402

from models.ZenkerModel import ZenkerODE  # noqa: E402


def sample_parameters(
    batch_size: int, rng: np.random.Generator
) -> Dict[str, np.ndarray]:
    """Sample physiologically plausible states and parameters for testing.

    Returns arrays shaped [batch_size].
    """
    # State variables
    p_a = rng.uniform(60.0, 140.0, size=batch_size)  # mmHg
    p_v = rng.uniform(2.0, 20.0, size=batch_size)  # mmHg
    s_reflex = rng.uniform(0.0, 1.0, size=batch_size)
    sv = rng.uniform(40.0, 120.0, size=batch_size)  # ml

    # Parameters (kept positive and reasonable)
    f_hr_min = rng.uniform(0.7, 1.2, size=batch_size)  # Hz (42-72 bpm)
    f_hr_max = rng.uniform(2.0, 3.3, size=batch_size)  # Hz (120-198 bpm)
    # Ensure f_hr_max > f_hr_min per-sample
    mask = f_hr_max <= f_hr_min
    f_hr_max[mask] = f_hr_min[mask] + 0.5

    r_tpr_min = rng.uniform(0.3, 1.2, size=batch_size)
    r_tpr_max = rng.uniform(1.3, 3.0, size=batch_size)
    # Ensure r_tpr_max > r_tpr_min per-sample
    mask = r_tpr_max <= r_tpr_min
    r_tpr_max[mask] = r_tpr_min[mask] + 0.5

    r_tpr_mod = rng.uniform(
        -0.1, 0.1, size=batch_size
    )  # small modulation so total stays > 0

    ca = rng.uniform(2.0, 6.0, size=batch_size)
    cv = rng.uniform(60.0, 150.0, size=batch_size)

    k_width = rng.uniform(0.05, 0.3, size=batch_size)
    p_aset = rng.uniform(70.0, 110.0, size=batch_size)
    tau = rng.uniform(10.0, 40.0, size=batch_size)

    return {
        "p_a": p_a,
        "p_v": p_v,
        "s_reflex": s_reflex,
        "sv": sv,
        "f_hr_min": f_hr_min,
        "f_hr_max": f_hr_max,
        "r_tpr_min": r_tpr_min,
        "r_tpr_max": r_tpr_max,
        "r_tpr_mod": r_tpr_mod,
        "ca": ca,
        "cv": cv,
        "k_width": k_width,
        "p_aset": p_aset,
        "tau": tau,
    }


def build_torch_state_matrix(
    params: Dict[str, np.ndarray], device: torch.device
) -> torch.Tensor:
    """Pack state + parameters into the torch layout expected by zenker_derivatives.

    Order: [p_a, p_v, s_reflex, sv, r_tpr_mod, f_hr_max, f_hr_min, r_tpr_max, r_tpr_min, ca, cv, k_width, p_aset, tau]
    """
    y = np.stack(
        [
            params["p_a"],
            params["p_v"],
            params["s_reflex"],
            params["sv"],
            params["r_tpr_mod"],
            params["f_hr_max"],
            params["f_hr_min"],
            params["r_tpr_max"],
            params["r_tpr_min"],
            params["ca"],
            params["cv"],
            params["k_width"],
            params["p_aset"],
            params["tau"],
        ],
        axis=1,
    )
    y_torch = torch.tensor(y, dtype=torch.float32, device=device)
    return y_torch


def compare_derivatives(
    batch_size: int = 32, seed: int = 123, atol: float = 1e-5, rtol: float = 1e-5
) -> Tuple[bool, Dict[str, float]]:
    """Compare numpy (ZenkerODE) derivatives and torch (zenker_derivatives) across random samples.

    Returns a tuple (all_close, max_abs_diffs)
    """
    rng = np.random.default_rng(seed)
    params = sample_parameters(batch_size, rng)

    device = torch.device("cpu")
    y_torch = build_torch_state_matrix(params, device)

    # Torch derivatives (batched)
    (
        dpa_dt_t,  # [B,1]
        dpv_dt_t,
        ds_dt_t,
        dsv_dt_t,
        *_,
    ) = zenker_derivatives(y_torch, device, expert_start_index=0)

    dpa_dt_t = dpa_dt_t.squeeze(1).cpu().numpy()
    dpv_dt_t = dpv_dt_t.squeeze(1).cpu().numpy()
    ds_dt_t = ds_dt_t.squeeze(1).cpu().numpy()
    dsv_dt_t = dsv_dt_t.squeeze(1).cpu().numpy()

    # Numpy derivatives (per-sample because ZenkerODE holds scalar params)
    dpa_dt_n = np.zeros(batch_size, dtype=np.float64)
    dpv_dt_n = np.zeros(batch_size, dtype=np.float64)
    ds_dt_n = np.zeros(batch_size, dtype=np.float64)
    dsv_dt_n = np.zeros(batch_size, dtype=np.float64)

    for i in range(batch_size):
        model = ZenkerODE(
            p_a_init=params["p_a"][i],
            p_v_init=params["p_v"][i],
            s_reflex_init=params["s_reflex"][i],
            sv_init=params["sv"][i],
            f_hr_max=params["f_hr_max"][i],
            f_hr_min=params["f_hr_min"][i],
            r_tpr_max=params["r_tpr_max"][i],
            r_tpr_min=params["r_tpr_min"][i],
            r_tpr_mod=params["r_tpr_mod"][i],
            ca=params["ca"][i],
            cv=params["cv"][i],
            k_width=params["k_width"][i],
            p_aset=params["p_aset"][i],
            tau=params["tau"][i],
            use_physiological_clamping=False,  # avoid clamping for strict equality
        )

        # State vector for derivative evaluation
        y_np = np.array(
            [
                params["p_a"][i],
                params["p_v"][i],
                params["s_reflex"][i],
                params["sv"][i],
            ]
        )

        d_np = model.derivatives(y_np, t=0.0)
        dpa_dt_n[i] = d_np[0]
        dpv_dt_n[i] = d_np[1]
        ds_dt_n[i] = d_np[2]
        dsv_dt_n[i] = d_np[3]

    diffs = {
        "dpa_dt": float(np.max(np.abs(dpa_dt_t - dpa_dt_n))),
        "dpv_dt": float(np.max(np.abs(dpv_dt_t - dpv_dt_n))),
        "ds_dt": float(np.max(np.abs(ds_dt_t - ds_dt_n))),
        "dsv_dt": float(np.max(np.abs(dsv_dt_t - dsv_dt_n))),
    }

    all_close = (
        np.allclose(dpa_dt_t, dpa_dt_n, rtol=rtol, atol=atol)
        and np.allclose(dpv_dt_t, dpv_dt_n, rtol=rtol, atol=atol)
        and np.allclose(ds_dt_t, ds_dt_n, rtol=rtol, atol=atol)
        and np.allclose(dsv_dt_t, dsv_dt_n, rtol=rtol, atol=atol)
    )

    return all_close, diffs


def main() -> int:
    torch.set_grad_enabled(False)
    seed = int(os.environ.get("ZC_SEED", "123"))
    batch_size = int(os.environ.get("ZC_BATCH", "64"))
    atol = float(os.environ.get("ZC_ATOL", "1e-5"))
    rtol = float(os.environ.get("ZC_RTL", "1e-5"))

    ok, diffs = compare_derivatives(
        batch_size=batch_size, seed=seed, atol=atol, rtol=rtol
    )

    status = "PASS" if ok else "FAIL"
    print(f"Zenker derivatives consistency: {status}")
    print(
        "Max |difference|: "
        f"dpa_dt={diffs['dpa_dt']:.3e}, "
        f"dpv_dt={diffs['dpv_dt']:.3e}, "
        f"ds_dt={diffs['ds_dt']:.3e}, "
        f"dsv_dt={diffs['dsv_dt']:.3e}"
    )

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
