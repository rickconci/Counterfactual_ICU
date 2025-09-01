import os
import sys

import numpy as np


def _ensure_src_on_path() -> None:
    this_dir = os.path.dirname(os.path.abspath(__file__))
    src_root = os.path.abspath(os.path.join(this_dir, ".."))
    if src_root not in sys.path:
        sys.path.insert(0, src_root)


_ensure_src_on_path()

from models.ZenkerModel import ZenkerODE  # noqa: E402


def _as_state_feedback(values_or_fn, t_grid=None):
    """Return u(t, y) callable. Scalars/arrays become time-only and ignore y."""
    if callable(values_or_fn):
        try:
            narg = values_or_fn.__code__.co_argcount
        except Exception:
            narg = 1
        return values_or_fn if narg >= 2 else (lambda t, y: float(values_or_fn(t)))
    if isinstance(values_or_fn, (int, float)):
        val = float(values_or_fn)
        return lambda t, y: val
    arr = np.asarray(values_or_fn, dtype=float)
    if t_grid is None or arr.shape[0] != len(t_grid):
        raise ValueError("Control arrays must match provided t_grid length")
    return lambda t, y: float(np.interp(t, t_grid, arr))


def make_pa_pv_trajectory_tracker(
    model: ZenkerODE,
    t_grid,
    pa_ref,
    pv_ref,
    mode="pdff",
    kv=0.15,
    ka1=0.03,
    ka2=None,
    w=(1.0, 1.0, 1.0),
    deriv_smooth=3,
    eps=1e-9,
):
    """Return four callables u1..u4: (t, y) -> float to track pa*(t), pv*(t)."""
    import numpy as _np
    from scipy.signal import savgol_filter as _savgol

    t_grid = _np.asarray(t_grid, dtype=float)
    if t_grid.ndim != 1:
        raise ValueError("t_grid must be 1D")
    dt = _np.gradient(t_grid)

    def _as_fn(ref):
        if callable(ref):
            return ref
        arr = _np.asarray(ref, dtype=float)
        if arr.shape[0] != t_grid.shape[0]:
            raise ValueError("Reference arrays must match t_grid length")
        return lambda t: float(_np.interp(t, t_grid, arr))

    pa_star_fn = _as_fn(pa_ref)
    pv_star_fn = _as_fn(pv_ref)

    pa_star = _np.array([pa_star_fn(ti) for ti in t_grid])
    pv_star = _np.array([pv_star_fn(ti) for ti in t_grid])

    if (
        deriv_smooth
        and deriv_smooth > 0
        and deriv_smooth % 2 == 1
        and deriv_smooth >= 3
    ):
        pa_dot = _savgol(pa_star, deriv_smooth, 2, deriv=1, delta=_np.mean(dt))
        pa_ddot = _savgol(pa_star, deriv_smooth, 2, deriv=2, delta=_np.mean(dt))
        pv_dot = _savgol(pv_star, deriv_smooth, 2, deriv=1, delta=_np.mean(dt))
    else:
        pa_dot = _np.gradient(pa_star, t_grid)
        pa_ddot = _np.gradient(pa_dot, t_grid)
        pv_dot = _np.gradient(pv_star, t_grid)

    pa_star_of_t = lambda t: float(_np.interp(t, t_grid, pa_star))
    pa_dot_of_t = lambda t: float(_np.interp(t, t_grid, pa_dot))
    pa_ddot_of_t = lambda t: float(_np.interp(t, t_grid, pa_ddot))
    pv_star_of_t = lambda t: float(_np.interp(t, t_grid, pv_star))
    pv_dot_of_t = lambda t: float(_np.interp(t, t_grid, pv_dot))

    if ka2 is None:
        ka2 = 2.0 * (ka1**0.5)

    w = _np.asarray(w, dtype=float).clip(min=eps)
    W = _np.diag(w)

    rspan = model.r_tpr_max - model.r_tpr_min
    fspan = model.f_hr_max - model.f_hr_min
    cv = model.cv
    k = model.k_width
    pset = model.p_aset
    tau = model.tau

    def ctrl(t, y):
        pa, pv, s, sv, ca, rmod = y
        r = s * rspan + model.r_tpr_min + rmod
        fhr = s * fspan + model.f_hr_min
        r = max(r, eps)
        ca_safe = max(ca, eps)
        dVadt = -(pa - pv) / r + sv * fhr
        A = dVadt / ca_safe

        v2_ff = pv_dot_of_t(t)
        v1_ff = pa_ddot_of_t(t)
        if mode == "pdff":
            v2 = v2_ff - kv * (pv - pv_star_of_t(t))
            v1 = v1_ff - ka1 * (pa - pa_star_of_t(t)) - ka2 * (A - pa_dot_of_t(t))
        elif mode == "ff":
            v2, v1 = v2_ff, v1_ff
        else:
            raise ValueError("mode must be 'ff' or 'pdff'")

        u1 = v2 - ((pa - pv) / (cv * r) - sv * fhr / cv)

        sig = 1.0 / (1.0 + _np.exp(-k * (pa - pset)))
        sdot = (1.0 / tau) * (1.0 - sig - s)
        rdot0 = rspan * sdot
        fhrd = fspan * sdot
        dPdot0 = A - ((pa - pv) / (cv * r) - sv * fhr / cv)
        Fp0 = -(dPdot0 * r - (pa - pv) * rdot0) / (r**2) + sv * fhrd
        a_x = Fp0 / ca_safe

        b1 = 1.0 / (ca_safe * r)
        b2 = fhr / ca_safe
        b3 = -A / ca_safe
        b4 = (pa - pv) / (ca_safe * (r**2))

        r_need = v1 - a_x - b1 * u1

        bvec = _np.array([b2, b3, b4], dtype=float)
        denom = float(bvec @ W @ bvec) + eps
        u_w = (r_need / denom) * (W @ bvec)
        u2, u3, u4 = map(float, u_w)
        return u1, u2, u3, u4

    return (
        lambda t, y: ctrl(t, y)[0],
        lambda t, y: ctrl(t, y)[1],
        lambda t, y: ctrl(t, y)[2],
        lambda t, y: ctrl(t, y)[3],
    )


def main():
    import matplotlib

    matplotlib.use("Agg")

    zen = ZenkerODE()

    # Build a 10-minute horizon
    t_grid = np.arange(0.0, 600.0, 0.1)

    # Example references (smooth, within clamps)
    pa_ref = 95.0 + 10.0 * np.sin(2 * np.pi * t_grid / 120.0)
    pv_ref = 6.0 + 1.0 * np.sin(2 * np.pi * t_grid / 180.0)

    u1, u2, u3, u4 = make_pa_pv_trajectory_tracker(
        zen,
        t_grid,
        pa_ref,
        pv_ref,
        mode="pdff",
        kv=0.12,
        ka1=0.025,
        ka2=None,
        w=(1.0, 5.0, 5.0),
    )

    # Run simulation with feedback controls
    t, sol_aug, U = zen.simulate_with_controls(
        t_span=t_grid[-1],
        dt=0.1,
        controls={"u1_dpv": u1, "u2_dsv": u2, "u3_dca": u3, "u4_drtpr": u4},
        t_grid=t_grid,
        clamp_states=True,
    )

    # Unified figure: desired vs actual p_a/p_v + controls, shared x-axis
    import matplotlib.pyplot as plt

    try:
        plt.switch_backend("Agg")
    except Exception:
        pass

    pa = sol_aug[:, 0]
    pv = sol_aug[:, 1]
    pa_ref = 95.0 + 10.0 * np.sin(2 * np.pi * t / 120.0)
    pv_ref = 6.0 + 1.0 * np.sin(2 * np.pi * t / 180.0)

    nrows = 4
    fig, axes = plt.subplots(nrows, 1, figsize=(9, 9), sharex=True)

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
    # Finite-difference derivatives of controls
    dt_arr = np.maximum(np.diff(t), 1e-9)
    dU = np.vstack([np.zeros((1, U.shape[1])), np.diff(U, axis=0) / dt_arr[:, None]])
    # Integrated control subplot
    for i in range(U.shape[1]):
        axes[2].plot(t, U[:, i], linewidth=1.6, label=labels[i])
    axes[2].axhline(0.0, color="black", linestyle=":", alpha=0.4)
    axes[2].set_ylabel("controls")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(loc="upper right", ncol=2, fontsize=8)
    # Derivative subplot
    for i in range(U.shape[1]):
        axes[3].plot(t, dU[:, i], linewidth=1.2, label=f"d({labels[i]})/dt")
    axes[3].axhline(0.0, color="black", linestyle=":", alpha=0.4)
    axes[3].set_ylabel("d(controls)/dt")
    axes[3].set_xlabel("Time (s)")
    axes[3].grid(True, alpha=0.3)
    axes[3].legend(loc="upper right", ncol=2, fontsize=8)

    plt.tight_layout()
    out_all = os.path.join(os.path.dirname(__file__), "zenker_tracking_unified.png")
    plt.savefig(out_all, dpi=300, bbox_inches="tight")
    print(f"[PLOT] Saved: {out_all}")


if __name__ == "__main__":
    main()
