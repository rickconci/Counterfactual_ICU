import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import odeint


class ZenkerODE:
    """
    Zenker cardiovascular model ODE class - Volume-based implementation
    """

    def __init__(
        self,
        # Initial conditions
        p_a_init=78.937,  # Initial arterial pressure (mmHg) – aligned with Hybrid_SDE
        p_v_init=8.505,  # Initial venous pressure (mmHg) – aligned with Hybrid_SDE
        s_reflex_init=0.5,  # Initial reflex state (0-1)
        sv_init=80.0,  # Initial stroke volume (ml) – midpoint of [40,120]
        # Cardiovascular parameters
        f_hr_max=2.5,  # Maximum heart rate (Hz) – midpoint of [2.0,3.0]
        f_hr_min=1.0,  # Minimum heart rate (Hz) = 30 bpm  # 0.5
        r_tpr_max=2.1,  # Maximum total peripheral resistance – midpoint of [1.8,2.4]
        r_tpr_min=0.525,  # Minimum total peripheral resistance – midpoint of [0.45,0.6]
        r_tpr_mod=0.0,  # TPR modulation
        ca=4.0,  # Arterial compliance  # 2.0
        cv=105.0,  # Venous compliance – midpoint of [90,120]
        k_width=0.20,  # Sigmoid steepness – midpoint of [0.1,0.3]
        p_aset=70.0,  # Arterial pressure setpoint  # 100.0
        tau=20.0,  # Reflex time constant (slower for stability)  # 20.0
        # Enable physiological clamping
        use_physiological_clamping=True,
    ):
        """
        Initialize the Zenker ODE model exactly as specified.
        """

        # Store initial conditions
        self.p_a_init = p_a_init
        self.p_v_init = p_v_init
        self.s_reflex_init = s_reflex_init
        self.sv_init = sv_init

        # Store cardiovascular parameters (individual attributes, not dictionary)
        self.f_hr_max = f_hr_max
        self.f_hr_min = f_hr_min
        self.r_tpr_max = r_tpr_max
        self.r_tpr_min = r_tpr_min
        self.r_tpr_mod = r_tpr_mod
        self.ca = ca
        self.cv = cv
        self.k_width = k_width
        self.p_aset = p_aset
        self.tau = tau

        # Physiological clamping
        self.use_physiological_clamping = use_physiological_clamping

        # Define physiological ranges for clamping (min, max)
        self.physio_ranges = {
            "p_a": (40.0, 220.0),  # Arterial pressure (mmHg)
            "p_v": (0.0, 39.0),  # Venous pressure (mmHg)
            "s_reflex": (0.0, 1.0),  # Reflex state (normalized)
            "sv": (20.0, 150.0),  # Stroke volume (ml)
        }

        # Convert to numpy arrays for efficient clamping
        self.physio_min = np.array(
            [self.physio_ranges[k][0] for k in ["p_a", "p_v", "s_reflex", "sv"]]
        )
        self.physio_max = np.array(
            [self.physio_ranges[k][1] for k in ["p_a", "p_v", "s_reflex", "sv"]]
        )

        # Initialize solution storage
        self.t = None
        self.solution = None

    def get_initial_conditions(self):
        """Get initial conditions as a numpy array."""
        return np.array(
            [self.p_a_init, self.p_v_init, self.s_reflex_init, self.sv_init]
        )

    def apply_physiological_clamps(self, y):
        """Apply physiological bounds to state vector."""
        if self.use_physiological_clamping:
            return np.clip(y, self.physio_min, self.physio_max)
        return y

    def derivatives(self, y, t):
        """
        Compute derivatives with SCALING DEBUG.

        Args:
            y: State vector [p_a, p_v, s_reflex, sv]
            t: Time

        Returns:
            dy_dt: Derivative vector
        """

        # Apply physiological clamping if enabled
        y_clamped = self.apply_physiological_clamps(y)
        p_a, p_v, s, sv = y_clamped

        # Calculate heart rate and resistance exactly as specified
        f_hr = s * (self.f_hr_max - self.f_hr_min) + self.f_hr_min  # Already in Hz
        r_tpr = s * (self.r_tpr_max - self.r_tpr_min) + self.r_tpr_min + self.r_tpr_mod

        # Volume derivatives exactly as specified - now units are consistent!
        outflow = (p_a - p_v) / r_tpr  # ml/s
        inflow = sv * f_hr  # ml/s
        dva_dt = -1.0 * outflow + inflow
        # Pressure derivatives exactly as specified
        dpa_dt = dva_dt / (self.ca)  # * 100.)
        # dpv_dt = -100*self.ca*dpa_dt / (self.cv) #* 10.)
        dpv_dt = -dva_dt / self.cv

        # Reflex derivative exactly as specified
        sigmoid = 1.0 / (1 + np.exp(-self.k_width * (p_a - self.p_aset)))
        ds_dt = (1.0 / self.tau) * (1.0 - sigmoid - s)

        # Stroke volume derivative
        dsv_dt = 0

        return np.array([dpa_dt, dpv_dt, ds_dt, dsv_dt])

    def integrate(self, t_span=1200.0, dt=0.1):
        """
        Integrate the ODE system over time and plot results.

        Args:
            t_span: Total integration time (seconds, default 1200)
            dt: Time step (seconds, default 0.1)

        Returns:
            t: Time array
            solution: Solution array [time, states]
        """
        # Create time array
        self.t = np.arange(0, t_span + dt, dt)

        # Get initial conditions and apply clamping if enabled
        y0 = self.get_initial_conditions()
        y0 = self.apply_physiological_clamps(y0)

        # Integrate ODE
        self.solution = odeint(self.derivatives, y0, self.t)

        # Post-integration clamping if enabled
        if self.use_physiological_clamping:
            self.solution = np.clip(self.solution, self.physio_min, self.physio_max)

        return self.t, self.solution

    def plot(self, figsize=(15, 10)):
        """Plot the integration results."""
        if self.solution is None:
            print("No solution to plot. Run integrate() first.")
            return

        fig, axes = plt.subplots(2, 3, figsize=figsize)
        fig.suptitle(
            "Zenker Cardiovascular Model - Volume-Based Implementation", fontsize=16
        )

        # Extract state variables
        p_a = self.solution[:, 0]
        p_v = self.solution[:, 1]
        s_reflex = self.solution[:, 2]
        sv = self.solution[:, 3]

        # Calculate derived variables for visualization
        f_hr_values = []
        r_tpr_values = []
        for s in s_reflex:
            f_hr = (s * (self.f_hr_max - self.f_hr_min) + self.f_hr_min) * 60
            r_tpr = (
                s * (self.r_tpr_max - self.r_tpr_min) + self.r_tpr_min + self.r_tpr_mod
            )
            f_hr_values.append(f_hr)
            r_tpr_values.append(r_tpr)

        f_hr_values = np.array(f_hr_values)
        r_tpr_values = np.array(r_tpr_values)

        # Plot arterial pressure
        axes[0, 0].plot(self.t, p_a, "r-", linewidth=2)
        axes[0, 0].set_title("Arterial Pressure")
        axes[0, 0].set_xlabel("Time (s)")
        axes[0, 0].set_ylabel("Pressure (mmHg)")
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].axhline(
            y=self.p_aset,
            color="red",
            linestyle="--",
            alpha=0.5,
            label=f"Setpoint ({self.p_aset})",
        )
        axes[0, 0].set_ylim(40, 220)
        axes[0, 0].legend()

        # Plot venous pressure
        axes[0, 1].plot(self.t, p_v, "b-", linewidth=2)
        axes[0, 1].set_title("Venous Pressure")
        axes[0, 1].set_xlabel("Time (s)")
        axes[0, 1].set_ylabel("Pressure (mmHg)")
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_ylim(0, 39)

        # Plot reflex state
        axes[0, 2].plot(self.t, s_reflex, "g-", linewidth=2)
        axes[0, 2].set_title("Reflex State")
        axes[0, 2].set_xlabel("Time (s)")
        axes[0, 2].set_ylabel("Reflex State")
        axes[0, 2].grid(True, alpha=0.3)
        axes[0, 2].set_ylim(0, 1)

        # Plot stroke volume
        axes[1, 0].plot(self.t, sv, "m-", linewidth=2)
        axes[1, 0].set_title("Stroke Volume")
        axes[1, 0].set_xlabel("Time (s)")
        axes[1, 0].set_ylabel("Volume (ml)")
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_ylim(20, 150)

        # Plot heart rate (convert Hz to bpm for display)
        axes[1, 1].plot(self.t, f_hr_values, "c-", linewidth=2)
        axes[1, 1].set_title("Heart Rate")
        axes[1, 1].set_xlabel("Time (s)")
        axes[1, 1].set_ylabel("Rate (bpm)")
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_ylim(self.f_hr_min * 60, self.f_hr_max * 60)

        # Plot total peripheral resistance
        axes[1, 2].plot(self.t, r_tpr_values, "orange", linewidth=2)
        axes[1, 2].set_title("Total Peripheral Resistance")
        axes[1, 2].set_xlabel("Time (s)")
        axes[1, 2].set_ylabel("Resistance")
        axes[1, 2].grid(True, alpha=0.3)
        axes[1, 2].set_ylim(self.r_tpr_min, self.r_tpr_max)

        plt.tight_layout()
        plt.show()

        # Check for physiological bounds violations
        if self.use_physiological_clamping:
            violations = self.check_physiological_bounds()
            if violations:
                print("\n=== Physiological Bounds Violations ===")
                for var_name, violation_info in violations.items():
                    print(
                        f"{var_name}: {violation_info['total_violations']} violations"
                    )
                    print(
                        f"  At min bound ({violation_info['bounds'][0]}): {violation_info['at_min_bound']} times"
                    )
                    print(
                        f"  At max bound ({violation_info['bounds'][1]}): {violation_info['at_max_bound']} times"
                    )
            else:
                print("\n=== No Physiological Bounds Violations Detected ===")

    def check_physiological_bounds(self):
        """Check if any variables hit physiological bounds during integration."""
        if self.solution is None:
            return {}

        violations = {}
        var_names = ["p_a", "p_v", "s_reflex", "sv"]

        for i, var_name in enumerate(var_names):
            min_val, max_val = self.physio_ranges[var_name]
            var_data = self.solution[:, i]

            at_min = np.sum(np.abs(var_data - min_val) < 1e-6)
            at_max = np.sum(np.abs(var_data - max_val) < 1e-6)

            if at_min > 0 or at_max > 0:
                violations[var_name] = {
                    "at_min_bound": at_min,
                    "at_max_bound": at_max,
                    "total_violations": at_min + at_max,
                    "bounds": (min_val, max_val),
                }

        return violations

    def get_current_state(self, t_index):
        """Get the cardiovascular state at a specific time index."""
        if self.solution is None:
            print("No solution available. Run integrate() first.")
            return None

        p_a, p_v, s, sv = self.solution[t_index]
        f_hr = s * (self.f_hr_max - self.f_hr_min) + self.f_hr_min
        r_tpr = s * (self.r_tpr_max - self.r_tpr_min) + self.r_tpr_min + self.r_tpr_mod

        return {
            "time": self.t[t_index],
            "p_a": p_a,
            "p_v": p_v,
            "s_reflex": s,
            "sv": sv,
            "f_hr": f_hr,
            "r_tpr": r_tpr,
        }

    def _interp_control(self, t_grid, values_or_fn):
        """Return a callable u(t) from array/scalar/callable control input.

        - callable: returned as-is
        - scalar: constant function
        - array-like: linear interpolation over t_grid (clamped to endpoints)
        """
        if callable(values_or_fn):
            return values_or_fn
        if isinstance(values_or_fn, (int, float)):
            val = float(values_or_fn)
            return lambda t: val
        arr = np.asarray(values_or_fn, dtype=float)
        if arr.ndim != 1 or arr.shape[0] != len(t_grid):
            raise ValueError("Control arrays must be 1D and match the time grid length")
        t_grid = np.asarray(t_grid, dtype=float)

        def fn(t):
            return float(np.interp(t, t_grid, arr))

        return fn

    def simulate_with_controls(
        self, t_span=1200.0, dt=0.1, controls=None, t_grid=None, clamp_states=True
    ):
        """Simulate ODE with time-varying controls affecting venous, SV, ca, r_tpr_mod.

        Controls dictionary keys:
        - "u1_dpv": additive to dpv_dt
        - "u2_dsv": sets dsv_dt
        - "u3_dca": additive to d(ca)/dt (treat ca as dynamic state)
        - "u4_drtpr": additive to d(r_tpr_mod)/dt (treat r_tpr_mod as dynamic state)

        Each control can be scalar, len(t_grid) array, or callable u(t).

        Returns (t, sol_aug, control_series), where sol_aug columns are
        [p_a, p_v, s_reflex, sv, ca, r_tpr_mod].
        """
        # Build time grid
        if t_grid is None:
            t = np.arange(0.0, t_span + dt, dt, dtype=float)
        else:
            t = np.asarray(t_grid, dtype=float)

        controls = controls or {}

        # Support state-feedback controls: each u accepts (t, y_aug)
        def _as_state_feedback(values_or_fn, t_grid_local=None):
            if callable(values_or_fn):
                try:
                    narg = values_or_fn.__code__.co_argcount
                except Exception:
                    narg = 1
                return (
                    values_or_fn
                    if narg >= 2
                    else (lambda tt, y: float(values_or_fn(tt)))
                )
            if isinstance(values_or_fn, (int, float)):
                val = float(values_or_fn)
                return lambda tt, y: val
            arr = np.asarray(values_or_fn, dtype=float)
            if t_grid_local is None or arr.shape[0] != len(t_grid_local):
                raise ValueError("Control arrays must match provided t_grid length")
            return lambda tt, y: float(np.interp(tt, t_grid_local, arr))

        u1 = _as_state_feedback(controls.get("u1_dpv", 0.0), t)
        u2 = _as_state_feedback(controls.get("u2_dsv", 0.0), t)
        u3 = _as_state_feedback(controls.get("u3_dca", 0.0), t)
        u4 = _as_state_feedback(controls.get("u4_drtpr", 0.0), t)

        # Initial augmented state: include dynamic ca and r_tpr_mod
        y0_aug = np.array(
            [
                self.p_a_init,
                self.p_v_init,
                self.s_reflex_init,
                self.sv_init,
                self.ca,
                self.r_tpr_mod,
            ],
            dtype=float,
        )

        phys_min = self.physio_min
        phys_max = self.physio_max

        def f_aug(y, tt):
            p_a, p_v, s, sv, ca_dyn, r_tpr_mod_dyn = y

            # Clamp physiological states if enabled
            if clamp_states and self.use_physiological_clamping:
                p_a = np.clip(p_a, phys_min[0], phys_max[0])
                p_v = np.clip(p_v, phys_min[1], phys_max[1])
                s = np.clip(s, phys_min[2], phys_max[2])
                sv = np.clip(sv, phys_min[3], phys_max[3])

            # Derived parameters
            f_hr = s * (self.f_hr_max - self.f_hr_min) + self.f_hr_min
            r_tpr = (
                s * (self.r_tpr_max - self.r_tpr_min) + self.r_tpr_min + r_tpr_mod_dyn
            )

            # Hemodynamics
            outflow = (p_a - p_v) / r_tpr
            inflow = sv * f_hr
            dva_dt = -1.0 * outflow + inflow
            dpa_dt = dva_dt / ca_dyn
            dpv_dt = -dva_dt / self.cv

            # Reflex
            sigmoid = 1.0 / (1 + np.exp(-self.k_width * (p_a - self.p_aset)))
            ds_dt = (1.0 / self.tau) * (1.0 - sigmoid - s)

            # Controls (state-feedback enabled)
            y_aug = np.array([p_a, p_v, s, sv, ca_dyn, r_tpr_mod_dyn], dtype=float)
            dpv_dt = dpv_dt + u1(tt, y_aug)
            dsv_dt = u2(tt, y_aug)
            dca_dt = u3(tt, y_aug)
            drtpr_dt = u4(tt, y_aug)

            return np.array(
                [dpa_dt, dpv_dt, ds_dt, dsv_dt, dca_dt, drtpr_dt], dtype=float
            )

        sol_aug = odeint(f_aug, y0_aug, t)

        # Save base solution for existing plot() API
        self.t = t
        self.solution = sol_aug[:, :4]

        # Evaluate realized controls along the trajectory using state-feedback
        ctrl_vals = []
        for i, tt in enumerate(t):
            pa_i, pv_i, s_i, sv_i = self.solution[i]
            ca_i = sol_aug[i, 4]
            rmod_i = sol_aug[i, 5]
            y_aug = np.array([pa_i, pv_i, s_i, sv_i, ca_i, rmod_i], dtype=float)
            ctrl_vals.append(
                [
                    u1(tt, y_aug),
                    u2(tt, y_aug),
                    u3(tt, y_aug),
                    u4(tt, y_aug),
                ]
            )
        control_series = np.asarray(ctrl_vals, dtype=float)

        return t, sol_aug, control_series

    def plot_control_impact(
        self, t, sol_aug, control_series, save_path=None, figsize=(8, 8)
    ):
        """Plot pressures and control profiles with approximate derivatives."""
        try:
            plt.switch_backend("Agg")
        except Exception:
            pass

        p_a = sol_aug[:, 0]
        p_v = sol_aug[:, 1]

        # Finite differences of controls
        dt_arr = np.maximum(np.diff(t), 1e-6)
        du = np.vstack(
            [
                np.zeros((1, control_series.shape[1])),
                np.diff(control_series, axis=0) / dt_arr[:, None],
            ]
        )

        num_controls = control_series.shape[1]
        nrows = 1 + num_controls
        fig, axes = plt.subplots(nrows, 1, figsize=figsize, sharex=True)
        if nrows == 1:
            axes = [axes]

        # Pressures
        ax0 = axes[0]
        ax0.plot(t, p_a, "r-", linewidth=2.0, label="Arterial BP")
        ax0.plot(t, p_v, "b-", linewidth=2.0, label="Venous BP")
        ax0.set_ylabel("Pressure (mmHg)")
        ax0.set_xlim(t[0], t[-1])
        ax0.grid(True, alpha=0.3)
        ax0.legend(loc="upper right")

        labels = ["u1: +dpv_dt", "u2: dsv_dt", "u3: dca/dt", "u4: d(r_tpr_mod)/dt"]
        for i in range(num_controls):
            axi = axes[i + 1]
            axi.plot(t, control_series[:, i], linewidth=1.8, label=labels[i])
            axi.plot(
                t,
                du[:, i],
                linestyle=":",
                linewidth=1.2,
                alpha=0.8,
                label=f"d({labels[i]})/dt",
            )
            axi.set_ylabel(labels[i])
            axi.grid(True, alpha=0.3)
            axi.axhline(y=0.0, color="black", linestyle=":", alpha=0.4)
            axi.legend(loc="upper right", fontsize=8)

        axes[-1].set_xlabel("Time (s)")
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"[PLOT] Saved control impact plot: {save_path}")
        plt.close()


# Example usage
if __name__ == "__main__":
    print("🚨 === VENOUS PRESSURE COLLAPSE DIAGNOSIS === 🚨")
    print("The issue: SCALING IMBALANCE between arterial and venous pressure!")
    print("")
    print("❌ Problem with current scaling:")
    print("  • dpa_dt = dva_dt / (ca * 100)")
    print("  • dpv_dt = dvv_dt / (cv * 10)")
    print("  • With ca=2, cv=4: arterial scale=200, venous scale=40")
    print("  • Venous pressure changes 5x FASTER than arterial!")
    print("  • Result: Venous pressure hits zero in seconds!")

    print("\n📊 Testing current parameters to show the problem...")
    model = ZenkerODE(
        p_a_init=100,
        p_v_init=4,
        s_reflex_init=0.5,
        sv_init=70.0,
        ca=4,
        cv=111,  # Current values causing imbalance
        f_hr_max=3.0,
        f_hr_min=1.0,
        use_physiological_clamping=True,
    )

    t_short, _ = model.integrate(t_span=20.0, dt=0.1)  # Short test to see collapse

    if (
        input("\nTry full integration with balanced scaling? (y/n): ")
        .lower()
        .startswith("y")
    ):
        t_full, _ = model.integrate(t_span=1200.0, dt=0.1)

    model.plot()  # creates the figure
    plt.savefig("zenker_plot.png", dpi=300, bbox_inches="tight")

    # --- Control impact demo ---
    print("\n⚙️ Running control impact demo with time-varying controls...")
    T = 600.0
    dt = 1.0
    t_demo = np.arange(0, T + dt, dt)

    U1 = np.zeros_like(t_demo)
    U1[(t_demo >= 100) & (t_demo <= 200)] = 0.02  # add to dpv_dt

    U2 = np.zeros_like(t_demo)  # set dsv_dt

    U3 = np.zeros_like(t_demo)
    U3[(t_demo >= 300) & (t_demo <= 500)] = 0.005  # d(ca)/dt

    U4 = np.zeros_like(t_demo)
    U4[(t_demo >= 400)] = -0.0005  # d(r_tpr_mod)/dt

    t_out, sol_aug, controls = model.simulate_with_controls(
        t_grid=t_demo,
        controls={
            "u1_dpv": U1,
            "u2_dsv": U2,
            "u3_dca": U3,
            "u4_drtpr": U4,
        },
    )

    out_path = "zenker_control_impact_demo.png"
    model.plot_control_impact(t_out, sol_aug, controls, save_path=out_path)
    print(f"[PLOT] Saved control impact plot: {out_path}")
