import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint


class ZenkerODE:
    """
    Zenker cardiovascular model ODE class - Volume-based implementation
    """

    def __init__(self,
                 # Initial conditions
                 p_a_init=100.0,  # Initial arterial pressure (mmHg)
                 p_v_init=8.0,  # Initial venous pressure (mmHg)
                 s_reflex_init=0.5,  # Initial reflex state (0-1)
                 sv_init=70.0,  # Initial stroke volume (ml)

                 # Cardiovascular parameters
                 f_hr_max=3.0,  # Maximum heart rate (Hz) = 180 bpm  # 3.0
                 f_hr_min=1.0,  # Minimum heart rate (Hz) = 30 bpm  # 0.5
                 r_tpr_max=2.13,  # Maximum total peripheral resistance  # 2.0
                 r_tpr_min=0.53,  # Minimum total peripheral resistance  # 0.8
                 r_tpr_mod=0.0,  # TPR modulation
                 ca=4.0,  # Arterial compliance (scaled by *100)  # 2.0
                 cv=111.0,  # Venous compliance (scaled by *10) - LARGER for balance  # 20.0
                 k_width=0.18,  # Sigmoid steepness for reflex (gentle)  # 0.05
                 p_aset=70.0,  # Arterial pressure setpoint  # 100.0
                 tau=20.0,  # Reflex time constant (slower for stability)  # 20.0

                # Enable physiological clamping
                 use_physiological_clamping=True):
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
            'p_a': (40.0, 220.0),  # Arterial pressure (mmHg)
            'p_v': (0.0, 39.0),  # Venous pressure (mmHg)
            's_reflex': (0.0, 1.0),  # Reflex state (normalized)
            'sv': (20.0, 150.0)  # Stroke volume (ml)
        }

        # Convert to numpy arrays for efficient clamping
        self.physio_min = np.array([self.physio_ranges[k][0] for k in ['p_a', 'p_v', 's_reflex', 'sv']])
        self.physio_max = np.array([self.physio_ranges[k][1] for k in ['p_a', 'p_v', 's_reflex', 'sv']])

        # Initialize solution storage
        self.t = None
        self.solution = None

    def get_initial_conditions(self):
        """Get initial conditions as a numpy array."""
        return np.array([self.p_a_init, self.p_v_init, self.s_reflex_init, self.sv_init])

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
        dva_dt = -1. * outflow + inflow
        # Pressure derivatives exactly as specified
        dpa_dt = dva_dt / (self.ca) #* 100.)
        #dpv_dt = -100*self.ca*dpa_dt / (self.cv) #* 10.)
        dpv_dt = -dva_dt / self.cv


        # Reflex derivative exactly as specified
        sigmoid = 1. / (1 + np.exp(-self.k_width * (p_a - self.p_aset)))
        ds_dt = (1. / self.tau) * (1. - sigmoid - s)

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
        fig.suptitle('Zenker Cardiovascular Model - Volume-Based Implementation', fontsize=16)

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
            r_tpr = s * (self.r_tpr_max - self.r_tpr_min) + self.r_tpr_min + self.r_tpr_mod
            f_hr_values.append(f_hr)
            r_tpr_values.append(r_tpr)

        f_hr_values = np.array(f_hr_values)
        r_tpr_values = np.array(r_tpr_values)

        # Plot arterial pressure
        axes[0, 0].plot(self.t, p_a, 'r-', linewidth=2)
        axes[0, 0].set_title('Arterial Pressure')
        axes[0, 0].set_xlabel('Time (s)')
        axes[0, 0].set_ylabel('Pressure (mmHg)')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].axhline(y=self.p_aset, color='red', linestyle='--', alpha=0.5, label=f'Setpoint ({self.p_aset})')
        axes[0, 0].set_ylim(40, 220)
        axes[0, 0].legend()

        # Plot venous pressure
        axes[0, 1].plot(self.t, p_v, 'b-', linewidth=2)
        axes[0, 1].set_title('Venous Pressure')
        axes[0, 1].set_xlabel('Time (s)')
        axes[0, 1].set_ylabel('Pressure (mmHg)')
        axes[0, 1].grid(True, alpha=0.3)
        axes[0, 1].set_ylim(0, 39)

        # Plot reflex state
        axes[0, 2].plot(self.t, s_reflex, 'g-', linewidth=2)
        axes[0, 2].set_title('Reflex State')
        axes[0, 2].set_xlabel('Time (s)')
        axes[0, 2].set_ylabel('Reflex State')
        axes[0, 2].grid(True, alpha=0.3)
        axes[0, 2].set_ylim(0, 1)

        # Plot stroke volume
        axes[1, 0].plot(self.t, sv, 'm-', linewidth=2)
        axes[1, 0].set_title('Stroke Volume')
        axes[1, 0].set_xlabel('Time (s)')
        axes[1, 0].set_ylabel('Volume (ml)')
        axes[1, 0].grid(True, alpha=0.3)
        axes[1, 0].set_ylim(20, 150)

        # Plot heart rate (convert Hz to bpm for display)
        axes[1, 1].plot(self.t, f_hr_values, 'c-', linewidth=2)
        axes[1, 1].set_title('Heart Rate')
        axes[1, 1].set_xlabel('Time (s)')
        axes[1, 1].set_ylabel('Rate (bpm)')
        axes[1, 1].grid(True, alpha=0.3)
        axes[1, 1].set_ylim(self.f_hr_min * 60, self.f_hr_max * 60)

        # Plot total peripheral resistance
        axes[1, 2].plot(self.t, r_tpr_values, 'orange', linewidth=2)
        axes[1, 2].set_title('Total Peripheral Resistance')
        axes[1, 2].set_xlabel('Time (s)')
        axes[1, 2].set_ylabel('Resistance')
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
                    print(f"{var_name}: {violation_info['total_violations']} violations")
                    print(f"  At min bound ({violation_info['bounds'][0]}): {violation_info['at_min_bound']} times")
                    print(f"  At max bound ({violation_info['bounds'][1]}): {violation_info['at_max_bound']} times")
            else:
                print("\n=== No Physiological Bounds Violations Detected ===")

    def check_physiological_bounds(self):
        """Check if any variables hit physiological bounds during integration."""
        if self.solution is None:
            return {}

        violations = {}
        var_names = ['p_a', 'p_v', 's_reflex', 'sv']

        for i, var_name in enumerate(var_names):
            min_val, max_val = self.physio_ranges[var_name]
            var_data = self.solution[:, i]

            at_min = np.sum(np.abs(var_data - min_val) < 1e-6)
            at_max = np.sum(np.abs(var_data - max_val) < 1e-6)

            if at_min > 0 or at_max > 0:
                violations[var_name] = {
                    'at_min_bound': at_min,
                    'at_max_bound': at_max,
                    'total_violations': at_min + at_max,
                    'bounds': (min_val, max_val)
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
            'time': self.t[t_index],
            'p_a': p_a,
            'p_v': p_v,
            's_reflex': s,
            'sv': sv,
            'f_hr': f_hr,
            'r_tpr': r_tpr
        }


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
        use_physiological_clamping=True
    )

    t_short, _ = model.integrate(t_span=20.0, dt=0.1)  # Short test to see collapse

    if input("\nTry full integration with balanced scaling? (y/n): ").lower().startswith('y'):
        t_full, _ = model.integrate(t_span=1200.0, dt=0.1)

    model.plot()  # creates the figure
    plt.savefig('zenker_plot.png', dpi=300, bbox_inches='tight')