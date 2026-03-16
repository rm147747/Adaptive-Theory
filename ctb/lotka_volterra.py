"""
Lotka-Volterra competition model for tumor dynamics.

This module implements the two-population competitive Lotka-Volterra system
used to model therapy-sensitive (S) and therapy-resistant (R) tumor clones.

Mathematical formulation:
    dS/dt = r_S * S * (1 - (S + α_SR * R) / K) - u(t) * d_S * S
    dR/dt = r_R * R * (1 - (R + α_RS * S) / K) - u(t) * d_R * R

Parameter sources:
    - Growth rates (r_S, r_R): Zhang et al., Nature Communications 2017, Table 1
      Derived from cell line doubling times (LNCaP, PC-3), scaled to 10% of in vitro values
    - Competition coefficients (α): Estimated ranges from Strobl et al., Cancer Research 2024
    - Drug kill rates (d_S, d_R): Calibrated to match published abiraterone response curves

References:
    [1] Zhang J, Cunningham JJ, Brown JS, Gatenby RA. Integrating evolutionary dynamics
        into treatment of metastatic castrate-resistant prostate cancer.
        Nat Commun. 2017;8:1816. doi:10.1038/s41467-017-01968-5
    [2] Zhang J, Cunningham J, Brown J, Gatenby R. Evolution-based mathematical models
        significantly prolong response to abiraterone in metastatic castrate-resistant
        prostate cancer. eLife. 2022;11:e76284. doi:10.7554/eLife.76284
    [3] Strobl MAR et al. Mathematical Model-Driven Deep Learning Enables Personalized
        Adaptive Therapy. Cancer Res. 2024;84(11):1929. doi:10.1158/0008-5472.CAN-23-2040
"""

from dataclasses import dataclass, field
import numpy as np
from scipy.integrate import solve_ivp


@dataclass
class LVParams:
    """
    Lotka-Volterra model parameters.

    All parameters have documented sources and biological justification.
    Default values correspond to the mCRPC abiraterone scenario from Zhang et al. 2017.
    """

    # Growth rates (day^-1)
    # Source: Zhang et al. 2017, Table 1. LNCaP doubling time → r_S; PC-3 → r_R
    # Scaled to 10% of in vitro values for in vivo approximation
    r_S: float = 0.0278  # sensitive cell growth rate
    r_R: float = 0.02    # resistant cell growth rate (lower = cost of resistance)

    # Carrying capacity (normalized)
    K: float = 1.0

    # Competition coefficients (dimensionless)
    # α_SR: effect of resistant cells on sensitive growth
    # α_RS: effect of sensitive cells on resistant growth
    # α_RS > 1 means sensitive cells strongly suppress resistant (key for AT)
    # Source: estimated from Strobl et al. 2024 parameter fitting
    alpha_SR: float = 0.8
    alpha_RS: float = 1.5

    # Drug-induced death rates (day^-1)
    # d_S >> d_R: drug selectively kills sensitive cells
    # Source: calibrated to match Zhang et al. 2017 abiraterone response curves
    d_S: float = 0.018
    d_R: float = 0.001  # minimal drug effect on resistant cells

    def validate(self) -> list[str]:
        """Validate parameters are biologically plausible."""
        errors = []
        if self.r_S <= 0:
            errors.append(f"r_S must be positive, got {self.r_S}")
        if self.r_R <= 0:
            errors.append(f"r_R must be positive, got {self.r_R}")
        if self.K <= 0:
            errors.append(f"K must be positive, got {self.K}")
        if self.alpha_SR < 0:
            errors.append(f"alpha_SR must be non-negative, got {self.alpha_SR}")
        if self.alpha_RS < 0:
            errors.append(f"alpha_RS must be non-negative, got {self.alpha_RS}")
        if self.d_S < 0:
            errors.append(f"d_S must be non-negative, got {self.d_S}")
        if self.d_R < 0:
            errors.append(f"d_R must be non-negative, got {self.d_R}")
        if self.d_R > self.d_S:
            errors.append(f"d_R ({self.d_R}) > d_S ({self.d_S}): drug kills resistant more than sensitive")
        return errors


def lotka_volterra_rhs(t: float, y: np.ndarray, params: LVParams,
                       dose_intensity: float) -> list[float]:
    """
    Right-hand side of the Lotka-Volterra ODE system.

    Args:
        t: current time (days)
        y: state vector [S, R] (population abundances)
        params: model parameters
        dose_intensity: u(t), treatment intensity in [0, 1]

    Returns:
        [dS/dt, dR/dt]
    """
    S = max(y[0], 0.0)
    R = max(y[1], 0.0)
    u = dose_intensity

    dS = (params.r_S * S * (1.0 - (S + params.alpha_SR * R) / params.K)
          - u * params.d_S * S)
    dR = (params.r_R * R * (1.0 - (R + params.alpha_RS * S) / params.K)
          - u * params.d_R * R)

    return [dS, dR]


def simulate_euler(params: LVParams, S0: float, R0: float,
                   dose_schedule: callable, t_end: float = 1500,
                   dt: float = 1.0) -> dict:
    """
    Simulate tumor dynamics using forward Euler method.

    We use Euler instead of solve_ivp for the stepwise adaptive controller
    because the dose function depends on the current state (feedback control).

    Args:
        params: LV model parameters
        S0: initial sensitive population
        R0: initial resistant population
        dose_schedule: callable(t, N_current, N0, state, step_index) -> float in [0,1]
        t_end: simulation horizon (days)
        dt: time step (days)

    Returns:
        dict with keys: t, S, R, N, dose, TTP, cumulative_dose, R_fraction_final
    """
    errors = params.validate()
    if errors:
        raise ValueError(f"Invalid parameters: {errors}")

    N0 = S0 + R0
    n_steps = int(t_end / dt)
    t_arr = np.arange(0, t_end, dt)

    S_arr = np.zeros(n_steps)
    R_arr = np.zeros(n_steps)
    dose_arr = np.zeros(n_steps)

    S_arr[0] = S0
    R_arr[0] = R0

    # State dict for adaptive controllers
    controller_state = {"treating": True, "nadir": N0}

    for i in range(1, n_steps):
        s, r = S_arr[i - 1], R_arr[i - 1]
        n = s + r

        # Get dose from schedule
        u = dose_schedule(t_arr[i], n, N0, controller_state, i)
        dose_arr[i] = u

        # Forward Euler step
        dS, dR = lotka_volterra_rhs(t_arr[i - 1], [s, r], params, u)
        S_arr[i] = max(s + dS * dt, 1e-10)
        R_arr[i] = max(r + dR * dt, 1e-10)

    N_arr = S_arr + R_arr

    # Compute time to progression (TTP)
    ttp = compute_ttp(t_arr, N_arr, N0)

    return {
        "t": t_arr,
        "S": S_arr,
        "R": R_arr,
        "N": N_arr,
        "dose": dose_arr,
        "TTP": ttp,
        "cumulative_dose": float(np.sum(dose_arr) * dt),
        "R_fraction_final": float(R_arr[-1] / N_arr[-1]) if N_arr[-1] > 0 else 0.0,
    }


def simulate_ivp(params: LVParams, S0: float, R0: float,
                 dose_intensity: float, t_end: float = 1500) -> dict:
    """
    Simulate with constant dose using scipy solve_ivp (RK45).

    Use this for continuous-dose policies (MTD, metronomic) where
    dose does not depend on current state.

    Args:
        params: LV model parameters
        S0: initial sensitive population
        R0: initial resistant population
        dose_intensity: constant u(t) in [0, 1]
        t_end: simulation horizon (days)

    Returns:
        dict with keys: t, S, R, N, TTP, cumulative_dose, R_fraction_final
    """
    t_eval = np.arange(0, t_end, 1.0)

    sol = solve_ivp(
        fun=lambda t, y: lotka_volterra_rhs(t, y, params, dose_intensity),
        t_span=(0, t_end),
        y0=[S0, R0],
        t_eval=t_eval,
        method="RK45",
        max_step=1.0,
    )

    S = np.clip(sol.y[0], 0, None)
    R = np.clip(sol.y[1], 0, None)
    N = S + R
    N0 = S0 + R0
    ttp = compute_ttp(sol.t, N, N0)

    return {
        "t": sol.t,
        "S": S,
        "R": R,
        "N": N,
        "dose": np.full_like(sol.t, dose_intensity),
        "TTP": ttp,
        "cumulative_dose": float(dose_intensity * t_end),
        "R_fraction_final": float(R[-1] / N[-1]) if N[-1] > 0 else 0.0,
    }


def compute_ttp(t: np.ndarray, N: np.ndarray, N0: float) -> float:
    """
    Compute time to progression (TTP).

    Definition: first time after nadir at which tumor burden N(t) returns
    to the initial value N0. If no progression occurs, returns t[-1].

    Args:
        t: time array
        N: tumor burden array
        N0: initial tumor burden

    Returns:
        TTP in days
    """
    # Find nadir in first half of simulation
    search_window = min(len(N), int(len(N) * 0.6))
    nadir_idx = np.argmin(N[:search_window])

    if nadir_idx < 5:
        return 0.0

    # Find first time after nadir where N >= N0
    post_nadir = N[nadir_idx:]
    progression_indices = np.where(post_nadir >= N0)[0]

    if len(progression_indices) == 0:
        return float(t[-1])  # no progression

    return float(t[nadir_idx + progression_indices[0]])
