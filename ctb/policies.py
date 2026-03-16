"""
Treatment policy definitions for the Computational Tumor Board.

Each policy is a callable with signature:
    policy(t, N_current, N0, state, step_index) -> float

where the return value is the dose intensity u(t) in [0, 1].

Policies implemented:
    - MTD: Maximum Tolerated Dose (continuous full dose)
    - AT50: Adaptive therapy with 50% decline threshold
    - AT30: Adaptive therapy with 30% decline threshold
    - Intermittent: Fixed on/off cycling
    - Metronomic: Continuous reduced dose

References:
    [1] Zhang J et al. Nat Commun. 2017;8:1816. (AT50 protocol)
    [2] Gatenby RA et al. Cancer Res. 2009;69(11):4894-4903. (Adaptive therapy concept)
    [3] Staňková K et al. JAMA Oncol. 2019;5(1):96-103. (Stackelberg game framework)
"""


def mtd_policy(t, N_current, N0, state, step_index) -> float:
    """Maximum Tolerated Dose: continuous full dose u(t) = 1."""
    return 1.0


def metronomic_policy(dose_fraction: float = 0.5):
    """Metronomic: continuous reduced dose u(t) = dose_fraction."""
    def policy(t, N_current, N0, state, step_index) -> float:
        return dose_fraction
    policy.__name__ = f"metronomic_{int(dose_fraction * 100)}"
    return policy


def adaptive_policy(decline_threshold: float = 0.50,
                    resume_threshold: float = 1.0,
                    check_interval_days: int = 14):
    """
    Adaptive therapy policy.

    Treatment is ON until tumor burden declines by `decline_threshold` fraction
    from baseline (N0). Treatment is paused until burden recovers to
    `resume_threshold` × N0, then restarted.

    Args:
        decline_threshold: fraction decline from N0 to trigger pause (e.g., 0.50 = 50%)
        resume_threshold: fraction of N0 to trigger restart (e.g., 1.0 = baseline)
        check_interval_days: re-evaluation interval (simulates clinical visits)

    Protocol source:
        AT50 follows Zhang et al. 2017: abiraterone discontinued at ≥50% PSA decline,
        resumed at PSA return to baseline. AT30 is a more aggressive variant.
    """
    def policy(t, N_current, N0, state, step_index) -> float:
        if step_index % check_interval_days == 0:
            if state["treating"]:
                state["nadir"] = min(state.get("nadir", N0), N_current)
                if N_current <= N0 * (1.0 - decline_threshold):
                    state["treating"] = False
            else:
                if N_current >= N0 * resume_threshold:
                    state["treating"] = True
                    state["nadir"] = N_current
        return 1.0 if state["treating"] else 0.0

    policy.__name__ = f"AT{int(decline_threshold * 100)}_resume{int(resume_threshold * 100)}"
    return policy


def intermittent_policy(on_days: int = 28, off_days: int = 14):
    """
    Fixed intermittent schedule: on for X days, off for Y days.

    Args:
        on_days: days of treatment per cycle
        off_days: days of treatment holiday per cycle
    """
    cycle = on_days + off_days

    def policy(t, N_current, N0, state, step_index) -> float:
        return 1.0 if (t % cycle) < on_days else 0.0

    policy.__name__ = f"int_{on_days}_{off_days}"
    return policy


# ═══════════════════════════════════════════════════════════
# POLICY SPACE: all candidate policies for CTB grid search
# ═══════════════════════════════════════════════════════════

POLICY_SPACE = {
    "MTD": mtd_policy,
    "AT50": adaptive_policy(0.50, 1.0, 14),
    "AT30": adaptive_policy(0.30, 0.90, 14),
    "AT40": adaptive_policy(0.40, 0.95, 14),
    "AT30_fast": adaptive_policy(0.30, 0.85, 7),
    "int_28_14": intermittent_policy(28, 14),
    "int_21_14": intermittent_policy(21, 14),
    "int_28_28": intermittent_policy(28, 28),
    "metro_50": metronomic_policy(0.50),
}
