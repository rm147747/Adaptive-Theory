"""
ctDNA closed-loop monitoring and model recalibration.

This module implements Layer 4 of the CTB: serial ctDNA measurements
are used to recalibrate the evolutionary model state and re-evaluate
treatment policies in a closed-loop fashion.

The ctDNA serves as a dual-input evolutionary sensor:
    - Tumor fraction (B̂): global disease burden proxy
    - VAF ratio (q̂_R): relative clonal composition proxy

At each monitoring timepoint:
    1. OBSERVE: parse ctDNA measurements
    2. ESTIMATE: map observations to model state (S, R)
    3. PREDICT: simulate forward under candidate policies
    4. RANK: select best policy by utility function
    5. REPORT: generate clinical decision-support output

IMPORTANT: VAF trajectories are proxies for relative clone dynamics,
not exact measurements of clonal abundance. See docs/METHODS.md §2.2.

References:
    [1] ctDNA monitoring review: npj Precision Oncology 2025;9:84
    [2] Clonal dynamics from ctDNA: Genome Medicine 2021;13:79
    [3] Adaptive abiraterone ctDNA: Annala M et al., Cancer Discov 2018
"""

from dataclasses import dataclass, field
import numpy as np
from .lotka_volterra import LVParams, simulate_euler
from .optimizer import ctb_select_policy
from .policies import POLICY_SPACE


@dataclass
class CtdnaTimepoint:
    """A single ctDNA measurement."""
    day: float
    tumor_fraction: float           # total tumor-derived DNA fraction
    sensitive_vafs: dict = None     # {mutation_id: VAF} for S-associated mutations
    resistant_vafs: dict = None     # {mutation_id: VAF} for R-associated mutations
    read_depths: dict = None        # {mutation_id: total_reads} for uncertainty

    def __post_init__(self):
        self.sensitive_vafs = self.sensitive_vafs or {}
        self.resistant_vafs = self.resistant_vafs or {}
        self.read_depths = self.read_depths or {}


@dataclass
class ClosedLoopReport:
    """Output of a single closed-loop update cycle."""
    day: float
    observed_tumor_fraction: float
    observed_q_R: float
    estimated_S: float
    estimated_R: float
    estimated_N: float
    R_fraction: float
    sigma_tf: float
    recommended_policy: str
    recommended_utility: float
    top3_policies: list
    safety_flags: list
    status: str  # "normal", "below_lod", "progression_signal", "r_expanding"


class ClosedLoopCTB:
    """
    Closed-loop CTB decision-support system.

    Maintains tumor model state and updates with each ctDNA observation.
    Generates treatment policy recommendations at each monitoring timepoint.

    Usage:
        ctb = ClosedLoopCTB(params, S0, R0, tf_baseline)
        report = ctb.update(ctdna_timepoint)

    The system is designed for research use only and provides
    model-suggested strategies, not clinical recommendations.
    """

    # Limit of detection for tumor fraction
    LOD = 0.001

    def __init__(self, params: LVParams, S0: float, R0: float,
                 tf_baseline: float, cns_risk: bool = False,
                 t_horizon: float = 730):
        """
        Initialize the closed-loop CTB.

        Args:
            params: Lotka-Volterra model parameters
            S0, R0: initial tumor state
            tf_baseline: baseline tumor fraction from first ctDNA
            cns_risk: if True, treatment pauses are contraindicated
            t_horizon: forward simulation horizon (days)
        """
        self.params = params
        self.S_current = S0
        self.R_current = R0
        self.N0 = S0 + R0
        self.tf_baseline = tf_baseline
        self.cns_risk = cns_risk
        self.t_horizon = t_horizon

        self.history: list[ClosedLoopReport] = []

    def update(self, obs: CtdnaTimepoint) -> ClosedLoopReport:
        """
        Process a new ctDNA observation and generate recommendation.

        This is the core closed-loop algorithm:
            1. OBSERVE: extract tumor fraction and clone composition
            2. ESTIMATE: map to model state (S, R)
            3. PREDICT: simulate all policies forward
            4. RANK: select best by utility
            5. REPORT: generate output with safety flags

        Args:
            obs: ctDNA measurement timepoint

        Returns:
            ClosedLoopReport with recommendation and diagnostics
        """
        # ── Step 1: OBSERVE ──
        if obs.tumor_fraction < self.LOD:
            return self._below_lod_report(obs)

        # Compute resistant fraction from sentinel VAFs
        q_R = self._estimate_q_R(obs)

        # ── Step 2: ESTIMATE ──
        # Map ctDNA observations to model state
        N_est = (obs.tumor_fraction / self.tf_baseline) * self.N0
        N_est = min(N_est, self.params.K * 2.0)  # cap at 2×K

        R_est = N_est * q_R
        S_est = N_est * (1.0 - q_R)

        # Observation uncertainty (binomial sampling error)
        mean_reads = np.mean(list(obs.read_depths.values())) if obs.read_depths else 300
        tf_clipped = min(obs.tumor_fraction, 0.999)  # prevent negative under sqrt
        sigma_tf = np.sqrt(
            tf_clipped * (1.0 - tf_clipped) / max(mean_reads, 1)
        )

        # Update internal state
        self.S_current = S_est
        self.R_current = R_est

        # ── Step 3 & 4: PREDICT + RANK ──
        # Build policy space (may be filtered by safety constraints)
        policy_space = self._get_safe_policies()

        ctb_result = ctb_select_policy(
            self.params, S_est, R_est,
            t_end=self.t_horizon,
            policy_space=policy_space,
        )

        # ── Step 5: REPORT ──
        safety_flags = self._compute_safety_flags(obs, N_est, q_R)
        status = self._determine_status(safety_flags)

        report = ClosedLoopReport(
            day=obs.day,
            observed_tumor_fraction=round(obs.tumor_fraction, 4),
            observed_q_R=round(q_R, 4),
            estimated_S=round(S_est, 4),
            estimated_R=round(R_est, 4),
            estimated_N=round(S_est + R_est, 4),
            R_fraction=round(q_R, 4),
            sigma_tf=round(sigma_tf, 5),
            recommended_policy=ctb_result["recommended"]["policy"],
            recommended_utility=ctb_result["recommended"]["utility"],
            top3_policies=[
                {"policy": p["policy"], "utility": p["utility"],
                 "TTP": p["TTP_days"]}
                for p in ctb_result["all_policies"][:3]
            ],
            safety_flags=safety_flags,
            status=status,
        )

        self.history.append(report)
        return report

    def _estimate_q_R(self, obs: CtdnaTimepoint) -> float:
        """
        Estimate resistant clone fraction from sentinel mutation VAFs.

        Formula:
            q̂_R = Σ VAF_resistance / Σ VAF_all_tracked

        This is a PROXY for relative clone composition, not an exact
        measurement. See docs/METHODS.md §2.2 for limitations.
        """
        sum_r = sum(obs.resistant_vafs.values()) if obs.resistant_vafs else 0.0
        sum_s = sum(obs.sensitive_vafs.values()) if obs.sensitive_vafs else 0.0
        total = sum_r + sum_s

        if total < 1e-6:
            # No signal — use prior estimate
            if self.history:
                return self.history[-1].observed_q_R
            return self.R_current / (self.S_current + self.R_current)

        return sum_r / total

    def _compute_safety_flags(self, obs: CtdnaTimepoint,
                               N_est: float, q_R: float) -> list[str]:
        """Check safety constraints and generate flags."""
        flags = []

        # CNS risk
        if self.cns_risk:
            flags.append("CNS_RISK: treatment pauses contraindicated")

        # Resistant dominance
        if q_R > 0.50:
            flags.append(f"R_DOMINANT: resistant fraction {q_R:.0%} > 50%")

        # Resistant expansion (rising trend)
        if q_R > 0.10 and len(self.history) >= 1:
            prev_qr = self.history[-1].observed_q_R
            if q_R > prev_qr * 1.3:  # >30% increase
                flags.append(f"R_EXPANDING: q_R rose from {prev_qr:.1%} to {q_R:.1%}")

        # Progression signal (3 consecutive N increases)
        if len(self.history) >= 2:
            recent_n = [h.estimated_N for h in self.history[-2:]] + [N_est]
            if all(recent_n[i + 1] > recent_n[i] for i in range(len(recent_n) - 1)):
                flags.append("PROGRESSION_SIGNAL: burden rising for 3+ timepoints")

        # Persistence filter: single-point VAF fluctuation
        if q_R > 0.05 and len(self.history) >= 1:
            prev_qr = self.history[-1].observed_q_R
            if prev_qr < 0.03 and q_R > 0.10:
                flags.append("CAUTION: sudden q_R jump — confirm at next timepoint")

        return flags

    def _determine_status(self, flags: list[str]) -> str:
        """Classify overall status from safety flags."""
        if any("PROGRESSION" in f for f in flags):
            return "progression_signal"
        if any("R_DOMINANT" in f for f in flags):
            return "r_dominant"
        if any("R_EXPANDING" in f for f in flags):
            return "r_expanding"
        return "normal"

    def _get_safe_policies(self) -> dict:
        """Filter policies by safety constraints."""
        if not self.cns_risk:
            return POLICY_SPACE

        # CNS risk: only allow continuous treatment (no pauses)
        safe = {}
        for name, fn in POLICY_SPACE.items():
            if name in ("MTD", "metro_50"):
                safe[name] = fn
        return safe if safe else POLICY_SPACE  # fallback

    def _below_lod_report(self, obs: CtdnaTimepoint) -> ClosedLoopReport:
        """Generate report when ctDNA is below limit of detection."""
        return ClosedLoopReport(
            day=obs.day,
            observed_tumor_fraction=obs.tumor_fraction,
            observed_q_R=0.0,
            estimated_S=self.S_current,
            estimated_R=self.R_current,
            estimated_N=self.S_current + self.R_current,
            R_fraction=0.0,
            sigma_tf=0.0,
            recommended_policy="continue_current",
            recommended_utility=0.0,
            top3_policies=[],
            safety_flags=["BELOW_LOD: tumor fraction below detection. Consider imaging."],
            status="below_lod",
        )

    def get_summary(self) -> dict:
        """Return summary of all monitoring timepoints."""
        return {
            "n_timepoints": len(self.history),
            "timeline": [
                {
                    "day": h.day,
                    "tf": h.observed_tumor_fraction,
                    "q_R": h.observed_q_R,
                    "status": h.status,
                    "policy": h.recommended_policy,
                    "flags": h.safety_flags,
                }
                for h in self.history
            ],
        }
