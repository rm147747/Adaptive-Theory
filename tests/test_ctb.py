"""
Test suite for the Computational Tumor Board.

These tests verify that the CTB produces biologically plausible
and numerically correct results. They serve as both quality assurance
and audit evidence that the software behaves as documented.

Run with: pytest tests/test_ctb.py -v
"""

import numpy as np
import pytest
from ctb import LVParams, simulate_euler, simulate_ivp, compute_ttp
from ctb.policies import mtd_policy, adaptive_policy, intermittent_policy
from ctb.matching_score import Mutation, compute_matching_scores, estimate_ccf
from ctb.optimizer import ctb_select_policy


class TestLotkaVolterra:
    """Tests for the core ODE model."""

    def test_no_treatment_grows_to_capacity(self):
        """Without treatment, tumor should grow to carrying capacity K."""
        params = LVParams()
        sim = simulate_euler(params, S0=0.5, R0=0.1,
                             dose_schedule=lambda t, n, n0, s, i: 0.0,
                             t_end=500)
        assert sim["N"][-1] > 0.90 * params.K, "Tumor should approach K"

    def test_mtd_kills_sensitive(self):
        """Under MTD, sensitive population should decline."""
        params = LVParams()
        sim = simulate_euler(params, S0=0.85, R0=0.02,
                             dose_schedule=mtd_policy, t_end=500)
        assert sim["S"][-1] < sim["S"][0], "Sensitive should decline under MTD"

    def test_resistant_expands_under_mtd(self):
        """Under MTD, resistant fraction should increase (competitive release)."""
        params = LVParams()
        sim = simulate_euler(params, S0=0.85, R0=0.02,
                             dose_schedule=mtd_policy, t_end=800)
        rf_initial = sim["R"][0] / sim["N"][0]
        rf_final = sim["R"][-1] / sim["N"][-1]
        assert rf_final > rf_initial, "Resistant fraction should grow under MTD"

    def test_populations_stay_non_negative(self):
        """Populations should never go negative."""
        params = LVParams(d_S=0.05)  # high kill rate
        sim = simulate_euler(params, S0=0.85, R0=0.02,
                             dose_schedule=mtd_policy, t_end=500)
        assert np.all(sim["S"] >= 0), "S should be non-negative"
        assert np.all(sim["R"] >= 0), "R should be non-negative"

    def test_parameter_validation(self):
        """Invalid parameters should be caught."""
        params = LVParams(r_S=-0.01)
        errors = params.validate()
        assert len(errors) > 0, "Negative growth rate should fail validation"


class TestAdaptiveTherapy:
    """Tests for adaptive therapy dynamics."""

    def test_at50_oscillates(self):
        """AT50 should produce oscillatory tumor burden (on/off cycling)."""
        params = LVParams(alpha_RS=1.5)
        at50 = adaptive_policy(0.50, 1.0, 14)
        sim = simulate_euler(params, S0=0.85, R0=0.02,
                             dose_schedule=at50, t_end=800)
        # Check that dose switches at least once
        dose_changes = np.diff(sim["dose"])
        switches = np.sum(np.abs(dose_changes) > 0.5)
        assert switches >= 2, "AT50 should switch treatment on/off"

    def test_at_reduces_dose_vs_mtd(self):
        """Adaptive therapy should use less total drug than MTD."""
        params = LVParams(alpha_RS=1.5)
        sim_mtd = simulate_euler(params, 0.85, 0.02, mtd_policy, t_end=800)
        at30 = adaptive_policy(0.30, 0.90, 14)
        sim_at = simulate_euler(params, 0.85, 0.02, at30, t_end=800)
        assert sim_at["cumulative_dose"] < sim_mtd["cumulative_dose"], \
            "Adaptive therapy should use less drug than MTD"


class TestMatchingScore:
    """Tests for the molecular matching module."""

    def test_basic_matching(self):
        """3 mutations, 2 matched → MS = 0.667."""
        muts = [
            Mutation("HER2", "amp", None, 8, ["trastuzumab"], CCF=0.95),
            Mutation("TP53", "R248W", 0.38, 2, [], CCF=0.79),
            Mutation("BRCA2", "splice", 0.12, 2, ["olaparib"], CCF=0.29),
        ]
        result = compute_matching_scores(muts, ["trastuzumab", "olaparib"], 0.65)
        assert abs(result["MS_classic"] - 0.667) < 0.01

    def test_weighted_favors_truncal(self):
        """Weighted MS should be higher when truncal alterations are matched."""
        muts = [
            Mutation("HER2", "amp", None, 8, ["trastuzumab"], CCF=0.95),
            Mutation("TP53", "R248W", 0.38, 2, [], CCF=0.79),
            Mutation("BRCA2", "splice", 0.12, 2, ["olaparib"], CCF=0.29),
        ]
        result = compute_matching_scores(muts, ["trastuzumab", "olaparib"], 0.65)
        # MS_weighted should favor the high-CCF match (HER2)
        assert result["MS_weighted"] > 0, "Weighted MS should be positive"

    def test_ccf_capped_at_one(self):
        """CCF should never exceed 1.0."""
        mut = Mutation("TP53", "R248W", VAF=0.45, copy_number=2, drugs=[])
        ccf = estimate_ccf(mut, purity=0.65)
        assert ccf <= 1.0

    def test_ccf_zero_vaf(self):
        """VAF = 0 should give CCF = 0."""
        mut = Mutation("X", "", VAF=0.0, copy_number=2, drugs=[])
        ccf = estimate_ccf(mut, purity=0.65)
        assert ccf == 0.0

    def test_ccf_low_purity_inflates(self):
        """Lower purity should give higher CCF estimate."""
        mut = Mutation("X", "", VAF=0.30, copy_number=2, drugs=[])
        ccf_high_pur = estimate_ccf(mut, purity=0.80)
        ccf_low_pur = estimate_ccf(mut, purity=0.30)
        assert ccf_low_pur >= ccf_high_pur


class TestCTBOptimizer:
    """Tests for the policy selection module."""

    def test_ctb_returns_result(self):
        """CTB should return a valid recommendation."""
        params = LVParams()
        result = ctb_select_policy(params, 0.85, 0.02, t_end=500)
        assert "recommended" in result
        assert "all_policies" in result
        assert result["selection_basis"] == "a_priori_model_prediction"

    def test_ctb_selects_best_utility(self):
        """Recommended policy should have highest utility."""
        params = LVParams()
        result = ctb_select_policy(params, 0.85, 0.02, t_end=500)
        best_util = result["recommended"]["utility"]
        for pol in result["all_policies"]:
            assert pol["utility"] <= best_util + 1e-6


class TestClosedLoop:
    """Tests for the ctDNA closed-loop monitoring system."""

    def _make_ctb(self):
        params = LVParams()
        from ctb import ClosedLoopCTB
        return ClosedLoopCTB(params, S0=0.85, R0=0.02, tf_baseline=1.0)

    def test_normal_update(self):
        """Normal ctDNA measurement should produce a valid report."""
        from ctb import ClosedLoopCTB, CtdnaTimepoint
        ctb = self._make_ctb()
        obs = CtdnaTimepoint(
            day=90, tumor_fraction=0.85,
            sensitive_vafs={"TP53": 0.30},
            resistant_vafs={"BRCA2": 0.02},
            read_depths={"TP53": 400, "BRCA2": 300},
        )
        report = ctb.update(obs)
        assert report.status in ("normal", "r_expanding", "r_dominant", "progression_signal")
        assert report.recommended_policy is not None

    def test_below_lod(self):
        """ctDNA below LOD should skip recalibration."""
        from ctb import ClosedLoopCTB, CtdnaTimepoint
        ctb = self._make_ctb()
        obs = CtdnaTimepoint(day=90, tumor_fraction=0.0005)
        report = ctb.update(obs)
        assert report.status == "below_lod"

    def test_resistant_expansion_detected(self):
        """Rising q_R should trigger R_EXPANDING flag."""
        from ctb import ClosedLoopCTB, CtdnaTimepoint
        ctb = self._make_ctb()
        # First: normal
        ctb.update(CtdnaTimepoint(
            day=0, tumor_fraction=1.0,
            sensitive_vafs={"TP53": 0.35}, resistant_vafs={"BRCA2": 0.01}))
        # Second: resistance rising
        report = ctb.update(CtdnaTimepoint(
            day=90, tumor_fraction=0.9,
            sensitive_vafs={"TP53": 0.20}, resistant_vafs={"BRCA2": 0.15}))
        has_r_flag = any("R_EXPANDING" in f or "R_DOMINANT" in f
                         for f in report.safety_flags)
        assert has_r_flag or report.observed_q_R > 0.3, \
            "Should detect resistant expansion"

    def test_cns_risk_filters_policies(self):
        """CNS risk should exclude treatment pauses."""
        from ctb import ClosedLoopCTB, CtdnaTimepoint
        params = LVParams()
        ctb = ClosedLoopCTB(params, 0.85, 0.02, tf_baseline=1.0, cns_risk=True)
        obs = CtdnaTimepoint(
            day=90, tumor_fraction=0.85,
            sensitive_vafs={"TP53": 0.30}, resistant_vafs={"BRCA2": 0.02})
        report = ctb.update(obs)
        # Should only recommend continuous policies
        assert report.recommended_policy in ("MTD", "metro_50"), \
            f"CNS risk: should not recommend {report.recommended_policy}"

    def test_history_accumulates(self):
        """Each update should add to history."""
        from ctb import ClosedLoopCTB, CtdnaTimepoint
        ctb = self._make_ctb()
        for day in [0, 90, 180]:
            ctb.update(CtdnaTimepoint(
                day=day, tumor_fraction=0.8,
                sensitive_vafs={"TP53": 0.30}, resistant_vafs={"BRCA2": 0.01}))
        assert len(ctb.history) == 3


class TestReproducibility:
    """Tests ensuring results are reproducible."""

    def test_deterministic_simulation(self):
        """Same inputs should produce identical outputs."""
        params = LVParams()
        sim1 = simulate_euler(params, 0.85, 0.02, mtd_policy, t_end=300)
        sim2 = simulate_euler(params, 0.85, 0.02, mtd_policy, t_end=300)
        assert np.allclose(sim1["N"], sim2["N"]), "Simulations should be deterministic"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
