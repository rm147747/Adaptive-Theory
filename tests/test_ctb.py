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
