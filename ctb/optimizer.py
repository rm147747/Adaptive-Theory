"""
CTB policy optimizer via discretized Stackelberg search.

The CTB evaluates all candidate treatment policies for a given patient's
ecological state and selects the one maximizing a composite utility function.

Game-theoretic interpretation:
    - Physician (leader) selects treatment policy from candidate set
    - Tumor (follower) responds via evolutionary dynamics (Lotka-Volterra)
    - Physician evaluates tumor's best response to each policy
    - Policy maximizing physician utility is selected

This is a discretized approximation to the full Stackelberg equilibrium.

Utility function:
    U = w_TTP × (TTP / T_max) - w_dose × (cumulative_dose / T_max) - w_resist × R_final_fraction

Default weights:
    w_TTP = 1.0   (maximize time to progression)
    w_dose = 0.3   (minimize drug exposure / toxicity)
    w_resist = 0.2 (minimize resistant clone dominance)

References:
    [1] Staňková K et al. JAMA Oncol. 2019;5(1):96-103.
    [2] Salvioli M et al. Dyn Games Appl. 2025;15:1750-1769.
    [3] Ganzfried S. Games. 2024;15(6):45.
"""

from .lotka_volterra import LVParams, simulate_euler
from .policies import POLICY_SPACE
import copy


def evaluate_policy(params: LVParams, S0: float, R0: float,
                    policy_name: str, policy_fn: callable,
                    t_end: float = 1500,
                    w_ttp: float = 1.0,
                    w_dose: float = 0.3,
                    w_resist: float = 0.2) -> dict:
    """
    Evaluate a single policy for a given patient state.

    The policy is simulated forward from (S0, R0) and scored by the utility function.
    Selection occurs BEFORE outcome observation — the CTB uses the model prediction,
    not the actual outcome, to make its decision.

    Args:
        params: Lotka-Volterra parameters for this patient
        S0, R0: initial tumor state
        policy_name: human-readable policy identifier
        policy_fn: callable implementing the dosing schedule
        t_end: simulation horizon
        w_ttp, w_dose, w_resist: utility function weights

    Returns:
        dict with policy name, simulation metrics, and utility score
    """
    sim = simulate_euler(params, S0, R0, policy_fn, t_end=t_end)

    ttp_norm = sim["TTP"] / t_end
    dose_norm = sim["cumulative_dose"] / t_end
    rf = sim["R_fraction_final"]

    utility = w_ttp * ttp_norm - w_dose * dose_norm - w_resist * rf

    return {
        "policy": policy_name,
        "TTP_days": round(sim["TTP"], 1),
        "cumulative_dose": round(sim["cumulative_dose"], 1),
        "R_fraction_final": round(rf, 4),
        "utility": round(utility, 4),
    }


def ctb_select_policy(params: LVParams, S0: float, R0: float,
                      t_end: float = 1500,
                      w_ttp: float = 1.0,
                      w_dose: float = 0.3,
                      w_resist: float = 0.2,
                      policy_space: dict = None) -> dict:
    """
    CTB policy selection: evaluate all candidates, return best.

    IMPORTANT: This function selects the policy a priori based on
    the patient's initial ecological state. It does NOT peek at
    outcomes — the model simulates predicted trajectories under
    each policy and selects the one with highest predicted utility.

    Args:
        params: patient-specific Lotka-Volterra parameters
        S0, R0: initial tumor state
        t_end: simulation horizon
        w_ttp, w_dose, w_resist: utility weights
        policy_space: dict of {name: callable} (defaults to POLICY_SPACE)

    Returns:
        dict with:
            - recommended: best policy evaluation
            - all_policies: sorted list of all evaluations
            - selection_basis: "a_priori_model_prediction"
    """
    if policy_space is None:
        policy_space = POLICY_SPACE

    results = []
    for name, fn in policy_space.items():
        result = evaluate_policy(params, S0, R0, name, fn, t_end,
                                 w_ttp, w_dose, w_resist)
        results.append(result)

    # Sort by utility (descending)
    results.sort(key=lambda x: x["utility"], reverse=True)

    return {
        "recommended": results[0],
        "all_policies": results,
        "selection_basis": "a_priori_model_prediction",
        "utility_weights": {"w_ttp": w_ttp, "w_dose": w_dose, "w_resist": w_resist},
    }
