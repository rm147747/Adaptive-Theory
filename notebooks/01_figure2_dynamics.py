# %% [markdown]
# # Notebook 01: Evolutionary Dynamics — MTD vs AT50 vs CTB
#
# **Purpose**: Reproduce Figure 2 of the companion paper.
# Simulates three treatment policies on a single mCRPC virtual patient
# and compares tumor burden, resistant fraction, and drug exposure.
#
# **Audit trail**:
# - All parameters sourced from `data/parameters/parameter_provenance.json`
# - No external data — all curves generated from ODE model
# - Random seed: N/A (deterministic simulation)
# - Code is self-contained and reproduces exact figure values

# %% [markdown]
# ## 1. Setup and parameter loading

# %%
import sys
sys.path.insert(0, "..")

import numpy as np
import json
from ctb import LVParams, simulate_euler
from ctb.policies import mtd_policy, adaptive_policy, POLICY_SPACE

# Load parameter provenance
with open("../data/parameters/parameter_provenance.json") as f:
    provenance = json.load(f)

print("Parameter source:", provenance["zhang_2017_mCRPC"]["source"])
print("DOI:", provenance["zhang_2017_mCRPC"]["doi"])

# %% [markdown]
# ## 2. Define parameters
#
# These are the default mCRPC parameters documented in `parameter_provenance.json`.
# Each value has a published source or explicit justification.

# %%
params = LVParams(
    r_S=0.0278,       # Zhang 2017, LNCaP-derived
    r_R=0.02,         # cost of resistance (r_R < r_S)
    K=1.0,            # normalized carrying capacity
    alpha_SR=0.8,     # moderate reverse competition
    alpha_RS=1.5,     # strong competitive suppression (key for AT)
    d_S=0.018,        # abiraterone kill rate on sensitive cells
    d_R=0.001,        # minimal effect on resistant cells
)

S0 = 0.85   # 85% sensitive at baseline
R0 = 0.02   # 2% resistant at baseline
N0 = S0 + R0
T_END = 1500

# Validate
errors = params.validate()
assert not errors, f"Parameter validation failed: {errors}"
print(f"Parameters valid. S0={S0}, R0={R0}, N0={N0}")
print(f"Initial resistant fraction: {R0/N0:.1%}")

# %% [markdown]
# ## 3. Simulate three policies

# %%
# Policy 1: MTD (continuous maximum dose)
sim_mtd = simulate_euler(params, S0, R0, mtd_policy, t_end=T_END)
print(f"MTD:  TTP={sim_mtd['TTP']:.0f}d, dose={sim_mtd['cumulative_dose']:.0f}, "
      f"R_final={sim_mtd['R_fraction_final']:.3f}")

# Policy 2: AT50 (adaptive, 50% decline threshold)
at50_fn = adaptive_policy(decline_threshold=0.50, resume_threshold=1.0, check_interval_days=14)
sim_at50 = simulate_euler(params, S0, R0, at50_fn, t_end=T_END)
print(f"AT50: TTP={sim_at50['TTP']:.0f}d, dose={sim_at50['cumulative_dose']:.0f}, "
      f"R_final={sim_at50['R_fraction_final']:.3f}")

# Policy 3: CTB-selected (grid search over all policies)
from ctb import ctb_select_policy
ctb_result = ctb_select_policy(params, S0, R0, t_end=T_END)
best = ctb_result["recommended"]
print(f"CTB ({best['policy']}): TTP={best['TTP_days']:.0f}d, "
      f"dose={best['cumulative_dose']:.0f}, "
      f"R_final={best['R_fraction_final']:.3f}, "
      f"utility={best['utility']:.4f}")
print(f"\nSelection basis: {ctb_result['selection_basis']}")

# Simulate the CTB-selected policy for plotting
ctb_policy_fn = POLICY_SPACE[best["policy"]]
sim_ctb = simulate_euler(params, S0, R0, ctb_policy_fn, t_end=T_END)

# %% [markdown]
# ## 4. Compute derived metrics

# %%
# Resistant fractions over time
rf_mtd = sim_mtd["R"] / sim_mtd["N"]
rf_at50 = sim_at50["R"] / sim_at50["N"]
rf_ctb = sim_ctb["R"] / sim_ctb["N"]

# Summary table
print("\n══════════════════════════════════════════════════")
print(f"{'Policy':<10} {'TTP (d)':>8} {'Dose':>8} {'R_final':>8} {'Dose/MTD':>10}")
print("──────────────────────────────────────────────────")
print(f"{'MTD':<10} {sim_mtd['TTP']:>8.0f} {sim_mtd['cumulative_dose']:>8.0f} "
      f"{sim_mtd['R_fraction_final']:>8.3f} {'100%':>10}")
print(f"{'AT50':<10} {sim_at50['TTP']:>8.0f} {sim_at50['cumulative_dose']:>8.0f} "
      f"{sim_at50['R_fraction_final']:>8.3f} "
      f"{sim_at50['cumulative_dose']/sim_mtd['cumulative_dose']*100:>9.0f}%")
print(f"{'CTB':>10} {sim_ctb['TTP']:>8.0f} {sim_ctb['cumulative_dose']:>8.0f} "
      f"{sim_ctb['R_fraction_final']:>8.3f} "
      f"{sim_ctb['cumulative_dose']/sim_mtd['cumulative_dose']*100:>9.0f}%")
print("══════════════════════════════════════════════════")

# %% [markdown]
# ## 5. Plot Figure 2

# %%
import matplotlib.pyplot as plt

fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

t = sim_mtd["t"]

# Panel A: Tumor burden
ax = axes[0]
ax.plot(t, sim_mtd["N"], color="#E24B4A", linewidth=1.5, label="MTD")
ax.plot(t, sim_at50["N"], color="#378ADD", linewidth=1.5, label="AT50")
ax.plot(t, sim_ctb["N"], color="#1D9E75", linewidth=1.5, label=f"CTB ({best['policy']})")
ax.axhline(y=N0, color="gray", linestyle="--", linewidth=0.5, alpha=0.5)
ax.set_ylabel("Tumor burden N(t)")
ax.set_ylim(0.3, 1.0)
ax.legend(frameon=False)
ax.set_title("A — Tumor burden trajectories", loc="left", fontweight="bold", fontsize=11)

# Panel B: Resistant fraction
ax = axes[1]
ax.plot(t, rf_mtd, color="#E24B4A", linewidth=1.5, label="MTD")
ax.plot(t, rf_at50, color="#378ADD", linewidth=1.5, label="AT50")
ax.plot(t, rf_ctb, color="#1D9E75", linewidth=1.5, label=f"CTB ({best['policy']})")
ax.set_ylabel("R(t) / N(t)")
ax.set_ylim(0, 1.05)
ax.annotate("competitive release", xy=(600, 0.85), color="#E24B4A", fontsize=9)
ax.annotate("competitive suppression", xy=(900, 0.05), color="#1D9E75", fontsize=9)
ax.set_title("B — Resistant clone fraction", loc="left", fontweight="bold", fontsize=11)

# Panel C: Treatment intensity
ax = axes[2]
ax.plot(t, sim_mtd["dose"], color="#E24B4A", linewidth=1.5, label="MTD")
ax.step(t, sim_at50["dose"], color="#378ADD", linewidth=1.5, label="AT50", where="post")
ax.step(t, sim_ctb["dose"], color="#1D9E75", linewidth=1.5, label=f"CTB ({best['policy']})", where="post")
ax.set_ylabel("u(t)")
ax.set_ylim(-0.1, 1.2)
ax.set_yticks([0, 1])
ax.set_yticklabels(["OFF", "ON"])
ax.set_xlabel("Time (days)")
ax.set_title("C — Treatment intensity", loc="left", fontweight="bold", fontsize=11)

plt.tight_layout()
plt.savefig("../figures/figure2_evolutionary_dynamics.png", dpi=300, bbox_inches="tight")
plt.savefig("../figures/figure2_evolutionary_dynamics.pdf", bbox_inches="tight")
print("Figure 2 saved to figures/")
plt.show()

# %% [markdown]
# ## 6. Save raw data for audit

# %%
# Save the exact numerical values used in the figure
step = 5  # downsample for readability
output = {
    "metadata": {
        "description": "Raw simulation data for Figure 2",
        "parameters": {k: v for k, v in params.__dict__.items()},
        "initial_conditions": {"S0": S0, "R0": R0},
        "simulation_horizon_days": T_END,
        "ctb_selected_policy": best["policy"],
        "generated_by": "notebooks/01_figure2_dynamics.py",
    },
    "t": t[::step].tolist(),
    "mtd": {"N": sim_mtd["N"][::step].tolist(), "R_frac": rf_mtd[::step].tolist()},
    "at50": {"N": sim_at50["N"][::step].tolist(), "R_frac": rf_at50[::step].tolist()},
    "ctb": {"N": sim_ctb["N"][::step].tolist(), "R_frac": rf_ctb[::step].tolist()},
}

with open("../data/outputs/figure2_data.json", "w") as f:
    json.dump(output, f, indent=2)

print("Raw data saved to data/outputs/figure2_data.json")
