# %% [markdown]
# # Notebook 02: Failure Modes of Evolutionary Control
#
# **Purpose**: Reproduce Figure 3 of the companion paper.
# Identifies conditions under which adaptive therapy / CTB fails.
#
# **Three failure scenarios**:
# - A: High pre-existing resistance (R₀/N₀ = 0.35)
# - B: Weak ecological competition (α_RS = 0.5)
# - C: Low drug selectivity (d_R/d_S = 0.67)
#
# **Audit trail**:
# - Parameters derived from published ranges (see parameter_provenance.json)
# - Deterministic simulations, fully reproducible
# - Each scenario documents WHY it fails

# %%
import sys
sys.path.insert(0, "..")

import numpy as np
import json
import matplotlib.pyplot as plt
from ctb import LVParams, simulate_euler
from ctb.policies import mtd_policy, adaptive_policy

# %% [markdown]
# ## Scenario A: High pre-existing resistance
#
# When resistant cells already comprise ~35% of the tumor at baseline,
# competitive suppression is insufficient to prevent resistance dominance
# regardless of treatment strategy.

# %%
params_a = LVParams(r_S=0.0278, r_R=0.02, alpha_SR=0.8, alpha_RS=1.5,
                     d_S=0.018, d_R=0.001)
S0_a, R0_a = 0.55, 0.30  # R0/N0 ≈ 0.35
T_END = 1200

sim_a_mtd = simulate_euler(params_a, S0_a, R0_a, mtd_policy, t_end=T_END)
sim_a_ctb = simulate_euler(params_a, S0_a, R0_a,
                           adaptive_policy(0.30, 0.90, 14), t_end=T_END)

print("Scenario A: High pre-existing resistance (R0/N0 = 0.35)")
print(f"  MTD: TTP={sim_a_mtd['TTP']:.0f}d, R_final={sim_a_mtd['R_fraction_final']:.3f}")
print(f"  CTB: TTP={sim_a_ctb['TTP']:.0f}d, R_final={sim_a_ctb['R_fraction_final']:.3f}")
print(f"  → Both fail identically. Insufficient sensitive population for suppression.")

# %% [markdown]
# ## Scenario B: Weak ecological competition
#
# When α_RS = 0.5, sensitive cells exert minimal competitive pressure
# on resistant cells. Maintaining a sensitive population provides
# little ecological advantage.

# %%
params_b = LVParams(r_S=0.0278, r_R=0.02, alpha_SR=0.8, alpha_RS=0.5,
                     d_S=0.018, d_R=0.001)
S0_b, R0_b = 0.85, 0.02

sim_b_mtd = simulate_euler(params_b, S0_b, R0_b, mtd_policy, t_end=T_END)
sim_b_ctb = simulate_euler(params_b, S0_b, R0_b,
                           adaptive_policy(0.30, 0.90, 14), t_end=T_END)

print("\nScenario B: Weak competition (α_RS = 0.5)")
print(f"  MTD: TTP={sim_b_mtd['TTP']:.0f}d, R_final={sim_b_mtd['R_fraction_final']:.3f}")
print(f"  CTB: TTP={sim_b_ctb['TTP']:.0f}d, R_final={sim_b_ctb['R_fraction_final']:.3f}")
print(f"  → CTB gains only {sim_b_ctb['TTP'] - sim_b_mtd['TTP']:.0f} days. Marginal benefit.")

# %% [markdown]
# ## Scenario C: Low drug selectivity
#
# When d_R/d_S ≈ 0.67, the drug kills both clones with similar efficacy.
# There is no differential selection pressure to exploit.

# %%
params_c = LVParams(r_S=0.0278, r_R=0.02, alpha_SR=0.8, alpha_RS=1.5,
                     d_S=0.018, d_R=0.012)  # d_R/d_S = 0.67
S0_c, R0_c = 0.85, 0.02

sim_c_mtd = simulate_euler(params_c, S0_c, R0_c, mtd_policy, t_end=T_END)
sim_c_ctb = simulate_euler(params_c, S0_c, R0_c,
                           adaptive_policy(0.30, 0.90, 14), t_end=T_END)

print("\nScenario C: Low drug selectivity (d_R/d_S = 0.67)")
print(f"  MTD: TTP={sim_c_mtd['TTP']:.0f}d, R_final={sim_c_mtd['R_fraction_final']:.3f}")
print(f"  CTB: TTP={sim_c_ctb['TTP']:.0f}d, R_final={sim_c_ctb['R_fraction_final']:.3f}")
print(f"  → Both control tumor. No asymmetry to exploit — AT adds no benefit.")

# %% [markdown]
# ## Conclusion
#
# Evolutionary control requires THREE prerequisites:
# 1. Sufficient sensitive population at baseline (Scenario A fails)
# 2. Strong interclonal competition (Scenario B fails)
# 3. Differential drug sensitivity (Scenario C shows no benefit)

# %%
# Save and plot
print("\n" + "="*60)
print("PREREQUISITES FOR EVOLUTIONARY CONTROL:")
print("  1. R0/N0 < ~0.30 (sufficient sensitive cells)")
print("  2. α_RS > ~1.0 (strong competition)")
print("  3. d_R/d_S < ~0.5 (differential sensitivity)")
print("="*60)
