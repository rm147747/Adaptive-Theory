"""
Computational Tumor Board (CTB) — Evolutionary Modeling Framework

A four-layer decision-support system integrating molecular matching,
clonal architecture, evolutionary dynamics, and ctDNA monitoring
for adaptive cancer therapy.

Author: Raphael Brandão, MD PhD(c)
License: MIT
"""

from .lotka_volterra import LVParams, simulate_euler, simulate_ivp, compute_ttp
from .policies import POLICY_SPACE, mtd_policy, adaptive_policy, intermittent_policy, metronomic_policy
from .matching_score import Mutation, compute_matching_scores, estimate_ccf
from .optimizer import ctb_select_policy, evaluate_policy

__version__ = "0.1.0"
__author__ = "Raphael Brandão"
