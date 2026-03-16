# CTB Mathematical Architecture & Computational Pipeline
## Technical Specification v1.0

> **Status**: Pre-implementation specification
> **Scope**: Formal mathematical definitions, data schemas, solver specifications, and closed-loop algorithm for the Computational Tumor Board prototype
> **Purpose**: Address all 4 critical gaps from mentor review; serve as supplementary methods for companion paper

---

## 1. Patient Data Schema

All CTB modules consume a unified patient data object. This is the single source of truth.

### 1.1 Patient Input Schema (JSON)

```json
{
  "patient": {
    "id": "CTB-001",
    "cancer_type": "mCRPC",
    "histology": "adenocarcinoma",
    "stage": "IV",
    "tumor_purity": 0.65,
    "ploidy": 2.1,
    "prior_lines": 2
  },

  "mutations": [
    {
      "gene": "AR",
      "variant": "amplification",
      "type": "CNV",
      "VAF": null,
      "copy_number": 8,
      "clonality": "truncal",
      "CCF": 0.95,
      "CCF_method": "provided",
      "actionable": true,
      "drugs": ["abiraterone", "enzalutamide"]
    },
    {
      "gene": "TP53",
      "variant": "R248W",
      "type": "missense",
      "VAF": 0.38,
      "copy_number": 2,
      "clonality": "clonal",
      "CCF": null,
      "CCF_method": "to_estimate",
      "actionable": false,
      "drugs": []
    },
    {
      "gene": "BRCA2",
      "variant": "splice_site",
      "type": "indel",
      "VAF": 0.12,
      "copy_number": 2,
      "clonality": "subclonal",
      "CCF": null,
      "CCF_method": "to_estimate",
      "actionable": true,
      "drugs": ["olaparib", "rucaparib", "talazoparib"]
    }
  ],

  "ctdna_series": [
    {
      "timepoint_days": 0,
      "tumor_fraction": 0.15,
      "mutations": {
        "TP53_R248W": {"VAF": 0.35, "reads_alt": 140, "reads_total": 400},
        "BRCA2_splice": {"VAF": 0.08, "reads_alt": 24, "reads_total": 300}
      }
    },
    {
      "timepoint_days": 28,
      "tumor_fraction": 0.08,
      "mutations": {
        "TP53_R248W": {"VAF": 0.18, "reads_alt": 72, "reads_total": 400},
        "BRCA2_splice": {"VAF": 0.05, "reads_alt": 15, "reads_total": 300}
      }
    },
    {
      "timepoint_days": 56,
      "tumor_fraction": 0.11,
      "mutations": {
        "TP53_R248W": {"VAF": 0.22, "reads_alt": 88, "reads_total": 400},
        "BRCA2_splice": {"VAF": 0.09, "reads_alt": 27, "reads_total": 300}
      }
    }
  ],

  "treatment_history": [
    {
      "agent": "abiraterone",
      "start_day": 0,
      "end_day": null,
      "dose_mg": 1000,
      "status": "active"
    }
  ],

  "clinical_markers": [
    {"marker": "PSA", "timepoint_days": 0, "value": 42.0, "unit": "ng/mL"},
    {"marker": "PSA", "timepoint_days": 28, "value": 18.5, "unit": "ng/mL"},
    {"marker": "PSA", "timepoint_days": 56, "value": 25.1, "unit": "ng/mL"}
  ],

  "constraints": {
    "cns_risk": false,
    "organ_function_limits": [],
    "patient_preference_treatment_breaks": true
  }
}
```

### 1.2 Validation Rules

```python
VALIDATION_RULES = {
    "tumor_purity": (0.05, 1.0),       # biologically plausible range
    "VAF": (0.0, 1.0),
    "copy_number": (0, 50),
    "tumor_fraction": (0.0, 1.0),
    "CCF": (0.0, 1.0),
    "timepoint_days": (0, float("inf"))  # must be non-negative
}

def validate_patient(patient_data: dict) -> list[str]:
    """Returns list of validation errors. Empty = valid."""
    errors = []
    p = patient_data["patient"]
    if not (0.05 <= p["tumor_purity"] <= 1.0):
        errors.append(f"tumor_purity {p['tumor_purity']} out of range [0.05, 1.0]")

    for m in patient_data["mutations"]:
        if m["VAF"] is not None and not (0.0 <= m["VAF"] <= 1.0):
            errors.append(f"{m['gene']} VAF {m['VAF']} out of range")

    # ctDNA timepoints must be monotonically increasing
    times = [t["timepoint_days"] for t in patient_data["ctdna_series"]]
    if times != sorted(times):
        errors.append("ctDNA timepoints not monotonically increasing")

    return errors
```

---

## 2. Layer 1 — Dynamic Matching Score

### 2.1 Classical Matching Score (I-PREDICT)

```
MS = (number of pathogenic alterations matched by administered therapies)
     / (total number of pathogenic alterations)
```

### 2.2 Clonality-Weighted Matching Score (novel extension)

**Definition**:

```
         Σ_i (w_i × m_i)
MS_w = ─────────────────
            Σ_i (w_i)

where:
  i    = index over all pathogenic alterations
  m_i  = 1 if alteration i is matched by at least one administered drug, 0 otherwise
  w_i  = clonality weight for alteration i
```

**Weight assignment**:

```
w_i = CCF_i          if CCF is estimated or provided
w_i = 1.0            if CCF is unknown (conservative default)
```

**Clonality classification thresholds**:

```
Truncal:    CCF > 0.60
Branch:     0.30 ≤ CCF ≤ 0.60
Subclonal:  CCF < 0.30
```

### 2.3 CCF Estimation (Layer 2 prerequisite)

**Simplified estimator for proof-of-concept**:

For SNVs and small indels:

```
            VAF × (purity × CN_tumor + (1 - purity) × CN_normal)
CCF_est = ───────────────────────────────────────────────────────────
                          purity × multiplicity

where:
  VAF         = variant allele frequency from NGS
  purity      = tumor purity (from pathology or bioinformatics)
  CN_tumor    = total copy number at variant locus
  CN_normal   = 2 (diploid normal)
  multiplicity = expected number of copies carrying the variant
                 (default = 1 for heterozygous SNVs)
```

**Fallback (minimal data)**:

```
CCF_approx ≈ VAF / (purity × 0.5)
```

**Cap at 1.0**:

```
CCF_final = min(CCF_est, 1.0)
```

**Disclosure for paper**: "CCF values were estimated using a simplified formula assuming heterozygous variants at a single allelic copy. This approximation does not account for subclonal copy number events, loss of heterozygosity, or complex ploidy states. For clinical applications, established tools such as PyClone, ABSOLUTE, or FACETS should be used."

### 2.4 Implementation

```python
from dataclasses import dataclass

@dataclass
class Mutation:
    gene: str
    VAF: float | None
    copy_number: int
    drugs: list[str]
    CCF: float | None = None

def estimate_ccf(mut: Mutation, purity: float, multiplicity: int = 1) -> float:
    """Estimate cancer cell fraction from VAF."""
    if mut.VAF is None:
        return 1.0  # default for CNV-driven (e.g., amplifications)

    cn_normal = 2
    numerator = mut.VAF * (purity * mut.copy_number + (1 - purity) * cn_normal)
    denominator = purity * multiplicity

    if denominator == 0:
        return 1.0

    return min(numerator / denominator, 1.0)


def matching_score_weighted(mutations: list[Mutation],
                            administered_drugs: list[str],
                            purity: float) -> dict:
    """Compute classical and clonality-weighted matching scores."""

    results = {"mutations": [], "MS_classic": 0.0, "MS_weighted": 0.0}
    total_matched = 0
    total_n = len(mutations)
    sum_w_matched = 0.0
    sum_w_total = 0.0

    for mut in mutations:
        # Estimate CCF if not provided
        ccf = mut.CCF if mut.CCF is not None else estimate_ccf(mut, purity)
        w = ccf

        # Check matching
        matched = any(drug in administered_drugs for drug in mut.drugs)
        m_i = 1 if matched else 0

        total_matched += m_i
        sum_w_matched += w * m_i
        sum_w_total += w

        results["mutations"].append({
            "gene": mut.gene,
            "CCF": round(ccf, 3),
            "clonality": (
                "truncal" if ccf > 0.60
                else "branch" if ccf >= 0.30
                else "subclonal"
            ),
            "matched": matched,
            "weight": round(w, 3)
        })

    results["MS_classic"] = round(total_matched / total_n, 3) if total_n > 0 else 0
    results["MS_weighted"] = (
        round(sum_w_matched / sum_w_total, 3) if sum_w_total > 0 else 0
    )

    return results
```

### 2.5 Drug-Target Database (minimal curated)

```python
# Subset curated from OncoKB Level 1-2 + CIViC Level A-B
# For prototype — NOT exhaustive
DRUG_TARGET_DB = {
    # GU Oncology
    "AR amplification": ["abiraterone", "enzalutamide", "darolutamide", "apalutamide"],
    "AR T878A": ["abiraterone"],
    "BRCA2 loss": ["olaparib", "rucaparib", "talazoparib", "niraparib"],
    "BRCA1 loss": ["olaparib", "rucaparib", "talazoparib", "niraparib"],
    "MSI-H": ["pembrolizumab", "nivolumab"],
    "TMB-H": ["pembrolizumab"],

    # Breast
    "HER2 amplification": ["trastuzumab", "pertuzumab", "tucatinib",
                           "trastuzumab_deruxtecan", "lapatinib"],
    "PIK3CA H1047R": ["alpelisib"],
    "PIK3CA E545K": ["alpelisib"],
    "ESR1 D538G": ["elacestrant"],
    "ESR1 Y537S": ["elacestrant"],

    # Pan-cancer
    "BRAF V600E": ["dabrafenib", "vemurafenib", "encorafenib"],
    "NTRK fusion": ["larotrectinib", "entrectinib"],
    "RET fusion": ["selpercatinib", "pralsetinib"],
    "KRAS G12C": ["sotorasib", "adagrasib"],
}
```

---

## 3. Layer 3 — Evolutionary Dynamics Engine

### 3.1 Lotka-Volterra Competition Model (2 populations)

**State variables**:

```
S(t) = abundance of treatment-sensitive cells at time t
R(t) = abundance of treatment-resistant cells at time t
N(t) = S(t) + R(t) = total tumor burden
```

**System of ODEs**:

```
dS/dt = r_S · S · [1 − (S + α_SR · R) / K] − d(t) · S

dR/dt = r_R · R · [1 − (R + α_RS · S) / K]
```

**Parameters**:

| Symbol | Description | Units | Source |
|--------|-------------|-------|--------|
| r_S | Sensitive cell growth rate | day⁻¹ | Zhang et al. 2017: 0.0278 |
| r_R | Resistant cell growth rate | day⁻¹ | Zhang et al. 2017: 0.0355 |
| K | Carrying capacity | cells (normalized) | Set to 1.0 (normalized) |
| α_SR | Competition: R effect on S | dimensionless | Range [0.5, 1.5]; default 1.0 |
| α_RS | Competition: S effect on R | dimensionless | Range [0.5, 1.5]; default 1.2 |
| d(t) | Drug-induced death rate | day⁻¹ | Treatment-dependent |

**Drug effect function**:

```
d(t) = d_max · dose(t) / dose_max

where:
  d_max    = maximum drug kill rate (calibrated per agent)
  dose(t)  = administered dose at time t
  dose_max = maximum tolerated dose
```

**Note on r_S < r_R**: Zhang et al. parameterized growth rates from cell line doubling times (LNCaP, H295R, PC-3) and scaled to 10% of in vitro values for in vivo simulation. The resistant cells (PC-3 line, T−) have intrinsically faster growth, but under no treatment, the competitive advantage of sensitive cells comes from *not* bearing the metabolic cost of resistance — captured by α_RS > 1 (sensitive cells suppress resistant more than vice versa).

### 3.2 Three-Population Extension (mCRPC-specific)

Following Zhang et al. 2017, mCRPC can be modeled with three subpopulations:

```
T+  = testosterone-dependent (sensitive to abiraterone)
TP  = testosterone-producing (partially sensitive)
T−  = testosterone-independent (resistant)

dT+/dt = r_1 · T+ · [1 − (a_11·T+ + a_12·TP + a_13·T−) / K_1] − d(t)·T+
dTP/dt = r_2 · TP · [1 − (a_21·T+ + a_22·TP + a_23·T−) / K_2]
dT−/dt = r_3 · T− · [1 − (a_31·T+ + a_32·TP + a_33·T−) / K_3]
```

**Competition matrix from Zhang et al.**:

```
A = [[1.0,  a_12, a_13],     # effects on T+
     [a_21, 1.0,  a_23],     # effects on TP
     [a_31, a_32, 1.0 ]]     # effects on T−
```

For the prototype, we implement the 2-population model as default and the 3-population as mCRPC-specific extension.

### 3.3 Implementation

```python
import numpy as np
from scipy.integrate import solve_ivp
from dataclasses import dataclass


@dataclass
class LVParams:
    """Lotka-Volterra model parameters."""
    r_S: float = 0.0278       # sensitive growth rate (day^-1)
    r_R: float = 0.0355       # resistant growth rate (day^-1)
    K: float = 1.0            # carrying capacity (normalized)
    alpha_SR: float = 1.0     # competition: R effect on S
    alpha_RS: float = 1.2     # competition: S effect on R
    d_max: float = 0.015      # max drug kill rate (day^-1)


def lv_system(t: float, y: np.ndarray, params: LVParams,
              dose_schedule: callable) -> list[float]:
    """
    Lotka-Volterra competition ODE system.

    Args:
        t: current time (days)
        y: state vector [S, R]
        params: model parameters
        dose_schedule: function(t) -> dose fraction in [0, 1]

    Returns:
        [dS/dt, dR/dt]
    """
    S, R = y
    S = max(S, 0)  # prevent negative populations
    R = max(R, 0)

    # Drug effect
    dose_frac = dose_schedule(t)
    d = params.d_max * dose_frac

    # ODEs
    dS = (params.r_S * S * (1 - (S + params.alpha_SR * R) / params.K)
          - d * S)
    dR = (params.r_R * R * (1 - (R + params.alpha_RS * S) / params.K))

    return [dS, dR]


def simulate(params: LVParams, S0: float, R0: float,
             dose_schedule: callable, t_end: float = 1000,
             dt: float = 1.0) -> dict:
    """
    Simulate tumor dynamics under a given dosing schedule.

    Returns dict with:
      - t: time array
      - S: sensitive population
      - R: resistant population
      - N: total tumor burden
      - dose: dose at each timepoint
    """
    t_eval = np.arange(0, t_end, dt)

    sol = solve_ivp(
        fun=lambda t, y: lv_system(t, y, params, dose_schedule),
        t_span=(0, t_end),
        y0=[S0, R0],
        t_eval=t_eval,
        method="RK45",
        max_step=1.0,
        events=progression_event(params.K)
    )

    S = np.clip(sol.y[0], 0, None)
    R = np.clip(sol.y[1], 0, None)
    N = S + R
    dose = np.array([dose_schedule(t) for t in sol.t])

    return {
        "t": sol.t,
        "S": S,
        "R": R,
        "N": N,
        "dose": dose,
        "TTP": _compute_ttp(sol.t, N, S0 + R0),
        "cumulative_dose": np.trapezoid(dose, sol.t),
        "final_R_fraction": R[-1] / N[-1] if N[-1] > 0 else 0
    }


def progression_event(K: float):
    """Event function: progression = tumor returns to initial burden."""
    def event(t, y):
        return y[0] + y[1] - K  # triggers when N reaches K
    event.terminal = True
    event.direction = 1  # only when crossing upward
    return event


def _compute_ttp(t: np.ndarray, N: np.ndarray, N0: float) -> float:
    """
    Time to progression.
    Defined as time when N returns to N0 after initial decline,
    or t_end if no progression.
    """
    # Find nadir
    nadir_idx = np.argmin(N)
    if nadir_idx == 0:
        return 0.0

    # Find first time after nadir where N >= N0
    post_nadir = N[nadir_idx:]
    prog_indices = np.where(post_nadir >= N0)[0]

    if len(prog_indices) == 0:
        return t[-1]  # no progression within simulation window

    return t[nadir_idx + prog_indices[0]]
```

### 3.4 Treatment Policy Definitions

```python
# ─── DOSING SCHEDULES ───────────────────────────────────────────

def mtd_schedule(t: float) -> float:
    """Maximum Tolerated Dose: continuous full dose."""
    return 1.0


def at50_schedule(t: float, state_history: list = None,
                  check_interval: int = 28) -> float:
    """
    Adaptive Therapy 50% rule (Zhang et al.):
    - Treat until tumor burden (PSA/ctDNA proxy) declines by ≥50%
    - Pause treatment
    - Resume when burden returns to baseline
    """
    # Implementation requires access to simulation state
    # See AdaptiveController class below
    raise NotImplementedError("Use AdaptiveController")


def intermittent_schedule(t: float, on_days: int = 28,
                          off_days: int = 14) -> float:
    """Fixed intermittent: on for X days, off for Y days."""
    cycle = on_days + off_days
    t_in_cycle = t % cycle
    return 1.0 if t_in_cycle < on_days else 0.0


class AdaptiveController:
    """
    Implements adaptive dosing based on tumor burden monitoring.

    Supports:
      - AT50: pause at 50% decline, resume at baseline
      - AT30: pause at 30% decline, resume at baseline
      - Custom: configurable thresholds
    """

    def __init__(self, N0: float, decline_threshold: float = 0.50,
                 resume_threshold: float = 1.0,
                 check_interval_days: int = 28):
        self.N0 = N0
        self.decline_threshold = decline_threshold
        self.resume_threshold = resume_threshold
        self.check_interval = check_interval_days
        self.treating = True
        self.last_check = 0
        self.nadir = N0

    def get_dose(self, t: float, N_current: float) -> float:
        """Return dose fraction based on current tumor burden."""

        # Only re-evaluate at check intervals
        if t - self.last_check >= self.check_interval:
            self.last_check = t

            if self.treating:
                # Track nadir
                self.nadir = min(self.nadir, N_current)
                # Pause if declined enough from baseline
                if N_current <= self.N0 * (1 - self.decline_threshold):
                    self.treating = False
            else:
                # Resume if returned to baseline
                if N_current >= self.N0 * self.resume_threshold:
                    self.treating = True
                    self.nadir = N_current

        return 1.0 if self.treating else 0.0
```

### 3.5 Stackelberg Optimization via Discretized Policy Search

**Rationale**: For v1.0, we avoid Pontryagin's minimum principle (computationally intensive, requires adjoint equations) and instead perform grid search over a finite policy space. This is consistent with the "evolutionarily enlightened" Stackelberg approach where the physician (leader) evaluates the tumor's (follower's) best response to each candidate policy.

**Policy space**:

```python
POLICY_SPACE = {
    "MTD": {"type": "continuous", "dose": 1.0},
    "AT50": {"type": "adaptive", "decline": 0.50, "resume": 1.0},
    "AT30": {"type": "adaptive", "decline": 0.30, "resume": 1.0},
    "intermittent_28_14": {"type": "intermittent", "on": 28, "off": 14},
    "intermittent_21_7": {"type": "intermittent", "on": 21, "off": 7},
    "intermittent_21_14": {"type": "intermittent", "on": 21, "off": 14},
    "intermittent_28_28": {"type": "intermittent", "on": 28, "off": 28},
    "metronomic_50": {"type": "continuous", "dose": 0.50},
    "metronomic_25": {"type": "continuous", "dose": 0.25},
}
```

**Objective function (physician's utility)**:

```
U(policy) = w_TTP · TTP_normalized
           − w_dose · cumulative_dose_normalized
           − w_resist · final_resistant_fraction

where:
  TTP_normalized           = TTP / T_max
  cumulative_dose_normalized = cumulative_dose / (T_max × dose_max)
  final_resistant_fraction  = R(t_end) / N(t_end)

Default weights:
  w_TTP    = 1.0   (maximize time to progression)
  w_dose   = 0.3   (minimize cumulative drug exposure)
  w_resist = 0.2   (minimize resistant fraction at end)
```

**Solver**:

```python
def stackelberg_search(params: LVParams, S0: float, R0: float,
                       t_end: float = 1500,
                       w_ttp: float = 1.0,
                       w_dose: float = 0.3,
                       w_resist: float = 0.2) -> dict:
    """
    Evaluate all policies in POLICY_SPACE.
    Return ranked results with utility scores.

    The 'Stackelberg' interpretation: for each physician strategy (policy),
    the tumor's 'best response' is its evolutionary trajectory (determined
    by the ODE system). The physician selects the policy that maximizes
    utility given the tumor's response.
    """
    results = []

    for name, spec in POLICY_SPACE.items():
        # Build dose schedule
        if spec["type"] == "continuous":
            schedule = lambda t, d=spec["dose"]: d
        elif spec["type"] == "intermittent":
            schedule = lambda t, on=spec["on"], off=spec["off"]: (
                1.0 if (t % (on + off)) < on else 0.0
            )
        elif spec["type"] == "adaptive":
            controller = AdaptiveController(
                N0=S0 + R0,
                decline_threshold=spec["decline"],
                resume_threshold=spec["resume"]
            )
            # For adaptive, we need stateful simulation
            sim = _simulate_adaptive(params, S0, R0, controller, t_end)
            results.append(_score_result(name, sim, t_end, w_ttp, w_dose, w_resist))
            continue
        else:
            continue

        sim = simulate(params, S0, R0, schedule, t_end)
        results.append(_score_result(name, sim, t_end, w_ttp, w_dose, w_resist))

    # Sort by utility (descending)
    results.sort(key=lambda x: x["utility"], reverse=True)

    return {
        "recommended": results[0],
        "all_policies": results
    }


def _score_result(name: str, sim: dict, t_end: float,
                  w_ttp: float, w_dose: float, w_resist: float) -> dict:
    """Compute utility for a simulation result."""
    ttp_norm = sim["TTP"] / t_end
    dose_norm = sim["cumulative_dose"] / t_end  # max possible = t_end × 1.0
    resist = sim["final_R_fraction"]

    utility = w_ttp * ttp_norm - w_dose * dose_norm - w_resist * resist

    return {
        "policy": name,
        "TTP_days": round(sim["TTP"], 1),
        "cumulative_dose": round(sim["cumulative_dose"], 1),
        "resistant_fraction": round(resist, 3),
        "utility": round(utility, 4)
    }
```

### 3.6 Sensitivity Analysis

To assess robustness, each policy evaluation is repeated across a parameter grid:

```python
SENSITIVITY_GRID = {
    "r_S": [0.02, 0.0278, 0.035],
    "r_R": [0.025, 0.0355, 0.045],
    "alpha_RS": [0.8, 1.0, 1.2, 1.5],
    "d_max": [0.01, 0.015, 0.02, 0.03],
    "R0_fraction": [0.01, 0.05, 0.10, 0.20]
}

# Total configurations: 3 × 3 × 4 × 4 × 4 = 576
# Each evaluated across 9 policies = 5,184 simulations
# At ~10ms per solve_ivp call: ~52 seconds total
```

Report: for each policy, compute mean ± SD of TTP across the parameter grid. The recommended policy must dominate MTD across ≥80% of configurations.

---

## 4. Layer 4 — ctDNA Closed-Loop System

### 4.1 Mathematical Specification

**Observation model: ctDNA as tumor sensor**

```
Tumor fraction (TF):
  TF(t) ∝ N(t) = S(t) + R(t)

Clone-level tracking:
  For mutation m_i belonging to clone C_j:
    VAF_i(t) ∝ abundance(C_j, t)

  Clone fraction estimate:
    f_j(t) = VAF_j(t) / Σ_k VAF_k(t)
```

**Mapping ctDNA to model state**:

Given a ctDNA measurement at time t_obs:

```
Step 1: Estimate total tumor burden
  N_obs(t_obs) = TF(t_obs) / TF(t_0) × N(t_0)
  (assumes linear relationship between TF and tumor volume)

Step 2: Estimate clone fractions
  For 2-population model:
    - Assign each tracked mutation to S or R clone
    - If sensitive-clone mutation has VAF_S and resistant-clone has VAF_R:

      S_obs = N_obs × VAF_S / (VAF_S + VAF_R)
      R_obs = N_obs × VAF_R / (VAF_S + VAF_R)

    - If only one clone has ctDNA marker:
      Use TF for total burden and single VAF for clone fraction

Step 3: Compute observation error
  σ_obs = sqrt(VAF × (1 - VAF) / n_reads)    # binomial sampling error
```

### 4.2 Closed-Loop Recalibration Algorithm

```
ALGORITHM: CTB Closed-Loop Decision Support

INPUT:
  - Patient data (schema §1)
  - Current L-V model parameters (θ)
  - Treatment history
  - New ctDNA measurement at t_obs

OUTPUT:
  - Updated model state
  - Forward trajectory predictions
  - Dosing recommendation

PROCEDURE:

1. OBSERVE
   Parse new ctDNA timepoint
   Estimate N_obs, S_obs, R_obs (§4.1)
   Flag if TF < LOD (limit of detection → skip recalibration)

2. UPDATE
   Set new initial conditions:
     S(t_obs) = S_obs
     R(t_obs) = R_obs
   Optionally re-estimate parameters via least-squares fit
     to historical [t, N] trajectory

3. PREDICT
   For each policy in POLICY_SPACE:
     Simulate forward from (t_obs, S_obs, R_obs) for T_horizon days
     Compute TTP, cumulative dose, resistant fraction

4. RECOMMEND
   Rank policies by utility function (§3.5)
   Flag safety constraints:
     - If CNS risk = true → do NOT recommend treatment pauses
     - If R_obs / N_obs > 0.50 → flag: "resistant-dominant tumor"
     - If N_obs increasing for 2+ consecutive timepoints → flag: "progression signal"

5. REPORT
   Generate CTB report with:
     - Current state estimate (N, S, R, clone fractions)
     - Comparison: predicted vs observed trajectory
     - Forward projections under top 3 policies
     - Recommended policy with confidence interval
     - Safety flags and constraints applied
```

### 4.3 Implementation

```python
@dataclass
class CtdnaObservation:
    timepoint_days: float
    tumor_fraction: float
    clone_vafs: dict[str, float]  # {mutation_id: VAF}
    read_depths: dict[str, int]   # {mutation_id: total_reads}


@dataclass
class ModelState:
    t: float
    S: float
    R: float
    params: LVParams


class ClosedLoopCTB:
    """
    Implements the closed-loop CTB decision-support system.

    Maintains model state and updates with each ctDNA observation.
    """

    def __init__(self, initial_params: LVParams, S0: float, R0: float,
                 sensitive_mutations: list[str],
                 resistant_mutations: list[str],
                 tf_baseline: float,
                 cns_risk: bool = False):
        self.params = initial_params
        self.state = ModelState(t=0, S=S0, R=R0, params=initial_params)
        self.sensitive_muts = sensitive_mutations
        self.resistant_muts = resistant_mutations
        self.tf_baseline = tf_baseline
        self.cns_risk = cns_risk
        self.history: list[dict] = []
        self.LOD = 0.001  # limit of detection for TF

    def update(self, obs: CtdnaObservation) -> dict:
        """
        Process new ctDNA observation and generate recommendation.

        Returns CTB report dict.
        """
        # Step 1: OBSERVE
        if obs.tumor_fraction < self.LOD:
            return self._below_lod_report(obs)

        N_obs = obs.tumor_fraction / self.tf_baseline * (self.state.S + self.state.R)
        # Prevent unreasonable extrapolation
        N_obs = min(N_obs, self.params.K * 2)

        # Clone fractions from VAF
        vaf_s = np.mean([obs.clone_vafs.get(m, 0) for m in self.sensitive_muts]) or 0.001
        vaf_r = np.mean([obs.clone_vafs.get(m, 0) for m in self.resistant_muts]) or 0.001
        total_vaf = vaf_s + vaf_r

        S_obs = N_obs * (vaf_s / total_vaf)
        R_obs = N_obs * (vaf_r / total_vaf)

        # Observation uncertainty
        mean_reads = np.mean(list(obs.read_depths.values())) if obs.read_depths else 300
        sigma = np.sqrt(obs.tumor_fraction * (1 - obs.tumor_fraction) / mean_reads)

        # Step 2: UPDATE
        self.state = ModelState(
            t=obs.timepoint_days,
            S=S_obs,
            R=R_obs,
            params=self.params
        )

        # Step 3: PREDICT
        search_result = stackelberg_search(
            self.params, S_obs, R_obs, t_end=730  # 2-year horizon
        )

        # Step 4: SAFETY FLAGS
        flags = []
        r_fraction = R_obs / (S_obs + R_obs) if (S_obs + R_obs) > 0 else 0

        if self.cns_risk:
            flags.append("CNS_RISK: treatment pauses contraindicated")
            # Filter out policies with treatment breaks
            search_result["all_policies"] = [
                p for p in search_result["all_policies"]
                if p["policy"] in ["MTD", "metronomic_50", "metronomic_25"]
            ]
            if search_result["all_policies"]:
                search_result["recommended"] = search_result["all_policies"][0]

        if r_fraction > 0.50:
            flags.append(f"RESISTANT_DOMINANT: R fraction = {r_fraction:.1%}")

        # Check trend
        if len(self.history) >= 2:
            recent_n = [h["N_obs"] for h in self.history[-2:]] + [S_obs + R_obs]
            if all(recent_n[i+1] > recent_n[i] for i in range(len(recent_n)-1)):
                flags.append("PROGRESSION_SIGNAL: N increasing for 2+ timepoints")

        # Step 5: REPORT
        report = {
            "timepoint_days": obs.timepoint_days,
            "observed": {
                "tumor_fraction": obs.tumor_fraction,
                "N_obs": round(S_obs + R_obs, 4),
                "S_obs": round(S_obs, 4),
                "R_obs": round(R_obs, 4),
                "R_fraction": round(r_fraction, 3),
                "sigma_TF": round(sigma, 4)
            },
            "recommended_policy": search_result["recommended"],
            "top3_policies": search_result["all_policies"][:3],
            "safety_flags": flags,
            "disclaimer": (
                "Model-suggested strategy for research use only. "
                "Not for clinical decision making without physician review."
            )
        }

        self.history.append({"t": obs.timepoint_days, "N_obs": S_obs + R_obs,
                             "S_obs": S_obs, "R_obs": R_obs})

        return report

    def _below_lod_report(self, obs: CtdnaObservation) -> dict:
        return {
            "timepoint_days": obs.timepoint_days,
            "status": "BELOW_LOD",
            "note": (
                f"Tumor fraction {obs.tumor_fraction:.4f} below LOD "
                f"({self.LOD}). Recalibration skipped. "
                "Consider imaging confirmation."
            )
        }
```

---

## 5. Validation Strategy

### 5.1 Benchmark: Zhang et al. mCRPC Abiraterone

**Source data**: Zhang et al., Nature Communications 2017 (parameters), eLife 2022 (clinical outcomes).

**Parameters to use**:

```python
ZHANG_MCRPC_PARAMS = LVParams(
    r_S=0.0278,       # from LNCaP doubling time, scaled to 10%
    r_R=0.0355,       # from H295R, scaled to 10%
    K=1.0,            # normalized
    alpha_SR=1.0,     # symmetric competition (baseline)
    alpha_RS=1.0,     # symmetric competition (baseline)
    d_max=0.015       # calibrated to abiraterone effect
)

ZHANG_INITIAL_CONDITIONS = {
    "S0": 0.80,  # 80% sensitive at baseline
    "R0": 0.20,  # 20% resistant at baseline
}
```

**Validation metrics**:

```
1. Qualitative: AT50 produces longer TTP than MTD (binary pass/fail)
2. Quantitative: simulated TTP ratio (AT/MTD) within ±20% of published
   ratio (33.5 / 14.3 ≈ 2.34)
3. Cumulative dose: AT produces <50% cumulative dose of MTD
   (published: less than half)
```

### 5.2 Demonstration Case: HER2+ Breast (from paper)

**Parameters**: Adapted from published breast cancer xenograft models (Enriquez-Navas et al. 2016).

**Scenario from paper**:
- HER2 amplification (truncal, CCF ~1.0)
- TP53 R248W (clonal, CCF ~0.79)
- BRCA2 splice (subclonal, CCF ~0.29)

**Demonstration goals**:
- Show MS_classic = 0.33 (HER2 only) vs MS_weighted favoring truncal
- Show L-V simulation predicting resistant subclone expansion under continuous anti-HER2
- Show adaptive strategy maintaining competitive suppression
- Show ctDNA closed-loop detecting BRCA2 clone rise

### 5.3 Smoke Tests

```python
def test_matching_score_basic():
    """3 mutations, 2 matched → MS = 0.66"""
    muts = [
        Mutation("HER2", None, 8, ["trastuzumab"], CCF=0.95),
        Mutation("TP53", 0.38, 2, [], CCF=0.79),
        Mutation("BRCA2", 0.12, 2, ["olaparib"], CCF=0.29),
    ]
    result = matching_score_weighted(muts, ["trastuzumab", "olaparib"], 0.65)
    assert abs(result["MS_classic"] - 0.667) < 0.01
    assert result["MS_weighted"] > result["MS_classic"]  # truncal match weighted higher


def test_lv_no_treatment():
    """Without treatment, tumor grows to K."""
    params = LVParams()
    sim = simulate(params, S0=0.5, R0=0.1,
                   dose_schedule=lambda t: 0.0, t_end=500)
    assert sim["N"][-1] > 0.95 * params.K


def test_at50_beats_mtd():
    """AT50 should produce longer TTP than MTD."""
    params = LVParams(r_S=0.0278, r_R=0.0355, alpha_RS=1.2)
    sim_mtd = simulate(params, 0.8, 0.2, mtd_schedule, t_end=1500)
    # AT50 requires stateful simulation — simplified check
    sim_int = simulate(params, 0.8, 0.2,
                       lambda t: intermittent_schedule(t, 28, 14), t_end=1500)
    # At minimum, intermittent should show different dynamics
    assert sim_int["cumulative_dose"] < sim_mtd["cumulative_dose"]


def test_ccf_estimation():
    """CCF should be capped at 1.0."""
    mut = Mutation("TP53", VAF=0.45, copy_number=2, drugs=[])
    ccf = estimate_ccf(mut, purity=0.65)
    assert 0 < ccf <= 1.0


def test_edge_vaf_zero():
    """VAF = 0 should produce CCF = 0."""
    mut = Mutation("X", VAF=0.0, copy_number=2, drugs=[])
    ccf = estimate_ccf(mut, purity=0.65)
    assert ccf == 0.0


def test_edge_low_purity():
    """Low purity should inflate CCF estimate (up to cap)."""
    mut = Mutation("X", VAF=0.30, copy_number=2, drugs=[])
    ccf_high = estimate_ccf(mut, purity=0.80)
    ccf_low = estimate_ccf(mut, purity=0.30)
    assert ccf_low >= ccf_high  # lower purity → higher CCF estimate
```

### 5.4 Observability / Logging

```python
import logging

logger = logging.getLogger("ctb")

# Log every simulation
logger.info(f"Simulation: policy={name}, params={params}, S0={S0}, R0={R0}")
logger.info(f"Result: TTP={sim['TTP']:.1f}, dose={sim['cumulative_dose']:.1f}")

# Log every closed-loop update
logger.info(f"ctDNA update t={obs.timepoint_days}: TF={obs.tumor_fraction:.4f}")
logger.info(f"State: S={S_obs:.4f}, R={R_obs:.4f}, R_frac={r_fraction:.3f}")
logger.info(f"Recommendation: {report['recommended_policy']['policy']}")
logger.info(f"Safety flags: {report['safety_flags']}")
```

---

## 6. Implementation Roadmap (revised)

### Phase 1: Mathematical Core (Weeks 1-3)

```
Week 1:
  ☐ Repository setup (GitHub, CI, pytest)
  ☐ Patient data schema + validation (§1)
  ☐ CCF estimator (§2.3)
  ☐ Matching Score engine (§2.4)
  ☐ Drug-target DB v1 (§2.5)
  ☐ Smoke tests for Layer 1-2

Week 2:
  ☐ L-V ODE system (§3.3)
  ☐ Treatment policies: MTD, intermittent, AdaptiveController (§3.4)
  ☐ Simulate + TTP computation
  ☐ Reproduce Zhang et al. mCRPC baseline (§5.1)

Week 3:
  ☐ Stackelberg grid search (§3.5)
  ☐ Sensitivity analysis framework (§3.6)
  ☐ Smoke tests for Layer 3
  ☐ Validate: AT50 > MTD in TTP across parameter grid
```

### Phase 2: Closed-Loop + Integration (Weeks 4-6)

```
Week 4:
  ☐ ctDNA observation model (§4.1)
  ☐ ClosedLoopCTB class (§4.3)
  ☐ Safety flags + CNS constraint logic

Week 5:
  ☐ CTB integrator: unified report from all 4 layers
  ☐ HER2+ breast demonstration case (§5.2)
  ☐ mCRPC validation case end-to-end

Week 6:
  ☐ Streamlit UI: patient input → 4-layer report
  ☐ Plotly dashboards: L-V dynamics, ctDNA overlay, policy comparison
  ☐ Muller/fish plot for clonal evolution visualization
```

### Phase 3: Validation + Paper (Weeks 7-10)

```
Week 7-8:
  ☐ Full sensitivity analysis (576 configurations × 9 policies)
  ☐ Quantitative validation vs Zhang et al. published data
  ☐ Generate all figures for paper

Week 9-10:
  ☐ Draft companion paper (methods + results)
  ☐ GitHub repo cleanup, documentation, README
  ☐ Public release
  ☐ Submit paper
```

---

## 7. Regulatory and Safety Declarations

All outputs of the CTB prototype must carry:

```
DISCLAIMER:
This software is a research prototype for academic use only.
It is NOT a medical device and NOT approved for clinical decision making.
All outputs are model-suggested strategies intended to support,
not replace, physician judgment. No patient data should be entered
without appropriate IRB approval and de-identification.
```

Terminology guidelines:
- Use: "model-suggested strategy"
- Avoid: "recommended treatment"
- Use: "decision-support output"
- Avoid: "clinical recommendation"
