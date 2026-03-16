# Computational Tumor Board: Methods

> **Document**: Supplementary Methods for companion paper
> **Version**: 1.0 — incorporates all mentor review feedback
> **Intended section**: Materials and Methods (or Online Methods)

---

## 1. Overview of the CTB Framework

The Computational Tumor Board (CTB) is a four-layer decision-support system that integrates molecular profiling, clonal architecture analysis, evolutionary tumor dynamics modeling, and longitudinal circulating tumor DNA (ctDNA) monitoring into a unified closed-loop framework. The system processes patient genomic and liquid biopsy data to generate model-suggested adaptive treatment strategies. All recommendations are intended for research use and require physician review before any clinical application.

The CTB operates on a single patient data object and produces, at each monitoring timepoint, a structured report containing: (a) a clonality-weighted molecular matching score, (b) an estimate of current tumor state (total burden and clonal composition), (c) forward simulations under candidate treatment policies, and (d) a ranked policy recommendation with safety constraints.

---

## 2. Model Variables and Definitions

### 2.1 Tumor state (latent)

The tumor is modeled as two competing subpopulations:

- **S(t)**: abundance of treatment-sensitive cells at time *t*
- **R(t)**: abundance of treatment-resistant cells at time *t*
- **N(t) = S(t) + R(t)**: total tumor burden (normalized to carrying capacity)

### 2.2 Observable quantities

The CTB uses ctDNA as a dual-input evolutionary sensor, measuring two distinct quantities:

**A. Global tumor burden**

$$\hat{B}(t) = \text{tumor fraction in plasma at time } t$$

This reflects the total amount of tumor-derived DNA in circulation and serves as a proxy for aggregate disease burden. Tumor fraction may be estimated from variant allele frequencies of clonal (truncal) mutations, from methylation-based approaches, or reported directly by clinical ctDNA assays.

**B. Relative clonal composition**

$$\hat{q}_R(t) = \frac{\displaystyle\sum_{i \in \mathcal{R}} \text{VAF}_i(t)}{\displaystyle\sum_{j \in \mathcal{M}} \text{VAF}_j(t)}$$

where:
- $\mathcal{R}$ = set of mutations designated as resistance-associated sentinels
- $\mathcal{M}$ = set of all tracked mutations (sensitive + resistant sentinels)

$\hat{q}_R(t)$ is interpreted as a proxy for the relative proportion of resistant cells among all tumor cells shedding ctDNA. It does not represent exact clonal abundance; rather, it captures the directional trend in clonal competition.

### 2.3 Distinction between the two observation layers

| Quantity | What it measures | Biological meaning | Limitation |
|----------|------------------|--------------------|------------|
| $\hat{B}(t)$ | Total ctDNA signal | How much tumor exists | Does not distinguish who is winning |
| $\hat{q}_R(t)$ | Relative VAF ratio | Who is gaining competitive advantage | VAF ≠ true clone fraction; affected by CN, purity, shedding |

This separation is critical because a declining total burden can coexist with rising resistant fraction — a scenario invisible to burden-only monitoring but detectable through dual-input tracking.

### 2.4 Sentinel mutation assignment

In the current implementation, the user assigns each tracked ctDNA mutation to one of two categories:

- **Sensitive-associated**: mutations expected to be present predominantly in treatment-responsive clones (e.g., truncal driver mutations targeted by the administered therapy)
- **Resistance-associated**: mutations expected to enrich under therapeutic pressure (e.g., known resistance mutations, subclonal variants in alternative pathways)

This assignment is performed prior to analysis and informed by the molecular tumor board's biological interpretation of the patient's genomic landscape. Automated sentinel classification is deferred to future versions.

---

## 3. Clonality-Weighted Molecular Matching Score

### 3.1 Classical matching score

Following the I-PREDICT framework (Sicklick et al., JCO 2026), the classical matching score (MS) is defined as:

$$\text{MS} = \frac{|\{i : m_i = 1\}|}{n}$$

where $n$ is the total number of pathogenic alterations and $m_i = 1$ if alteration $i$ is targeted by at least one administered therapy.

### 3.2 Clonality-weighted extension

We extend the MS to incorporate clonal hierarchy:

$$\text{MS}_w = \frac{\displaystyle\sum_{i=1}^{n} w_i \cdot m_i}{\displaystyle\sum_{i=1}^{n} w_i}$$

where $w_i = \text{CCF}_i$ (cancer cell fraction) for alteration $i$.

This weighting ensures that matching a truncal alteration (CCF ≈ 1.0) contributes more to the score than matching a subclonal alteration (CCF < 0.30), reflecting the biological principle that truncal drivers represent the largest therapeutic target.

### 3.3 Cancer cell fraction estimation

When CCF is not provided by an external tool, we estimate it from variant allele frequency:

$$\text{CCF}_{\text{est}} = \frac{\text{VAF} \times \left(p \cdot \text{CN}_{\text{tumor}} + (1-p) \cdot \text{CN}_{\text{normal}}\right)}{p \times M}$$

where:
- $p$ = tumor purity
- $\text{CN}_{\text{tumor}}$ = total copy number at the variant locus
- $\text{CN}_{\text{normal}}$ = 2 (diploid)
- $M$ = expected allelic multiplicity (default = 1 for heterozygous SNVs)
- CCF is capped at 1.0

This simplified estimator does not account for subclonal copy number events, loss of heterozygosity, or complex ploidy states. For clinical applications, established tools (PyClone-VI, ABSOLUTE, FACETS) should be used. The estimator is sufficient for proof-of-concept demonstration of the clonality-weighting principle.

### 3.4 Clonality classification

Alterations are classified by CCF into three tiers:

| Tier | CCF range | Interpretation |
|------|-----------|----------------|
| Truncal | > 0.60 | Present in majority of tumor cells; highest therapeutic priority |
| Branch | 0.30–0.60 | Present in a substantial subpopulation |
| Subclonal | < 0.30 | Present in a minority; may expand under selective pressure |

---

## 4. Evolutionary Dynamics Model

### 4.1 Lotka-Volterra competition system

Tumor dynamics are modeled using a two-population Lotka-Volterra competition system:

$$\frac{dS}{dt} = r_S \cdot S \cdot \left(1 - \frac{S + \alpha_{SR} \cdot R}{K}\right) - u(t) \cdot d_S \cdot S$$

$$\frac{dR}{dt} = r_R \cdot R \cdot \left(1 - \frac{R + \alpha_{RS} \cdot S}{K}\right) - u(t) \cdot d_R \cdot R$$

| Parameter | Description | Units |
|-----------|-------------|-------|
| $r_S, r_R$ | Intrinsic growth rates | day$^{-1}$ |
| $K$ | Shared carrying capacity | normalized to 1.0 |
| $\alpha_{SR}$ | Competitive effect of R on S | dimensionless |
| $\alpha_{RS}$ | Competitive effect of S on R | dimensionless |
| $d_S$ | Drug-induced death rate for S | day$^{-1}$ |
| $d_R$ | Drug-induced death rate for R | day$^{-1}$; $d_R \ll d_S$ |
| $u(t)$ | Treatment intensity at time $t$ | $\in [0, 1]$ |

The key biological assumption is $d_S \gg d_R$: therapy selectively kills sensitive cells while having minimal effect on resistant cells. The competitive suppression parameter $\alpha_{RS} > 1$ encodes the principle that sensitive cells, when abundant, exert strong competitive pressure on resistant cells — the biological shield that adaptive therapy seeks to preserve.

### 4.2 Treatment intensity function

The treatment intensity $u(t)$ maps clinical dosing to the model:

$$u(t) = \frac{\text{dose}(t)}{\text{dose}_{\text{max}}}$$

where $\text{dose}_{\text{max}}$ is the maximum tolerated dose for the specific agent. Under MTD, $u(t) = 1$ continuously. Under adaptive or intermittent strategies, $u(t) \in \{0, 1\}$ or $u(t) \in [0, 1]$.

### 4.3 Parameterization

For the mCRPC validation case, parameters are derived from Zhang et al. (Nature Communications, 2017):

| Parameter | Value | Source |
|-----------|-------|--------|
| $r_S$ | 0.0278 day$^{-1}$ | LNCaP doubling time, scaled to 10% |
| $r_R$ | 0.0355 day$^{-1}$ | PC-3 doubling time, scaled to 10% |
| $K$ | 1.0 | normalized |
| $\alpha_{SR}$ | 1.0 | baseline symmetric |
| $\alpha_{RS}$ | 1.2 | sensitive cells suppress resistant |
| $d_S$ | 0.015 day$^{-1}$ | calibrated to abiraterone response |
| $d_R$ | 0.0 day$^{-1}$ | fully resistant |

For the HER2+ breast cancer demonstration case, parameters are adapted from preclinical adaptive therapy models (Enriquez-Navas et al., Science Translational Medicine, 2016) and adjusted to produce clinically plausible dynamics.

### 4.4 Numerical integration

The ODE system is solved using the explicit Runge-Kutta method of order 4(5) (Dormand-Prince) as implemented in SciPy's `solve_ivp`, with a maximum step size of 1 day to ensure accurate capture of treatment switching events. Population values are clipped to $[0, \infty)$ at each evaluation to prevent numerical artifacts.

---

## 5. Observation Model and State Update

### 5.1 Mapping ctDNA to model state

At each ctDNA monitoring timepoint $t_k$, the CTB performs the following state update:

**Step 1. Estimate total tumor burden**

$$\hat{N}(t_k) = \frac{\hat{B}(t_k)}{\hat{B}(t_0)} \cdot N(t_0)$$

where $\hat{B}(t_0)$ is the baseline tumor fraction and $N(t_0) = S_0 + R_0$ is the initial modeled burden. This assumes a linear relationship between plasma tumor fraction and total tumor volume — an approximation supported by empirical correlations in metastatic settings but acknowledged as imperfect (e.g., ctDNA shedding rates vary by tumor type and anatomic site).

**Step 2. Estimate clonal composition**

$$\hat{R}(t_k) = \hat{N}(t_k) \cdot \hat{q}_R(t_k)$$

$$\hat{S}(t_k) = \hat{N}(t_k) \cdot \left(1 - \hat{q}_R(t_k)\right)$$

where $\hat{q}_R(t_k)$ is the relative resistant fraction estimated from sentinel mutation VAFs (Section 2.2B).

**Step 3. Quantify observation uncertainty**

The uncertainty in the tumor fraction estimate is approximated by the binomial sampling error of the ctDNA measurement:

$$\sigma_{\hat{B}} = \sqrt{\frac{\hat{B}(t_k) \cdot (1 - \hat{B}(t_k))}{\bar{n}_{\text{reads}}}}$$

where $\bar{n}_{\text{reads}}$ is the mean sequencing depth across tracked loci.

### 5.2 Reliability filters

To avoid reacting to noise, the state update applies two filters:

1. **Limit of detection (LOD)**: If $\hat{B}(t_k) < \text{LOD}$ (default: 0.001), the system reports "below detection" and skips recalibration. The prior model state is carried forward.

2. **Persistence rule**: A change in $\hat{q}_R$ is considered reliable only if the directional trend (rising or falling) is consistent across at least 2 consecutive timepoints. Single-timepoint fluctuations in low-VAF mutations are flagged but do not trigger policy changes.

---

## 6. Policy Evaluation

### 6.1 Candidate treatment policies

The CTB evaluates each candidate policy by simulating the Lotka-Volterra system forward from the current state $(\hat{S}(t_k), \hat{R}(t_k))$ over a defined horizon $T_{\text{horizon}}$ (default: 730 days).

The v1 policy space includes:

| Policy | Description | $u(t)$ |
|--------|-------------|--------|
| MTD | Continuous maximum tolerated dose | $u(t) = 1$ always |
| AT50 | Adaptive: pause at 50% decline from baseline, resume at baseline | Binary, state-dependent |
| AT30 | Adaptive: pause at 30% decline from baseline, resume at baseline | Binary, state-dependent |
| Intermittent 28/14 | 28 days on, 14 days off | Cyclic binary |
| Intermittent 21/7 | 21 days on, 7 days off | Cyclic binary |
| Intermittent 21/14 | 21 days on, 14 days off | Cyclic binary |
| Intermittent 28/28 | 28 days on, 28 days off | Cyclic binary |
| Metronomic 50% | Continuous at 50% dose | $u(t) = 0.5$ always |
| Metronomic 25% | Continuous at 25% dose | $u(t) = 0.25$ always |

### 6.2 Outcome metrics

For each simulated policy, three outcomes are computed:

1. **Time to progression (TTP)**: defined as the first time after nadir at which $N(t) \geq N(t_k)$ — i.e., when tumor burden returns to its value at the time of the simulation. If no progression occurs within $T_{\text{horizon}}$, TTP is set to $T_{\text{horizon}}$.

2. **Cumulative drug exposure**: $\displaystyle\int_{t_k}^{t_k + T_{\text{horizon}}} u(t) \, dt$, representing total treatment intensity over the simulation window.

3. **Resistant dominance at endpoint**: $q_R(t_{\text{end}}) = R(t_{\text{end}}) / N(t_{\text{end}})$, capturing whether the policy leads to a resistant-dominated tumor state.

### 6.3 Policy ranking (physician utility function)

Policies are ranked by a composite utility score:

$$U = \text{TTP}_{\text{norm}} - \lambda \cdot \text{Dose}_{\text{norm}} - \mu \cdot q_R(t_{\text{end}})$$

where:
- $\text{TTP}_{\text{norm}} = \text{TTP} / T_{\text{horizon}}$
- $\text{Dose}_{\text{norm}} = \text{cumulative dose} / T_{\text{horizon}}$
- $\lambda = 0.3$ (dose penalty weight)
- $\mu = 0.2$ (resistant dominance penalty weight)

The interpretation is: the CTB selects the policy that maximizes time to progression while penalizing excessive drug exposure and the emergence of resistant-dominant disease. The weights $\lambda$ and $\mu$ are configurable and may be adjusted based on clinical context (e.g., increasing $\mu$ for tumors with known rapid resistance evolution).

### 6.4 Game-theoretic interpretation

This policy search corresponds to a discretized Stackelberg game:

- The **physician (leader)** selects a treatment policy from the candidate set.
- The **tumor (follower)** responds via its evolutionary trajectory, determined by the ODE system.
- The physician evaluates the tumor's best response to each policy and selects the policy maximizing their utility function.

Unlike analytical Stackelberg solutions (e.g., Pontryagin's minimum principle), the discretized approach is computationally efficient, clinically interpretable, and sufficient for a proof-of-concept demonstration. Continuous-action Stackelberg optimization and reinforcement learning extensions are planned for future versions.

---

## 7. Closed-Loop Simulation

### 7.1 Algorithm

The CTB closed-loop operates as follows:

```
ALGORITHM: CTB Closed-Loop Decision Support

INPUT:
  Patient genomic profile (mutations, CCF estimates)
  Baseline ctDNA (tumor fraction, sentinel VAFs)
  Treatment history
  L-V model parameters (θ)

INITIALIZATION:
  Compute MS_classic and MS_weighted (Section 3)
  Classify clonal architecture (truncal/branch/subclonal)
  Set initial model state: S(0), R(0) from baseline clonal composition
  Run initial policy search → generate baseline recommendation

LOOP (at each ctDNA monitoring timepoint t_k):
  1. OBSERVE
     - Parse tumor fraction B̂(t_k)
     - Parse sentinel mutation VAFs
     - If B̂ < LOD → report "below detection", carry forward prior state

  2. ESTIMATE
     - Compute N̂(t_k) from B̂ ratio to baseline
     - Compute q̂_R(t_k) from sentinel VAF ratio
     - Derive Ŝ(t_k), R̂(t_k)
     - Apply persistence filter (≥2 timepoints for trend confirmation)

  3. COMPARE
     - Compare observed state to previous model prediction
     - Compute prediction error: ε = |N̂(t_k) - N_predicted(t_k)|
     - If ε > threshold → flag model recalibration needed

  4. PREDICT
     - Simulate all candidate policies forward from (Ŝ, R̂) at t_k
     - Compute TTP, cumulative dose, resistant fraction for each

  5. RANK
     - Score each policy by utility function U
     - Apply safety constraints (Section 7.2)

  6. REPORT
     - Output: current state, prediction vs observation comparison,
       top 3 policies with scores, recommended policy, safety flags
     - Visualization: tumor burden timeline with ctDNA overlay,
       clonal composition bar, policy comparison chart
```

### 7.2 Safety constraints

The following constraints are applied after policy ranking and may override the utility-maximizing recommendation:

| Constraint | Condition | Action |
|------------|-----------|--------|
| CNS risk | Patient has CNS metastases or high-risk CNS histology | Exclude all policies with treatment pauses; only MTD or metronomic policies permitted |
| Resistant dominance | $\hat{q}_R(t_k) > 0.50$ | Flag "resistant-dominant tumor — adaptive therapy unlikely to benefit"; recommend MTD or combination strategy |
| Progression signal | $\hat{N}(t_k) > \hat{N}(t_{k-1}) > \hat{N}(t_{k-2})$ (3 consecutive rises) | Flag "sustained progression — consider treatment change" |
| Below LOD | $\hat{B}(t_k) < \text{LOD}$ | Skip recalibration; recommend imaging confirmation; carry forward prior state |

---

## 8. Sensitivity Analysis

To assess the robustness of policy recommendations to parameter uncertainty, each policy evaluation is repeated across a grid of biologically plausible parameter values:

| Parameter | Grid values |
|-----------|-------------|
| $r_S$ | 0.020, 0.028, 0.035 |
| $r_R$ | 0.025, 0.036, 0.045 |
| $\alpha_{RS}$ | 0.8, 1.0, 1.2, 1.5 |
| $d_S$ | 0.010, 0.015, 0.020, 0.030 |
| $R_0 / N_0$ | 0.01, 0.05, 0.10, 0.20 |

This yields 576 parameter configurations. For each configuration, all 9 policies are simulated, producing 5,184 total simulations. We report: (a) mean ± SD of TTP for each policy across configurations, (b) the fraction of configurations in which each policy dominates MTD, and (c) the policy with the highest mean utility.

A policy is considered robustly superior to MTD if it produces longer TTP in ≥80% of parameter configurations.

---

## 9. Validation

### 9.1 mCRPC benchmark (Zhang et al.)

The CTB is validated by reproducing published results from the adaptive abiraterone trial in metastatic castration-resistant prostate cancer (Zhang et al., Nature Communications 2017; eLife 2022):

**Qualitative validation**: AT50 produces longer TTP than MTD (pass/fail).

**Quantitative validation**: The simulated TTP ratio (AT50/MTD) falls within ±20% of the published ratio (33.5 months / 14.3 months ≈ 2.34).

**Drug exposure validation**: AT50 produces less than 50% of the cumulative drug dose of MTD, consistent with the published finding of "less than half."

### 9.2 HER2+ breast cancer demonstration

The CTB is demonstrated on a synthetic case derived from the Perspective paper:

- HER2 amplification (truncal, CCF ≈ 1.0) → trastuzumab
- TP53 R248W (clonal, CCF ≈ 0.79) → no targeted agent
- BRCA2 splice (subclonal, CCF ≈ 0.29) → olaparib

Demonstration goals:
1. MS_classic = 0.67; MS_weighted > MS_classic (truncal matching contributes more)
2. Continuous anti-HER2 therapy leads to competitive release of BRCA2 subclone in simulation
3. Adaptive strategy maintains interclonal competition and delays BRCA2 expansion
4. Closed-loop ctDNA detects rising BRCA2 sentinel VAF and triggers policy re-evaluation

### 9.3 Smoke tests

| Test | Input | Expected output |
|------|-------|-----------------|
| Burden falls, resistant rises | $\hat{B}$ ↓, $\hat{q}_R$ ↑ | Apparent response detected; alert for resistant expansion |
| Burden rises, resistant dominates | $\hat{B}$ ↑, $\hat{q}_R$ > 0.5 | Recommend treatment intensification; flag resistant dominance |
| Burden rises, sensitive still dominant | $\hat{B}$ ↑, $\hat{q}_R$ < 0.3 | Interpret as re-expansion of sensitive clone; adaptive therapy may remain valid |
| ctDNA below LOD | $\hat{B}$ < 0.001 | Skip recalibration; recommend imaging confirmation |
| Missing timepoint | Gap in ctDNA series | Carry forward prior state; increase uncertainty flag |

---

## 10. Software Implementation

The CTB is implemented in Python 3.11+ using the following dependencies:

| Component | Library | Purpose |
|-----------|---------|---------|
| ODE integration | SciPy `solve_ivp` | Lotka-Volterra simulation |
| Numerical computation | NumPy | Array operations |
| Visualization | Plotly | Interactive dashboards |
| Web interface | Streamlit | Prototype UI |
| Data validation | Pydantic | Input schema enforcement |
| Testing | pytest | Unit and integration tests |

Source code is available at [repository URL] under [license]. All analyses are reproducible from the provided parameter files and synthetic patient data.

---

## 11. Limitations

The following limitations are explicitly acknowledged:

1. **CCF estimation is approximate.** The simplified formula used in this prototype does not account for subclonal copy number variation, loss of heterozygosity, or complex ploidy. Clinical implementations should use dedicated tools (PyClone-VI, ABSOLUTE).

2. **VAF trajectories are proxies, not exact measurements of clonal abundance.** VAF is influenced by tumor fraction, copy number, sequencing depth, and shedding heterogeneity. We interpret VAF ratios as indicators of relative clonal trend, not absolute population sizes.

3. **The Lotka-Volterra model is a simplification.** It does not capture spatial heterogeneity, phenotypic plasticity, immune interactions, or pharmacokinetic variability. More complex models (agent-based, reaction-diffusion) may be required for specific clinical scenarios.

4. **The ctDNA-to-model mapping assumes linearity.** The relationship between plasma tumor fraction and in vivo tumor volume is approximate and may not hold in oligometastatic disease, CNS-confined tumors, or tumors with low ctDNA shedding.

5. **Prospective validation is absent.** The CTB framework is validated retrospectively against published data. Clinical implementation requires a prospective trial comparing CTB-guided therapy to standard of care.

6. **Sentinel mutation assignment is manual.** Classification of mutations as sensitive- or resistance-associated relies on expert biological interpretation and is not automated in v1.

---

## 12. Regulatory Statement

This software is a research prototype developed for academic investigation. It is **not** a medical device and has **not** been evaluated or approved by any regulatory authority. All outputs are **model-suggested strategies** intended to support, not replace, clinical judgment. No real patient data should be used without appropriate institutional review board approval and data de-identification procedures.

---

## Key Formulation (for abstract/introduction)

> The CTB models longitudinal ctDNA as a dual-input evolutionary sensor: total tumor fraction informs global disease burden, while mutation-level VAF trajectories provide an approximate readout of relative clonal composition. At each monitoring timepoint, these signals are used to recalibrate the state of a Lotka–Volterra tumor ecosystem model, enabling closed-loop simulation of alternative adaptive treatment policies.
