# 9. Partially observed control interpretation

The CTB decision framework can be interpreted as a model-based controller for a partially observed evolutionary system.

## Latent state

The true tumor ecosystem state at time *t* is represented by:

$$x_t = (S(t),\; R(t))$$

consisting of the abundances of therapy-sensitive and therapy-resistant clonal populations. This state is not directly measurable in the clinic.

## Actions

At each decision timepoint, the clinician selects a treatment action:

$$a_t \in \mathcal{A}$$

where $\mathcal{A}$ is the set of candidate treatment policies (e.g., MTD, AT50, intermittent schedules).

## Transition dynamics

The tumor evolves in response to the selected action according to:

$$x_{t+1} \sim P(x_{t+1} \mid x_t, a_t)$$

In the CTB framework, this transition is approximated deterministically by the Lotka–Volterra competition system described in Section 4. A stochastic extension incorporating demographic noise or parameter uncertainty is deferred to future work.

## Observations

The clinician does not observe $x_t$ directly. Instead, longitudinal ctDNA provides noisy, partial observations:

$$o_t = \left(\hat{B}(t),\; \hat{q}_R(t)\right)$$

where $\hat{B}(t)$ is the estimated tumor burden (from plasma tumor fraction) and $\hat{q}_R(t)$ is the inferred resistant clone fraction (from sentinel mutation VAF ratios). These observations follow an implicit observation model:

$$o_t \sim P(o_t \mid x_t)$$

in which measurement noise arises from sequencing depth, ctDNA shedding variability, and the imperfect relationship between VAF and true clonal abundance (Section 2.2).

## Belief-state approximation

At each ctDNA monitoring timepoint, the CTB reconstructs an approximate belief about the current tumor state:

$$\hat{x}_t = \left(\hat{S}(t),\; \hat{R}(t)\right)$$

derived from the observation-to-state mapping described in Section 5. This point estimate serves as the initial condition for forward simulation of candidate treatment policies. In the language of partially observed Markov decision processes (POMDPs), this corresponds to a maximum-likelihood belief-state update — a computationally tractable approximation to full Bayesian belief tracking.

## Policy selection

Given the estimated state $\hat{x}_t$, the clinician selects the action that maximizes the composite utility function:

$$a_t^* = \arg\max_{a \in \mathcal{A}} \; U(a \mid \hat{x}_t)$$

where

$$U = \text{TTP}_{\text{norm}} - \lambda \cdot \text{Dose}_{\text{norm}} - \mu \cdot q_R(t_{\text{end}})$$

balancing time to progression, cumulative drug exposure, and resistant clone dominance (Section 6.3).

## Interpretation

Under this framing, the CTB operates as a partially observed sequential decision system in which:

- the tumor ecosystem is the **environment** with latent dynamics
- ctDNA measurements are **observations** providing imperfect windows into the evolutionary state
- the oncologist is the **agent** selecting actions to maximize long-term patient utility
- the Lotka–Volterra model is the **internal world model** used to predict consequences of candidate actions

This interpretation is conceptually consistent with the POMDP framework widely used in sequential decision-making under uncertainty. However, the current CTB implementation uses deterministic forward simulation and point-estimate state reconstruction rather than full belief-state tracking or stochastic policy optimization. The POMDP framing is presented here as an organizing principle that clarifies the information structure of the problem and motivates future extensions, including:

- Bayesian state estimation with explicit observation noise models
- Robust policy optimization under parameter uncertainty
- Integration of reinforcement learning for continuous policy improvement from accumulated clinical data

These extensions would allow the CTB to move from a deterministic model-based controller toward a probabilistic decision system capable of quantifying confidence in its recommendations — a property essential for clinical adoption.

---

*Note on positioning*: This subsection is intended to follow the closed-loop monitoring algorithm (Section 7) and precede the sensitivity analysis (Section 8) in the final manuscript. It bridges the current deterministic implementation with the broader decision-theoretic framework, providing a conceptual roadmap for v2.0 without overclaiming the current prototype's capabilities.
