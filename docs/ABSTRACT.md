# CTB Companion Paper: Abstract and Central Claim

---

## Central claim (one sentence)

> **Precision oncology treats tumors as static genomic targets; we introduce a computational tumor board that reframes cancer therapy as dynamic control of an evolving ecosystem under partial observation, using ctDNA as a real-time evolutionary sensor.**

---

## Short claim (for abstract/grants/talks)

> **The CTB converts precision oncology from static mutation matching into closed-loop evolutionary control guided by longitudinal ctDNA.**

---

## Abstract

**Background.** Precision oncology has demonstrated that matching therapies to molecular alterations improves outcomes, as shown by the I-PREDICT trial where the degree of biomarker matching correlated linearly with survival. However, current frameworks treat the tumor as a static molecular portrait, ignoring that cancer is a dynamically evolving ecosystem of competing clonal populations. Concurrently, adaptive therapy trials have provided proof-of-concept that evolution-informed treatment strategies can prolong time to progression. These paradigms remain disconnected: precision oncology optimizes what to target, while evolutionary therapy optimizes when and how to treat.

**Methods.** We developed the Computational Tumor Board (CTB), a four-layer decision-support framework that integrates: (1) clonality-weighted molecular matching, extending the I-PREDICT matching score to prioritize truncal over subclonal alterations; (2) clonal hierarchy estimation from variant allele frequencies; (3) evolutionary dynamics simulation using a Lotka–Volterra competition model with Stackelberg game-theoretic policy optimization; and (4) closed-loop monitoring via serial circulating tumor DNA (ctDNA), which serves as a dual-input evolutionary sensor providing both global tumor burden and relative clonal composition. At each ctDNA timepoint, the system recalibrates the tumor state and re-evaluates candidate treatment policies. The framework can be interpreted as a model-based controller for a partially observed evolutionary system.

**Results.** Using parameters derived from published adaptive therapy trials in metastatic castration-resistant prostate cancer, the CTB-optimized policy (selected from a candidate space of nine strategies via a composite utility function) achieved time to progression exceeding 1,500 days with less than 1% resistant clone fraction, compared to 755 days under maximum tolerated dose (MTD) with 100% resistance. Cumulative drug exposure was reduced by approximately 50%. Systematic evaluation of failure modes identified three prerequisites for evolutionary control: sufficient sensitive cell population at baseline, strong interclonal competition, and differential drug sensitivity between clones. Closed-loop ctDNA monitoring detected emergent resistance approximately 180 days before model-predicted progression in a simulated clinical scenario with acquired resistance mutation.

**Conclusions.** The CTB provides the first integrated, open-source framework operationalizing the concept of the oncologist as a strategic player in the evolutionary game between competing tumor subclones. By converting ctDNA from a static biomarker into a longitudinal evolutionary sensor, the framework enables adaptive therapeutic strategies that maintain competitive suppression of resistant clones. Prospective validation is required, and the current implementation uses deterministic simulation with point-estimate state reconstruction. The CTB is proposed as decision support, not decision replacement, and represents a candidate framework for the next generation of evolution-informed precision oncology.

**Keywords:** computational tumor board; evolutionary game theory; adaptive therapy; circulating tumor DNA; Lotka–Volterra; Stackelberg game; precision oncology; clonal dynamics

---

## Closing sentence of introduction (the sentence that defines acceptance)

> Together, these components establish a computational tumor board that enables adaptive cancer therapy through closed-loop evolutionary control informed by longitudinal ctDNA measurements — converting the oncologist from a passive executor of fixed protocols into an active strategic player in the evolutionary game between competing tumor subclones.

---

## Figure sequence

| Figure | What it proves | Claim component |
|--------|---------------|-----------------|
| Figure 1 | CTB framework (4 layers + closed-loop + Stackelberg) | This is the system |
| Figure 2 | CTB prolongs control, reduces resistance, halves drug exposure | It works |
| Figure 3 | Three prerequisite failures (high R₀, weak competition, low selectivity) | We know when it doesn't work |
| Figure 4 | ctDNA detects model-reality divergence 180 days early | The sensor works |

---

## Target journals (ranked by fit)

1. **npj Precision Oncology** — computational framework + precision oncology + open-source. Perfect fit for companion paper.
2. **JCO Clinical Cancer Informatics** — clinical decision support + computational oncology. Strong fit.
3. **Cancer Research Communications** — novel computational framework with biological validation. Good fit.
4. **Cancer Research** — if Figure 2/3 validation is expanded with additional tumor types. Stretch target.

---

## The one thing the editor needs to understand in 10 seconds

The CTB is the first software that connects molecular matching → clonal hierarchy → evolutionary simulation → ctDNA monitoring in a single closed loop.

Everything else published does one piece. Nobody has connected all four.

That's the paper.
