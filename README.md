# Computational Tumor Board (CTB)

**Evolutionary Modeling Framework for Adaptive Cancer Therapy**

[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://python.org)

> **Precision oncology treats tumors as static genomic targets. The CTB reframes cancer therapy as dynamic control of an evolving ecosystem under partial observation.**

## Overview

The Computational Tumor Board (CTB) is a four-layer decision-support framework that integrates:

1. **Genomic targeting** — Clonality-weighted molecular matching score
2. **Clonal hierarchy** — CCF estimation and truncal/subclonal classification
3. **Evolutionary simulation** — Lotka-Volterra dynamics with Stackelberg policy optimization
4. **ctDNA monitoring** — Closed-loop recalibration using longitudinal liquid biopsy

This repository contains the companion software for the paper:

>  *The Oncologist as the Third Player: Evolutionary Game Theory, Precision Molecular Matching, and the Case for Computational Tumor Boards.* (In preparation)

## ⚠️ Disclaimer

**This software is a research prototype for academic use only.** It is NOT a medical device and has NOT been evaluated by any regulatory authority. All outputs are model-suggested strategies intended to support, not replace, clinical judgment.

## Repository Structure

```
ctb-repo/
├── ctb/                          # Core Python package
│   ├── __init__.py               # Package exports
│   ├── lotka_volterra.py         # ODE model with full parameter provenance
│   ├── policies.py               # Treatment policy implementations
│   ├── matching_score.py         # Molecular matching + CCF estimation
│   └── optimizer.py              # Stackelberg grid-search policy selector
│
├── notebooks/                    # Reproducible analysis notebooks
│   ├── 01_figure2_dynamics.py    # Figure 2: MTD vs AT50 vs CTB
│   ├── 02_figure3_failure_modes.py  # Figure 3: When evolutionary control fails
│   └── 03_figure5_virtual_cohort.py # Figure 5: 500-patient benchmark
│
├── data/
│   ├── parameters/
│   │   └── parameter_provenance.json  # Every parameter with published source
│   ├── outputs/                  # Generated data (reproducible from notebooks)
│   └── published_references/     # Reference PDFs / data from cited papers
│
├── tests/
│   └── test_ctb.py              # Unit + integration tests (pytest)
│
├── docs/
│   ├── METHODS.md               # Full Methods section for the paper
│   ├── MATHEMATICAL_ARCHITECTURE.md  # Equations + pipeline specification
│   └── POMDP_FRAMING.md         # Partially observed control interpretation
│
├── figures/                     # Generated figures (from notebooks)
├── requirements.txt
├── LICENSE                      # MIT
└── README.md                    # This file
```

## Reproducibility

### All data is generated, not imported

Every number in every figure comes from the Lotka-Volterra ODE model implemented in `ctb/lotka_volterra.py`. There are no external datasets, no pre-computed results, and no hidden data sources.

### Audit trail

- **Parameters**: Every model parameter has a published source documented in `data/parameters/parameter_provenance.json`
- **Random seeds**: All stochastic elements (virtual patient sampling) use fixed seeds (seed=42)
- **Notebooks**: Each figure has a dedicated notebook that generates it from scratch
- **Tests**: `pytest tests/test_ctb.py` validates all core computations

### To reproduce all results

```bash
# Install dependencies
pip install -r requirements.txt

# Run tests
pytest tests/test_ctb.py -v

# Reproduce figures
cd notebooks
python 01_figure2_dynamics.py
python 02_figure3_failure_modes.py
python 03_figure5_virtual_cohort.py
```

## Key Results

| Metric | MTD | AT50 | CTB |
|--------|-----|------|-----|
| Median TTP (days) | 755 | 1054 | >1500 |
| Cumulative dose (% of MTD) | 100% | 91% | 50% |
| Resistant fraction at endpoint | 100% | 100% | <1% |

*Virtual cohort (n=500)*: No single fixed policy was universally optimal. The CTB achieved the highest median utility (0.65) by adapting its strategy to each patient's ecological regime.

## Parameter Sources

| Parameter | Value | Source |
|-----------|-------|--------|
| r_S (sensitive growth) | 0.0278 day⁻¹ | Zhang et al., Nat Commun 2017 |
| r_R (resistant growth) | 0.02 day⁻¹ | Cost of resistance assumption |
| α_RS (competition) | 1.5 | Strobl et al., Cancer Res 2024 |
| d_S (drug kill rate) | 0.018 day⁻¹ | Calibrated to abiraterone response |

Full provenance: see `data/parameters/parameter_provenance.json`

## References

1. Zhang J et al. Nat Commun. 2017;8:1816.
2. Zhang J et al. eLife. 2022;11:e76284.
3. Strobl MAR et al. Cancer Res. 2024;84(11):1929.
4. Salvioli M et al. Dyn Games Appl. 2025;15:1750-1769.
5. Sicklick JK et al. JCO. 2026. (I-PREDICT)
6. Staňková K et al. JAMA Oncol. 2019;5(1):96-103.

## License

MIT License. Copyright (c) 2026 Raphael Brandão.

## Contact

Raphael Brandão, MD PhD(c)
Medical Coordinator of Oncology, Rede São Camilo, São Paulo
Founder, First Oncologia
