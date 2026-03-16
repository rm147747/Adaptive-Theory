"""
Molecular matching score calculator.

Implements:
    1. Classical I-PREDICT Matching Score (MS)
    2. Clonality-Weighted Matching Score (MS_w) — novel extension

The clonality-weighted MS prioritizes truncal alterations over subclonal ones,
reflecting the biological principle that truncal drivers represent the largest
therapeutic target within the tumor.

References:
    [1] Sicklick JK et al. Investigation of Profile-Related Evidence Determining
        Individualized Cancer Therapy (I-PREDICT) N-of-1 precision oncology study.
        J Clin Oncol. 2026. doi:10.1200/JCO-25-01453
"""

from dataclasses import dataclass
import math


@dataclass
class Mutation:
    """A pathogenic alteration with optional clonality information."""
    gene: str
    variant: str = ""
    VAF: float | None = None       # variant allele frequency
    copy_number: int = 2           # total copy number at locus
    drugs: list[str] = None        # list of matched drugs
    CCF: float | None = None       # cancer cell fraction (if known)

    def __post_init__(self):
        if self.drugs is None:
            self.drugs = []


def estimate_ccf(mut: Mutation, purity: float, multiplicity: int = 1) -> float:
    """
    Estimate cancer cell fraction from VAF.

    Formula:
        CCF = VAF × (purity × CN_tumor + (1 - purity) × CN_normal) / (purity × multiplicity)

    This is a simplified estimator. For clinical use, tools like PyClone-VI,
    ABSOLUTE, or FACETS should be used.

    Args:
        mut: mutation with VAF and copy_number
        purity: tumor purity (0-1)
        multiplicity: expected allelic copies carrying the variant (default 1)

    Returns:
        Estimated CCF, capped at 1.0

    Reference:
        McGranahan N, Swanton C. Clonal heterogeneity and tumor evolution.
        Cell. 2017;168(4):613-628. doi:10.1016/j.cell.2017.01.018
    """
    if mut.VAF is None or mut.VAF == 0:
        return 0.0 if mut.VAF == 0 else 1.0  # default for CNV-driven

    cn_normal = 2
    numerator = mut.VAF * (purity * mut.copy_number + (1 - purity) * cn_normal)
    denominator = purity * multiplicity

    if denominator <= 0:
        return 1.0

    return min(numerator / denominator, 1.0)


def classify_clonality(ccf: float) -> str:
    """
    Classify alteration by clonality tier.

    Thresholds:
        Truncal:    CCF > 0.60
        Branch:     0.30 ≤ CCF ≤ 0.60
        Subclonal:  CCF < 0.30
    """
    if ccf > 0.60:
        return "truncal"
    elif ccf >= 0.30:
        return "branch"
    else:
        return "subclonal"


def compute_matching_scores(mutations: list[Mutation],
                            administered_drugs: list[str],
                            purity: float) -> dict:
    """
    Compute classical and clonality-weighted matching scores.

    Classical MS (I-PREDICT):
        MS = (# matched alterations) / (# total pathogenic alterations)

    Clonality-weighted MS (novel):
        MS_w = Σ(w_i × m_i) / Σ(w_i)
        where w_i = CCF_i, m_i = 1 if matched

    Args:
        mutations: list of pathogenic alterations
        administered_drugs: list of drug names being administered
        purity: tumor purity for CCF estimation

    Returns:
        dict with MS_classic, MS_weighted, and per-mutation details
    """
    if not mutations:
        return {"MS_classic": 0.0, "MS_weighted": 0.0, "mutations": []}

    admin_set = {d.lower() for d in administered_drugs}
    details = []
    total_matched = 0
    sum_w_matched = 0.0
    sum_w_total = 0.0

    for mut in mutations:
        # Estimate CCF
        ccf = mut.CCF if mut.CCF is not None else estimate_ccf(mut, purity)
        w = ccf

        # Check drug matching
        matched = any(d.lower() in admin_set for d in mut.drugs)
        m_i = 1 if matched else 0

        total_matched += m_i
        sum_w_matched += w * m_i
        sum_w_total += w

        details.append({
            "gene": mut.gene,
            "variant": mut.variant,
            "CCF": round(ccf, 3),
            "clonality": classify_clonality(ccf),
            "matched": matched,
            "weight": round(w, 3),
            "matched_drugs": [d for d in mut.drugs if d.lower() in admin_set],
        })

    n = len(mutations)
    ms_classic = round(total_matched / n, 3) if n > 0 else 0.0
    ms_weighted = round(sum_w_matched / sum_w_total, 3) if sum_w_total > 0 else 0.0

    return {
        "MS_classic": ms_classic,
        "MS_weighted": ms_weighted,
        "n_alterations": n,
        "n_matched": total_matched,
        "mutations": details,
    }


# ═══════════════════════════════════════════════════════════
# DRUG-TARGET DATABASE (curated subset)
# Source: OncoKB Level 1-2, CIViC Level A-B
# Last updated: March 2026
# ═══════════════════════════════════════════════════════════

DRUG_TARGET_DB = {
    # GU oncology
    "AR amplification": ["abiraterone", "enzalutamide", "darolutamide", "apalutamide"],
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
    # Pan-cancer
    "BRAF V600E": ["dabrafenib", "vemurafenib", "encorafenib"],
    "NTRK fusion": ["larotrectinib", "entrectinib"],
    "RET fusion": ["selpercatinib", "pralsetinib"],
    "KRAS G12C": ["sotorasib", "adagrasib"],
}
