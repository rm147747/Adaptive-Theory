"""
Configuration and presets for the CTB Streamlit app.
"""

# Disease-specific presets for structured input
PRESETS = {
    "mCRPC": {
        "mutations": [
            ("AR", "amplification", 0.0, 8, "abiraterone, enzalutamide", 0.95),
            ("TP53", "R248W", 0.38, 2, "", None),
            ("BRCA2", "splice_site", 0.12, 2, "olaparib, rucaparib", None),
        ],
        "drugs": "abiraterone",
    },
    "HER2+ Breast": {
        "mutations": [
            ("HER2", "amplification", 0.0, 12, "trastuzumab, pertuzumab, tucatinib", 0.95),
            ("TP53", "R248W", 0.42, 2, "", None),
            ("PIK3CA", "H1047R", 0.15, 2, "alpelisib", None),
        ],
        "drugs": "trastuzumab",
    },
    "Other": {
        "mutations": [("GENE1", "variant1", 0.30, 2, "drug_a", None)],
        "drugs": "drug_a",
    },
}

# Default ecological parameters
DEFAULT_R0_FRAC = 0.02
DEFAULT_ALPHA_RS = 1.5
DEFAULT_D_S = 0.018

# Parameter bounds (hard limits)
BOUNDS = {
    "r0_fraction": (0.01, 0.40),
    "alpha_rs": (0.3, 2.0),
    "d_s": (0.005, 0.035),
    "purity": (0.10, 1.00),
}

# Disclaimer text
DISCLAIMER = (
    "⚠️ **Research prototype** — All outputs are model-suggested strategies "
    "for academic use only. Not a medical device. Not for clinical decision "
    "making without independent physician review. Parameters marked "
    "'CTB-inferred' were estimated by the computational model, not observed clinically."
)

DISCLAIMER_HTML = (
    '<div style="background:#F5F5F2;border-radius:8px;padding:.75rem 1rem;'
    'font-size:.75rem;color:#888780;margin-top:1.5rem;text-align:center">'
    '⚠️ <strong>Research prototype</strong> — Model-suggested strategies only. '
    'Not for clinical decisions. Parameters marked <em>CTB-inferred</em> were '
    'estimated by the computational model, not observed clinically. '
    '<a href="https://github.com/rm147747/Adaptive-Theory">GitHub</a> · v0.3.1'
    '</div>'
)
