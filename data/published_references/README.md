# Published References

This directory is intended to store reference materials from published papers
used to derive model parameters. These are NOT data inputs to the CTB —
they are audit documentation showing where parameter values come from.

## Key references to include:

1. **Zhang et al. 2017** — Table 1, Supplementary Methods
   - Source of growth rate parameters (r_S, r_R)
   - Source of adaptive abiraterone protocol (AT50)
   - DOI: 10.1038/s41467-017-01968-5

2. **Strobl et al. 2024** — Supplementary Section S4
   - Patient fits used to validate parameter ranges
   - Competition coefficient estimates
   - DOI: 10.1158/0008-5472.CAN-23-2040

3. **Sicklick et al. 2026** — Main text, Table 1
   - I-PREDICT matching score definition
   - DOI: 10.1200/JCO-25-01453

## Note

No patient-level data from these papers is used in the CTB.
Only published aggregate parameters (growth rates, competition coefficients)
are extracted and documented in `parameters/parameter_provenance.json`.
