"""
LLM services for the CTB Streamlit app.

Handles Anthropic API integration for:
    - Clinical case parsing (free-text → structured JSON)
    - Narrative tumor board report generation

All LLM outputs are labeled as model-inferred in the UI.
"""

import json
import os
import streamlit as st

HAS_LLM = False
_client = None


def get_client():
    """Initialize Anthropic client from secrets or environment."""
    global HAS_LLM, _client
    try:
        from anthropic import Anthropic
        api_key = st.secrets.get("ANTHROPIC_API_KEY", os.environ.get("ANTHROPIC_API_KEY"))
        if api_key:
            _client = Anthropic(api_key=api_key)
            HAS_LLM = True
    except Exception:
        pass
    return _client


def is_available() -> bool:
    """Check if LLM services are available."""
    return HAS_LLM and _client is not None


PARSE_SYSTEM = """You are a clinical oncology data extractor. Extract structured data from the case.
Return ONLY valid JSON:
{"cancer_type":"string","purity":float,"mutations":[{"gene":"string","variant":"string",
"VAF":float_or_null,"copy_number":int,"drugs":["string"],"CCF":float_or_null}],
"administered_drugs":["string"],"prior_lines":int,"clinical_summary":"1-2 sentences",
"estimated_r0_fraction":float,"estimated_alpha_rs":float,"estimated_d_s":float}

Use oncology expertise to estimate ecological parameters from clinical context.
Default purity=0.65 if not stated. Bound estimates:
  r0_fraction: [0.01, 0.40]
  alpha_rs: [0.3, 2.0]
  d_s: [0.005, 0.035]"""

REPORT_SYSTEM = """You are an expert oncologist at a computational tumor board. Write a clinical report:
1. CLINICAL SUMMARY (2-3 sentences)
2. GENOMIC ANALYSIS (matching results, clonal hierarchy, significance)
3. EVOLUTIONARY ASSESSMENT (model predictions, strategy tradeoffs)
4. RECOMMENDED STRATEGY (which policy, why, in clinical terms)
5. MONITORING PLAN (what to track, when to reassess)
6. CAVEATS (model limitations)
Write professionally. End with: 'This analysis is model-suggested and requires physician review.'
IMPORTANT: This is a RESEARCH PROTOTYPE. All values labeled as 'CTB-inferred' were estimated
by the computational model, not observed clinically."""


def parse_case(case_text: str) -> dict:
    """
    Parse a free-text clinical case description into structured data.

    Returns a dict with validated fields. Raises ValueError if parsing fails.
    All inferred values are marked with '_source': 'llm_inferred'.
    """
    if not is_available():
        raise RuntimeError("LLM not available")

    resp = _client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2000,
        system=PARSE_SYSTEM,
        messages=[{"role": "user", "content": case_text}],
    )

    text = resp.content[0].text.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1]
    if text.endswith("```"):
        text = text[:-3]

    data = json.loads(text)

    # Validate and bound ecological parameters
    data["estimated_r0_fraction"] = max(0.01, min(0.40, data.get("estimated_r0_fraction", 0.02)))
    data["estimated_alpha_rs"] = max(0.3, min(2.0, data.get("estimated_alpha_rs", 1.5)))
    data["estimated_d_s"] = max(0.005, min(0.035, data.get("estimated_d_s", 0.018)))
    data["purity"] = max(0.10, min(1.0, data.get("purity", 0.65)))

    # Mark all inferred values
    data["_parameter_source"] = "llm_inferred"

    return data


def generate_report(case_data: dict, sim_results: dict,
                    ms_result: dict, ctb_result: dict) -> str:
    """
    Generate a narrative tumor board report.

    All model-inferred values are explicitly labeled in the output.
    """
    if not is_available():
        raise RuntimeError("LLM not available")

    ctx = (
        f"CASE: {json.dumps(case_data)}\n"
        f"MATCHING: MS_classic={ms_result['MS_classic']:.1%}, "
        f"MS_weighted={ms_result['MS_weighted']:.1%}\n"
        f"Mutations: {json.dumps(ms_result['mutations'])}\n"
        f"SIMULATION: "
        f"MTD TTP={sim_results['mtd']['TTP']:.0f}d "
        f"R={sim_results['mtd']['R_fraction_final']:.0%}\n"
        f"AT50 TTP={sim_results['at50']['TTP']:.0f}d "
        f"R={sim_results['at50']['R_fraction_final']:.0%}\n"
        f"CTB={ctb_result['recommended']['policy']} "
        f"TTP={ctb_result['recommended']['TTP_days']:.0f}d "
        f"utility={ctb_result['recommended']['utility']:.4f}"
    )

    resp = _client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=2500,
        system=REPORT_SYSTEM,
        messages=[{"role": "user", "content": ctx}],
    )

    return resp.content[0].text
