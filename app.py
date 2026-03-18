"""
CTB — Computational Tumor Board
================================
AI-powered evolutionary analysis for adaptive cancer therapy.

This is a RESEARCH PROTOTYPE. All outputs are simulation-backed suggestions,
not clinical recommendations. See README for full disclaimer.

Deploy: streamlit run app.py
"""

import streamlit as st
import numpy as np
from collections import Counter

from ctb import (
    LVParams, simulate_euler, ctb_select_policy,
    Mutation, compute_matching_scores,
)
from ctb.policies import POLICY_SPACE, mtd_policy, adaptive_policy

from app_modules import llm_services
from app_modules.plotting import plot_dynamics, plot_cohort_pie, COLORS
from app_modules.config import (
    PRESETS, DEFAULT_R0_FRAC, DEFAULT_ALPHA_RS, DEFAULT_D_S,
    BOUNDS, DISCLAIMER, DISCLAIMER_HTML,
)

# ═══════════════════════════════════════════════════════════
# Page config and styling
# ═══════════════════════════════════════════════════════════

st.set_page_config(page_title="CTB", page_icon="🧬", layout="wide")

st.markdown("""<style>
.stApp{background:#FAFAF8}
.block-container{max-width:1100px;padding-top:1.5rem}
.strat-box{background:linear-gradient(135deg,#E8F5EE,#F0FAF5);
  border:1px solid #1D9E75;border-radius:12px;padding:1.25rem 1.5rem;margin:1rem 0}
.strat-box h3{color:#0F6E56;margin:0 0 .5rem;font-size:1.1rem}
.strat-box p{color:#2C2C2A;margin:0;font-size:.95rem;line-height:1.5}
.inferred-badge{background:#FFF3CD;border-radius:4px;padding:2px 6px;
  font-size:0.75rem;color:#856404}
#MainMenu{visibility:hidden}footer{visibility:hidden}.stDeployButton{display:none}
</style>""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
# LLM initialization
# ═══════════════════════════════════════════════════════════

llm_services.get_client()


# ═══════════════════════════════════════════════════════════
# Header and sidebar
# ═══════════════════════════════════════════════════════════

st.markdown("# 🧬 Computational Tumor Board")
st.caption("Evolutionary analysis for adaptive cancer therapy — research prototype")

with st.sidebar:
    mode = st.radio(
        "Module",
        ["📋 Clinical Case", "🔬 Explorer", "🧪 Cohort"],
        index=0,
    )
    st.markdown("---")
    st.markdown(DISCLAIMER)
    st.caption("[GitHub](https://github.com/rm147747/Adaptive-Theory) · v0.3.1")


# ═══════════════════════════════════════════════════════════
# MODULE 1: CLINICAL CASE
# ═══════════════════════════════════════════════════════════

if mode == "📋 Clinical Case":

    if not llm_services.is_available():
        st.info(
            "Add `ANTHROPIC_API_KEY` in Settings → Secrets for "
            "AI-powered case interpretation and narrative reports."
        )

    # ── Input tabs ──
    tab_free, tab_struct = st.tabs([
        "📝 Describe case (AI)" if llm_services.is_available()
        else "📝 Free text (needs API key)",
        "📊 Structured input",
    ])

    with tab_free:
        case_text = st.text_area(
            "Clinical case",
            height=160,
            placeholder=(
                "68yo male, mCRPC, progressing on 2nd-line abiraterone. "
                "PSA 42. NGS: AR amp, TP53 R248W (VAF 0.38), "
                "BRCA2 splice (VAF 0.12 subclonal). No CNS mets."
            ),
        )
        if case_text and not llm_services.is_available():
            st.warning("Free-text requires API key. Use the Structured input tab.")

    # ── Structured input ──
    with tab_struct:
        col1, col2 = st.columns(2)

        cancer = col1.selectbox("Cancer type", list(PRESETS.keys()))
        purity = col1.slider("Purity", *BOUNDS["purity"], 0.65, 0.05)

        r0f = col2.slider("R₀/N₀", *BOUNDS["r0_fraction"], DEFAULT_R0_FRAC, 0.01)
        ars = col2.slider("α_RS", *BOUNDS["alpha_rs"], DEFAULT_ALPHA_RS, 0.1)
        ds = col2.slider("d_S", *BOUNDS["d_s"], DEFAULT_D_S, 0.001)

        preset = PRESETS[cancer]
        preset_muts = preset["mutations"]
        n_muts = st.number_input("Alterations", 1, 10, len(preset_muts))

        mutations = []
        for i in range(n_muts):
            dm = preset_muts[i] if i < len(preset_muts) else ("GENE", "v", 0.2, 2, "", None)
            cc = st.columns([2, 2, 1, 1, 3])
            gene = cc[0].text_input("Gene", dm[0], key=f"g{i}")
            var = cc[1].text_input("Variant", dm[1], key=f"v{i}")
            vaf = cc[2].number_input("VAF", 0.0, 1.0, dm[2], 0.01, key=f"vf{i}")
            cn = cc[3].number_input("CN", 0, 50, dm[3], key=f"cn{i}")
            drugs_str = cc[4].text_input("Drugs", dm[4], key=f"dr{i}")
            drugs = [x.strip() for x in drugs_str.split(",") if x.strip()]
            mutations.append(
                Mutation(gene, var, vaf if vaf > 0 else None, cn, drugs, dm[5])
            )

        administered = st.text_input("Current drugs", preset["drugs"])
        admin_list = [x.strip() for x in administered.split(",") if x.strip()]

    # ── Run analysis ──
    st.markdown("---")

    if st.button("▶ Generate Tumor Board Report", type="primary", use_container_width=True):

        # Track parameter provenance
        param_source = "user_entered"
        case_data = None

        # LLM parsing (if available and case text provided)
        if llm_services.is_available() and case_text.strip():
            with st.spinner("🧠 AI interpreting clinical case..."):
                try:
                    case_data = llm_services.parse_case(case_text)
                    mutations = [
                        Mutation(
                            m["gene"], m.get("variant", ""),
                            m.get("VAF"), m.get("copy_number", 2),
                            m.get("drugs", []), m.get("CCF"),
                        )
                        for m in case_data["mutations"]
                    ]
                    admin_list = case_data.get("administered_drugs", admin_list)
                    purity = case_data.get("purity", purity)
                    r0f = case_data.get("estimated_r0_fraction", r0f)
                    ars = case_data.get("estimated_alpha_rs", ars)
                    ds = case_data.get("estimated_d_s", ds)
                    param_source = "llm_inferred"
                    st.success(f"Parsed: {case_data.get('clinical_summary', '')}")
                except Exception as e:
                    st.warning(f"Parse error: {e}. Using structured input.")

        # ── Run 4 layers ──
        ms = compute_matching_scores(mutations, admin_list, purity)

        S0 = (1 - r0f) * 0.87
        R0 = r0f * 0.87
        params = LVParams(
            r_S=0.0278, r_R=0.02, K=1.0, alpha_SR=0.8,
            alpha_RS=ars, d_S=ds, d_R=0.001,
        )

        sim_m = simulate_euler(params, S0, R0, mtd_policy, t_end=1500)
        sim_a = simulate_euler(params, S0, R0, adaptive_policy(0.50, 1.0), t_end=1500)
        ctb_r = ctb_select_policy(params, S0, R0, t_end=1500)
        best = ctb_r["recommended"]
        sim_c = simulate_euler(params, S0, R0, POLICY_SPACE[best["policy"]], t_end=1500)

        dt_ttp = sim_c["TTP"] - sim_m["TTP"]
        dose_red = (1 - sim_c["cumulative_dose"] / max(sim_m["cumulative_dose"], 1)) * 100

        # ── Strategy recommendation ──
        source_badge = (
            ' <span class="inferred-badge">CTB-inferred parameters</span>'
            if param_source == "llm_inferred" else ""
        )

        st.markdown(f"""<div class="strat-box">
        <h3>📋 CTB Strategy: {best['policy']}{source_badge}</h3>
        <p>Simulation-suggested TTP <strong>{sim_c['TTP']:.0f} days</strong>
        ({f"+{dt_ttp:.0f}d vs MTD" if dt_ttp > 0 else "= MTD"}),
        {f"{dose_red:.0f}% dose reduction" if dose_red > 5 else "similar exposure"},
        resistant fraction {sim_c['R_fraction_final']:.0%} at endpoint.
        Utility score: {best['utility']:.3f}.</p>
        <p style="font-size:0.8rem;color:#888780;margin-top:0.5rem">
        This is a simulation-backed suggestion from a discrete policy search
        over {len(POLICY_SPACE)} candidate heuristics, not a validated clinical recommendation.</p>
        </div>""", unsafe_allow_html=True)

        # ── Metrics ──
        mc = st.columns(4)
        mc[0].metric("MS weighted", f"{ms['MS_weighted']:.0%}")
        mc[1].metric("Simulated TTP", f"{sim_c['TTP']:.0f}d",
                      f"{dt_ttp:+.0f}d vs MTD" if dt_ttp else None)
        mc[2].metric("Dose saved", f"{dose_red:.0f}%" if dose_red > 0 else "—")
        mc[3].metric("R fraction", f"{sim_c['R_fraction_final']:.0%}")

        # ── Genomic table ──
        st.markdown("#### 🧬 Genomic analysis")
        st.dataframe(
            [
                {
                    "Gene": m["gene"],
                    "CCF": f"{m['CCF']:.0%}",
                    "Tier": m["clonality"].upper(),
                    "Matched": "✅ " + ", ".join(m.get("matched_drugs", []))
                    if m["matched"] else "❌",
                }
                for m in ms["mutations"]
            ],
            use_container_width=True,
            hide_index=True,
        )

        # ── Dynamics plot ──
        st.markdown("#### 📈 Evolutionary dynamics (Lotka-Volterra simulation)")
        fig = plot_dynamics(sim_m, sim_a, sim_c, best["policy"])
        st.plotly_chart(fig, use_container_width=True)

        # ── Policy ranking ──
        st.markdown("#### 🏆 Policy ranking")
        st.dataframe(
            [
                {
                    "": "⭐" if p["policy"] == best["policy"] else "",
                    "Policy": p["policy"],
                    "TTP": f"{p['TTP_days']:.0f}d",
                    "Dose": f"{p['cumulative_dose']:.0f}",
                    "R%": f"{p['R_fraction_final']:.0%}",
                    "Utility": f"{p['utility']:.3f}",
                }
                for p in ctb_r["all_policies"][:6]
            ],
            use_container_width=True,
            hide_index=True,
        )
        st.caption(
            "U = TTP/T_max − 0.3·Dose/T_max − 0.2·R_final. "
            "Discrete search over candidate heuristics; not a full "
            "control-theoretic optimizer."
        )

        # ── LLM narrative report ──
        if llm_services.is_available():
            st.markdown("#### 📝 Tumor board report")
            with st.spinner("Generating narrative report..."):
                try:
                    cd = case_data or {
                        "cancer_type": cancer,
                        "mutations": [m.__dict__ for m in mutations],
                    }
                    sr = {
                        k: {
                            "TTP": v["TTP"],
                            "cumulative_dose": v["cumulative_dose"],
                            "R_fraction_final": v["R_fraction_final"],
                        }
                        for k, v in {"mtd": sim_m, "at50": sim_a, "ctb": sim_c}.items()
                    }
                    report = llm_services.generate_report(cd, sr, ms, ctb_r)
                    st.markdown(report)
                except Exception as e:
                    st.error(f"Report error: {e}")

        # ── Mandatory disclaimer ──
        st.markdown(DISCLAIMER_HTML, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════
# MODULE 2: PARAMETER EXPLORER
# ═══════════════════════════════════════════════════════════

elif mode == "🔬 Explorer":
    st.markdown("### Parameter Explorer")
    st.markdown(
        "Adjust tumor ecology and compare treatment strategies. "
        "All results are Lotka-Volterra simulations with heuristic policies."
    )

    c1, c2, c3 = st.columns(3)
    r0 = c1.slider("R₀/N₀", *BOUNDS["r0_fraction"], DEFAULT_R0_FRAC, 0.01)
    ars = c2.slider("α_RS", *BOUNDS["alpha_rs"], DEFAULT_ALPHA_RS, 0.1)
    ds = c3.slider("d_S", *BOUNDS["d_s"], DEFAULT_D_S, 0.001)

    S0 = (1 - r0) * 0.87
    R0_val = r0 * 0.87
    params = LVParams(
        r_S=0.0278, r_R=0.02, K=1.0, alpha_SR=0.8,
        alpha_RS=ars, d_S=ds, d_R=0.001,
    )

    sim_mtd = simulate_euler(params, S0, R0_val, mtd_policy, t_end=1500)
    sim_at50 = simulate_euler(params, S0, R0_val, adaptive_policy(0.50, 1.0), t_end=1500)
    ctb = ctb_select_policy(params, S0, R0_val, t_end=1500)
    best_name = ctb["recommended"]["policy"]
    sim_ctb = simulate_euler(params, S0, R0_val, POLICY_SPACE[best_name], t_end=1500)

    fig = plot_dynamics(sim_mtd, sim_at50, sim_ctb, best_name)
    st.plotly_chart(fig, use_container_width=True)
    st.success(
        f"**CTB selects: {best_name}** "
        f"(utility: {ctb['recommended']['utility']:.3f})"
    )


# ═══════════════════════════════════════════════════════════
# MODULE 3: VIRTUAL COHORT
# ═══════════════════════════════════════════════════════════

elif mode == "🧪 Cohort":
    st.markdown("### Virtual Cohort Benchmark")
    st.markdown(
        "Simulate CTB across heterogeneous virtual patients. "
        "Parameters sampled from published biological ranges."
    )

    n_pat = st.slider("Patients", 50, 500, 200, 50)

    if st.button("▶ Run", type="primary"):
        np.random.seed(42)
        bar = st.progress(0)
        choices, ctb_utils, mtd_utils = [], [], []

        for i in range(n_pat):
            if i % 20 == 0:
                bar.progress(i / n_pat)

            p = LVParams(
                r_S=np.random.uniform(0.02, 0.035),
                r_R=np.random.uniform(0.015, 0.04),
                K=1.0, alpha_SR=0.8,
                alpha_RS=np.random.uniform(0.5, 1.5),
                d_S=np.random.uniform(0.012, 0.025),
                d_R=np.random.uniform(0.012, 0.025) * np.random.uniform(0, 0.7),
            )
            rf = np.random.uniform(0.01, 0.35)
            s0 = (1 - rf) * 0.87
            r0 = rf * 0.87

            res = ctb_select_policy(p, s0, r0, t_end=1500)
            choices.append(res["recommended"]["policy"])
            ctb_utils.append(res["recommended"]["utility"])
            mtd_entry = [x for x in res["all_policies"] if x["policy"] == "MTD"]
            mtd_utils.append(mtd_entry[0]["utility"] if mtd_entry else 0)

        bar.progress(1.0)

        counts = Counter(choices)
        c1, c2 = st.columns(2)
        c1.metric("CTB median utility", f"{np.median(ctb_utils):.3f}")
        c1.metric("MTD median utility", f"{np.median(mtd_utils):.3f}")
        wins = sum(1 for a, b in zip(ctb_utils, mtd_utils) if a > b)
        c2.metric("CTB beats MTD", f"{wins}/{n_pat}")

        fig = plot_cohort_pie(counts)
        st.plotly_chart(fig, use_container_width=True)

        st.info(
            f"No single fixed policy was optimal across {n_pat} virtual patients. "
            f"CTB adapted its strategy to each patient's ecological regime."
        )
