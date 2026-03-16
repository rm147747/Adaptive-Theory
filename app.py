"""
Computational Tumor Board (CTB) — Streamlit Prototype
=====================================================
Interactive 4-layer decision-support system for adaptive cancer therapy.

Companion software for:
    Brandão R, Scott J. "The Oncologist as the Third Player."

Deploy: streamlit run app.py
Cloud:  https://ctb-adaptive.streamlit.app

DISCLAIMER: Research prototype only. Not for clinical decision making.
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json

from ctb import (
    LVParams, simulate_euler, ctb_select_policy,
    Mutation, compute_matching_scores, estimate_ccf,
    ClosedLoopCTB, CtdnaTimepoint,
)
from ctb.policies import POLICY_SPACE, mtd_policy, adaptive_policy, intermittent_policy

# ═══════════════════════════════════════════════════════════
# PAGE CONFIG
# ═══════════════════════════════════════════════════════════

st.set_page_config(
    page_title="CTB — Computational Tumor Board",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ═══════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════

with st.sidebar:
    st.title("🧬 CTB")
    st.caption("Computational Tumor Board")

    st.markdown("---")

    mode = st.radio(
        "Module",
        ["🎯 Full CTB Pipeline", "📊 Policy Comparison", "🧪 Virtual Cohort", "🔬 ctDNA Closed-Loop"],
        index=0,
    )

    st.markdown("---")

    st.markdown(
        "**⚠️ Research use only**  \n"
        "Model-suggested strategies.  \n"
        "Not for clinical decisions."
    )

    st.markdown("---")
    st.caption(
        "[GitHub](https://github.com/rm147747/Adaptive-Theory) · "
        "CTB v0.2.0 · MIT License"
    )


# ═══════════════════════════════════════════════════════════
# COLORS
# ═══════════════════════════════════════════════════════════

C_MTD = "#E24B4A"
C_AT50 = "#378ADD"
C_CTB = "#1D9E75"
C_GRAY = "#888780"
C_CORAL = "#D85A30"
C_PURPLE = "#7F77DD"


# ═══════════════════════════════════════════════════════════
# MODULE 1: FULL CTB PIPELINE
# ═══════════════════════════════════════════════════════════

if mode == "🎯 Full CTB Pipeline":

    st.header("Computational Tumor Board — 4-Layer Analysis")
    st.markdown(
        "Enter patient genomic data to run the full CTB pipeline: "
        "molecular matching → clonal hierarchy → evolutionary simulation → policy recommendation."
    )

    # ── Patient input ──
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Patient profile")
        cancer_type = st.selectbox("Cancer type", ["mCRPC", "HER2+ Breast", "Custom"])
        purity = st.slider("Tumor purity", 0.10, 1.00, 0.65, 0.05)

    with col2:
        st.subheader("Tumor ecology")
        r0_frac = st.slider("Initial resistant fraction (R₀/N₀)", 0.01, 0.40, 0.02, 0.01)
        alpha_rs = st.slider("Competition coefficient (α_RS)", 0.3, 2.0, 1.5, 0.1,
                             help="Higher = sensitive cells suppress resistant more strongly")
        d_s = st.slider("Drug kill rate (d_S)", 0.005, 0.040, 0.018, 0.001)

    # ── Mutations ──
    st.subheader("Layer 1 — Genomic targeting")

    if cancer_type == "mCRPC":
        default_muts = [
            {"gene": "AR", "variant": "amplification", "VAF": None, "CN": 8,
             "drugs": "abiraterone, enzalutamide", "CCF": 0.95},
            {"gene": "TP53", "variant": "R248W", "VAF": 0.38, "CN": 2,
             "drugs": "", "CCF": None},
            {"gene": "BRCA2", "variant": "splice_site", "VAF": 0.12, "CN": 2,
             "drugs": "olaparib, rucaparib", "CCF": None},
        ]
        default_drugs = "abiraterone"
    elif cancer_type == "HER2+ Breast":
        default_muts = [
            {"gene": "HER2", "variant": "amplification", "VAF": None, "CN": 12,
             "drugs": "trastuzumab, pertuzumab, tucatinib", "CCF": 0.95},
            {"gene": "TP53", "variant": "R248W", "VAF": 0.42, "CN": 2,
             "drugs": "", "CCF": None},
            {"gene": "BRCA2", "variant": "splice_site", "VAF": 0.10, "CN": 2,
             "drugs": "olaparib, talazoparib", "CCF": None},
        ]
        default_drugs = "trastuzumab, olaparib"
    else:
        default_muts = [
            {"gene": "GENE1", "variant": "mut1", "VAF": 0.35, "CN": 2,
             "drugs": "drug_a", "CCF": None},
        ]
        default_drugs = "drug_a"

    n_muts = st.number_input("Number of alterations", 1, 10, len(default_muts))

    mutations = []
    for i in range(n_muts):
        dm = default_muts[i] if i < len(default_muts) else {
            "gene": f"GENE{i+1}", "variant": "", "VAF": 0.20, "CN": 2, "drugs": "", "CCF": None
        }
        cols = st.columns([2, 2, 1, 1, 3])
        gene = cols[0].text_input(f"Gene", dm["gene"], key=f"gene_{i}")
        variant = cols[1].text_input(f"Variant", dm["variant"], key=f"var_{i}")
        vaf = cols[2].number_input(f"VAF", 0.0, 1.0, dm["VAF"] or 0.0, 0.01, key=f"vaf_{i}")
        cn = cols[3].number_input(f"CN", 0, 50, dm["CN"], key=f"cn_{i}")
        drugs_str = cols[4].text_input(f"Matched drugs", dm["drugs"], key=f"drugs_{i}")

        drugs = [d.strip() for d in drugs_str.split(",") if d.strip()]
        mutations.append(Mutation(
            gene=gene, variant=variant,
            VAF=vaf if vaf > 0 else None,
            copy_number=cn, drugs=drugs,
            CCF=dm.get("CCF"),
        ))

    administered = st.text_input("Administered drugs (comma-separated)", default_drugs)
    admin_list = [d.strip() for d in administered.split(",") if d.strip()]

    # ── RUN CTB ──
    if st.button("▶ Run CTB Analysis", type="primary", use_container_width=True):

        # Layer 1: Matching Score
        ms_result = compute_matching_scores(mutations, admin_list, purity)

        st.markdown("---")
        st.subheader("Layer 1 — Matching scores")

        mc1, mc2 = st.columns(2)
        mc1.metric("Classical MS (I-PREDICT)", f"{ms_result['MS_classic']:.1%}")
        mc2.metric("Clonality-weighted MS", f"{ms_result['MS_weighted']:.1%}",
                    delta=f"{(ms_result['MS_weighted'] - ms_result['MS_classic'])*100:+.1f}pp vs classic")

        # Mutation table
        mut_data = []
        for m in ms_result["mutations"]:
            mut_data.append({
                "Gene": m["gene"],
                "CCF": f"{m['CCF']:.1%}",
                "Clonality": m["clonality"],
                "Matched": "✅" if m["matched"] else "❌",
                "Weight": f"{m['weight']:.3f}",
            })
        st.dataframe(mut_data, use_container_width=True, hide_index=True)

        # Layer 2: Clonal Hierarchy
        st.subheader("Layer 2 — Clonal hierarchy")
        tiers = {"truncal": 0, "branch": 0, "subclonal": 0}
        for m in ms_result["mutations"]:
            tiers[m["clonality"]] += 1
        tc1, tc2, tc3 = st.columns(3)
        tc1.metric("Truncal", tiers["truncal"])
        tc2.metric("Branch", tiers["branch"])
        tc3.metric("Subclonal", tiers["subclonal"])

        # Layer 3: Evolutionary Simulation
        st.markdown("---")
        st.subheader("Layer 3 — Evolutionary simulation")

        S0 = (1 - r0_frac) * 0.87
        R0 = r0_frac * 0.87
        params = LVParams(
            r_S=0.0278, r_R=0.02, K=1.0,
            alpha_SR=0.8, alpha_RS=alpha_rs,
            d_S=d_s, d_R=0.001,
        )

        # CTB selects best policy
        ctb_result = ctb_select_policy(params, S0, R0, t_end=1500)
        best = ctb_result["recommended"]

        # Simulate top policies for comparison
        sim_mtd = simulate_euler(params, S0, R0, mtd_policy, t_end=1500)
        at50_fn = adaptive_policy(0.50, 1.0, 14)
        sim_at50 = simulate_euler(params, S0, R0, at50_fn, t_end=1500)
        ctb_fn = POLICY_SPACE[best["policy"]]
        sim_ctb = simulate_euler(params, S0, R0, ctb_fn, t_end=1500)

        # Metrics
        st.success(f"**CTB recommendation: {best['policy']}** (utility = {best['utility']:.4f})")

        rc1, rc2, rc3 = st.columns(3)
        rc1.metric("MTD TTP", f"{sim_mtd['TTP']:.0f}d")
        rc2.metric("AT50 TTP", f"{sim_at50['TTP']:.0f}d")
        rc3.metric(f"CTB TTP ({best['policy']})", f"{sim_ctb['TTP']:.0f}d",
                    delta=f"+{sim_ctb['TTP'] - sim_mtd['TTP']:.0f}d vs MTD")

        # Plot
        fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                            subplot_titles=("Tumor burden N(t)",
                                           "Resistant fraction R(t)/N(t)",
                                           "Treatment intensity u(t)"),
                            vertical_spacing=0.08)

        t = sim_mtd["t"]

        # Panel A
        fig.add_trace(go.Scatter(x=t, y=sim_mtd["N"], name="MTD",
                                 line=dict(color=C_MTD, width=2)), row=1, col=1)
        fig.add_trace(go.Scatter(x=t, y=sim_at50["N"], name="AT50",
                                 line=dict(color=C_AT50, width=2)), row=1, col=1)
        fig.add_trace(go.Scatter(x=t, y=sim_ctb["N"], name=f"CTB ({best['policy']})",
                                 line=dict(color=C_CTB, width=2)), row=1, col=1)

        # Panel B
        rf_mtd = sim_mtd["R"] / sim_mtd["N"]
        rf_at50 = sim_at50["R"] / sim_at50["N"]
        rf_ctb = sim_ctb["R"] / sim_ctb["N"]
        fig.add_trace(go.Scatter(x=t, y=rf_mtd, name="MTD", showlegend=False,
                                 line=dict(color=C_MTD, width=2)), row=2, col=1)
        fig.add_trace(go.Scatter(x=t, y=rf_at50, name="AT50", showlegend=False,
                                 line=dict(color=C_AT50, width=2)), row=2, col=1)
        fig.add_trace(go.Scatter(x=t, y=rf_ctb, name=f"CTB", showlegend=False,
                                 line=dict(color=C_CTB, width=2)), row=2, col=1)

        # Panel C
        fig.add_trace(go.Scatter(x=t, y=sim_mtd["dose"], name="MTD", showlegend=False,
                                 line=dict(color=C_MTD, width=1.5)), row=3, col=1)
        fig.add_trace(go.Scatter(x=t, y=sim_at50["dose"], name="AT50", showlegend=False,
                                 line=dict(color=C_AT50, width=1.5, shape="hv")), row=3, col=1)
        fig.add_trace(go.Scatter(x=t, y=sim_ctb["dose"], name="CTB", showlegend=False,
                                 line=dict(color=C_CTB, width=1.5, shape="hv")), row=3, col=1)

        fig.update_layout(height=700, template="plotly_white",
                          legend=dict(orientation="h", yanchor="bottom", y=1.02))
        fig.update_yaxes(title_text="N(t)", row=1, col=1, range=[0.2, 1.05])
        fig.update_yaxes(title_text="R/N", row=2, col=1, range=[0, 1.05])
        fig.update_yaxes(title_text="u(t)", row=3, col=1, range=[-0.1, 1.2])
        fig.update_xaxes(title_text="Time (days)", row=3, col=1)

        st.plotly_chart(fig, use_container_width=True)

        # Policy ranking table
        st.subheader("All policies ranked")
        pol_data = []
        for p in ctb_result["all_policies"]:
            pol_data.append({
                "Policy": p["policy"],
                "TTP (days)": f"{p['TTP_days']:.0f}",
                "Cumulative dose": f"{p['cumulative_dose']:.0f}",
                "R final": f"{p['R_fraction_final']:.1%}",
                "Utility": f"{p['utility']:.4f}",
                "Rank": "⭐" if p["policy"] == best["policy"] else "",
            })
        st.dataframe(pol_data, use_container_width=True, hide_index=True)

        st.caption(
            f"Utility = TTP/T_max − 0.3 × Dose/T_max − 0.2 × R_final. "
            f"Selection: a priori, based on initial ecological state."
        )


# ═══════════════════════════════════════════════════════════
# MODULE 2: POLICY COMPARISON
# ═══════════════════════════════════════════════════════════

elif mode == "📊 Policy Comparison":

    st.header("Policy Comparison — Parameter Explorer")
    st.markdown("Adjust tumor ecology parameters and see how different treatment strategies perform.")

    col1, col2, col3 = st.columns(3)
    r0_frac = col1.slider("R₀/N₀", 0.01, 0.40, 0.02, 0.01)
    alpha_rs = col2.slider("α_RS", 0.3, 2.0, 1.5, 0.1)
    d_s = col3.slider("d_S", 0.005, 0.040, 0.018, 0.001)

    S0 = (1 - r0_frac) * 0.87
    R0 = r0_frac * 0.87
    params = LVParams(r_S=0.0278, r_R=0.02, K=1.0, alpha_SR=0.8,
                       alpha_RS=alpha_rs, d_S=d_s, d_R=0.001)

    policies_to_show = {
        "MTD": mtd_policy,
        "AT50": adaptive_policy(0.50, 1.0),
        "AT30": adaptive_policy(0.30, 0.90),
        "Int 28/14": intermittent_policy(28, 14),
        "Metro 50%": POLICY_SPACE["metro_50"],
    }

    colors = [C_MTD, C_AT50, C_PURPLE, C_CORAL, C_GRAY]

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        subplot_titles=("Tumor burden", "Resistant fraction"),
                        vertical_spacing=0.10)

    results = []
    for i, (name, fn) in enumerate(policies_to_show.items()):
        sim = simulate_euler(params, S0, R0, fn, t_end=1500)
        rf = sim["R"] / sim["N"]
        results.append({"name": name, "sim": sim, "rf": rf})

        fig.add_trace(go.Scatter(x=sim["t"], y=sim["N"], name=name,
                                 line=dict(color=colors[i], width=2)), row=1, col=1)
        fig.add_trace(go.Scatter(x=sim["t"], y=rf, name=name, showlegend=False,
                                 line=dict(color=colors[i], width=2)), row=2, col=1)

    fig.update_layout(height=500, template="plotly_white",
                      legend=dict(orientation="h", yanchor="bottom", y=1.02))
    fig.update_yaxes(range=[0.2, 1.05], row=1, col=1)
    fig.update_yaxes(range=[0, 1.05], row=2, col=1)
    fig.update_xaxes(title_text="Time (days)", row=2, col=1)

    st.plotly_chart(fig, use_container_width=True)

    # Metrics table
    cols = st.columns(len(results))
    for i, r in enumerate(results):
        with cols[i]:
            st.metric(r["name"], f"TTP {r['sim']['TTP']:.0f}d")
            st.caption(f"Dose: {r['sim']['cumulative_dose']:.0f}")
            st.caption(f"R final: {r['sim']['R_fraction_final']:.1%}")


# ═══════════════════════════════════════════════════════════
# MODULE 3: VIRTUAL COHORT
# ═══════════════════════════════════════════════════════════

elif mode == "🧪 Virtual Cohort":

    st.header("Virtual Patient Cohort Benchmark")
    st.markdown(
        "Test CTB across a heterogeneous virtual population. "
        "Demonstrates that no single fixed policy is universally optimal."
    )

    n_patients = st.slider("Number of virtual patients", 50, 500, 200, 50)

    if st.button("▶ Run Cohort Simulation", type="primary", use_container_width=True):

        np.random.seed(42)
        progress = st.progress(0)
        status = st.empty()

        all_results = {name: [] for name in POLICY_SPACE}
        ctb_results = []
        ctb_choices = []

        for pi in range(n_patients):
            if pi % 20 == 0:
                progress.progress(pi / n_patients)
                status.text(f"Simulating patient {pi+1}/{n_patients}...")

            r0_frac = np.random.uniform(0.01, 0.35)
            alpha_rs = np.random.uniform(0.5, 1.5)
            r_s = np.random.uniform(0.02, 0.035)
            r_r = np.random.uniform(0.015, 0.04)
            d_s = np.random.uniform(0.012, 0.025)
            dr_ratio = np.random.uniform(0.0, 0.7)

            p = LVParams(r_S=r_s, r_R=r_r, K=1.0, alpha_SR=0.8,
                          alpha_RS=alpha_rs, d_S=d_s, d_R=d_s * dr_ratio)
            S0 = (1 - r0_frac) * 0.87
            R0 = r0_frac * 0.87

            ctb = ctb_select_policy(p, S0, R0, t_end=1500)
            ctb_results.append(ctb["recommended"])
            ctb_choices.append(ctb["recommended"]["policy"])

            for pol in ctb["all_policies"]:
                if pol["policy"] in all_results:
                    all_results[pol["policy"]].append(pol)

        progress.progress(1.0)
        status.text(f"Complete: {n_patients} patients × {len(POLICY_SPACE)} policies")

        # Results
        st.markdown("---")

        # Utility comparison
        st.subheader("Composite utility by policy")

        policies_plot = ["MTD", "AT50", "AT30", "int_28_14", "metro_50"]
        labels_plot = ["MTD", "AT50", "AT30", "Int 28/14", "Metro 50%"]
        colors_plot = [C_MTD, C_AT50, C_PURPLE, C_CORAL, C_GRAY, C_CTB]

        fig = go.Figure()

        for i, pname in enumerate(policies_plot):
            utils = [d["utility"] for d in all_results[pname]]
            fig.add_trace(go.Box(y=utils, name=labels_plot[i],
                                 marker_color=colors_plot[i], boxmean=True))

        ctb_utils = [d["utility"] for d in ctb_results]
        fig.add_trace(go.Box(y=ctb_utils, name="CTB",
                             marker_color=C_CTB, boxmean=True))

        fig.update_layout(height=400, template="plotly_white",
                          yaxis_title="Utility score", showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

        # CTB choices
        st.subheader("CTB policy selection distribution")
        from collections import Counter
        counts = Counter(ctb_choices)

        fig2 = go.Figure(go.Pie(
            labels=list(counts.keys()),
            values=list(counts.values()),
            hole=0.4,
            textinfo="label+percent",
        ))
        fig2.update_layout(height=350)
        st.plotly_chart(fig2, use_container_width=True)

        st.info(
            f"**Key finding**: No single fixed policy was optimal across all "
            f"{n_patients} virtual patients. The CTB adapted its strategy to "
            f"each patient's ecological regime."
        )


# ═══════════════════════════════════════════════════════════
# MODULE 4: ctDNA CLOSED-LOOP
# ═══════════════════════════════════════════════════════════

elif mode == "🔬 ctDNA Closed-Loop":

    st.header("ctDNA Closed-Loop Monitoring")
    st.markdown(
        "Simulate serial ctDNA monitoring with optional acquired resistance. "
        "Demonstrates how the CTB detects model-reality divergence."
    )

    col1, col2 = st.columns(2)
    with col1:
        resistance_day = st.slider("Day of resistance emergence", 0, 800, 360, 30,
                                    help="Set to 0 for no acquired resistance")
        r_R_post = st.slider("Resistant growth rate after event", 0.02, 0.06, 0.035, 0.005)
    with col2:
        monitor_interval = st.slider("ctDNA monitoring interval (days)", 30, 180, 90, 30)
        noise_cv = st.slider("Measurement noise (CV)", 0.0, 0.15, 0.05, 0.01)

    if st.button("▶ Run Closed-Loop Simulation", type="primary", use_container_width=True):

        np.random.seed(42)

        params = LVParams(r_S=0.0278, r_R=0.02, K=1.0, alpha_SR=0.8,
                           alpha_RS=1.5, d_S=0.018, d_R=0.001)
        S0, R0 = 0.85, 0.02
        N0 = S0 + R0
        T_END = 900
        dt = 1.0
        t_eval = np.arange(0, T_END, dt)

        # Simulate true tumor with resistance shift
        S_true, R_true = [S0], [R0]
        state_ctrl = {"treating": True, "nadir": N0}
        at30 = adaptive_policy(0.30, 0.90, 14)

        for i in range(1, len(t_eval)):
            s, r = S_true[-1], R_true[-1]
            n = s + r
            u = at30(t_eval[i], n, N0, state_ctrl, i)

            r_R_eff = 0.02 if (resistance_day == 0 or t_eval[i] < resistance_day) else r_R_post
            a_RS_eff = 1.5 if (resistance_day == 0 or t_eval[i] < resistance_day) else 1.0

            dS = params.r_S * s * (1 - (s + params.alpha_SR * r) / params.K) - u * params.d_S * s
            dR = r_R_eff * r * (1 - (r + a_RS_eff * s) / params.K) - u * params.d_R * r

            S_true.append(max(s + dS * dt, 1e-10))
            R_true.append(max(r + dR * dt, 1e-10))

        S_true = np.array(S_true)
        R_true = np.array(R_true)
        N_true = S_true + R_true
        Rf_true = R_true / N_true

        # Model prediction (unaware of change)
        sim_pred = simulate_euler(params, S0, R0, adaptive_policy(0.30, 0.90, 14), t_end=T_END)

        # Generate ctDNA measurements
        ctdna_days = list(range(0, T_END, monitor_interval))
        measurements = []

        for day in ctdna_days:
            idx = int(day)
            if idx >= len(S_true):
                break
            tf_true = N_true[idx] / N0
            qr_true = R_true[idx] / N_true[idx]

            tf_obs = max(0.001, tf_true * (1 + np.random.normal(0, noise_cv)))
            qr_obs = max(0.0, min(1.0, qr_true * (1 + np.random.normal(0, noise_cv * 1.5))))

            measurements.append(CtdnaTimepoint(
                day=day, tumor_fraction=round(tf_obs, 4),
                sensitive_vafs={"TP53": round((1 - qr_obs) * tf_obs * 0.4, 4)},
                resistant_vafs={"BRCA2": round(qr_obs * tf_obs * 0.4, 4)},
                read_depths={"TP53": 400, "BRCA2": 300},
            ))

        # Run closed-loop
        ctb_loop = ClosedLoopCTB(params, S0, R0, tf_baseline=measurements[0].tumor_fraction)
        reports = []
        for obs in measurements:
            report = ctb_loop.update(obs)
            reports.append(report)

        # Plot
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            subplot_titles=("Tumor burden: true vs model",
                                           "Resistant fraction: true vs model"),
                            vertical_spacing=0.10)

        fig.add_trace(go.Scatter(x=t_eval, y=N_true, name="True tumor",
                                 line=dict(color=C_CTB, width=2.5)), row=1, col=1)
        fig.add_trace(go.Scatter(x=sim_pred["t"], y=sim_pred["N"], name="Model prediction",
                                 line=dict(color=C_AT50, width=1.5, dash="dash")), row=1, col=1)
        fig.add_trace(go.Scatter(
            x=[m.day for m in measurements],
            y=[m.tumor_fraction * N0 / measurements[0].tumor_fraction for m in measurements],
            name="ctDNA", mode="markers",
            marker=dict(color=C_CORAL, size=8, symbol="triangle-down")), row=1, col=1)

        fig.add_trace(go.Scatter(x=t_eval, y=Rf_true, name="True R fraction", showlegend=False,
                                 line=dict(color=C_CTB, width=2.5)), row=2, col=1)
        rf_pred = sim_pred["R"] / sim_pred["N"]
        fig.add_trace(go.Scatter(x=sim_pred["t"], y=rf_pred, name="Model R prediction", showlegend=False,
                                 line=dict(color=C_AT50, width=1.5, dash="dash")), row=2, col=1)
        fig.add_trace(go.Scatter(
            x=[r.day for r in reports],
            y=[r.observed_q_R for r in reports],
            name="ctDNA q_R", mode="markers", showlegend=False,
            marker=dict(color=C_CORAL, size=8, symbol="triangle-down")), row=2, col=1)

        if resistance_day > 0:
            fig.add_vline(x=resistance_day, line_dash="dot", line_color="gray",
                          annotation_text="resistance acquired", row=1, col=1)
            fig.add_vline(x=resistance_day, line_dash="dot", line_color="gray", row=2, col=1)

        fig.update_layout(height=550, template="plotly_white",
                          legend=dict(orientation="h", yanchor="bottom", y=1.02))
        fig.update_yaxes(title_text="N(t)", range=[0.3, 1.1], row=1, col=1)
        fig.update_yaxes(title_text="R/N", range=[0, 1.0], row=2, col=1)
        fig.update_xaxes(title_text="Time (days)", row=2, col=1)

        st.plotly_chart(fig, use_container_width=True)

        # Monitoring log
        st.subheader("Monitoring log")
        log_data = []
        for r in reports:
            status_emoji = {"normal": "🟢", "r_expanding": "🟡",
                           "r_dominant": "🔴", "progression_signal": "🔴",
                           "below_lod": "⚪"}.get(r.status, "⚪")
            log_data.append({
                "Day": int(r.day),
                "TF": f"{r.observed_tumor_fraction:.3f}",
                "q_R": f"{r.observed_q_R:.1%}",
                "Status": f"{status_emoji} {r.status}",
                "Policy": r.recommended_policy,
                "Flags": " | ".join(r.safety_flags) if r.safety_flags else "—",
            })
        st.dataframe(log_data, use_container_width=True, hide_index=True)

        # Detection summary
        first_flag = next((r for r in reports if r.safety_flags), None)
        if first_flag and resistance_day > 0:
            lead = first_flag.day - resistance_day
            st.success(
                f"**Detection**: ctDNA flagged resistance at day {int(first_flag.day)} "
                f"({lead:.0f} days after resistance emergence at day {resistance_day})"
            )
