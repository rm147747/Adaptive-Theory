"""
CTB — Computational Tumor Board
AI-powered evolutionary tumor board for adaptive cancer therapy.
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import os

from ctb import (
    LVParams, simulate_euler, ctb_select_policy,
    Mutation, compute_matching_scores,
)
from ctb.policies import POLICY_SPACE, mtd_policy, adaptive_policy

# ── Page config ──
st.set_page_config(page_title="CTB", page_icon="🧬", layout="wide")

# ── Colors ──
CP = "#1D9E75"
CR = "#E24B4A"
CI = "#378ADD"
CU = "#7F77DD"
CG = "#888780"

# ── CSS ──
st.markdown("""<style>
.stApp{background:#FAFAF8}
.block-container{max-width:1100px;padding-top:1.5rem}
.strat-box{background:linear-gradient(135deg,#E8F5EE,#F0FAF5);border:1px solid #1D9E75;
  border-radius:12px;padding:1.25rem 1.5rem;margin:1rem 0}
.strat-box h3{color:#0F6E56;margin:0 0 .5rem;font-size:1.1rem}
.strat-box p{color:#2C2C2A;margin:0;font-size:.95rem;line-height:1.5}
.disc{background:#F5F5F2;border-radius:8px;padding:.75rem 1rem;font-size:.75rem;
  color:#888780;margin-top:1.5rem;text-align:center}
#MainMenu{visibility:hidden}footer{visibility:hidden}.stDeployButton{display:none}
</style>""", unsafe_allow_html=True)

# ── LLM setup ──
HAS_LLM = False
client = None
try:
    from anthropic import Anthropic
    api_key = st.secrets.get("ANTHROPIC_API_KEY", os.environ.get("ANTHROPIC_API_KEY"))
    if api_key:
        client = Anthropic(api_key=api_key)
        HAS_LLM = True
except Exception:
    pass

def parse_case(text):
    resp = client.messages.create(
        model="claude-sonnet-4-20250514", max_tokens=2000,
        system="""You are a clinical oncology data extractor. Extract structured data from the case.
Return ONLY valid JSON:
{"cancer_type":"string","purity":float,"mutations":[{"gene":"string","variant":"string",
"VAF":float_or_null,"copy_number":int,"drugs":["string"],"CCF":float_or_null}],
"administered_drugs":["string"],"prior_lines":int,"clinical_summary":"1-2 sentences",
"estimated_r0_fraction":float,"estimated_alpha_rs":float,"estimated_d_s":float}
Use oncology expertise. Default purity=0.65. Estimate ecological parameters from context.""",
        messages=[{"role":"user","content":text}])
    t = resp.content[0].text.strip()
    if t.startswith("```"): t = t.split("\n",1)[1]
    if t.endswith("```"): t = t[:-3]
    return json.loads(t)

def generate_report(case_d, sim_r, ms_r, ctb_r):
    ctx = f"""CASE: {json.dumps(case_d)}
MATCHING: MS_classic={ms_r['MS_classic']:.1%}, MS_weighted={ms_r['MS_weighted']:.1%}
Mutations: {json.dumps(ms_r['mutations'])}
SIMULATION: MTD TTP={sim_r['mtd']['TTP']:.0f}d R={sim_r['mtd']['R_fraction_final']:.0%}
AT50 TTP={sim_r['at50']['TTP']:.0f}d R={sim_r['at50']['R_fraction_final']:.0%}
CTB={ctb_r['recommended']['policy']} TTP={ctb_r['recommended']['TTP_days']:.0f}d utility={ctb_r['recommended']['utility']:.4f}"""
    resp = client.messages.create(
        model="claude-sonnet-4-20250514", max_tokens=2500,
        system="""You are an expert oncologist at a computational tumor board. Write a clinical report:
1. CLINICAL SUMMARY (2-3 sentences)
2. GENOMIC ANALYSIS (matching results, clonal hierarchy, significance)
3. EVOLUTIONARY ASSESSMENT (model predictions, strategy tradeoffs)
4. RECOMMENDED STRATEGY (which policy, why, in clinical terms)
5. MONITORING PLAN (what to track, when to reassess)
6. CAVEATS (model limitations)
Write professionally. End with: 'This analysis is model-suggested and requires physician review.'""",
        messages=[{"role":"user","content":ctx}])
    return resp.content[0].text

# ── Header ──
st.markdown("# 🧬 Computational Tumor Board")
st.caption("AI-powered evolutionary analysis for adaptive cancer therapy")

# ── Sidebar ──
with st.sidebar:
    mode = st.radio("Module", ["📋 Clinical Case","🔬 Explorer","🧪 Cohort"], index=0)
    st.markdown("---")
    st.caption("⚠️ Research use only")
    st.caption("[GitHub](https://github.com/rm147747/Adaptive-Theory) · v0.3.0")

# ═══════════════ CLINICAL CASE ═══════════════
if mode == "📋 Clinical Case":

    if not HAS_LLM:
        st.info("Add `ANTHROPIC_API_KEY` in Secrets for AI-powered case interpretation and narrative reports.")

    tab1, tab2 = st.tabs(["📝 Describe case (AI)" if HAS_LLM else "📝 Free text (needs API key)","📊 Structured input"])

    with tab1:
        case_text = st.text_area("Clinical case", height=160,
            placeholder="68yo male, mCRPC, progressing on 2nd-line abiraterone. PSA 42. NGS: AR amp, TP53 R248W (VAF 0.38), BRCA2 splice (VAF 0.12 subclonal). No CNS mets.")
        if case_text and not HAS_LLM:
            st.warning("Free-text requires API key. Use Structured input tab.")

    # Structured defaults
    PRESETS = {
        "mCRPC": ([("AR","amplification",0.0,8,"abiraterone, enzalutamide",0.95),
                    ("TP53","R248W",0.38,2,"",None),
                    ("BRCA2","splice_site",0.12,2,"olaparib, rucaparib",None)],"abiraterone"),
        "HER2+ Breast":([("HER2","amplification",0.0,12,"trastuzumab, pertuzumab, tucatinib",0.95),
                          ("TP53","R248W",0.42,2,"",None),
                          ("PIK3CA","H1047R",0.15,2,"alpelisib",None)],"trastuzumab"),
    }

    with tab2:
        c1, c2 = st.columns(2)
        cancer = c1.selectbox("Cancer type", ["mCRPC","HER2+ Breast","Other"])
        purity = c1.slider("Purity", 0.10, 1.00, 0.65, 0.05)
        r0f = c2.slider("R₀/N₀", 0.01, 0.40, 0.02, 0.01)
        ars = c2.slider("α_RS", 0.3, 2.0, 1.5, 0.1)
        ds = c2.slider("d_S", 0.005, 0.035, 0.018, 0.001)

        preset_muts, preset_drugs = PRESETS.get(cancer, ([("GENE","var",0.3,2,"drug",None)],"drug"))
        n = st.number_input("Alterations", 1, 10, len(preset_muts))
        mutations = []
        for i in range(n):
            dm = preset_muts[i] if i < len(preset_muts) else ("GENE","v",0.2,2,"",None)
            cc = st.columns([2,2,1,1,3])
            g = cc[0].text_input("Gene",dm[0],key=f"g{i}")
            v = cc[1].text_input("Variant",dm[1],key=f"v{i}")
            vf = cc[2].number_input("VAF",0.0,1.0,dm[2],0.01,key=f"vf{i}")
            cn = cc[3].number_input("CN",0,50,dm[3],key=f"cn{i}")
            dr = cc[4].text_input("Drugs",dm[4],key=f"dr{i}")
            drugs = [x.strip() for x in dr.split(",") if x.strip()]
            mutations.append(Mutation(g,v,vf if vf>0 else None,cn,drugs,dm[5]))
        administered = st.text_input("Current drugs", preset_drugs)
        admin_list = [x.strip() for x in administered.split(",") if x.strip()]

    st.markdown("---")
    if st.button("▶ Generate Tumor Board Report", type="primary", use_container_width=True):

        # LLM parsing if available
        case_data = None
        if HAS_LLM and case_text.strip():
            with st.spinner("🧠 AI interpreting clinical case..."):
                try:
                    case_data = parse_case(case_text)
                    mutations = [Mutation(m["gene"],m.get("variant",""),m.get("VAF"),
                                 m.get("copy_number",2),m.get("drugs",[]),m.get("CCF"))
                                 for m in case_data["mutations"]]
                    admin_list = case_data.get("administered_drugs", admin_list)
                    purity = case_data.get("purity", purity)
                    r0f = case_data.get("estimated_r0_fraction", r0f)
                    ars = case_data.get("estimated_alpha_rs", ars)
                    ds = case_data.get("estimated_d_s", ds)
                    st.success(f"Parsed: {case_data.get('clinical_summary','')}")
                except Exception as e:
                    st.warning(f"Parse error: {e}. Using structured input.")

        # Run 4 layers
        ms = compute_matching_scores(mutations, admin_list, purity)
        S0 = (1-r0f)*0.87; R0 = r0f*0.87
        params = LVParams(r_S=0.0278,r_R=0.02,K=1.0,alpha_SR=0.8,alpha_RS=ars,d_S=ds,d_R=0.001)
        sim_m = simulate_euler(params,S0,R0,mtd_policy,t_end=1500)
        sim_a = simulate_euler(params,S0,R0,adaptive_policy(0.50,1.0),t_end=1500)
        ctb_r = ctb_select_policy(params,S0,R0,t_end=1500)
        best = ctb_r["recommended"]
        sim_c = simulate_euler(params,S0,R0,POLICY_SPACE[best["policy"]],t_end=1500)

        dt = sim_c["TTP"]-sim_m["TTP"]
        dr = (1-sim_c["cumulative_dose"]/max(sim_m["cumulative_dose"],1))*100

        # ── Strategy box ──
        st.markdown(f"""<div class="strat-box">
        <h3>📋 CTB Strategy: {best['policy']}</h3>
        <p>Predicted TTP <strong>{sim_c['TTP']:.0f} days</strong>
        ({f"+{dt:.0f}d vs MTD" if dt>0 else "= MTD"}),
        {f"{dr:.0f}% dose reduction" if dr>5 else "similar exposure"},
        resistant fraction {sim_c['R_fraction_final']:.0%} at endpoint.
        Utility score: {best['utility']:.3f}.</p>
        </div>""", unsafe_allow_html=True)

        # ── Metrics ──
        mc = st.columns(4)
        mc[0].metric("MS weighted",f"{ms['MS_weighted']:.0%}")
        mc[1].metric("CTB TTP",f"{sim_c['TTP']:.0f}d",f"{dt:+.0f}d" if dt else None)
        mc[2].metric("Dose saved",f"{dr:.0f}%" if dr>0 else "—")
        mc[3].metric("R fraction",f"{sim_c['R_fraction_final']:.0%}")

        # ── Genomic table ──
        st.markdown("#### 🧬 Genomic analysis")
        st.dataframe([{"Gene":m["gene"],"CCF":f"{m['CCF']:.0%}","Tier":m["clonality"].upper(),
            "Matched":"✅ "+", ".join(m.get("matched_drugs",[])) if m["matched"] else "❌"}
            for m in ms["mutations"]], use_container_width=True, hide_index=True)

        # ── Dynamics plot ──
        st.markdown("#### 📈 Evolutionary dynamics")
        t = sim_m["t"]
        fig = make_subplots(rows=2,cols=1,shared_xaxes=True,row_heights=[.6,.4],vertical_spacing=.06)
        for s,n,c in [(sim_m,"MTD",CR),(sim_a,"AT50",CI),(sim_c,f"CTB ({best['policy']})",CP)]:
            w = 2.5 if "CTB" in n else 1.5
            fig.add_trace(go.Scatter(x=t,y=s["N"],name=n,line=dict(color=c,width=w)),row=1,col=1)
            fig.add_trace(go.Scatter(x=t,y=s["R"]/s["N"],showlegend=False,line=dict(color=c,width=w)),row=2,col=1)
        fig.update_layout(height=420,template="plotly_white",margin=dict(l=50,r=20,t=30,b=40),
            legend=dict(orientation="h",yanchor="bottom",y=1.02,xanchor="center",x=.5))
        fig.update_yaxes(title_text="Burden",range=[.2,1.05],row=1,col=1)
        fig.update_yaxes(title_text="R/N",range=[0,1.05],row=2,col=1)
        fig.update_xaxes(title_text="Days",row=2,col=1)
        st.plotly_chart(fig, use_container_width=True)

        # ── Policy ranking ──
        st.markdown("#### 🏆 Policy ranking")
        st.dataframe([{"":"⭐" if p["policy"]==best["policy"] else "","Policy":p["policy"],
            "TTP":f"{p['TTP_days']:.0f}d","Dose":f"{p['cumulative_dose']:.0f}",
            "R%":f"{p['R_fraction_final']:.0%}","Utility":f"{p['utility']:.3f}"}
            for p in ctb_r["all_policies"][:6]], use_container_width=True, hide_index=True)
        st.caption("U = TTP/T_max − 0.3·Dose/T_max − 0.2·R_final. A priori selection.")

        # ── LLM Narrative ──
        if HAS_LLM:
            st.markdown("#### 📝 Tumor board report")
            with st.spinner("Generating narrative report..."):
                try:
                    cd = case_data or {"cancer_type":cancer,"mutations":[m.__dict__ for m in mutations]}
                    sr = {k:{"TTP":v["TTP"],"cumulative_dose":v["cumulative_dose"],
                          "R_fraction_final":v["R_fraction_final"]}
                          for k,v in {"mtd":sim_m,"at50":sim_a,"ctb":sim_c}.items()}
                    report = generate_report(cd, sr, ms, ctb_r)
                    st.markdown(report)
                except Exception as e:
                    st.error(f"Report error: {e}")

        st.markdown('<div class="disc">⚠️ Research prototype. Model-suggested strategies only. '
                    'Not for clinical decisions. <a href="https://github.com/rm147747/Adaptive-Theory">GitHub</a></div>',
                    unsafe_allow_html=True)

# ═══════════════ EXPLORER ═══════════════
elif mode == "🔬 Explorer":
    st.markdown("### Parameter Explorer")
    c1,c2,c3 = st.columns(3)
    r0=c1.slider("R₀/N₀",0.01,0.40,0.02,0.01); ars=c2.slider("α_RS",0.3,2.0,1.5,0.1)
    ds=c3.slider("d_S",0.005,0.035,0.018,0.001)
    S0=(1-r0)*.87; R0=r0*.87
    params=LVParams(r_S=0.0278,r_R=0.02,K=1,alpha_SR=0.8,alpha_RS=ars,d_S=ds,d_R=0.001)
    pols={"MTD":(mtd_policy,CR),"AT50":(adaptive_policy(.5,1),CI),"AT30":(adaptive_policy(.3,.9),CU),"Metro 50%":(POLICY_SPACE["metro_50"],CG)}
    ctb=ctb_select_policy(params,S0,R0,t_end=1500)
    bn=ctb["recommended"]["policy"]
    fig=make_subplots(rows=2,cols=1,shared_xaxes=True,vertical_spacing=.08,subplot_titles=("Tumor burden","Resistant fraction"))
    for nm,(fn,cl) in pols.items():
        s=simulate_euler(params,S0,R0,fn,t_end=1500);w=2.5 if nm==bn else 1.5
        fig.add_trace(go.Scatter(x=s["t"],y=s["N"],name=nm,line=dict(color=cl,width=w)),row=1,col=1)
        fig.add_trace(go.Scatter(x=s["t"],y=s["R"]/s["N"],showlegend=False,line=dict(color=cl,width=w)),row=2,col=1)
    fig.update_layout(height=500,template="plotly_white",legend=dict(orientation="h",yanchor="bottom",y=1.02))
    fig.update_yaxes(range=[.2,1.05],row=1,col=1);fig.update_yaxes(range=[0,1.05],row=2,col=1)
    fig.update_xaxes(title_text="Days",row=2,col=1)
    st.plotly_chart(fig,use_container_width=True)
    st.success(f"**CTB selects: {bn}** (utility: {ctb['recommended']['utility']:.3f})")

# ═══════════════ COHORT ═══════════════
elif mode == "🧪 Cohort":
    st.markdown("### Virtual Cohort Benchmark")
    np_n=st.slider("Patients",50,500,200,50)
    if st.button("▶ Run",type="primary"):
        np.random.seed(42);bar=st.progress(0);ch=[];cu=[];mu=[]
        for i in range(np_n):
            if i%20==0:bar.progress(i/np_n)
            p=LVParams(r_S=np.random.uniform(.02,.035),r_R=np.random.uniform(.015,.04),K=1,alpha_SR=0.8,
                alpha_RS=np.random.uniform(.5,1.5),d_S=np.random.uniform(.012,.025),
                d_R=np.random.uniform(.012,.025)*np.random.uniform(0,.7))
            rf=np.random.uniform(.01,.35);s0=(1-rf)*.87;r0=rf*.87
            r=ctb_select_policy(p,s0,r0,t_end=1500)
            ch.append(r["recommended"]["policy"]);cu.append(r["recommended"]["utility"])
            mx=[x for x in r["all_policies"] if x["policy"]=="MTD"]
            mu.append(mx[0]["utility"] if mx else 0)
        bar.progress(1.0)
        from collections import Counter;cnt=Counter(ch)
        c1,c2=st.columns(2)
        c1.metric("CTB median utility",f"{np.median(cu):.3f}")
        c1.metric("MTD median utility",f"{np.median(mu):.3f}")
        c2.metric("CTB beats MTD",f"{sum(1 for a,b in zip(cu,mu) if a>b)}/{np_n}")
        fig=go.Figure(go.Pie(labels=list(cnt.keys()),values=list(cnt.values()),hole=.4,textinfo="label+percent"))
        fig.update_layout(height=350,title="CTB policy selection")
        st.plotly_chart(fig,use_container_width=True)
        st.info(f"No single fixed policy optimal across {np_n} patients. CTB adapts to each tumor ecology.")
