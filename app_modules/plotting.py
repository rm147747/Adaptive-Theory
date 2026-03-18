"""
Plotting utilities for the CTB Streamlit app.

All charts use Plotly for interactivity.
Color palette is consistent across all modules.
"""

import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Consistent color palette
COLORS = {
    "primary": "#1D9E75",   # CTB green
    "danger": "#E24B4A",    # MTD red
    "info": "#378ADD",      # AT50 blue
    "purple": "#7F77DD",    # AT30
    "gray": "#888780",      # metronomic
    "muted": "#888780",
}


def plot_dynamics(sim_mtd, sim_at50, sim_ctb, best_policy_name: str):
    """
    Create a 2-panel dynamics plot: tumor burden + resistant fraction.

    Args:
        sim_mtd, sim_at50, sim_ctb: simulation result dicts from simulate_euler
        best_policy_name: name of the CTB-selected policy

    Returns:
        plotly Figure
    """
    t = sim_mtd["t"]

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[0.6, 0.4], vertical_spacing=0.06,
    )

    # Panel A: Tumor burden
    traces = [
        ("MTD", sim_mtd, COLORS["danger"], 1.5),
        ("AT50", sim_at50, COLORS["info"], 1.5),
        (f"CTB ({best_policy_name})", sim_ctb, COLORS["primary"], 2.5),
    ]

    for name, sim, color, width in traces:
        fig.add_trace(
            go.Scatter(x=t, y=sim["N"], name=name,
                       line=dict(color=color, width=width)),
            row=1, col=1,
        )
        fig.add_trace(
            go.Scatter(x=t, y=sim["R"] / sim["N"], showlegend=False,
                       line=dict(color=color, width=width)),
            row=2, col=1,
        )

    fig.update_layout(
        height=420, template="plotly_white",
        margin=dict(l=50, r=20, t=30, b=40),
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="center", x=0.5),
    )
    fig.update_yaxes(title_text="Tumor burden N(t)", range=[0.2, 1.05], row=1, col=1)
    fig.update_yaxes(title_text="Resistant fraction R/N", range=[0, 1.05], row=2, col=1)
    fig.update_xaxes(title_text="Time (days)", row=2, col=1)

    return fig


def plot_cohort_pie(counts: dict):
    """Create a pie chart of CTB policy selections across a virtual cohort."""
    fig = go.Figure(
        go.Pie(
            labels=list(counts.keys()),
            values=list(counts.values()),
            hole=0.4,
            textinfo="label+percent",
        )
    )
    fig.update_layout(height=350, title="CTB policy selection distribution")
    return fig
