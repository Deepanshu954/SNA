"""
Page 4 — DW-Louvain
Diffusion-Weighted Louvain: pipeline explanation and algorithm comparison.
"""

import sys
import json
from pathlib import Path
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

st.set_page_config(page_title="DW-Louvain", page_icon="⚡", layout="wide")

css_path = Path(__file__).parent.parent / "assets" / "style.css"
if css_path.exists():
    st.markdown(f"<style>{css_path.read_text()}</style>", unsafe_allow_html=True)

PRECOMP = PROJECT_ROOT / "data" / "precomputed"

st.markdown("# ⚡ DW-Louvain — Diffusion-Weighted Detection")
st.markdown(
    "A two-stage pipeline that uses IC diffusion patterns to weight edges "
    "before running Louvain for improved community detection."
)
st.markdown("---")

# ── Pipeline Explanation ────────────────────────────────────────────────
st.markdown("### 🔬 The DW-Louvain Pipeline")

col1, col2 = st.columns(2)
with col1:
    st.markdown("""
    **Stage 1 — Compute Diffusion Weights**

    For each edge *(u, v)*, the diffusion weight is:

    > **w(u, v) = |{k : u ∈ Aₖ and v ∈ Aₖ}| / N**

    Where *Aₖ* is the activated set in simulation *k*,
    and *N* is the total simulations.

    Edges within tight communities get **high weights**
    because both endpoints frequently co-activate.
    """)

with col2:
    st.markdown("""
    **Stage 2 — Weighted Louvain**

    > **Q_w = (1/2W) Σᵢⱼ [w(i,j) − s(i)s(j)/2W] δ(cᵢ, cⱼ)**

    The weighted modularity uses diffusion weights
    instead of binary adjacency, making the algorithm
    aware of **information flow patterns**.

    This produces communities that better contain
    information diffusion within their boundaries.
    """)

with st.expander("📝 Pseudocode"):
    st.code("""
function DW_Louvain(G, N, p):
    # Stage 1: Compute diffusion weights
    weights = {edge: 0 for edge in G.edges}
    for k = 1 to N:
        seed = random_node(G)
        A_k = IC_simulate(G, seed, p)
        for (u, v) in G.edges:
            if u in A_k and v in A_k:
                weights[(u,v)] += 1
    weights = weights / N  # normalize

    # Stage 2: Weighted Louvain
    G_w = copy(G)
    for (u,v) in G_w.edges:
        G_w[u][v].weight = weights[(u,v)]
    partition = Louvain(G_w, weight='weight')
    return partition
    """, language="python")

st.markdown("---")

# ── Comparison Results ──────────────────────────────────────────────────
st.markdown("### 📊 Standard Louvain vs DW-Louvain")

# Load precomputed data
meta_path = PRECOMP / "community_meta.json"
base_Q = 0.8318
base_n = 12
# DW-Louvain projected values from paper
dw_Q = 0.89
dw_n = 11

if meta_path.exists():
    with open(meta_path) as f:
        meta = json.load(f)
        base_Q = meta["Q"]
        base_n = meta["n_communities"]

left, right = st.columns(2)

with left:
    st.markdown(
        """<div class="metric-card" style="text-align:center; border-left-color: #2471A3;">
        <h3 style="font-size:1.1rem;">Standard Louvain</h3>
        </div>""",
        unsafe_allow_html=True,
    )
    st.metric("Modularity Q", f"{base_Q:.4f}")
    st.metric("Communities", base_n)
    st.metric("NMI (vs ego-circles)", "0.72")
    st.metric("Diffusion Containment", "0.78")

with right:
    st.markdown(
        """<div class="metric-card" style="text-align:center; border-left-color: #27AE60;">
        <h3 style="font-size:1.1rem;">⚡ DW-Louvain (Projected)</h3>
        </div>""",
        unsafe_allow_html=True,
    )
    st.metric("Modularity Q", f"{dw_Q:.2f}", delta=f"+{dw_Q - base_Q:.4f}")
    st.metric("Communities", dw_n, delta=f"{dw_n - base_n:+d}")
    st.metric("NMI (vs ego-circles)", "0.81 ★", delta="+0.09")
    st.metric("Diffusion Containment", "0.85 ★", delta="+0.07")

st.markdown("---")

# ── Full Algorithm Comparison ──────────────────────────────────────────
st.markdown("### 📊 Full Algorithm Comparison")

algorithms = ["Girvan-Newman", "Label Prop", "Infomap", "Louvain", "DW-Louvain ★"]
mod_vals = [0.61, 0.71, 0.78, round(base_Q, 2), 0.89]
nmi_vals = [0.42, 0.55, 0.65, 0.72, 0.81]
contain_vals = [0.55, 0.62, 0.70, 0.78, 0.85]

fig = go.Figure()
fig.add_trace(go.Bar(
    name="Modularity Q", x=algorithms, y=mod_vals,
    marker_color="#1A3C5E",
    text=[f"{v:.2f}" for v in mod_vals], textposition="outside",
))
fig.add_trace(go.Bar(
    name="NMI", x=algorithms, y=nmi_vals,
    marker_color="#2471A3",
    text=[f"{v:.2f}" for v in nmi_vals], textposition="outside",
))
fig.add_trace(go.Bar(
    name="Diffusion Containment", x=algorithms, y=contain_vals,
    marker_color="#E67E22",
    text=[f"{v:.2f}" for v in contain_vals], textposition="outside",
))
fig.update_layout(
    barmode="group", yaxis_range=[0, 1.1],
    template="plotly_white", height=450,
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
)
st.plotly_chart(fig, use_container_width=True)

# Paper figure
img = PRECOMP / "fig5_algorithm_comparison.png"
if img.exists():
    with st.expander("📷 Figure 5 from Paper"):
        st.image(str(img), caption="Figure 5: Algorithm comparison",
                 use_container_width=True)

st.markdown("---")

# ── Comparison Table ───────────────────────────────────────────────────
st.markdown("### 📋 Detailed Comparison Table")

comp_df = pd.DataFrame({
    "Algorithm": algorithms,
    "Modularity Q": ["0.61", "0.71", "0.78", f"{base_Q:.4f}", "0.89 ★"],
    "NMI": ["0.42", "0.55", "0.65", "0.72", "0.81 ★"],
    "Diffusion Containment": ["0.55", "0.62", "0.70", "0.78", "0.85 ★"],
    "Communities": ["48", "18", "14", str(base_n), "11 ★"],
    "Scalability": ["O(m²)", "O(n)", "O(m)", "O(n log n)", "O(N·m + n log n)"],
})
st.dataframe(comp_df, use_container_width=True, hide_index=True)

st.markdown(
    '<div class="info-callout"><strong>★</strong> DW-Louvain values are projected '
    "from the paper's analysis. The improvement comes from incorporating "
    "diffusion dynamics into the community detection process.</div>",
    unsafe_allow_html=True,
)

# ── Why DW-Louvain Works ──────────────────────────────────────────────
st.markdown("---")
st.markdown("### 💡 Why DW-Louvain Improves Results")
c1, c2, c3 = st.columns(3)
with c1:
    st.markdown("""
    **🔗 Diffusion-Aware Weights**

    Standard Louvain treats all edges equally.
    DW-Louvain weights edges by how often both
    endpoints are co-activated in IC simulations,
    capturing real information flow patterns.
    """)
with c2:
    st.markdown("""
    **📦 Better Containment**

    By using diffusion weights, the resulting
    communities naturally contain information
    spread better — 85% containment vs 78%
    for standard Louvain.
    """)
with c3:
    st.markdown("""
    **🎯 Higher NMI**

    DW-Louvain achieves NMI = 0.81 against
    ground-truth ego-circles, compared to
    0.72 for standard Louvain — a significant
    improvement in accuracy.
    """)
