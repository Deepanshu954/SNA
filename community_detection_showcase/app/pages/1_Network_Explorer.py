"""
Page 1 — Network Explorer
Displays precomputed network visualization, statistics, and degree distribution.
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

st.set_page_config(page_title="Network Explorer", page_icon="🌐", layout="wide")

css_path = Path(__file__).parent.parent / "assets" / "style.css"
if css_path.exists():
    st.markdown(f"<style>{css_path.read_text()}</style>", unsafe_allow_html=True)

PRECOMP = PROJECT_ROOT / "data" / "precomputed"


@st.cache_data
def load_data():
    with open(PRECOMP / "summary.json") as f:
        summary = json.load(f)
    with open(PRECOMP / "stats.json") as f:
        stats = json.load(f)
    return summary, stats


if not (PRECOMP / "summary.json").exists():
    st.error("Run `python3 precompute.py` first!")
    st.stop()

summary, stats = load_data()

st.markdown("# 🌐 Network Explorer")
st.markdown("Explore the Facebook Social Circles network — 4,039 nodes, 88,234 edges.")
st.markdown("---")

# ── Stats Cards ─────────────────────────────────────────────────────────
st.markdown("### 📊 Network Statistics")
c1, c2, c3, c4, c5 = st.columns(5)

for col, label, val in [
    (c1, "Nodes", f'{stats["nodes"]:,}'),
    (c2, "Edges", f'{stats["edges"]:,}'),
    (c3, "Avg Degree", str(stats["avg_degree"])),
    (c4, "Clustering", str(stats["avg_clustering"])),
    (c5, "Density", str(stats["density"])),
]:
    with col:
        st.markdown(
            f'<div class="metric-card"><h3>{label}</h3><p>{val}</p></div>',
            unsafe_allow_html=True,
        )

st.markdown("---")

# ── Network Visualization ──────────────────────────────────────────────
st.markdown("### 🕸️ Network Visualization")

tab1, tab2 = st.tabs(["📍 Top 500 Nodes (Clear)", "🌍 Full Network (All 4,039)"])

with tab1:
    img = PRECOMP / "fig_network.png"
    if img.exists():
        st.image(str(img), caption="Community-coloured network — Top 500 nodes by degree",
                 use_container_width=True)
    else:
        st.warning("Network image not found. Run precompute.py")

with tab2:
    img = PRECOMP / "fig_network_full.png"
    if img.exists():
        st.image(str(img), caption="Complete Facebook Social Network — all 4,039 nodes",
                 use_container_width=True)
    else:
        st.warning("Full network image not found. Run precompute.py")

st.markdown("---")

# ── Degree Distribution ────────────────────────────────────────────────
st.markdown("### 📈 Degree Distribution")

# Load graph for degree dist
from src.graph_utils import load_graph, degree_distribution

@st.cache_data
def get_degree_data():
    G = load_graph(str(PROJECT_ROOT / "data" / "facebook_combined.txt"))
    return degree_distribution(G)

degrees, counts = get_degree_data()

fig = go.Figure()
fig.add_trace(go.Bar(
    x=degrees.tolist(), y=counts.tolist(),
    marker_color="#1A3C5E",
    marker_line_color="white", marker_line_width=0.5,
))
fig.update_layout(
    title="Degree Distribution (Log-Log Scale)",
    xaxis_title="Degree", yaxis_title="Count",
    xaxis_type="log", yaxis_type="log",
    template="plotly_white", height=400,
)
st.plotly_chart(fig, use_container_width=True)

# Also show the precomputed figure
img = PRECOMP / "fig1_degree_dist.png"
if img.exists():
    with st.expander("📷 Figure 1 from Paper"):
        st.image(str(img), caption="Figure 1: Log-log degree distribution confirming scale-free topology",
                 use_container_width=True)

st.markdown("---")

# ── Top Hub Nodes ───────────────────────────────────────────────────────
st.markdown("### 🏆 Top 10 Hub Nodes")
hubs = stats.get("top_hubs", [])
if hubs:
    hub_df = pd.DataFrame(hubs, columns=["Node ID", "Degree"])
    hub_df.index = range(1, len(hub_df) + 1)
    hub_df.index.name = "Rank"
    st.dataframe(hub_df, use_container_width=True)

st.markdown(
    '<div class="info-callout"><strong>Scale-Free Property:</strong> '
    "The network follows a power-law degree distribution — a few hub nodes "
    "have extremely high connectivity while most nodes have few connections. "
    "This is characteristic of real-world social networks.</div>",
    unsafe_allow_html=True,
)
