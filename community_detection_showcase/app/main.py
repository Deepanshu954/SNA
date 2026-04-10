"""
main.py — Streamlit entry point for Community Detection Showcase.
Loads precomputed data for instant display.
"""

import sys
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

# ── Page Config ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Community Detection Showcase",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load CSS
css_path = Path(__file__).parent / "assets" / "style.css"
if css_path.exists():
    st.markdown(f"<style>{css_path.read_text()}</style>", unsafe_allow_html=True)

PRECOMP = PROJECT_ROOT / "data" / "precomputed"


@st.cache_data
def load_summary():
    with open(PRECOMP / "summary.json") as f:
        return json.load(f)


# ── Check precomputed data exists ───────────────────────────────────────
if not (PRECOMP / "summary.json").exists():
    st.error("⚠️ Precomputed data not found! Run `python3 precompute.py` first.")
    st.code("cd community_detection_showcase && python3 precompute.py", language="bash")
    st.stop()

summary = load_summary()

# ── Landing Page ────────────────────────────────────────────────────────
st.markdown("# 🔬 Community Detection in Social Networks")
st.markdown("### Interactive Showcase — Information Diffusion Models")
st.markdown("---")

st.markdown(
    """
    <div class="info-callout">
    <strong>Research Paper:</strong> This project explores community detection
    in social networks using information diffusion models. We apply the <em>Louvain algorithm</em>
    to the <strong>Facebook Social Circles</strong> dataset (~4,039 nodes, ~88,000 edges) and
    analyze how information propagates within and across detected communities using the
    <strong>Independent Cascade (IC)</strong> model. We further propose <strong>DW-Louvain</strong>,
    a diffusion-weighted variant that leverages diffusion patterns to improve community quality.
    </div>
    """,
    unsafe_allow_html=True,
)

# ── KPI Cards ───────────────────────────────────────────────────────────
st.markdown("### 📊 Dataset Overview")
c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown(
        f'<div class="metric-card"><h3>Nodes</h3><p>{summary["nodes"]:,}</p></div>',
        unsafe_allow_html=True,
    )
with c2:
    st.markdown(
        f'<div class="metric-card"><h3>Edges</h3><p>{summary["edges"]:,}</p></div>',
        unsafe_allow_html=True,
    )
with c3:
    st.markdown(
        f'<div class="metric-card"><h3>Modularity Q</h3><p>{summary["modularity_Q"]}</p></div>',
        unsafe_allow_html=True,
    )
with c4:
    st.markdown(
        f'<div class="metric-card"><h3>Communities</h3><p>{summary["n_communities"]}</p></div>',
        unsafe_allow_html=True,
    )

st.markdown("---")

c1, c2, c3 = st.columns(3)
c1.metric("Average Degree", summary["avg_degree"])
c2.metric("Clustering Coefficient", summary["avg_clustering"])
c3.metric("Network Density", summary["density"])

# ── Preview: Network Graph ──────────────────────────────────────────────
st.markdown("---")
st.markdown("### 🌐 Network Preview")
net_img = PRECOMP / "fig_network.png"
if net_img.exists():
    st.image(str(net_img), caption="Facebook Social Network — Community Structure (Top 500 Nodes)",
             use_container_width=True)

# ── Navigation ──────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    """
    <div class="success-callout">
    <strong>👈 Navigate</strong> using the sidebar to explore:
    <ol>
        <li><strong>Network Explorer</strong> — interactive graph visualization & statistics</li>
        <li><strong>Community Detection</strong> — Louvain algorithm results</li>
        <li><strong>Diffusion Simulator</strong> — IC model simulation & analysis</li>
        <li><strong>DW-Louvain</strong> — diffusion-weighted community detection</li>
        <li><strong>Results Dashboard</strong> — summary of all findings</li>
    </ol>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    '<div class="footer">Community Detection Showcase · '
    "Harshit Singh Shakya, Manjeet Singh Jhakar, Deepanshu Chauhan · "
    "Bennett University 2024-25</div>",
    unsafe_allow_html=True,
)
