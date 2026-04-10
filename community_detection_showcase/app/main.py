"""
main.py — Streamlit entry point for Community Detection Showcase.

Run with: streamlit run app/main.py
"""

import sys
from pathlib import Path

# Add project root to sys.path so 'src' can be imported
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
from src.graph_utils import load_graph, get_stats

# ── Page Configuration ──────────────────────────────────────────────────
st.set_page_config(
    page_title="Community Detection Showcase",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load custom CSS
css_path = Path(__file__).parent / "assets" / "style.css"
if css_path.exists():
    st.markdown(f"<style>{css_path.read_text()}</style>", unsafe_allow_html=True)

# ── Cache the graph ─────────────────────────────────────────────────────
@st.cache_data
def cached_load_graph():
    data_path = PROJECT_ROOT / "data" / "facebook_combined.txt"
    return load_graph(str(data_path))


@st.cache_data
def cached_get_stats(_G):
    return get_stats(_G)


# ── Landing Page ────────────────────────────────────────────────────────
st.markdown(
    '<h1 class="section-header">🔬 Community Detection in Social Networks</h1>',
    unsafe_allow_html=True,
)
st.markdown(
    "### Interactive Showcase — Information Diffusion Models",
)
st.markdown("---")

# Abstract
st.markdown(
    """
    <div class="info-callout">
    <strong>Research Paper Abstract:</strong> This project explores community detection
    in social networks using information diffusion models. We apply the <em>Louvain algorithm</em>
    to the <strong>Facebook Social Circles</strong> dataset (~4,039 nodes, ~88,000 edges) and
    analyze how information propagates within and across detected communities using the
    <strong>Independent Cascade (IC)</strong> model. We further propose <strong>DW-Louvain</strong>,
    a diffusion-weighted variant that leverages diffusion patterns to improve community quality.
    </div>
    """,
    unsafe_allow_html=True,
)

# Load graph and stats
try:
    G = cached_load_graph()
    stats = cached_get_stats(G)

    # KPI Cards
    st.markdown("### 📊 Dataset Overview")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.markdown(
            f"""<div class="metric-card">
            <h3>Nodes</h3><p>{stats['nodes']:,}</p>
            </div>""",
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f"""<div class="metric-card">
            <h3>Edges</h3><p>{stats['edges']:,}</p>
            </div>""",
            unsafe_allow_html=True,
        )
    with col3:
        st.markdown(
            f"""<div class="metric-card">
            <h3>Modularity Q</h3><p>0.8318</p>
            </div>""",
            unsafe_allow_html=True,
        )
    with col4:
        st.markdown(
            f"""<div class="metric-card">
            <h3>Communities</h3><p>12</p>
            </div>""",
            unsafe_allow_html=True,
        )

    st.markdown("---")

    # Quick stats
    c1, c2, c3 = st.columns(3)
    c1.metric("Average Degree", stats["avg_degree"])
    c2.metric("Clustering Coefficient", stats["avg_clustering"])
    c3.metric("Network Density", stats["density"])

except FileNotFoundError as e:
    st.error(f"⚠️ Dataset not found: {e}")
    st.info(
        "Please download `facebook_combined.txt` from "
        "[SNAP Stanford](https://snap.stanford.edu/data/ego-Facebook.html) "
        "or [Kaggle](https://www.kaggle.com/datasets/pypiahmad/social-circles) "
        "and place it in the `data/` folder."
    )

# Navigation hint
st.markdown("---")
st.markdown(
    """
    <div class="success-callout">
    <strong>👈 Navigate</strong> using the sidebar to explore:
    <ol>
        <li><strong>Network Explorer</strong> — interactive graph visualization</li>
        <li><strong>Community Detection</strong> — Louvain algorithm with tunable resolution</li>
        <li><strong>Diffusion Simulator</strong> — IC model simulation and analysis</li>
        <li><strong>DW-Louvain</strong> — diffusion-weighted community detection</li>
        <li><strong>Results Dashboard</strong> — summary of all findings</li>
    </ol>
    </div>
    """,
    unsafe_allow_html=True,
)

# Footer
st.markdown(
    '<div class="footer">Community Detection Showcase · Based on research by '
    "Harshit Singh Shakya, Manjeet Singh Jhakar, Deepanshu Chauhan · "
    "Bennett University 2024-25</div>",
    unsafe_allow_html=True,
)
