"""
Page 1 — Network Explorer
Interactive network visualization, statistics, and degree distribution.
"""

import sys
from pathlib import Path
import streamlit as st
import streamlit.components.v1 as components
import plotly.graph_objects as go
import pandas as pd
import tempfile

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.graph_utils import load_graph, get_stats, get_top_hubs, degree_distribution

st.set_page_config(page_title="Network Explorer", page_icon="🌐", layout="wide")

# Load CSS
css_path = Path(__file__).parent.parent / "assets" / "style.css"
if css_path.exists():
    st.markdown(f"<style>{css_path.read_text()}</style>", unsafe_allow_html=True)


@st.cache_data
def cached_load_graph():
    return load_graph(str(PROJECT_ROOT / "data" / "facebook_combined.txt"))


st.markdown('<h1 class="section-header">🌐 Network Explorer</h1>', unsafe_allow_html=True)
st.markdown("Explore the Facebook Social Circles network interactively.")
st.markdown("---")

G = cached_load_graph()
stats = get_stats(G)

# ── Sidebar ─────────────────────────────────────────────────────────────
st.sidebar.header("🔧 Display Settings")
max_nodes_option = st.sidebar.selectbox(
    "Max nodes to display",
    [200, 500, 1000, "All"],
    index=0,
)
layout_seed = st.sidebar.slider("Spring layout seed", 1, 100, 42)

max_nodes = G.number_of_nodes() if max_nodes_option == "All" else int(max_nodes_option)

# ── Stats Cards ─────────────────────────────────────────────────────────
st.markdown("### 📊 Network Statistics")
c1, c2, c3, c4, c5 = st.columns(5)

with c1:
    st.markdown(
        f'<div class="metric-card"><h3>Nodes</h3><p>{stats["nodes"]:,}</p></div>',
        unsafe_allow_html=True,
    )
with c2:
    st.markdown(
        f'<div class="metric-card"><h3>Edges</h3><p>{stats["edges"]:,}</p></div>',
        unsafe_allow_html=True,
    )
with c3:
    st.markdown(
        f'<div class="metric-card"><h3>Avg Degree</h3><p>{stats["avg_degree"]}</p></div>',
        unsafe_allow_html=True,
    )
with c4:
    st.markdown(
        f'<div class="metric-card"><h3>Clustering</h3><p>{stats["avg_clustering"]}</p></div>',
        unsafe_allow_html=True,
    )
with c5:
    st.markdown(
        f'<div class="metric-card"><h3>Density</h3><p>{stats["density"]}</p></div>',
        unsafe_allow_html=True,
    )

st.markdown("---")

# ── Interactive Graph (Pyvis) ───────────────────────────────────────────
st.markdown("### 🕸️ Interactive Network Graph")
with st.spinner("Building interactive graph..."):
    from pyvis.network import Network
    import networkx as nx

    # Sample nodes for display
    if max_nodes < G.number_of_nodes():
        # Get the top nodes by degree for a representative sample
        sorted_nodes = sorted(G.degree(), key=lambda x: x[1], reverse=True)
        selected = [n for n, _ in sorted_nodes[:max_nodes]]
        G_sample = G.subgraph(selected).copy()
    else:
        G_sample = G

    net = Network(height="550px", width="100%", bgcolor="#fafafa",
                  font_color="#333333")
    net.toggle_physics(True)
    net.barnes_hut(spring_length=100, spring_strength=0.01)

    # Scale node sizes by degree
    max_deg = max(dict(G_sample.degree()).values()) if G_sample.nodes() else 1
    for node in G_sample.nodes():
        deg = G_sample.degree(node)
        size = 5 + (deg / max_deg) * 25
        net.add_node(node, label=str(node), size=size,
                     color="#2471A3", title=f"Node {node}\nDegree: {deg}")

    for u, v in G_sample.edges():
        net.add_edge(u, v, color="#CCCCCC")

    # Save and display
    with tempfile.NamedTemporaryFile(suffix=".html", delete=False, mode="w") as f:
        net.save_graph(f.name)
        html_path = f.name

    with open(html_path, "r") as f:
        html_content = f.read()

    components.html(html_content, height=570, scrolling=False)

    st.caption(f"Showing {G_sample.number_of_nodes()} nodes, {G_sample.number_of_edges()} edges")

    # Download button
    st.download_button(
        "📥 Download Graph as HTML",
        data=html_content,
        file_name="network_graph.html",
        mime="text/html",
    )

st.markdown("---")

# ── Degree Distribution (Plotly) ────────────────────────────────────────
st.markdown("### 📈 Degree Distribution")
degrees, counts = degree_distribution(G)

fig = go.Figure()
fig.add_trace(go.Bar(
    x=degrees.tolist(),
    y=counts.tolist(),
    marker_color="#1A3C5E",
    marker_line_color="white",
    marker_line_width=0.5,
))
fig.update_layout(
    title="Degree Distribution (Log-Log Scale)",
    xaxis_title="Degree",
    yaxis_title="Count",
    xaxis_type="log",
    yaxis_type="log",
    template="plotly_white",
    height=400,
)
st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# ── Top Hub Nodes ───────────────────────────────────────────────────────
st.markdown("### 🏆 Top 10 Hub Nodes")
hubs = get_top_hubs(G, n=10)
hub_df = pd.DataFrame(hubs, columns=["Node ID", "Degree"])
hub_df.index = range(1, len(hub_df) + 1)
hub_df.index.name = "Rank"
st.dataframe(hub_df, width="stretch")
