"""
Page 2 — Community Detection
Run Louvain algorithm with tunable resolution, visualize communities.
"""

import sys
from pathlib import Path
import streamlit as st
import streamlit.components.v1 as components
import plotly.graph_objects as go
import pandas as pd
import tempfile
import matplotlib.cm as cm
import matplotlib.colors as mcolors

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.graph_utils import load_graph
from src.community_detection import detect_communities, community_stats, get_community_colors

st.set_page_config(page_title="Community Detection", page_icon="🏘️", layout="wide")

css_path = Path(__file__).parent.parent / "assets" / "style.css"
if css_path.exists():
    st.markdown(f"<style>{css_path.read_text()}</style>", unsafe_allow_html=True)


@st.cache_data
def cached_load_graph():
    return load_graph(str(PROJECT_ROOT / "data" / "facebook_combined.txt"))


st.markdown('<h1 class="section-header">🏘️ Community Detection</h1>', unsafe_allow_html=True)
st.markdown("Run the Louvain algorithm to detect communities in the Facebook network.")
st.markdown("---")

G = cached_load_graph()

# ── Sidebar ─────────────────────────────────────────────────────────────
st.sidebar.header("🔧 Louvain Parameters")
resolution = st.sidebar.slider("Resolution parameter", 0.5, 2.0, 1.0, step=0.1)
seed = st.sidebar.number_input("Random seed", value=42, min_value=0, max_value=9999)
run_button = st.sidebar.button("🚀 Detect Communities", type="primary", width="stretch")

# ── Session state for partition ─────────────────────────────────────────
if "partition" not in st.session_state:
    st.session_state["partition"] = None
    st.session_state["modularity"] = None

if run_button:
    with st.spinner("Running Louvain community detection..."):
        partition, modularity = detect_communities(G, resolution=resolution, seed=int(seed))
        st.session_state["partition"] = partition
        st.session_state["modularity"] = modularity

partition = st.session_state["partition"]
modularity = st.session_state["modularity"]

if partition is None:
    st.info("👈 Click **Detect Communities** in the sidebar to run the Louvain algorithm.")
    st.stop()

# ── Results ─────────────────────────────────────────────────────────────
n_communities = len(set(partition.values()))
comm_sizes = {}
for node, cid in partition.items():
    comm_sizes.setdefault(cid, 0)
    comm_sizes[cid] += 1
largest = max(comm_sizes.values())
smallest = min(comm_sizes.values())

# Metric row
st.markdown("### 📊 Detection Results")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Modularity Q", f"{modularity:.4f}")
c2.metric("Communities", n_communities)
c3.metric("Largest Community", f"{largest} nodes")
c4.metric("Smallest Community", f"{smallest} nodes")

st.markdown("---")

# ── Community-coloured graph + bar chart ────────────────────────────────
left_col, right_col = st.columns([3, 2])

with left_col:
    st.markdown("### 🕸️ Community-Coloured Network")
    with st.spinner("Building community graph..."):
        from pyvis.network import Network

        node_colors = get_community_colors(partition)

        # Sample top 500 nodes for display performance
        sorted_nodes = sorted(G.degree(), key=lambda x: x[1], reverse=True)
        selected = [n for n, _ in sorted_nodes[:500]]
        G_sample = G.subgraph(selected).copy()

        net = Network(height="500px", width="100%", bgcolor="#fafafa",
                      font_color="#333333")
        net.barnes_hut(spring_length=80)

        max_deg = max(dict(G_sample.degree()).values()) if G_sample.nodes() else 1
        for node in G_sample.nodes():
            deg = G_sample.degree(node)
            size = 5 + (deg / max_deg) * 25
            color = node_colors.get(node, "#999999")
            comm_id = partition.get(node, -1)
            net.add_node(node, label=str(node), size=size, color=color,
                         title=f"Node {node}\nCommunity: {comm_id}\nDegree: {deg}")

        for u, v in G_sample.edges():
            # Color edge if both nodes in same community
            if partition.get(u) == partition.get(v):
                edge_color = node_colors.get(u, "#CCCCCC") + "66"  # semi-transparent
            else:
                edge_color = "#CCCCCC44"
            net.add_edge(u, v, color=edge_color)

        with tempfile.NamedTemporaryFile(suffix=".html", delete=False, mode="w") as f:
            net.save_graph(f.name)
            html_path = f.name

        with open(html_path, "r") as f:
            components.html(f.read(), height=520, scrolling=False)

with right_col:
    st.markdown("### 📊 Community Size Distribution")
    sorted_comms = sorted(comm_sizes.items(), key=lambda x: x[1], reverse=True)
    c_ids = [str(c) for c, _ in sorted_comms]
    c_sizes = [s for _, s in sorted_comms]

    import matplotlib.pyplot as mplt
    cmap = mplt.colormaps.get_cmap("tab20").resampled(max(len(c_ids), 1))
    colors = [mcolors.to_hex(cmap(i)) for i in range(len(c_ids))]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=c_ids, y=c_sizes,
        marker_color=colors,
        marker_line_color="white",
        marker_line_width=1,
    ))
    fig.update_layout(
        xaxis_title="Community ID",
        yaxis_title="Number of Nodes",
        template="plotly_white",
        height=500,
        showlegend=False,
    )
    import numpy as np
    mean_size = np.mean(c_sizes)
    fig.add_hline(y=mean_size, line_dash="dash", line_color="red",
                  annotation_text=f"Mean = {mean_size:.0f}")
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# ── Community Stats Table ───────────────────────────────────────────────
st.markdown("### 📋 Per-Community Statistics")
stats_df = community_stats(G, partition)
st.dataframe(stats_df, width="stretch")

# ── Download partition as CSV ───────────────────────────────────────────
partition_df = pd.DataFrame(
    [(node, cid) for node, cid in partition.items()],
    columns=["node_id", "community_id"]
).sort_values("node_id")

st.download_button(
    "📥 Download Partition as CSV",
    data=partition_df.to_csv(index=False),
    file_name="community_partition.csv",
    mime="text/csv",
)
