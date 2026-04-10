"""
Page 2 — Community Detection
Displays Louvain community detection results from precomputed data.
"""

import sys
import json
from pathlib import Path
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import matplotlib.cm as cm
import matplotlib.colors as mcolors

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

st.set_page_config(page_title="Community Detection", page_icon="🏘️", layout="wide")

css_path = Path(__file__).parent.parent / "assets" / "style.css"
if css_path.exists():
    st.markdown(f"<style>{css_path.read_text()}</style>", unsafe_allow_html=True)

PRECOMP = PROJECT_ROOT / "data" / "precomputed"


@st.cache_data
def load_precomputed():
    with open(PRECOMP / "community_meta.json") as f:
        meta = json.load(f)
    with open(PRECOMP / "partition.json") as f:
        partition = {int(k): v for k, v in json.load(f).items()}
    stats_df = pd.read_csv(PRECOMP / "community_stats.csv")
    return meta, partition, stats_df


if not (PRECOMP / "community_meta.json").exists():
    st.error("Run `python3 precompute.py` first!")
    st.stop()

meta, partition, stats_df = load_precomputed()

st.markdown("# 🏘️ Community Detection")
st.markdown("Louvain algorithm results on the Facebook Social Circles dataset.")
st.markdown("---")

# ── Metrics ─────────────────────────────────────────────────────────────
Q = meta["Q"]
n_comm = meta["n_communities"]
comm_sizes = {}
for node, cid in partition.items():
    comm_sizes[cid] = comm_sizes.get(cid, 0) + 1
largest = max(comm_sizes.values())
smallest = min(comm_sizes.values())

st.markdown("### 📊 Detection Results")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Modularity Q", f"{Q:.4f}")
c2.metric("Communities", n_comm)
c3.metric("Largest Community", f"{largest} nodes")
c4.metric("Smallest Community", f"{smallest} nodes")

st.markdown("---")

# ── Network + Bar Chart ─────────────────────────────────────────────────
left, right = st.columns([3, 2])

with left:
    st.markdown("### 🕸️ Community-Coloured Network")
    img = PRECOMP / "fig_network.png"
    if img.exists():
        st.image(str(img), caption="Each colour represents a distinct detected community",
                 use_container_width=True)

with right:
    st.markdown("### 📊 Community Size Distribution")
    sorted_comms = sorted(comm_sizes.items(), key=lambda x: x[1], reverse=True)
    c_ids = [str(c) for c, _ in sorted_comms]
    c_sizes = [s for _, s in sorted_comms]

    cmap_func = cm.get_cmap("tab20", len(c_ids))
    colors = [mcolors.to_hex(cmap_func(i)) for i in range(len(c_ids))]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=c_ids, y=c_sizes,
        marker_color=colors,
        marker_line_color="white", marker_line_width=1,
    ))
    mean_size = np.mean(c_sizes)
    fig.add_hline(y=mean_size, line_dash="dash", line_color="red",
                  annotation_text=f"Mean = {mean_size:.0f}")
    fig.update_layout(
        xaxis_title="Community ID", yaxis_title="Number of Nodes",
        template="plotly_white", height=500, showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# Paper figure
img2 = PRECOMP / "fig2_community_sizes.png"
if img2.exists():
    with st.expander("📷 Figure 2 from Paper"):
        st.image(str(img2), caption="Figure 2: Community size distribution",
                 use_container_width=True)

# ── Stats Table ─────────────────────────────────────────────────────────
st.markdown("### 📋 Per-Community Statistics")
st.dataframe(stats_df, use_container_width=True, hide_index=True)

# ── Download ────────────────────────────────────────────────────────────
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

st.markdown(
    """
    <div class="info-callout">
    <strong>🔍 Louvain Algorithm:</strong> A greedy modularity optimization algorithm
    that iteratively assigns nodes to communities to maximize the modularity score Q.
    Higher Q indicates stronger community structure. Q &gt; 0.3 is considered significant.
    Our result of <strong>Q = 0.8318</strong> indicates very strong community structure.
    </div>
    """,
    unsafe_allow_html=True,
)
