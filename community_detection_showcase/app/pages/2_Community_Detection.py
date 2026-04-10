"""
Page 2 — Community Detection
Displays results from 4 community detection algorithms.
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
    algo_comp = {}
    algo_diffusion = {}
    if (PRECOMP / "algo_comparison.json").exists():
        with open(PRECOMP / "algo_comparison.json") as f:
            algo_comp = json.load(f)
    if (PRECOMP / "algo_diffusion.json").exists():
        with open(PRECOMP / "algo_diffusion.json") as f:
            algo_diffusion = json.load(f)
    return meta, partition, stats_df, algo_comp, algo_diffusion


if not (PRECOMP / "community_meta.json").exists():
    st.error("Run `python3 precompute.py` first!")
    st.stop()

meta, partition, stats_df, algo_comp, algo_diffusion = load_precomputed()

st.markdown("# 🏘️ Community Detection")
st.markdown("Comparing **4 algorithms** on the Facebook Social Circles dataset.")
st.markdown("---")

# ── Algorithm Comparison Bar Chart ──────────────────────────────────────
if algo_comp and algo_diffusion:
    st.markdown("### 📊 4-Algorithm Comparison")

    algo_names = list(algo_comp.keys())
    q_vals = [algo_comp[n]["Q"] for n in algo_names]
    n_comms = [algo_comp[n]["n_communities"] for n in algo_names]
    contain_vals = [algo_diffusion.get(n, {}).get("containment", 0) for n in algo_names]
    intra_vals = [algo_diffusion.get(n, {}).get("intra_pct", 0) / 100 for n in algo_names]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name="Modularity Q", x=algo_names, y=q_vals,
        marker_color="#1A3C5E",
        text=[f"{v:.4f}" for v in q_vals], textposition="outside",
    ))
    fig.add_trace(go.Bar(
        name="Diffusion Containment", x=algo_names, y=contain_vals,
        marker_color="#2471A3",
        text=[f"{v:.4f}" for v in contain_vals], textposition="outside",
    ))
    fig.add_trace(go.Bar(
        name="Intra-Community %", x=algo_names, y=intra_vals,
        marker_color="#E67E22",
        text=[f"{v:.2f}" for v in intra_vals], textposition="outside",
    ))
    fig.update_layout(
        barmode="group", yaxis_range=[0, 1.15],
        template="plotly_white", height=450,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Metric cards for each algorithm
    cols = st.columns(len(algo_names))
    best_q = max(q_vals)
    for i, (name, col) in enumerate(zip(algo_names, cols)):
        with col:
            star = " ★" if q_vals[i] == best_q else ""
            st.markdown(
                f"""<div class="metric-card" style="text-align:center;">
                <h3 style="font-size:0.9rem;">{name}{star}</h3>
                <p style="font-size:1.5rem; margin:0;">{q_vals[i]:.4f}</p>
                <p style="font-size:0.75rem; margin:0; color:#666;">
                {n_comms[i]} comms · {algo_diffusion.get(name,{}).get('intra_pct','?')}% intra</p>
                </div>""",
                unsafe_allow_html=True,
            )

    st.markdown("---")

    # ── Comparison Table ────────────────────────────────────────────────
    st.markdown("### 📋 Detailed Comparison Table")
    comp_df = pd.DataFrame({
        "Algorithm": algo_names,
        "Modularity Q": [str(algo_comp[n]["Q"]) for n in algo_names],
        "Communities": [str(algo_comp[n]["n_communities"]) for n in algo_names],
        "Containment": [str(algo_diffusion.get(n, {}).get("containment", "N/A")) for n in algo_names],
        "Intra %": [str(algo_diffusion.get(n, {}).get("intra_pct", "N/A")) for n in algo_names],
    })
    st.dataframe(comp_df, use_container_width=True, hide_index=True)

st.markdown("---")

# ── Visual Comparison: 4 algorithms side by side ────────────────────────
st.markdown("### 🕸️ Network Visualization per Algorithm")

# 2x2 composite image
composite_img = PRECOMP / "fig_all_algorithms_comparison.png"
if composite_img.exists():
    st.image(str(composite_img),
             caption="Side-by-side: Louvain, Label Propagation, Greedy Modularity, Fluid Communities",
             use_container_width=True)

# Individual tabs
algo_tabs = st.tabs(["Louvain", "Label Propagation", "Greedy Modularity", "Fluid Communities"])
for tab, name in zip(algo_tabs, ["louvain", "label_propagation", "greedy_modularity", "fluid_communities"]):
    with tab:
        img = PRECOMP / f"fig_network_{name}.png"
        if img.exists():
            st.image(str(img), use_container_width=True)
        else:
            st.info(f"Image for {name} not found")

st.markdown("---")

# ── Louvain-Specific Details ────────────────────────────────────────────
st.markdown("### 📊 Louvain Community Size Distribution")

Q = meta["Q"]
n_comm = meta["n_communities"]
comm_sizes = {}
for node, cid in partition.items():
    comm_sizes[cid] = comm_sizes.get(cid, 0) + 1

sorted_comms = sorted(comm_sizes.items(), key=lambda x: x[1], reverse=True)
c_ids = [str(c) for c, _ in sorted_comms]
c_sizes = [s for _, s in sorted_comms]
cmap_func = cm.get_cmap("tab20", len(c_ids))
colors = [mcolors.to_hex(cmap_func(i)) for i in range(len(c_ids))]

fig_bar = go.Figure()
fig_bar.add_trace(go.Bar(x=c_ids, y=c_sizes, marker_color=colors,
                          marker_line_color="white", marker_line_width=1))
mean_size = np.mean(c_sizes)
fig_bar.add_hline(y=mean_size, line_dash="dash", line_color="red",
                  annotation_text=f"Mean = {mean_size:.0f}")
fig_bar.update_layout(xaxis_title="Community ID", yaxis_title="Number of Nodes",
                      template="plotly_white", height=400, showlegend=False)
st.plotly_chart(fig_bar, use_container_width=True)

# ── Stats Table ─────────────────────────────────────────────────────────
st.markdown("### 📋 Per-Community Statistics (Louvain)")
st.dataframe(stats_df, use_container_width=True, hide_index=True)

# Download Partition
partition_df = pd.DataFrame(
    [(node, cid) for node, cid in partition.items()],
    columns=["node_id", "community_id"]
).sort_values("node_id")
st.download_button(
    "📥 Download Louvain Partition CSV", data=partition_df.to_csv(index=False),
    file_name="community_partition.csv", mime="text/csv",
)

st.markdown(
    """<div class="info-callout">
    <strong>🔍 Key Insight:</strong> All 4 algorithms detect similar high-level community structure,
    but <strong>Louvain</strong> consistently achieves the highest modularity and best
    intra-community diffusion containment on this dataset.</div>""",
    unsafe_allow_html=True,
)
