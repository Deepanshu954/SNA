"""
Page 5 — Results Dashboard
Summary of all findings with all paper figures, algorithm comparison, and downloads.
"""

import sys
import json
import io
import zipfile
from pathlib import Path
import streamlit as st
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

st.set_page_config(page_title="Results Dashboard", page_icon="📊", layout="wide")

css_path = Path(__file__).parent.parent / "assets" / "style.css"
if css_path.exists():
    st.markdown(f"<style>{css_path.read_text()}</style>", unsafe_allow_html=True)

PRECOMP = PROJECT_ROOT / "data" / "precomputed"


@st.cache_data
def load_data():
    with open(PRECOMP / "summary.json") as f:
        summary = json.load(f)
    algo_comp = {}
    algo_diff = {}
    if (PRECOMP / "algo_comparison.json").exists():
        with open(PRECOMP / "algo_comparison.json") as f:
            algo_comp = json.load(f)
    if (PRECOMP / "algo_diffusion.json").exists():
        with open(PRECOMP / "algo_diffusion.json") as f:
            algo_diff = json.load(f)
    return summary, algo_comp, algo_diff


if not (PRECOMP / "summary.json").exists():
    st.error("Run `python3 precompute.py` first!")
    st.stop()

summary, algo_comp, algo_diff = load_data()

st.markdown("# 📊 Results Dashboard")
st.markdown("Complete summary of research findings with downloadable figures.")
st.markdown("---")

# ── KPI Row ─────────────────────────────────────────────────────────────
st.markdown("### 🏆 Key Results")
c1, c2, c3, c4 = st.columns(4)
for col, label, val in [
    (c1, "Best Modularity Q", str(summary["modularity_Q"])),
    (c2, "Communities", str(summary["n_communities"])),
    (c3, "Intra-Community", f'{summary["intra_pct"]}%'),
    (c4, "Speed Ratio", f'{summary["speed_ratio"]}×'),
]:
    with col:
        st.markdown(
            f'<div class="metric-card"><h3>{label}</h3><p>{val}</p></div>',
            unsafe_allow_html=True,
        )

st.markdown("---")

# ── Algorithm Comparison Table ──────────────────────────────────────────
if algo_comp:
    st.markdown("### 📋 Algorithm Comparison (All Computed)")
    algo_names = list(algo_comp.keys())
    comp_df = pd.DataFrame({
        "Algorithm": algo_names,
        "Modularity Q": [str(algo_comp[n]["Q"]) for n in algo_names],
        "Communities": [str(algo_comp[n]["n_communities"]) for n in algo_names],
        "Containment": [str(algo_diff.get(n, {}).get("containment", "N/A")) for n in algo_names],
        "Intra %": [str(algo_diff.get(n, {}).get("intra_pct", "N/A")) for n in algo_names],
    })
    st.dataframe(comp_df, use_container_width=True, hide_index=True)
    st.download_button(
        "📥 Download Algorithm Comparison CSV",
        data=comp_df.to_csv(index=False),
        file_name="algorithm_comparison.csv", mime="text/csv",
    )
    st.markdown("---")

# ── All Paper Figures (2x3 grid) ───────────────────────────────────────
st.markdown("### 📈 Paper Figures (Reproduced)")

figures = [
    ("Figure 1: Degree Distribution", "fig1_degree_dist.png",
     "Log-log degree distribution confirming scale-free power-law topology"),
    ("Figure 2: Community Sizes", "fig2_community_sizes.png",
     "Community size distribution from Louvain detection"),
    ("Figure 3: Diffusion Spread", "fig3_diffusion_spread.png",
     "Information spread over time steps (mean ± std)"),
    ("Figure 4: Intra vs Inter", "fig4_intra_inter_pie.png",
     "Proportion of intra vs inter-community activations"),
    ("Figure 5: Algorithm Comparison", "fig5_algorithm_comparison.png",
     "Modularity, Containment, and Intra% across 4 algorithms"),
    ("Figure 6: Results Dashboard", "fig6_results_dashboard.png",
     "Three-panel summary dashboard"),
]

row1 = st.columns(3)
row2 = st.columns(3)

for i, (title, filename, caption) in enumerate(figures):
    col = row1[i] if i < 3 else row2[i - 3]
    img_path = PRECOMP / filename
    with col:
        st.markdown(f"**{title}**")
        if img_path.exists():
            st.image(str(img_path), caption=caption, use_container_width=True)
            with open(img_path, "rb") as f:
                st.download_button(
                    f"📥 Download", data=f.read(),
                    file_name=filename, mime="image/png",
                    key=f"dl_{filename}",
                )
        else:
            st.warning(f"{filename} not found")

st.markdown("---")

# ── 4-Algorithm Network Comparison ─────────────────────────────────────
st.markdown("### 🌐 Network Visualizations")

composite = PRECOMP / "fig_all_algorithms_comparison.png"
if composite.exists():
    st.image(str(composite),
             caption="4 Community Detection Algorithms — Side-by-Side Comparison",
             use_container_width=True)

net_col1, net_col2 = st.columns(2)
with net_col1:
    st.markdown("**Community-Coloured Network (Louvain)**")
    img = PRECOMP / "fig_network.png"
    if img.exists():
        st.image(str(img), use_container_width=True)
with net_col2:
    st.markdown("**Diffusion Activation Network**")
    img = PRECOMP / "fig_network_diffusion.png"
    if img.exists():
        st.image(str(img), use_container_width=True)

st.markdown("---")

# ── Download All as ZIP ─────────────────────────────────────────────────
st.markdown("### 📦 Download All")

zip_buf = io.BytesIO()
with zipfile.ZipFile(zip_buf, "w") as zf:
    for _, filename, _ in figures:
        img_path = PRECOMP / filename
        if img_path.exists():
            zf.write(img_path, filename)
    for extra in ["fig_network.png", "fig_network_full.png", "fig_network_diffusion.png",
                   "fig_all_algorithms_comparison.png",
                   "fig_network_louvain.png", "fig_network_label_propagation.png",
                   "fig_network_greedy_modularity.png", "fig_network_fluid_communities.png"]:
        p = PRECOMP / extra
        if p.exists():
            zf.write(p, extra)
zip_buf.seek(0)

st.download_button(
    "📥 Download All Figures as ZIP",
    data=zip_buf.getvalue(),
    file_name="paper_figures.zip", mime="application/zip",
    type="primary", use_container_width=True,
)

st.markdown("---")

# ── Full Summary Stats ─────────────────────────────────────────────────
st.markdown("### 📋 Full Summary Statistics")
summary_df = pd.DataFrame([
    {"Metric": "Nodes", "Value": str(summary["nodes"])},
    {"Metric": "Edges", "Value": str(summary["edges"])},
    {"Metric": "Average Degree", "Value": str(summary["avg_degree"])},
    {"Metric": "Clustering Coefficient", "Value": str(summary["avg_clustering"])},
    {"Metric": "Network Density", "Value": str(summary["density"])},
    {"Metric": "Modularity Q (Best)", "Value": str(summary["modularity_Q"])},
    {"Metric": "Communities (Louvain)", "Value": str(summary["n_communities"])},
    {"Metric": "Mean IC Spread", "Value": str(summary["mean_spread"])},
    {"Metric": "Spread Std Dev", "Value": str(summary["spread_std"])},
    {"Metric": "Intra-Community %", "Value": f'{summary["intra_pct"]}%'},
    {"Metric": "Speed Ratio (inter/intra)", "Value": f'{summary["speed_ratio"]}×'},
    {"Metric": "Diffusion Containment", "Value": str(summary["containment"])},
    {"Metric": "Avg Conductance", "Value": str(summary["avg_conductance"])},
])
st.dataframe(summary_df, use_container_width=True, hide_index=True)

st.download_button(
    "📥 Download Summary CSV",
    data=summary_df.to_csv(index=False),
    file_name="summary_stats.csv", mime="text/csv",
)

st.markdown(
    '<div class="footer">Community Detection Showcase — Bennett University 2024-25</div>',
    unsafe_allow_html=True,
)
