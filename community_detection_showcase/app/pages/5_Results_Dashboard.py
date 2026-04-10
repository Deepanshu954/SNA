"""
Page 5 — Results Dashboard
Summary of all findings with all paper figures and download capabilities.
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
def load_summary():
    with open(PRECOMP / "summary.json") as f:
        return json.load(f)


if not (PRECOMP / "summary.json").exists():
    st.error("Run `python3 precompute.py` first!")
    st.stop()

summary = load_summary()

st.markdown("# 📊 Results Dashboard")
st.markdown("Complete summary of research findings with downloadable figures.")
st.markdown("---")

# ── KPI Row ─────────────────────────────────────────────────────────────
st.markdown("### 🏆 Key Results")
c1, c2, c3, c4 = st.columns(4)

for col, label, val in [
    (c1, "Modularity Q", str(summary["modularity_Q"])),
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
     "Modularity, NMI, and Containment across 5 algorithms"),
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

# ── Network Figures ────────────────────────────────────────────────────
st.markdown("### 🌐 Network Visualizations")
net_col1, net_col2 = st.columns(2)

with net_col1:
    st.markdown("**Community-Coloured Network**")
    img = PRECOMP / "fig_network.png"
    if img.exists():
        st.image(str(img), caption="Top 500 nodes coloured by community",
                 use_container_width=True)

with net_col2:
    st.markdown("**Diffusion Activation Network**")
    img = PRECOMP / "fig_network_diffusion.png"
    if img.exists():
        st.image(str(img), caption="Nodes coloured by activation time step",
                 use_container_width=True)

st.markdown("---")

# ── Download All as ZIP ─────────────────────────────────────────────────
st.markdown("### 📦 Download All")

zip_buf = io.BytesIO()
with zipfile.ZipFile(zip_buf, "w") as zf:
    for _, filename, _ in figures:
        img_path = PRECOMP / filename
        if img_path.exists():
            zf.write(img_path, filename)
    # Also add network figures
    for extra in ["fig_network.png", "fig_network_full.png", "fig_network_diffusion.png"]:
        p = PRECOMP / extra
        if p.exists():
            zf.write(p, extra)
zip_buf.seek(0)

st.download_button(
    "📥 Download All Figures as ZIP",
    data=zip_buf.getvalue(),
    file_name="paper_figures.zip",
    mime="application/zip",
    type="primary",
    use_container_width=True,
)

st.markdown("---")

# ── Metrics Table ──────────────────────────────────────────────────────
st.markdown("### 📋 Algorithm Comparison Table")

comp_df = pd.DataFrame({
    "Algorithm": ["Girvan-Newman", "Label Propagation", "Infomap",
                   "Louvain", "DW-Louvain ★"],
    "Modularity Q": ["0.61", "0.71", "0.78", str(summary["modularity_Q"]), "0.89 ★"],
    "NMI": ["0.42", "0.55", "0.65", "0.72", "0.81 ★"],
    "Diffusion Containment": ["0.55", "0.62", "0.70", "0.78", "0.85 ★"],
    "Communities": ["48", "18", "14", str(summary["n_communities"]), "11 ★"],
    "Scalability": ["O(m²)", "O(n)", "O(m)", "O(n log n)", "O(N·m + n log n)"],
})
st.dataframe(comp_df, use_container_width=True, hide_index=True)

st.download_button(
    "📥 Download Comparison CSV",
    data=comp_df.to_csv(index=False),
    file_name="algorithm_comparison.csv",
    mime="text/csv",
)

# ── Summary Stats ──────────────────────────────────────────────────────
st.markdown("---")
st.markdown("### 📋 Full Summary Statistics")

summary_df = pd.DataFrame([
    {"Metric": "Nodes", "Value": str(summary["nodes"])},
    {"Metric": "Edges", "Value": str(summary["edges"])},
    {"Metric": "Average Degree", "Value": str(summary["avg_degree"])},
    {"Metric": "Clustering Coefficient", "Value": str(summary["avg_clustering"])},
    {"Metric": "Network Density", "Value": str(summary["density"])},
    {"Metric": "Modularity Q", "Value": str(summary["modularity_Q"])},
    {"Metric": "Communities Detected", "Value": str(summary["n_communities"])},
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
    file_name="summary_stats.csv",
    mime="text/csv",
)

st.markdown(
    '<div class="footer">Community Detection Showcase · All figures reproduced from research paper · '
    "Bennett University 2024-25</div>",
    unsafe_allow_html=True,
)
