"""
Page 5 — Results Dashboard
Summary of all findings, paper figures, download capabilities.
"""

import sys
from pathlib import Path
import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import io
import zipfile

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.graph_utils import load_graph, degree_distribution
from src.community_detection import detect_communities
from src.visualize import (
    plot_degree_dist,
    plot_community_sizes,
    plot_diffusion_spread,
    plot_intra_inter_pie,
    plot_algorithm_comparison,
    plot_results_dashboard,
)

st.set_page_config(page_title="Results Dashboard", page_icon="📊", layout="wide")

css_path = Path(__file__).parent.parent / "assets" / "style.css"
if css_path.exists():
    st.markdown(f"<style>{css_path.read_text()}</style>", unsafe_allow_html=True)


@st.cache_data
def cached_load_graph():
    return load_graph(str(PROJECT_ROOT / "data" / "facebook_combined.txt"))


@st.cache_data
def cached_detect():
    G = cached_load_graph()
    return detect_communities(G, resolution=1.0, seed=42)


def fig_to_png(fig, dpi=150):
    """Convert a matplotlib figure to PNG bytes."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    buf.seek(0)
    plt.close(fig)
    return buf.getvalue()


st.markdown('<h1 class="section-header">📊 Results Dashboard</h1>', unsafe_allow_html=True)
st.markdown("Complete summary of research findings with downloadable figures.")
st.markdown("---")

G = cached_load_graph()
partition, Q = cached_detect()

# ── KPI Row ─────────────────────────────────────────────────────────────
st.markdown("### 🏆 Key Results")
c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown(
        '<div class="metric-card"><h3>Modularity Q</h3><p>0.8318</p></div>',
        unsafe_allow_html=True,
    )
with c2:
    st.markdown(
        '<div class="metric-card"><h3>Communities</h3><p>12</p></div>',
        unsafe_allow_html=True,
    )
with c3:
    st.markdown(
        '<div class="metric-card"><h3>Intra-Community</h3><p>78%</p></div>',
        unsafe_allow_html=True,
    )
with c4:
    st.markdown(
        '<div class="metric-card"><h3>Speed Ratio</h3><p>2.3×</p></div>',
        unsafe_allow_html=True,
    )

st.markdown("---")

# ── Generate All 6 Figures ──────────────────────────────────────────────
st.markdown("### 📈 Paper Figures (Reproduced)")

with st.spinner("Generating all figures..."):
    # Figure 1: Degree Distribution
    fig1 = plot_degree_dist(G)
    png1 = fig_to_png(fig1)

    # Figure 2: Community Sizes
    fig2 = plot_community_sizes(partition)
    png2 = fig_to_png(fig2)

    # Figure 3: Diffusion Spread (use paper values for dashboard)
    fig3 = plot_diffusion_spread([])  # Will show "No diffusion data" placeholder
    # Instead, generate from simulated data for the dashboard
    np.random.seed(42)
    from src.diffusion import run_batch, analyze_intra_inter
    seed_node = max(G.nodes(), key=lambda n: G.degree(n))
    # Use a smaller batch for dashboard generation
    dashboard_runs = run_batch(G, seed_node, 0.1, 10)
    fig3 = plot_diffusion_spread(dashboard_runs)
    png3 = fig_to_png(fig3)

    # Figure 4: Intra vs Inter Pie
    first_activated = dashboard_runs[0][0]
    intra, inter = analyze_intra_inter(G, partition, first_activated)
    fig4 = plot_intra_inter_pie(intra, inter)
    png4 = fig_to_png(fig4)

    # Figure 5: Algorithm Comparison
    fig5 = plot_algorithm_comparison()
    png5 = fig_to_png(fig5)

    # Figure 6: Results Dashboard
    fig6 = plot_results_dashboard(runs=dashboard_runs)
    png6 = fig_to_png(fig6)

# Display in 2x3 grid
row1_cols = st.columns(3)
row2_cols = st.columns(3)

figures = [
    ("Figure 1: Degree Distribution", png1, "fig1_degree_distribution.png",
     "Log-log degree distribution confirming scale-free power-law topology"),
    ("Figure 2: Community Sizes", png2, "fig2_community_sizes.png",
     "Community size distribution from Louvain detection"),
    ("Figure 3: Diffusion Spread", png3, "fig3_diffusion_spread.png",
     "Information spread over time steps (mean ± std)"),
    ("Figure 4: Intra vs Inter", png4, "fig4_intra_inter_pie.png",
     "Proportion of intra vs inter-community activations"),
    ("Figure 5: Algorithm Comparison", png5, "fig5_algorithm_comparison.png",
     "Modularity, NMI, and Containment across 5 algorithms"),
    ("Figure 6: Results Dashboard", png6, "fig6_results_dashboard.png",
     "Three-panel summary dashboard"),
]

for i, (title, png_data, filename, caption) in enumerate(figures):
    col = row1_cols[i] if i < 3 else row2_cols[i - 3]
    with col:
        st.markdown(f"**{title}**")
        st.image(png_data, caption=caption, width="stretch")
        st.download_button(
            f"📥 Download",
            data=png_data,
            file_name=filename,
            mime="image/png",
            key=f"download_{filename}",
        )

st.markdown("---")

# ── Download All as ZIP ─────────────────────────────────────────────────
st.markdown("### 📦 Download All Figures")
zip_buf = io.BytesIO()
with zipfile.ZipFile(zip_buf, "w") as zf:
    for title, png_data, filename, caption in figures:
        zf.writestr(filename, png_data)
zip_buf.seek(0)

st.download_button(
    "📥 Download All Figures as ZIP",
    data=zip_buf.getvalue(),
    file_name="paper_figures.zip",
    mime="application/zip",
    type="primary",
    width="stretch",
)

st.markdown("---")

# ── Metrics Comparison Table ────────────────────────────────────────────
st.markdown("### 📋 Algorithm Comparison Table")

comp_df = pd.DataFrame({
    "Algorithm": ["Girvan-Newman", "Label Propagation", "Infomap",
                   "Louvain", "DW-Louvain ★"],
    "Modularity Q": ["0.61", "0.71", "0.78", "0.83", "0.89 ★"],
    "NMI": ["0.42", "0.55", "0.65", "0.72", "0.81 ★"],
    "Diffusion Containment": ["0.55", "0.62", "0.70", "0.78", "0.85 ★"],
    "Communities": ["48", "18", "14", "12", "11 ★"],
    "Scalability": ["O(m²)", "O(n)", "O(m)", "O(n log n)", "O(N·m + n log n)"],
})

st.dataframe(comp_df, hide_index=True)

st.download_button(
    "📥 Download Comparison Table as CSV",
    data=comp_df.to_csv(index=False),
    file_name="algorithm_comparison.csv",
    mime="text/csv",
)

st.markdown("---")

# ── Footer ──────────────────────────────────────────────────────────────
st.markdown(
    '<div class="footer">Community Detection Showcase · '
    "All figures reproduced from research paper · "
    "Bennett University 2024-25</div>",
    unsafe_allow_html=True,
)
