"""
Page 3 — Diffusion Simulator
IC model simulation with seed strategies, spread analysis, and visualizations.
"""

import sys
from pathlib import Path
import streamlit as st
import streamlit.components.v1 as components
import plotly.graph_objects as go
import numpy as np
import tempfile

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.graph_utils import load_graph
from src.community_detection import detect_communities, get_community_colors
from src.diffusion import run_batch, analyze_spread, analyze_intra_inter, diffusion_speed_comparison

st.set_page_config(page_title="Diffusion Simulator", page_icon="🌊", layout="wide")

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


st.markdown('<h1 class="section-header">🌊 Diffusion Simulator</h1>', unsafe_allow_html=True)
st.markdown("Simulate the Independent Cascade (IC) model and analyze information spread.")
st.markdown("---")

G = cached_load_graph()
partition, _ = cached_detect()

# ── Sidebar ─────────────────────────────────────────────────────────────
st.sidebar.header("🔧 Simulation Parameters")

seed_strategy = st.sidebar.radio(
    "Seed Node Strategy",
    ["Highest Degree", "Random", "Custom Node ID"],
)

if seed_strategy == "Custom Node ID":
    custom_node = st.sidebar.number_input(
        "Node ID", min_value=0, max_value=max(G.nodes()), value=0
    )
elif seed_strategy == "Random":
    import random as _rand
    custom_node = _rand.choice(list(G.nodes()))
    st.sidebar.info(f"Random seed node: **{custom_node}**")
else:
    custom_node = None

prob = st.sidebar.slider("Propagation probability (p)", 0.01, 0.50, 0.10, step=0.01)
n_runs = st.sidebar.slider("Number of simulation runs", 10, 200, 100, step=10)
run_button = st.sidebar.button("🚀 Run Simulation", type="primary", width="stretch")

# Determine seed node
if seed_strategy == "Highest Degree":
    seed_node = max(G.nodes(), key=lambda n: G.degree(n))
else:
    seed_node = custom_node

st.sidebar.markdown("---")
st.sidebar.info(f"**Seed Node:** {seed_node}  \n**Degree:** {G.degree(seed_node)}")

# ── Session state ───────────────────────────────────────────────────────
if "diffusion_runs" not in st.session_state:
    st.session_state["diffusion_runs"] = None

if run_button:
    progress_bar = st.progress(0, text="Running IC simulations...")

    def update_progress(i, total):
        progress_bar.progress(i / total, text=f"Simulation {i}/{total}")

    with st.spinner("Simulating diffusion..."):
        runs = run_batch(G, seed_node, prob, n_runs, callback=update_progress)
        st.session_state["diffusion_runs"] = runs
        st.session_state["diffusion_seed"] = seed_node

    progress_bar.progress(1.0, text="✅ Simulation complete!")

runs = st.session_state.get("diffusion_runs")
if runs is None:
    st.info("👈 Configure parameters and click **Run Simulation** in the sidebar.")
    st.stop()

# ── Analysis ────────────────────────────────────────────────────────────
spread_stats = analyze_spread(runs)

# Use the first run for detailed analysis
first_activated, first_activation_time = runs[0]
intra, inter = analyze_intra_inter(G, partition, first_activated)
speed = diffusion_speed_comparison(G, partition, runs)

# Count communities reached
communities_reached = len(set(partition[n] for n in first_activated if n in partition))

intra_pct = round(intra / (intra + inter) * 100, 1) if (intra + inter) > 0 else 0

# ── Metric Cards ────────────────────────────────────────────────────────
st.markdown("### 📊 Simulation Results")
c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown(
        f'<div class="metric-card"><h3>Mean Spread</h3><p>{spread_stats["mean"]}</p></div>',
        unsafe_allow_html=True,
    )
with c2:
    st.markdown(
        f'<div class="metric-card"><h3>Std Dev</h3><p>{spread_stats["std"]}</p></div>',
        unsafe_allow_html=True,
    )
with c3:
    st.markdown(
        f'<div class="metric-card"><h3>Communities Reached</h3><p>{communities_reached}</p></div>',
        unsafe_allow_html=True,
    )
with c4:
    st.markdown(
        f'<div class="metric-card"><h3>Intra-Community</h3><p>{intra_pct}%</p></div>',
        unsafe_allow_html=True,
    )

st.markdown("---")

# ── Spread Curve (Plotly) ──────────────────────────────────────────────
st.markdown("### 📈 Cumulative Spread Over Time")

# Compute cumulative activations per step
max_step = 0
for _, at in runs:
    if at:
        max_step = max(max_step, max(at.values()))

step_range = list(range(0, max_step + 1))
curves = []
for _, at in runs:
    curve = []
    for s in step_range:
        count = sum(1 for t in at.values() if t <= s)
        curve.append(count)
    curves.append(curve)

curves_np = np.array(curves)
mean_curve = curves_np.mean(axis=0)
std_curve = curves_np.std(axis=0)

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=step_range, y=(mean_curve + std_curve).tolist(),
    mode="lines", line=dict(width=0), showlegend=False,
))
fig.add_trace(go.Scatter(
    x=step_range, y=(mean_curve - std_curve).tolist(),
    mode="lines", line=dict(width=0),
    fill="tonexty", fillcolor="rgba(36,113,163,0.2)",
    name="±1 Std Dev",
))
fig.add_trace(go.Scatter(
    x=step_range, y=mean_curve.tolist(),
    mode="lines+markers", line=dict(color="#1A3C5E", width=2),
    name="Mean Spread",
))
fig.update_layout(
    xaxis_title="Time Step",
    yaxis_title="Cumulative Activated Nodes",
    template="plotly_white",
    height=400,
    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
)
st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# ── Graph by activation time + Pie chart ───────────────────────────────
left_col, right_col = st.columns([3, 2])

with left_col:
    st.markdown("### 🕸️ Network Coloured by Activation Time")
    with st.spinner("Building diffusion graph..."):
        from pyvis.network import Network

        sorted_nodes = sorted(G.degree(), key=lambda x: x[1], reverse=True)
        selected = [n for n, _ in sorted_nodes[:300]]
        G_sample = G.subgraph(selected).copy()

        net = Network(height="450px", width="100%", bgcolor="#fafafa",
                      font_color="#333333")
        net.barnes_hut(spring_length=80)

        at = first_activation_time
        max_t = max(at.values()) if at else 1

        import matplotlib.pyplot as mplt
        import matplotlib.colors as mc

        cmap = mplt.colormaps.get_cmap("YlOrRd")
        for node in G_sample.nodes():
            deg = G_sample.degree(node)
            size = 5 + (deg / max(dict(G_sample.degree()).values())) * 25
            if node in at:
                t = at[node]
                color = mc.to_hex(cmap(t / max_t))
                title = f"Node {node}\nActivated: step {t}\nDegree: {deg}"
            else:
                color = "#CCCCCC"
                title = f"Node {node}\nNot activated\nDegree: {deg}"
            net.add_node(node, label=str(node), size=size, color=color, title=title)

        for u, v in G_sample.edges():
            net.add_edge(u, v, color="#CCCCCC44")

        with tempfile.NamedTemporaryFile(suffix=".html", delete=False, mode="w") as f:
            net.save_graph(f.name)
            html_path = f.name

        with open(html_path, "r") as f:
            components.html(f.read(), height=470, scrolling=False)

    st.caption("Nodes coloured from yellow (early) to red (late). Grey = not activated.")

with right_col:
    st.markdown("### 🥧 Intra vs Inter-Community")
    fig_pie = go.Figure(data=[go.Pie(
        labels=["Intra-Community", "Inter-Community"],
        values=[intra, inter],
        hole=0.3,
        marker_colors=["#1A3C5E", "#E67E22"],
        textfont=dict(color="white", size=14),
    )])
    fig_pie.update_layout(
        template="plotly_white",
        height=350,
        margin=dict(t=20, b=20, l=20, r=20),
    )
    st.plotly_chart(fig_pie, use_container_width=True)

    # Speed callout
    st.markdown(
        f"""
        <div class="info-callout">
        <strong>💡 Key Finding:</strong> Intra-community spread is
        <strong>{speed['speed_ratio']}×</strong> faster than inter-community spread.
        <br><br>
        Avg intra steps: <strong>{speed['avg_intra_steps']}</strong> |
        Avg inter steps: <strong>{speed['avg_inter_steps']}</strong>
        </div>
        """,
        unsafe_allow_html=True,
    )
