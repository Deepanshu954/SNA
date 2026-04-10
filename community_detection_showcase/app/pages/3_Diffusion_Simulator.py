"""
Page 3 — Diffusion Simulator
Displays IC model simulation results from precomputed data.
"""

import sys
import json
from pathlib import Path
import streamlit as st
import plotly.graph_objects as go
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

st.set_page_config(page_title="Diffusion Simulator", page_icon="🌊", layout="wide")

css_path = Path(__file__).parent.parent / "assets" / "style.css"
if css_path.exists():
    st.markdown(f"<style>{css_path.read_text()}</style>", unsafe_allow_html=True)

PRECOMP = PROJECT_ROOT / "data" / "precomputed"


@st.cache_data
def load_diffusion():
    with open(PRECOMP / "diffusion.json") as f:
        return json.load(f)


if not (PRECOMP / "diffusion.json").exists():
    st.error("Run `python3 precompute.py` first!")
    st.stop()

data = load_diffusion()

st.markdown("# 🌊 Diffusion Simulator")
st.markdown("Independent Cascade (IC) model simulation results on the Facebook network.")
st.markdown("---")

# ── How IC Works ────────────────────────────────────────────────────────
with st.expander("📖 How the Independent Cascade Model Works", expanded=False):
    st.markdown("""
    1. **Seed Selection**: Start with a seed node (highest-degree hub)
    2. **Propagation**: At each time step, each newly activated node attempts
       to activate each of its neighbours with probability **p = 0.1**
    3. **Cascade**: Activated nodes get one chance to spread; the process
       continues until no new activations occur
    4. **Analysis**: We run **100 independent simulations** and analyze the spread

    ```python
    def simulate_ic(G, seed_node, prob=0.1):
        activated = {seed_node}
        queue = [seed_node]
        while queue:
            next_queue = []
            for node in queue:
                for nbr in G.neighbors(node):
                    if nbr not in activated and random() < prob:
                        activated.add(nbr)
                        next_queue.append(nbr)
            queue = next_queue
        return activated
    ```
    """)

# ── Metric Cards ────────────────────────────────────────────────────────
st.markdown("### 📊 Simulation Results")
st.markdown(f"**Seed Node:** {data['seed_node']} (degree {data['seed_degree']})  |  "
            f"**p = {data['prob']}**  |  **{data['n_runs']} runs**")

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown(
        f'<div class="metric-card"><h3>Mean Spread</h3><p>{data["spread_stats"]["mean"]}</p></div>',
        unsafe_allow_html=True,
    )
with c2:
    st.markdown(
        f'<div class="metric-card"><h3>Std Dev</h3><p>{data["spread_stats"]["std"]}</p></div>',
        unsafe_allow_html=True,
    )
with c3:
    st.markdown(
        f'<div class="metric-card"><h3>Intra-Community</h3><p>{data["intra_pct"]}%</p></div>',
        unsafe_allow_html=True,
    )
with c4:
    st.markdown(
        f'<div class="metric-card"><h3>Speed Ratio</h3><p>{data["speed"]["speed_ratio"]}×</p></div>',
        unsafe_allow_html=True,
    )

st.markdown("---")

# ── Spread Curve ────────────────────────────────────────────────────────
st.markdown("### 📈 Cumulative Spread Over Time")

steps = data["step_range"]
mean_c = np.array(data["mean_curve"])
std_c = np.array(data["std_curve"])

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=steps, y=(mean_c + std_c).tolist(),
    mode="lines", line=dict(width=0), showlegend=False,
))
fig.add_trace(go.Scatter(
    x=steps, y=(mean_c - std_c).tolist(),
    mode="lines", line=dict(width=0),
    fill="tonexty", fillcolor="rgba(36,113,163,0.2)",
    name="±1 Std Dev",
))
fig.add_trace(go.Scatter(
    x=steps, y=mean_c.tolist(),
    mode="lines+markers", line=dict(color="#1A3C5E", width=2.5),
    name="Mean Spread",
))
fig.update_layout(
    xaxis_title="Time Step", yaxis_title="Cumulative Activated Nodes",
    template="plotly_white", height=400,
    legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01),
)
st.plotly_chart(fig, use_container_width=True)

st.markdown("---")

# ── Diffusion Network + Pie Chart ──────────────────────────────────────
left, right = st.columns([3, 2])

with left:
    st.markdown("### 🕸️ Activation Time Network")
    img = PRECOMP / "fig_network_diffusion.png"
    if img.exists():
        st.image(str(img),
                 caption="Yellow = early activation, Red = late activation, Grey = not activated",
                 use_container_width=True)

with right:
    st.markdown("### 🥧 Intra vs Inter-Community")
    intra = data["avg_intra"]
    inter = data["avg_inter"]
    fig_pie = go.Figure(data=[go.Pie(
        labels=["Intra-Community", "Inter-Community"],
        values=[intra, inter],
        hole=0.35,
        marker_colors=["#1A3C5E", "#E67E22"],
        textfont=dict(color="white", size=14),
    )])
    fig_pie.update_layout(
        template="plotly_white", height=350,
        margin=dict(t=20, b=20, l=20, r=20),
    )
    st.plotly_chart(fig_pie, use_container_width=True)

    st.markdown(
        f"""
        <div class="info-callout">
        <strong>💡 Key Finding:</strong> Intra-community spread is
        <strong>{data['speed']['speed_ratio']}×</strong> faster than inter-community spread.
        <br><br>
        Avg intra steps: <strong>{data['speed']['avg_intra_steps']}</strong> |
        Avg inter steps: <strong>{data['speed']['avg_inter_steps']}</strong>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.markdown("---")

# ── Paper figures ──────────────────────────────────────────────────────
col1, col2 = st.columns(2)
with col1:
    img = PRECOMP / "fig3_diffusion_spread.png"
    if img.exists():
        st.image(str(img), caption="Figure 3: Diffusion spread over time (mean ± std)",
                 use_container_width=True)
with col2:
    img = PRECOMP / "fig4_intra_inter_pie.png"
    if img.exists():
        st.image(str(img), caption="Figure 4: Intra vs inter-community activation proportion",
                 use_container_width=True)

# ── Summary Table ──────────────────────────────────────────────────────
st.markdown("### 📋 Detailed Spread Statistics")
import pandas as pd
spread_df = pd.DataFrame({
    "Metric": ["Mean Spread", "Std Dev", "Max Spread", "Min Spread",
                "Max Steps", "Intra-Community %", "Speed Ratio (inter/intra)"],
    "Value": [
        data["spread_stats"]["mean"],
        data["spread_stats"]["std"],
        data["spread_stats"]["max"],
        data["spread_stats"]["min"],
        data["spread_stats"]["max_steps"],
        f'{data["intra_pct"]}%',
        f'{data["speed"]["speed_ratio"]}×',
    ],
})
st.dataframe(spread_df, use_container_width=True, hide_index=True)
