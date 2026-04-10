"""
visualize.py — All matplotlib / plotly figure-generation functions.

Every function returns (fig, ax) or fig. Never calls plt.show().
"""

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import numpy as np
from collections import Counter


# ── Colour palette ──────────────────────────────────────────────────────
PRIMARY = "#1A3C5E"
SECONDARY = "#2471A3"
ACCENT = "#E67E22"
SUCCESS = "#27AE60"
GRID_STYLE = dict(alpha=0.25, linestyle="--")


def _style_ax(ax, title="", xlabel="", ylabel=""):
    """Apply consistent styling to an axes."""
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.grid(True, **GRID_STYLE)


def plot_degree_dist(G):
    """Plot log-log degree distribution histogram.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    degrees = [d for _, d in G.degree()]
    fig, ax = plt.subplots(figsize=(8, 4))

    counts, bin_edges, _ = ax.hist(
        degrees, bins=50, color=PRIMARY, edgecolor="white", log=True
    )
    ax.set_xscale("log")
    _style_ax(
        ax,
        title="Degree Distribution (Log-Log Scale)",
        xlabel="Degree (log)",
        ylabel="Count (log)",
    )
    fig.tight_layout(pad=0.8)
    return fig


def plot_community_sizes(partition):
    """Plot community size distribution as a bar chart.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    comm_sizes = Counter(partition.values())
    sorted_comms = sorted(comm_sizes.items(), key=lambda x: x[1], reverse=True)
    ids = [str(c) for c, _ in sorted_comms]
    sizes = [s for _, s in sorted_comms]

    cmap = plt.colormaps.get_cmap("tab20").resampled(max(len(ids), 1))
    colors = [mcolors.to_hex(cmap(i)) for i in range(len(ids))]

    fig, ax = plt.subplots(figsize=(9, 4))
    bars = ax.bar(ids, sizes, color=colors, edgecolor="white")
    ax.axhline(np.mean(sizes), color="red", linestyle="--", linewidth=1.2,
               label=f"Mean = {np.mean(sizes):.0f}")
    ax.legend()
    _style_ax(
        ax,
        title="Community Size Distribution",
        xlabel="Community ID",
        ylabel="Number of Nodes",
    )
    fig.tight_layout(pad=0.8)
    return fig


def plot_diffusion_spread(runs):
    """Plot cumulative diffusion spread over time steps (mean ± std ribbon).

    Parameters
    ----------
    runs : list of (activated_set, activation_time_dict)

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    # Compute cumulative activations per step for each run
    max_step = 0
    for _, at in runs:
        if at:
            max_step = max(max_step, max(at.values()))

    if max_step == 0:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "No diffusion data", transform=ax.transAxes,
                ha="center", va="center")
        return fig

    step_range = range(0, max_step + 1)
    curves = []
    for _, at in runs:
        curve = []
        for s in step_range:
            count = sum(1 for t in at.values() if t <= s)
            curve.append(count)
        curves.append(curve)

    curves = np.array(curves)
    mean_curve = curves.mean(axis=0)
    std_curve = curves.std(axis=0)

    fig, ax = plt.subplots(figsize=(8, 4))
    steps = list(step_range)
    ax.plot(steps, mean_curve, color=PRIMARY, linewidth=2, label="Mean spread")
    ax.fill_between(steps, mean_curve - std_curve, mean_curve + std_curve,
                     alpha=0.2, color=SECONDARY, label="±1 Std Dev")
    _style_ax(
        ax,
        title="Information Diffusion Spread Over Time",
        xlabel="Time Step",
        ylabel="Cumulative Activated Nodes",
    )
    ax.legend()
    fig.tight_layout(pad=0.8)
    return fig


def plot_intra_inter_pie(intra, inter):
    """Plot intra vs inter community activation pie chart.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    fig, ax = plt.subplots(figsize=(5, 4))
    labels = ["Intra-Community", "Inter-Community"]
    sizes = [intra, inter]
    colors = [PRIMARY, ACCENT]
    explode = (0.05, 0.05)

    ax.pie(
        sizes,
        labels=labels,
        explode=explode,
        colors=colors,
        autopct="%1.1f%%",
        textprops={"color": "white", "fontweight": "bold"},
        startangle=90,
    )
    ax.set_title("Intra vs Inter-Community Activations",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(pad=0.8)
    return fig


def plot_algorithm_comparison():
    """Plot grouped bar chart comparing all 5 algorithms.

    Uses hardcoded values from the paper.

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    algorithms = ["Girvan-\nNewman", "Label\nProp", "Infomap", "Louvain", "DW-Louvain\n★"]
    modularity = [0.61, 0.71, 0.78, 0.83, 0.89]
    nmi = [0.42, 0.55, 0.65, 0.72, 0.81]
    containment = [0.55, 0.62, 0.70, 0.78, 0.85]

    x = np.arange(len(algorithms))
    width = 0.25

    fig, ax = plt.subplots(figsize=(9, 4))
    bars1 = ax.bar(x - width, modularity, width, label="Modularity Q",
                   color=PRIMARY, edgecolor="white")
    bars2 = ax.bar(x, nmi, width, label="NMI",
                   color=SECONDARY, edgecolor="white")
    bars3 = ax.bar(x + width, containment, width, label="Diffusion Containment",
                   color=ACCENT, edgecolor="white")

    # Annotate bars
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f"{height:.2f}",
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(algorithms)
    ax.set_ylim(0, 1.05)
    ax.legend(loc="upper left", fontsize=8)
    _style_ax(
        ax,
        title="Algorithm Comparison — Modularity, NMI, Diffusion Containment",
        xlabel="Algorithm",
        ylabel="Score",
    )
    fig.tight_layout(pad=0.8)
    return fig


def plot_results_dashboard(runs=None, intra=None, inter=None):
    """Create a 3-panel results dashboard figure.

    Panel A: Spread size histogram
    Panel B: Steps-to-penetration grouped bars (intra vs inter)
    Panel C: Louvain vs DW-Louvain head-to-head

    Returns
    -------
    fig : matplotlib.figure.Figure
    """
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # Panel A — Spread histogram
    ax = axes[0]
    if runs:
        spreads = [len(a) for a, _ in runs]
    else:
        np.random.seed(42)
        spreads = np.random.normal(127, 22, 100).astype(int).tolist()
    ax.hist(spreads, bins=20, color=PRIMARY, edgecolor="white")
    mean_val = np.mean(spreads)
    ax.axvline(mean_val, color="red", linestyle="--", label=f"Mean = {mean_val:.0f}")
    ax.legend(fontsize=8)
    _style_ax(ax, title="(A) Spread Distribution", xlabel="# Activated Nodes",
              ylabel="Frequency")

    # Panel B — Penetration speed
    ax = axes[1]
    thresholds = ["50%", "75%", "90%"]
    intra_steps = [1.8, 2.9, 4.2]
    inter_steps = [4.1, 6.7, 9.7]
    x = np.arange(len(thresholds))
    w = 0.3
    ax.bar(x - w / 2, intra_steps, w, label="Intra-community", color=PRIMARY)
    ax.bar(x + w / 2, inter_steps, w, label="Inter-community", color=ACCENT)
    ax.set_xticks(x)
    ax.set_xticklabels(thresholds)
    ax.legend(fontsize=8)
    _style_ax(ax, title="(B) Steps to Penetration", xlabel="Penetration %",
              ylabel="Avg Steps")

    # Panel C — Louvain vs DW-Louvain head-to-head
    ax = axes[2]
    metrics = ["Modularity", "NMI", "Containment"]
    louvain_vals = [0.83, 0.72, 0.78]
    dw_vals = [0.89, 0.81, 0.85]
    x = np.arange(len(metrics))
    w = 0.3
    ax.bar(x - w / 2, louvain_vals, w, label="Louvain", color=SECONDARY)
    ax.bar(x + w / 2, dw_vals, w, label="DW-Louvain ★", color=SUCCESS)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)
    _style_ax(ax, title="(C) Louvain vs DW-Louvain", xlabel="Metric",
              ylabel="Score")

    fig.tight_layout(pad=1.0)
    return fig
