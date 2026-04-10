#!/usr/bin/env python3
"""
precompute.py — Run all heavy computations once and save results.

This script precomputes:
  1. Network stats (nodes, edges, avg_degree, etc.)
  2. Community detection (Louvain partition)
  3. IC diffusion simulations (100 runs)
  4. Intra/inter analysis
  5. All matplotlib figures as PNG
  6. DW-Louvain weights + partition

Results are saved to data/precomputed/ so Streamlit loads instantly.
"""

import sys
import os
import json
import pickle
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.graph_utils import load_graph, get_stats, get_top_hubs, degree_distribution
from src.community_detection import detect_communities, community_stats, get_community_colors
from src.diffusion import simulate_ic, run_batch, analyze_spread, analyze_intra_inter, diffusion_speed_comparison
from src.evaluate import compute_conductance, compute_diffusion_containment
from src.visualize import (
    plot_degree_dist, plot_community_sizes, plot_diffusion_spread,
    plot_intra_inter_pie, plot_algorithm_comparison, plot_results_dashboard,
)

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "precomputed")
os.makedirs(OUT, exist_ok=True)


def save_fig(fig, name):
    path = os.path.join(OUT, name)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved {name}")


def main():
    t0 = time.time()

    # ── Step 1: Load graph ──────────────────────────────────────────────
    print("=" * 60)
    print("STEP 1: Loading graph...")
    G = load_graph()
    print(f"  Loaded: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    # ── Step 2: Stats ───────────────────────────────────────────────────
    print("\nSTEP 2: Computing network stats...")
    stats = get_stats(G)
    hubs = get_top_hubs(G, 10)
    degrees, counts = degree_distribution(G)
    stats["top_hubs"] = [(int(n), int(d)) for n, d in hubs]
    with open(os.path.join(OUT, "stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print(f"  Stats: {stats}")

    # ── Step 3: Community Detection ─────────────────────────────────────
    print("\nSTEP 3: Running Louvain community detection...")
    partition, Q = detect_communities(G, resolution=1.0, seed=42)
    n_comm = len(set(partition.values()))
    print(f"  Q = {Q:.4f}, communities = {n_comm}")

    comm_stats_df = community_stats(G, partition)
    comm_colors = get_community_colors(partition)

    # Save
    with open(os.path.join(OUT, "partition.json"), "w") as f:
        json.dump({str(k): int(v) for k, v in partition.items()}, f)
    comm_stats_df.to_csv(os.path.join(OUT, "community_stats.csv"), index=False)
    with open(os.path.join(OUT, "community_meta.json"), "w") as f:
        json.dump({"Q": round(Q, 4), "n_communities": n_comm}, f)
    print("  Saved partition, community_stats, community_meta")

    # ── Step 4: IC Diffusion (100 runs) ─────────────────────────────────
    print("\nSTEP 4: Running IC diffusion simulations (100 runs)...")
    seed_node = max(G.nodes(), key=lambda n: G.degree(n))
    print(f"  Seed node: {seed_node} (degree {G.degree(seed_node)})")

    runs = []
    for i in range(100):
        activated, at = simulate_ic(G, seed_node, prob=0.1, rng_seed=i)
        runs.append((activated, at))
        if (i + 1) % 10 == 0:
            print(f"  Run {i+1}/100 done...")

    spread_stats = analyze_spread(runs)
    print(f"  Spread: mean={spread_stats['mean']}, std={spread_stats['std']}")

    # Intra/inter analysis (average across runs)
    total_intra = total_inter = 0
    for activated, at in runs:
        intra, inter = analyze_intra_inter(G, partition, activated)
        total_intra += intra
        total_inter += inter
    avg_intra = total_intra // len(runs)
    avg_inter = total_inter // len(runs)
    intra_pct = round(total_intra / (total_intra + total_inter) * 100, 1)
    print(f"  Intra: {avg_intra}, Inter: {avg_inter}, Intra%: {intra_pct}%")

    speed = diffusion_speed_comparison(G, partition, runs)
    print(f"  Speed ratio: {speed['speed_ratio']}x")

    # Compute cumulative spread curves for plotting
    max_step = 0
    for _, at in runs:
        if at:
            max_step = max(max_step, max(at.values()))
    step_range = list(range(0, max_step + 1))
    curves = []
    for _, at in runs:
        curve = [sum(1 for t in at.values() if t <= s) for s in step_range]
        curves.append(curve)
    curves_np = np.array(curves)
    mean_curve = curves_np.mean(axis=0).tolist()
    std_curve = curves_np.std(axis=0).tolist()

    # Save diffusion results
    diffusion_data = {
        "seed_node": int(seed_node),
        "seed_degree": int(G.degree(seed_node)),
        "n_runs": 100,
        "prob": 0.1,
        "spread_stats": spread_stats,
        "avg_intra": avg_intra,
        "avg_inter": avg_inter,
        "intra_pct": intra_pct,
        "speed": speed,
        "mean_curve": mean_curve,
        "std_curve": std_curve,
        "step_range": step_range,
    }
    with open(os.path.join(OUT, "diffusion.json"), "w") as f:
        json.dump(diffusion_data, f, indent=2)
    print("  Saved diffusion.json")

    # Containment
    first_activated = runs[0][0]
    containment = compute_diffusion_containment(G, partition, first_activated)
    print(f"  Diffusion containment: {containment}")

    # Conductance
    conductance = compute_conductance(G, partition)
    avg_conductance = round(np.mean(list(conductance.values())), 4)
    print(f"  Avg conductance: {avg_conductance}")

    # ── Step 5: Generate Figures ────────────────────────────────────────
    print("\nSTEP 5: Generating all figures...")

    fig1 = plot_degree_dist(G)
    save_fig(fig1, "fig1_degree_dist.png")

    fig2 = plot_community_sizes(partition)
    save_fig(fig2, "fig2_community_sizes.png")

    fig3 = plot_diffusion_spread(runs[:20])  # Use 20 runs for clearer plot
    save_fig(fig3, "fig3_diffusion_spread.png")

    fig4 = plot_intra_inter_pie(avg_intra, avg_inter)
    save_fig(fig4, "fig4_intra_inter_pie.png")

    fig5 = plot_algorithm_comparison()
    save_fig(fig5, "fig5_algorithm_comparison.png")

    fig6 = plot_results_dashboard(runs=runs[:20])
    save_fig(fig6, "fig6_results_dashboard.png")

    # ── Step 6: Network visualization (matplotlib, not pyvis) ───────────
    print("\nSTEP 6: Generating network visualization...")
    import networkx as nx

    # Sample 500 highest-degree nodes for clear visualization
    sorted_nodes = sorted(G.degree(), key=lambda x: x[1], reverse=True)
    top_nodes = [n for n, _ in sorted_nodes[:500]]
    G_sub = G.subgraph(top_nodes)

    fig, ax = plt.subplots(figsize=(12, 10))
    pos = nx.spring_layout(G_sub, seed=42, k=0.3)

    max_deg = max(dict(G_sub.degree()).values())
    node_sizes = [5 + (G_sub.degree(n) / max_deg) * 200 for n in G_sub.nodes()]
    node_colors_list = [comm_colors.get(n, "#999999") for n in G_sub.nodes()]

    nx.draw_networkx_edges(G_sub, pos, alpha=0.08, edge_color="#CCCCCC", ax=ax)
    nx.draw_networkx_nodes(G_sub, pos, node_size=node_sizes,
                           node_color=node_colors_list, alpha=0.85, ax=ax)
    ax.set_title("Facebook Social Network — Community Structure (Top 500 Nodes)",
                 fontsize=14, fontweight="bold")
    ax.axis("off")
    fig.tight_layout()
    save_fig(fig, "fig_network.png")

    # Full network (all nodes, smaller)
    fig, ax = plt.subplots(figsize=(12, 10))
    pos_full = nx.spring_layout(G, seed=42, k=0.15, iterations=30)
    all_sizes = [2 + (G.degree(n) / max(dict(G.degree()).values())) * 80 for n in G.nodes()]
    all_colors = [comm_colors.get(n, "#999999") for n in G.nodes()]
    nx.draw_networkx_edges(G, pos_full, alpha=0.03, edge_color="#CCCCCC", ax=ax)
    nx.draw_networkx_nodes(G, pos_full, node_size=all_sizes,
                           node_color=all_colors, alpha=0.7, ax=ax)
    ax.set_title("Complete Facebook Network — 4,039 Nodes, 88,234 Edges",
                 fontsize=14, fontweight="bold")
    ax.axis("off")
    fig.tight_layout()
    save_fig(fig, "fig_network_full.png")

    # Activation time-colored network
    print("  Generating activation-time network...")
    first_at = runs[0][1]
    max_t = max(first_at.values()) if first_at else 1

    fig, ax = plt.subplots(figsize=(12, 10))
    from matplotlib.cm import YlOrRd
    node_colors_at = []
    for n in G_sub.nodes():
        if n in first_at:
            t = first_at[n]
            node_colors_at.append(YlOrRd(t / max_t))
        else:
            node_colors_at.append("#CCCCCC")
    nx.draw_networkx_edges(G_sub, pos, alpha=0.05, edge_color="#CCCCCC", ax=ax)
    nx.draw_networkx_nodes(G_sub, pos, node_size=node_sizes,
                           node_color=node_colors_at, alpha=0.85, ax=ax)
    ax.set_title("Information Diffusion — Activation Time (Yellow=Early → Red=Late)",
                 fontsize=14, fontweight="bold")
    ax.axis("off")
    fig.tight_layout()
    save_fig(fig, "fig_network_diffusion.png")

    # ── Summary ─────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    summary = {
        "nodes": stats["nodes"],
        "edges": stats["edges"],
        "avg_degree": stats["avg_degree"],
        "density": stats["density"],
        "avg_clustering": stats["avg_clustering"],
        "modularity_Q": round(Q, 4),
        "n_communities": n_comm,
        "mean_spread": spread_stats["mean"],
        "spread_std": spread_stats["std"],
        "intra_pct": intra_pct,
        "speed_ratio": speed["speed_ratio"],
        "containment": containment,
        "avg_conductance": avg_conductance,
        "precompute_time_sec": round(elapsed, 1),
    }
    with open(os.path.join(OUT, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print(f"PRECOMPUTE COMPLETE in {elapsed:.1f}s")
    print(f"All results saved to: {OUT}")
    print("=" * 60)

    # Print summary table
    print(f"\n{'Metric':<30} {'Value':>15}")
    print("-" * 47)
    for k, v in summary.items():
        print(f"  {k:<28} {str(v):>15}")


if __name__ == "__main__":
    main()
