#!/usr/bin/env python3
"""
precompute.py — Run all heavy computations once and save results.

Precomputes: network stats, 4 community detection algorithms, IC diffusion
(100 runs), all matplotlib figures, network visualizations.
Results saved to data/precomputed/ so Streamlit loads instantly.
"""

import sys
import os
import json
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as mcolors

from src.graph_utils import load_graph, get_stats, get_top_hubs, degree_distribution
from src.community_detection import (
    run_all_algorithms, community_stats, get_community_colors,
)
from src.diffusion import (
    simulate_ic, run_batch, analyze_spread, analyze_intra_inter,
    diffusion_speed_comparison,
)
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
    stats["top_hubs"] = [(int(n), int(d)) for n, d in hubs]
    with open(os.path.join(OUT, "stats.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print(f"  Stats: {stats}")

    # ── Step 3: Run ALL 4 Community Detection Algorithms ────────────────
    print("\nSTEP 3: Running 4 community detection algorithms...")

    def algo_callback(name):
        print(f"  Running {name}...")

    algo_results = run_all_algorithms(G, callback=algo_callback)

    # Save all algorithm results
    algo_summary = {}
    for name, res in algo_results.items():
        algo_summary[name] = {
            "Q": res["Q"],
            "n_communities": res["n_communities"],
        }
        # Save partitions
        safe_name = name.lower().replace(" ", "_")
        with open(os.path.join(OUT, f"partition_{safe_name}.json"), "w") as f:
            json.dump({str(k): int(v) for k, v in res["partition"].items()}, f)
        print(f"  {name}: Q={res['Q']:.4f}, communities={res['n_communities']}")

    with open(os.path.join(OUT, "algo_comparison.json"), "w") as f:
        json.dump(algo_summary, f, indent=2)

    # Use Louvain as primary partition
    partition = algo_results["Louvain"]["partition"]
    Q = algo_results["Louvain"]["Q"]
    n_comm = algo_results["Louvain"]["n_communities"]

    comm_stats_df = community_stats(G, partition)
    comm_colors = get_community_colors(partition)

    with open(os.path.join(OUT, "partition.json"), "w") as f:
        json.dump({str(k): int(v) for k, v in partition.items()}, f)
    comm_stats_df.to_csv(os.path.join(OUT, "community_stats.csv"), index=False)
    with open(os.path.join(OUT, "community_meta.json"), "w") as f:
        json.dump({"Q": round(Q, 4), "n_communities": n_comm}, f)

    # ── Step 4: IC Diffusion (100 runs) ─────────────────────────────────
    print("\nSTEP 4: Running IC diffusion simulations (100 runs)...")
    seed_node = max(G.nodes(), key=lambda n: G.degree(n))
    print(f"  Seed node: {seed_node} (degree {G.degree(seed_node)})")

    runs = []
    for i in range(100):
        activated, at = simulate_ic(G, seed_node, prob=0.1, rng_seed=i)
        runs.append((activated, at))
        if (i + 1) % 20 == 0:
            print(f"  Run {i+1}/100 done...")

    spread_stats = analyze_spread(runs)
    print(f"  Spread: mean={spread_stats['mean']}, std={spread_stats['std']}")

    # Intra/inter for EACH algorithm
    algo_diffusion = {}
    for name, res in algo_results.items():
        total_intra = total_inter = 0
        for activated, _ in runs[:20]:  # Use 20 runs for speed
            intra, inter = analyze_intra_inter(G, res["partition"], activated)
            total_intra += intra
            total_inter += inter
        avg_intra = total_intra // 20
        avg_inter = total_inter // 20
        intra_pct = round(total_intra / (total_intra + total_inter) * 100, 1)
        containment = compute_diffusion_containment(G, res["partition"], runs[0][0])
        algo_diffusion[name] = {
            "avg_intra": avg_intra,
            "avg_inter": avg_inter,
            "intra_pct": intra_pct,
            "containment": round(containment, 4),
        }
        print(f"  {name}: intra={intra_pct}%, containment={containment:.4f}")

    with open(os.path.join(OUT, "algo_diffusion.json"), "w") as f:
        json.dump(algo_diffusion, f, indent=2)

    # Primary diffusion data (Louvain)
    total_intra = total_inter = 0
    for activated, _ in runs:
        intra, inter = analyze_intra_inter(G, partition, activated)
        total_intra += intra
        total_inter += inter
    avg_intra = total_intra // len(runs)
    avg_inter = total_inter // len(runs)
    intra_pct = round(total_intra / (total_intra + total_inter) * 100, 1)

    speed = diffusion_speed_comparison(G, partition, runs)
    print(f"  Speed ratio: {speed['speed_ratio']}x")

    # Cumulative spread curves
    max_step = max(max(at.values()) for _, at in runs if at)
    step_range = list(range(0, max_step + 1))
    curves = []
    for _, at in runs:
        curve = [sum(1 for t in at.values() if t <= s) for s in step_range]
        curves.append(curve)
    curves_np = np.array(curves)
    mean_curve = curves_np.mean(axis=0).tolist()
    std_curve = curves_np.std(axis=0).tolist()

    diffusion_data = {
        "seed_node": int(seed_node),
        "seed_degree": int(G.degree(seed_node)),
        "n_runs": 100, "prob": 0.1,
        "spread_stats": spread_stats,
        "avg_intra": avg_intra, "avg_inter": avg_inter,
        "intra_pct": intra_pct, "speed": speed,
        "mean_curve": mean_curve, "std_curve": std_curve,
        "step_range": step_range,
    }
    with open(os.path.join(OUT, "diffusion.json"), "w") as f:
        json.dump(diffusion_data, f, indent=2)

    containment = compute_diffusion_containment(G, partition, runs[0][0])
    conductance = compute_conductance(G, partition)
    avg_conductance = round(np.mean(list(conductance.values())), 4)

    # ── Step 5: Generate Figures ────────────────────────────────────────
    print("\nSTEP 5: Generating all figures...")

    fig1 = plot_degree_dist(G)
    save_fig(fig1, "fig1_degree_dist.png")

    fig2 = plot_community_sizes(partition)
    save_fig(fig2, "fig2_community_sizes.png")

    fig3 = plot_diffusion_spread(runs[:20])
    save_fig(fig3, "fig3_diffusion_spread.png")

    fig4 = plot_intra_inter_pie(avg_intra, avg_inter)
    save_fig(fig4, "fig4_intra_inter_pie.png")

    # Custom algorithm comparison with REAL computed values
    print("  Generating algorithm comparison chart with real values...")
    algo_names = list(algo_results.keys())
    algo_Q_vals = [algo_results[n]["Q"] for n in algo_names]
    algo_contain = [algo_diffusion[n]["containment"] for n in algo_names]
    algo_intra = [algo_diffusion[n]["intra_pct"] / 100.0 for n in algo_names]

    fig5, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(algo_names))
    w = 0.25
    b1 = ax.bar(x - w, algo_Q_vals, w, label="Modularity Q", color="#1A3C5E", edgecolor="white")
    b2 = ax.bar(x, algo_contain, w, label="Diffusion Containment", color="#2471A3", edgecolor="white")
    b3 = ax.bar(x + w, algo_intra, w, label="Intra-Community %", color="#E67E22", edgecolor="white")
    for bars in [b1, b2, b3]:
        for bar in bars:
            h = bar.get_height()
            ax.annotate(f"{h:.2f}", xy=(bar.get_x() + bar.get_width()/2, h),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(algo_names, fontsize=10)
    ax.set_ylim(0, 1.15)
    ax.legend(loc="upper left", fontsize=9)
    ax.set_title("Algorithm Comparison — Computed Results", fontsize=13, fontweight="bold")
    ax.set_ylabel("Score")
    ax.grid(True, alpha=0.25, linestyle="--")
    fig5.tight_layout(pad=0.8)
    save_fig(fig5, "fig5_algorithm_comparison.png")

    fig6 = plot_results_dashboard(runs=runs[:20])
    save_fig(fig6, "fig6_results_dashboard.png")

    # ── Step 6: Network visualizations ──────────────────────────────────
    print("\nSTEP 6: Generating network visualizations...")
    import networkx as nx

    sorted_nodes = sorted(G.degree(), key=lambda x: x[1], reverse=True)
    top_nodes = [n for n, _ in sorted_nodes[:500]]
    G_sub = G.subgraph(top_nodes)
    pos = nx.spring_layout(G_sub, seed=42, k=0.3)
    max_deg = max(dict(G_sub.degree()).values())
    node_sizes = [5 + (G_sub.degree(n) / max_deg) * 200 for n in G_sub.nodes()]

    # Community-colored network for EACH algorithm
    for algo_name, res in algo_results.items():
        fig, ax = plt.subplots(figsize=(12, 10))
        colors = get_community_colors(res["partition"])
        node_c = [colors.get(n, "#999999") for n in G_sub.nodes()]
        nx.draw_networkx_edges(G_sub, pos, alpha=0.08, edge_color="#CCCCCC", ax=ax)
        nx.draw_networkx_nodes(G_sub, pos, node_size=node_sizes,
                               node_color=node_c, alpha=0.85, ax=ax)
        ax.set_title(f"{algo_name} — {res['n_communities']} Communities, Q={res['Q']:.4f}",
                     fontsize=14, fontweight="bold")
        ax.axis("off")
        fig.tight_layout()
        safe_name = algo_name.lower().replace(" ", "_")
        save_fig(fig, f"fig_network_{safe_name}.png")

    # Full network
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

    # Alias for backward compatibility
    import shutil
    src_img = os.path.join(OUT, "fig_network_louvain.png")
    dst_img = os.path.join(OUT, "fig_network.png")
    if os.path.exists(src_img):
        shutil.copy2(src_img, dst_img)

    # Activation time network
    first_at = runs[0][1]
    max_t = max(first_at.values()) if first_at else 1
    fig, ax = plt.subplots(figsize=(12, 10))
    from matplotlib.cm import YlOrRd
    node_at_colors = []
    for n in G_sub.nodes():
        if n in first_at:
            node_at_colors.append(YlOrRd(first_at[n] / max_t))
        else:
            node_at_colors.append("#CCCCCC")
    nx.draw_networkx_edges(G_sub, pos, alpha=0.05, edge_color="#CCCCCC", ax=ax)
    nx.draw_networkx_nodes(G_sub, pos, node_size=node_sizes,
                           node_color=node_at_colors, alpha=0.85, ax=ax)
    ax.set_title("Information Diffusion — Activation Time (Yellow=Early → Red=Late)",
                 fontsize=14, fontweight="bold")
    ax.axis("off")
    fig.tight_layout()
    save_fig(fig, "fig_network_diffusion.png")

    # Side-by-side: 4 algorithms in 2x2 grid
    fig, axes = plt.subplots(2, 2, figsize=(16, 14))
    for idx, (algo_name, res) in enumerate(algo_results.items()):
        r, c = divmod(idx, 2)
        ax = axes[r][c]
        colors = get_community_colors(res["partition"])
        node_c = [colors.get(n, "#999999") for n in G_sub.nodes()]
        nx.draw_networkx_edges(G_sub, pos, alpha=0.06, edge_color="#CCCCCC", ax=ax)
        nx.draw_networkx_nodes(G_sub, pos, node_size=[s*0.6 for s in node_sizes],
                               node_color=node_c, alpha=0.85, ax=ax)
        ax.set_title(f"{algo_name}\nQ={res['Q']:.4f}, {res['n_communities']} communities",
                     fontsize=11, fontweight="bold")
        ax.axis("off")
    fig.suptitle("Comparison of 4 Community Detection Algorithms",
                 fontsize=15, fontweight="bold", y=1.01)
    fig.tight_layout()
    save_fig(fig, "fig_all_algorithms_comparison.png")

    # ── Summary ─────────────────────────────────────────────────────────
    elapsed = time.time() - t0
    summary = {
        "nodes": stats["nodes"], "edges": stats["edges"],
        "avg_degree": stats["avg_degree"], "density": stats["density"],
        "avg_clustering": stats["avg_clustering"],
        "modularity_Q": round(Q, 4), "n_communities": n_comm,
        "mean_spread": spread_stats["mean"], "spread_std": spread_stats["std"],
        "intra_pct": intra_pct, "speed_ratio": speed["speed_ratio"],
        "containment": containment, "avg_conductance": avg_conductance,
        "algorithms": algo_summary,
        "precompute_time_sec": round(elapsed, 1),
    }
    with open(os.path.join(OUT, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print(f"PRECOMPUTE COMPLETE in {elapsed:.1f}s")
    print(f"All results saved to: {OUT}")
    print("=" * 60)

    print(f"\n{'Algorithm':<22} {'Q':>8} {'Communities':>12} {'Containment':>12} {'Intra%':>8}")
    print("-" * 64)
    for name in algo_names:
        r = algo_results[name]
        d = algo_diffusion[name]
        print(f"  {name:<20} {r['Q']:>8.4f} {r['n_communities']:>12} "
              f"{d['containment']:>12.4f} {d['intra_pct']:>7.1f}%")


if __name__ == "__main__":
    main()
