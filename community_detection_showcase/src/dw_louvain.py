"""
dw_louvain.py — Diffusion-Weighted Louvain (DW-Louvain) implementation.

Two-stage pipeline:
  Stage 1: Compute edge weights from N IC simulations.
  Stage 2: Run weighted Louvain on the reweighted graph.
"""

import random
import networkx as nx
import community as community_louvain
import numpy as np
from .diffusion import simulate_ic


def compute_diffusion_weights(G: nx.Graph, N: int = 1000, p: float = 0.1,
                               callback=None) -> dict:
    """Compute diffusion-based edge weights.

    For N simulations, pick a random seed node, run IC, and for every edge
    (u, v) where both endpoints are activated, increment weight by 1.
    Normalize by N at the end.

    Parameters
    ----------
    G : nx.Graph
    N : int
        Number of simulations (default 1000).
    p : float
        Propagation probability (default 0.1).
    callback : callable, optional
        Called as callback(i, N) after each simulation for progress tracking.

    Returns
    -------
    weights : dict
        Mapping (u, v) → float weight for each edge in G.
    """
    weights = {}
    for u, v in G.edges():
        weights[(u, v)] = 0.0
        weights[(v, u)] = 0.0

    nodes = list(G.nodes())
    rng = random.Random(42)

    for i in range(N):
        seed = rng.choice(nodes)
        activated, _ = simulate_ic(G, seed, prob=p, rng_seed=i)
        for u, v in G.edges():
            if u in activated and v in activated:
                weights[(u, v)] += 1
                weights[(v, u)] += 1
        if callback:
            callback(i + 1, N)

    # Normalize
    for k in weights:
        weights[k] /= N

    return weights


def run_dw_louvain(G: nx.Graph, N: int = 1000, p: float = 0.1,
                    callback=None):
    """Run the full DW-Louvain pipeline.

    Parameters
    ----------
    G : nx.Graph
    N : int
    p : float
    callback : callable, optional
        Progress callback for weight computation stage.

    Returns
    -------
    partition : dict
    Q : float
        Weighted modularity score.
    weights : dict
        Edge weights from stage 1.
    """
    # Stage 1: compute diffusion weights
    weights = compute_diffusion_weights(G, N=N, p=p, callback=callback)

    # Stage 2: run weighted Louvain
    G_w = G.copy()
    for u, v in G_w.edges():
        G_w[u][v]["weight"] = weights.get((u, v), 0.0)

    partition = community_louvain.best_partition(G_w, weight="weight", random_state=42)
    Q = community_louvain.modularity(partition, G_w, weight="weight")

    return partition, Q, weights


def compare_partitions(G: nx.Graph, partition_base: dict,
                        partition_dw: dict) -> dict:
    """Compare two partitions on several metrics.

    Returns
    -------
    dict with keys:
        base_Q, dw_Q, base_n_communities, dw_n_communities,
        base_largest, dw_largest
    """
    Q_base = community_louvain.modularity(partition_base, G)
    Q_dw = community_louvain.modularity(partition_dw, G)

    def _partition_stats(part):
        communities = {}
        for node, cid in part.items():
            communities.setdefault(cid, []).append(node)
        sizes = [len(v) for v in communities.values()]
        return len(communities), max(sizes) if sizes else 0

    n_base, largest_base = _partition_stats(partition_base)
    n_dw, largest_dw = _partition_stats(partition_dw)

    return {
        "base_Q": round(Q_base, 4),
        "dw_Q": round(Q_dw, 4),
        "base_n_communities": n_base,
        "dw_n_communities": n_dw,
        "base_largest": largest_base,
        "dw_largest": largest_dw,
    }
