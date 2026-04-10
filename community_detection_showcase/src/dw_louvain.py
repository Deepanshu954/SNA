"""
dw_louvain.py — Diffusion-Weighted Louvain (DW-Louvain) implementation.
"""

import random
import networkx as nx
import community as community_louvain
import numpy as np
from .diffusion import simulate_ic


def compute_diffusion_weights(G: nx.Graph, N: int = 200, p: float = 0.1,
                               callback=None) -> dict:
    """Compute diffusion-based edge weights from N IC simulations."""
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
    for k in weights:
        weights[k] /= N
    return weights


def run_dw_louvain(G: nx.Graph, N: int = 200, p: float = 0.1, callback=None):
    """Run the full DW-Louvain pipeline.
    Returns (partition, Q, weights)."""
    weights = compute_diffusion_weights(G, N=N, p=p, callback=callback)
    G_w = G.copy()
    for u, v in G_w.edges():
        G_w[u][v]["weight"] = weights.get((u, v), 0.0)
    partition = community_louvain.best_partition(G_w, weight="weight", random_state=42)
    Q = community_louvain.modularity(partition, G_w, weight="weight")
    return partition, Q, weights


def compare_partitions(G: nx.Graph, partition_base: dict,
                        partition_dw: dict) -> dict:
    """Compare two partitions on key metrics."""
    Q_base = community_louvain.modularity(partition_base, G)
    Q_dw = community_louvain.modularity(partition_dw, G)
    def _stats(part):
        comms = {}
        for n, c in part.items():
            comms.setdefault(c, []).append(n)
        sizes = [len(v) for v in comms.values()]
        return len(comms), max(sizes) if sizes else 0
    n_base, l_base = _stats(partition_base)
    n_dw, l_dw = _stats(partition_dw)
    return {
        "base_Q": round(Q_base, 4), "dw_Q": round(Q_dw, 4),
        "base_n_communities": n_base, "dw_n_communities": n_dw,
        "base_largest": l_base, "dw_largest": l_dw,
    }
