"""
community_detection.py — Louvain community detection and community statistics.
"""

import networkx as nx
import community as community_louvain
import pandas as pd
import numpy as np
import matplotlib.cm as cm
import matplotlib.colors as mcolors


def detect_communities(G: nx.Graph, resolution: float = 1.0, seed: int = 42):
    """Run Louvain community detection.
    Returns (partition_dict, modularity_score)."""
    partition = community_louvain.best_partition(
        G, resolution=resolution, random_state=seed
    )
    modularity = community_louvain.modularity(partition, G)
    return partition, modularity


def community_stats(G: nx.Graph, partition: dict) -> pd.DataFrame:
    """Compute per-community statistics."""
    communities = {}
    for node, comm_id in partition.items():
        communities.setdefault(comm_id, []).append(node)

    rows = []
    for comm_id in sorted(communities.keys()):
        members = communities[comm_id]
        size = len(members)
        degrees_in_comm = [(n, G.degree(n)) for n in members]
        degrees_in_comm.sort(key=lambda x: x[1], reverse=True)
        top3 = [str(n) for n, _ in degrees_in_comm[:3]]

        subgraph = G.subgraph(members)
        internal_edges = subgraph.number_of_edges()
        max_possible = size * (size - 1) / 2 if size > 1 else 1
        internal_density = round(internal_edges / max_possible, 4) if max_possible > 0 else 0.0

        rows.append({
            "community_id": comm_id,
            "size": size,
            "top_3_nodes": ", ".join(top3),
            "internal_density": internal_density,
        })
    return pd.DataFrame(rows)


def get_community_colors(partition: dict) -> dict:
    """Assign a hex color to each node based on its community using tab20."""
    n_communities = len(set(partition.values()))
    cmap = cm.get_cmap("tab20", max(n_communities, 20))
    node_colors = {}
    for node, comm_id in partition.items():
        rgba = cmap(comm_id % 20)
        node_colors[node] = mcolors.to_hex(rgba)
    return node_colors
