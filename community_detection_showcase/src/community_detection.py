"""
community_detection.py — Multiple community detection algorithms.
Supports: Louvain, Label Propagation, Greedy Modularity, Fluid Communities.
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


def detect_label_propagation(G: nx.Graph):
    """Run Label Propagation community detection.
    Returns (partition_dict, modularity_score)."""
    communities = list(nx.community.label_propagation_communities(G))
    partition = {}
    for cid, comm in enumerate(communities):
        for node in comm:
            partition[node] = cid
    modularity = nx.community.modularity(G, communities)
    return partition, modularity


def detect_greedy_modularity(G: nx.Graph):
    """Run Greedy Modularity (Clauset-Newman-Moore) community detection.
    Returns (partition_dict, modularity_score)."""
    communities = list(nx.community.greedy_modularity_communities(G, cutoff=1, best_n=None))
    partition = {}
    for cid, comm in enumerate(communities):
        for node in comm:
            partition[node] = cid
    modularity = nx.community.modularity(G, communities)
    return partition, modularity


def detect_fluid_communities(G: nx.Graph, k: int = 15, seed: int = 42):
    """Run Asynchronous Fluid Communities detection.
    Requires specifying k (number of communities).
    Returns (partition_dict, modularity_score)."""
    # Fluid communities requires connected graph
    if not nx.is_connected(G):
        # Use largest connected component
        largest_cc = max(nx.connected_components(G), key=len)
        G_cc = G.subgraph(largest_cc).copy()
    else:
        G_cc = G

    communities = list(nx.community.asyn_fluidc(G_cc, k=k, seed=seed))
    partition = {}
    for cid, comm in enumerate(communities):
        for node in comm:
            partition[node] = cid
    # For nodes not in largest CC, assign to community -1
    for node in G.nodes():
        if node not in partition:
            partition[node] = k
    modularity = nx.community.modularity(G, [{n for n, c in partition.items() if c == cid}
                                              for cid in set(partition.values())])
    return partition, modularity


def run_all_algorithms(G: nx.Graph, callback=None):
    """Run all 4 algorithms and return results dict.

    Returns dict of {name: {'partition': dict, 'Q': float, 'n_communities': int, ...}}
    """
    results = {}

    # 1. Louvain
    if callback:
        callback("Louvain")
    partition, Q = detect_communities(G)
    n_comm = len(set(partition.values()))
    results["Louvain"] = {
        "partition": partition,
        "Q": round(Q, 4),
        "n_communities": n_comm,
    }

    # 2. Label Propagation
    if callback:
        callback("Label Propagation")
    partition_lp, Q_lp = detect_label_propagation(G)
    n_comm_lp = len(set(partition_lp.values()))
    results["Label Propagation"] = {
        "partition": partition_lp,
        "Q": round(Q_lp, 4),
        "n_communities": n_comm_lp,
    }

    # 3. Greedy Modularity
    if callback:
        callback("Greedy Modularity")
    partition_gm, Q_gm = detect_greedy_modularity(G)
    n_comm_gm = len(set(partition_gm.values()))
    results["Greedy Modularity"] = {
        "partition": partition_gm,
        "Q": round(Q_gm, 4),
        "n_communities": n_comm_gm,
    }

    # 4. Fluid Communities
    if callback:
        callback("Fluid Communities")
    k_target = results["Louvain"]["n_communities"]  # Match Louvain's count
    partition_fc, Q_fc = detect_fluid_communities(G, k=k_target)
    n_comm_fc = len(set(partition_fc.values()))
    results["Fluid Communities"] = {
        "partition": partition_fc,
        "Q": round(Q_fc, 4),
        "n_communities": n_comm_fc,
    }

    return results


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
            "community_id": comm_id, "size": size,
            "top_3_nodes": ", ".join(top3), "internal_density": internal_density,
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
