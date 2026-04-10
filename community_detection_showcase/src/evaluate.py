"""
evaluate.py — Evaluation metrics: NMI, conductance, diffusion containment.
"""

import networkx as nx
import numpy as np
from sklearn.metrics import normalized_mutual_info_score


def compute_nmi(partition: dict, ground_truth: dict) -> float:
    """Compute Normalized Mutual Information between two partitions.

    Parameters
    ----------
    partition : dict
        Mapping node → community_id (predicted).
    ground_truth : dict
        Mapping node → community_id (ground truth).

    Returns
    -------
    float
        NMI score in [0, 1].
    """
    # Align on common nodes
    common_nodes = sorted(set(partition.keys()) & set(ground_truth.keys()))
    if not common_nodes:
        return 0.0

    labels_pred = [partition[n] for n in common_nodes]
    labels_true = [ground_truth[n] for n in common_nodes]

    return float(normalized_mutual_info_score(labels_true, labels_pred))


def compute_conductance(G: nx.Graph, partition: dict) -> dict:
    """Compute conductance for each community.

    Conductance of community S:
        conductance(S) = cut(S, V\\S) / min(vol(S), vol(V\\S))

    where cut(S, V\\S) is the number of edges between S and V\\S,
    and vol(S) is the sum of degrees of nodes in S.

    Returns
    -------
    dict
        Mapping community_id → conductance value.
    """
    communities = {}
    for node, comm_id in partition.items():
        communities.setdefault(comm_id, set()).add(node)

    all_nodes = set(G.nodes())
    conductances = {}

    for comm_id, members in communities.items():
        complement = all_nodes - members

        # Cut edges: edges from members to non-members
        cut = 0
        for u in members:
            for v in G.neighbors(u):
                if v in complement:
                    cut += 1

        # Volume
        vol_s = sum(G.degree(n) for n in members)
        vol_complement = sum(G.degree(n) for n in complement)

        denom = min(vol_s, vol_complement)
        conductances[comm_id] = round(cut / denom, 4) if denom > 0 else 0.0

    return conductances


def compute_diffusion_containment(G: nx.Graph, partition: dict,
                                   activated: set) -> float:
    """Compute diffusion containment score.

    Fraction of activated edges that are intra-community.

    Returns
    -------
    float
        Containment score in [0, 1].
    """
    intra = 0
    total = 0

    for u, v in G.edges():
        if u in activated and v in activated:
            total += 1
            if partition.get(u) == partition.get(v):
                intra += 1

    return round(intra / total, 4) if total > 0 else 0.0
