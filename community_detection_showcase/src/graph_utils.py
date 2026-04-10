"""
graph_utils.py — Graph loading and basic statistics.
"""

import networkx as nx
import numpy as np
from collections import Counter
from pathlib import Path


def load_graph(path: str = None) -> nx.Graph:
    """Load the Facebook social circles graph from an edge list file."""
    if path is None:
        candidates = [
            Path(__file__).resolve().parent.parent / "data" / "facebook_combined.txt",
            Path("data/facebook_combined.txt"),
        ]
        for candidate in candidates:
            if candidate.exists():
                path = str(candidate)
                break
        if path is None:
            raise FileNotFoundError(
                "Could not find facebook_combined.txt. "
                "Please provide the path explicitly."
            )
    G = nx.read_edgelist(path, create_using=nx.Graph(), nodetype=int)
    return G


def get_stats(G: nx.Graph) -> dict:
    """Return a dictionary of basic network statistics.
    Uses a fast sample for clustering coefficient."""
    n = G.number_of_nodes()
    m = G.number_of_edges()
    degrees = [d for _, d in G.degree()]
    avg_degree = np.mean(degrees) if n > 0 else 0.0
    density = nx.density(G)

    # Fast clustering: sample 200 random nodes to avoid O(n*d^2) on full graph
    import random
    random.seed(42)
    if n > 500:
        sample = random.sample(list(G.nodes()), min(200, n))
        avg_clustering = nx.average_clustering(G, nodes=sample)
    else:
        avg_clustering = nx.average_clustering(G)

    return {
        "nodes": n,
        "edges": m,
        "avg_degree": round(float(avg_degree), 2),
        "density": round(density, 6),
        "avg_clustering": round(avg_clustering, 4),
    }


def get_top_hubs(G: nx.Graph, n: int = 10) -> list:
    """Return the top-n nodes by degree as a list of (node, degree) tuples."""
    degree_seq = sorted(G.degree(), key=lambda x: x[1], reverse=True)
    return degree_seq[:n]


def degree_distribution(G: nx.Graph):
    """Compute the degree distribution."""
    deg_sequence = [d for _, d in G.degree()]
    counter = Counter(deg_sequence)
    degrees = np.array(sorted(counter.keys()))
    counts = np.array([counter[d] for d in degrees])
    return degrees, counts
