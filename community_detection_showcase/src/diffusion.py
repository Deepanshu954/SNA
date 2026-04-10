"""
diffusion.py — Independent Cascade (IC) model simulation and analysis.
"""

import random
from collections import deque
import numpy as np
import networkx as nx


def simulate_ic(G: nx.Graph, seed_node: int, prob: float = 0.1, rng_seed=None):
    """Run one Independent Cascade simulation starting from seed_node.

    Parameters
    ----------
    G : nx.Graph
    seed_node : int
        Starting node for the cascade.
    prob : float
        Transmission probability along each edge (default 0.1).
    rng_seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    activated : set
        Set of all activated nodes.
    activation_time : dict
        Mapping node → time step at which it was activated.
    """
    rng = random.Random(rng_seed)
    activated = {seed_node}
    queue = deque([seed_node])
    activation_time = {seed_node: 0}
    step = 0

    while queue:
        step += 1
        next_queue = []
        for node in list(queue):
            for nbr in G.neighbors(node):
                if nbr not in activated and rng.random() < prob:
                    activated.add(nbr)
                    next_queue.append(nbr)
                    activation_time[nbr] = step
        queue = deque(next_queue)

    return activated, activation_time


def run_batch(G: nx.Graph, seed_node: int, prob: float, n_runs: int, callback=None):
    """Run multiple IC simulations and collect results.

    Parameters
    ----------
    G : nx.Graph
    seed_node : int
    prob : float
    n_runs : int
    callback : callable, optional
        Called as callback(i, n_runs) after each run for progress tracking.

    Returns
    -------
    results : list of (activated_set, activation_time_dict)
    """
    results = []
    for i in range(n_runs):
        activated, activation_time = simulate_ic(G, seed_node, prob=prob, rng_seed=i)
        results.append((activated, activation_time))
        if callback:
            callback(i + 1, n_runs)
    return results


def analyze_spread(runs: list) -> dict:
    """Analyze the spread statistics across multiple runs.

    Parameters
    ----------
    runs : list of (activated_set, activation_time_dict)

    Returns
    -------
    dict with keys: mean, std, max, min, max_steps
    """
    spreads = [len(a) for a, _ in runs]
    max_steps_list = [max(at.values()) if at else 0 for _, at in runs]

    return {
        "mean": round(float(np.mean(spreads)), 1),
        "std": round(float(np.std(spreads)), 1),
        "max": int(np.max(spreads)),
        "min": int(np.min(spreads)),
        "max_steps": round(float(np.mean(max_steps_list)), 1),
    }


def analyze_intra_inter(G: nx.Graph, partition: dict, activated: set):
    """Count intra-community and inter-community activations.

    For every edge (u, v) where both u and v are activated,
    classify it as intra (same community) or inter (different community).

    Returns
    -------
    intra : int
    inter : int
    """
    intra = 0
    inter = 0
    for u, v in G.edges():
        if u in activated and v in activated:
            if partition.get(u) == partition.get(v):
                intra += 1
            else:
                inter += 1
    return intra, inter


def diffusion_speed_comparison(G: nx.Graph, partition: dict, runs: list) -> dict:
    """Compare intra-community vs inter-community diffusion speed.

    For each run, compute the average time step at which intra-community
    nodes were activated vs inter-community nodes.

    Returns
    -------
    dict with keys: avg_intra_steps, avg_inter_steps, speed_ratio
    """
    all_intra_times = []
    all_inter_times = []

    # Determine the community of the seed node (step 0)
    for activated, activation_time in runs:
        if not activation_time:
            continue
        # Find seed node (step 0)
        seed = [n for n, t in activation_time.items() if t == 0]
        if not seed:
            continue
        seed_community = partition.get(seed[0])

        for node, step in activation_time.items():
            if step == 0:
                continue
            if partition.get(node) == seed_community:
                all_intra_times.append(step)
            else:
                all_inter_times.append(step)

    avg_intra = float(np.mean(all_intra_times)) if all_intra_times else 0.0
    avg_inter = float(np.mean(all_inter_times)) if all_inter_times else 0.0
    speed_ratio = round(avg_inter / avg_intra, 1) if avg_intra > 0 else 0.0

    return {
        "avg_intra_steps": round(avg_intra, 2),
        "avg_inter_steps": round(avg_inter, 2),
        "speed_ratio": speed_ratio,
    }
