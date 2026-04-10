"""
diffusion.py — Independent Cascade (IC) model simulation and analysis.
"""

import random
from collections import deque
import numpy as np
import networkx as nx


def simulate_ic(G: nx.Graph, seed_node: int, prob: float = 0.1, rng_seed=None):
    """Run one Independent Cascade simulation starting from seed_node.
    Returns (activated_set, activation_time_dict)."""
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
    """Run multiple IC simulations and collect results."""
    results = []
    for i in range(n_runs):
        activated, activation_time = simulate_ic(G, seed_node, prob=prob, rng_seed=i)
        results.append((activated, activation_time))
        if callback:
            callback(i + 1, n_runs)
    return results


def analyze_spread(runs: list) -> dict:
    """Analyze the spread statistics across multiple runs."""
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
    """Count intra vs inter-community activated edges.
    Returns (intra_count, inter_count)."""
    intra = inter = 0
    for u, v in G.edges():
        if u in activated and v in activated:
            if partition.get(u) == partition.get(v):
                intra += 1
            else:
                inter += 1
    return intra, inter


def diffusion_speed_comparison(G: nx.Graph, partition: dict, runs: list) -> dict:
    """Compare intra vs inter-community diffusion speed."""
    all_intra_times = []
    all_inter_times = []
    for activated, activation_time in runs:
        if not activation_time:
            continue
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
