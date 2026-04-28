"""Synthetic graph dataset generator for GNN benchmarking."""

import numpy as np
from typing import Dict, Any, Tuple


def generate_graph_dataset(
    n_nodes: int = 500,
    n_edges: int = 1000,
    n_features: int = 10,
    seed: int = 42,
) -> Dict[str, Any]:
    """Generate synthetic graph dataset for link prediction."""
    rng = np.random.RandomState(seed)

    # Generate node features
    X = rng.randn(n_nodes, n_features).astype(np.float32)

    # Generate edges
    edges = rng.randint(0, n_nodes, (n_edges, 2))
    edge_labels = rng.randint(0, 2, n_edges)

    return {
        "node_features": X,
        "edges": edges,
        "edge_labels": edge_labels,
        "n_nodes": n_nodes,
        "n_edges": n_edges,
        "n_features": n_features,
    }
