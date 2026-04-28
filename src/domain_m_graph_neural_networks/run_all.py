"""Run all approaches for Domain M: Graph Neural Networks."""

import json
from pathlib import Path
from typing import Dict, Any, Optional, List
from .data_generator import generate_graph_dataset
from .approach_01_gcn import GCN
from .approach_02_graphsage import GraphSAGE
from .approach_03_gat import GAT
from .approach_04_10 import (
    ChebNet, GIN, NodeToNodeSimilarity, SkipGram, EnsembleGraphMethods, NGNN, LearningToRankGraphs
)


APPROACHES = [GCN, GraphSAGE, GAT, ChebNet, GIN, NodeToNodeSimilarity, SkipGram, EnsembleGraphMethods, NGNN, LearningToRankGraphs]


def run_all_approaches(
    n_nodes: int = 500,
    n_edges: int = 1000,
    output_dir: str = "results/domain_m",
    save_results: bool = True,
    n_runs: int = 1,
    seed: int = 42,
    seed_list: Optional[List[int]] = None,
    smoke_test: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """Run all approaches for Domain M: Graph Neural Networks."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    if smoke_test:
        n_nodes = 100
        n_edges = 200
        n_runs = 1

    seeds = seed_list if seed_list else [seed + i * 1000 for i in range(n_runs)]
    all_results = []

    for run_idx, current_seed in enumerate(seeds):
        dataset = generate_graph_dataset(n_nodes=n_nodes, n_edges=n_edges, seed=current_seed)
        X = dataset["node_features"]
        edges = dataset["edges"]
        edge_labels = dataset["edge_labels"]

        run_results = {"run": run_idx + 1, "seed": current_seed, "approaches": {}}

        for approach_class in APPROACHES:
            try:
                model = approach_class(seed=current_seed)
                model.fit(X, edges)
                y_pred = model.predict(edges)
                accuracy = (y_pred == edge_labels).mean()
                run_results["approaches"][model.name] = {"accuracy": float(accuracy), "status": "success"}
            except Exception as e:
                run_results["approaches"][model.name] = {"error": str(e), "status": "failed"}

        all_results.append(run_results)

    if save_results:
        results_file = Path(output_dir) / "results_aggregated.json"
        with open(results_file, "w") as f:
            json.dump(all_results, f, indent=2)

    return {"all_results": all_results}
