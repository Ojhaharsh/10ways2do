"""Run all approaches for Domain N: Few-Shot Learning."""
import json
from pathlib import Path
from typing import Dict, Any, Optional, List
from .data_generator import generate_few_shot_dataset
from .approaches import *

APPROACHES = [PrototypicalNetworks, MatchingNetworks, RelationNetwork, MAML, TaskAugmentedMAML, TransductiveTransfer, Timm, EnsembleFewShot, OptimalTransportFewShot, ContextualFewShot]

def run_all_approaches(n_ways: int = 5, n_shots: int = 5, n_queries: int = 15, output_dir: str = "results/domain_n", save_results: bool = True, n_runs: int = 1, seed: int = 42, seed_list: Optional[List[int]] = None, smoke_test: bool = False, **kwargs) -> Dict[str, Any]:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    if smoke_test:
        n_runs = 1
    seeds = seed_list if seed_list else [seed + i * 1000 for i in range(n_runs)]
    all_results = []
    for run_idx, current_seed in enumerate(seeds):
        dataset = generate_few_shot_dataset(n_ways=n_ways, n_shots=n_shots, n_queries=n_queries, seed=current_seed)
        X_support = dataset["X_support"]
        y_support = dataset["y_support"]
        X_query = dataset["X_query"]
        y_query = dataset["y_query"]
        run_results = {"run": run_idx + 1, "seed": current_seed, "approaches": {}}
        for approach_class in APPROACHES:
            try:
                model = approach_class(seed=current_seed)
                model.fit(X_support, y_support)
                y_pred = model.predict(X_query)
                accuracy = (y_pred == y_query).mean()
                run_results["approaches"][model.name] = {"accuracy": float(accuracy), "status": "success"}
            except Exception as e:
                run_results["approaches"][model.name] = {"error": str(e), "status": "failed"}
        all_results.append(run_results)
    if save_results:
        results_file = Path(output_dir) / "results_aggregated.json"
        with open(results_file, "w") as f:
            json.dump(all_results, f, indent=2)
    return {"all_results": all_results}
