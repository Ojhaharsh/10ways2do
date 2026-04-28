"""Run all approaches for Domain O: Reinforcement Learning."""
import json
from pathlib import Path
from typing import Dict, Any, Optional, List
from .data_generator import generate_bandit_environment, simulate_bandit
from .strategies import *

STRATEGIES = [EpsilonGreedy, UCB, Thompson, LinUCB, ContextualThompson, NeuralContextual, EnsembleBANDIT, OnlineOptimization, AdaptiveAllocation, MetaBandit]

def run_all_approaches(n_arms: int = 10, n_steps: int = 1000, output_dir: str = "results/domain_o", save_results: bool = True, n_runs: int = 1, seed: int = 42, seed_list: Optional[List[int]] = None, smoke_test: bool = False, **kwargs) -> Dict[str, Any]:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    if smoke_test:
        n_steps = 100
        n_runs = 1
    seeds = seed_list if seed_list else [seed + i * 1000 for i in range(n_runs)]
    all_results = []
    for run_idx, current_seed in enumerate(seeds):
        env = generate_bandit_environment(n_arms=n_arms, n_steps=n_steps, seed=current_seed)
        run_results = {"run": run_idx + 1, "seed": current_seed, "approaches": {}}
        for strategy_class in STRATEGIES:
            try:
                strategy = strategy_class(seed=current_seed)
                strategy.fit()
                result = simulate_bandit(strategy.select_arm, env, seed=current_seed)
                run_results["approaches"][strategy.name] = {"reward": result["total_reward"], "regret": result["regret"], "status": "success"}
            except Exception as e:
                run_results["approaches"][strategy.name] = {"error": str(e), "status": "failed"}
        all_results.append(run_results)
    if save_results:
        results_file = Path(output_dir) / "results_aggregated.json"
        with open(results_file, "w") as f:
            json.dump(all_results, f, indent=2)
    return {"all_results": all_results}
