"""Run all approaches for Domain P: Multimodal Learning."""
import json
from pathlib import Path
from typing import Dict, Any, Optional, List
from .data_generator import generate_multimodal_dataset
from .approaches import *

APPROACHES = [CLIP, VisualBERT, ALBEF, Flamingo, LLaVA, BLIPv2, EnsembleMultimodal, CrossModalAttention, MoCo, AdaptiveMultimodal]

def run_all_approaches(n_samples: int = 1000, output_dir: str = "results/domain_p", save_results: bool = True, n_runs: int = 1, seed: int = 42, seed_list: Optional[List[int]] = None, smoke_test: bool = False, **kwargs) -> Dict[str, Any]:
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    if smoke_test:
        n_samples = 100
        n_runs = 1
    seeds = seed_list if seed_list else [seed + i * 1000 for i in range(n_runs)]
    all_results = []
    for run_idx, current_seed in enumerate(seeds):
        dataset = generate_multimodal_dataset(n_samples=n_samples, seed=current_seed)
        n_train = int(0.8 * n_samples)
        images_train, images_test = dataset["images"][:n_train], dataset["images"][n_train:]
        texts_train, texts_test = dataset["texts"][:n_train], dataset["texts"][n_train:]
        labels_train, labels_test = dataset["labels"][:n_train], dataset["labels"][n_train:]
        run_results = {"run": run_idx + 1, "seed": current_seed, "approaches": {}}
        for approach_class in APPROACHES:
            try:
                model = approach_class(seed=current_seed)
                model.fit(images_train, texts_train, labels_train)
                y_pred = model.predict(images_test, texts_test)
                accuracy = (y_pred == labels_test).mean()
                run_results["approaches"][model.name] = {"accuracy": float(accuracy), "status": "success"}
            except Exception as e:
                run_results["approaches"][model.name] = {"error": str(e), "status": "failed"}
        all_results.append(run_results)
    if save_results:
        results_file = Path(output_dir) / "results_aggregated.json"
        with open(results_file, "w") as f:
            json.dump(all_results, f, indent=2)
    return {"all_results": all_results}
