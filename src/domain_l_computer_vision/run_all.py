"""Run all approaches for Domain L: Computer Vision."""

import json
from pathlib import Path
from typing import Dict, Any, Optional, List
from ..core.base_model import BaseApproach
from ..core.evaluation import evaluate_classification
from ..core.metrics import compute_classification_metrics
from .data_generator import generate_image_dataset

# Import all approaches
from .approach_01_basic_cnn import BasicCNN
from .approach_02_vgg_style import VGGStyle
from .approach_03_resnet_style import ResNetStyle
from .approach_04_pretrained_efficientnet import PretrainedEfficientNet
from .approach_05_vision_transformer import VisionTransformer
from .approach_06_mobilenet_lightweight import MobileNetLightweight
from .approach_07_ensemble_voting import EnsembleVoting
from .approach_08_data_augmentation import DataAugmentationRegularization
from .approach_09_self_supervised import SelfSupervisedLearning
from .approach_10_neural_architecture_search import NeuralArchitectureSearch


APPROACHES = [
    BasicCNN,
    VGGStyle,
    ResNetStyle,
    PretrainedEfficientNet,
    VisionTransformer,
    MobileNetLightweight,
    EnsembleVoting,
    DataAugmentationRegularization,
    SelfSupervisedLearning,
    NeuralArchitectureSearch,
]


def run_all_approaches(
    n_train: int = 1000,
    n_test: int = 200,
    output_dir: str = "results/domain_l",
    save_results: bool = True,
    n_runs: int = 1,
    seed: int = 42,
    seed_list: Optional[List[int]] = None,
    smoke_test: bool = False,
    **kwargs,
) -> Dict[str, Any]:
    """Run all approaches for Domain L: Computer Vision."""
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    if smoke_test:
        n_train = 100
        n_test = 20
        n_runs = 1

    seeds = seed_list if seed_list else [seed + i * 1000 for i in range(n_runs)]

    all_results = []

    for run_idx, current_seed in enumerate(seeds):
        dataset = generate_image_dataset(
            n_train=n_train,
            n_test=n_test,
            seed=current_seed,
            output_dir=f"{output_dir}/data",
        )

        X_train = dataset["X_train"]
        y_train = dataset["y_train"]
        X_test = dataset["X_test"]
        y_test = dataset["y_test"]

        run_results = {
            "run": run_idx + 1,
            "seed": current_seed,
            "approaches": {},
        }

        for approach_class in APPROACHES:
            try:
                model = approach_class(seed=current_seed)
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                y_proba = model.predict_proba(X_test)

                metrics = compute_classification_metrics(y_test, y_pred, y_proba)
                run_results["approaches"][model.name] = {
                    "metrics": metrics,
                    "status": "success",
                }
            except Exception as e:
                run_results["approaches"][model.name] = {
                    "error": str(e),
                    "status": "failed",
                }

        all_results.append(run_results)

    comparison = _aggregate_results(all_results)

    if save_results:
        results_file = Path(output_dir) / "results_aggregated.json"
        with open(results_file, "w") as f:
            json.dump(all_results, f, indent=2)

        comparison_file = Path(output_dir) / "comparison_canonical.csv"
        comparison.to_csv(comparison_file, index=False)

    return {
        "all_results": all_results,
        "comparison": comparison,
    }


def _aggregate_results(all_results: List[Dict]) -> Any:
    """Aggregate results across all runs."""
    import pandas as pd

    aggregated = []
    for approach_name in APPROACHES[0].__dict__.get("name", ""):
        metrics_by_run = []
        for run in all_results:
            if approach_name in run["approaches"]:
                approach_result = run["approaches"][approach_name]
                if approach_result.get("status") == "success":
                    metrics_by_run.append(approach_result["metrics"])

        if metrics_by_run:
            aggregated.append({
                "approach": approach_name,
                "mean_accuracy": sum(m.get("accuracy", 0) for m in metrics_by_run) / len(metrics_by_run),
                "runs": len(metrics_by_run),
            })

    return pd.DataFrame(aggregated) if aggregated else pd.DataFrame()
