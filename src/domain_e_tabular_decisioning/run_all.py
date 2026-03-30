"""Run tabular decisioning benchmark with reproducible multi-seed evaluation."""

import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .approach_01_rule_based import RuleBasedDecisioner
from .approach_02_linear import LinearDecisioner
from .approach_03_svm import SVMDecisioner
from .approach_04_tree import TreeDecisioner
from .approach_05_boosting import BoostingDecisioner
from .approach_06_neural_mlp import NeuralDecisioner
from .approach_07_instance_based import InstanceDecisioner
from .approach_08_probabilistic import ProbabilisticDecisioner
from .approach_09_ensemble import EnsembleDecisioner
from .approach_10_systems import SystemsDecisioner
from .data_generator import create_tabular_decision_dataset
from ..core.benchmark_schema import (
    validate_aggregated_results,
    validate_comparison_dataframe,
    validate_run_row,
)
from ..core.benchmark_utils import (
    aggregate_numeric_dicts,
    compute_significance_vs_best,
    create_run_manifest,
    resolve_seed_list,
    save_json,
    set_global_seed,
)


CANONICAL_CATEGORIES = [
    "rule_based",
    "linear",
    "svm",
    "tree",
    "boosting",
    "neural",
    "instance_based",
    "probabilistic",
    "ensemble",
    "systems",
]

BUDGET_TRAIN_TIME_CAP_SECONDS = 30.0
BUDGET_TRAIN_TIME_CAP_SECONDS_FULL = 300.0
BUDGET_MEMORY_CAP_MB = 4096.0
BUDGET_TUNING_TRIALS_CAP = 0


def _category_for_approach(name: str) -> str:
    low = name.lower()
    if "rule" in low:
        return "rule_based"
    if "logistic" in low or "linear" in low:
        return "linear"
    if "svm" in low:
        return "svm"
    if "random forest" in low or "tree" in low:
        return "tree"
    if "boost" in low or "xgboost" in low:
        return "boosting"
    if "neural" in low or "mlp" in low:
        return "neural"
    if "instance" in low or "knn" in low:
        return "instance_based"
    if "naive bayes" in low or "probabilistic" in low:
        return "probabilistic"
    if "ensemble" in low:
        return "ensemble"
    return "systems"


def _safe_metric(summary: Dict[str, Dict[str, float]], key: str, stat: str = "mean") -> float:
    return summary.get(key, {}).get(stat, np.nan)


def _compute_decision_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray) -> Dict[str, float]:
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    try:
        metrics["roc_auc"] = float(roc_auc_score(y_true, y_score))
    except Exception:
        metrics["roc_auc"] = 0.0
    try:
        metrics["pr_auc"] = float(average_precision_score(y_true, y_score))
    except Exception:
        metrics["pr_auc"] = 0.0
    return metrics


def _build_canonical_table(variant_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for category in CANONICAL_CATEGORIES:
        candidates = variant_df[variant_df["Category"] == category]
        if candidates.empty:
            rows.append(
                {
                    "Category": category,
                    "Approach": "N/A",
                    "F1 Mean": np.nan,
                    "F1 Std": np.nan,
                    "F1 CI95 Low": np.nan,
                    "F1 CI95 High": np.nan,
                    "F1 p-value vs Best": np.nan,
                    "Accuracy Mean": np.nan,
                    "ROC-AUC Mean": np.nan,
                    "Run Success Rate": 0.0,
                    "Available": False,
                }
            )
            continue

        best_idx = candidates["F1 Mean"].fillna(-np.inf).idxmax()
        best = candidates.loc[best_idx]
        rows.append(
            {
                "Category": category,
                "Approach": best["Approach"],
                "F1 Mean": float(best["F1 Mean"]),
                "F1 Std": float(best["F1 Std"]),
                "F1 CI95 Low": float(best["F1 CI95 Low"]),
                "F1 CI95 High": float(best["F1 CI95 High"]),
                "F1 p-value vs Best": float(best["F1 p-value vs Best"])
                if pd.notna(best["F1 p-value vs Best"])
                else np.nan,
                "Accuracy Mean": float(best["Accuracy Mean"]),
                "ROC-AUC Mean": float(best["ROC-AUC Mean"]),
                "Run Success Rate": float(best["Run Success Rate"]),
                "Available": True,
            }
        )
    return pd.DataFrame(rows)


def run_all_approaches(
    n_samples: int = 6000,
    n_features: int = 24,
    save_results: bool = True,
    output_dir: str = "results/domain_e",
    n_runs: int = 1,
    seed: int = 42,
    seed_list: Optional[List[int]] = None,
    smoke_test: bool = False,
) -> Dict[str, Any]:
    """Run tabular decisioning approaches with repeated seeded runs."""

    print("=" * 70)
    print("TABULAR DECISIONING: 10 Approaches Comparison")
    print("=" * 70)

    seeds = resolve_seed_list(n_runs=n_runs, seed=seed, seed_list=seed_list)
    train_time_cap_seconds = BUDGET_TRAIN_TIME_CAP_SECONDS if smoke_test else BUDGET_TRAIN_TIME_CAP_SECONDS_FULL
    memory_cap_mb = BUDGET_MEMORY_CAP_MB
    tuning_trials_cap = BUDGET_TUNING_TRIALS_CAP
    all_run_results: List[List[Dict[str, Any]]] = []

    for run_idx, run_seed in enumerate(seeds, start=1):
        print(f"\n[Run {run_idx}/{len(seeds)}] Seed={run_seed}")
        set_global_seed(run_seed)

        print("\n[1/4] Generating dataset...")
        dataset = create_tabular_decision_dataset(n_samples=n_samples, n_features=n_features)

        x_train, y_train = dataset["train"]["X"], dataset["train"]["y"]
        x_val, y_val = dataset["val"]["X"], dataset["val"]["y"]
        x_test, y_test = dataset["test"]["X"], dataset["test"]["y"]

        print(f"  Train: {x_train.shape}, positive ratio: {y_train.mean():.4f}")
        print(f"  Test: {x_test.shape}, positive ratio: {y_test.mean():.4f}")

        print("\n[2/4] Initializing approaches...")
        if smoke_test:
            approaches = [
                RuleBasedDecisioner(),
                LinearDecisioner(),
                TreeDecisioner({"n_estimators": 50}),
            ]
        else:
            approaches = [
                RuleBasedDecisioner(),
                LinearDecisioner(),
                SVMDecisioner(),
                TreeDecisioner(),
                BoostingDecisioner(),
                NeuralDecisioner(),
                InstanceDecisioner(),
                ProbabilisticDecisioner(),
                EnsembleDecisioner(),
                SystemsDecisioner(),
            ]
        print(f"  Initialized {len(approaches)} approaches")

        print("\n[3/4] Evaluating approaches...")
        run_results: List[Dict[str, Any]] = []

        for approach in approaches:
            print(f"\n  Evaluating: {approach.name}")
            category = _category_for_approach(approach.name)
            try:
                start_time = time.time()
                approach.train(x_train, y_train, x_val, y_val)
                train_time = time.time() - start_time
                print(f"    Training time: {train_time:.2f}s")

                start_time = time.time()
                y_pred = approach.predict(x_test)
                y_score = approach.score(x_test) if hasattr(approach, "score") else y_pred.astype(float)
                inference_time = time.time() - start_time

                metrics = _compute_decision_metrics(y_test, y_pred, y_score)
                print(f"    Accuracy: {metrics['accuracy']:.4f}")
                print(f"    F1: {metrics['f1']:.4f}")
                print(f"    ROC-AUC: {metrics['roc_auc']:.4f}")

                row = {
                    "name": approach.name,
                    "category": category,
                    "philosophy": approach.get_philosophy(),
                    "metrics": metrics,
                    "training_time": train_time,
                    "inference_time": inference_time,
                    "seed": run_seed,
                    "train_time_cap_seconds": train_time_cap_seconds,
                    "memory_cap_mb": memory_cap_mb,
                    "tuning_trials_cap": tuning_trials_cap,
                    "out_of_budget": train_time > train_time_cap_seconds,
                    "success": True,
                }
                validate_run_row(row, required_metric_keys=["accuracy", "f1", "roc_auc"])
                run_results.append(row)
            except Exception as e:
                print(f"    Error: {str(e)}")
                import traceback

                traceback.print_exc()
                row = {
                    "name": approach.name,
                    "category": category,
                    "seed": run_seed,
                    "train_time_cap_seconds": train_time_cap_seconds,
                    "memory_cap_mb": memory_cap_mb,
                    "tuning_trials_cap": tuning_trials_cap,
                    "out_of_budget": False,
                    "success": False,
                    "error": str(e),
                }
                validate_run_row(row)
                run_results.append(row)

        all_run_results.append(run_results)

    print("\n[4/4] Aggregating multi-seed results...")

    by_name: Dict[str, List[Dict[str, Any]]] = {}
    for run_results in all_run_results:
        for row in run_results:
            by_name.setdefault(row["name"], []).append(row)

    comparison_data: List[Dict[str, Any]] = []
    aggregated_results: List[Dict[str, Any]] = []
    total_runs = len(seeds)
    f1_samples_by_name: Dict[str, List[float]] = {}

    for name, rows in by_name.items():
        successful = [r for r in rows if r.get("success", False)]
        out_of_budget_count = sum(1 for r in rows if r.get("out_of_budget", False))
        budget_summary = {
            "train_time_cap_seconds": train_time_cap_seconds,
            "memory_cap_mb": memory_cap_mb,
            "tuning_trials_cap": tuning_trials_cap,
            "out_of_budget_count": out_of_budget_count,
            "out_of_budget_rate": out_of_budget_count / total_runs,
        }

        if not successful:
            aggregated_results.append(
                {
                    "name": name,
                    "category": rows[0].get("category", "systems"),
                    "success_rate": 0.0,
                    "budget_summary": budget_summary,
                    "success": False,
                }
            )
            continue

        metric_summaries = aggregate_numeric_dicts([r["metrics"] for r in successful])
        timing_summaries = aggregate_numeric_dicts(
            [{"training_time": r["training_time"], "inference_time": r["inference_time"]} for r in successful]
        )
        success_rate = len(successful) / total_runs

        f1_samples = [float(r["metrics"].get("f1", np.nan)) for r in successful]
        f1_samples = [v for v in f1_samples if np.isfinite(v)]
        if f1_samples:
            f1_samples_by_name[name] = f1_samples

        aggregated_results.append(
            {
                "name": name,
                "category": successful[0]["category"],
                "success_rate": success_rate,
                "budget_summary": budget_summary,
                "metrics_summary": metric_summaries,
                "timing_summary": timing_summaries,
                "success": True,
            }
        )

        comparison_data.append(
            {
                "Approach": name,
                "Category": successful[0]["category"],
                "F1 Mean": _safe_metric(metric_summaries, "f1", "mean"),
                "F1 Std": _safe_metric(metric_summaries, "f1", "std"),
                "F1 CI95 Low": _safe_metric(metric_summaries, "f1", "ci95_low"),
                "F1 CI95 High": _safe_metric(metric_summaries, "f1", "ci95_high"),
                "F1 p-value vs Best": np.nan,
                "Accuracy Mean": _safe_metric(metric_summaries, "accuracy", "mean"),
                "ROC-AUC Mean": _safe_metric(metric_summaries, "roc_auc", "mean"),
                "PR-AUC Mean": _safe_metric(metric_summaries, "pr_auc", "mean"),
                "Train Time Mean": _safe_metric(timing_summaries, "training_time", "mean"),
                "Inference Time Mean": _safe_metric(timing_summaries, "inference_time", "mean"),
                "Run Success Rate": success_rate,
                "Out Of Budget Rate": budget_summary["out_of_budget_rate"],
            }
        )

    f1_significance = compute_significance_vs_best(f1_samples_by_name, higher_is_better=True)

    for row in aggregated_results:
        if row.get("success", False):
            row["significance_vs_best"] = f1_significance.get(row["name"], {
                "best_approach": row["name"],
                "best_mean": row["metrics_summary"].get("f1", {}).get("mean", 0.0),
                "higher_is_better": True,
                "alpha": 0.05,
                "is_best": True,
                "mean_diff_vs_best": 0.0,
                "p_value": None,
                "cohens_d_vs_best": None,
                "significantly_better_than_best": False,
            })

    for row in comparison_data:
        sig = f1_significance.get(row["Approach"], {})
        row["F1 p-value vs Best"] = sig.get("p_value", np.nan)

    validate_aggregated_results(
        aggregated_results,
        metric_summary_key="metrics_summary",
        required_metric_keys=["accuracy", "f1", "roc_auc"],
        timing_summary_key="timing_summary",
        required_timing_keys=["training_time", "inference_time"],
    )

    variant_df = pd.DataFrame(comparison_data).sort_values(by="F1 Mean", ascending=False)
    canonical_df = _build_canonical_table(variant_df)

    validate_comparison_dataframe(
        canonical_df,
        required_columns=["Category", "Approach", "F1 Mean", "F1 Std", "Run Success Rate", "Available"],
        label="canonical",
    )
    validate_comparison_dataframe(
        variant_df,
        required_columns=["Approach", "Category", "F1 Mean", "F1 Std", "Run Success Rate"],
        label="variant",
    )

    print("\n" + "=" * 70)
    print("RESULTS SUMMARY (CANONICAL 10 CATEGORIES)")
    print("=" * 70)
    print(canonical_df.to_string(index=False))

    if save_results:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        save_json(
            output_path / "run_manifest.json",
            create_run_manifest(
                domain="domain_e",
                config={
                    "n_samples": n_samples,
                    "n_features": n_features,
                    "n_runs": len(seeds),
                    "smoke_test": smoke_test,
                    "train_time_cap_seconds": train_time_cap_seconds,
                    "memory_cap_mb": memory_cap_mb,
                    "tuning_trials_cap": tuning_trials_cap,
                },
                seed_list=seeds,
            ),
        )
        save_json(output_path / "results_raw_by_run.json", {"runs": all_run_results})
        save_json(output_path / "results_aggregated.json", {"approaches": aggregated_results})
        canonical_df.to_csv(output_path / "comparison_canonical.csv", index=False)
        variant_df.to_csv(output_path / "comparison_variants.csv", index=False)
        print(f"\nResults saved to {output_path}")

    return {
        "canonical_comparison": canonical_df.to_dict("records"),
        "variant_comparison": variant_df.to_dict("records"),
        "raw_results": all_run_results,
        "aggregated_results": aggregated_results,
    }
