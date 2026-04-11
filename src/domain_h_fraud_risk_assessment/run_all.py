"""Run fraud risk assessment benchmark with reproducible multi-seed evaluation."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .data_generator import create_fraud_risk_dataset
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
    "classical_ml",
    "svm",
    "tree_based",
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


def _safe_metric(summary: Dict[str, Dict[str, float]], key: str, stat: str = "mean") -> float:
    return summary.get(key, {}).get(stat, np.nan)


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


def _fraud_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_score: np.ndarray) -> Dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, y_score)),
    }


def _approach_specs(smoke_test: bool):
    specs = [
        (
            "Rule-Based Fraud Flags",
            "rule_based",
            lambda seed: None,
            lambda model, x_train, y_train, x_test: (
                (x_test[:, 0] + x_test[:, 1] > np.quantile(x_train[:, 0] + x_train[:, 1], 0.9)).astype(int),
                (x_test[:, 0] + x_test[:, 1]).astype(float),
            ),
        ),
        (
            "Classical ML (Logistic)",
            "classical_ml",
            lambda seed: make_pipeline(StandardScaler(), LogisticRegression(max_iter=320, random_state=seed)),
            None,
        ),
        (
            "Tree-Based (Random Forest)",
            "tree_based",
            lambda seed: RandomForestClassifier(n_estimators=120, random_state=seed),
            None,
        ),
        (
            "SVM Fraud Detector",
            "svm",
            lambda seed: make_pipeline(StandardScaler(), SVC(probability=True, random_state=seed)),
            None,
        ),
        (
            "Boosting Fraud Detector",
            "boosting",
            lambda seed: GradientBoostingClassifier(random_state=seed),
            None,
        ),
        (
            "Neural MLP Fraud Detector",
            "neural",
            lambda seed: make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(32,), max_iter=220, random_state=seed)),
            None,
        ),
        (
            "Instance-Based KNN",
            "instance_based",
            lambda seed: make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=9)),
            None,
        ),
        (
            "Probabilistic Naive Bayes",
            "probabilistic",
            lambda seed: GaussianNB(),
            None,
        ),
        (
            "Ensemble Voting",
            "ensemble",
            lambda seed: VotingClassifier(
                estimators=[
                    ("lr", make_pipeline(StandardScaler(), LogisticRegression(max_iter=250, random_state=seed))),
                    ("rf", RandomForestClassifier(n_estimators=80, random_state=seed)),
                    ("nb", GaussianNB()),
                ],
                voting="soft",
            ),
            None,
        ),
        (
            "Systems Fraud Wrapper",
            "systems",
            lambda seed: RandomForestClassifier(n_estimators=60, max_depth=6, random_state=seed),
            None,
        ),
    ]

    if smoke_test:
        return specs[:3]
    return specs


def run_all_approaches(
    n_samples: int = 6000,
    n_features: int = 26,
    save_results: bool = True,
    output_dir: str = "results/domain_h",
    n_runs: int = 1,
    seed: int = 42,
    seed_list: Optional[List[int]] = None,
    smoke_test: bool = False,
) -> Dict[str, Any]:
    print("=" * 70)
    print("FRAUD RISK ASSESSMENT: 10 Approaches Comparison")
    print("=" * 70)

    seeds = resolve_seed_list(n_runs=n_runs, seed=seed, seed_list=seed_list)
    train_time_cap_seconds = BUDGET_TRAIN_TIME_CAP_SECONDS if smoke_test else BUDGET_TRAIN_TIME_CAP_SECONDS_FULL
    memory_cap_mb = BUDGET_MEMORY_CAP_MB
    tuning_trials_cap = BUDGET_TUNING_TRIALS_CAP

    all_run_results: List[List[Dict[str, Any]]] = []

    for run_idx, run_seed in enumerate(seeds, start=1):
        print(f"\n[Run {run_idx}/{len(seeds)}] Seed={run_seed}")
        set_global_seed(run_seed)

        dataset = create_fraud_risk_dataset(n_samples=n_samples, n_features=n_features, seed=run_seed)
        x_train, y_train = dataset["train"]["X"], dataset["train"]["y"]
        x_test, y_test = dataset["test"]["X"], dataset["test"]["y"]

        run_results: List[Dict[str, Any]] = []
        for name, category, model_factory, custom_eval in _approach_specs(smoke_test=smoke_test):
            try:
                start = time.time()
                if custom_eval is None:
                    model = model_factory(run_seed)
                    model.fit(x_train, y_train)
                    train_time = time.time() - start

                    infer_start = time.time()
                    y_pred = model.predict(x_test)
                    if hasattr(model, "predict_proba"):
                        y_score = model.predict_proba(x_test)[:, 1]
                    elif hasattr(model, "decision_function"):
                        y_score = model.decision_function(x_test)
                    else:
                        y_score = y_pred.astype(float)
                    inference_time = time.time() - infer_start
                else:
                    model = model_factory(run_seed)
                    train_time = time.time() - start
                    infer_start = time.time()
                    y_pred, y_score = custom_eval(model, x_train, y_train, x_test)
                    inference_time = time.time() - infer_start

                metrics = _fraud_metrics(y_test, y_pred, y_score)

                row = {
                    "name": name,
                    "category": category,
                    "philosophy": {
                        "mental_model": category,
                        "strengths": "Fast comparative baseline",
                        "weaknesses": "Synthetic workload assumptions",
                        "best_for": "Fraud risk triage prototyping",
                    },
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
            except Exception as exc:
                row = {
                    "name": name,
                    "category": category,
                    "seed": run_seed,
                    "train_time_cap_seconds": train_time_cap_seconds,
                    "memory_cap_mb": memory_cap_mb,
                    "tuning_trials_cap": tuning_trials_cap,
                    "out_of_budget": False,
                    "success": False,
                    "error": str(exc),
                }
                validate_run_row(row)
                run_results.append(row)

        all_run_results.append(run_results)

    by_name: Dict[str, List[Dict[str, Any]]] = {}
    for run_results in all_run_results:
        for row in run_results:
            by_name.setdefault(row["name"], []).append(row)

    aggregated_results: List[Dict[str, Any]] = []
    comparison_data: List[Dict[str, Any]] = []
    f1_samples_by_name: Dict[str, List[float]] = {}
    total_runs = len(seeds)

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
                "Run Success Rate": success_rate,
                "Out Of Budget Rate": budget_summary["out_of_budget_rate"],
            }
        )

    significance = compute_significance_vs_best(f1_samples_by_name, higher_is_better=True)
    for row in aggregated_results:
        if row.get("success", False):
            row["significance_vs_best"] = significance.get(
                row["name"],
                {
                    "best_approach": row["name"],
                    "best_mean": row["metrics_summary"].get("f1", {}).get("mean", 0.0),
                    "higher_is_better": True,
                    "alpha": 0.05,
                    "is_best": True,
                    "mean_diff_vs_best": 0.0,
                    "p_value": None,
                    "cohens_d_vs_best": None,
                    "significantly_better_than_best": False,
                },
            )

    for row in comparison_data:
        row["F1 p-value vs Best"] = significance.get(row["Approach"], {}).get("p_value", np.nan)

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

    if save_results:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        save_json(
            output_path / "run_manifest.json",
            create_run_manifest(
                domain="domain_h",
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

    return {
        "canonical_comparison": canonical_df.to_dict("records"),
        "variant_comparison": variant_df.to_dict("records"),
        "raw_results": all_run_results,
        "aggregated_results": aggregated_results,
    }
