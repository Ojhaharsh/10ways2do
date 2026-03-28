"""Run anomaly detection benchmark with reproducible multi-seed evaluation."""

import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np

from .data_generator import create_anomaly_dataset, add_noise_to_data
from .approach_01_statistical import StatisticalAnomalyDetector, MADDetector
from .approach_02_distance_based import KNNAnomalyDetector, LOFDetector
from .approach_03_tree_based import IsolationForestDetector
from .approach_04_autoencoder import AutoencoderDetector, VAEDetector
from .approach_05_rnn_lstm import LSTMAnomalyDetector
from .approach_06_transformer import TransformerAnomalyDetector
from .approach_07_graph_based import GraphAnomalyDetector
from .approach_08_ensemble import EnsembleAnomalyDetector
from .approach_09_hybrid import HybridAnomalyDetector

from ..core.metrics import compute_anomaly_metrics
from ..core.benchmark_utils import (
    aggregate_numeric_dicts,
    create_run_manifest,
    resolve_seed_list,
    save_json,
    set_global_seed,
)
from ..core.benchmark_schema import (
    validate_aggregated_results,
    validate_comparison_dataframe,
    validate_run_row,
)


CANONICAL_CATEGORIES = [
    "statistical",
    "distance",
    "tree",
    "autoencoder",
    "rnn_lstm",
    "transformer",
    "graph",
    "ensemble",
    "hybrid",
    "systems",
]


def _category_for_approach(name: str) -> str:
    low = name.lower()
    if "statistical" in low or "mad" in low:
        return "statistical"
    if "knn" in low or "lof" in low or "distance" in low:
        return "distance"
    if "isolation" in low or "forest" in low or "tree" in low:
        return "tree"
    if "autoencoder" in low or "vae" in low:
        return "autoencoder"
    if "lstm" in low or "rnn" in low:
        return "rnn_lstm"
    if "transformer" in low:
        return "transformer"
    if "graph" in low:
        return "graph"
    if "ensemble" in low:
        return "ensemble"
    if "hybrid" in low:
        return "hybrid"
    return "systems"


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
                    "Precision Mean": np.nan,
                    "Recall Mean": np.nan,
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
                "Precision Mean": float(best["Precision Mean"]),
                "Recall Mean": float(best["Recall Mean"]),
                "Run Success Rate": float(best["Run Success Rate"]),
                "Available": True,
            }
        )
    return pd.DataFrame(rows)


def run_all_approaches(
    n_train: int = 5000,
    n_val: int = 1000,
    n_test: int = 1000,
    save_results: bool = True,
    output_dir: str = "results/domain_b",
    n_runs: int = 1,
    seed: int = 42,
    seed_list: Optional[List[int]] = None,
    smoke_test: bool = False,
) -> Dict[str, Any]:
    """Run anomaly detection approaches with repeated seeded runs."""
    
    print("=" * 70)
    print("ANOMALY DETECTION: 10 Approaches Comparison")
    print("=" * 70)
    
    seeds = resolve_seed_list(n_runs=n_runs, seed=seed, seed_list=seed_list)
    all_run_results: List[List[Dict[str, Any]]] = []

    for run_idx, run_seed in enumerate(seeds, start=1):
        print(f"\n[Run {run_idx}/{len(seeds)}] Seed={run_seed}")
        set_global_seed(run_seed)

        print("\n[1/4] Generating dataset...")
        dataset = create_anomaly_dataset(n_train, n_val, n_test)

        X_train, y_train = dataset['train']['X'], dataset['train']['y']
        X_val, y_val = dataset['val']['X'], dataset['val']['y']
        X_test, y_test = dataset['test']['X'], dataset['test']['y']

        print(f"  Train: {X_train.shape}, Anomaly ratio: {y_train.mean():.4f}")
        print(f"  Test: {X_test.shape}, Anomaly ratio: {y_test.mean():.4f}")

        print("\n[2/4] Initializing approaches...")
        if smoke_test:
            approaches = [
                StatisticalAnomalyDetector({'method': 'zscore'}),
                MADDetector(),
                IsolationForestDetector({'n_estimators': 20}),
            ]
        else:
            approaches = [
                StatisticalAnomalyDetector({'method': 'zscore'}),
                MADDetector(),
                KNNAnomalyDetector({'k': 5}),
                LOFDetector({'n_neighbors': 20}),
                IsolationForestDetector({'n_estimators': 100}),
                AutoencoderDetector({'epochs': 20, 'hidden_dims': [16, 8]}),
                LSTMAnomalyDetector({'epochs': 10, 'seq_len': 10}),
                TransformerAnomalyDetector({'epochs': 10, 'seq_len': 20}),
                GraphAnomalyDetector({'k': 10}),
                EnsembleAnomalyDetector(),
                HybridAnomalyDetector()
            ]
        print(f"  Initialized {len(approaches)} approaches")

        print("\n[3/4] Evaluating approaches...")
        run_results: List[Dict[str, Any]] = []

        for approach in approaches:
            print(f"\n  Evaluating: {approach.name}")
            category = _category_for_approach(approach.name)
            try:
                start_time = time.time()
                approach.train(X_train, y_train, X_val, y_val)
                train_time = time.time() - start_time
                print(f"    Training time: {train_time:.2f}s")

                start_time = time.time()
                y_pred = approach.predict(X_test)
                y_scores = approach.score(X_test) if hasattr(approach, 'score') else None
                inference_time = time.time() - start_time

                metrics = compute_anomaly_metrics(y_test, y_pred, y_scores)

                print(f"    Precision: {metrics['precision']:.4f}")
                print(f"    Recall: {metrics['recall']:.4f}")
                print(f"    F1: {metrics['f1']:.4f}")

                row = {
                    'name': approach.name,
                    'category': category,
                    'philosophy': approach.get_philosophy(),
                    'metrics': metrics,
                    'training_time': train_time,
                    'inference_time': inference_time,
                    'seed': run_seed,
                    'success': True
                }
                validate_run_row(row, required_metric_keys=['precision', 'recall', 'f1'])
                run_results.append(row)
            except Exception as e:
                print(f"    Error: {str(e)}")
                import traceback
                traceback.print_exc()
                row = {
                    'name': approach.name,
                    'category': category,
                    'seed': run_seed,
                    'success': False,
                    'error': str(e)
                }
                validate_run_row(row)
                run_results.append(row)

        all_run_results.append(run_results)

    print("\n[4/4] Aggregating multi-seed results...")

    by_name: Dict[str, List[Dict[str, Any]]] = {}
    by_name_success: Dict[str, int] = {}
    for run_results in all_run_results:
        for row in run_results:
            name = row['name']
            by_name.setdefault(name, []).append(row)
            if row.get('success', False):
                by_name_success[name] = by_name_success.get(name, 0) + 1

    comparison_data: List[Dict[str, Any]] = []
    aggregated_results: List[Dict[str, Any]] = []
    total_runs = len(seeds)

    for name, rows in by_name.items():
        successful = [r for r in rows if r.get('success', False)]
        if not successful:
            aggregated_results.append({
                'name': name,
                'category': rows[0].get('category', 'systems'),
                'success_rate': 0.0,
                'success': False,
            })
            continue

        metric_summary = aggregate_numeric_dicts([r['metrics'] for r in successful])
        time_summary = aggregate_numeric_dicts(
            [{'training_time': r.get('training_time', np.nan), 'inference_time': r.get('inference_time', np.nan)} for r in successful]
        )

        success_rate = by_name_success.get(name, 0) / total_runs
        category = successful[0].get('category', 'systems')

        aggregated_results.append({
            'name': name,
            'category': category,
            'success_rate': success_rate,
            'metrics_summary': metric_summary,
            'timing_summary': time_summary,
            'success': True,
        })

        comparison_data.append({
            'Approach': name,
            'Category': category,
            'Precision Mean': _safe_metric(metric_summary, 'precision', 'mean'),
            'Precision Std': _safe_metric(metric_summary, 'precision', 'std'),
            'Recall Mean': _safe_metric(metric_summary, 'recall', 'mean'),
            'Recall Std': _safe_metric(metric_summary, 'recall', 'std'),
            'F1 Mean': _safe_metric(metric_summary, 'f1', 'mean'),
            'F1 Std': _safe_metric(metric_summary, 'f1', 'std'),
            'ROC-AUC Mean': _safe_metric(metric_summary, 'roc_auc', 'mean'),
            'Training Time Mean (s)': _safe_metric(time_summary, 'training_time', 'mean'),
            'Run Success Rate': success_rate,
        })

    variant_comparison_df = pd.DataFrame(comparison_data)
    canonical_comparison_df = _build_canonical_table(variant_comparison_df)

    validate_aggregated_results(
        aggregated_results,
        metric_summary_key='metrics_summary',
        required_metric_keys=['precision', 'recall', 'f1'],
        timing_summary_key='timing_summary',
        required_timing_keys=['training_time'],
    )
    validate_comparison_dataframe(
        variant_comparison_df,
        required_columns=[
            'Approach',
            'Category',
            'Precision Mean',
            'Recall Mean',
            'F1 Mean',
            'Run Success Rate',
        ],
        label='variant',
    )
    validate_comparison_dataframe(
        canonical_comparison_df,
        required_columns=[
            'Category',
            'Approach',
            'F1 Mean',
            'F1 Std',
            'Run Success Rate',
            'Available',
        ],
        label='canonical',
    )
    
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY (CANONICAL 10 CATEGORIES)")
    print("=" * 70)
    print(canonical_comparison_df.to_string(index=False))
    
    # Save
    if save_results:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        save_json(
            output_path / "run_manifest.json",
            create_run_manifest(
                domain="domain_b",
                config={
                    'n_train': n_train,
                    'n_val': n_val,
                    'n_test': n_test,
                    'n_runs': len(seeds),
                    'smoke_test': smoke_test,
                },
                seed_list=seeds,
            ),
        )
        save_json(output_path / "results_raw_by_run.json", {'runs': all_run_results})
        save_json(output_path / "results_aggregated.json", {'approaches': aggregated_results})

        variant_comparison_df.to_csv(output_path / "comparison_variants.csv", index=False)
        canonical_comparison_df.to_csv(output_path / "comparison_canonical.csv", index=False)
        print(f"\nResults saved to {output_path}")

    return {
        'results_by_run': all_run_results,
        'results_aggregated': aggregated_results,
        'comparison_variants': variant_comparison_df,
        'comparison': canonical_comparison_df,
    }


if __name__ == "__main__":
    run_all_approaches(n_train=2000, n_val=500, n_test=500)