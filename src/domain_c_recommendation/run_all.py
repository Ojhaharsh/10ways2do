"""Run recommendation benchmark with reproducible multi-seed evaluation."""

import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np

from .data_generator import create_recommendation_dataset
from .approach_01_popularity import PopularityRecommender
from .approach_02_collaborative_filtering import UserBasedCF, ItemBasedCF
from .approach_03_content_based import ContentBasedRecommender
from .approach_04_matrix_factorization import SVDRecommender, ALSRecommender
from .approach_05_deep_learning import NCFRecommender
from .approach_06_sequence_based import SequentialRecommender
from .approach_07_graph_neural import GraphNeuralRecommender
from .approach_08_transformer import TransformerRecommender
from .approach_09_hybrid import HybridRecommender

from ..core.metrics import compute_ranking_metrics
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
    "popularity",
    "collaborative_filtering",
    "content_based",
    "matrix_factorization",
    "deep_learning",
    "sequence",
    "graph",
    "transformer",
    "hybrid",
    "systems",
]


def _category_for_approach(name: str) -> str:
    low = name.lower()
    if "popularity" in low:
        return "popularity"
    if "user" in low or "item" in low or "cf" in low:
        return "collaborative_filtering"
    if "content" in low:
        return "content_based"
    if "svd" in low or "als" in low or "factor" in low:
        return "matrix_factorization"
    if "ncf" in low or "deep" in low:
        return "deep_learning"
    if "sequence" in low:
        return "sequence"
    if "graph" in low:
        return "graph"
    if "transformer" in low:
        return "transformer"
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
                    "NDCG@10 Mean": np.nan,
                    "NDCG@10 Std": np.nan,
                    "Recall@10 Mean": np.nan,
                    "MRR Mean": np.nan,
                    "Run Success Rate": 0.0,
                    "Available": False,
                }
            )
            continue
        best_idx = candidates["NDCG@10 Mean"].fillna(-np.inf).idxmax()
        best = candidates.loc[best_idx]
        rows.append(
            {
                "Category": category,
                "Approach": best["Approach"],
                "NDCG@10 Mean": float(best["NDCG@10 Mean"]),
                "NDCG@10 Std": float(best["NDCG@10 Std"]),
                "Recall@10 Mean": float(best["Recall@10 Mean"]),
                "MRR Mean": float(best["MRR Mean"]),
                "Run Success Rate": float(best["Run Success Rate"]),
                "Available": True,
            }
        )
    return pd.DataFrame(rows)


def run_all_approaches(
    n_users: int = 500,
    n_items: int = 200,
    save_results: bool = True,
    output_dir: str = "results/domain_c",
    n_runs: int = 1,
    seed: int = 42,
    seed_list: Optional[List[int]] = None,
    smoke_test: bool = False,
) -> Dict[str, Any]:
    """Run recommendation approaches with repeated seeded runs."""
    
    print("=" * 70)
    print("RECOMMENDATION: 10 Approaches Comparison")
    print("=" * 70)
    
    seeds = resolve_seed_list(n_runs=n_runs, seed=seed, seed_list=seed_list)
    all_run_results: List[List[Dict[str, Any]]] = []

    for run_idx, run_seed in enumerate(seeds, start=1):
        print(f"\n[Run {run_idx}/{len(seeds)}] Seed={run_seed}")
        set_global_seed(run_seed)

        print("\n[1/4] Generating dataset...")
        dataset = create_recommendation_dataset(n_users, n_items, sparsity=0.95)

        train_matrix = dataset['train_matrix']
        train_interactions = dataset['train_interactions']
        test_items = dataset['test_items']

        print(f"  Users: {n_users}, Items: {n_items}")
        print(f"  Train interactions: {len(train_interactions)}")
        print(f"  Test users with items: {len(test_items)}")

        print("\n[2/4] Initializing approaches...")
        if smoke_test:
            approaches = [
                PopularityRecommender(),
                UserBasedCF({'k_neighbors': 10}),
                SVDRecommender({'n_factors': 10}),
            ]
        else:
            approaches = [
                PopularityRecommender(),
                UserBasedCF({'k_neighbors': 30}),
                ItemBasedCF({'k_neighbors': 30}),
                ContentBasedRecommender(),
                SVDRecommender({'n_factors': 30}),
                ALSRecommender({'n_factors': 30, 'iterations': 10}),
                NCFRecommender({'epochs': 10, 'embed_dim': 32}),
                SequentialRecommender({'epochs': 10}),
                GraphNeuralRecommender({'epochs': 20, 'embed_dim': 32}),
                TransformerRecommender({'epochs': 10}),
                HybridRecommender()
            ]
        print(f"  Initialized {len(approaches)} approaches")

        print("\n[3/4] Evaluating approaches...")
        run_results: List[Dict[str, Any]] = []
        test_user_ids = list(test_items.keys())
        y_true = [test_items[u] for u in test_user_ids]

        for approach in approaches:
            print(f"\n  Evaluating: {approach.name}")
            category = _category_for_approach(approach.name)

            try:
                start_time = time.time()
                if hasattr(approach, 'train'):
                    if 'item_features' in approach.train.__code__.co_varnames:
                        approach.train(
                            train_matrix,
                            train_interactions,
                            item_features=dataset.get('item_features'),
                        )
                    else:
                        approach.train(train_matrix, train_interactions)
                train_time = time.time() - start_time

                start_time = time.time()
                y_pred = approach.predict(test_user_ids, k=20)
                inference_time = time.time() - start_time

                metrics = compute_ranking_metrics(y_true, y_pred, k_values=[5, 10, 20])

                print(f"    NDCG@10: {metrics.get('ndcg@10', 0):.4f}")
                print(f"    Recall@10: {metrics.get('recall@10', 0):.4f}")
                print(f"    MRR: {metrics.get('mrr', 0):.4f}")

                row = {
                    'name': approach.name,
                    'category': category,
                    'philosophy': approach.get_philosophy(),
                    'metrics': metrics,
                    'training_time': train_time,
                    'inference_time': inference_time,
                    'seed': run_seed,
                    'success': True,
                }
                validate_run_row(row, required_metric_keys=['ndcg@10', 'recall@10', 'mrr'])
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
                    'error': str(e),
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

    total_runs = len(seeds)
    comparison_data: List[Dict[str, Any]] = []
    aggregated_results: List[Dict[str, Any]] = []

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
            'NDCG@10 Mean': _safe_metric(metric_summary, 'ndcg@10', 'mean'),
            'NDCG@10 Std': _safe_metric(metric_summary, 'ndcg@10', 'std'),
            'Recall@10 Mean': _safe_metric(metric_summary, 'recall@10', 'mean'),
            'MRR Mean': _safe_metric(metric_summary, 'mrr', 'mean'),
            'Training Time Mean (s)': _safe_metric(time_summary, 'training_time', 'mean'),
            'Run Success Rate': success_rate,
        })

    variant_comparison_df = pd.DataFrame(comparison_data)
    canonical_comparison_df = _build_canonical_table(variant_comparison_df)

    validate_aggregated_results(
        aggregated_results,
        metric_summary_key='metrics_summary',
        required_metric_keys=['ndcg@10', 'recall@10', 'mrr'],
        timing_summary_key='timing_summary',
        required_timing_keys=['training_time'],
    )
    validate_comparison_dataframe(
        variant_comparison_df,
        required_columns=[
            'Approach',
            'Category',
            'NDCG@10 Mean',
            'Recall@10 Mean',
            'MRR Mean',
            'Run Success Rate',
        ],
        label='variant',
    )
    validate_comparison_dataframe(
        canonical_comparison_df,
        required_columns=[
            'Category',
            'Approach',
            'NDCG@10 Mean',
            'NDCG@10 Std',
            'Run Success Rate',
            'Available',
        ],
        label='canonical',
    )
    
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY (CANONICAL 10 CATEGORIES)")
    print("=" * 70)
    print(canonical_comparison_df.to_string(index=False))
    
    if save_results:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        save_json(
            output_path / "run_manifest.json",
            create_run_manifest(
                domain="domain_c",
                config={
                    'n_users': n_users,
                    'n_items': n_items,
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
    run_all_approaches(n_users=300, n_items=150)