"""Run time series benchmark with reproducible multi-seed evaluation."""

import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np

from .data_generator import create_timeseries_dataset
from .approach_01_statistical import ARIMAForecaster
from .approach_02_exponential_smoothing import ExpSmoothingForecaster
from .approach_03_tree_based import TreeBasedForecaster
from .approach_04_rnn_lstm import LSTMForecaster
from .approach_05_cnn_temporal import TCNForecaster
from .approach_06_transformer import TransformerForecaster
from .approach_07_neural_prophet import ProphetStyleForecaster
from .approach_08_ensemble import EnsembleForecaster
from .approach_09_hybrid import HybridForecaster

from ..core.metrics import compute_timeseries_metrics
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
    "exponential_smoothing",
    "tree",
    "rnn_lstm",
    "cnn",
    "transformer",
    "prophet_style",
    "ensemble",
    "hybrid",
    "systems",
]


def _category_for_approach(name: str) -> str:
    low = name.lower()
    if "arima" in low:
        return "statistical"
    if "smoothing" in low:
        return "exponential_smoothing"
    if "tree" in low or "forest" in low:
        return "tree"
    if "lstm" in low or "rnn" in low:
        return "rnn_lstm"
    if "tcn" in low or "cnn" in low:
        return "cnn"
    if "transformer" in low:
        return "transformer"
    if "prophet" in low:
        return "prophet_style"
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
                    "RMSE Mean": np.nan,
                    "RMSE Std": np.nan,
                    "MAE Mean": np.nan,
                    "MASE Mean": np.nan,
                    "Run Success Rate": 0.0,
                    "Available": False,
                }
            )
            continue

        best_idx = candidates["RMSE Mean"].fillna(np.inf).idxmin()
        best = candidates.loc[best_idx]
        rows.append(
            {
                "Category": category,
                "Approach": best["Approach"],
                "RMSE Mean": float(best["RMSE Mean"]),
                "RMSE Std": float(best["RMSE Std"]),
                "MAE Mean": float(best["MAE Mean"]),
                "MASE Mean": float(best["MASE Mean"]),
                "Run Success Rate": float(best["Run Success Rate"]),
                "Available": True,
            }
        )
    return pd.DataFrame(rows)


def run_all_approaches(
    n_samples: int = 2000,
    forecast_horizon: int = 24,
    lookback: int = 168,
    save_results: bool = True,
    output_dir: str = "results/domain_d",
    n_runs: int = 1,
    seed: int = 42,
    seed_list: Optional[List[int]] = None,
    smoke_test: bool = False,
) -> Dict[str, Any]:
    """Run time series approaches with repeated seeded runs."""
    
    print("=" * 70)
    print("TIME SERIES FORECASTING: 10 Approaches Comparison")
    print("=" * 70)
    
    seeds = resolve_seed_list(n_runs=n_runs, seed=seed, seed_list=seed_list)
    all_run_results: List[List[Dict[str, Any]]] = []

    for run_idx, run_seed in enumerate(seeds, start=1):
        print(f"\n[Run {run_idx}/{len(seeds)}] Seed={run_seed}")
        set_global_seed(run_seed)

        print("\n[1/4] Generating dataset...")
        dataset = create_timeseries_dataset(
            n_samples=n_samples,
            forecast_horizon=forecast_horizon,
            lookback=lookback,
        )

        X_train, y_train = dataset['train']['X'], dataset['train']['y']
        X_val, y_val = dataset['val']['X'], dataset['val']['y']
        X_test, y_test = dataset['test']['X'], dataset['test']['y']
        train_series = dataset['train_series']

        print(f"  Train: {X_train.shape}, Test: {X_test.shape}")
        print(f"  Lookback: {lookback}, Horizon: {forecast_horizon}")

        print("\n[2/4] Initializing approaches...")
        if smoke_test:
            approaches = [
                ExpSmoothingForecaster({'seasonal_periods': max(2, forecast_horizon)}),
                TreeBasedForecaster({'n_estimators': 20}),
            ]
        else:
            approaches = [
                ARIMAForecaster({'order': (2, 1, 2)}),
                ExpSmoothingForecaster({'seasonal_periods': 24}),
                TreeBasedForecaster({'n_estimators': 50}),
                ProphetStyleForecaster({'seasonality_periods': [24, 168]}),
                LSTMForecaster({'epochs': 20, 'hidden_dim': 32}),
                TCNForecaster({'epochs': 20, 'hidden_dim': 32}),
                TransformerForecaster({'epochs': 20, 'd_model': 32}),
                EnsembleForecaster(),
                HybridForecaster(),
            ]
        print(f"  Initialized {len(approaches)} approaches")

        print("\n[3/4] Evaluating approaches...")
        run_results: List[Dict[str, Any]] = []

        naive_pred = np.tile(X_test[:, -1:, :], (1, forecast_horizon, 1))

        for approach in approaches:
            print(f"\n  Evaluating: {approach.name}")
            category = _category_for_approach(approach.name)

            try:
                start_time = time.time()
                approach.train(
                    X_train,
                    y_train,
                    X_val,
                    y_val,
                    train_series=train_series,
                    forecast_horizon=forecast_horizon,
                )
                train_time = time.time() - start_time
                print(f"    Training time: {train_time:.2f}s")

                start_time = time.time()
                y_pred = approach.predict(X_test)
                inference_time = time.time() - start_time

                if y_pred.shape != y_test.shape:
                    print(f"    Shape mismatch: pred={y_pred.shape}, test={y_test.shape}")
                    y_pred = y_pred.reshape(y_test.shape)

                y_test_flat = y_test.reshape(-1)
                y_pred_flat = y_pred.reshape(-1)
                naive_flat = naive_pred.reshape(-1)
                metrics = compute_timeseries_metrics(y_test_flat, y_pred_flat, naive_flat)

                print(f"    MAE: {metrics['mae']:.4f}")
                print(f"    RMSE: {metrics['rmse']:.4f}")
                print(f"    MASE: {metrics.get('mase', 0):.4f}")

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
                validate_run_row(row, required_metric_keys=['mae', 'rmse'])
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
            'MAE Mean': _safe_metric(metric_summary, 'mae', 'mean'),
            'MAE Std': _safe_metric(metric_summary, 'mae', 'std'),
            'RMSE Mean': _safe_metric(metric_summary, 'rmse', 'mean'),
            'RMSE Std': _safe_metric(metric_summary, 'rmse', 'std'),
            'MASE Mean': _safe_metric(metric_summary, 'mase', 'mean'),
            'Training Time Mean (s)': _safe_metric(time_summary, 'training_time', 'mean'),
            'Run Success Rate': success_rate,
        })

    variant_comparison_df = pd.DataFrame(comparison_data)
    canonical_comparison_df = _build_canonical_table(variant_comparison_df)

    validate_aggregated_results(
        aggregated_results,
        metric_summary_key='metrics_summary',
        required_metric_keys=['mae', 'rmse'],
        timing_summary_key='timing_summary',
        required_timing_keys=['training_time'],
    )
    validate_comparison_dataframe(
        variant_comparison_df,
        required_columns=[
            'Approach',
            'Category',
            'MAE Mean',
            'RMSE Mean',
            'Run Success Rate',
        ],
        label='variant',
    )
    validate_comparison_dataframe(
        canonical_comparison_df,
        required_columns=[
            'Category',
            'Approach',
            'RMSE Mean',
            'RMSE Std',
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
                domain="domain_d",
                config={
                    'n_samples': n_samples,
                    'forecast_horizon': forecast_horizon,
                    'lookback': lookback,
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
    run_all_approaches(n_samples=1000, forecast_horizon=12, lookback=48)