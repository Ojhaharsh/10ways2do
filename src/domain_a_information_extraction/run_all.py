"""Run information extraction benchmark with reproducible multi-seed evaluation."""

import json
import time
from pathlib import Path
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np

from .data_generator import create_ie_dataset, get_ie_fields, ResumeGenerator
from .approach_01_rule_based import RuleBasedIE
from .approach_02_classical_ml import ClassicalMLIE
from .approach_03_tree_based import TreeBasedIE
from .approach_04_rnn_lstm import RNNLSTMIE
from .approach_05_cnn_sequence import CNNIE
from .approach_06_transformer_scratch import TransformerScratchIE
from .approach_07_pretrained_transformer import PretrainedTransformerIE, HAS_TRANSFORMERS
from .approach_08_prompt_llm import PromptLLMIE
from .approach_09_hybrid import HybridIE, EnsembleIE
from .approach_10_systems import SystemsWrapper, SystemsEvaluator

from ..core.metrics import compute_ie_metrics
from ..core.evaluation import Evaluator, EvaluationResult
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
    "rule_based",
    "classical_ml",
    "tree_based",
    "rnn_lstm",
    "cnn",
    "transformer_scratch",
    "pretrained_transformer",
    "llm_prompting",
    "hybrid",
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
    if "classical" in low or "ml" in low:
        return "classical_ml"
    if "tree" in low or "xgboost" in low or "forest" in low:
        return "tree_based"
    if "rnn" in low or "lstm" in low:
        return "rnn_lstm"
    if "cnn" in low:
        return "cnn"
    if "scratch" in low and "transformer" in low:
        return "transformer_scratch"
    if "pretrained" in low or "bert" in low:
        return "pretrained_transformer"
    if "llm" in low or "prompt" in low:
        return "llm_prompting"
    if "hybrid" in low or "ensemble" in low:
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
                    "Exact Match Mean": np.nan,
                    "Exact Match Std": np.nan,
                    "Partial Match Mean": np.nan,
                    "Run Success Rate": 0.0,
                    "Available": False,
                }
            )
            continue

        best_idx = candidates["Exact Match Mean"].fillna(-np.inf).idxmax()
        best = candidates.loc[best_idx]
        rows.append(
            {
                "Category": category,
                "Approach": best["Approach"],
                "Exact Match Mean": float(best["Exact Match Mean"]),
                "Exact Match Std": float(best["Exact Match Std"]),
                "Partial Match Mean": float(best["Partial Match Mean"]),
                "Run Success Rate": float(best["Run Success Rate"]),
                "Available": True,
            }
        )
    return pd.DataFrame(rows)


def run_all_approaches(
    n_train: int = 2000,
    n_val: int = 500,
    n_test: int = 500,
    save_results: bool = True,
    output_dir: str = "results/domain_a",
    n_runs: int = 1,
    seed: int = 42,
    seed_list: Optional[List[int]] = None,
    smoke_test: bool = False,
) -> Dict[str, Any]:
    """
    Run all IE approaches and compare.
    """
    
    print("=" * 70)
    print("INFORMATION EXTRACTION: 10 Approaches Comparison")
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
        dataset = create_ie_dataset(n_train, n_val, n_test)

        X_train, y_train = dataset['train']['X'], dataset['train']['y']
        X_val, y_val = dataset['val']['X'], dataset['val']['y']
        X_test, y_test = dataset['test']['X'], dataset['test']['y']
        fields = dataset['fields']

        print(f"  Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

        print("\n[2/4] Initializing approaches...")
        if smoke_test:
            approaches = [
                RuleBasedIE(),
                ClassicalMLIE(),
                TreeBasedIE(),
            ]
        else:
            approaches = [
                RuleBasedIE(),
                ClassicalMLIE(),
                TreeBasedIE(),
                RNNLSTMIE({'epochs': 5, 'hidden_dim': 128}),
                CNNIE({'epochs': 5}),
                TransformerScratchIE({'epochs': 5, 'd_model': 128, 'num_layers': 2}),
                PromptLLMIE({'simulate': True}),
                HybridIE(),
            ]

        if HAS_TRANSFORMERS:
            try:
                approaches.insert(-1, PretrainedTransformerIE({'epochs': 1}))
            except Exception:
                print("  Warning: Could not initialize pretrained transformer")

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
                approach.metrics.training_time = train_time
                print(f"    Training time: {train_time:.2f}s")

                start_time = time.time()
                y_pred = approach.predict(X_test)
                inference_time = time.time() - start_time
                print(f"    Inference time: {inference_time:.2f}s ({len(X_test)} samples)")

                metrics = compute_ie_metrics(y_test, y_pred, fields)
                approach.metrics.primary_metrics = metrics

                print(f"    Overall Exact Match: {metrics.get('overall_exact_match', 0):.4f}")
                print(f"    Overall Partial Match: {metrics.get('overall_partial_match', 0):.4f}")

                approach.measure_latency([X_test[0]], n_iterations=50)
                print(f"    Latency p95: {approach.metrics.inference_latency_p95:.2f}ms")

                approach.metrics.failure_cases = approach.collect_failure_cases(
                    X_test, y_test, y_pred, n_cases=5
                )

                row = {
                    'name': approach.name,
                    'category': category,
                    'philosophy': approach.get_philosophy(),
                    'metrics': approach.metrics.to_dict(),
                    'seed': run_seed,
                    'train_time_cap_seconds': train_time_cap_seconds,
                    'memory_cap_mb': memory_cap_mb,
                    'tuning_trials_cap': tuning_trials_cap,
                    'out_of_budget': train_time > train_time_cap_seconds,
                    'success': True,
                }
                validate_run_row(
                    {
                        'name': row['name'],
                        'category': row['category'],
                        'seed': row['seed'],
                        'train_time_cap_seconds': row['train_time_cap_seconds'],
                        'memory_cap_mb': row['memory_cap_mb'],
                        'tuning_trials_cap': row['tuning_trials_cap'],
                        'out_of_budget': row['out_of_budget'],
                        'success': row['success'],
                        'metrics': row['metrics'].get('primary_metrics', {}),
                    },
                    required_metric_keys=['overall_exact_match', 'overall_partial_match'],
                )
                run_results.append(row)
            except Exception as e:
                print(f"    Error: {str(e)}")
                import traceback
                traceback.print_exc()
                row = {
                    'name': approach.name,
                    'category': category,
                    'seed': run_seed,
                    'train_time_cap_seconds': train_time_cap_seconds,
                    'memory_cap_mb': memory_cap_mb,
                    'tuning_trials_cap': tuning_trials_cap,
                    'out_of_budget': False,
                    'philosophy': approach.get_philosophy(),
                    'metrics': {},
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
        out_of_budget_count = sum(1 for r in rows if r.get('out_of_budget', False))
        budget_summary = {
            'train_time_cap_seconds': train_time_cap_seconds,
            'memory_cap_mb': memory_cap_mb,
            'tuning_trials_cap': tuning_trials_cap,
            'out_of_budget_count': out_of_budget_count,
            'out_of_budget_rate': out_of_budget_count / total_runs,
        }
        if not successful:
            aggregated_results.append({
                'name': name,
                'category': rows[0].get('category', 'systems'),
                'success_rate': 0.0,
                'budget_summary': budget_summary,
                'success': False,
            })
            continue

        primary_metrics = [r['metrics'].get('primary_metrics', {}) for r in successful]
        systems_metrics = [
            {
                'training_time': r['metrics'].get('training_time', np.nan),
                'inference_latency_p95': r['metrics'].get('inference_latency_p95', np.nan),
                'memory_inference_mb': r['metrics'].get('memory_inference_mb', np.nan),
                'model_size_mb': r['metrics'].get('model_size_mb', np.nan),
            }
            for r in successful
        ]
        primary_summary = aggregate_numeric_dicts(primary_metrics)
        systems_summary = aggregate_numeric_dicts(systems_metrics)

        success_rate = by_name_success.get(name, 0) / total_runs
        category = successful[0].get('category', 'systems')

        aggregated_results.append({
            'name': name,
            'category': category,
            'success_rate': success_rate,
            'primary_metrics_summary': primary_summary,
            'systems_metrics_summary': systems_summary,
            'budget_summary': budget_summary,
            'success': True,
        })

        comparison_data.append({
            'Approach': name,
            'Category': category,
            'Exact Match Mean': _safe_metric(primary_summary, 'overall_exact_match', 'mean'),
            'Exact Match Std': _safe_metric(primary_summary, 'overall_exact_match', 'std'),
            'Partial Match Mean': _safe_metric(primary_summary, 'overall_partial_match', 'mean'),
            'Latency p95 Mean (ms)': _safe_metric(systems_summary, 'inference_latency_p95', 'mean'),
            'Run Success Rate': success_rate,
        })

    variant_comparison_df = pd.DataFrame(comparison_data)
    canonical_comparison_df = _build_canonical_table(variant_comparison_df)

    validate_aggregated_results(
        aggregated_results,
        metric_summary_key='primary_metrics_summary',
        required_metric_keys=['overall_exact_match', 'overall_partial_match'],
        timing_summary_key='systems_metrics_summary',
        required_timing_keys=['inference_latency_p95'],
    )
    validate_comparison_dataframe(
        variant_comparison_df,
        required_columns=[
            'Approach',
            'Category',
            'Exact Match Mean',
            'Partial Match Mean',
            'Run Success Rate',
        ],
        label='variant',
    )
    validate_comparison_dataframe(
        canonical_comparison_df,
        required_columns=[
            'Category',
            'Approach',
            'Exact Match Mean',
            'Exact Match Std',
            'Run Success Rate',
            'Available',
        ],
        label='canonical',
    )
    
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY (CANONICAL 10 CATEGORIES)")
    print("=" * 70)
    print(canonical_comparison_df.to_string(index=False))
    
    # Save results
    if save_results:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        save_json(
            output_path / "run_manifest.json",
            create_run_manifest(
                domain="domain_a",
                config={
                    'n_train': n_train,
                    'n_val': n_val,
                    'n_test': n_test,
                    'n_runs': len(seeds),
                    'smoke_test': smoke_test,
                    'train_time_cap_seconds': train_time_cap_seconds,
                    'memory_cap_mb': memory_cap_mb,
                    'tuning_trials_cap': tuning_trials_cap,
                },
                seed_list=seeds,
            ),
        )
        save_json(output_path / "results_raw_by_run.json", {'runs': all_run_results})
        save_json(output_path / "results_aggregated.json", {'approaches': aggregated_results})

        variant_comparison_df.to_csv(output_path / "comparison_variants.csv", index=False)
        canonical_comparison_df.to_csv(output_path / "comparison_canonical.csv", index=False)

        # Generate report from first successful run for narrative details.
        first_successful_run = next((run for run in all_run_results if any(r.get('success', False) for r in run)), [])
        report = generate_report(first_successful_run, variant_comparison_df)
        with open(output_path / "report.md", 'w') as f:
            f.write(report)
        
        print(f"\nResults saved to {output_path}")
    
    return {
        'results_by_run': all_run_results,
        'results_aggregated': aggregated_results,
        'comparison_variants': variant_comparison_df,
        'comparison': canonical_comparison_df,
        'dataset_info': {
            'n_train': n_train,
            'n_val': n_val,
            'n_test': n_test,
            'fields': get_ie_fields(),
            'n_runs': len(seeds),
            'seeds': seeds,
        }
    }


def generate_report(results: List[Dict], comparison_df: pd.DataFrame) -> str:
    """Generate markdown report"""
    
    report = []
    report.append("# Information Extraction: 10 Approaches Comparison\n")
    report.append("## Executive Summary\n")

    exact_col = 'Exact Match' if 'Exact Match' in comparison_df.columns else 'Exact Match Mean'
    latency_col = 'Latency p95 (ms)' if 'Latency p95 (ms)' in comparison_df.columns else 'Latency p95 Mean (ms)'
    
    # Best performers
    if not comparison_df.empty and exact_col in comparison_df.columns and latency_col in comparison_df.columns:
        best_accuracy = comparison_df.loc[comparison_df[exact_col].idxmax()]
        best_speed = comparison_df.loc[comparison_df[latency_col].idxmin()]
        
        report.append(f"- **Best Accuracy**: {best_accuracy['Approach']} "
                     f"(Exact Match: {best_accuracy[exact_col]:.4f})\n")
        report.append(f"- **Fastest**: {best_speed['Approach']} "
                     f"(Latency: {best_speed[latency_col]:.2f}ms)\n")
    
    report.append("\n## Comparison Table\n")
    try:
        report.append(comparison_df.to_markdown(index=False))
    except ImportError:
        # Fall back when optional dependency "tabulate" is not installed.
        report.append(comparison_df.to_string(index=False))
    report.append("\n")
    
    report.append("\n## Approach Philosophies\n")
    for result in results:
        if result['success']:
            report.append(f"\n### {result['name']}\n")
            philosophy = result['philosophy']
            report.append(f"**Mental Model**: {philosophy.get('mental_model', 'N/A')}\n\n")
            report.append(f"**Strengths**: {philosophy.get('strengths', 'N/A')}\n\n")
            report.append(f"**Weaknesses**: {philosophy.get('weaknesses', 'N/A')}\n\n")
            report.append(f"**Best For**: {philosophy.get('best_for', 'N/A')}\n")
    
    report.append("\n## Key Insights\n")
    report.append("""
1. **Rule-based approaches** offer highest precision for well-defined patterns (email, phone) but fail on unstructured fields.

2. **Classical ML** provides a good balance of interpretability and performance with minimal compute.

3. **Deep learning approaches** (RNN, CNN, Transformer) require more data and compute but capture complex patterns.

4. **Pretrained transformers** offer best accuracy with transfer learning but have higher latency and memory footprint.

5. **LLM approaches** provide flexibility without training but have cost and latency concerns.

6. **Hybrid approaches** often perform best in production by combining strengths of multiple methods.
""")
    
    return "\n".join(report)


if __name__ == "__main__":
    run_all_approaches(n_train=1000, n_val=200, n_test=200)