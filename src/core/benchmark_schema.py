"""Validation utilities for benchmark result schemas."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np


def _require(row: Dict[str, Any], key: str) -> Any:
    if key not in row:
        raise ValueError(f"Benchmark row missing required key: {key}")
    return row[key]


def _validate_numeric(value: Any, name: str) -> None:
    if not isinstance(value, (int, float)):
        raise ValueError(f"Benchmark metric '{name}' must be numeric, got {type(value).__name__}")
    if not np.isfinite(float(value)):
        raise ValueError(f"Benchmark metric '{name}' must be finite, got {value}")


def _validate_summary_stats(summary: Dict[str, Any], metric_key: str) -> None:
    if metric_key not in summary:
        raise ValueError(f"Aggregated summary missing metric '{metric_key}'")

    stats = summary[metric_key]
    if not isinstance(stats, dict):
        raise ValueError(f"Summary stats for '{metric_key}' must be a dict")

    for stat_key in ["mean", "std", "n", "sem", "ci95_low", "ci95_high"]:
        if stat_key not in stats:
            raise ValueError(f"Summary stats for '{metric_key}' missing '{stat_key}'")

    _validate_numeric(stats["mean"], f"{metric_key}.mean")
    _validate_numeric(stats["std"], f"{metric_key}.std")
    _validate_numeric(stats["sem"], f"{metric_key}.sem")
    _validate_numeric(stats["ci95_low"], f"{metric_key}.ci95_low")
    _validate_numeric(stats["ci95_high"], f"{metric_key}.ci95_high")

    if not isinstance(stats["n"], (int, float)):
        raise ValueError(f"Summary stats for '{metric_key}' has non-numeric n")
    if int(stats["n"]) < 1:
        raise ValueError(f"Summary stats for '{metric_key}' must have n >= 1")


def validate_run_row(row: Dict[str, Any], required_metric_keys: Optional[List[str]] = None) -> None:
    """Validate per-approach per-run benchmark rows.

    Required base keys:
    - name (str)
    - category (str)
    - seed (int)
    - success (bool)

    Success rows must include metrics (dict) and any required metric keys.
    Failure rows must include error (str).
    """
    if not isinstance(row, dict):
        raise ValueError("Benchmark row must be a dict")

    name = _require(row, "name")
    category = _require(row, "category")
    seed = _require(row, "seed")
    success = _require(row, "success")

    if not isinstance(name, str) or not name.strip():
        raise ValueError("Benchmark row 'name' must be a non-empty string")
    if not isinstance(category, str) or not category.strip():
        raise ValueError("Benchmark row 'category' must be a non-empty string")
    if not isinstance(seed, int):
        raise ValueError("Benchmark row 'seed' must be an int")
    if not isinstance(success, bool):
        raise ValueError("Benchmark row 'success' must be a bool")

    train_time_cap_seconds = _require(row, "train_time_cap_seconds")
    memory_cap_mb = _require(row, "memory_cap_mb")
    tuning_trials_cap = _require(row, "tuning_trials_cap")
    out_of_budget = _require(row, "out_of_budget")

    _validate_numeric(train_time_cap_seconds, "train_time_cap_seconds")
    if memory_cap_mb is not None:
        _validate_numeric(memory_cap_mb, "memory_cap_mb")
    if not isinstance(tuning_trials_cap, int) or tuning_trials_cap < 0:
        raise ValueError("Benchmark row 'tuning_trials_cap' must be a non-negative int")
    if not isinstance(out_of_budget, bool):
        raise ValueError("Benchmark row 'out_of_budget' must be a bool")

    if success:
        metrics = _require(row, "metrics")
        if not isinstance(metrics, dict) or not metrics:
            raise ValueError("Successful benchmark row must contain non-empty dict 'metrics'")

        for metric_key in (required_metric_keys or []):
            if metric_key not in metrics:
                raise ValueError(f"Benchmark row metrics missing required key: {metric_key}")
            _validate_numeric(metrics[metric_key], metric_key)
    else:
        error = _require(row, "error")
        if not isinstance(error, str) or not error.strip():
            raise ValueError("Failed benchmark row must include non-empty string 'error'")


def validate_aggregated_results(
    aggregated_results: List[Dict[str, Any]],
    metric_summary_key: str,
    required_metric_keys: List[str],
    timing_summary_key: Optional[str] = None,
    required_timing_keys: Optional[List[str]] = None,
) -> None:
    """Validate aggregated per-approach results structure and required summaries."""
    if not isinstance(aggregated_results, list):
        raise ValueError("Aggregated results must be a list")
    if not aggregated_results:
        raise ValueError("Aggregated results must not be empty")

    for row in aggregated_results:
        if not isinstance(row, dict):
            raise ValueError("Each aggregated result row must be a dict")

        _require(row, "name")
        _require(row, "category")
        success_rate = _require(row, "success_rate")
        success = _require(row, "success")

        _validate_numeric(success_rate, "success_rate")
        if not (0.0 <= float(success_rate) <= 1.0):
            raise ValueError("success_rate must be in [0, 1]")
        if not isinstance(success, bool):
            raise ValueError("Aggregated row 'success' must be bool")

        budget_summary = _require(row, "budget_summary")
        if not isinstance(budget_summary, dict):
            raise ValueError("Aggregated row 'budget_summary' must be dict")
        for budget_key in [
            "train_time_cap_seconds",
            "memory_cap_mb",
            "tuning_trials_cap",
            "out_of_budget_count",
            "out_of_budget_rate",
        ]:
            if budget_key not in budget_summary:
                raise ValueError(f"Aggregated budget_summary missing '{budget_key}'")

        _validate_numeric(budget_summary["train_time_cap_seconds"], "budget_summary.train_time_cap_seconds")
        if budget_summary["memory_cap_mb"] is not None:
            _validate_numeric(budget_summary["memory_cap_mb"], "budget_summary.memory_cap_mb")

        tuning_trials_cap = budget_summary["tuning_trials_cap"]
        out_of_budget_count = budget_summary["out_of_budget_count"]
        out_of_budget_rate = budget_summary["out_of_budget_rate"]
        if not isinstance(tuning_trials_cap, int) or tuning_trials_cap < 0:
            raise ValueError("budget_summary.tuning_trials_cap must be a non-negative int")
        if not isinstance(out_of_budget_count, int) or out_of_budget_count < 0:
            raise ValueError("budget_summary.out_of_budget_count must be a non-negative int")
        _validate_numeric(out_of_budget_rate, "budget_summary.out_of_budget_rate")
        if not (0.0 <= float(out_of_budget_rate) <= 1.0):
            raise ValueError("budget_summary.out_of_budget_rate must be in [0, 1]")

        if not success:
            continue

        metric_summary = _require(row, metric_summary_key)
        if not isinstance(metric_summary, dict):
            raise ValueError(f"Aggregated row '{metric_summary_key}' must be dict")
        for metric_key in required_metric_keys:
            _validate_summary_stats(metric_summary, metric_key)

        if timing_summary_key is not None and required_timing_keys is not None:
            timing_summary = _require(row, timing_summary_key)
            if not isinstance(timing_summary, dict):
                raise ValueError(f"Aggregated row '{timing_summary_key}' must be dict")
            for timing_key in required_timing_keys:
                _validate_summary_stats(timing_summary, timing_key)

        significance = _require(row, "significance_vs_best")
        if not isinstance(significance, dict):
            raise ValueError("Aggregated row 'significance_vs_best' must be dict")

        for key in ["best_approach", "best_mean", "higher_is_better", "alpha", "is_best", "mean_diff_vs_best", "p_value", "significantly_better_than_best"]:
            if key not in significance:
                raise ValueError(f"Aggregated row significance_vs_best missing '{key}'")

        if not isinstance(significance["best_approach"], str) or not significance["best_approach"].strip():
            raise ValueError("significance_vs_best.best_approach must be a non-empty string")
        _validate_numeric(significance["best_mean"], "significance_vs_best.best_mean")
        if not isinstance(significance["higher_is_better"], bool):
            raise ValueError("significance_vs_best.higher_is_better must be bool")
        _validate_numeric(significance["alpha"], "significance_vs_best.alpha")
        if not isinstance(significance["is_best"], bool):
            raise ValueError("significance_vs_best.is_best must be bool")
        _validate_numeric(significance["mean_diff_vs_best"], "significance_vs_best.mean_diff_vs_best")
        p_value = significance["p_value"]
        if p_value is not None:
            _validate_numeric(p_value, "significance_vs_best.p_value")
            if not (0.0 <= float(p_value) <= 1.0):
                raise ValueError("significance_vs_best.p_value must be in [0, 1]")
        if not isinstance(significance["significantly_better_than_best"], bool):
            raise ValueError("significance_vs_best.significantly_better_than_best must be bool")


def validate_comparison_dataframe(df: Any, required_columns: List[str], label: str) -> None:
    """Validate comparison dataframe has expected columns."""
    if not hasattr(df, "columns"):
        raise ValueError(f"{label} comparison object must have 'columns'")

    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise ValueError(f"{label} comparison missing columns: {missing}")
