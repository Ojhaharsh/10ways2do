"""
Shared utilities for reproducible benchmark runs.
"""

from __future__ import annotations

import json
import math
import os
import platform
import random
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


BENCHMARK_PROTOCOL_VERSION = "1.0.0"


def set_global_seed(seed: int) -> None:
    """Set global random seeds for reproducibility across common libraries."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        try:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
        except Exception:
            pass
    except Exception:
        pass


def resolve_seed_list(n_runs: int, seed: int = 42, seed_list: Optional[List[int]] = None) -> List[int]:
    """Build seed list for repeated runs.

    If seed_list is provided, it is used as-is. Otherwise creates [seed, seed+1, ...].
    """
    if seed_list:
        return seed_list
    return [seed + i for i in range(max(1, n_runs))]


def _safe_package_version(pkg_name: str) -> Optional[str]:
    try:
        return metadata.version(pkg_name)
    except Exception:
        return None


def _get_git_commit_hash() -> Optional[str]:
    """Return the current git commit hash when available."""
    # CI environments often expose the commit directly.
    env_sha = os.getenv("GITHUB_SHA")
    if env_sha:
        return env_sha

    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip() or None
    except Exception:
        return None


def create_run_manifest(domain: str, config: Dict[str, Any], seed_list: List[int]) -> Dict[str, Any]:
    """Create run manifest with environment and dependency metadata."""
    packages = [
        "numpy",
        "pandas",
        "scikit-learn",
        "torch",
        "transformers",
        "xgboost",
        "lightgbm",
        "catboost",
        "statsmodels",
        "prophet",
    ]

    package_versions = {}
    for pkg in packages:
        version = _safe_package_version(pkg)
        if version is not None:
            package_versions[pkg] = version

    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "domain": domain,
        "benchmark_protocol_version": BENCHMARK_PROTOCOL_VERSION,
        "git_commit_hash": _get_git_commit_hash(),
        "config": config,
        "seeds": seed_list,
        "environment": {
            "python_version": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "executable": sys.executable,
        },
        "package_versions": package_versions,
    }


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    """Save a dict as pretty JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)


def aggregate_numeric_dicts(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    """Aggregate numeric keys in a list of dicts into statistical summaries."""
    values_by_key: Dict[str, List[float]] = {}

    for row in rows:
        for k, v in row.items():
            if isinstance(v, (int, float)) and np.isfinite(v):
                values_by_key.setdefault(k, []).append(float(v))

    summary: Dict[str, Dict[str, float]] = {}
    for k, vals in values_by_key.items():
        arr = np.array(vals, dtype=float)
        n = int(arr.shape[0])
        mean = float(np.mean(arr))
        std = float(np.std(arr))
        sem = float(std / np.sqrt(n)) if n > 0 else float("nan")
        ci95_half = 1.96 * sem if np.isfinite(sem) else float("nan")
        summary[k] = {
            "mean": mean,
            "std": std,
            "n": n,
            "sem": sem,
            "ci95_low": float(mean - ci95_half) if np.isfinite(ci95_half) else float("nan"),
            "ci95_high": float(mean + ci95_half) if np.isfinite(ci95_half) else float("nan"),
        }
    return summary


def _welch_ttest_pvalue_normal_approx(sample_a: List[float], sample_b: List[float]) -> Optional[float]:
    """Approximate two-sided p-value for Welch's t-test using normal tail."""
    if len(sample_a) < 2 or len(sample_b) < 2:
        return None

    a = np.array(sample_a, dtype=float)
    b = np.array(sample_b, dtype=float)

    mean_a = float(np.mean(a))
    mean_b = float(np.mean(b))
    var_a = float(np.var(a, ddof=1))
    var_b = float(np.var(b, ddof=1))

    denom = np.sqrt((var_a / len(a)) + (var_b / len(b)))
    if not np.isfinite(denom) or denom <= 0:
        return None

    z = abs((mean_a - mean_b) / denom)
    p_value = float(math.erfc(z / np.sqrt(2.0)))
    return p_value


def _cohens_d_directional(sample_a: List[float], sample_b: List[float]) -> Optional[float]:
    """Compute directional Cohen's d between two samples using pooled std.

    Returns None when d cannot be estimated reliably (e.g., too few points or
    near-zero pooled variance).
    """
    if len(sample_a) < 2 or len(sample_b) < 2:
        return None

    a = np.array(sample_a, dtype=float)
    b = np.array(sample_b, dtype=float)

    mean_a = float(np.mean(a))
    mean_b = float(np.mean(b))
    var_a = float(np.var(a, ddof=1))
    var_b = float(np.var(b, ddof=1))

    n_a = len(a)
    n_b = len(b)
    denom_df = (n_a + n_b - 2)
    if denom_df <= 0:
        return None

    pooled_var = (((n_a - 1) * var_a) + ((n_b - 1) * var_b)) / denom_df
    if not np.isfinite(pooled_var) or pooled_var <= 0:
        return None

    pooled_std = float(np.sqrt(pooled_var))
    if not np.isfinite(pooled_std) or pooled_std <= 0:
        return None

    return float((mean_a - mean_b) / pooled_std)


def compute_significance_vs_best(
    metric_samples_by_name: Dict[str, List[float]],
    higher_is_better: bool,
    alpha: float = 0.05,
) -> Dict[str, Dict[str, Any]]:
    """Compute per-approach significance against the best mean approach."""
    clean_samples: Dict[str, List[float]] = {}
    for name, samples in metric_samples_by_name.items():
        clean = [float(v) for v in samples if isinstance(v, (int, float)) and np.isfinite(v)]
        if clean:
            clean_samples[name] = clean

    if not clean_samples:
        return {}

    means = {name: float(np.mean(vals)) for name, vals in clean_samples.items()}
    best_name = max(means, key=means.get) if higher_is_better else min(means, key=means.get)
    best_mean = means[best_name]
    best_samples = clean_samples[best_name]

    result: Dict[str, Dict[str, Any]] = {}
    for name, samples in clean_samples.items():
        mean_value = means[name]
        if name == best_name:
            p_value: Optional[float] = 1.0
            cohens_d_vs_best: Optional[float] = 0.0
        else:
            p_value = _welch_ttest_pvalue_normal_approx(samples, best_samples)
            raw_d = _cohens_d_directional(samples, best_samples)
            cohens_d_vs_best = None if raw_d is None else (raw_d if higher_is_better else -raw_d)

        if higher_is_better:
            mean_diff_vs_best = mean_value - best_mean
            significantly_better = (
                p_value is not None and p_value < alpha and mean_value > best_mean
            )
        else:
            mean_diff_vs_best = best_mean - mean_value
            significantly_better = (
                p_value is not None and p_value < alpha and mean_value < best_mean
            )

        result[name] = {
            "best_approach": best_name,
            "best_mean": best_mean,
            "higher_is_better": higher_is_better,
            "alpha": alpha,
            "is_best": name == best_name,
            "mean_diff_vs_best": float(mean_diff_vs_best),
            "p_value": p_value,
            "cohens_d_vs_best": cohens_d_vs_best,
            "significantly_better_than_best": significantly_better,
        }

    return result
