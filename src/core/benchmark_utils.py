"""
Shared utilities for reproducible benchmark runs.
"""

from __future__ import annotations

import json
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
    """Aggregate numeric keys in a list of dicts into mean/std summaries."""
    values_by_key: Dict[str, List[float]] = {}

    for row in rows:
        for k, v in row.items():
            if isinstance(v, (int, float)) and np.isfinite(v):
                values_by_key.setdefault(k, []).append(float(v))

    summary: Dict[str, Dict[str, float]] = {}
    for k, vals in values_by_key.items():
        arr = np.array(vals, dtype=float)
        summary[k] = {
            "mean": float(np.mean(arr)),
            "std": float(np.std(arr)),
            "n": int(arr.shape[0]),
        }
    return summary
