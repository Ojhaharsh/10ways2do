"""Repository-level validation for benchmark artifact directories."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import pandas as pd

REQUIRED_FILES = [
    "run_manifest.json",
    "results_raw_by_run.json",
    "results_aggregated.json",
    "comparison_variants.csv",
    "comparison_canonical.csv",
]

DEFAULT_DOMAIN_DIRS = [
    "domain_a",
    "domain_b",
    "domain_c",
    "domain_d",
    "domain_e",
    "domain_f",
    "domain_g",
    "domain_h",
    "domain_i",
    "domain_j",
    "domain_k",
]


class ArtifactValidationError(ValueError):
    """Raised when one or more benchmark artifact checks fail."""



def _load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)



def _validate_manifest(path: Path, expected_domain: str, errors: List[str]) -> None:
    try:
        manifest = _load_json(path)
    except Exception as exc:
        errors.append(f"{path}: cannot parse JSON ({exc})")
        return

    if manifest.get("domain") != expected_domain:
        errors.append(
            f"{path}: expected domain='{expected_domain}' but found '{manifest.get('domain')}'"
        )



def _validate_aggregated(path: Path, errors: List[str]) -> None:
    try:
        payload = _load_json(path)
    except Exception as exc:
        errors.append(f"{path}: cannot parse JSON ({exc})")
        return

    approaches = payload.get("approaches")
    if not isinstance(approaches, list) or not approaches:
        errors.append(f"{path}: missing non-empty 'approaches' list")
        return

    for idx, row in enumerate(approaches):
        if not isinstance(row, dict):
            errors.append(f"{path}: approaches[{idx}] is not an object")
            continue
        if "budget_summary" not in row:
            errors.append(f"{path}: approaches[{idx}] missing 'budget_summary'")
        if row.get("success") and "significance_vs_best" not in row:
            errors.append(f"{path}: approaches[{idx}] missing 'significance_vs_best' for success row")



def _validate_raw(path: Path, errors: List[str]) -> None:
    try:
        payload = _load_json(path)
    except Exception as exc:
        errors.append(f"{path}: cannot parse JSON ({exc})")
        return

    runs = payload.get("runs")
    if not isinstance(runs, list):
        errors.append(f"{path}: missing 'runs' list")
        return

    for run_idx, run_rows in enumerate(runs):
        if not isinstance(run_rows, list):
            errors.append(f"{path}: runs[{run_idx}] is not a list")
            continue
        for row_idx, row in enumerate(run_rows):
            if not isinstance(row, dict):
                errors.append(f"{path}: runs[{run_idx}][{row_idx}] is not an object")
                continue
            for key in [
                "train_time_cap_seconds",
                "memory_cap_mb",
                "tuning_trials_cap",
                "out_of_budget",
            ]:
                if key not in row:
                    errors.append(f"{path}: runs[{run_idx}][{row_idx}] missing '{key}'")



def _validate_csv(path: Path, required_columns: Sequence[str], label: str, errors: List[str]) -> None:
    try:
        df = pd.read_csv(path)
    except Exception as exc:
        errors.append(f"{path}: cannot read CSV ({exc})")
        return

    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        errors.append(f"{path}: {label} missing columns {missing}")

    ci_low_cols = [c for c in df.columns if c.endswith("CI95 Low")]
    ci_high_cols = [c for c in df.columns if c.endswith("CI95 High")]
    p_value_cols = [c for c in df.columns if "p-value vs Best" in c]
    if not ci_low_cols:
        errors.append(f"{path}: {label} missing CI95 Low columns")
    if not ci_high_cols:
        errors.append(f"{path}: {label} missing CI95 High columns")
    if not p_value_cols:
        errors.append(f"{path}: {label} missing p-value columns")



def validate_domain_artifacts(domain_dir: Path) -> List[str]:
    """Validate one domain artifact directory and return a list of errors."""
    errors: List[str] = []

    expected_domain = domain_dir.name
    for filename in REQUIRED_FILES:
        file_path = domain_dir / filename
        if not file_path.exists():
            errors.append(f"{file_path}: missing required file")

    if errors:
        return errors

    _validate_manifest(domain_dir / "run_manifest.json", expected_domain, errors)
    _validate_aggregated(domain_dir / "results_aggregated.json", errors)
    _validate_raw(domain_dir / "results_raw_by_run.json", errors)

    _validate_csv(
        domain_dir / "comparison_canonical.csv",
        required_columns=["Category", "Available"],
        label="canonical",
        errors=errors,
    )
    _validate_csv(
        domain_dir / "comparison_variants.csv",
        required_columns=["Approach", "Category"],
        label="variants",
        errors=errors,
    )

    return errors



def validate_results_tree(
    results_dir: str | Path = "results",
    domain_dirs: Iterable[str] | None = None,
) -> None:
    """Validate all domain artifacts under results_dir.

    Raises:
        ArtifactValidationError: if any checks fail.
    """
    root = Path(results_dir)
    if not root.exists():
        raise ArtifactValidationError(f"Results directory not found: {root}")

    domain_names = list(domain_dirs) if domain_dirs is not None else DEFAULT_DOMAIN_DIRS
    all_errors: List[str] = []

    for domain_name in domain_names:
        domain_path = root / domain_name
        if not domain_path.exists():
            all_errors.append(f"{domain_path}: domain directory not found")
            continue
        all_errors.extend(validate_domain_artifacts(domain_path))

    if all_errors:
        preview = "\n".join(f"- {err}" for err in all_errors[:25])
        if len(all_errors) > 25:
            preview += f"\n- ... and {len(all_errors) - 25} more"
        raise ArtifactValidationError(
            f"Artifact validation failed with {len(all_errors)} issue(s):\n{preview}"
        )
