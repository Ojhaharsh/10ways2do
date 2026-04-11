import json
from pathlib import Path

import pandas as pd
import pytest

from src.core.artifact_validator import ArtifactValidationError, validate_results_tree


def _write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f)


def _create_minimal_domain_artifacts(domain_dir: Path):
    _write_json(
        domain_dir / "run_manifest.json",
        {
            "domain": domain_dir.name,
            "benchmark_protocol_version": "1.0.0",
        },
    )
    _write_json(
        domain_dir / "results_raw_by_run.json",
        {
            "runs": [
                [
                    {
                        "train_time_cap_seconds": 30.0,
                        "memory_cap_mb": 4096.0,
                        "tuning_trials_cap": 0,
                        "out_of_budget": False,
                    }
                ]
            ]
        },
    )
    _write_json(
        domain_dir / "results_aggregated.json",
        {
            "approaches": [
                {
                    "name": "Dummy",
                    "category": "rule_based",
                    "success": True,
                    "budget_summary": {
                        "train_time_cap_seconds": 30.0,
                        "memory_cap_mb": 4096.0,
                        "tuning_trials_cap": 0,
                        "out_of_budget_count": 0,
                        "out_of_budget_rate": 0.0,
                    },
                    "significance_vs_best": {
                        "best_approach": "Dummy",
                    },
                }
            ]
        },
    )

    pd.DataFrame(
        [
            {
                "Category": "rule_based",
                "Available": True,
                "F1 CI95 Low": 0.1,
                "F1 CI95 High": 0.9,
                "F1 p-value vs Best": 1.0,
            }
        ]
    ).to_csv(domain_dir / "comparison_canonical.csv", index=False)

    pd.DataFrame(
        [
            {
                "Approach": "Dummy",
                "Category": "rule_based",
                "F1 CI95 Low": 0.1,
                "F1 CI95 High": 0.9,
                "F1 p-value vs Best": 1.0,
            }
        ]
    ).to_csv(domain_dir / "comparison_variants.csv", index=False)


def test_validate_results_tree_passes_on_minimal_contract(tmp_path):
    results_dir = tmp_path / "results"
    domain_names = [
        "domain_a",
        "domain_b",
        "domain_c",
        "domain_d",
        "domain_e",
        "domain_f",
        "domain_g",
        "domain_h",
        "domain_i",
    ]

    for name in domain_names:
        _create_minimal_domain_artifacts(results_dir / name)

    validate_results_tree(results_dir=results_dir)


def test_validate_results_tree_fails_on_missing_required_file(tmp_path):
    results_dir = tmp_path / "results"
    domain_names = [
        "domain_a",
        "domain_b",
        "domain_c",
        "domain_d",
        "domain_e",
        "domain_f",
        "domain_g",
        "domain_h",
        "domain_i",
    ]

    for name in domain_names:
        _create_minimal_domain_artifacts(results_dir / name)

    (results_dir / "domain_i" / "comparison_variants.csv").unlink()

    with pytest.raises(ArtifactValidationError):
        validate_results_tree(results_dir=results_dir)
