"""
Smoke tests for end-to-end benchmark artifact generation.
"""

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.domain_a_information_extraction.run_all import run_all_approaches as run_a
from src.domain_b_anomaly_detection.run_all import run_all_approaches as run_b
from src.domain_c_recommendation.run_all import run_all_approaches as run_c
from src.domain_d_time_series.run_all import run_all_approaches as run_d


REQUIRED_FILES = [
    "run_manifest.json",
    "results_raw_by_run.json",
    "results_aggregated.json",
    "comparison_variants.csv",
    "comparison_canonical.csv",
]


@pytest.mark.parametrize(
    "domain_name,runner,kwargs",
    [
        (
            "domain_a",
            run_a,
            {
                "n_train": 80,
                "n_val": 20,
                "n_test": 20,
                "n_runs": 1,
                "seed": 42,
                "smoke_test": True,
            },
        ),
        (
            "domain_b",
            run_b,
            {
                "n_train": 120,
                "n_val": 40,
                "n_test": 40,
                "n_runs": 1,
                "seed": 42,
                "smoke_test": True,
            },
        ),
        (
            "domain_c",
            run_c,
            {
                "n_users": 60,
                "n_items": 40,
                "n_runs": 1,
                "seed": 42,
                "smoke_test": True,
            },
        ),
        (
            "domain_d",
            run_d,
            {
                "n_samples": 300,
                "forecast_horizon": 6,
                "lookback": 24,
                "n_runs": 1,
                "seed": 42,
                "smoke_test": True,
            },
        ),
    ],
)
def test_smoke_artifacts(domain_name, runner, kwargs, tmp_path):
    output_dir = tmp_path / domain_name
    runner(save_results=True, output_dir=str(output_dir), **kwargs)

    for filename in REQUIRED_FILES:
        assert (output_dir / filename).exists(), f"Missing artifact: {filename}"

    with open(output_dir / "run_manifest.json", encoding="utf-8") as f:
        manifest = json.load(f)
    assert manifest["domain"] == domain_name
    assert manifest["config"]["smoke_test"] is True

    with open(output_dir / "results_aggregated.json", encoding="utf-8") as f:
        aggregated = json.load(f)
    assert "approaches" in aggregated
    assert isinstance(aggregated["approaches"], list)
    assert len(aggregated["approaches"]) > 0

    canonical = pd.read_csv(output_dir / "comparison_canonical.csv")
    assert "Category" in canonical.columns
    assert "Available" in canonical.columns
    assert len(canonical) == 10

    variants = pd.read_csv(output_dir / "comparison_variants.csv")
    assert "Approach" in variants.columns
    assert "Category" in variants.columns
