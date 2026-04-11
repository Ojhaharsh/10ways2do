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
from src.domain_e_tabular_decisioning.run_all import run_all_approaches as run_e
from src.domain_f_cyber_threat_hunting.run_all import run_all_approaches as run_f
from src.domain_g_operations_optimization.run_all import run_all_approaches as run_g
from src.domain_h_fraud_risk_assessment.run_all import run_all_approaches as run_h
from src.domain_i_capacity_planning.run_all import run_all_approaches as run_i
from src.core.benchmark_utils import BENCHMARK_PROTOCOL_VERSION


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
        (
            "domain_e",
            run_e,
            {
                "n_samples": 600,
                "n_features": 16,
                "n_runs": 1,
                "seed": 42,
                "smoke_test": True,
            },
        ),
        (
            "domain_f",
            run_f,
            {
                "n_samples": 800,
                "n_features": 20,
                "n_runs": 1,
                "seed": 42,
                "smoke_test": True,
            },
        ),
        (
            "domain_g",
            run_g,
            {
                "n_samples": 800,
                "n_features": 16,
                "n_runs": 1,
                "seed": 42,
                "smoke_test": True,
            },
        ),
        (
            "domain_h",
            run_h,
            {
                "n_samples": 800,
                "n_features": 20,
                "n_runs": 1,
                "seed": 42,
                "smoke_test": True,
            },
        ),
        (
            "domain_i",
            run_i,
            {
                "n_samples": 800,
                "n_features": 16,
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
    assert manifest["benchmark_protocol_version"] == BENCHMARK_PROTOCOL_VERSION
    assert "git_commit_hash" in manifest
    assert manifest["config"]["smoke_test"] is True

    with open(output_dir / "results_aggregated.json", encoding="utf-8") as f:
        aggregated = json.load(f)
    assert "approaches" in aggregated
    assert isinstance(aggregated["approaches"], list)
    assert len(aggregated["approaches"]) > 0
    for row in aggregated["approaches"]:
        assert "budget_summary" in row
        budget_summary = row["budget_summary"]
        for key in [
            "train_time_cap_seconds",
            "memory_cap_mb",
            "tuning_trials_cap",
            "out_of_budget_count",
            "out_of_budget_rate",
        ]:
            assert key in budget_summary
        if row.get("success"):
            assert "significance_vs_best" in row
            significance = row["significance_vs_best"]
            for key in [
                "best_approach",
                "best_mean",
                "higher_is_better",
                "alpha",
                "is_best",
                "mean_diff_vs_best",
                "p_value",
                "cohens_d_vs_best",
                "significantly_better_than_best",
            ]:
                assert key in significance

    with open(output_dir / "results_raw_by_run.json", encoding="utf-8") as f:
        raw = json.load(f)
    assert "runs" in raw
    for run_rows in raw["runs"]:
        for row in run_rows:
            for key in ["train_time_cap_seconds", "memory_cap_mb", "tuning_trials_cap", "out_of_budget"]:
                assert key in row

    canonical = pd.read_csv(output_dir / "comparison_canonical.csv")
    assert "Category" in canonical.columns
    assert "Available" in canonical.columns
    assert len(canonical) == 10
    assert any(col.endswith("CI95 Low") for col in canonical.columns)
    assert any(col.endswith("CI95 High") for col in canonical.columns)
    assert any("p-value vs Best" in col for col in canonical.columns)

    variants = pd.read_csv(output_dir / "comparison_variants.csv")
    assert "Approach" in variants.columns
    assert "Category" in variants.columns
    assert any(col.endswith("CI95 Low") for col in variants.columns)
    assert any(col.endswith("CI95 High") for col in variants.columns)
    assert any("p-value vs Best" in col for col in variants.columns)
