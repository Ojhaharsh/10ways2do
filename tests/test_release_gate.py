import json
from pathlib import Path

import pytest

from src.core.benchmark_utils import BENCHMARK_PROTOCOL_VERSION
from src.core.release_gate import ReleaseGateError, run_release_gate


def _write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f)


def _write_csv(path: Path, header: str, row: str):
    path.write_text(f"{header}\n{row}\n", encoding="utf-8")


def _create_domain_bundle(domain_dir: Path):
    _write_json(
        domain_dir / "run_manifest.json",
        {
            "domain": domain_dir.name,
            "benchmark_protocol_version": BENCHMARK_PROTOCOL_VERSION,
            "git_commit_hash": "abc123",
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

    _write_csv(
        domain_dir / "comparison_canonical.csv",
        "Category,Available,F1 CI95 Low,F1 CI95 High,F1 p-value vs Best",
        "rule_based,True,0.1,0.9,1.0",
    )
    _write_csv(
        domain_dir / "comparison_variants.csv",
        "Approach,Category,F1 CI95 Low,F1 CI95 High,F1 p-value vs Best",
        "Dummy,rule_based,0.1,0.9,1.0",
    )


def _create_report(path: Path):
    path.write_text(
        "\n".join(
            [
                "# ML Philosophy Benchmark: Complete Report",
                "# Information Extraction: Benchmark Report",
                "## Statistical Significance",
                "# Anomaly Detection: Benchmark Report",
                "## Statistical Significance",
                "# Recommendation: Benchmark Report",
                "## Statistical Significance",
                "# Time Series Forecasting: Benchmark Report",
                "## Statistical Significance",
                "# Tabular Decisioning: Benchmark Report",
                "## Statistical Significance",
                "## Cross-Domain Statistical Summary",
                "## Cross-Domain Pareto Frontier",
            ]
        ),
        encoding="utf-8",
    )


def _create_frontier(path: Path):
    domains = ["domain_a", "domain_b", "domain_c", "domain_d", "domain_e"]
    path.write_text(
        json.dumps(
            {
                "domains": [
                    {
                        "domain": domain,
                        "champion": {"name": "Dummy", "extraordinary_index": 1.0},
                        "pareto_frontier": [{"name": "Dummy", "extraordinary_index": 1.0}],
                    }
                    for domain in domains
                ],
                "cross_domain_generalists": [
                    {"name": "Dummy", "avg_extraordinary_index": 1.0, "domains_covered": 1}
                ],
            }
        ),
        encoding="utf-8",
    )


def test_release_gate_passes_on_valid_bundle(tmp_path):
    results_dir = tmp_path / "results"
    for name in ["domain_a", "domain_b", "domain_c", "domain_d", "domain_e"]:
        _create_domain_bundle(results_dir / name)
    _create_report(results_dir / "REPORT.md")
    _create_frontier(results_dir / "CROSS_DOMAIN_FRONTIER.json")

    run_release_gate(results_dir=results_dir)


def test_release_gate_fails_when_report_missing_section(tmp_path):
    results_dir = tmp_path / "results"
    for name in ["domain_a", "domain_b", "domain_c", "domain_d", "domain_e"]:
        _create_domain_bundle(results_dir / name)
    (results_dir / "REPORT.md").write_text("# Incomplete report", encoding="utf-8")
    _create_frontier(results_dir / "CROSS_DOMAIN_FRONTIER.json")

    with pytest.raises(ReleaseGateError):
        run_release_gate(results_dir=results_dir)


def test_release_gate_fails_when_frontier_semantics_invalid(tmp_path):
    results_dir = tmp_path / "results"
    for name in ["domain_a", "domain_b", "domain_c", "domain_d", "domain_e"]:
        _create_domain_bundle(results_dir / name)
    _create_report(results_dir / "REPORT.md")

    # Missing domain coverage and empty Pareto list should fail semantic frontier checks.
    (results_dir / "CROSS_DOMAIN_FRONTIER.json").write_text(
        json.dumps(
            {
                "domains": [
                    {
                        "domain": "domain_a",
                        "champion": {"name": "Dummy", "extraordinary_index": 1.2},
                        "pareto_frontier": [],
                    }
                ],
                "cross_domain_generalists": [],
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ReleaseGateError):
        run_release_gate(results_dir=results_dir)
