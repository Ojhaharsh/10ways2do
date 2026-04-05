import json
from pathlib import Path

from src.core.benchmark_utils import BENCHMARK_PROTOCOL_VERSION
from src.core.release_snapshot import create_release_snapshot


def _write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f)


def _write_csv(path: Path, header: str, rows: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{header}\n{rows}\n", encoding="utf-8")


def _seed_domain_artifacts(results_dir: Path, domain_name: str, metric_col: str, best_approach: str):
    domain_dir = results_dir / domain_name
    _write_json(
        domain_dir / "run_manifest.json",
        {
            "domain": domain_name,
            "benchmark_protocol_version": BENCHMARK_PROTOCOL_VERSION,
            "git_commit_hash": "deadbeef",
            "generated_at_utc": "2026-03-31T00:00:00+00:00",
            "seeds": [42, 43, 44],
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
                    "name": best_approach,
                    "category": "rule_based",
                    "success": True,
                    "budget_summary": {
                        "train_time_cap_seconds": 30.0,
                        "memory_cap_mb": 4096.0,
                        "tuning_trials_cap": 0,
                        "out_of_budget_count": 0,
                        "out_of_budget_rate": 0.0,
                    },
                    "significance_vs_best": {"best_approach": best_approach},
                }
            ]
        },
    )

    _write_csv(
        domain_dir / "comparison_canonical.csv",
        f"Category,Approach,{metric_col},Run Success Rate,Available,F1 CI95 Low,F1 CI95 High,F1 p-value vs Best",
        f"rule_based,{best_approach},0.8,1.0,True,0.7,0.9,1.0",
    )
    _write_csv(
        domain_dir / "comparison_variants.csv",
        "Approach,Category,F1 Mean,F1 CI95 Low,F1 CI95 High,F1 p-value vs Best",
        f"{best_approach},rule_based,0.8,0.7,0.9,1.0",
    )


def _seed_report(results_dir: Path):
    report = "\n".join(
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
    )
    (results_dir / "REPORT.md").write_text(report, encoding="utf-8")


def _seed_frontier(results_dir: Path):
    payload = {
        "domains": [
            {
                "domain": domain,
                "champion": {"name": "Rule-Based IE", "extraordinary_index": 0.88},
                "pareto_frontier": [{"name": "Rule-Based IE", "extraordinary_index": 0.88}],
            }
            for domain in ["domain_a", "domain_b", "domain_c", "domain_d", "domain_e"]
        ],
        "cross_domain_generalists": [
            {"name": "Rule-Based IE", "avg_extraordinary_index": 0.88, "domains_covered": 1}
        ],
    }
    (results_dir / "CROSS_DOMAIN_FRONTIER.json").write_text(json.dumps(payload), encoding="utf-8")


def _seed_strategy_playbook(results_dir: Path):
    playbook = {
        "scenarios": {
            "balanced_production": {
                "recommendations": [{"domain": "domain_a", "recommended_approach": "Rule-Based IE"}]
            },
            "accuracy_first": {
                "recommendations": [{"domain": "domain_a", "recommended_approach": "Rule-Based IE"}]
            },
            "latency_first": {
                "recommendations": [{"domain": "domain_a", "recommended_approach": "Rule-Based IE"}]
            },
            "reliability_first": {
                "recommendations": [{"domain": "domain_a", "recommended_approach": "Rule-Based IE"}]
            },
        }
    }
    (results_dir / "STRATEGY_PLAYBOOK.json").write_text(json.dumps(playbook), encoding="utf-8")
    (results_dir / "STRATEGY_PLAYBOOK.md").write_text("# Strategy Playbook\n", encoding="utf-8")


def test_create_release_snapshot_creates_expected_files(tmp_path):
    results_dir = tmp_path / "results"
    snapshots_dir = tmp_path / "releases"

    _seed_domain_artifacts(results_dir, "domain_a", "Overall Exact Match Mean", "Rule-Based IE")
    _seed_domain_artifacts(results_dir, "domain_b", "F1 Mean", "Statistical")
    _seed_domain_artifacts(results_dir, "domain_c", "NDCG@10 Mean", "Popularity")
    _seed_domain_artifacts(results_dir, "domain_d", "MAE Mean", "Exp Smoothing")
    _seed_domain_artifacts(results_dir, "domain_e", "F1 Mean", "Linear")
    _seed_report(results_dir)
    _seed_frontier(results_dir)
    _seed_strategy_playbook(results_dir)

    out = create_release_snapshot("v1.1-test", results_dir=results_dir, snapshots_root=snapshots_dir)

    assert out.exists()
    assert (out / "snapshot.json").exists()
    assert (out / "SNAPSHOT.md").exists()

    payload = json.loads((out / "snapshot.json").read_text(encoding="utf-8"))
    assert payload["snapshot_tag"] == "v1.1-test"
    assert payload["benchmark_protocol_version"] == BENCHMARK_PROTOCOL_VERSION
    assert payload["protocol_version"] == BENCHMARK_PROTOCOL_VERSION
    assert len(payload["domain_summaries"]) == 5
    assert len(payload["domains"]) == 5
    assert "domain_a" in payload["domains"]
    assert "comparison_canonical.csv" in payload["domains"]["domain_a"]["artifacts"]
    assert payload["report_artifacts"]["report"] == "REPORT.md"
    assert payload["report_artifacts"]["cross_domain_frontier"] == "CROSS_DOMAIN_FRONTIER.json"
    assert payload["report_artifacts"]["strategy_playbook_json"] == "STRATEGY_PLAYBOOK.json"
    assert payload["report_artifacts"]["strategy_playbook_markdown"] == "STRATEGY_PLAYBOOK.md"
    assert (out / "REPORT.md").exists()
    assert (out / "CROSS_DOMAIN_FRONTIER.json").exists()
    assert (out / "STRATEGY_PLAYBOOK.json").exists()
    assert (out / "STRATEGY_PLAYBOOK.md").exists()
