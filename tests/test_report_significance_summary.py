"""Tests for cross-domain significance summary in reports."""

import json
from pathlib import Path

from src.analysis.report_generator import ReportGenerator


def _write_domain_aggregated(base_dir: Path, domain: str, approach_name: str, best_mean: float) -> None:
    domain_dir = base_dir / domain
    domain_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "approaches": [
            {
                "name": approach_name,
                "category": "hybrid",
                "success_rate": 1.0,
                "budget_summary": {
                    "train_time_cap_seconds": 30.0,
                    "memory_cap_mb": None,
                    "tuning_trials_cap": 0,
                    "out_of_budget_count": 0,
                    "out_of_budget_rate": 0.0,
                },
                "significance_vs_best": {
                    "best_approach": approach_name,
                    "best_mean": best_mean,
                    "higher_is_better": True,
                    "alpha": 0.05,
                    "is_best": True,
                    "mean_diff_vs_best": 0.0,
                    "p_value": 1.0,
                    "significantly_better_than_best": False,
                },
                "success": True,
            },
            {
                "name": "RunnerUp",
                "category": "classical_ml",
                "success_rate": 1.0,
                "budget_summary": {
                    "train_time_cap_seconds": 30.0,
                    "memory_cap_mb": None,
                    "tuning_trials_cap": 0,
                    "out_of_budget_count": 0,
                    "out_of_budget_rate": 0.0,
                },
                "significance_vs_best": {
                    "best_approach": approach_name,
                    "best_mean": best_mean,
                    "higher_is_better": True,
                    "alpha": 0.05,
                    "is_best": False,
                    "mean_diff_vs_best": -0.1,
                    "p_value": 0.01,
                    "significantly_better_than_best": False,
                },
                "success": True,
            },
        ]
    }
    (domain_dir / "results_aggregated.json").write_text(json.dumps(payload), encoding="utf-8")


def test_full_report_includes_cross_domain_statistical_summary(tmp_path: Path) -> None:
    _write_domain_aggregated(tmp_path, "domain_a", "AlphaModel", 0.91)
    _write_domain_aggregated(tmp_path, "domain_b", "BetaModel", 0.82)

    report = ReportGenerator(results_dir=str(tmp_path)).generate_full_report()

    assert "## Cross-Domain Statistical Summary" in report
    assert "Information Extraction: best=AlphaModel" in report
    assert "Anomaly Detection: best=BetaModel" in report
    assert "Most frequent domain winner" in report


def test_save_report_persists_statistical_summary(tmp_path: Path) -> None:
    _write_domain_aggregated(tmp_path, "domain_a", "AlphaModel", 0.91)

    generator = ReportGenerator(results_dir=str(tmp_path))
    output_path = tmp_path / "REPORT.md"
    generator.save_report(str(output_path))

    saved = output_path.read_text(encoding="utf-8")
    assert "## Cross-Domain Statistical Summary" in saved
    assert "Information Extraction: best=AlphaModel" in saved
