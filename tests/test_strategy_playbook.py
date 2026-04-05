import json
from pathlib import Path

from src.analysis.strategy_playbook import StrategyPlaybookGenerator


def _write_frontier(results_dir: Path):
    payload = {
        "generated_at_utc": "2026-04-05T00:00:00+00:00",
        "domains": [
            {
                "domain": "domain_a",
                "domain_name": "Information Extraction",
                "champion": {"name": "A1", "extraordinary_index": 0.82},
                "pareto_frontier": [
                    {
                        "name": "A1",
                        "extraordinary_index": 0.82,
                        "quality_score": 0.9,
                        "speed_score": 0.7,
                        "resilience": 0.8,
                        "consistency": 1.0,
                    },
                    {
                        "name": "A2",
                        "extraordinary_index": 0.79,
                        "quality_score": 0.85,
                        "speed_score": 0.9,
                        "resilience": 0.75,
                        "consistency": 1.0,
                    },
                ],
            },
            {
                "domain": "domain_b",
                "domain_name": "Anomaly Detection",
                "champion": {"name": "B1", "extraordinary_index": 0.8},
                "pareto_frontier": [
                    {
                        "name": "B1",
                        "extraordinary_index": 0.8,
                        "quality_score": 0.88,
                        "speed_score": 0.65,
                        "resilience": 0.85,
                        "consistency": 0.95,
                    }
                ],
            },
        ],
        "cross_domain_generalists": [
            {"name": "A1", "avg_extraordinary_index": 0.81, "domains_covered": 2}
        ],
    }
    (results_dir / "CROSS_DOMAIN_FRONTIER.json").write_text(json.dumps(payload), encoding="utf-8")


def test_strategy_playbook_generator_writes_json_and_markdown(tmp_path: Path):
    _write_frontier(tmp_path)

    outputs = StrategyPlaybookGenerator(results_dir=str(tmp_path)).save()

    assert outputs["json"].exists()
    assert outputs["markdown"].exists()

    payload = json.loads(outputs["json"].read_text(encoding="utf-8"))
    scenarios = payload["scenarios"]

    assert "balanced_production" in scenarios
    assert "accuracy_first" in scenarios
    assert "latency_first" in scenarios
    assert "reliability_first" in scenarios

    balanced = scenarios["balanced_production"]["recommendations"]
    assert len(balanced) == 2
    assert balanced[0]["recommended_approach"] in {"A1", "A2"}

    markdown = outputs["markdown"].read_text(encoding="utf-8")
    assert "# Strategy Playbook" in markdown
    assert "## Scenario: balanced_production" in markdown


def test_strategy_playbook_prefers_speed_in_latency_scenario(tmp_path: Path):
    _write_frontier(tmp_path)

    outputs = StrategyPlaybookGenerator(results_dir=str(tmp_path)).save()
    payload = json.loads(outputs["json"].read_text(encoding="utf-8"))

    latency_recs = payload["scenarios"]["latency_first"]["recommendations"]
    domain_a = [r for r in latency_recs if r["domain"] == "domain_a"][0]
    assert domain_a["recommended_approach"] == "A2"
