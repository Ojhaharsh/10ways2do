import json
from pathlib import Path

from src.analysis.policy_simulator import PolicySimulator


def _write_frontier(results_dir: Path):
    payload = {
        "generated_at_utc": "2026-04-05T00:00:00+00:00",
        "domains": [
            {
                "domain": "domain_a",
                "domain_name": "Information Extraction",
                "champion": {"name": "AccurateA", "extraordinary_index": 0.8},
                "pareto_frontier": [
                    {
                        "name": "AccurateA",
                        "quality_score": 0.95,
                        "speed_score": 0.30,
                        "resilience": 0.8,
                        "consistency": 1.0,
                    },
                    {
                        "name": "FastA",
                        "quality_score": 0.75,
                        "speed_score": 0.95,
                        "resilience": 0.7,
                        "consistency": 0.95,
                    },
                ],
            }
        ],
    }
    (results_dir / "CROSS_DOMAIN_FRONTIER.json").write_text(json.dumps(payload), encoding="utf-8")


def test_policy_simulator_prefers_fast_model_when_speed_weight_high(tmp_path: Path):
    _write_frontier(tmp_path)

    outputs = PolicySimulator(results_dir=str(tmp_path)).save(
        weights={
            "quality_score": 0.1,
            "speed_score": 0.8,
            "resilience": 0.05,
            "consistency": 0.05,
        },
        mins={
            "quality_score": 0.0,
            "speed_score": 0.0,
            "resilience": 0.0,
            "consistency": 0.0,
        },
        policy_name="latency_ops",
        top_k=2,
    )

    payload = json.loads(outputs["json"].read_text(encoding="utf-8"))
    selected = payload["domains"][0]["selected"]
    assert selected["name"] == "FastA"


def test_policy_simulator_falls_back_when_constraints_too_strict(tmp_path: Path):
    _write_frontier(tmp_path)

    outputs = PolicySimulator(results_dir=str(tmp_path)).save(
        weights={
            "quality_score": 0.5,
            "speed_score": 0.2,
            "resilience": 0.2,
            "consistency": 0.1,
        },
        mins={
            "quality_score": 0.99,
            "speed_score": 0.99,
            "resilience": 0.99,
            "consistency": 0.99,
        },
        policy_name="strict_policy",
        top_k=2,
    )

    payload = json.loads(outputs["json"].read_text(encoding="utf-8"))
    domain_row = payload["domains"][0]
    assert domain_row["constraints_satisfied"] is False
    assert "No feasible candidate" in domain_row["reason"]


def test_policy_optimizer_writes_outputs_and_best_policy(tmp_path: Path):
    _write_frontier(tmp_path)

    outputs = PolicySimulator(results_dir=str(tmp_path)).save_optimization(
        mins={
            "quality_score": 0.0,
            "speed_score": 0.0,
            "resilience": 0.0,
            "consistency": 0.0,
        },
        policy_name="search_balanced",
        objective="balanced",
        weight_step=0.5,
        top_k=2,
        top_n=3,
    )

    assert outputs["json"].exists()
    assert outputs["markdown"].exists()

    payload = json.loads(outputs["json"].read_text(encoding="utf-8"))
    best = payload["best_policy"]
    weights = best["weights"]
    assert abs(sum(weights.values()) - 1.0) < 1e-8
    assert payload["objective"] == "balanced"
    assert len(payload["top_policies"]) >= 1


def test_policy_optimizer_respects_max_configs(tmp_path: Path):
    _write_frontier(tmp_path)

    outputs = PolicySimulator(results_dir=str(tmp_path)).save_optimization(
        policy_name="limited_search",
        objective="max_coverage",
        weight_step=0.25,
        max_configs=4,
        top_n=2,
    )

    payload = json.loads(outputs["json"].read_text(encoding="utf-8"))
    assert payload["search_space_size"] == 4


def test_policy_frontier_optimizer_writes_outputs_and_stability_bands(tmp_path: Path):
    _write_frontier(tmp_path)

    outputs = PolicySimulator(results_dir=str(tmp_path)).save_frontier_optimization(
        policy_name="frontier_search",
        weight_step=0.5,
        top_k=2,
        top_n=5,
    )

    assert outputs["json"].exists()
    assert outputs["markdown"].exists()

    payload = json.loads(outputs["json"].read_text(encoding="utf-8"))
    assert payload["mode"] == "multi_objective_pareto"
    assert payload["frontier_size"] >= 1
    assert len(payload["frontier_policies"]) >= 1

    bands = payload["stability_bands"]
    for metric in ["coverage_rate", "avg_selected_policy_score", "worst_selected_policy_score"]:
        assert metric in bands
        assert set(bands[metric].keys()) == {"p10", "p50", "p90"}


def test_policy_frontier_optimizer_respects_max_configs(tmp_path: Path):
    _write_frontier(tmp_path)

    outputs = PolicySimulator(results_dir=str(tmp_path)).save_frontier_optimization(
        policy_name="frontier_limited",
        weight_step=0.25,
        max_configs=3,
        top_n=3,
    )

    payload = json.loads(outputs["json"].read_text(encoding="utf-8"))
    assert payload["search_space_size"] == 3
