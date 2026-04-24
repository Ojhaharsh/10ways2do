import json

from src.analysis.benchmark_card import BenchmarkCardGenerator


def _write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _seed_manifest(results_dir, domain, protocol="1.0.0", commit="abc123"):
    _write_json(
        results_dir / domain / "run_manifest.json",
        {
            "domain": domain,
            "benchmark_protocol_version": protocol,
            "git_commit_hash": commit,
        },
    )


def test_benchmark_card_generator_writes_expected_artifacts(tmp_path):
    results_dir = tmp_path / "results"

    frontier_payload = {
        "weights": {"quality": 0.45, "speed": 0.25, "resilience": 0.20, "consistency": 0.10},
        "domains": [
            {
                "domain": "domain_a",
                "domain_name": "Information Extraction",
                "champion": {"name": "Rule-Based IE", "extraordinary_index": 0.91},
                "pareto_frontier": [{"name": "Rule-Based IE", "extraordinary_index": 0.91}],
            },
            {
                "domain": "domain_b",
                "domain_name": "Anomaly Detection",
                "champion": {"name": "Statistical", "extraordinary_index": 0.88},
                "pareto_frontier": [{"name": "Statistical", "extraordinary_index": 0.88}],
            },
        ],
        "cross_domain_generalists": [
            {"name": "Rule-Based IE", "avg_extraordinary_index": 0.91, "domains_covered": 1},
            {"name": "Statistical", "avg_extraordinary_index": 0.88, "domains_covered": 1},
        ],
    }
    _write_json(results_dir / "CROSS_DOMAIN_FRONTIER.json", frontier_payload)

    _seed_manifest(results_dir, "domain_a")
    _seed_manifest(results_dir, "domain_b")

    output_dir = tmp_path / "release_bundle"
    generator = BenchmarkCardGenerator(results_dir=str(results_dir))
    outputs = generator.save(output_dir=str(output_dir))

    assert outputs["json"].endswith("BENCHMARK_CARD.json")
    assert outputs["markdown"].endswith("BENCHMARK_CARD.md")

    card_json_path = output_dir / "BENCHMARK_CARD.json"
    card_md_path = output_dir / "BENCHMARK_CARD.md"

    assert card_json_path.exists()
    assert card_md_path.exists()

    payload = json.loads(card_json_path.read_text(encoding="utf-8"))
    assert payload["domain_coverage"]["observed"] == 2
    assert "domain_c" in payload["domain_coverage"]["missing_in_frontier"]
    assert payload["protocol_versions"]["1.0.0"] == 2
    assert len(payload["champions"]) == 2

    markdown = card_md_path.read_text(encoding="utf-8")
    assert "# Benchmark Card" in markdown
    assert "## Domain Champions" in markdown
