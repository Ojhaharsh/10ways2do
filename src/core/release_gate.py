"""Release-gate checks for benchmark readiness."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, List

from .artifact_validator import DEFAULT_DOMAIN_DIRS, ArtifactValidationError, validate_results_tree
from .benchmark_utils import BENCHMARK_PROTOCOL_VERSION


class ReleaseGateError(ValueError):
    """Raised when release-gate checks fail."""


EXPECTED_DOMAIN_HEADERS = [
    "# Information Extraction: Benchmark Report",
    "# Anomaly Detection: Benchmark Report",
    "# Recommendation: Benchmark Report",
    "# Time Series Forecasting: Benchmark Report",
    "# Tabular Decisioning: Benchmark Report",
    "# Cyber Threat Hunting: Benchmark Report",
    "# Operations Optimization: Benchmark Report",
    "# Fraud Risk Assessment: Benchmark Report",
    "# Capacity Planning: Benchmark Report",
    "# Model Risk Monitoring: Benchmark Report",
    "# Infrastructure Cost Forecasting: Benchmark Report",
]



def _load_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)



def _validate_manifest_protocol(results_dir: Path, domain_dirs: Iterable[str], errors: List[str]) -> None:
    for domain_name in domain_dirs:
        manifest_path = results_dir / domain_name / "run_manifest.json"
        if not manifest_path.exists():
            errors.append(f"{manifest_path}: missing run manifest")
            continue

        try:
            manifest = _load_json(manifest_path)
        except Exception as exc:
            errors.append(f"{manifest_path}: cannot parse JSON ({exc})")
            continue

        protocol_version = manifest.get("benchmark_protocol_version")
        if protocol_version != BENCHMARK_PROTOCOL_VERSION:
            errors.append(
                f"{manifest_path}: protocol version '{protocol_version}' does not match expected '{BENCHMARK_PROTOCOL_VERSION}'"
            )

        if "git_commit_hash" not in manifest:
            errors.append(f"{manifest_path}: missing 'git_commit_hash' field")



def _validate_report(report_path: Path, errors: List[str]) -> None:
    if not report_path.exists():
        errors.append(f"{report_path}: report not found")
        return

    try:
        report_text = report_path.read_text(encoding="utf-8")
    except Exception as exc:
        errors.append(f"{report_path}: cannot read report ({exc})")
        return

    if "## Cross-Domain Statistical Summary" not in report_text:
        errors.append(f"{report_path}: missing 'Cross-Domain Statistical Summary' section")
    if "## Cross-Domain Pareto Frontier" not in report_text:
        errors.append(f"{report_path}: missing 'Cross-Domain Pareto Frontier' section")

    for header in EXPECTED_DOMAIN_HEADERS:
        if header not in report_text:
            errors.append(f"{report_path}: missing domain section '{header}'")

    significance_count = report_text.count("## Statistical Significance")
    if significance_count < len(EXPECTED_DOMAIN_HEADERS):
        errors.append(
            f"{report_path}: expected at least {len(EXPECTED_DOMAIN_HEADERS)} 'Statistical Significance' sections, found {significance_count}"
        )


def _validate_frontier_artifact(path: Path, errors: List[str]) -> None:
    if not path.exists():
        errors.append(f"{path}: missing cross-domain frontier artifact")
        return

    try:
        payload = _load_json(path)
    except Exception as exc:
        errors.append(f"{path}: cannot parse JSON ({exc})")
        return

    if not isinstance(payload.get("domains"), list):
        errors.append(f"{path}: missing 'domains' list")
        return

    if not isinstance(payload.get("cross_domain_generalists"), list):
        errors.append(f"{path}: missing 'cross_domain_generalists' list")
        return

    domains = payload.get("domains", [])
    if not domains:
        errors.append(f"{path}: 'domains' list must be non-empty")
        return

    domain_map = {}
    for idx, row in enumerate(domains):
        if not isinstance(row, dict):
            errors.append(f"{path}: domains[{idx}] is not an object")
            continue

        domain_name = row.get("domain")
        if not isinstance(domain_name, str) or not domain_name.strip():
            errors.append(f"{path}: domains[{idx}] missing non-empty 'domain'")
            continue

        domain_map[domain_name] = row

    missing_domains = [d for d in DEFAULT_DOMAIN_DIRS if d not in domain_map]
    if missing_domains:
        errors.append(f"{path}: missing expected domains {missing_domains}")

    for domain_name, row in domain_map.items():
        champion = row.get("champion")
        if not isinstance(champion, dict):
            errors.append(f"{path}: domain '{domain_name}' missing champion object")
            continue

        champion_name = champion.get("name")
        champion_index = champion.get("extraordinary_index")
        if not isinstance(champion_name, str) or not champion_name.strip():
            errors.append(f"{path}: domain '{domain_name}' champion missing non-empty name")
        if not isinstance(champion_index, (int, float)):
            errors.append(f"{path}: domain '{domain_name}' champion missing numeric extraordinary_index")
        elif not (0.0 <= float(champion_index) <= 1.0):
            errors.append(
                f"{path}: domain '{domain_name}' champion extraordinary_index out of range [0,1]: {champion_index}"
            )

        pareto_frontier = row.get("pareto_frontier")
        if not isinstance(pareto_frontier, list) or not pareto_frontier:
            errors.append(f"{path}: domain '{domain_name}' pareto_frontier must be non-empty list")
            continue

        for p_idx, candidate in enumerate(pareto_frontier):
            if not isinstance(candidate, dict):
                errors.append(f"{path}: domain '{domain_name}' pareto_frontier[{p_idx}] is not an object")
                continue

            name = candidate.get("name")
            score = candidate.get("extraordinary_index")
            if not isinstance(name, str) or not name.strip():
                errors.append(
                    f"{path}: domain '{domain_name}' pareto_frontier[{p_idx}] missing non-empty name"
                )
            if not isinstance(score, (int, float)):
                errors.append(
                    f"{path}: domain '{domain_name}' pareto_frontier[{p_idx}] missing numeric extraordinary_index"
                )
            elif not (0.0 <= float(score) <= 1.0):
                errors.append(
                    f"{path}: domain '{domain_name}' pareto_frontier[{p_idx}] extraordinary_index out of range [0,1]: {score}"
                )

    generalists = payload.get("cross_domain_generalists", [])
    if not generalists:
        errors.append(f"{path}: 'cross_domain_generalists' must be non-empty")


def _validate_strategy_playbook(json_path: Path, md_path: Path, errors: List[str]) -> None:
    if not json_path.exists():
        errors.append(f"{json_path}: missing strategy playbook artifact")
        return

    if not md_path.exists():
        errors.append(f"{md_path}: missing strategy playbook markdown")
        return

    try:
        payload = _load_json(json_path)
    except Exception as exc:
        errors.append(f"{json_path}: cannot parse JSON ({exc})")
        return

    scenarios = payload.get("scenarios")
    if not isinstance(scenarios, dict) or not scenarios:
        errors.append(f"{json_path}: missing non-empty 'scenarios' object")
        return

    required_scenarios = [
        "balanced_production",
        "accuracy_first",
        "latency_first",
        "reliability_first",
    ]
    for scenario_name in required_scenarios:
        if scenario_name not in scenarios:
            errors.append(f"{json_path}: missing scenario '{scenario_name}'")
            continue

        scenario_payload = scenarios[scenario_name]
        if not isinstance(scenario_payload, dict):
            errors.append(f"{json_path}: scenario '{scenario_name}' must be an object")
            continue

        recommendations = scenario_payload.get("recommendations")
        if not isinstance(recommendations, list) or not recommendations:
            errors.append(f"{json_path}: scenario '{scenario_name}' has no recommendations")

    try:
        markdown = md_path.read_text(encoding="utf-8")
    except Exception as exc:
        errors.append(f"{md_path}: cannot read markdown ({exc})")
        return

    if "# Strategy Playbook" not in markdown:
        errors.append(f"{md_path}: missing Strategy Playbook title")



def run_release_gate(results_dir: str | Path = "results", require_report: bool = True) -> None:
    """Run release-gate checks and raise on failure."""
    root = Path(results_dir)
    if not root.exists():
        raise ReleaseGateError(f"Results directory not found: {root}")

    errors: List[str] = []

    try:
        validate_results_tree(results_dir=root, domain_dirs=DEFAULT_DOMAIN_DIRS)
    except ArtifactValidationError as exc:
        errors.append(str(exc))

    _validate_manifest_protocol(root, DEFAULT_DOMAIN_DIRS, errors)

    if require_report:
        _validate_report(root / "REPORT.md", errors)
        _validate_frontier_artifact(root / "CROSS_DOMAIN_FRONTIER.json", errors)
        _validate_strategy_playbook(
            root / "STRATEGY_PLAYBOOK.json",
            root / "STRATEGY_PLAYBOOK.md",
            errors,
        )

    if errors:
        preview = "\n".join(f"- {err}" for err in errors[:25])
        if len(errors) > 25:
            preview += f"\n- ... and {len(errors) - 25} more"
        raise ReleaseGateError(f"Release gate failed with {len(errors)} issue(s):\n{preview}")
