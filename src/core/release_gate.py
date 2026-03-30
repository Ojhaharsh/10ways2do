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

    for header in EXPECTED_DOMAIN_HEADERS:
        if header not in report_text:
            errors.append(f"{report_path}: missing domain section '{header}'")

    significance_count = report_text.count("## Statistical Significance")
    if significance_count < len(EXPECTED_DOMAIN_HEADERS):
        errors.append(
            f"{report_path}: expected at least {len(EXPECTED_DOMAIN_HEADERS)} 'Statistical Significance' sections, found {significance_count}"
        )



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

    if errors:
        preview = "\n".join(f"- {err}" for err in errors[:25])
        if len(errors) > 25:
            preview += f"\n- ... and {len(errors) - 25} more"
        raise ReleaseGateError(f"Release gate failed with {len(errors)} issue(s):\n{preview}")
