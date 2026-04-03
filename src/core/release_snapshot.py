"""Versioned release snapshot generation utilities."""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from .artifact_validator import DEFAULT_DOMAIN_DIRS, REQUIRED_FILES
from .benchmark_utils import BENCHMARK_PROTOCOL_VERSION
from .release_gate import run_release_gate


class ReleaseSnapshotError(ValueError):
    """Raised when a release snapshot cannot be generated."""


DOMAIN_LABELS = {
    "domain_a": "Information Extraction",
    "domain_b": "Anomaly Detection",
    "domain_c": "Recommendation",
    "domain_d": "Time Series Forecasting",
    "domain_e": "Tabular Decisioning",
}


LOWER_IS_BETTER_TOKENS = ["mae", "rmse", "mape", "mase", "loss", "error", "latency", "time"]


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _is_lower_better(metric_column_name: str) -> bool:
    key = metric_column_name.lower().replace(" mean", "")
    return any(token in key for token in LOWER_IS_BETTER_TOKENS)


def _pick_primary_metric_column(columns: List[str]) -> str:
    candidates = [c for c in columns if c.endswith(" Mean")]
    if not candidates:
        raise ReleaseSnapshotError("No primary metric mean column found in canonical comparison")

    preferred = [
        "NDCG@10 Mean",
        "F1 Mean",
        "Overall Exact Match Mean",
        "MAE Mean",
    ]
    for name in preferred:
        if name in candidates:
            return name

    return candidates[0]


def _domain_summary(domain_dir: Path) -> Dict[str, Any]:
    manifest = _load_json(domain_dir / "run_manifest.json")
    canonical = pd.read_csv(domain_dir / "comparison_canonical.csv")

    metric_col = _pick_primary_metric_column(list(canonical.columns))
    available = canonical[canonical.get("Available", True) == True]  # noqa: E712
    if available.empty:
        raise ReleaseSnapshotError(f"No available canonical approaches found in {domain_dir}")

    numeric_metric = pd.to_numeric(available[metric_col], errors="coerce")
    available = available.assign(_metric=numeric_metric).dropna(subset=["_metric"])
    if available.empty:
        raise ReleaseSnapshotError(f"No numeric primary metric values found in {domain_dir}")

    lower_is_better = _is_lower_better(metric_col)
    best_row = available.sort_values("_metric", ascending=lower_is_better).iloc[0]

    return {
        "domain": domain_dir.name,
        "domain_label": DOMAIN_LABELS.get(domain_dir.name, domain_dir.name),
        "seed_count": len(manifest.get("seeds", [])),
        "best_approach": str(best_row["Approach"]),
        "primary_metric": metric_col.replace(" Mean", ""),
        "primary_metric_mean": float(best_row["_metric"]),
        "higher_is_better": not lower_is_better,
        "run_success_rate": float(best_row.get("Run Success Rate", 0.0)),
        "manifest": {
            "benchmark_protocol_version": manifest.get("benchmark_protocol_version"),
            "git_commit_hash": manifest.get("git_commit_hash"),
            "generated_at_utc": manifest.get("generated_at_utc"),
        },
    }


def _build_snapshot_markdown(tag: str, snapshot: Dict[str, Any]) -> str:
    lines = []
    lines.append(f"# Release Snapshot: {tag}")
    lines.append("")
    lines.append(f"Generated UTC: {snapshot['generated_at_utc']}")
    lines.append(f"Protocol Version: {snapshot['benchmark_protocol_version']}")
    lines.append("")
    lines.append("## Release Gate Status")
    lines.append("")
    lines.append("- Artifact validation: PASS")
    lines.append("- Release gate checks: PASS")
    lines.append("")
    lines.append("## Domain Summary")
    lines.append("")
    lines.append("| Domain | Best Approach | Primary Metric | Mean | Success Rate |")
    lines.append("|--------|---------------|----------------|------|--------------|")

    for row in snapshot["domain_summaries"]:
        lines.append(
            f"| {row['domain_label']} | {row['best_approach']} | {row['primary_metric']} | {row['primary_metric_mean']:.4f} | {row['run_success_rate']:.2f} |"
        )

    lines.append("")
    lines.append("## Artifacts")
    lines.append("")
    lines.append("- snapshot.json")
    lines.append("- SNAPSHOT.md")
    lines.append("")
    return "\n".join(lines)


def create_release_snapshot(
    tag: str,
    results_dir: str | Path = "results",
    snapshots_root: str | Path = "releases",
) -> Path:
    """Create a versioned release snapshot after running release-gate checks."""
    clean_tag = str(tag).strip()
    if not clean_tag:
        raise ReleaseSnapshotError("Snapshot tag must be a non-empty string")

    results_path = Path(results_dir)
    snapshots_path = Path(snapshots_root)
    output_dir = snapshots_path / clean_tag

    run_release_gate(results_dir=results_path, require_report=True)

    domain_summaries = []
    domain_artifacts: Dict[str, Dict[str, Any]] = {}
    for domain_name in DEFAULT_DOMAIN_DIRS:
        domain_path = results_path / domain_name
        domain_summaries.append(_domain_summary(domain_path))

        copied_files: Dict[str, str] = {}
        target_domain_dir = output_dir / domain_name
        target_domain_dir.mkdir(parents=True, exist_ok=True)

        for filename in REQUIRED_FILES:
            source = domain_path / filename
            if not source.exists():
                continue
            destination = target_domain_dir / filename
            shutil.copy2(source, destination)
            copied_files[filename] = filename

        domain_artifacts[domain_name] = {
            "artifacts": copied_files,
        }

    report_artifacts: Dict[str, str] = {}
    report_path = results_path / "REPORT.md"
    if report_path.exists():
        destination = output_dir / "REPORT.md"
        shutil.copy2(report_path, destination)
        report_artifacts["report"] = "REPORT.md"

    frontier_path = results_path / "CROSS_DOMAIN_FRONTIER.json"
    if frontier_path.exists():
        destination = output_dir / "CROSS_DOMAIN_FRONTIER.json"
        shutil.copy2(frontier_path, destination)
        report_artifacts["cross_domain_frontier"] = "CROSS_DOMAIN_FRONTIER.json"

    snapshot = {
        "snapshot_tag": clean_tag,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_version": BENCHMARK_PROTOCOL_VERSION,
        "benchmark_protocol_version": BENCHMARK_PROTOCOL_VERSION,
        "results_dir": str(results_path),
        "domain_summaries": domain_summaries,
        "domains": domain_artifacts,
        "report_artifacts": report_artifacts,
    }

    output_dir.mkdir(parents=True, exist_ok=True)

    with (output_dir / "snapshot.json").open("w", encoding="utf-8") as f:
        json.dump(snapshot, f, indent=2)

    markdown = _build_snapshot_markdown(clean_tag, snapshot)
    (output_dir / "SNAPSHOT.md").write_text(markdown, encoding="utf-8")

    return output_dir
