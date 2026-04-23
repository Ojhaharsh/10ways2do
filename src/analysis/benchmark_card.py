"""Benchmark card generation for release-facing consumption."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

from ..core.artifact_validator import DEFAULT_DOMAIN_DIRS


class BenchmarkCardError(ValueError):
    """Raised when benchmark card artifacts cannot be generated."""


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


class BenchmarkCardGenerator:
    """Build release-grade benchmark card artifacts from result bundles."""

    def __init__(self, results_dir: str = "results"):
        self.results_dir = Path(results_dir)

    def _load_frontier(self) -> Dict[str, Any]:
        path = self.results_dir / "CROSS_DOMAIN_FRONTIER.json"
        if not path.exists():
            raise BenchmarkCardError(f"Missing frontier artifact: {path}")
        payload = _load_json(path)
        if not isinstance(payload.get("domains"), list):
            raise BenchmarkCardError(f"Invalid frontier artifact structure: {path}")
        return payload

    def _manifest_summary(self) -> Tuple[List[str], List[str], Dict[str, int]]:
        protocol_versions: Dict[str, int] = {}
        commit_hashes = set()
        missing_domains: List[str] = []

        for domain in DEFAULT_DOMAIN_DIRS:
            manifest_path = self.results_dir / domain / "run_manifest.json"
            if not manifest_path.exists():
                missing_domains.append(domain)
                continue

            manifest = _load_json(manifest_path)
            protocol = str(manifest.get("benchmark_protocol_version", "unknown"))
            protocol_versions[protocol] = protocol_versions.get(protocol, 0) + 1

            commit_hash = manifest.get("git_commit_hash")
            if isinstance(commit_hash, str) and commit_hash.strip():
                commit_hashes.add(commit_hash.strip())

        return sorted(commit_hashes), missing_domains, protocol_versions

    def build_payload(self) -> Dict[str, Any]:
        frontier = self._load_frontier()
        commits, missing_manifest_domains, protocol_versions = self._manifest_summary()

        domains = frontier.get("domains", [])
        domain_names = [row.get("domain") for row in domains if isinstance(row, dict)]
        domain_names = [name for name in domain_names if isinstance(name, str)]

        champions = []
        for row in domains:
            if not isinstance(row, dict):
                continue
            champion = row.get("champion") if isinstance(row.get("champion"), dict) else {}
            champions.append(
                {
                    "domain": row.get("domain", "unknown"),
                    "domain_name": row.get("domain_name", row.get("domain", "unknown")),
                    "champion": champion.get("name", "unknown"),
                    "extraordinary_index": champion.get("extraordinary_index"),
                }
            )

        generalists = frontier.get("cross_domain_generalists", [])
        if not isinstance(generalists, list):
            generalists = []

        top_generalists = [
            {
                "name": row.get("name", "unknown"),
                "avg_extraordinary_index": row.get("avg_extraordinary_index"),
                "domains_covered": row.get("domains_covered"),
            }
            for row in generalists[:5]
            if isinstance(row, dict)
        ]

        expected_domains = list(DEFAULT_DOMAIN_DIRS)
        missing_frontier_domains = [name for name in expected_domains if name not in set(domain_names)]

        payload = {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "results_dir": str(self.results_dir),
            "domain_coverage": {
                "expected": len(expected_domains),
                "observed": len(domain_names),
                "missing_in_frontier": missing_frontier_domains,
                "missing_manifest": missing_manifest_domains,
            },
            "protocol_versions": protocol_versions,
            "git_commit_hashes": commits,
            "weights": frontier.get("weights", {}),
            "champions": champions,
            "top_generalists": top_generalists,
        }
        return payload

    def _render_markdown(self, payload: Dict[str, Any]) -> str:
        lines: List[str] = []
        lines.append("# Benchmark Card")
        lines.append("")
        lines.append(f"Generated UTC: {payload.get('generated_at_utc', 'N/A')}")
        lines.append(f"Results Directory: {payload.get('results_dir', 'N/A')}")
        lines.append("")

        coverage = payload.get("domain_coverage", {})
        lines.append("## Coverage")
        lines.append("")
        lines.append(f"- Expected domains: {coverage.get('expected', 'N/A')}")
        lines.append(f"- Observed domains: {coverage.get('observed', 'N/A')}")
        lines.append(f"- Missing in frontier: {coverage.get('missing_in_frontier', [])}")
        lines.append(f"- Missing manifests: {coverage.get('missing_manifest', [])}")
        lines.append("")

        lines.append("## Protocol and Commit Provenance")
        lines.append("")
        lines.append(f"- Protocol versions: {payload.get('protocol_versions', {})}")
        lines.append(f"- Commit hashes: {payload.get('git_commit_hashes', [])}")
        lines.append("")

        lines.append("## Domain Champions")
        lines.append("")
        lines.append("| Domain | Champion | Extraordinary Index |")
        lines.append("|--------|----------|---------------------|")
        for row in payload.get("champions", []):
            lines.append(
                f"| {row.get('domain_name', row.get('domain', 'unknown'))} | "
                f"{row.get('champion', 'unknown')} | {row.get('extraordinary_index', 'N/A')} |"
            )
        lines.append("")

        lines.append("## Top Cross-Domain Generalists")
        lines.append("")
        lines.append("| Approach | Avg Extraordinary Index | Domains Covered |")
        lines.append("|----------|--------------------------|-----------------|")
        for row in payload.get("top_generalists", []):
            lines.append(
                f"| {row.get('name', 'unknown')} | {row.get('avg_extraordinary_index', 'N/A')} | "
                f"{row.get('domains_covered', 'N/A')} |"
            )
        lines.append("")

        return "\n".join(lines)

    def save(self, output_dir: str | Path | None = None) -> Dict[str, str]:
        payload = self.build_payload()
        target = Path(output_dir) if output_dir is not None else self.results_dir
        target.mkdir(parents=True, exist_ok=True)

        json_path = target / "BENCHMARK_CARD.json"
        md_path = target / "BENCHMARK_CARD.md"

        with json_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        md_path.write_text(self._render_markdown(payload), encoding="utf-8")

        return {
            "json": str(json_path),
            "markdown": str(md_path),
        }
