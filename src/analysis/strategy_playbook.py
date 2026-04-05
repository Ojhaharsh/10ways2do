"""Scenario-based strategy playbook generation from frontier artifacts."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


class StrategyPlaybookError(ValueError):
    """Raised when strategy playbook generation cannot proceed."""


SCENARIO_WEIGHTS: Dict[str, Dict[str, float]] = {
    "balanced_production": {
        "quality_score": 0.45,
        "speed_score": 0.25,
        "resilience": 0.20,
        "consistency": 0.10,
    },
    "accuracy_first": {
        "quality_score": 0.70,
        "speed_score": 0.05,
        "resilience": 0.15,
        "consistency": 0.10,
    },
    "latency_first": {
        "quality_score": 0.25,
        "speed_score": 0.55,
        "resilience": 0.10,
        "consistency": 0.10,
    },
    "reliability_first": {
        "quality_score": 0.35,
        "speed_score": 0.10,
        "resilience": 0.35,
        "consistency": 0.20,
    },
}


class StrategyPlaybookGenerator:
    """Create recommendation playbooks from CROSS_DOMAIN_FRONTIER artifacts."""

    def __init__(self, results_dir: str = "results"):
        self.results_dir = Path(results_dir)

    def _load_frontier(self) -> Dict[str, Any]:
        frontier_path = self.results_dir / "CROSS_DOMAIN_FRONTIER.json"
        if not frontier_path.exists():
            raise StrategyPlaybookError(
                f"Frontier artifact not found: {frontier_path}. Generate report first."
            )

        with frontier_path.open("r", encoding="utf-8") as f:
            payload = json.load(f)

        domains = payload.get("domains")
        if not isinstance(domains, list) or not domains:
            raise StrategyPlaybookError("Frontier artifact has no domain entries.")
        return payload

    @staticmethod
    def _composite(candidate: Dict[str, Any], weights: Dict[str, float]) -> float:
        return (
            float(candidate.get("quality_score", 0.0)) * weights["quality_score"]
            + float(candidate.get("speed_score", 0.0)) * weights["speed_score"]
            + float(candidate.get("resilience", 0.0)) * weights["resilience"]
            + float(candidate.get("consistency", 0.0)) * weights["consistency"]
        )

    def _build_playbook_payload(self, frontier: Dict[str, Any]) -> Dict[str, Any]:
        domains = frontier.get("domains", [])
        scenarios: Dict[str, Any] = {}

        for scenario_name, weights in SCENARIO_WEIGHTS.items():
            scenario_rows: List[Dict[str, Any]] = []
            recommendation_count = 0

            for domain_row in domains:
                domain = domain_row.get("domain", "unknown")
                domain_name = domain_row.get("domain_name", domain)
                candidates = domain_row.get("pareto_frontier", [])
                if not isinstance(candidates, list) or not candidates:
                    scenario_rows.append(
                        {
                            "domain": domain,
                            "domain_name": domain_name,
                            "recommended_approach": None,
                            "scenario_score": None,
                            "runner_up": None,
                            "notes": "No Pareto candidates available",
                        }
                    )
                    continue

                ranked = sorted(
                    candidates,
                    key=lambda c: self._composite(c, weights),
                    reverse=True,
                )
                best = ranked[0]
                runner_up = ranked[1] if len(ranked) > 1 else None
                recommendation_count += 1

                scenario_rows.append(
                    {
                        "domain": domain,
                        "domain_name": domain_name,
                        "recommended_approach": best.get("name"),
                        "scenario_score": round(self._composite(best, weights), 6),
                        "runner_up": runner_up.get("name") if isinstance(runner_up, dict) else None,
                        "champion_alignment": best.get("name") == domain_row.get("champion", {}).get("name"),
                    }
                )

            scenarios[scenario_name] = {
                "weights": dict(weights),
                "coverage": {
                    "domains_with_recommendations": recommendation_count,
                    "total_domains": len(domains),
                },
                "recommendations": scenario_rows,
            }

        return {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_frontier_generated_at_utc": frontier.get("generated_at_utc"),
            "scenarios": scenarios,
            "cross_domain_generalists": frontier.get("cross_domain_generalists", []),
        }

    @staticmethod
    def _to_markdown(payload: Dict[str, Any]) -> str:
        lines: List[str] = []
        lines.append("# Strategy Playbook")
        lines.append("")
        lines.append("Scenario-based recommendations generated from the cross-domain Pareto frontier.")
        lines.append("")

        for scenario_name, scenario_payload in payload.get("scenarios", {}).items():
            lines.append(f"## Scenario: {scenario_name}")
            lines.append("")
            weights = scenario_payload.get("weights", {})
            lines.append(
                "Weights: "
                f"quality={weights.get('quality_score', 0.0):.2f}, "
                f"speed={weights.get('speed_score', 0.0):.2f}, "
                f"resilience={weights.get('resilience', 0.0):.2f}, "
                f"consistency={weights.get('consistency', 0.0):.2f}"
            )
            lines.append("")
            lines.append("| Domain | Recommended Approach | Scenario Score | Runner-up |")
            lines.append("|--------|----------------------|----------------|-----------|")
            for rec in scenario_payload.get("recommendations", []):
                score = rec.get("scenario_score")
                score_text = f"{score:.4f}" if isinstance(score, (int, float)) else "N/A"
                lines.append(
                    f"| {rec.get('domain_name', rec.get('domain', 'unknown'))} "
                    f"| {rec.get('recommended_approach') or 'N/A'} "
                    f"| {score_text} "
                    f"| {rec.get('runner_up') or 'N/A'} |"
                )
            lines.append("")

        generalists = payload.get("cross_domain_generalists", [])
        if isinstance(generalists, list) and generalists:
            lines.append("## Cross-Domain Generalists")
            lines.append("")
            for row in generalists[:5]:
                lines.append(
                    f"- {row.get('name', 'unknown')}: "
                    f"avg_index={row.get('avg_extraordinary_index', 'N/A')} "
                    f"across {row.get('domains_covered', 'N/A')} domains"
                )

        lines.append("")
        return "\n".join(lines)

    def save(self) -> Dict[str, Path]:
        """Generate and persist playbook artifacts under results_dir."""
        frontier = self._load_frontier()
        payload = self._build_playbook_payload(frontier)

        json_path = self.results_dir / "STRATEGY_PLAYBOOK.json"
        md_path = self.results_dir / "STRATEGY_PLAYBOOK.md"

        self.results_dir.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        md_path.write_text(self._to_markdown(payload), encoding="utf-8")

        return {
            "json": json_path,
            "markdown": md_path,
        }