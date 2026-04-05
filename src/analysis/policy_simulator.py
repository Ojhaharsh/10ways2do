"""What-if policy simulation over cross-domain frontier candidates."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


class PolicySimulationError(ValueError):
    """Raised when policy simulation cannot run."""


def _normalize_weights(weights: Dict[str, float]) -> Dict[str, float]:
    total = sum(max(0.0, float(v)) for v in weights.values())
    if total <= 0:
        raise PolicySimulationError("Policy weights must sum to a positive value")
    return {k: max(0.0, float(v)) / total for k, v in weights.items()}


class PolicySimulator:
    """Simulate domain-wise strategy selection under custom policy settings."""

    def __init__(self, results_dir: str = "results"):
        self.results_dir = Path(results_dir)

    def _load_frontier(self) -> Dict[str, Any]:
        path = self.results_dir / "CROSS_DOMAIN_FRONTIER.json"
        if not path.exists():
            raise PolicySimulationError(
                f"Missing frontier artifact at {path}. Run report generation first."
            )
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        domains = payload.get("domains")
        if not isinstance(domains, list) or not domains:
            raise PolicySimulationError("Frontier artifact has no domain entries")
        return payload

    @staticmethod
    def _score(candidate: Dict[str, Any], weights: Dict[str, float]) -> float:
        return (
            float(candidate.get("quality_score", 0.0)) * weights["quality_score"]
            + float(candidate.get("speed_score", 0.0)) * weights["speed_score"]
            + float(candidate.get("resilience", 0.0)) * weights["resilience"]
            + float(candidate.get("consistency", 0.0)) * weights["consistency"]
        )

    @staticmethod
    def _meets_constraints(candidate: Dict[str, Any], mins: Dict[str, float]) -> bool:
        for metric, threshold in mins.items():
            value = float(candidate.get(metric, 0.0))
            if value < threshold:
                return False
        return True

    def simulate(
        self,
        weights: Dict[str, float],
        mins: Optional[Dict[str, float]] = None,
        policy_name: str = "custom_policy",
        top_k: int = 3,
    ) -> Dict[str, Any]:
        frontier = self._load_frontier()
        normalized = _normalize_weights(weights)
        mins = mins or {
            "quality_score": 0.0,
            "speed_score": 0.0,
            "resilience": 0.0,
            "consistency": 0.0,
        }

        domain_results: List[Dict[str, Any]] = []
        satisfied_constraints = 0

        for domain in frontier.get("domains", []):
            domain_key = domain.get("domain", "unknown")
            domain_name = domain.get("domain_name", domain_key)
            candidates = domain.get("pareto_frontier", [])

            if not isinstance(candidates, list) or not candidates:
                domain_results.append(
                    {
                        "domain": domain_key,
                        "domain_name": domain_name,
                        "selected": None,
                        "alternatives": [],
                        "constraints_satisfied": False,
                        "reason": "No candidates available",
                    }
                )
                continue

            enriched = []
            for candidate in candidates:
                row = dict(candidate)
                row["policy_score"] = round(self._score(row, normalized), 6)
                row["constraints_satisfied"] = self._meets_constraints(row, mins)
                enriched.append(row)

            feasible = [r for r in enriched if r["constraints_satisfied"]]
            ranked = sorted(enriched, key=lambda r: r["policy_score"], reverse=True)
            feasible_ranked = sorted(feasible, key=lambda r: r["policy_score"], reverse=True)

            if feasible_ranked:
                selected = feasible_ranked[0]
                satisfied_constraints += 1
                reason = "Best feasible candidate under policy"
            else:
                selected = ranked[0]
                reason = "No feasible candidate met constraints; selected best policy score"

            alternatives = [
                {
                    "name": item.get("name"),
                    "policy_score": item.get("policy_score"),
                    "constraints_satisfied": item.get("constraints_satisfied"),
                }
                for item in ranked[: max(1, top_k)]
            ]

            domain_results.append(
                {
                    "domain": domain_key,
                    "domain_name": domain_name,
                    "selected": {
                        "name": selected.get("name"),
                        "policy_score": selected.get("policy_score"),
                        "constraints_satisfied": selected.get("constraints_satisfied"),
                    },
                    "alternatives": alternatives,
                    "constraints_satisfied": bool(selected.get("constraints_satisfied")),
                    "reason": reason,
                }
            )

        return {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "source_frontier_generated_at_utc": frontier.get("generated_at_utc"),
            "policy_name": policy_name,
            "weights": normalized,
            "minimum_constraints": mins,
            "domains": domain_results,
            "summary": {
                "domains_total": len(domain_results),
                "domains_meeting_constraints": satisfied_constraints,
            },
        }

    @staticmethod
    def _to_markdown(payload: Dict[str, Any]) -> str:
        lines: List[str] = []
        lines.append(f"# Policy Simulation: {payload.get('policy_name', 'custom_policy')}")
        lines.append("")

        weights = payload.get("weights", {})
        lines.append(
            "Weights: "
            f"quality={weights.get('quality_score', 0.0):.2f}, "
            f"speed={weights.get('speed_score', 0.0):.2f}, "
            f"resilience={weights.get('resilience', 0.0):.2f}, "
            f"consistency={weights.get('consistency', 0.0):.2f}"
        )
        lines.append("")
        lines.append("| Domain | Selected | Policy Score | Constraints Met |")
        lines.append("|--------|----------|--------------|-----------------|")

        for domain in payload.get("domains", []):
            selected = domain.get("selected") or {}
            score = selected.get("policy_score")
            score_text = f"{score:.4f}" if isinstance(score, (int, float)) else "N/A"
            lines.append(
                f"| {domain.get('domain_name', domain.get('domain', 'unknown'))} "
                f"| {selected.get('name', 'N/A')} "
                f"| {score_text} "
                f"| {'yes' if domain.get('constraints_satisfied') else 'no'} |"
            )

        summary = payload.get("summary", {})
        lines.append("")
        lines.append(
            "Coverage: "
            f"{summary.get('domains_meeting_constraints', 0)}/"
            f"{summary.get('domains_total', 0)} domains meet constraints"
        )
        lines.append("")
        return "\n".join(lines)

    def save(
        self,
        weights: Dict[str, float],
        mins: Optional[Dict[str, float]] = None,
        policy_name: str = "custom_policy",
        top_k: int = 3,
    ) -> Dict[str, Path]:
        payload = self.simulate(weights=weights, mins=mins, policy_name=policy_name, top_k=top_k)

        slug = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in policy_name).strip("_")
        slug = slug or "custom_policy"

        json_path = self.results_dir / f"POLICY_SIMULATION_{slug}.json"
        md_path = self.results_dir / f"POLICY_SIMULATION_{slug}.md"
        self.results_dir.mkdir(parents=True, exist_ok=True)

        json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        md_path.write_text(self._to_markdown(payload), encoding="utf-8")
        return {"json": json_path, "markdown": md_path}