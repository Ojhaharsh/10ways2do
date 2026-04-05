"""What-if policy simulation over cross-domain frontier candidates."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from itertools import product
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

    @staticmethod
    def _weight_grid(weight_step: float) -> List[Dict[str, float]]:
        if weight_step <= 0 or weight_step > 1:
            raise PolicySimulationError("weight_step must be in (0, 1]")

        inverse = round(1.0 / weight_step)
        if abs(inverse * weight_step - 1.0) > 1e-8:
            raise PolicySimulationError("weight_step must divide 1.0 exactly (e.g., 0.5, 0.25, 0.2, 0.1)")

        grid: List[Dict[str, float]] = []
        for a, b, c in product(range(inverse + 1), repeat=3):
            d = inverse - (a + b + c)
            if d < 0:
                continue
            grid.append(
                {
                    "quality_score": a * weight_step,
                    "speed_score": b * weight_step,
                    "resilience": c * weight_step,
                    "consistency": d * weight_step,
                }
            )
        return grid

    @staticmethod
    def _evaluate_objective(summary: Dict[str, Any], objective: str) -> float:
        total = max(1, int(summary.get("domains_total", 0)))
        covered = int(summary.get("domains_meeting_constraints", 0))
        avg_selected = float(summary.get("avg_selected_policy_score", 0.0))
        coverage_rate = covered / total

        if objective == "max_coverage":
            return coverage_rate + 0.01 * avg_selected
        if objective == "max_score":
            return avg_selected + 0.01 * coverage_rate
        if objective == "balanced":
            return 0.65 * coverage_rate + 0.35 * avg_selected

        raise PolicySimulationError(f"Unknown optimization objective: {objective}")

    @staticmethod
    def _percentile(values: List[float], q: float) -> float:
        if not values:
            return 0.0
        if q <= 0:
            return float(min(values))
        if q >= 100:
            return float(max(values))

        sorted_vals = sorted(float(v) for v in values)
        pos = (len(sorted_vals) - 1) * (q / 100.0)
        lo = int(pos)
        hi = min(lo + 1, len(sorted_vals) - 1)
        frac = pos - lo
        return sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac

    def _evaluate_weight_candidate(
        self,
        weights: Dict[str, float],
        mins: Dict[str, float],
        policy_name: str,
        top_k: int,
        objective: Optional[str] = None,
    ) -> Dict[str, Any]:
        payload = self.simulate(
            weights=weights,
            mins=mins,
            policy_name=policy_name,
            top_k=top_k,
        )

        selected_scores: List[float] = []
        for domain in payload.get("domains", []):
            selected = domain.get("selected") if isinstance(domain.get("selected"), dict) else None
            score = selected.get("policy_score") if selected else None
            if isinstance(score, (int, float)):
                selected_scores.append(float(score))

        avg_selected_score = sum(selected_scores) / len(selected_scores) if selected_scores else 0.0
        worst_selected_score = min(selected_scores) if selected_scores else 0.0
        payload["summary"]["avg_selected_policy_score"] = round(avg_selected_score, 6)
        payload["summary"]["worst_selected_policy_score"] = round(worst_selected_score, 6)

        total = max(1, int(payload["summary"].get("domains_total", 0)))
        covered = int(payload["summary"].get("domains_meeting_constraints", 0))
        payload["summary"]["coverage_rate"] = round(covered / total, 6)

        row = {
            "weights": payload["weights"],
            "summary": payload["summary"],
            "domains": payload["domains"],
        }

        if objective is not None:
            objective_value = self._evaluate_objective(payload["summary"], objective=objective)
            row["objective_value"] = round(float(objective_value), 6)

        return row

    @staticmethod
    def _pareto_non_dominated(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Return non-dominated policies over coverage, avg score, and worst score."""
        frontier: List[Dict[str, Any]] = []
        for candidate in rows:
            c_summary = candidate.get("summary", {})
            c_cov = float(c_summary.get("coverage_rate", 0.0))
            c_avg = float(c_summary.get("avg_selected_policy_score", 0.0))
            c_worst = float(c_summary.get("worst_selected_policy_score", 0.0))

            dominated = False
            for other in rows:
                if other is candidate:
                    continue
                o_summary = other.get("summary", {})
                o_cov = float(o_summary.get("coverage_rate", 0.0))
                o_avg = float(o_summary.get("avg_selected_policy_score", 0.0))
                o_worst = float(o_summary.get("worst_selected_policy_score", 0.0))

                better_or_equal = o_cov >= c_cov and o_avg >= c_avg and o_worst >= c_worst
                strictly_better = o_cov > c_cov or o_avg > c_avg or o_worst > c_worst
                if better_or_equal and strictly_better:
                    dominated = True
                    break

            if not dominated:
                frontier.append(candidate)

        frontier.sort(
            key=lambda r: (
                float(r.get("summary", {}).get("coverage_rate", 0.0)),
                float(r.get("summary", {}).get("avg_selected_policy_score", 0.0)),
                float(r.get("summary", {}).get("worst_selected_policy_score", 0.0)),
            ),
            reverse=True,
        )
        return frontier

    def optimize(
        self,
        mins: Optional[Dict[str, float]] = None,
        policy_name: str = "optimized_policy",
        objective: str = "balanced",
        weight_step: float = 0.25,
        top_k: int = 3,
        max_configs: Optional[int] = None,
        top_n: int = 5,
    ) -> Dict[str, Any]:
        mins = mins or {
            "quality_score": 0.0,
            "speed_score": 0.0,
            "resilience": 0.0,
            "consistency": 0.0,
        }

        candidates = self._weight_grid(weight_step=weight_step)
        if isinstance(max_configs, int) and max_configs > 0:
            candidates = candidates[:max_configs]

        evaluated: List[Dict[str, Any]] = []

        for idx, weights in enumerate(candidates):
            evaluated.append(
                self._evaluate_weight_candidate(
                    weights=weights,
                    mins=mins,
                    policy_name=f"{policy_name}_candidate_{idx}",
                    top_k=top_k,
                    objective=objective,
                )
            )

        if not evaluated:
            raise PolicySimulationError("No candidate policies generated for optimization")

        ranked = sorted(evaluated, key=lambda r: r["objective_value"], reverse=True)
        best = ranked[0]

        return {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "policy_name": policy_name,
            "objective": objective,
            "weight_step": weight_step,
            "minimum_constraints": mins,
            "search_space_size": len(candidates),
            "best_policy": best,
            "top_policies": ranked[: max(1, top_n)],
        }

    def optimize_frontier(
        self,
        mins: Optional[Dict[str, float]] = None,
        policy_name: str = "frontier_policy",
        weight_step: float = 0.25,
        top_k: int = 3,
        max_configs: Optional[int] = None,
        top_n: int = 10,
    ) -> Dict[str, Any]:
        mins = mins or {
            "quality_score": 0.0,
            "speed_score": 0.0,
            "resilience": 0.0,
            "consistency": 0.0,
        }

        candidates = self._weight_grid(weight_step=weight_step)
        if isinstance(max_configs, int) and max_configs > 0:
            candidates = candidates[:max_configs]

        evaluated: List[Dict[str, Any]] = []
        for idx, weights in enumerate(candidates):
            evaluated.append(
                self._evaluate_weight_candidate(
                    weights=weights,
                    mins=mins,
                    policy_name=f"{policy_name}_candidate_{idx}",
                    top_k=top_k,
                    objective=None,
                )
            )

        if not evaluated:
            raise PolicySimulationError("No candidate policies generated for frontier optimization")

        frontier = self._pareto_non_dominated(evaluated)

        coverage_values = [float(r.get("summary", {}).get("coverage_rate", 0.0)) for r in frontier]
        avg_values = [float(r.get("summary", {}).get("avg_selected_policy_score", 0.0)) for r in frontier]
        worst_values = [float(r.get("summary", {}).get("worst_selected_policy_score", 0.0)) for r in frontier]

        stability_bands = {
            "coverage_rate": {
                "p10": round(self._percentile(coverage_values, 10), 6),
                "p50": round(self._percentile(coverage_values, 50), 6),
                "p90": round(self._percentile(coverage_values, 90), 6),
            },
            "avg_selected_policy_score": {
                "p10": round(self._percentile(avg_values, 10), 6),
                "p50": round(self._percentile(avg_values, 50), 6),
                "p90": round(self._percentile(avg_values, 90), 6),
            },
            "worst_selected_policy_score": {
                "p10": round(self._percentile(worst_values, 10), 6),
                "p50": round(self._percentile(worst_values, 50), 6),
                "p90": round(self._percentile(worst_values, 90), 6),
            },
        }

        return {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "policy_name": policy_name,
            "mode": "multi_objective_pareto",
            "weight_step": weight_step,
            "minimum_constraints": mins,
            "search_space_size": len(candidates),
            "frontier_size": len(frontier),
            "stability_bands": stability_bands,
            "frontier_policies": frontier[: max(1, top_n)],
        }

    @staticmethod
    def _optimization_to_markdown(payload: Dict[str, Any]) -> str:
        lines: List[str] = []
        lines.append(f"# Policy Optimization: {payload.get('policy_name', 'optimized_policy')}")
        lines.append("")
        lines.append(
            f"Objective: {payload.get('objective', 'balanced')} | "
            f"Search Space: {payload.get('search_space_size', 0)} candidates"
        )
        lines.append("")

        best = payload.get("best_policy", {})
        weights = best.get("weights", {})
        summary = best.get("summary", {})
        lines.append("## Best Policy")
        lines.append("")
        lines.append(
            "Weights: "
            f"quality={weights.get('quality_score', 0.0):.2f}, "
            f"speed={weights.get('speed_score', 0.0):.2f}, "
            f"resilience={weights.get('resilience', 0.0):.2f}, "
            f"consistency={weights.get('consistency', 0.0):.2f}"
        )
        lines.append(
            "Coverage: "
            f"{summary.get('domains_meeting_constraints', 0)}/{summary.get('domains_total', 0)} | "
            f"Avg selected score={summary.get('avg_selected_policy_score', 0.0):.4f}"
        )
        lines.append("")
        lines.append("| Domain | Selected | Score | Constraints Met |")
        lines.append("|--------|----------|-------|-----------------|")
        for row in best.get("domains", []):
            selected = row.get("selected", {})
            score = selected.get("policy_score")
            score_txt = f"{score:.4f}" if isinstance(score, (int, float)) else "N/A"
            lines.append(
                f"| {row.get('domain_name', row.get('domain', 'unknown'))} "
                f"| {selected.get('name', 'N/A')} "
                f"| {score_txt} "
                f"| {'yes' if row.get('constraints_satisfied') else 'no'} |"
            )

        lines.append("")
        lines.append("## Top Policies")
        lines.append("")
        lines.append("| Rank | Objective | Coverage | Avg Score | Weights (Q,S,R,C) |")
        lines.append("|------|-----------|----------|-----------|-------------------|")
        for idx, row in enumerate(payload.get("top_policies", []), start=1):
            s = row.get("summary", {})
            w = row.get("weights", {})
            lines.append(
                f"| {idx} | {row.get('objective_value', 0.0):.4f} "
                f"| {s.get('domains_meeting_constraints', 0)}/{s.get('domains_total', 0)} "
                f"| {s.get('avg_selected_policy_score', 0.0):.4f} "
                f"| ({w.get('quality_score', 0.0):.2f}, {w.get('speed_score', 0.0):.2f}, {w.get('resilience', 0.0):.2f}, {w.get('consistency', 0.0):.2f}) |"
            )

        lines.append("")
        return "\n".join(lines)

    def save_optimization(
        self,
        mins: Optional[Dict[str, float]] = None,
        policy_name: str = "optimized_policy",
        objective: str = "balanced",
        weight_step: float = 0.25,
        top_k: int = 3,
        max_configs: Optional[int] = None,
        top_n: int = 5,
    ) -> Dict[str, Path]:
        payload = self.optimize(
            mins=mins,
            policy_name=policy_name,
            objective=objective,
            weight_step=weight_step,
            top_k=top_k,
            max_configs=max_configs,
            top_n=top_n,
        )

        slug = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in policy_name).strip("_")
        slug = slug or "optimized_policy"

        json_path = self.results_dir / f"POLICY_OPTIMIZATION_{slug}.json"
        md_path = self.results_dir / f"POLICY_OPTIMIZATION_{slug}.md"
        self.results_dir.mkdir(parents=True, exist_ok=True)

        json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        md_path.write_text(self._optimization_to_markdown(payload), encoding="utf-8")
        return {"json": json_path, "markdown": md_path}

    @staticmethod
    def _frontier_to_markdown(payload: Dict[str, Any]) -> str:
        lines: List[str] = []
        lines.append(f"# Policy Frontier Optimization: {payload.get('policy_name', 'frontier_policy')}")
        lines.append("")
        lines.append(
            f"Search Space: {payload.get('search_space_size', 0)} candidates | "
            f"Frontier Size: {payload.get('frontier_size', 0)}"
        )
        lines.append("")

        bands = payload.get("stability_bands", {})
        lines.append("## Stability Bands")
        lines.append("")
        for metric, stats in bands.items():
            lines.append(
                f"- {metric}: p10={stats.get('p10', 0.0):.4f}, "
                f"p50={stats.get('p50', 0.0):.4f}, p90={stats.get('p90', 0.0):.4f}"
            )
        lines.append("")

        lines.append("## Non-Dominated Policies")
        lines.append("")
        lines.append("| Rank | Coverage | Avg Score | Worst Score | Weights (Q,S,R,C) |")
        lines.append("|------|----------|-----------|-------------|-------------------|")
        for idx, row in enumerate(payload.get("frontier_policies", []), start=1):
            s = row.get("summary", {})
            w = row.get("weights", {})
            lines.append(
                f"| {idx} "
                f"| {s.get('coverage_rate', 0.0):.4f} "
                f"| {s.get('avg_selected_policy_score', 0.0):.4f} "
                f"| {s.get('worst_selected_policy_score', 0.0):.4f} "
                f"| ({w.get('quality_score', 0.0):.2f}, {w.get('speed_score', 0.0):.2f}, {w.get('resilience', 0.0):.2f}, {w.get('consistency', 0.0):.2f}) |"
            )
        lines.append("")
        return "\n".join(lines)

    def save_frontier_optimization(
        self,
        mins: Optional[Dict[str, float]] = None,
        policy_name: str = "frontier_policy",
        weight_step: float = 0.25,
        top_k: int = 3,
        max_configs: Optional[int] = None,
        top_n: int = 10,
    ) -> Dict[str, Path]:
        payload = self.optimize_frontier(
            mins=mins,
            policy_name=policy_name,
            weight_step=weight_step,
            top_k=top_k,
            max_configs=max_configs,
            top_n=top_n,
        )

        slug = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in policy_name).strip("_")
        slug = slug or "frontier_policy"

        json_path = self.results_dir / f"POLICY_FRONTIER_{slug}.json"
        md_path = self.results_dir / f"POLICY_FRONTIER_{slug}.md"
        self.results_dir.mkdir(parents=True, exist_ok=True)

        json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        md_path.write_text(self._frontier_to_markdown(payload), encoding="utf-8")
        return {"json": json_path, "markdown": md_path}