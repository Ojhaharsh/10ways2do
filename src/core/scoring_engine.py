"""
8-Axis Scoring Engine for 10ways2do Benchmark Platform.

Computes a multi-dimensional radar score for each evaluated model:
1. Accuracy — Raw correctness across challenges
2. Speed — Response latency normalized by difficulty
3. Cost — Dollar cost per evaluation (token efficiency)
4. Robustness — Performance stability across difficulty levels
5. Fairness — Consistency across challenge types and domains
6. Consistency — Reproducibility across repeated evaluations
7. Generalization — Cross-domain performance breadth
8. Efficiency — Token economy (quality per token spent)

The composite "10w2d Score" is a configurable weighted aggregate.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path

import numpy as np


@dataclass
class AxisScore:
    """Score on a single evaluation axis."""

    axis: str
    score: float  # 0.0 - 1.0
    raw_value: float  # Original metric value
    percentile: Optional[float] = None  # vs. other models, if available
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "axis": self.axis,
            "score": round(self.score, 4),
            "raw_value": round(self.raw_value, 4) if isinstance(self.raw_value, float) else self.raw_value,
        }
        if self.percentile is not None:
            d["percentile"] = round(self.percentile, 2)
        if self.details:
            d["details"] = self.details
        return d


@dataclass
class RadarProfile:
    """Full 8-axis radar profile for a model."""

    model_id: str
    axes: Dict[str, AxisScore] = field(default_factory=dict)
    composite_score: float = 0.0
    weights_used: Dict[str, float] = field(default_factory=dict)
    rank: Optional[int] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def scores_array(self) -> List[float]:
        """Ordered scores for radar chart rendering."""
        ordered_axes = ["accuracy", "speed", "cost", "robustness", "fairness", "consistency", "generalization", "efficiency"]
        return [self.axes.get(a, AxisScore(axis=a, score=0, raw_value=0)).score for a in ordered_axes]

    @property
    def axis_labels(self) -> List[str]:
        return ["Accuracy", "Speed", "Cost", "Robustness", "Fairness", "Consistency", "Generalization", "Efficiency"]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            "composite_score": round(self.composite_score, 4),
            "rank": self.rank,
            "axes": {k: v.to_dict() for k, v in self.axes.items()},
            "weights_used": {k: round(v, 3) for k, v in self.weights_used.items()},
            "scores_array": [round(s, 4) for s in self.scores_array],
            "axis_labels": self.axis_labels,
            "metadata": self.metadata,
        }

    def to_leaderboard_row(self) -> Dict[str, Any]:
        """Compact format for leaderboard rendering."""
        return {
            "model": self.model_id,
            "score": round(self.composite_score, 3),
            "rank": self.rank,
            "accuracy": round(self.axes.get("accuracy", AxisScore("accuracy", 0, 0)).score, 3),
            "speed": round(self.axes.get("speed", AxisScore("speed", 0, 0)).score, 3),
            "cost": round(self.axes.get("cost", AxisScore("cost", 0, 0)).score, 3),
            "robustness": round(self.axes.get("robustness", AxisScore("robustness", 0, 0)).score, 3),
            "fairness": round(self.axes.get("fairness", AxisScore("fairness", 0, 0)).score, 3),
            "consistency": round(self.axes.get("consistency", AxisScore("consistency", 0, 0)).score, 3),
            "generalization": round(self.axes.get("generalization", AxisScore("generalization", 0, 0)).score, 3),
            "efficiency": round(self.axes.get("efficiency", AxisScore("efficiency", 0, 0)).score, 3),
        }


# ──────────────────────────────────────────────────────────────────────────────
# Scoring Profiles
# ──────────────────────────────────────────────────────────────────────────────

SCORING_PROFILES = {
    "balanced": {
        "accuracy": 0.20,
        "speed": 0.10,
        "cost": 0.10,
        "robustness": 0.15,
        "fairness": 0.10,
        "consistency": 0.10,
        "generalization": 0.15,
        "efficiency": 0.10,
    },
    "enterprise": {
        "accuracy": 0.20,
        "speed": 0.15,
        "cost": 0.20,
        "robustness": 0.20,
        "fairness": 0.10,
        "consistency": 0.05,
        "generalization": 0.05,
        "efficiency": 0.05,
    },
    "research": {
        "accuracy": 0.30,
        "speed": 0.05,
        "cost": 0.05,
        "robustness": 0.10,
        "fairness": 0.05,
        "consistency": 0.10,
        "generalization": 0.25,
        "efficiency": 0.10,
    },
    "safety": {
        "accuracy": 0.10,
        "speed": 0.05,
        "cost": 0.05,
        "robustness": 0.30,
        "fairness": 0.25,
        "consistency": 0.15,
        "generalization": 0.05,
        "efficiency": 0.05,
    },
    "speed_optimized": {
        "accuracy": 0.15,
        "speed": 0.30,
        "cost": 0.20,
        "robustness": 0.05,
        "fairness": 0.05,
        "consistency": 0.05,
        "generalization": 0.05,
        "efficiency": 0.15,
    },
}


# ──────────────────────────────────────────────────────────────────────────────
# Scoring Engine
# ──────────────────────────────────────────────────────────────────────────────

class ScoringEngine:
    """
    Computes 8-axis radar scores from evaluation results.

    Usage:
        engine = ScoringEngine(profile="balanced")
        profile = engine.compute_profile(evaluation_report)
        leaderboard = engine.rank_models([profile1, profile2, ...])
    """

    def __init__(self, profile: str = "balanced", custom_weights: Optional[Dict[str, float]] = None):
        if custom_weights:
            self.weights = custom_weights.copy()
        elif profile in SCORING_PROFILES:
            self.weights = SCORING_PROFILES[profile].copy()
        else:
            raise ValueError(f"Unknown profile '{profile}'. Available: {list(SCORING_PROFILES.keys())}")

        # Normalize
        total = sum(self.weights.values())
        self.weights = {k: v / total for k, v in self.weights.items()}
        self.profile_name = profile

    def compute_profile(
        self,
        evaluation_report,  # DomainEvaluationReport from process_evaluator
        additional_domain_reports: Optional[List] = None,
    ) -> RadarProfile:
        """
        Compute a full 8-axis radar profile from evaluation results.

        Args:
            evaluation_report: Primary DomainEvaluationReport
            additional_domain_reports: Optional list of reports from other domains
                                       (used for computing generalization)
        """
        profile = RadarProfile(
            model_id=evaluation_report.model_id,
            weights_used=self.weights.copy(),
        )

        results = evaluation_report.results

        if not results:
            return profile

        # ── Axis 1: Accuracy ──
        accuracy_raw = evaluation_report.accuracy
        accuracy_score = accuracy_raw  # Already 0-1
        profile.axes["accuracy"] = AxisScore(
            axis="accuracy",
            score=accuracy_score,
            raw_value=accuracy_raw,
            details={
                "n_correct": evaluation_report.n_correct,
                "n_total": evaluation_report.n_total,
                "mean_answer_score": evaluation_report.mean_answer_score,
            },
        )

        # ── Axis 2: Speed ──
        avg_latency = evaluation_report.total_latency_ms / max(1, evaluation_report.n_total)
        # Normalize: <1s = 1.0, 1-5s = 0.7-1.0, 5-15s = 0.3-0.7, >15s = <0.3
        speed_score = self._normalize_latency(avg_latency)
        profile.axes["speed"] = AxisScore(
            axis="speed",
            score=speed_score,
            raw_value=avg_latency,
            details={"total_latency_ms": evaluation_report.total_latency_ms},
        )

        # ── Axis 3: Cost ──
        cost_per_challenge = evaluation_report.total_cost_usd / max(1, evaluation_report.n_total)
        cost_score = self._normalize_cost(cost_per_challenge)
        profile.axes["cost"] = AxisScore(
            axis="cost",
            score=cost_score,
            raw_value=cost_per_challenge,
            details={"total_cost_usd": evaluation_report.total_cost_usd},
        )

        # ── Axis 4: Robustness ──
        difficulty_scores = evaluation_report.scores_by_difficulty
        robustness_score = self._compute_robustness(difficulty_scores)
        profile.axes["robustness"] = AxisScore(
            axis="robustness",
            score=robustness_score,
            raw_value=robustness_score,
            details={"scores_by_difficulty": difficulty_scores},
        )

        # ── Axis 5: Fairness ──
        fairness_score = self._compute_fairness(results)
        profile.axes["fairness"] = AxisScore(
            axis="fairness",
            score=fairness_score,
            raw_value=fairness_score,
            details={},
        )

        # ── Axis 6: Consistency ──
        consistency_score = self._compute_consistency(results)
        profile.axes["consistency"] = AxisScore(
            axis="consistency",
            score=consistency_score,
            raw_value=consistency_score,
            details={},
        )

        # ── Axis 7: Generalization ──
        if additional_domain_reports:
            gen_score = self._compute_generalization(
                evaluation_report, additional_domain_reports
            )
        else:
            gen_score = accuracy_score * 0.8  # Estimate from single domain
        profile.axes["generalization"] = AxisScore(
            axis="generalization",
            score=gen_score,
            raw_value=gen_score,
            details={},
        )

        # ── Axis 8: Efficiency ──
        tokens_per_correct = evaluation_report.total_tokens / max(1, evaluation_report.n_correct)
        efficiency_score = self._normalize_token_efficiency(tokens_per_correct)
        profile.axes["efficiency"] = AxisScore(
            axis="efficiency",
            score=efficiency_score,
            raw_value=tokens_per_correct,
            details={"total_tokens": evaluation_report.total_tokens},
        )

        # ── Composite Score ──
        profile.composite_score = sum(
            self.weights.get(axis, 0) * axis_score.score
            for axis, axis_score in profile.axes.items()
        )

        profile.metadata = {
            "profile": self.profile_name,
            "domain": evaluation_report.domain,
            "session_id": evaluation_report.session_id,
        }

        return profile

    def rank_models(self, profiles: List[RadarProfile]) -> List[RadarProfile]:
        """Rank models by composite score and assign rank numbers."""
        sorted_profiles = sorted(profiles, key=lambda p: p.composite_score, reverse=True)
        for i, profile in enumerate(sorted_profiles):
            profile.rank = i + 1

            # Compute percentiles for each axis
            for axis in profile.axes:
                all_scores = [p.axes[axis].score for p in sorted_profiles if axis in p.axes]
                if all_scores:
                    score = profile.axes[axis].score
                    percentile = (sum(1 for s in all_scores if s <= score) / len(all_scores)) * 100
                    profile.axes[axis].percentile = percentile

        return sorted_profiles

    def generate_leaderboard(self, profiles: List[RadarProfile]) -> Dict[str, Any]:
        """Generate a complete leaderboard data structure."""
        ranked = self.rank_models(profiles)

        return {
            "generated_at_utc": None,  # Will be set by caller
            "scoring_profile": self.profile_name,
            "weights": self.weights,
            "n_models": len(ranked),
            "leaderboard": [p.to_leaderboard_row() for p in ranked],
            "full_profiles": [p.to_dict() for p in ranked],
        }

    # ──────────────────────────────────────────────────────────────────────
    # Normalization functions
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def _normalize_latency(avg_latency_ms: float) -> float:
        """Convert average latency to 0-1 score (lower latency = higher score)."""
        if avg_latency_ms <= 500:
            return 1.0
        elif avg_latency_ms <= 1000:
            return 0.95
        elif avg_latency_ms <= 3000:
            return 0.85 - 0.15 * ((avg_latency_ms - 1000) / 2000)
        elif avg_latency_ms <= 10000:
            return 0.7 - 0.3 * ((avg_latency_ms - 3000) / 7000)
        elif avg_latency_ms <= 30000:
            return 0.4 - 0.2 * ((avg_latency_ms - 10000) / 20000)
        else:
            return max(0.05, 0.2 - 0.15 * min(1.0, (avg_latency_ms - 30000) / 30000))

    @staticmethod
    def _normalize_cost(cost_per_challenge: float) -> float:
        """Convert per-challenge cost to 0-1 score (lower cost = higher score)."""
        if cost_per_challenge <= 0:
            return 1.0  # Free (local model)
        elif cost_per_challenge <= 0.001:
            return 0.95
        elif cost_per_challenge <= 0.005:
            return 0.85
        elif cost_per_challenge <= 0.01:
            return 0.7
        elif cost_per_challenge <= 0.05:
            return 0.5
        elif cost_per_challenge <= 0.10:
            return 0.3
        else:
            return max(0.05, 0.3 - 0.25 * min(1.0, (cost_per_challenge - 0.10) / 0.50))

    @staticmethod
    def _normalize_token_efficiency(tokens_per_correct: float) -> float:
        """Convert tokens-per-correct-answer to 0-1 score."""
        if tokens_per_correct <= 0:
            return 0.5
        elif tokens_per_correct <= 100:
            return 1.0
        elif tokens_per_correct <= 300:
            return 0.85
        elif tokens_per_correct <= 800:
            return 0.7
        elif tokens_per_correct <= 2000:
            return 0.5
        else:
            return max(0.1, 0.5 - 0.4 * min(1.0, (tokens_per_correct - 2000) / 5000))

    @staticmethod
    def _compute_robustness(scores_by_difficulty: Dict[str, float]) -> float:
        """
        Robustness = how well performance holds across difficulty levels.
        High robustness = small drop from easy to hard.
        """
        if not scores_by_difficulty:
            return 0.5

        values = list(scores_by_difficulty.values())
        if len(values) < 2:
            return values[0] if values else 0.5

        # Compute coefficient of variation (lower = more robust)
        mean_score = np.mean(values)
        std_score = np.std(values)

        if mean_score < 1e-10:
            return 0.0

        cv = std_score / mean_score

        # Also check the drop from best to worst
        max_drop = max(values) - min(values)

        # Score: low CV and low max_drop = high robustness
        cv_score = max(0, 1.0 - cv * 2)  # CV of 0.5 maps to 0
        drop_score = max(0, 1.0 - max_drop)

        return 0.6 * cv_score + 0.4 * drop_score

    @staticmethod
    def _compute_fairness(results: list) -> float:
        """
        Fairness = consistent performance regardless of challenge type/content.
        We approximate via variance in answer scores.
        """
        if not results:
            return 0.5

        scores = [r.answer_score for r in results]
        if len(scores) < 3:
            return np.mean(scores)

        std = np.std(scores)
        mean = np.mean(scores)

        if mean < 1e-10:
            return 0.0

        # Low variance relative to mean = fair
        cv = std / mean
        return max(0, min(1.0, 1.0 - cv))

    @staticmethod
    def _compute_consistency(results: list) -> float:
        """
        Consistency = reliability of producing similar quality responses.
        Measured by the spread of overall scores.
        """
        if not results:
            return 0.5

        scores = [r.overall_score for r in results]
        if len(scores) < 3:
            return np.mean(scores)

        # Interquartile range
        q25, q75 = np.percentile(scores, [25, 75])
        iqr = q75 - q25

        # Low IQR = consistent
        return max(0, min(1.0, 1.0 - iqr * 2))

    @staticmethod
    def _compute_generalization(primary_report, other_reports: list) -> float:
        """
        Generalization = how well a model performs across different domains.
        High generalization = small spread between domain scores.
        """
        all_accuracies = [primary_report.accuracy]
        for report in other_reports:
            all_accuracies.append(report.accuracy)

        if len(all_accuracies) < 2:
            return primary_report.accuracy * 0.8

        mean_acc = np.mean(all_accuracies)
        min_acc = np.min(all_accuracies)

        # Penalize large gaps between best and worst domain
        gap = mean_acc - min_acc
        gap_penalty = max(0, 1.0 - gap * 3)

        return mean_acc * 0.7 + gap_penalty * 0.3


def list_profiles() -> List[str]:
    """List available scoring profile names."""
    return sorted(SCORING_PROFILES.keys())


def get_profile_weights(profile: str) -> Dict[str, float]:
    """Get weights for a scoring profile."""
    if profile not in SCORING_PROFILES:
        raise ValueError(f"Unknown profile '{profile}'. Available: {list(SCORING_PROFILES.keys())}")
    return SCORING_PROFILES[profile].copy()
