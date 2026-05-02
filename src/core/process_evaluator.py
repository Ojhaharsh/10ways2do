"""
Process Evaluator for 10ways2do Benchmark Platform.

Evaluates HOW a model reasons, not just what it outputs.
This is the key differentiator — most benchmarks only check final answers.

Evaluation dimensions:
1. Answer Correctness — Did it get the right answer?
2. Reasoning Quality — Is the reasoning chain sound?
3. Confidence Calibration — Does the model know what it knows?
4. Self-Correction — Can it catch and fix its own errors?
5. Format Compliance — Did it follow the output format instructions?
6. Efficiency — How many tokens/time did it take?
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np


@dataclass
class EvaluationResult:
    """Result of evaluating a single model response against a challenge."""

    challenge_id: str
    model_id: str

    # Scores (0.0 - 1.0)
    answer_score: float = 0.0
    reasoning_score: float = 0.0
    calibration_score: float = 0.0
    format_score: float = 0.0
    efficiency_score: float = 0.0

    # Composite
    overall_score: float = 0.0

    # Details
    answer_correct: bool = False
    partial_credit: float = 0.0
    reasoning_steps_detected: int = 0
    format_errors: List[str] = field(default_factory=list)

    # Cost
    latency_ms: float = 0.0
    tokens_used: int = 0
    cost_usd: float = 0.0

    # Metadata
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "challenge_id": self.challenge_id,
            "model_id": self.model_id,
            "answer_score": round(self.answer_score, 4),
            "reasoning_score": round(self.reasoning_score, 4),
            "calibration_score": round(self.calibration_score, 4),
            "format_score": round(self.format_score, 4),
            "efficiency_score": round(self.efficiency_score, 4),
            "overall_score": round(self.overall_score, 4),
            "answer_correct": self.answer_correct,
            "partial_credit": round(self.partial_credit, 4),
            "reasoning_steps_detected": self.reasoning_steps_detected,
            "format_errors": self.format_errors,
            "latency_ms": round(self.latency_ms, 2),
            "tokens_used": self.tokens_used,
            "cost_usd": round(self.cost_usd, 6),
            "details": self.details,
        }


@dataclass
class DomainEvaluationReport:
    """Aggregated evaluation report for a model across a challenge set."""

    model_id: str
    domain: str
    session_id: str

    # Per-challenge results
    results: List[EvaluationResult] = field(default_factory=list)

    # Aggregate scores
    mean_answer_score: float = 0.0
    mean_reasoning_score: float = 0.0
    mean_calibration_score: float = 0.0
    mean_format_score: float = 0.0
    mean_efficiency_score: float = 0.0
    mean_overall_score: float = 0.0

    # Breakdown by difficulty
    scores_by_difficulty: Dict[str, float] = field(default_factory=dict)

    # Cost summary
    total_latency_ms: float = 0.0
    total_tokens: int = 0
    total_cost_usd: float = 0.0

    # Statistics
    n_correct: int = 0
    n_total: int = 0
    accuracy: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_id": self.model_id,
            "domain": self.domain,
            "session_id": self.session_id,
            "n_challenges": self.n_total,
            "accuracy": round(self.accuracy, 4),
            "mean_scores": {
                "answer": round(self.mean_answer_score, 4),
                "reasoning": round(self.mean_reasoning_score, 4),
                "calibration": round(self.mean_calibration_score, 4),
                "format": round(self.mean_format_score, 4),
                "efficiency": round(self.mean_efficiency_score, 4),
                "overall": round(self.mean_overall_score, 4),
            },
            "scores_by_difficulty": {
                k: round(v, 4) for k, v in self.scores_by_difficulty.items()
            },
            "cost_summary": {
                "total_latency_ms": round(self.total_latency_ms, 2),
                "avg_latency_ms": round(self.total_latency_ms / max(1, self.n_total), 2),
                "total_tokens": self.total_tokens,
                "total_cost_usd": round(self.total_cost_usd, 6),
            },
            "n_correct": self.n_correct,
        }


# ──────────────────────────────────────────────────────────────────────────────
# Evaluator
# ──────────────────────────────────────────────────────────────────────────────

# Default weights for computing overall score
DEFAULT_WEIGHTS = {
    "answer": 0.45,
    "reasoning": 0.20,
    "calibration": 0.10,
    "format": 0.15,
    "efficiency": 0.10,
}


class ProcessEvaluator:
    """
    Evaluates model responses against challenges with multi-dimensional scoring.

    Usage:
        evaluator = ProcessEvaluator()
        result = evaluator.evaluate(challenge, model_response, model_id="gemini-2.5-pro")
    """

    def __init__(self, weights: Optional[Dict[str, float]] = None):
        self.weights = weights or DEFAULT_WEIGHTS.copy()
        # Normalize weights to sum to 1.0
        total = sum(self.weights.values())
        self.weights = {k: v / total for k, v in self.weights.items()}

    def evaluate(
        self,
        challenge,  # Challenge dataclass from dynamic_generator
        response_content: str,
        model_id: str = "unknown",
        latency_ms: float = 0.0,
        tokens_used: int = 0,
        cost_usd: float = 0.0,
        reasoning_trace: Optional[str] = None,
    ) -> EvaluationResult:
        """Evaluate a model's response against a challenge."""

        result = EvaluationResult(
            challenge_id=challenge.challenge_id,
            model_id=model_id,
            latency_ms=latency_ms,
            tokens_used=tokens_used,
            cost_usd=cost_usd,
        )

        # 1. Answer correctness
        answer_score, is_correct, partial = self._evaluate_answer(
            response_content, challenge.expected_answer, challenge.evaluation_rubric
        )
        result.answer_score = answer_score
        result.answer_correct = is_correct
        result.partial_credit = partial

        # 2. Reasoning quality
        result.reasoning_score, result.reasoning_steps_detected = self._evaluate_reasoning(
            response_content, reasoning_trace
        )

        # 3. Confidence calibration
        result.calibration_score = self._evaluate_calibration(
            response_content, is_correct
        )

        # 4. Format compliance
        result.format_score, result.format_errors = self._evaluate_format(
            response_content, challenge.expected_answer
        )

        # 5. Efficiency
        result.efficiency_score = self._evaluate_efficiency(
            latency_ms, tokens_used, challenge.difficulty.value
        )

        # Composite score
        result.overall_score = (
            self.weights["answer"] * result.answer_score
            + self.weights["reasoning"] * result.reasoning_score
            + self.weights["calibration"] * result.calibration_score
            + self.weights["format"] * result.format_score
            + self.weights["efficiency"] * result.efficiency_score
        )

        return result

    def evaluate_batch(
        self,
        challenges: list,
        responses: List[Dict[str, Any]],
        model_id: str = "unknown",
        session_id: str = "",
        domain: str = "",
    ) -> DomainEvaluationReport:
        """
        Evaluate a batch of responses and produce an aggregated report.

        Args:
            challenges: List of Challenge objects
            responses: List of dicts with keys: content, latency_ms, tokens_used, cost_usd, reasoning_trace
            model_id: Model identifier
            session_id: Challenge session ID
            domain: Domain name
        """
        report = DomainEvaluationReport(
            model_id=model_id,
            domain=domain,
            session_id=session_id,
        )

        by_difficulty = {}

        for challenge, resp in zip(challenges, responses):
            result = self.evaluate(
                challenge=challenge,
                response_content=resp.get("content", ""),
                model_id=model_id,
                latency_ms=resp.get("latency_ms", 0.0),
                tokens_used=resp.get("tokens_used", 0),
                cost_usd=resp.get("cost_usd", 0.0),
                reasoning_trace=resp.get("reasoning_trace"),
            )
            report.results.append(result)

            diff = challenge.difficulty.value
            if diff not in by_difficulty:
                by_difficulty[diff] = []
            by_difficulty[diff].append(result.overall_score)

        n = len(report.results)
        report.n_total = n

        if n > 0:
            report.mean_answer_score = np.mean([r.answer_score for r in report.results])
            report.mean_reasoning_score = np.mean([r.reasoning_score for r in report.results])
            report.mean_calibration_score = np.mean([r.calibration_score for r in report.results])
            report.mean_format_score = np.mean([r.format_score for r in report.results])
            report.mean_efficiency_score = np.mean([r.efficiency_score for r in report.results])
            report.mean_overall_score = np.mean([r.overall_score for r in report.results])

            report.n_correct = sum(1 for r in report.results if r.answer_correct)
            report.accuracy = report.n_correct / n

            report.total_latency_ms = sum(r.latency_ms for r in report.results)
            report.total_tokens = sum(r.tokens_used for r in report.results)
            report.total_cost_usd = sum(r.cost_usd for r in report.results)

            for diff, scores in by_difficulty.items():
                report.scores_by_difficulty[diff] = float(np.mean(scores))

        return report

    # ──────────────────────────────────────────────────────────────────────
    # Evaluation dimension implementations
    # ──────────────────────────────────────────────────────────────────────

    def _evaluate_answer(
        self,
        response: str,
        expected: Any,
        rubric: Dict[str, Any],
    ) -> Tuple[float, bool, float]:
        """Evaluate answer correctness. Returns (score, is_correct, partial_credit)."""
        response_clean = response.strip()

        if isinstance(expected, dict):
            return self._evaluate_json_answer(response_clean, expected, rubric)
        elif isinstance(expected, list):
            return self._evaluate_list_answer(response_clean, expected, rubric)
        elif isinstance(expected, (int, float)):
            return self._evaluate_numeric_answer(response_clean, expected, rubric)
        else:
            return self._evaluate_string_answer(response_clean, str(expected), rubric)

    def _evaluate_json_answer(
        self, response: str, expected: Dict, rubric: Dict
    ) -> Tuple[float, bool, float]:
        """Evaluate JSON-structured answers."""
        try:
            # Try to extract JSON from response
            parsed = self._extract_json(response)
            if parsed is None:
                return 0.0, False, 0.0

            # Field-level scoring
            total_fields = len(expected)
            matches = 0
            partial = 0.0

            for key, expected_val in expected.items():
                actual_val = parsed.get(key, "")
                if str(actual_val).strip().lower() == str(expected_val).strip().lower():
                    matches += 1
                    partial += 1.0
                elif str(expected_val).lower() in str(actual_val).lower():
                    partial += 0.5

            exact_ratio = matches / max(1, total_fields)
            partial_ratio = partial / max(1, total_fields)

            exact_w = rubric.get("exact_match_weight", 0.5)
            partial_w = rubric.get("partial_match_weight", 0.3)
            format_w = rubric.get("format_compliance_weight", 0.2)

            score = exact_w * exact_ratio + partial_w * partial_ratio + format_w * 1.0
            is_correct = matches == total_fields

            return min(1.0, score), is_correct, partial_ratio

        except Exception:
            return 0.0, False, 0.0

    def _evaluate_list_answer(
        self, response: str, expected: List, rubric: Dict
    ) -> Tuple[float, bool, float]:
        """Evaluate list-type answers (e.g., anomaly indices)."""
        try:
            parsed = self._extract_json(response)
            if parsed is None or not isinstance(parsed, list):
                return 0.0, False, 0.0

            expected_set = set(expected)
            actual_set = set(parsed)

            if len(expected_set) == 0 and len(actual_set) == 0:
                return 1.0, True, 1.0

            tp = len(expected_set & actual_set)
            precision = tp / max(1, len(actual_set))
            recall = tp / max(1, len(expected_set))

            prec_w = rubric.get("precision_weight", 0.4)
            rec_w = rubric.get("recall_weight", 0.4)
            fmt_w = rubric.get("format_compliance_weight", 0.2)

            score = prec_w * precision + rec_w * recall + fmt_w * 1.0
            is_correct = expected_set == actual_set
            partial = (precision + recall) / 2

            return min(1.0, score), is_correct, partial

        except Exception:
            return 0.0, False, 0.0

    def _evaluate_numeric_answer(
        self, response: str, expected: Union[int, float], rubric: Dict
    ) -> Tuple[float, bool, float]:
        """Evaluate numeric answers with tolerance."""
        try:
            # Extract numbers from response
            numbers = re.findall(r'-?\d+\.?\d*', response)
            if not numbers:
                return 0.0, False, 0.0

            # Try last number (usually the final answer)
            actual = float(numbers[-1])
            tolerance = rubric.get("numeric_tolerance", 0.01)

            if abs(expected) < 1e-10:
                error = abs(actual - expected)
            else:
                error = abs(actual - expected) / abs(expected)

            if error <= tolerance:
                return 1.0, True, 1.0
            elif error <= tolerance * 5:
                partial = max(0, 1.0 - error / (tolerance * 5))
                return partial * 0.7, False, partial
            else:
                return 0.0, False, 0.0

        except Exception:
            return 0.0, False, 0.0

    def _evaluate_string_answer(
        self, response: str, expected: str, rubric: Dict
    ) -> Tuple[float, bool, float]:
        """Evaluate string answers with fuzzy matching."""
        response_lower = response.lower().strip()
        expected_lower = expected.lower().strip()

        # Exact match
        if response_lower == expected_lower:
            return 1.0, True, 1.0

        # Check if expected is contained in response (model may add extra text)
        if expected_lower in response_lower:
            return 0.9, True, 0.9

        # Word overlap (Jaccard)
        response_words = set(response_lower.split())
        expected_words = set(expected_lower.split())
        if response_words and expected_words:
            jaccard = len(response_words & expected_words) / len(response_words | expected_words)
            if jaccard > 0.8:
                return jaccard * 0.8, False, jaccard
            elif jaccard > 0.5:
                return jaccard * 0.5, False, jaccard

        return 0.0, False, 0.0

    def _evaluate_reasoning(
        self, response: str, reasoning_trace: Optional[str] = None
    ) -> Tuple[float, int]:
        """
        Evaluate reasoning quality.
        Returns (score, n_steps_detected).
        """
        text = reasoning_trace or response

        # Detect reasoning patterns
        step_markers = [
            r'(?:step|first|second|third|then|next|finally|therefore|thus|hence|because|since)\b',
            r'(?:let me|let\'s|we can|we need|we know|consider|suppose|assume)\b',
            r'(?:\d+[\.\)]\s)',  # Numbered steps
            r'(?:→|=>|->)',  # Arrows
        ]

        steps = 0
        for pattern in step_markers:
            matches = re.findall(pattern, text.lower())
            steps += len(matches)

        # Heuristic quality scoring
        if steps == 0:
            score = 0.3  # Gave answer but no reasoning
        elif steps <= 2:
            score = 0.5
        elif steps <= 5:
            score = 0.7
        elif steps <= 10:
            score = 0.85
        else:
            score = 0.95

        # Bonus for showing work (mathematical expressions)
        math_patterns = re.findall(r'[\d\.\,]+\s*[\+\-\*\/\=]\s*[\d\.\,]+', text)
        if math_patterns:
            score = min(1.0, score + 0.05 * len(math_patterns))

        return score, steps

    def _evaluate_calibration(self, response: str, is_correct: bool) -> float:
        """
        Evaluate confidence calibration.
        Good calibration = confident when right, uncertain when wrong.
        """
        # Detect confidence signals
        high_confidence = any(w in response.lower() for w in [
            "definitely", "certainly", "clearly", "obviously", "no doubt",
            "i'm sure", "i am sure", "i'm confident", "100%",
        ])
        low_confidence = any(w in response.lower() for w in [
            "maybe", "perhaps", "might", "possibly", "not sure",
            "i think", "uncertain", "approximately", "roughly",
        ])
        hedging = any(w in response.lower() for w in [
            "however", "but", "although", "on the other hand", "alternatively",
        ])

        if is_correct:
            if high_confidence:
                return 1.0  # Correctly confident
            elif low_confidence:
                return 0.5  # Right but unsure (underconfident)
            else:
                return 0.7  # Neutral
        else:
            if high_confidence:
                return 0.1  # Wrong and confident (dangerous)
            elif low_confidence:
                return 0.8  # Wrong but expressed uncertainty (good calibration)
            elif hedging:
                return 0.6  # Wrong with hedging (okay)
            else:
                return 0.4  # Wrong without caveats

    def _evaluate_format(self, response: str, expected: Any) -> Tuple[float, List[str]]:
        """Evaluate format compliance."""
        errors = []

        if isinstance(expected, (dict, list)):
            parsed = self._extract_json(response)
            if parsed is None:
                errors.append("Expected JSON output but could not parse")
                return 0.3, errors
            if isinstance(expected, dict) and not isinstance(parsed, dict):
                errors.append(f"Expected JSON object, got {type(parsed).__name__}")
                return 0.5, errors
            if isinstance(expected, list) and not isinstance(parsed, list):
                errors.append(f"Expected JSON array, got {type(parsed).__name__}")
                return 0.5, errors
            return 1.0, errors

        # For string/numeric answers, check for excess verbosity
        response_clean = response.strip()
        if len(response_clean) > 500 and len(str(expected)) < 50:
            errors.append("Response much longer than expected answer")
            return 0.7, errors

        return 1.0, errors

    def _evaluate_efficiency(
        self, latency_ms: float, tokens_used: int, difficulty: str
    ) -> float:
        """
        Evaluate efficiency relative to difficulty.
        Faster/fewer tokens for easy problems = better.
        """
        # Expected latency budgets per difficulty (in ms)
        budgets = {
            "trivial": 3000,
            "easy": 5000,
            "medium": 10000,
            "hard": 20000,
            "expert": 45000,
        }
        budget = budgets.get(difficulty, 10000)

        if latency_ms <= 0:
            return 0.5  # Unknown latency

        if latency_ms <= budget * 0.3:
            return 1.0  # Excellent
        elif latency_ms <= budget:
            return 0.8  # Good
        elif latency_ms <= budget * 2:
            return 0.5  # Acceptable
        else:
            return 0.2  # Slow

    # ──────────────────────────────────────────────────────────────────────
    # Utilities
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    def _extract_json(text: str) -> Any:
        """Try to extract JSON from a model response."""
        # Try direct parse
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Try extracting from markdown code block
        json_block = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL)
        if json_block:
            try:
                return json.loads(json_block.group(1))
            except json.JSONDecodeError:
                pass

        # Try finding JSON-like content
        for pattern in [r'\{[^{}]*\}', r'\[[^\[\]]*\]']:
            match = re.search(pattern, text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group())
                except json.JSONDecodeError:
                    pass

        return None
