"""
Evaluation Pipeline for 10ways2do Benchmark Platform.

The top-level orchestrator that ties together:
- Model Adapter (connects to any AI model)
- Dynamic Generator (creates challenges)
- Process Evaluator (scores responses)
- Scoring Engine (computes radar profiles)
- Challenge Registry (tracks everything)

This is the single function you call to evaluate any model.
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from src.core.model_adapter import BaseModelAdapter, ModelResponse, create_adapter
from src.core.dynamic_generator import (
    ChallengeSet,
    generate_challenge_set,
    list_domains as list_challenge_domains,
)
from src.core.process_evaluator import ProcessEvaluator, DomainEvaluationReport
from src.core.scoring_engine import ScoringEngine, RadarProfile
from src.core.challenge_registry import ChallengeRegistry


SYSTEM_PROMPT = (
    "You are being evaluated by the 10ways2do AI benchmark. "
    "Answer each challenge accurately and concisely. "
    "Show your reasoning when solving problems. "
    "If the prompt asks for JSON output, respond with valid JSON only. "
    "If the prompt asks for a single value, respond with just that value."
)


def run_evaluation(
    provider: str,
    model_name: str,
    domains: Optional[List[str]] = None,
    n_challenges: int = 20,
    seed: Optional[int] = None,
    scoring_profile: str = "balanced",
    results_dir: str = "results",
    api_key: Optional[str] = None,
    api_base_url: Optional[str] = None,
    verbose: bool = True,
    **adapter_kwargs,
) -> Dict[str, Any]:
    """
    Run a full evaluation of a model against the 10ways2do benchmark.

    Args:
        provider: Model provider (gemini, perplexity, openai, together, mock, etc.)
        model_name: Model name (e.g., "gemini-2.5-flash", "Qwen/Qwen3-235B-A22B")
        domains: List of domains to evaluate (default: all available)
        n_challenges: Challenges per domain
        seed: Random seed (auto if None)
        scoring_profile: One of "balanced", "enterprise", "research", "safety"
        results_dir: Output directory
        api_key: API key (or set via environment variable)
        api_base_url: Custom API base URL
        verbose: Print progress
        **adapter_kwargs: Additional adapter config (temperature, max_tokens, etc.)

    Returns:
        Dict with evaluation results, radar profile, and metadata
    """
    if domains is None:
        domains = list_challenge_domains()

    if verbose:
        print("=" * 80)
        print("10WAYS2DO BENCHMARK EVALUATION")
        print("=" * 80)
        print(f"  Model:    {provider}/{model_name}")
        print(f"  Domains:  {', '.join(domains)}")
        print(f"  Challenges per domain: {n_challenges}")
        print(f"  Scoring profile: {scoring_profile}")
        print(f"  Seed: {seed or 'auto'}")
        print("=" * 80)

    # Initialize components
    adapter = create_adapter(
        provider=provider,
        model_name=model_name,
        api_key=api_key,
        api_base_url=api_base_url,
        **adapter_kwargs,
    )
    evaluator = ProcessEvaluator()
    engine = ScoringEngine(profile=scoring_profile)
    registry = ChallengeRegistry(results_dir=results_dir)

    model_id = f"{provider}/{model_name}"
    all_reports = []
    all_profiles = []
    total_start = time.time()

    for domain_idx, domain in enumerate(domains):
        if verbose:
            print(f"\n{'-' * 60}")
            print(f"  Domain {domain_idx + 1}/{len(domains)}: {domain}")
            print(f"{'-' * 60}")

        # 1. Generate challenges
        challenge_set = generate_challenge_set(
            domain=domain,
            n_challenges=n_challenges,
            seed=seed,
        )
        registry.register(challenge_set)

        if verbose:
            print(f"  Generated {len(challenge_set.challenges)} challenges "
                  f"(hash: {challenge_set.session_hash})")

        # 2. Run model against challenges
        responses = []
        for i, challenge in enumerate(challenge_set.challenges):
            if verbose and (i + 1) % 5 == 0:
                print(f"    Challenge {i + 1}/{len(challenge_set.challenges)}...")

            try:
                model_response = adapter.generate_with_tracking(
                    prompt=challenge.prompt,
                    system_prompt=SYSTEM_PROMPT,
                )
                responses.append({
                    "content": model_response.content,
                    "latency_ms": model_response.latency_ms,
                    "tokens_used": model_response.total_tokens,
                    "cost_usd": model_response.cost_usd,
                    "reasoning_trace": model_response.reasoning_trace,
                })
            except Exception as e:
                if verbose:
                    print(f"    [!] Challenge {i + 1} failed: {e}")
                responses.append({
                    "content": f"ERROR: {e}",
                    "latency_ms": 0.0,
                    "tokens_used": 0,
                    "cost_usd": 0.0,
                    "reasoning_trace": None,
                })

        # 3. Evaluate responses
        report = evaluator.evaluate_batch(
            challenges=challenge_set.challenges,
            responses=responses,
            model_id=model_id,
            session_id=challenge_set.session_id,
            domain=domain,
        )
        all_reports.append(report)

        # 4. Register evaluation
        registry.register_evaluation(
            session_id=challenge_set.session_id,
            model_id=model_id,
            report_dict=report.to_dict(),
        )

        if verbose:
            print(f"  [OK] Accuracy: {report.accuracy:.1%} "
                  f"({report.n_correct}/{report.n_total})")
            print(f"  [OK] Overall score: {report.mean_overall_score:.3f}")
            print(f"  [OK] Cost: ${report.total_cost_usd:.4f}")

    # 5. Compute radar profiles
    for i, report in enumerate(all_reports):
        other_reports = [r for j, r in enumerate(all_reports) if j != i]
        profile = engine.compute_profile(report, other_reports)
        all_profiles.append(profile)

    # 6. Compute aggregate profile
    if all_profiles:
        aggregate_profile = _aggregate_profiles(all_profiles, model_id, engine)
    else:
        aggregate_profile = RadarProfile(model_id=model_id)

    total_time = time.time() - total_start

    # 7. Save results
    output = _build_output(
        model_id=model_id,
        profiles=all_profiles,
        aggregate_profile=aggregate_profile,
        reports=all_reports,
        adapter_stats=adapter.get_session_stats(),
        total_time=total_time,
        scoring_profile=scoring_profile,
        seed=seed,
    )

    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)

    safe_name = model_name.replace("/", "_").replace("\\", "_")
    output_file = results_path / f"EVAL_{provider}_{safe_name}.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, default=str)

    registry.save()

    if verbose:
        print(f"\n{'=' * 80}")
        print(f"EVALUATION COMPLETE")
        print(f"{'=' * 80}")
        print(f"  Model: {model_id}")
        print(f"  Composite Score: {aggregate_profile.composite_score:.3f}")
        print(f"  Total Time: {total_time:.1f}s")
        print(f"  Total Cost: ${adapter.get_session_stats()['total_cost_usd']:.4f}")
        print(f"  Results: {output_file}")
        print()
        print("  Radar Profile:")
        for label, score in zip(aggregate_profile.axis_labels, aggregate_profile.scores_array):
            bar = "#" * int(score * 20) + "." * (20 - int(score * 20))
            print(f"    {label:>16s}  {bar}  {score:.3f}")
        print()

    return output


def _aggregate_profiles(
    profiles: List[RadarProfile],
    model_id: str,
    engine: ScoringEngine,
) -> RadarProfile:
    """Aggregate per-domain radar profiles into one overall profile."""
    from src.core.scoring_engine import AxisScore
    import numpy as np

    aggregate = RadarProfile(
        model_id=model_id,
        weights_used=engine.weights.copy(),
    )

    axes = ["accuracy", "speed", "cost", "robustness", "fairness", "consistency", "generalization", "efficiency"]

    for axis in axes:
        scores = [p.axes[axis].score for p in profiles if axis in p.axes]
        if scores:
            mean_score = float(np.mean(scores))
            aggregate.axes[axis] = AxisScore(
                axis=axis,
                score=mean_score,
                raw_value=mean_score,
                details={"per_domain_scores": {p.metadata.get("domain", "?"): p.axes[axis].score for p in profiles if axis in p.axes}},
            )

    aggregate.composite_score = sum(
        engine.weights.get(axis, 0) * aggregate.axes.get(axis, AxisScore(axis, 0, 0)).score
        for axis in axes
    )

    aggregate.metadata = {
        "type": "aggregate",
        "n_domains": len(profiles),
        "domains": [p.metadata.get("domain", "?") for p in profiles],
    }

    return aggregate


def _build_output(
    model_id: str,
    profiles: List[RadarProfile],
    aggregate_profile: RadarProfile,
    reports: List,
    adapter_stats: Dict,
    total_time: float,
    scoring_profile: str,
    seed: Optional[int],
) -> Dict[str, Any]:
    """Build the final output dict."""
    return {
        "benchmark": "10ways2do",
        "version": "2.0.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_id": model_id,
        "scoring_profile": scoring_profile,
        "seed": seed,
        "total_time_seconds": round(total_time, 2),
        "aggregate_profile": aggregate_profile.to_dict(),
        "leaderboard_row": aggregate_profile.to_leaderboard_row(),
        "per_domain": [
            {
                "domain": reports[i].domain,
                "report": reports[i].to_dict(),
                "profile": profiles[i].to_dict() if i < len(profiles) else None,
            }
            for i in range(len(reports))
        ],
        "adapter_stats": adapter_stats,
    }
