"""
Challenge Registry for 10ways2do Benchmark Platform.

Provides versioned tracking of generated challenge sets with
cryptographic hashes for reproducibility and anti-contamination.

Features:
- Every challenge set gets a unique ID and SHA-256 hash
- Challenge sets can be replayed via hash for reproducibility
- Registry is stored locally as JSON for persistence
- Anti-contamination: challenges generated fresh, never published as a static dataset
"""

from __future__ import annotations

import json
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


REGISTRY_VERSION = "1.0.0"


class ChallengeRegistry:
    """
    Persistent registry for tracking all generated challenge sets.

    Usage:
        registry = ChallengeRegistry(results_dir="results")
        registry.register(challenge_set)
        registry.save()

        # Later: replay a session
        session = registry.get_session("session_id")
    """

    def __init__(self, results_dir: str = "results"):
        self.results_dir = Path(results_dir)
        self.registry_path = self.results_dir / "challenge_registry.json"
        self._sessions: Dict[str, Dict[str, Any]] = {}
        self._evaluations: Dict[str, List[Dict[str, Any]]] = {}
        self._load()

    def _load(self):
        """Load existing registry from disk."""
        if self.registry_path.exists():
            try:
                with open(self.registry_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self._sessions = data.get("sessions", {})
                    self._evaluations = data.get("evaluations", {})
            except (json.JSONDecodeError, KeyError):
                self._sessions = {}
                self._evaluations = {}

    def save(self):
        """Persist registry to disk."""
        self.results_dir.mkdir(parents=True, exist_ok=True)
        data = {
            "registry_version": REGISTRY_VERSION,
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
            "n_sessions": len(self._sessions),
            "n_evaluations": sum(len(v) for v in self._evaluations.values()),
            "sessions": self._sessions,
            "evaluations": self._evaluations,
        }
        with open(self.registry_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)

    def register(self, challenge_set) -> str:
        """
        Register a generated challenge set.

        Args:
            challenge_set: ChallengeSet from dynamic_generator

        Returns:
            session_id
        """
        session_data = {
            "session_id": challenge_set.session_id,
            "session_hash": challenge_set.session_hash,
            "domain": challenge_set.domain,
            "seed": challenge_set.seed,
            "n_challenges": len(challenge_set.challenges),
            "difficulty_distribution": challenge_set.difficulty_distribution,
            "generator_version": challenge_set.generator_version,
            "generated_at_utc": challenge_set.generated_at_utc,
            "registered_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        self._sessions[challenge_set.session_id] = session_data
        return challenge_set.session_id

    def register_evaluation(
        self,
        session_id: str,
        model_id: str,
        report_dict: Dict[str, Any],
    ) -> None:
        """Register an evaluation result against a session."""
        eval_record = {
            "session_id": session_id,
            "model_id": model_id,
            "evaluated_at_utc": datetime.now(timezone.utc).isoformat(),
            "summary": {
                "accuracy": report_dict.get("accuracy", 0),
                "mean_overall": report_dict.get("mean_scores", {}).get("overall", 0),
                "total_cost_usd": report_dict.get("cost_summary", {}).get("total_cost_usd", 0),
                "n_challenges": report_dict.get("n_challenges", 0),
            },
        }

        if session_id not in self._evaluations:
            self._evaluations[session_id] = []
        self._evaluations[session_id].append(eval_record)

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session metadata by ID."""
        return self._sessions.get(session_id)

    def get_evaluations(self, session_id: str) -> List[Dict[str, Any]]:
        """Get all evaluations for a session."""
        return self._evaluations.get(session_id, [])

    def list_sessions(
        self,
        domain: Optional[str] = None,
        limit: int = 50,
    ) -> List[Dict[str, Any]]:
        """List registered sessions, optionally filtered by domain."""
        sessions = list(self._sessions.values())
        if domain:
            sessions = [s for s in sessions if s.get("domain") == domain]

        # Sort by registration time (most recent first)
        sessions.sort(key=lambda s: s.get("registered_at_utc", ""), reverse=True)
        return sessions[:limit]

    def list_model_results(self, model_id: str) -> List[Dict[str, Any]]:
        """Get all evaluation results for a specific model."""
        results = []
        for session_id, evals in self._evaluations.items():
            for ev in evals:
                if ev.get("model_id") == model_id:
                    ev_copy = ev.copy()
                    ev_copy["session_meta"] = self._sessions.get(session_id, {})
                    results.append(ev_copy)
        results.sort(key=lambda r: r.get("evaluated_at_utc", ""), reverse=True)
        return results

    def get_leaderboard_data(self, domain: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get the best result per model for leaderboard display.

        Returns list of {model_id, best_score, domain, session_id, ...}
        """
        best_by_model: Dict[str, Dict[str, Any]] = {}

        for session_id, evals in self._evaluations.items():
            session = self._sessions.get(session_id, {})
            if domain and session.get("domain") != domain:
                continue

            for ev in evals:
                mid = ev.get("model_id", "unknown")
                score = ev.get("summary", {}).get("mean_overall", 0)

                if mid not in best_by_model or score > best_by_model[mid].get("best_score", 0):
                    best_by_model[mid] = {
                        "model_id": mid,
                        "best_score": score,
                        "accuracy": ev.get("summary", {}).get("accuracy", 0),
                        "domain": session.get("domain", ""),
                        "session_id": session_id,
                        "evaluated_at": ev.get("evaluated_at_utc", ""),
                        "total_cost_usd": ev.get("summary", {}).get("total_cost_usd", 0),
                    }

        rows = list(best_by_model.values())
        rows.sort(key=lambda r: r["best_score"], reverse=True)
        return rows

    @property
    def stats(self) -> Dict[str, Any]:
        """Quick summary statistics."""
        unique_models = set()
        for evals in self._evaluations.values():
            for ev in evals:
                unique_models.add(ev.get("model_id", "unknown"))

        return {
            "n_sessions": len(self._sessions),
            "n_evaluations": sum(len(v) for v in self._evaluations.values()),
            "n_unique_models": len(unique_models),
            "domains": list(set(s.get("domain", "") for s in self._sessions.values())),
        }
