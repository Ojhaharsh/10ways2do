"""
Dynamic Challenge Generator for 10ways2do Benchmark Platform.

The contamination-proof core: generates unique, parameterized challenges
at evaluation time so no model can pre-train on the test set.

Each challenge:
- Is procedurally generated from parameterized templates
- Gets a unique cryptographic hash for reproducibility
- Can be replayed (same hash = same challenge) but not predicted
- Spans difficulty levels from trivial to expert

Supported challenge types per domain class:
- information_extraction: Entity/relation extraction from synthetic docs
- anomaly_detection: Pattern recognition in generated sequences
- reasoning: Multi-step logical and mathematical reasoning
- code_generation: Programming tasks with test cases
- classification: Categorization with controlled difficulty
- forecasting: Prediction from generated time series
"""

from __future__ import annotations

import hashlib
import json
import random
import string
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum

import numpy as np


class ChallengeDifficulty(Enum):
    TRIVIAL = "trivial"
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"
    EXPERT = "expert"


@dataclass
class Challenge:
    """A single evaluation challenge."""

    challenge_id: str
    domain: str
    challenge_type: str
    difficulty: ChallengeDifficulty
    prompt: str
    expected_answer: Any
    metadata: Dict[str, Any] = field(default_factory=dict)
    evaluation_rubric: Dict[str, Any] = field(default_factory=dict)
    max_score: float = 1.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "challenge_id": self.challenge_id,
            "domain": self.domain,
            "challenge_type": self.challenge_type,
            "difficulty": self.difficulty.value,
            "prompt": self.prompt,
            "expected_answer": self.expected_answer,
            "metadata": self.metadata,
            "evaluation_rubric": self.evaluation_rubric,
            "max_score": self.max_score,
        }


@dataclass
class ChallengeSet:
    """A set of challenges for a single evaluation session."""

    session_id: str
    session_hash: str
    challenges: List[Challenge]
    generated_at_utc: str
    generator_version: str
    domain: str
    seed: int
    difficulty_distribution: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "session_id": self.session_id,
            "session_hash": self.session_hash,
            "generated_at_utc": self.generated_at_utc,
            "generator_version": self.generator_version,
            "domain": self.domain,
            "seed": self.seed,
            "n_challenges": len(self.challenges),
            "difficulty_distribution": self.difficulty_distribution,
            "challenges": [c.to_dict() for c in self.challenges],
        }


# ──────────────────────────────────────────────────────────────────────────────
# Challenge Generators (one per challenge type)
# ──────────────────────────────────────────────────────────────────────────────

class ReasoningChallengeGenerator:
    """Generates multi-step reasoning challenges."""

    TEMPLATES = {
        ChallengeDifficulty.TRIVIAL: [
            {
                "template": "What is {a} + {b}?",
                "gen": lambda rng: {"a": rng.randint(1, 20), "b": rng.randint(1, 20)},
                "answer": lambda p: str(p["a"] + p["b"]),
            },
            {
                "template": "If a store has {a} apples and sells {b}, how many remain?",
                "gen": lambda rng: {"a": (x := rng.randint(10, 50)), "b": rng.randint(1, x)},
                "answer": lambda p: str(p["a"] - p["b"]),
            },
        ],
        ChallengeDifficulty.EASY: [
            {
                "template": "A train travels {speed} km/h for {hours} hours. How far does it go in kilometers?",
                "gen": lambda rng: {"speed": rng.randint(40, 200), "hours": rng.randint(1, 8)},
                "answer": lambda p: str(p["speed"] * p["hours"]),
            },
            {
                "template": "What is {a}% of {b}?",
                "gen": lambda rng: {"a": rng.choice([10, 15, 20, 25, 50, 75]), "b": rng.randint(100, 1000)},
                "answer": lambda p: str(round(p["a"] * p["b"] / 100, 2)),
            },
        ],
        ChallengeDifficulty.MEDIUM: [
            {
                "template": (
                    "A company's revenue was ${rev}M in Q1 and grew by {g1}% in Q2 and {g2}% in Q3. "
                    "What was the Q3 revenue in millions of dollars? Give just the number rounded to 2 decimal places."
                ),
                "gen": lambda rng: {
                    "rev": rng.randint(10, 500),
                    "g1": rng.randint(5, 30),
                    "g2": rng.randint(-10, 40),
                },
                "answer": lambda p: str(round(p["rev"] * (1 + p["g1"] / 100) * (1 + p["g2"] / 100), 2)),
            },
            {
                "template": (
                    "There are {n} teams in a tournament. Each team plays every other team exactly once. "
                    "How many total games are played?"
                ),
                "gen": lambda rng: {"n": rng.randint(4, 20)},
                "answer": lambda p: str(p["n"] * (p["n"] - 1) // 2),
            },
        ],
        ChallengeDifficulty.HARD: [
            {
                "template": (
                    "A system has {n} servers. Each server fails independently with probability {p}. "
                    "The system uses {k}-of-{n} redundancy (needs at least {k} servers working). "
                    "What is the probability the system is operational? Give the answer as a decimal rounded to 4 places."
                ),
                "gen": lambda rng: {
                    "n": (n := rng.randint(3, 7)),
                    "p": round(rng.uniform(0.01, 0.15), 2),
                    "k": rng.randint(1, n),
                },
                "answer": lambda p: str(round(
                    sum(
                        _comb(p["n"], i) * ((1 - p["p"]) ** i) * (p["p"] ** (p["n"] - i))
                        for i in range(p["k"], p["n"] + 1)
                    ), 4
                )),
            },
        ],
        ChallengeDifficulty.EXPERT: [
            {
                "template": (
                    "Consider a Markov chain with {n} states and the following transition matrix:\n{matrix}\n"
                    "What is the stationary distribution? Give probabilities for each state rounded to 3 decimal places, "
                    "separated by commas."
                ),
                "gen": lambda rng: _gen_markov_params(rng),
                "answer": lambda p: _solve_stationary(p["raw_matrix"]),
            },
        ],
    }

    def generate(self, rng: random.Random, difficulty: ChallengeDifficulty) -> Challenge:
        templates = self.TEMPLATES.get(difficulty, self.TEMPLATES[ChallengeDifficulty.MEDIUM])
        template_spec = rng.choice(templates)

        params = template_spec["gen"](rng)
        prompt = template_spec["template"].format(**params)
        answer = template_spec["answer"](params)

        challenge_id = _make_challenge_id("reasoning", prompt, answer)

        return Challenge(
            challenge_id=challenge_id,
            domain="reasoning",
            challenge_type="multi_step_reasoning",
            difficulty=difficulty,
            prompt=prompt,
            expected_answer=answer,
            metadata={"params": {k: v for k, v in params.items() if k != "raw_matrix"}},
            evaluation_rubric={
                "exact_match_weight": 0.7,
                "numeric_tolerance": 0.01,
                "reasoning_quality_weight": 0.3,
            },
        )


class InformationExtractionChallengeGenerator:
    """Generates entity extraction challenges from synthetic text."""

    ENTITY_TYPES = ["person", "organization", "location", "date", "amount", "product"]

    FIRST_NAMES = [
        "Alice", "Bob", "Carol", "David", "Eva", "Frank", "Grace", "Henry",
        "Iris", "Jack", "Karen", "Leo", "Maya", "Noah", "Olivia", "Peter",
        "Quinn", "Rachel", "Sam", "Tina", "Uma", "Victor", "Wendy", "Xander",
    ]
    LAST_NAMES = [
        "Anderson", "Brown", "Chen", "Davis", "Edwards", "Fischer", "Garcia",
        "Harris", "Ibrahim", "Johnson", "Kim", "Li", "Martinez", "Nakamura",
        "O'Brien", "Patel", "Quinn", "Robinson", "Singh", "Tanaka",
    ]
    COMPANIES = [
        "Nextera Systems", "Vortex Labs", "Pinnacle Corp", "Quantum Dynamics",
        "Evergreen Solutions", "Atlas Engineering", "Prism Analytics",
        "Horizon Technologies", "Meridian Group", "Catalyst Ventures",
    ]
    CITIES = [
        "San Francisco", "London", "Tokyo", "Berlin", "Sydney",
        "Toronto", "Mumbai", "São Paulo", "Singapore", "Dubai",
    ]
    PRODUCTS = [
        "DataStream Pro", "CloudVault Enterprise", "NeuralEdge Platform",
        "SecureSync Gateway", "QuantumCore SDK", "InsightFlow Dashboard",
    ]

    TEMPLATES = [
        (
            "{name}, CEO of {company}, announced a ${amount}M deal in {city} on {date}. "
            "The transaction involves the acquisition of {product}."
        ),
        (
            "According to {name} from {company}, the {product} platform processed "
            "${amount}M in transactions last quarter. The announcement was made in {city} on {date}."
        ),
        (
            "In {city}, {company} reported revenue of ${amount}M for the fiscal year ending {date}. "
            "{name}, the CFO, credited the success to the {product} initiative."
        ),
    ]

    def generate(self, rng: random.Random, difficulty: ChallengeDifficulty) -> Challenge:
        name = f"{rng.choice(self.FIRST_NAMES)} {rng.choice(self.LAST_NAMES)}"
        company = rng.choice(self.COMPANIES)
        city = rng.choice(self.CITIES)
        product = rng.choice(self.PRODUCTS)
        amount = round(rng.uniform(1.5, 999.9), 1)

        year = rng.randint(2023, 2026)
        month = rng.randint(1, 12)
        day = rng.randint(1, 28)
        date_str = f"{year}-{month:02d}-{day:02d}"

        template = rng.choice(self.TEMPLATES)
        text = template.format(
            name=name, company=company, city=city,
            product=product, amount=amount, date=date_str,
        )

        # Add noise for harder difficulties
        if difficulty in (ChallengeDifficulty.HARD, ChallengeDifficulty.EXPERT):
            noise_sentences = [
                f"Market analysts remain cautious about the sector.",
                f"The technology industry has seen significant shifts recently.",
                f"Several competitors are expected to respond with similar announcements.",
                f"Industry experts noted the timing was strategic.",
            ]
            insert_pos = rng.randint(0, 1)
            noise = rng.choice(noise_sentences)
            parts = text.split(". ")
            if insert_pos < len(parts):
                parts.insert(insert_pos + 1, noise)
            text = ". ".join(parts)

        expected = {
            "person": name,
            "organization": company,
            "location": city,
            "date": date_str,
            "amount": f"${amount}M",
            "product": product,
        }

        prompt = (
            f"Extract the following entities from the text below. "
            f"Return a JSON object with keys: person, organization, location, date, amount, product.\n\n"
            f"Text: {text}"
        )

        challenge_id = _make_challenge_id("ie", text, json.dumps(expected))

        return Challenge(
            challenge_id=challenge_id,
            domain="information_extraction",
            challenge_type="entity_extraction",
            difficulty=difficulty,
            prompt=prompt,
            expected_answer=expected,
            metadata={"source_text": text},
            evaluation_rubric={
                "exact_match_weight": 0.5,
                "partial_match_weight": 0.3,
                "format_compliance_weight": 0.2,
            },
        )


class AnomalyDetectionChallengeGenerator:
    """Generates anomaly detection challenges from synthetic sequences."""

    def generate(self, rng: random.Random, difficulty: ChallengeDifficulty) -> Challenge:
        np_rng = np.random.RandomState(rng.randint(0, 2**31))

        difficulty_params = {
            ChallengeDifficulty.TRIVIAL: {"n": 20, "anomaly_magnitude": 10.0},
            ChallengeDifficulty.EASY: {"n": 30, "anomaly_magnitude": 5.0},
            ChallengeDifficulty.MEDIUM: {"n": 50, "anomaly_magnitude": 3.0},
            ChallengeDifficulty.HARD: {"n": 80, "anomaly_magnitude": 2.0},
            ChallengeDifficulty.EXPERT: {"n": 100, "anomaly_magnitude": 1.5},
        }

        params = difficulty_params[difficulty]
        n = params["n"]
        mag = params["anomaly_magnitude"]

        # Generate normal sequence
        base = np_rng.normal(100, 10, n).tolist()

        # Insert anomalies
        n_anomalies = max(1, n // 10)
        anomaly_indices = sorted(rng.sample(range(n), n_anomalies))
        for idx in anomaly_indices:
            base[idx] += rng.choice([-1, 1]) * mag * 10

        sequence_str = ", ".join([f"{v:.1f}" for v in base])
        expected = sorted(anomaly_indices)

        prompt = (
            f"The following is a sequence of {n} measurements. "
            f"Identify the indices (0-based) of anomalous values. "
            f"Return a JSON array of indices.\n\n"
            f"Sequence: [{sequence_str}]"
        )

        challenge_id = _make_challenge_id("anomaly", sequence_str[:100], str(expected))

        return Challenge(
            challenge_id=challenge_id,
            domain="anomaly_detection",
            challenge_type="sequence_anomaly",
            difficulty=difficulty,
            prompt=prompt,
            expected_answer=expected,
            metadata={"n_points": n, "n_anomalies": n_anomalies, "anomaly_indices": expected},
            evaluation_rubric={
                "precision_weight": 0.4,
                "recall_weight": 0.4,
                "format_compliance_weight": 0.2,
            },
        )


class ClassificationChallengeGenerator:
    """Generates text classification challenges."""

    CATEGORIES = {
        "sentiment": {
            "positive": [
                "Absolutely loved {thing}! Best {category} I've ever {action}.",
                "Outstanding {thing}. The {feature} is remarkable and the {aspect} exceeded expectations.",
                "{thing} is phenomenal. Would recommend to anyone looking for quality {category}.",
            ],
            "negative": [
                "Terrible {thing}. Complete waste of {resource}. The {feature} was awful.",
                "Extremely disappointed with {thing}. The {aspect} was unacceptable.",
                "Would not recommend {thing} to anyone. Worst {category} experience ever.",
            ],
            "neutral": [
                "{thing} was okay. The {feature} was average and the {aspect} was nothing special.",
                "Decent {thing} for the {resource}. Neither great nor terrible.",
                "{thing} met basic expectations. The {category} could be better but works fine.",
            ],
        }
    }

    FILL_WORDS = {
        "thing": ["this product", "this service", "the experience", "the offering"],
        "category": ["service", "product", "solution", "platform"],
        "action": ["experienced", "used", "tried", "purchased"],
        "feature": ["quality", "design", "performance", "interface"],
        "aspect": ["customer support", "delivery", "value", "reliability"],
        "resource": ["money", "time", "effort", "investment"],
    }

    def generate(self, rng: random.Random, difficulty: ChallengeDifficulty) -> Challenge:
        task_type = "sentiment"
        categories = self.CATEGORIES[task_type]

        label = rng.choice(list(categories.keys()))
        template = rng.choice(categories[label])

        fills = {k: rng.choice(v) for k, v in self.FILL_WORDS.items()}
        text = template.format(**fills)

        # Add ambiguity for harder difficulties
        if difficulty in (ChallengeDifficulty.HARD, ChallengeDifficulty.EXPERT):
            contrasting = rng.choice([l for l in categories.keys() if l != label])
            contra_template = rng.choice(categories[contrasting])
            contra_text = contra_template.format(**{k: rng.choice(v) for k, v in self.FILL_WORDS.items()})
            text = f"{text} However, {contra_text.lower()}"

        prompt = (
            f"Classify the following text as 'positive', 'negative', or 'neutral'.\n\n"
            f"Text: \"{text}\"\n\n"
            f"Reply with only the classification label."
        )

        challenge_id = _make_challenge_id("classification", text, label)

        return Challenge(
            challenge_id=challenge_id,
            domain="classification",
            challenge_type="sentiment_analysis",
            difficulty=difficulty,
            prompt=prompt,
            expected_answer=label,
            metadata={"task_type": task_type, "source_text": text},
            evaluation_rubric={
                "exact_match_weight": 1.0,
            },
        )


# ──────────────────────────────────────────────────────────────────────────────
# Master Generator
# ──────────────────────────────────────────────────────────────────────────────

# Available domain generators
_DOMAIN_GENERATORS = {
    "reasoning": ReasoningChallengeGenerator,
    "information_extraction": InformationExtractionChallengeGenerator,
    "anomaly_detection": AnomalyDetectionChallengeGenerator,
    "classification": ClassificationChallengeGenerator,
}

# Dynamically register extended generators (deferred to avoid circular import)
_EXTENDED_LOADED = False

def _ensure_extended_generators():
    """Lazily load extended generators to avoid circular imports."""
    global _EXTENDED_LOADED
    if _EXTENDED_LOADED:
        return
    _EXTENDED_LOADED = True
    try:
        from src.core.extended_generators import (
            CodeGenerationChallengeGenerator,
            InstructionFollowingChallengeGenerator,
            LogicalDeductionChallengeGenerator,
            HallucinationDetectionChallengeGenerator,
            SummarizationChallengeGenerator,
            ToolUsePlanningChallengeGenerator,
        )
        _DOMAIN_GENERATORS.update({
            "code_generation": CodeGenerationChallengeGenerator,
            "instruction_following": InstructionFollowingChallengeGenerator,
            "logical_deduction": LogicalDeductionChallengeGenerator,
            "hallucination_resistance": HallucinationDetectionChallengeGenerator,
            "summarization": SummarizationChallengeGenerator,
            "tool_use_planning": ToolUsePlanningChallengeGenerator,
        })
    except ImportError as e:
        import warnings
        warnings.warn(f"Extended generators not available: {e}")

# Default difficulty distribution per challenge count
_DEFAULT_DIFFICULTY_DIST = {
    ChallengeDifficulty.TRIVIAL: 0.10,
    ChallengeDifficulty.EASY: 0.20,
    ChallengeDifficulty.MEDIUM: 0.35,
    ChallengeDifficulty.HARD: 0.25,
    ChallengeDifficulty.EXPERT: 0.10,
}

GENERATOR_VERSION = "2.0.0"


def generate_challenge_set(
    domain: str,
    n_challenges: int = 20,
    seed: Optional[int] = None,
    difficulty_distribution: Optional[Dict[ChallengeDifficulty, float]] = None,
) -> ChallengeSet:
    """
    Generate a fresh set of challenges for a domain.

    Args:
        domain: One of the registered domain generators
        n_challenges: Number of challenges to generate
        seed: Random seed for reproducibility (auto-generated if None)
        difficulty_distribution: Custom difficulty weights (must sum to ~1.0)

    Returns:
        ChallengeSet with unique challenges and a session hash
    """
    _ensure_extended_generators()
    if domain not in _DOMAIN_GENERATORS:
        raise ValueError(
            f"Unknown domain '{domain}'. Available: {', '.join(sorted(_DOMAIN_GENERATORS.keys()))}"
        )

    if seed is None:
        seed = int(time.time() * 1000) % (2**31)

    rng = random.Random(seed)
    dist = difficulty_distribution or _DEFAULT_DIFFICULTY_DIST

    # Compute how many challenges per difficulty level
    counts = {}
    remaining = n_challenges
    for diff, proportion in sorted(dist.items(), key=lambda x: x[1]):
        if diff == list(dist.keys())[-1]:
            counts[diff] = remaining
        else:
            c = max(1, round(n_challenges * proportion))
            counts[diff] = min(c, remaining)
            remaining -= counts[diff]

    generator = _DOMAIN_GENERATORS[domain]()
    challenges = []

    for difficulty, count in counts.items():
        for _ in range(count):
            challenge = generator.generate(rng, difficulty)
            challenges.append(challenge)

    # Shuffle to avoid predictable ordering
    rng.shuffle(challenges)

    # Compute session hash from all challenge IDs
    hash_input = "|".join(c.challenge_id for c in challenges)
    session_hash = hashlib.sha256(hash_input.encode()).hexdigest()[:16]
    session_id = f"{domain}_{session_hash}_{seed}"

    diff_dist = {}
    for c in challenges:
        diff_dist[c.difficulty.value] = diff_dist.get(c.difficulty.value, 0) + 1

    return ChallengeSet(
        session_id=session_id,
        session_hash=session_hash,
        challenges=challenges,
        generated_at_utc=datetime.now(timezone.utc).isoformat(),
        generator_version=GENERATOR_VERSION,
        domain=domain,
        seed=seed,
        difficulty_distribution=diff_dist,
    )


def list_domains() -> List[str]:
    """List all available challenge domains."""
    _ensure_extended_generators()
    return sorted(_DOMAIN_GENERATORS.keys())


# ──────────────────────────────────────────────────────────────────────────────
# Helper Functions
# ──────────────────────────────────────────────────────────────────────────────

def _make_challenge_id(domain: str, prompt: str, answer: str) -> str:
    """Create a deterministic challenge ID from content."""
    raw = f"{domain}:{prompt}:{answer}"
    return hashlib.sha256(raw.encode()).hexdigest()[:12]


def _comb(n: int, k: int) -> int:
    """Compute binomial coefficient C(n, k)."""
    if k < 0 or k > n:
        return 0
    if k == 0 or k == n:
        return 1
    k = min(k, n - k)
    result = 1
    for i in range(k):
        result = result * (n - i) // (i + 1)
    return result


def _gen_markov_params(rng: random.Random) -> Dict[str, Any]:
    """Generate a random transition matrix for a Markov chain challenge."""
    n = rng.randint(2, 4)
    matrix = []
    for _ in range(n):
        row = [rng.random() for _ in range(n)]
        total = sum(row)
        row = [round(v / total, 3) for v in row]
        # Fix rounding to ensure sum = 1.0
        diff = round(1.0 - sum(row), 3)
        row[-1] = round(row[-1] + diff, 3)
        matrix.append(row)

    matrix_str = "\n".join(
        "  [" + ", ".join(f"{v:.3f}" for v in row) + "]"
        for row in matrix
    )

    return {"n": n, "matrix": matrix_str, "raw_matrix": matrix}


def _solve_stationary(matrix: List[List[float]]) -> str:
    """Solve for the stationary distribution of a Markov chain."""
    try:
        mat = np.array(matrix, dtype=float)
        n = mat.shape[0]

        # Solve pi @ P = pi with constraint sum(pi) = 1
        # Equivalent to: (P^T - I) @ pi = 0, with sum = 1
        A = mat.T - np.eye(n)
        A[-1] = np.ones(n)
        b = np.zeros(n)
        b[-1] = 1.0

        pi = np.linalg.solve(A, b)
        return ", ".join(f"{v:.3f}" for v in pi)
    except Exception:
        return "Unable to compute"
