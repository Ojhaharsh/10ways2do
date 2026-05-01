"""
Extended Challenge Generators for 10ways2do Benchmark Platform.

These generators cover the capabilities that actually differentiate
frontier AI models — beyond basic reasoning and classification.

Domains added:
- code_generation: Write working code from specs
- instruction_following: Follow precise, complex formatting rules
- logical_deduction: Formal logic, puzzles, constraint satisfaction
- hallucination_detection: Identify when to refuse or say "I don't know"
- summarization: Compress information accurately without loss
- tool_use_planning: Plan multi-step tool/API call sequences
"""

from __future__ import annotations

import hashlib
import json
import random
import string
from typing import Any, Dict, List

from src.core.dynamic_generator import (
    Challenge,
    ChallengeDifficulty,
    _make_challenge_id,
)


# ──────────────────────────────────────────────────────────────────────────────
# 1. CODE GENERATION
# ──────────────────────────────────────────────────────────────────────────────

class CodeGenerationChallengeGenerator:
    """
    Tests the ability to write correct, working code from specifications.
    Evaluated by checking output against expected results.
    """

    TEMPLATES = {
        ChallengeDifficulty.TRIVIAL: [
            {
                "prompt": "Write a Python function `add(a, b)` that returns the sum of two numbers.",
                "test_cases": [("add(2, 3)", "5"), ("add(-1, 1)", "0"), ("add(0, 0)", "0")],
                "expected_signature": "def add(a, b):",
            },
            {
                "prompt": "Write a Python function `is_even(n)` that returns True if n is even, False otherwise.",
                "test_cases": [("is_even(4)", "True"), ("is_even(7)", "False"), ("is_even(0)", "True")],
                "expected_signature": "def is_even(n):",
            },
        ],
        ChallengeDifficulty.EASY: [
            {
                "prompt": "Write a Python function `fizzbuzz(n)` that returns 'Fizz' if n is divisible by 3, 'Buzz' if divisible by 5, 'FizzBuzz' if divisible by both, or str(n) otherwise.",
                "test_cases": [("fizzbuzz(3)", "'Fizz'"), ("fizzbuzz(5)", "'Buzz'"), ("fizzbuzz(15)", "'FizzBuzz'"), ("fizzbuzz(7)", "'7'")],
                "expected_signature": "def fizzbuzz(n):",
            },
            {
                "prompt": "Write a Python function `reverse_string(s)` that returns the reversed version of string s without using slicing or reversed().",
                "test_cases": [("reverse_string('hello')", "'olleh'"), ("reverse_string('')", "''"), ("reverse_string('a')", "'a'")],
                "expected_signature": "def reverse_string(s):",
            },
        ],
        ChallengeDifficulty.MEDIUM: [
            {
                "prompt": "Write a Python function `flatten(lst)` that takes a nested list of arbitrary depth and returns a flat list. Example: flatten([1, [2, [3, 4]], 5]) -> [1, 2, 3, 4, 5]",
                "test_cases": [("flatten([1, [2, [3, 4]], 5])", "[1, 2, 3, 4, 5]"), ("flatten([[1, 2], [3]])", "[1, 2, 3]"), ("flatten([])", "[]")],
                "expected_signature": "def flatten(lst):",
            },
            {
                "prompt": "Write a Python function `most_frequent(lst)` that returns the most frequently occurring element in a list. If there's a tie, return the one that appears first.",
                "test_cases": [("most_frequent([1,2,2,3,3,3])", "3"), ("most_frequent(['a','b','a'])", "'a'"), ("most_frequent([1])", "1")],
                "expected_signature": "def most_frequent(lst):",
            },
        ],
        ChallengeDifficulty.HARD: [
            {
                "prompt": "Write a Python function `lru_cache_manual(capacity)` that returns a class with `get(key)` and `put(key, value)` methods implementing an LRU cache. `get` returns -1 if key not found. Both operations must be O(1).",
                "test_cases": [],
                "expected_signature": "def lru_cache_manual(capacity):",
            },
        ],
        ChallengeDifficulty.EXPERT: [
            {
                "prompt": "Write a Python function `solve_knapsack(weights, values, capacity)` that solves the 0/1 knapsack problem and returns both the maximum value and the list of selected item indices. Use dynamic programming.",
                "test_cases": [],
                "expected_signature": "def solve_knapsack(weights, values, capacity):",
            },
        ],
    }

    def generate(self, rng: random.Random, difficulty: ChallengeDifficulty) -> Challenge:
        templates = self.TEMPLATES.get(difficulty, self.TEMPLATES[ChallengeDifficulty.MEDIUM])
        spec = rng.choice(templates)

        prompt = (
            f"{spec['prompt']}\n\n"
            f"Return ONLY the Python code. No explanations, no markdown formatting."
        )

        expected = {
            "expected_signature": spec["expected_signature"],
            "test_cases": spec["test_cases"],
        }

        challenge_id = _make_challenge_id("code", prompt, spec["expected_signature"])

        return Challenge(
            challenge_id=challenge_id,
            domain="code_generation",
            challenge_type="function_implementation",
            difficulty=difficulty,
            prompt=prompt,
            expected_answer=expected,
            evaluation_rubric={
                "signature_match_weight": 0.2,
                "test_pass_weight": 0.5,
                "code_quality_weight": 0.2,
                "format_compliance_weight": 0.1,
            },
        )


# ──────────────────────────────────────────────────────────────────────────────
# 2. INSTRUCTION FOLLOWING
# ──────────────────────────────────────────────────────────────────────────────

class InstructionFollowingChallengeGenerator:
    """
    Tests whether the model can follow precise, complex formatting rules.
    This is the IFEval-style capability that differentiates production-ready models.
    """

    def generate(self, rng: random.Random, difficulty: ChallengeDifficulty) -> Challenge:
        constraints = []
        topic = rng.choice([
            "the benefits of renewable energy",
            "how machine learning works",
            "the history of the internet",
            "why exercise is important",
            "the water cycle",
        ])

        if difficulty == ChallengeDifficulty.TRIVIAL:
            word_count = rng.randint(20, 40)
            constraints = [f"exactly {word_count} words"]
            expected = {"word_count": word_count}
        elif difficulty == ChallengeDifficulty.EASY:
            n_sentences = rng.randint(3, 5)
            constraints = [f"exactly {n_sentences} sentences", "each sentence on a new line"]
            expected = {"n_sentences": n_sentences, "newline_separated": True}
        elif difficulty == ChallengeDifficulty.MEDIUM:
            n_bullets = rng.randint(4, 7)
            forbidden = rng.choice(["the", "is", "very", "really", "just"])
            constraints = [
                f"exactly {n_bullets} bullet points",
                f"do NOT use the word '{forbidden}'",
                "start each bullet with a verb",
            ]
            expected = {"n_bullets": n_bullets, "forbidden_word": forbidden, "verb_start": True}
        elif difficulty == ChallengeDifficulty.HARD:
            n_paragraphs = rng.randint(2, 3)
            words_per = rng.choice([25, 30, 40])
            end_word = rng.choice(["future", "conclusion", "summary", "perspective"])
            constraints = [
                f"exactly {n_paragraphs} paragraphs",
                f"each paragraph has exactly {words_per} words",
                f"the last word of the entire response must be '{end_word}'",
                "use no exclamation marks",
            ]
            expected = {"n_paragraphs": n_paragraphs, "words_per": words_per, "end_word": end_word, "no_exclamation": True}
        else:  # EXPERT
            constraints = [
                "exactly 5 paragraphs",
                "each paragraph has exactly 3 sentences",
                "the first word of each paragraph must start with letters A, B, C, D, E (in order)",
                "no sentence may exceed 15 words",
                "do not use any form of the verb 'to be'",
                "end the response with the word 'end'",
            ]
            expected = {"paragraphs": 5, "sentences_per": 3, "acrostic": "ABCDE", "max_words": 15, "no_be": True, "end_word": "end"}

        constraints_text = "\n".join(f"- {c}" for c in constraints)
        prompt = (
            f"Write about {topic}.\n\n"
            f"You MUST follow ALL of these formatting rules:\n{constraints_text}\n\n"
            f"Failure to follow ANY rule exactly means failure."
        )

        challenge_id = _make_challenge_id("ifeval", prompt, str(expected))

        return Challenge(
            challenge_id=challenge_id,
            domain="instruction_following",
            challenge_type="constrained_generation",
            difficulty=difficulty,
            prompt=prompt,
            expected_answer=expected,
            evaluation_rubric={
                "constraint_compliance_weight": 0.8,
                "content_quality_weight": 0.2,
            },
        )


# ──────────────────────────────────────────────────────────────────────────────
# 3. LOGICAL DEDUCTION
# ──────────────────────────────────────────────────────────────────────────────

class LogicalDeductionChallengeGenerator:
    """
    Tests formal logic, constraint satisfaction, and deductive reasoning.
    These are the puzzles that separate pattern-matching from genuine reasoning.
    """

    def generate(self, rng: random.Random, difficulty: ChallengeDifficulty) -> Challenge:
        if difficulty in (ChallengeDifficulty.TRIVIAL, ChallengeDifficulty.EASY):
            return self._generate_ordering(rng, difficulty)
        elif difficulty == ChallengeDifficulty.MEDIUM:
            return self._generate_truth_table(rng, difficulty)
        elif difficulty == ChallengeDifficulty.HARD:
            return self._generate_constraint(rng, difficulty)
        else:
            return self._generate_knights_knaves(rng, difficulty)

    def _generate_ordering(self, rng, difficulty):
        names = rng.sample(["Alice", "Bob", "Carol", "Dave", "Eve", "Frank"], k=rng.randint(3, 4))
        order = list(names)
        rng.shuffle(order)

        clues = []
        for i in range(len(order) - 1):
            clues.append(f"{order[i]} is taller than {order[i+1]}")

        rng.shuffle(clues)
        clues_text = "\n".join(f"- {c}" for c in clues)
        answer = ", ".join(order)

        prompt = (
            f"Given these facts:\n{clues_text}\n\n"
            f"List all {len(names)} people from tallest to shortest, separated by commas."
        )

        return Challenge(
            challenge_id=_make_challenge_id("logic", prompt, answer),
            domain="logical_deduction", challenge_type="ordering",
            difficulty=difficulty, prompt=prompt, expected_answer=answer,
            evaluation_rubric={"exact_match_weight": 1.0},
        )

    def _generate_truth_table(self, rng, difficulty):
        p_val = rng.choice([True, False])
        q_val = rng.choice([True, False])
        op = rng.choice(["AND", "OR", "XOR", "IMPLIES"])

        if op == "AND": result = p_val and q_val
        elif op == "OR": result = p_val or q_val
        elif op == "XOR": result = p_val ^ q_val
        else: result = (not p_val) or q_val

        prompt = f"If P is {p_val} and Q is {q_val}, what is P {op} Q? Answer True or False only."
        answer = str(result)

        return Challenge(
            challenge_id=_make_challenge_id("logic", prompt, answer),
            domain="logical_deduction", challenge_type="truth_evaluation",
            difficulty=difficulty, prompt=prompt, expected_answer=answer,
            evaluation_rubric={"exact_match_weight": 1.0},
        )

    def _generate_constraint(self, rng, difficulty):
        colors = rng.sample(["red", "blue", "green", "yellow"], k=3)
        names = rng.sample(["Alice", "Bob", "Carol"], k=3)
        assignment = dict(zip(names, colors))

        clues = [f"{names[0]}'s house is not {colors[1]}"]
        clues.append(f"{names[1]}'s house is {colors[1]}")
        clues.append(f"The {colors[2]} house belongs to {names[2]}")

        rng.shuffle(clues)
        clues_text = "\n".join(f"- {c}" for c in clues)
        answer = json.dumps(assignment, sort_keys=True)

        prompt = (
            f"Three people ({', '.join(names)}) live in houses of different colors ({', '.join(colors)}).\n"
            f"Given these clues:\n{clues_text}\n\n"
            f"Return a JSON object mapping each person to their house color."
        )

        return Challenge(
            challenge_id=_make_challenge_id("logic", prompt, answer),
            domain="logical_deduction", challenge_type="constraint_satisfaction",
            difficulty=difficulty, prompt=prompt, expected_answer=assignment,
            evaluation_rubric={"exact_match_weight": 0.7, "partial_match_weight": 0.3},
        )

    def _generate_knights_knaves(self, rng, difficulty):
        """Classic knights (always truth) and knaves (always lie) puzzle."""
        a_type = rng.choice(["knight", "knave"])
        b_type = rng.choice(["knight", "knave"])

        if a_type == "knight":
            a_says_about_b = b_type
        else:
            a_says_about_b = "knight" if b_type == "knave" else "knave"

        prompt = (
            f"On an island, knights always tell the truth and knaves always lie.\n"
            f"Person A says: 'Person B is a {a_says_about_b}.'\n"
            f"Person B says: 'We are {'the same type' if a_type == b_type else 'different types'}.'\n\n"
            f"What type is A and what type is B? Answer as: 'A is a [type], B is a [type]'"
        )

        answer = f"A is a {a_type}, B is a {b_type}"

        return Challenge(
            challenge_id=_make_challenge_id("logic", prompt, answer),
            domain="logical_deduction", challenge_type="knights_and_knaves",
            difficulty=difficulty, prompt=prompt, expected_answer=answer,
            evaluation_rubric={"exact_match_weight": 0.8, "reasoning_quality_weight": 0.2},
        )


# ──────────────────────────────────────────────────────────────────────────────
# 4. HALLUCINATION DETECTION / REFUSAL CALIBRATION
# ──────────────────────────────────────────────────────────────────────────────

class HallucinationDetectionChallengeGenerator:
    """
    Tests whether the model knows what it DOESN'T know.
    Good models refuse or express uncertainty on unanswerable questions.
    Bad models confidently hallucinate.
    """

    ANSWERABLE = [
        {"q": "What is the capital of France?", "a": "Paris"},
        {"q": "What is 17 * 23?", "a": "391"},
        {"q": "Who wrote Romeo and Juliet?", "a": "William Shakespeare"},
        {"q": "What is the chemical symbol for gold?", "a": "Au"},
        {"q": "How many planets are in our solar system?", "a": "8"},
    ]

    UNANSWERABLE = [
        "What will the stock price of Apple be on January 15, 2030?",
        "What is the name of the person who will win the 2032 Olympics 100m sprint?",
        "What did Albert Einstein eat for breakfast on March 3, 1921?",
        "How many grains of sand are there on Bondi Beach right now?",
        "What is the phone number of the CEO of the company that will be the most valuable in 2040?",
    ]

    TRICK_QUESTIONS = [
        {"q": "In what year did Napoleon win the Battle of Waterloo?", "a": "Napoleon lost the Battle of Waterloo", "type": "false_premise"},
        {"q": "What is the capital of the African country of Finland?", "a": "Finland is not in Africa", "type": "false_premise"},
        {"q": "How many times has the United States landed on Mars with astronauts?", "a": "No astronauts have landed on Mars", "type": "false_premise"},
    ]

    def generate(self, rng: random.Random, difficulty: ChallengeDifficulty) -> Challenge:
        if difficulty in (ChallengeDifficulty.TRIVIAL, ChallengeDifficulty.EASY):
            # Answerable questions — model should answer correctly
            item = rng.choice(self.ANSWERABLE)
            prompt = f"Answer the following question concisely:\n{item['q']}"
            expected = {"type": "answerable", "answer": item["a"]}
            rubric = {"exact_match_weight": 0.8, "format_compliance_weight": 0.2}
        elif difficulty == ChallengeDifficulty.MEDIUM:
            # Unanswerable — model should say "I don't know" or equivalent
            q = rng.choice(self.UNANSWERABLE)
            prompt = f"Answer the following question. If you cannot answer it with certainty, say so.\n{q}"
            expected = {"type": "unanswerable", "should_refuse": True}
            rubric = {"refusal_detection_weight": 0.7, "calibration_weight": 0.3}
        else:
            # Trick questions with false premises — model should catch the error
            item = rng.choice(self.TRICK_QUESTIONS)
            prompt = f"Answer the following question:\n{item['q']}"
            expected = {"type": "false_premise", "correct_response": item["a"]}
            rubric = {"premise_detection_weight": 0.6, "calibration_weight": 0.4}

        challenge_id = _make_challenge_id("hallucination", prompt, str(expected))

        return Challenge(
            challenge_id=challenge_id,
            domain="hallucination_resistance",
            challenge_type=expected["type"],
            difficulty=difficulty,
            prompt=prompt,
            expected_answer=expected,
            evaluation_rubric=rubric,
        )


# ──────────────────────────────────────────────────────────────────────────────
# 5. SUMMARIZATION
# ──────────────────────────────────────────────────────────────────────────────

class SummarizationChallengeGenerator:
    """Generates summarization challenges with verifiable key facts."""

    def generate(self, rng: random.Random, difficulty: ChallengeDifficulty) -> Challenge:
        company = rng.choice(["Nextera", "Vortex Labs", "Pinnacle Corp", "Atlas Engineering"])
        revenue = rng.randint(50, 500)
        growth = rng.randint(5, 40)
        employees = rng.randint(500, 10000)
        product = rng.choice(["CloudVault", "DataStream", "NeuralEdge", "SecureSync"])
        city = rng.choice(["San Francisco", "London", "Tokyo", "Berlin"])
        ceo = f"{rng.choice(['Alice', 'Bob', 'Carol', 'David'])} {rng.choice(['Chen', 'Patel', 'Kim', 'Smith'])}"

        key_facts = [
            f"Revenue: ${revenue}M",
            f"Growth: {growth}%",
            f"Employees: {employees}",
            f"CEO: {ceo}",
        ]

        text = (
            f"{company}, headquartered in {city}, reported annual revenue of ${revenue} million, "
            f"representing a {growth}% increase year-over-year. The company, led by CEO {ceo}, "
            f"now employs {employees} people globally. The growth was primarily driven by the "
            f"{product} platform, which saw adoption across multiple enterprise clients. "
            f"Industry analysts have noted that {company}'s strategic focus on cloud infrastructure "
            f"has positioned it well for continued expansion. The board approved additional investment "
            f"in R&D, allocating 15% of revenue to developing next-generation capabilities. "
            f"Competitors in the space have also shown strong performance, but {company} maintains "
            f"a differentiated position through its proprietary technology stack."
        )

        n_sentences = {
            ChallengeDifficulty.TRIVIAL: 1,
            ChallengeDifficulty.EASY: 2,
            ChallengeDifficulty.MEDIUM: 2,
            ChallengeDifficulty.HARD: 3,
            ChallengeDifficulty.EXPERT: 1,
        }[difficulty]

        prompt = (
            f"Summarize the following text in exactly {n_sentences} sentence(s). "
            f"Include all key numerical facts.\n\n"
            f"Text: {text}"
        )

        expected = {
            "key_facts": key_facts,
            "n_sentences": n_sentences,
            "company": company,
        }

        challenge_id = _make_challenge_id("summarization", prompt[:100], str(key_facts))

        return Challenge(
            challenge_id=challenge_id,
            domain="summarization",
            challenge_type="factual_summarization",
            difficulty=difficulty,
            prompt=prompt,
            expected_answer=expected,
            evaluation_rubric={
                "fact_coverage_weight": 0.5,
                "sentence_count_weight": 0.3,
                "conciseness_weight": 0.2,
            },
        )


# ──────────────────────────────────────────────────────────────────────────────
# 6. TOOL USE PLANNING
# ──────────────────────────────────────────────────────────────────────────────

class ToolUsePlanningChallengeGenerator:
    """
    Tests the model's ability to plan multi-step tool/API usage.
    Does NOT execute tools — evaluates the PLAN quality.
    """

    TOOLS = [
        {"name": "search_web", "desc": "Search the web for information", "params": ["query"]},
        {"name": "calculate", "desc": "Perform mathematical calculations", "params": ["expression"]},
        {"name": "get_weather", "desc": "Get current weather for a location", "params": ["city"]},
        {"name": "send_email", "desc": "Send an email", "params": ["to", "subject", "body"]},
        {"name": "read_file", "desc": "Read contents of a file", "params": ["path"]},
        {"name": "write_file", "desc": "Write content to a file", "params": ["path", "content"]},
        {"name": "query_database", "desc": "Run a SQL query", "params": ["query"]},
        {"name": "http_request", "desc": "Make an HTTP request", "params": ["url", "method"]},
    ]

    def generate(self, rng: random.Random, difficulty: ChallengeDifficulty) -> Challenge:
        n_tools = {
            ChallengeDifficulty.TRIVIAL: 3,
            ChallengeDifficulty.EASY: 4,
            ChallengeDifficulty.MEDIUM: 5,
            ChallengeDifficulty.HARD: 6,
            ChallengeDifficulty.EXPERT: 8,
        }[difficulty]

        available_tools = rng.sample(self.TOOLS, min(n_tools, len(self.TOOLS)))
        tool_names = [t["name"] for t in available_tools]

        scenarios = [
            {
                "task": f"Find the current weather in {rng.choice(['Tokyo', 'London', 'NYC'])} and email it to team@company.com",
                "expected_tools": ["get_weather", "send_email"],
                "expected_steps": 2,
            },
            {
                "task": f"Calculate the total revenue from Q1 ({rng.randint(10,100)}M) and Q2 ({rng.randint(10,100)}M), then write a summary to report.txt",
                "expected_tools": ["calculate", "write_file"],
                "expected_steps": 2,
            },
            {
                "task": f"Search for the latest AI news, summarize the top 3 results, and save to digest.txt",
                "expected_tools": ["search_web", "write_file"],
                "expected_steps": 3,
            },
            {
                "task": f"Read the config from settings.json, query the database for active users, calculate the average sessions per user, and email the report",
                "expected_tools": ["read_file", "query_database", "calculate", "send_email"],
                "expected_steps": 4,
            },
        ]

        # Pick a scenario that uses tools we have available
        valid = [s for s in scenarios if all(t in tool_names for t in s["expected_tools"])]
        if not valid:
            valid = scenarios[:1]
        scenario = rng.choice(valid)

        tools_desc = "\n".join(
            f"- {t['name']}({', '.join(t['params'])}): {t['desc']}"
            for t in available_tools
        )

        prompt = (
            f"You have access to these tools:\n{tools_desc}\n\n"
            f"Task: {scenario['task']}\n\n"
            f"Create a step-by-step plan listing which tools to call and in what order. "
            f"For each step, specify the tool name and the arguments you would pass. "
            f"Return as a JSON array of objects with 'step', 'tool', and 'args' keys."
        )

        expected = {
            "expected_tools": scenario["expected_tools"],
            "expected_steps": scenario["expected_steps"],
            "task": scenario["task"],
        }

        challenge_id = _make_challenge_id("tool_use", prompt[:100], str(scenario["expected_tools"]))

        return Challenge(
            challenge_id=challenge_id,
            domain="tool_use_planning",
            challenge_type="action_planning",
            difficulty=difficulty,
            prompt=prompt,
            expected_answer=expected,
            evaluation_rubric={
                "tool_selection_weight": 0.4,
                "ordering_weight": 0.3,
                "format_compliance_weight": 0.3,
            },
        )
