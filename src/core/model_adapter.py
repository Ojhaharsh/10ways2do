"""
Universal Model Adapter for 10ways2do Benchmark Platform.

Provides a unified interface to evaluate any AI model — cloud APIs,
local models, or the existing hardcoded approaches — against the
benchmark's dynamic challenge system.

Supported backends:
- Google Gemini (gemini-2.5-pro, gemini-2.5-flash, etc.)
- Perplexity (sonar models)
- OpenAI-compatible (GPT-4o, Qwen via Together/OpenRouter, etc.)
- Local HuggingFace models
- Mock adapter for testing
"""

from __future__ import annotations

import json
import os
import time
import hashlib
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from pathlib import Path


@dataclass
class ModelResponse:
    """Standardized response from any model backend."""

    content: str
    raw_response: Dict[str, Any] = field(default_factory=dict)
    latency_ms: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    cost_usd: float = 0.0
    model_id: str = ""
    finish_reason: str = ""
    reasoning_trace: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "content": self.content,
            "latency_ms": self.latency_ms,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "cost_usd": self.cost_usd,
            "model_id": self.model_id,
            "finish_reason": self.finish_reason,
            "reasoning_trace": self.reasoning_trace,
        }


@dataclass
class ModelConfig:
    """Configuration for a model adapter."""

    provider: str  # gemini, perplexity, openai, local, mock
    model_name: str
    api_key: Optional[str] = None
    api_base_url: Optional[str] = None
    temperature: float = 0.0
    max_tokens: int = 4096
    timeout_seconds: float = 120.0
    extra_params: Dict[str, Any] = field(default_factory=dict)

    # Cost tracking (per 1M tokens)
    cost_per_1m_input: float = 0.0
    cost_per_1m_output: float = 0.0

    @property
    def display_name(self) -> str:
        return f"{self.provider}/{self.model_name}"


class BaseModelAdapter(ABC):
    """Abstract base for all model adapters."""

    def __init__(self, config: ModelConfig):
        self.config = config
        self._call_count = 0
        self._total_latency_ms = 0.0
        self._total_tokens = 0
        self._total_cost = 0.0

    @abstractmethod
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        structured_output: Optional[Dict] = None,
    ) -> ModelResponse:
        """Generate a response from the model."""
        pass

    def generate_with_tracking(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        structured_output: Optional[Dict] = None,
    ) -> ModelResponse:
        """Generate with automatic cost and latency tracking."""
        start = time.perf_counter()
        response = self.generate(prompt, system_prompt, structured_output)
        elapsed_ms = (time.perf_counter() - start) * 1000

        if response.latency_ms == 0.0:
            response.latency_ms = elapsed_ms

        self._call_count += 1
        self._total_latency_ms += response.latency_ms
        self._total_tokens += response.total_tokens
        self._total_cost += response.cost_usd

        return response

    def get_session_stats(self) -> Dict[str, Any]:
        """Get cumulative session statistics."""
        return {
            "provider": self.config.provider,
            "model": self.config.model_name,
            "calls": self._call_count,
            "total_latency_ms": round(self._total_latency_ms, 2),
            "avg_latency_ms": round(self._total_latency_ms / max(1, self._call_count), 2),
            "total_tokens": self._total_tokens,
            "total_cost_usd": round(self._total_cost, 6),
        }

    def reset_stats(self):
        self._call_count = 0
        self._total_latency_ms = 0.0
        self._total_tokens = 0
        self._total_cost = 0.0


class GeminiAdapter(BaseModelAdapter):
    """Google Gemini API adapter."""

    # Pricing per 1M tokens (as of 2026)
    PRICING = {
        "gemini-2.5-pro": {"input": 1.25, "output": 10.0},
        "gemini-2.5-flash": {"input": 0.15, "output": 0.60},
        "gemini-2.0-flash": {"input": 0.10, "output": 0.40},
    }

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                from google import genai
                api_key = self.config.api_key or os.environ.get("GEMINI_API_KEY", "")
                self._client = genai.Client(api_key=api_key)
            except ImportError:
                raise ImportError(
                    "google-genai package required. Install: pip install google-genai"
                )
        return self._client

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        structured_output: Optional[Dict] = None,
    ) -> ModelResponse:
        client = self._get_client()
        from google.genai import types

        config_kwargs = {
            "temperature": self.config.temperature,
            "max_output_tokens": self.config.max_tokens,
        }

        if system_prompt:
            config_kwargs["system_instruction"] = system_prompt

        if structured_output:
            config_kwargs["response_mime_type"] = "application/json"

        # Force 5s sleep to respect free tier 15 RPM limit
        time.sleep(5)
        start = time.perf_counter()
        
        max_retries = 5
        for attempt in range(max_retries):
            try:
                response = client.models.generate_content(
                    model=self.config.model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(**config_kwargs),
                )
                break
            except Exception as e:
                err_str = str(e)
                if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                    if attempt < max_retries - 1:
                        # Free tier is 15 RPM, wait 20s
                        time.sleep(20)
                        continue
                raise e

        latency_ms = (time.perf_counter() - start) * 1000

        content = response.text or ""
        input_tokens = getattr(response.usage_metadata, "prompt_token_count", 0) or 0
        output_tokens = getattr(response.usage_metadata, "candidates_token_count", 0) or 0
        total_tokens = input_tokens + output_tokens

        pricing = self.PRICING.get(self.config.model_name, {"input": 0, "output": 0})
        cost = (input_tokens * pricing["input"] + output_tokens * pricing["output"]) / 1_000_000

        # Extract reasoning trace if available (thinking models)
        reasoning_trace = None
        try:
            if hasattr(response, "candidates") and response.candidates:
                for part in response.candidates[0].content.parts:
                    if hasattr(part, "thought") and part.thought:
                        reasoning_trace = part.text
        except Exception:
            pass

        return ModelResponse(
            content=content,
            raw_response={"text": content},
            latency_ms=latency_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            cost_usd=cost,
            model_id=self.config.model_name,
            finish_reason="stop",
            reasoning_trace=reasoning_trace,
        )


class PerplexityAdapter(BaseModelAdapter):
    """Perplexity API adapter (OpenAI-compatible)."""

    PRICING = {
        "sonar-pro": {"input": 3.0, "output": 15.0},
        "sonar": {"input": 1.0, "output": 1.0},
        "sonar-deep-research": {"input": 2.0, "output": 8.0},
    }

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        config.api_base_url = config.api_base_url or "https://api.perplexity.ai"
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                import httpx
                api_key = self.config.api_key or os.environ.get("PERPLEXITY_API_KEY", "")
                self._client = httpx.Client(
                    base_url=self.config.api_base_url,
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    timeout=self.config.timeout_seconds,
                )
            except ImportError:
                raise ImportError("httpx required. Install: pip install httpx")
        return self._client

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        structured_output: Optional[Dict] = None,
    ) -> ModelResponse:
        client = self._get_client()

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.config.model_name,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }

        start = time.perf_counter()
        
        max_retries = 5
        resp = None
        for attempt in range(max_retries):
            try:
                resp = client.post("/chat/completions", json=payload)
                resp.raise_for_status()
                break
            except Exception as e:
                err_str = str(e)
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt) # Exponential backoff
                    continue
                raise e

        latency_ms = (time.perf_counter() - start) * 1000
        data = resp.json()

        content = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        input_tokens = usage.get("prompt_tokens", 0)
        output_tokens = usage.get("completion_tokens", 0)

        pricing = self.PRICING.get(self.config.model_name, {"input": 0, "output": 0})
        cost = (input_tokens * pricing["input"] + output_tokens * pricing["output"]) / 1_000_000

        return ModelResponse(
            content=content,
            raw_response=data,
            latency_ms=latency_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            cost_usd=cost,
            model_id=data.get("model", self.config.model_name),
            finish_reason=data["choices"][0].get("finish_reason", ""),
        )


class OpenAICompatibleAdapter(BaseModelAdapter):
    """
    Adapter for any OpenAI-compatible API.

    Works with: OpenAI, Together.ai, OpenRouter, Groq, local vLLM, etc.
    Use this for evaluating Qwen via Together/OpenRouter.
    """

    def __init__(self, config: ModelConfig):
        super().__init__(config)
        config.api_base_url = config.api_base_url or "https://api.openai.com/v1"
        self._client = None

    def _get_client(self):
        if self._client is None:
            try:
                import httpx
                api_key = self.config.api_key or os.environ.get("OPENAI_API_KEY", "")
                self._client = httpx.Client(
                    base_url=self.config.api_base_url,
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    timeout=self.config.timeout_seconds,
                )
            except ImportError:
                raise ImportError("httpx required. Install: pip install httpx")
        return self._client

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        structured_output: Optional[Dict] = None,
    ) -> ModelResponse:
        client = self._get_client()

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.config.model_name,
            "messages": messages,
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
        }

        if structured_output:
            payload["response_format"] = {"type": "json_object"}

        start = time.perf_counter()
        resp = client.post("/chat/completions", json=payload)
        latency_ms = (time.perf_counter() - start) * 1000

        resp.raise_for_status()
        data = resp.json()

        content = data["choices"][0]["message"]["content"]
        usage = data.get("usage", {})
        input_tokens = usage.get("prompt_tokens", 0)
        output_tokens = usage.get("completion_tokens", 0)

        cost = (
            input_tokens * self.config.cost_per_1m_input
            + output_tokens * self.config.cost_per_1m_output
        ) / 1_000_000

        return ModelResponse(
            content=content,
            raw_response=data,
            latency_ms=latency_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=input_tokens + output_tokens,
            cost_usd=cost,
            model_id=data.get("model", self.config.model_name),
            finish_reason=data["choices"][0].get("finish_reason", ""),
        )


class MockAdapter(BaseModelAdapter):
    """Mock adapter for testing the evaluation pipeline without API calls."""

    def __init__(self, config: ModelConfig, response_fn=None):
        super().__init__(config)
        self._response_fn = response_fn

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        structured_output: Optional[Dict] = None,
    ) -> ModelResponse:
        if self._response_fn:
            content = self._response_fn(prompt)
        else:
            # Smart default: try to extract expected format from prompt
            content = self._generate_mock_response(prompt, structured_output)

        time.sleep(0.05)  # Simulate 50ms latency

        token_est = len(prompt.split()) + len(content.split())

        return ModelResponse(
            content=content,
            raw_response={"mock": True},
            latency_ms=50.0,
            input_tokens=len(prompt.split()),
            output_tokens=len(content.split()),
            total_tokens=token_est,
            cost_usd=0.0,
            model_id="mock/mock-v1",
            finish_reason="stop",
        )

    def _generate_mock_response(self, prompt: str, structured_output: Optional[Dict]) -> str:
        """Generate a plausible mock response based on the prompt."""
        prompt_lower = prompt.lower()

        if structured_output or "json" in prompt_lower:
            return json.dumps({
                "answer": "mock_answer",
                "confidence": 0.75,
                "reasoning": "This is a mock reasoning trace for testing.",
            })

        if "classify" in prompt_lower or "category" in prompt_lower:
            return "Category A"
        if "extract" in prompt_lower:
            return json.dumps({"entity": "mock_entity", "value": "mock_value"})
        if "predict" in prompt_lower or "forecast" in prompt_lower:
            return "42.0"
        if "anomaly" in prompt_lower or "fraud" in prompt_lower:
            return "normal"

        return "This is a mock response from the 10ways2do benchmark testing adapter."


# ──────────────────────────────────────────────────────────────────────────────
# Factory
# ──────────────────────────────────────────────────────────────────────────────

_ADAPTER_REGISTRY = {
    "gemini": GeminiAdapter,
    "perplexity": PerplexityAdapter,
    "openai": OpenAICompatibleAdapter,
    "together": OpenAICompatibleAdapter,
    "openrouter": OpenAICompatibleAdapter,
    "groq": OpenAICompatibleAdapter,
    "local": OpenAICompatibleAdapter,
    "mock": MockAdapter,
}

# Default API base URLs for known providers
_DEFAULT_BASES = {
    "together": "https://api.together.xyz/v1",
    "openrouter": "https://openrouter.ai/api/v1",
    "groq": "https://api.groq.com/openai/v1",
}

# Default API key env var names
_DEFAULT_KEY_ENVS = {
    "gemini": "GEMINI_API_KEY",
    "perplexity": "PERPLEXITY_API_KEY",
    "openai": "OPENAI_API_KEY",
    "together": "TOGETHER_API_KEY",
    "openrouter": "OPENROUTER_API_KEY",
    "groq": "GROQ_API_KEY",
}


def create_adapter(
    provider: str,
    model_name: str,
    api_key: Optional[str] = None,
    api_base_url: Optional[str] = None,
    **kwargs,
) -> BaseModelAdapter:
    """
    Factory function to create a model adapter.

    Examples:
        # Gemini
        adapter = create_adapter("gemini", "gemini-2.5-flash", api_key="...")

        # Qwen via Together.ai
        adapter = create_adapter("together", "Qwen/Qwen3-235B-A22B", api_key="...")

        # Qwen via OpenRouter
        adapter = create_adapter("openrouter", "qwen/qwen3-235b-a22b", api_key="...")

        # Mock for testing
        adapter = create_adapter("mock", "mock-v1")
    """
    provider_lower = provider.lower()

    if provider_lower not in _ADAPTER_REGISTRY:
        raise ValueError(
            f"Unknown provider '{provider}'. "
            f"Available: {', '.join(sorted(_ADAPTER_REGISTRY.keys()))}"
        )

    # Resolve API key from env if not provided
    if api_key is None:
        env_var = _DEFAULT_KEY_ENVS.get(provider_lower, "")
        api_key = os.environ.get(env_var, "")

    # Resolve base URL for known providers
    if api_base_url is None:
        api_base_url = _DEFAULT_BASES.get(provider_lower)

    config = ModelConfig(
        provider=provider_lower,
        model_name=model_name,
        api_key=api_key,
        api_base_url=api_base_url,
        **kwargs,
    )

    adapter_cls = _ADAPTER_REGISTRY[provider_lower]
    return adapter_cls(config)


def list_providers() -> List[str]:
    """List all available model provider names."""
    return sorted(_ADAPTER_REGISTRY.keys())
