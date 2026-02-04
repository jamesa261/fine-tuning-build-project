from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Literal, Optional

from openai import OpenAI, AsyncOpenAI
from google import genai


Provider = Literal["openai", "google"]


@dataclass(frozen=True)
class ModelInfo:
    """Simple registry entry for supported models."""

    model: str
    provider: Provider
    aliases: List[str] = field(default_factory=list)
    description: Optional[str] = None


# Keep friendly keys stable for DVC params and CLI flags.
# The canonical model names reflect current API model identifiers.
MODEL_REGISTRY: Dict[str, ModelInfo] = {
    "gpt-5-mini": ModelInfo(
        model="gpt-5-mini",
        provider="openai",
        aliases=["gpt5-mini", "gpt5 mini", "gpt-5 mini"],
        description="OpenAI lightweight model with native JSON mode.",
    ),
    "gpt-5.2": ModelInfo(
        model="gpt-5.2",
        provider="openai",
        aliases=["gpt5.2", "gpt5 2"],
        description="Flagship OpenAI reasoning & generation model.",
    ),
    "gpt-4.1-mini": ModelInfo(
        model="gpt-4.1-mini",
        provider="openai",
        aliases=["gpt4.1-mini", "gpt-4.1mini"],
        description="Cost‑efficient OpenAI baseline compatible with structured outputs.",
    ),
    "gemini-3-flash": ModelInfo(
        model="gemini-3.0-flash",
        provider="google",
        aliases=["gemini-3.0-flash", "gemini3-flash"],
        description="Latest Gemini Flash 3.0 fast model.",
    ),
    "gemini-3-pro": ModelInfo(
        model="gemini-3.0-pro",
        provider="google",
        aliases=["gemini3-pro", "gemini-3.0-pro"],
        description="Gemini Pro 3.0 balanced quality model.",
    ),
    "gemini-2.5-flash": ModelInfo(
        model="gemini-2.5-flash",
        provider="google",
        aliases=["gemini-2.5-flash-exp"],
        description="Gemini 2.5 Flash with JSON schema support.",
    ),
    "gemma-3-27b-it": ModelInfo(
        model="gemma-3-27b-it",
        provider="google",
        aliases=["gemma-3-27b", "gemma-3-27b-instruct"],
        description="Gemma 3 instruction‑tuned model.",
    ),
}


def resolve_model(name: str) -> ModelInfo:
    key = name.lower().strip()
    for canonical, info in MODEL_REGISTRY.items():
        if key == canonical or key in info.aliases:
            return info
    raise ValueError(
        f"Unknown model '{name}'. Supported options: {', '.join(MODEL_REGISTRY.keys())}"
    )


def build_client(model_info: ModelInfo, api_key_env: Optional[str] = None, async_mode: bool = False):
    """Return the correct SDK client for the given model."""
    env_var = api_key_env or ("OPENAI_API_KEY" if model_info.provider == "openai" else "GEMINI_API_KEY")
    api_key = os.getenv(env_var)
    if not api_key:
        raise RuntimeError(f"Expected {env_var} to be set for {model_info.provider} calls.")

    if model_info.provider == "openai":
        return AsyncOpenAI(api_key=api_key) if async_mode else OpenAI(api_key=api_key)

    # google-genai client covers Gemini + Gemma
    return genai.Client(api_key=api_key)


def supported_model_names() -> Iterable[str]:
    return MODEL_REGISTRY.keys()
