"""
GuardRail Wonder - LLM Prompt Guardrails using Embedding-based Similarity
"""

from .core import PromptGuardrail, GuardrailResult, GuardrailDecision
from .embeddings import GeminiEmbeddingProvider, EmbeddingProvider
from .models import GuardrailConfig, PredefinedPrompt

__version__ = "0.1.0"

__all__ = [
    "PromptGuardrail",
    "GuardrailResult",
    "GuardrailDecision",
    "GeminiEmbeddingProvider",
    "EmbeddingProvider",
    "GuardrailConfig",
    "PredefinedPrompt",
]
