"""LLM client wrapper. One class for all chat causal LMs (Llama, Qwen, ...)."""

from .client import GenerationConfig, LLMClient

__all__ = ["LLMClient", "GenerationConfig"]
