"""
Insight client factory.

Returns the best available backend based on configured API keys.
With no keys present, falls back to TemplateInsightClient automatically.
"""
from __future__ import annotations

from src.insights.base import BaseInsightClient
from src.insights.template_client import TemplateInsightClient


def get_insight_client(
    anthropic_api_key: str | None = None,
    openai_api_key: str | None = None,
) -> BaseInsightClient:
    """
    Return the appropriate insight client.

    Priority:
      1. ClaudeInsightClient   — when anthropic_api_key is set (Phase 6+)
      2. OpenAIInsightClient   — when openai_api_key is set (Phase 6+)
      3. TemplateInsightClient — always available, no key required

    The Claude and OpenAI clients are not yet implemented; this factory
    returns the template client unconditionally until they are added.
    """
    # Future: add key-gated branches here when Claude/OpenAI clients are implemented.
    # if anthropic_api_key:
    #     from src.insights.claude_client import ClaudeInsightClient
    #     return ClaudeInsightClient(api_key=anthropic_api_key)
    # if openai_api_key:
    #     from src.insights.openai_client import OpenAIInsightClient
    #     return OpenAIInsightClient(api_key=openai_api_key)

    return TemplateInsightClient()
