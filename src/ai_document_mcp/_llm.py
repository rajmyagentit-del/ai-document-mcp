"""Small shared helpers used across modules that call the Anthropic API."""

from __future__ import annotations

import anthropic


def response_text(response: anthropic.types.Message) -> str:
    """Extract text from a Claude response's first content block.

    We only ever send plain text-generation requests (no tool use) in this
    codebase, so the first block should always be a TextBlock. Raising
    loudly if that assumption is ever wrong is safer than silently
    returning something incorrect (e.g. via getattr with a default).
    """
    block = response.content[0]
    if isinstance(block, anthropic.types.TextBlock):
        return block.text
    raise TypeError(f"Expected a TextBlock from Claude, got {type(block).__name__}")