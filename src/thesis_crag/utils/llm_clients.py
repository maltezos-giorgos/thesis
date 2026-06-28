"""Retry-wrapped LLM client with structured JSON output and fallback parsing."""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path
from typing import Any

import anthropic
from dotenv import load_dotenv
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

# Load .env from repo root if present (picks up ANTHROPIC_API_KEY without shell export)
load_dotenv(Path(__file__).parent.parent.parent.parent / ".env", override=False)

logger = logging.getLogger(__name__)

_client: anthropic.Anthropic | None = None


def _get_client() -> anthropic.Anthropic:
    global _client
    if _client is None:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise OSError(
                "ANTHROPIC_API_KEY is not set. "
                "Either export it in your shell or add it to a .env file at the repo root."
            )
        _client = anthropic.Anthropic(api_key=api_key)
    return _client


def _extract_json_from_text(text: str) -> dict[str, Any]:
    """Extract the first JSON object from text, even if surrounded by prose."""
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        return json.loads(match.group())
    raise ValueError(f"No JSON object found in LLM response: {text!r}")


@retry(
    retry=retry_if_exception_type((anthropic.RateLimitError, anthropic.APIStatusError)),
    wait=wait_exponential(multiplier=2, min=4, max=120),
    stop=stop_after_attempt(8),
    reraise=True,
)
def call_llm_with_validation(
    system_prompt: str,
    user_message: str,
    model: str = "claude-haiku-4-5-20251001",
    max_tokens: int = 512,
) -> dict[str, Any]:
    """Call the Anthropic API and parse the response as a JSON dict.

    Retries up to 8 attempts on rate-limit or server errors with exponential
    backoff (multiplier=2, min=4s, max=120s).
    Falls back to regex extraction if the model wraps JSON in prose.
    """
    client = _get_client()
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=system_prompt,
        messages=[{"role": "user", "content": user_message}],
    )
    text = response.content[0].text.strip()
    # Strip markdown code fences that Haiku sometimes wraps around JSON
    if text.startswith("```"):
        text = text.split("```", 2)[1]
        if text.startswith("json"):
            text = text[4:]
        text = text.strip()
    logger.debug("LLM raw response: %s", text)

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return _extract_json_from_text(text)
