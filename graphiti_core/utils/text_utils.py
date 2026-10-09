"""
Copyright 2024, Zep Software, Inc.
Copyright 2025-2026, Anton Repin <robot@pimeleon.org>

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pydantic import BaseModel

    from graphiti_core.llm_client.client import LLMClient
    from graphiti_core.nodes import EpisodicNode

# Maximum length for entity/community summaries
MAX_SUMMARY_CHARS = 1000

logger = logging.getLogger(__name__)

# Approximate characters per token (conservative estimate)
_CHARS_PER_TOKEN = 4

# Minimum token threshold for partial truncation of a boundary episode
_PARTIAL_TRUNCATION_THRESHOLD = 50


def truncate_at_sentence(text: str, max_chars: int) -> str:
    """
    Truncate text at or about max_chars while respecting sentence boundaries.

    Attempts to truncate at the last complete sentence before max_chars.
    If no sentence boundary is found before max_chars, truncates at max_chars.

    Args:
        text: The text to truncate
        max_chars: Maximum number of characters

    Returns:
        Truncated text
    """
    if not text or len(text) <= max_chars:
        return text

    # Find all sentence boundaries (., !, ?) up to max_chars
    truncated = text[:max_chars]

    # Look for sentence boundaries: period, exclamation, or question mark followed by space or end
    sentence_pattern = r'[.!?](?:\s|$)'
    matches = list(re.finditer(sentence_pattern, truncated))

    if matches:
        # Truncate at the last sentence boundary found
        last_match = matches[-1]
        return text[: last_match.end()].rstrip()

    # No sentence boundary found, truncate at max_chars
    return truncated.rstrip()


def build_previous_episodes_context(
    episodes: list[EpisodicNode],
    budget_tokens: int | None = None,
) -> list[dict[str, str | None]]:
    """Build previous episodes context for extraction prompts.

    Returns a list of dicts with ``'content'`` and ``'timestamp'`` keys.

    * ``budget_tokens=None`` → current behavior (backward compatible).
    * ``budget_tokens <= 0`` → empty list.
    * Otherwise walks episodes **newest-first** (``valid_at`` descending,
      falling back to list order) and includes whole episodes whose
      estimated serialized cost fits.  When the next episode doesn't fit
      but ``>= _PARTIAL_TRUNCATION_THRESHOLD`` tokens remain, the episode
      is included truncated via ``truncate_at_sentence``.
    * ``PREVIOUS_EPISODE_MAX_CHARS`` still applies as an upper bound per
      episode.  Episodes are returned in **chronological** order (same
      relative order as today), only containing the selected subset.
    """
    from graphiti_core import helpers
    from graphiti_core.utils.content_chunking import estimate_tokens

    max_chars = helpers.PREVIOUS_EPISODE_MAX_CHARS

    if budget_tokens is None:
        # No budget → include all episodes (current behavior)
        result: list[dict[str, str | None]] = []
        for ep in episodes:
            ts = ep.valid_at.isoformat() if ep.valid_at else None
            content = ep.content
            if max_chars > 0:
                content = truncate_at_sentence(content, max_chars)
            result.append({'content': content, 'timestamp': ts})
        return result

    if budget_tokens <= 0:
        # Zero or negative budget → no episodes fit
        return []

    # Sort episodes newest-first for budget allocation
    sorted_by_newest = sorted(
        episodes,
        key=lambda e: (e.valid_at or datetime.min),
        reverse=True,
    )

    remaining = budget_tokens
    selected: list[dict[str, str | None]] = []

    for ep in sorted_by_newest:
        if remaining < 0:
            break

        ts = ep.valid_at.isoformat() if ep.valid_at else None
        content = ep.content
        if max_chars > 0:
            content = truncate_at_sentence(content, max_chars)

        # Estimate token cost of this episode's dict
        episode_dict = {'content': content, 'timestamp': ts}
        episode_tokens = estimate_tokens(json.dumps(episode_dict)) + 10  # small overhead

        if episode_tokens <= remaining:
            remaining -= episode_tokens
            selected.append({'content': content, 'timestamp': ts})
        elif remaining >= _PARTIAL_TRUNCATION_THRESHOLD:
            # Partially truncate to fit remaining budget.
            # Iteratively shrink until the truncated dict fits.
            max_content_chars = remaining * _CHARS_PER_TOKEN
            for _ in range(5):  # max iterations to avoid infinite loops
                truncated = truncate_at_sentence(content, max_content_chars)
                truncated_tokens = (
                    estimate_tokens(json.dumps({'content': truncated, 'timestamp': ts})) + 10
                )
                if truncated_tokens <= remaining:
                    remaining -= truncated_tokens
                    selected.append({'content': truncated, 'timestamp': ts})
                    break  # stop after this episode
                # Shrink by 10% and retry
                max_content_chars = int(max_content_chars * 0.9)
                if max_content_chars < 10:
                    break

    # Return in chronological order (same as current behavior)
    return sorted(selected, key=lambda d: (d.get('timestamp') or ''))


def remaining_budget_for_previous_episodes(
    llm_client: LLMClient,
    prompt_builder,
    context: dict,
    response_model: type[BaseModel] | None = None,
) -> int | None:
    """Compute remaining token budget after rendering the prompt without episodes.

    Returns ``None`` when ``llm_client.input_token_budget()`` is ``None``,
    meaning the caller should fall back to ``budget_tokens=None`` (no
    budget-limited episode selection).

    Otherwise renders the prompt with ``previous_episodes=[]``, sums
    ``estimate_tokens`` over all rendered messages' content, adds the
    schema estimate (when ``response_model`` is given), and returns
    ``budget - fixed`` (may be ``<= 0``).
    """
    from graphiti_core.utils.content_chunking import estimate_tokens

    budget = llm_client.input_token_budget()
    # Guard against MagicMock in tests (not a real int or None)
    if budget is None or (not isinstance(budget, int)):
        return None

    # Render with empty previous_episodes to measure the "fixed" cost
    fixed_context = dict(context)
    fixed_context['previous_episodes'] = []
    messages = prompt_builder(fixed_context)

    # Messages may be list[Message] or list[str] (legacy prompts)
    fixed_tokens = 0
    for m in messages:
        content = m.content if hasattr(m, 'content') else m
        fixed_tokens += estimate_tokens(content)

    # Add schema estimate when using structured output
    if response_model is not None:
        schema_tokens = estimate_tokens(json.dumps(response_model.model_json_schema()))
        # Schema appears as a user message in some providers; add conservatively
        fixed_tokens += schema_tokens

    remaining = budget - fixed_tokens
    return max(remaining, 0)


def concatenate_episodes(episodes: list[EpisodicNode]) -> str:
    """Concatenate episode contents with enumerated headers.

    When given a single episode, returns its content as-is.
    When given multiple episodes, each is prefixed with an ``[Episode N]``
    header so the LLM can distinguish where one ends and the next begins.
    """
    if len(episodes) == 1:
        return episodes[0].content
    parts: list[str] = []
    for i, ep in enumerate(episodes):
        timestamp = ep.valid_at.isoformat() if ep.valid_at else 'unknown'
        parts.append(f'[Episode {i}] (timestamp: {timestamp})\n{ep.content}')
    return '\n\n'.join(parts)
