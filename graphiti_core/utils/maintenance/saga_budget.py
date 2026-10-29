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

import json
import logging
import time

from graphiti_core.graphiti_types import GraphitiClients, generate_prompt_response
from graphiti_core.llm_client import LLMClient
from graphiti_core.prompts import prompt_library
from graphiti_core.prompts.summarize_sagas import SagaSummary
from graphiti_core.utils.content_chunking import CHARS_PER_TOKEN, estimate_tokens
from graphiti_core.utils.maintenance.node_budget import _messages_cost
from graphiti_core.utils.text_utils import MAX_SUMMARY_CHARS, truncate_at_sentence
from graphiti_core.utils.token_budget import pack_by_budget

logger = logging.getLogger(__name__)

# Episode joiner used by the summarize_saga prompt
_SEP = '\n---\n'


def saga_batches(
    llm_client: LLMClient,
    saga_name: str,
    existing_summary: str,
    episode_contents: list[str],
) -> list[list[str]]:
    """Pack episodes, in order, into batches whose summarize_saga prompt fits the input budget."""
    budget = llm_client.input_token_budget()
    if not isinstance(budget, int):
        return [episode_contents]

    # Worst case running summary, so one up-front packing is valid for every batch
    fixed = _messages_cost(
        prompt_library.summarize_sagas.summarize_saga(
            {
                'saga_name': saga_name,
                'existing_summary': 'x' * max(MAX_SUMMARY_CHARS, len(existing_summary)),
                'episodes': [],
            }
        )
    ) + estimate_tokens(json.dumps(SagaSummary.model_json_schema()))
    room = budget - fixed
    if room <= 0:
        logger.warning(
            'summarize_saga: fixed prompt cost (%d tokens) exceeds the input-token budget %d',
            fixed,
            budget,
        )
        return [episode_contents]

    def cost(content: str) -> int:
        # +1: summing per-item floors can undercount the joined string
        return estimate_tokens(content + _SEP) + 1

    fitted = [
        truncate_at_sentence(c, (room - 2) * CHARS_PER_TOKEN - len(_SEP)) if cost(c) > room else c
        for c in episode_contents
    ]
    batches = pack_by_budget(fitted, cost, budget, fixed)
    if len(batches) > 1:
        logger.info(
            'summarize_saga: %d episodes in %d batches (budget=%d fixed=%d)',
            len(fitted),
            len(batches),
            budget,
            fixed,
        )
    return batches


async def summarize_saga_by_budget(
    llm_client: LLMClient,
    saga_name: str,
    existing_summary: str,
    episode_contents: list[str],
    clients: GraphitiClients | None = None,
) -> str:
    """Fold episodes into the saga summary in budgeted batches and return the final summary.

    Pure: persists nothing, so any LLM failure propagates before the caller saves.
    shortcut: a client that cancels on timeout discards the whole fold and a retry restarts it;
    upgrade to asyncio.shield plus per-saga in-flight dedupe in the MCP tool if summarize_saga
    timeouts are observed.
    """
    summary = existing_summary
    batches = saga_batches(llm_client, saga_name, existing_summary, episode_contents)
    for i, batch in enumerate(batches, 1):
        started = time.monotonic()
        response = await generate_prompt_response(
            llm_client,
            'summarize_sagas.summarize_saga',
            prompt_library.summarize_sagas.summarize_saga,
            {'saga_name': saga_name, 'existing_summary': summary, 'episodes': batch},
            clients=clients,
            response_model=SagaSummary,
        )
        # Empty reply keeps the running summary rather than wiping it
        summary = (response.get('summary', '') or summary)[:MAX_SUMMARY_CHARS]
        if len(batches) > 1:
            logger.info(
                'summarize_saga: batch %d/%d done in %.1fs',
                i,
                len(batches),
                time.monotonic() - started,
            )
    return summary
