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
from typing import Any

from pydantic import BaseModel

from graphiti_core.graphiti_types import GraphitiClients, generate_prompt_response
from graphiti_core.llm_client import LLMClient
from graphiti_core.nodes import EntityNode, EpisodeType, EpisodicNode
from graphiti_core.prompts import prompt_library
from graphiti_core.prompts.dedupe_nodes import NodeDuplicate, NodeResolutions
from graphiti_core.prompts.extract_nodes import SummarizedEntities
from graphiti_core.prompts.models import Message, PromptFunction
from graphiti_core.prompts.names import PromptName
from graphiti_core.utils.content_chunking import estimate_tokens
from graphiti_core.utils.text_utils import (
    build_previous_episodes_context,
    remaining_budget_for_previous_episodes,
)
from graphiti_core.utils.token_budget import pack_by_budget

logger = logging.getLogger(__name__)


def _messages_cost(messages: list[Message]) -> int:
    return sum(estimate_tokens(m.content if hasattr(m, 'content') else str(m)) for m in messages)


def _warn_fixed_overflow(label: str, fixed_cost: int, budget: int, context: dict[str, Any]) -> None:
    """Log one WARNING naming the component that makes the fixed prompt cost exceed the budget."""
    components = {
        k: estimate_tokens(
            v if isinstance(v, str) else json.dumps(v, default=str, ensure_ascii=False)
        )
        for k, v in context.items()
    }
    components['prompt template + schema'] = max(fixed_cost - sum(components.values()), 0)
    name, tokens = max(components.items(), key=lambda kv: kv[1])
    logger.warning(
        '%s: fixed prompt cost (%d tokens) exceeds the input-token budget %d; '
        'largest component: %s (%d tokens)',
        label,
        fixed_cost,
        budget,
        name,
        tokens,
    )


def extraction_prompt(source: EpisodeType) -> PromptFunction:
    """Prompt builder extract_nodes sends for an episode source (as in _call_extraction_llm)."""
    if source == EpisodeType.message:
        return prompt_library.extract_nodes.extract_message
    if source == EpisodeType.json:
        return prompt_library.extract_nodes.extract_json
    return prompt_library.extract_nodes.extract_text


def summary_prompt(use_episode_prompt: bool) -> tuple[PromptName, PromptFunction]:
    """Prompt name and builder a summary flight sends."""
    if use_episode_prompt:
        return (
            'extract_nodes.extract_entity_summaries_from_episodes',
            prompt_library.extract_nodes.extract_entity_summaries_from_episodes,
        )
    return (
        'extract_nodes.extract_summaries_batch',
        prompt_library.extract_nodes.extract_summaries_batch,
    )


def summary_entity_entry(node: EntityNode) -> dict[str, Any]:
    return {
        'name': node.name,
        'summary': node.summary,
        'entity_types': node.labels,
        'attributes': node.attributes,
    }


def with_budgeted_previous_episodes(
    label: str,
    llm_client: LLMClient,
    prompt_builder: PromptFunction,
    context: dict[str, Any],
    previous_episodes: list[EpisodicNode] | None,
    response_model: type[BaseModel],
) -> dict[str, Any]:
    """Return ``context`` with the previous episodes that fit the remaining input budget.

    ``context`` must render the prompt without previous episodes. ``None`` previous
    episodes send none.
    """
    episodes_context: list[dict[str, str | None]] = []
    # Computed even without previous episodes so a fixed-overhead overflow is still reported.
    budget_tokens = remaining_budget_for_previous_episodes(
        llm_client, prompt_builder, context, response_model=response_model
    )
    input_budget = llm_client.input_token_budget()
    if budget_tokens == 0 and isinstance(input_budget, int):
        fixed_cost = _messages_cost(prompt_builder({**context, 'previous_episodes': []}))
        fixed_cost += estimate_tokens(json.dumps(response_model.model_json_schema()))
        _warn_fixed_overflow(label, fixed_cost, input_budget, context)
    if previous_episodes is not None:
        episodes_context = build_previous_episodes_context(previous_episodes, budget_tokens)
    logger.debug(
        '%s: budget=%s fixed_estimate=%s episodes=%d/%d',
        label,
        budget_tokens,
        input_budget - budget_tokens
        if isinstance(input_budget, int) and budget_tokens is not None
        else 'N/A',
        len(episodes_context),
        len(previous_episodes) if previous_episodes is not None else 0,
    )
    return {**context, 'previous_episodes': episodes_context}


def pack_summary_flights(
    nodes: list[EntityNode],
    max_items: int,
    llm_client: LLMClient,
    prompt_builder: PromptFunction,
    base_context: dict[str, Any],
) -> list[list[EntityNode]]:
    """Partition nodes into flights of at most ``max_items`` that fit the input-token budget.

    ``base_context`` is the summary prompt context without entities and previous episodes;
    every flight sends it, so it counts as fixed cost. Previous episodes are filled per
    flight from what remains.
    """
    budget = llm_client.input_token_budget()
    if not isinstance(budget, int):
        return [nodes[i : i + max_items] for i in range(0, len(nodes), max_items)]

    fixed_cost = _messages_cost(prompt_builder(base_context))
    fixed_cost += estimate_tokens(json.dumps(SummarizedEntities.model_json_schema()))
    if fixed_cost >= budget:
        # No flight can fit; more flights would only resend the oversized episode
        _warn_fixed_overflow('summary flights', fixed_cost, budget, base_context)
        return [nodes[i : i + max_items] for i in range(0, len(nodes), max_items)]

    return pack_by_budget(
        nodes,
        # default=str: graph-hydrated attributes can hold datetimes or driver temporal types
        cost=lambda node: estimate_tokens(
            json.dumps(summary_entity_entry(node), ensure_ascii=False, default=str)
        ),
        budget=budget,
        fixed_cost=fixed_cost,
        max_items=max_items,
    )


def _dedupe_batches(llm_client: LLMClient, context: dict[str, Any]) -> list[list[dict[str, Any]]]:
    """Split the extracted nodes of a dedupe prompt so each call fits the input-token budget.

    Every batch carries all candidates, the episode and the previous episodes; only the
    extracted nodes are partitioned.
    """
    extracted_nodes_context: list[dict[str, Any]] = context['extracted_nodes']
    input_budget = llm_client.input_token_budget()
    if not isinstance(input_budget, int) or len(extracted_nodes_context) <= 1:
        return [extracted_nodes_context]

    fixed_cost = _messages_cost(
        prompt_library.dedupe_nodes.nodes({**context, 'extracted_nodes': []})
    )
    fixed_cost += estimate_tokens(json.dumps(NodeResolutions.model_json_schema()))
    batches = pack_by_budget(
        extracted_nodes_context,
        # ', ' joins entries; +1: summing per-entry floors undercounts the joined JSON
        cost=lambda entry: estimate_tokens(json.dumps(entry) + ', ') + 1,
        budget=input_budget,
        fixed_cost=fixed_cost,
    )
    if len(batches) > 1:
        logger.debug(
            '_resolve_with_llm: %d entities split into %d batches (budget=%d fixed=%d)',
            len(extracted_nodes_context),
            len(batches),
            input_budget,
            fixed_cost,
        )
    return batches


async def resolve_nodes_by_budget(
    llm_client: LLMClient,
    context: dict[str, Any],
    clients: GraphitiClients | None,
) -> list[NodeDuplicate]:
    """Run the dedupe_nodes prompt, split into sub-batches when it exceeds the budget.

    Resolution ids refer to ``context['extracted_nodes']`` as in a single call.
    """
    batches = _dedupe_batches(llm_client, context)
    node_resolutions: list[NodeDuplicate] = []
    for batch in batches:
        if len(batches) == 1:
            batch_context = context
        else:
            # The prompt numbers entities 0..N-1 per call, so each batch is re-numbered
            batch_context = dict(context)
            batch_context['extracted_nodes'] = [{**entry, 'id': i} for i, entry in enumerate(batch)]
        llm_response = await generate_prompt_response(
            llm_client,
            'dedupe_nodes.nodes',
            prompt_library.dedupe_nodes.nodes,
            batch_context,
            clients=clients,
            response_model=NodeResolutions,
        )
        batch_resolutions = NodeResolutions(**llm_response).entity_resolutions
        if len(batches) > 1:
            global_ids = [entry['id'] for entry in batch]
            for resolution in batch_resolutions:
                # Out-of-range ids map to -1 so they stay invalid instead of aliasing a node
                # of another batch
                resolution.id = global_ids[resolution.id] if 0 <= resolution.id < len(batch) else -1
        node_resolutions.extend(batch_resolutions)
    return node_resolutions
