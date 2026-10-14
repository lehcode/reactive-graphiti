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

from graphiti_core.edges import EntityEdge
from graphiti_core.graphiti_types import GraphitiClients, generate_prompt_response
from graphiti_core.llm_client import LLMClient
from graphiti_core.nodes import EntityNode, EpisodicNode
from graphiti_core.prompts import prompt_library
from graphiti_core.prompts.dedupe_edges import EdgeDuplicate
from graphiti_core.prompts.extract_edges import Edge as ExtractedEdge
from graphiti_core.prompts.extract_edges import ExtractedEdges
from graphiti_core.prompts.models import Message
from graphiti_core.utils.content_chunking import estimate_tokens, generate_covering_chunks
from graphiti_core.utils.maintenance.dedup_helpers import _normalize_string_exact
from graphiti_core.utils.text_utils import (
    build_previous_episodes_context,
    remaining_budget_for_previous_episodes,
)
from graphiti_core.utils.token_budget import pack_by_budget

logger = logging.getLogger(__name__)

# Upper bound on sequential LLM calls when edge extraction is split into covering chunks.
# Covering all node pairs needs about n^2 / k^2 chunks, which grows fast when the budget
# leaves room for few nodes; beyond this bound one over-budget call is sent instead.
MAX_EDGE_EXTRACTION_CHUNKS = 16


def _messages_cost(messages: list[Message]) -> int:
    return sum(estimate_tokens(m.content if hasattr(m, 'content') else str(m)) for m in messages)


def _covering_chunks(
    input_budget: int | None, fixed_cost: int, nodes: list[EntityNode], node_costs: list[int]
) -> tuple[list[tuple[list[EntityNode], list[int]]] | None, int]:
    """Return covering chunks of nodes when they do not fit next to the fixed prompt.

    ``None`` means one call with every node: the nodes fit, there is no budget, or the
    chunks needed exceed MAX_EDGE_EXTRACTION_CHUNKS.
    """
    if not (
        isinstance(input_budget, int)
        and len(nodes) > 2
        and fixed_cost + sum(node_costs) > input_budget
    ):
        return None, 0

    # Every pair of nodes must share a chunk, so a chunk holds at least 2 nodes
    k = (input_budget - fixed_cost) // max(max(node_costs), 1)
    if k < 2:
        logger.warning(
            'extract_edges: budget %d leaves room for %d nodes after fixed cost %d; '
            'using chunks of 2 (requests will exceed the budget)',
            input_budget,
            max(k, 0),
            fixed_cost,
        )
    k = min(max(k, 2), len(nodes) - 1)
    chunks = generate_covering_chunks(nodes, k)
    if len(chunks) > MAX_EDGE_EXTRACTION_CHUNKS:
        logger.warning(
            'extract_edges: %d nodes need %d covering chunks of %d (limit %d); '
            'sending one call over the input-token budget instead',
            len(nodes),
            len(chunks),
            k,
            MAX_EDGE_EXTRACTION_CHUNKS,
        )
        return None, k
    return chunks, k


async def _extract_edges_call(
    llm_client: LLMClient,
    context: dict[str, Any],
    previous_episodes: list[EpisodicNode],
    clients: GraphitiClients,
    max_tokens: int,
    group_id: str,
) -> tuple[list[ExtractedEdge], dict[str, Any], int | None]:
    """One extract_edges call; previous episodes fill the budget left by ``context``."""
    budget_tokens = remaining_budget_for_previous_episodes(
        llm_client,
        prompt_library.extract_edges.edge,
        context,
        response_model=ExtractedEdges,
    )
    call_context = dict(context)
    call_context['previous_episodes'] = build_previous_episodes_context(
        previous_episodes, budget_tokens
    )

    # Failures propagate to the caller
    llm_response = await generate_prompt_response(
        llm_client,
        'extract_edges.edge',
        prompt_library.extract_edges.edge,
        call_context,
        clients=clients,
        response_model=ExtractedEdges,
        max_tokens=max_tokens,
        group_id=group_id,
    )
    return ExtractedEdges(**llm_response).edges, call_context, budget_tokens


def _dedupe_extracted_edges(edges: list[ExtractedEdge]) -> list[ExtractedEdge]:
    # Pairs shared by several chunks can yield the same edge more than once. Unknown
    # entity names are kept and handled downstream, as for single-call output.
    deduped_edges: list[ExtractedEdge] = []
    seen_keys: set[tuple[str, str, str]] = set()
    for edge_data in edges:
        key = (
            edge_data.source_entity_name,
            edge_data.target_entity_name,
            _normalize_string_exact(edge_data.fact),
        )
        if key not in seen_keys:
            seen_keys.add(key)
            deduped_edges.append(edge_data)
    return deduped_edges


async def extract_edges_by_budget(
    llm_client: LLMClient,
    base_context: dict[str, Any],
    nodes: list[EntityNode],
    previous_episodes: list[EpisodicNode],
    *,
    clients: GraphitiClients,
    max_tokens: int,
    group_id: str,
) -> list[ExtractedEdge]:
    """Extract edges within the client's input-token budget.

    ``base_context`` is the extract_edges prompt context without nodes and previous
    episodes. All nodes go into one call when they fit next to the fixed prompt;
    otherwise the call is split over covering chunks so every node pair shares a call.
    """
    fixed_cost = _messages_cost(prompt_library.extract_edges.edge(base_context))
    fixed_cost += estimate_tokens(json.dumps(ExtractedEdges.model_json_schema()))

    nodes_context = [{'name': n.name, 'entity_types': n.labels} for n in nodes]
    node_costs = [estimate_tokens(json.dumps(entry)) for entry in nodes_context]
    chunks, k = _covering_chunks(llm_client.input_token_budget(), fixed_cost, nodes, node_costs)

    if chunks is None:
        edges, context, budget_tokens = await _extract_edges_call(
            llm_client,
            {**base_context, 'nodes': nodes_context},
            previous_episodes,
            clients,
            max_tokens,
            group_id,
        )
        logger.debug(
            'extract_edges: budget=%s episodes=%d/%d',
            budget_tokens,
            len(context['previous_episodes']),
            len(previous_episodes),
        )
        return edges

    all_edges_data: list[ExtractedEdge] = []
    for _chunk_items, chunk_indices in chunks:
        chunk_edges, _, _ = await _extract_edges_call(
            llm_client,
            {**base_context, 'nodes': [nodes_context[i] for i in chunk_indices]},
            previous_episodes,
            clients,
            max_tokens,
            group_id,
        )
        all_edges_data.extend(chunk_edges)

    deduped_edges = _dedupe_extracted_edges(all_edges_data)
    logger.debug(
        'extract_edges: %d nodes in %d covering chunks of <=%d, %d raw -> %d deduped edges',
        len(nodes),
        len(chunks),
        k,
        len(all_edges_data),
        len(deduped_edges),
    )
    return deduped_edges


def trim_resolution_candidates(
    llm_client: LLMClient,
    new_fact: str,
    related_edges: list[EntityEdge],
    existing_edges: list[EntityEdge],
) -> tuple[list[EntityEdge], list[EntityEdge]]:
    """Keep the highest-ranked candidates that fit the input-token budget.

    Related edges rank before invalidation candidates, so the kept candidates are a
    prefix of ``related_edges + existing_edges``. Returns the kept related edges and
    invalidation candidates; both are unchanged when there is no budget.
    """
    input_budget = llm_client.input_token_budget()
    if not isinstance(input_budget, int) or not (related_edges or existing_edges):
        return related_edges, existing_edges

    cost_context = {
        'existing_edges': [],
        'new_edge': new_fact,
        'edge_invalidation_candidates': [],
    }
    fixed_cost = _messages_cost(prompt_library.dedupe_edges.resolve_edge(cost_context))
    fixed_cost += estimate_tokens(json.dumps(EdgeDuplicate.model_json_schema()))

    offset = len(related_edges)
    all_candidates = [{'idx': i, 'fact': e.fact} for i, e in enumerate(related_edges)] + [
        {'idx': offset + i, 'fact': e.fact} for i, e in enumerate(existing_edges)
    ]
    batches = pack_by_budget(
        all_candidates,
        cost=lambda entry: estimate_tokens(json.dumps(entry)),
        budget=input_budget,
        fixed_cost=fixed_cost,
        max_items=len(all_candidates),
    )

    # Take the first batch (highest-ranked candidates)
    trimmed = batches[0] if batches else all_candidates[:1]
    kept_related = [related_edges[c['idx']] for c in trimmed if c['idx'] < offset]
    kept_invalidation = [existing_edges[c['idx'] - offset] for c in trimmed if c['idx'] >= offset]

    dropped_related = len(related_edges) - len(kept_related)
    dropped_invalidation = len(existing_edges) - len(kept_invalidation)
    if dropped_related or dropped_invalidation:
        # Dropped candidates cannot be marked duplicate or contradicted by this call
        logger.debug(
            'resolve_extracted_edge: input-token budget %d dropped %d of %d EXISTING FACTS '
            'and %d of %d INVALIDATION CANDIDATES (lowest ranked)',
            input_budget,
            dropped_related,
            len(related_edges),
            dropped_invalidation,
            len(existing_edges),
        )
    return kept_related, kept_invalidation
