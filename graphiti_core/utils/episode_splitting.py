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

import logging
from typing import TYPE_CHECKING, Any

from graphiti_core.edges import CommunityEdge, EntityEdge, EpisodicEdge
from graphiti_core.helpers import EPISODE_CONTENT_BUDGET_RATIO
from graphiti_core.nodes import CommunityNode, EntityNode, EpisodeType, SagaNode
from graphiti_core.utils.content_chunking import (
    CHUNK_OVERLAP_TOKENS,
    chunk_json_content,
    chunk_message_content,
    chunk_text_content,
    estimate_tokens,
)

if TYPE_CHECKING:
    from graphiti_core.graphiti import AddEpisodeResults, Graphiti

logger = logging.getLogger(__name__)


def chunk_episode_body(episode_body: str, source: EpisodeType, chunk_size: int) -> list[str]:
    """Split an episode body with the chunker matching its source type."""
    # The chunkers treat 0 as "use the env default", so keep both sizes positive.
    overlap = max(1, min(CHUNK_OVERLAP_TOKENS, chunk_size // 10))

    if source == EpisodeType.json:
        chunker = chunk_json_content
    elif source == EpisodeType.message:
        chunker = chunk_message_content
    else:
        chunker = chunk_text_content

    return chunker(episode_body, chunk_size, overlap)


def aggregate_part_results(results: 'list[AddEpisodeResults]') -> 'AddEpisodeResults':
    """Merge per-part results; nodes and edges are deduplicated by uuid, keeping the last."""
    from graphiti_core.graphiti import AddEpisodeResults

    episodic_edges: list[EpisodicEdge] = []
    communities: list[CommunityNode] = []
    community_edges: list[CommunityEdge] = []
    node_dedup: dict[str, EntityNode] = {}
    edge_dedup: dict[str, EntityEdge] = {}
    for r in results:
        episodic_edges.extend(r.episodic_edges)
        communities.extend(r.communities)
        community_edges.extend(r.community_edges)
        for node in r.nodes:
            node_dedup[node.uuid] = node
        for edge in r.edges:
            edge_dedup[edge.uuid] = edge

    return AddEpisodeResults(
        episode=results[-1].episode,
        episodic_edges=episodic_edges,
        nodes=list(node_dedup.values()),
        edges=list(edge_dedup.values()),
        communities=communities,
        community_edges=community_edges,
    )


async def run_episode_split(
    graphiti: 'Graphiti',
    *,
    name: str,
    episode_body: str,
    source: EpisodeType,
    budget: int,
    previous_episode_uuids: list[str] | None,
    saga: str | SagaNode | None,
    saga_previous_episode_uuid: str | None,
    **episode_kwargs: Any,
) -> 'AddEpisodeResults':
    """Split an oversized episode body into parts and process them sequentially.

    Each part is named ``'{name} (part i/N)'``.  Parts join ``saga`` if the caller
    provided one; otherwise a saga named ``name`` is created.  ``episode_kwargs`` are
    forwarded unchanged to ``Graphiti._add_episode_internal`` for every part.
    """
    ratio = EPISODE_CONTENT_BUDGET_RATIO
    parts = chunk_episode_body(episode_body, source, max(1, int(ratio * budget)))
    num_parts = len(parts)

    # If chunker returns a single part, process normally (no recursion)
    if num_parts == 1:
        return await graphiti._add_episode_internal(
            name=name,
            episode_body=episode_body,
            source=source,
            uuid=None,
            previous_episode_uuids=previous_episode_uuids,
            saga=saga,
            saga_previous_episode_uuid=saga_previous_episode_uuid,
            **episode_kwargs,
        )

    part_saga = saga if saga is not None else name
    prev_uuids = list(previous_episode_uuids) if previous_episode_uuids is not None else None

    # Log once when splitting
    logger.info(
        f'Splitting episode "{name}": estimated_tokens={estimate_tokens(episode_body)}, '
        f'budget={budget}, threshold={int(ratio * budget)}, parts={num_parts}'
    )

    results: list[AddEpisodeResults] = []
    for i, part_body in enumerate(parts):
        part_saga_previous = saga_previous_episode_uuid if i == 0 else results[-1].episode.uuid
        part_results = await graphiti._add_episode_internal(
            name=f'{name} (part {i + 1}/{num_parts})',
            episode_body=part_body,
            source=source,
            uuid=None,
            previous_episode_uuids=list(prev_uuids) if prev_uuids is not None else None,
            saga=part_saga,
            saga_previous_episode_uuid=part_saga_previous,
            **episode_kwargs,
        )
        results.append(part_results)

        # Accumulate previous-episode uuids for subsequent parts
        if prev_uuids is not None:
            prev_uuids.append(part_results.episode.uuid)

    return aggregate_part_results(results)
