"""
Copyright 2024, Zep Software, Inc.

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
from collections import defaultdict
from time import time

import numpy as np
from numpy._typing import NDArray

from graphiti_core.driver.driver import GraphDriver, GraphProvider
from graphiti_core.helpers import normalize_l2

from .utils import DEFAULT_MMR_LAMBDA

logger = logging.getLogger(__name__)


# takes in a list of rankings of uuids
def rrf(
    results: list[list[str]], rank_const=1, min_score: float = 0
) -> tuple[list[str], list[float]]:
    scores: dict[str, float] = defaultdict(float)
    for result in results:
        for i, uuid in enumerate(result):
            scores[uuid] += 1 / (i + rank_const)

    scored_uuids = [term for term in scores.items()]
    scored_uuids.sort(reverse=True, key=lambda term: term[1])

    sorted_uuids = [term[0] for term in scored_uuids]

    return [uuid for uuid in sorted_uuids if scores[uuid] >= min_score], [
        scores[uuid] for uuid in sorted_uuids if scores[uuid] >= min_score
    ]


async def node_distance_reranker(
    driver: GraphDriver,
    node_uuids: list[str],
    center_node_uuid: str,
    min_score: float = 0,
) -> tuple[list[str], list[float]]:
    if driver.search_interface:
        try:
            return await driver.search_interface.node_distance_reranker(
                driver, node_uuids, center_node_uuid, min_score
            )
        except NotImplementedError:
            pass

    # filter out node_uuid center node node uuid
    filtered_uuids = list(filter(lambda node_uuid: node_uuid != center_node_uuid, node_uuids))
    scores: dict[str, float] = {center_node_uuid: 0.0}

    query = """
    UNWIND $node_uuids AS node_uuid
    MATCH (center:Entity {uuid: $center_uuid})-[:RELATES_TO]-(n:Entity {uuid: node_uuid})
    RETURN 1 AS score, node_uuid AS uuid
    """
    if driver.provider == GraphProvider.KUZU:
        query = """
        UNWIND $node_uuids AS node_uuid
        MATCH (center:Entity {uuid: $center_uuid})-[:RELATES_TO]->(e:RelatesToNode_)-[:RELATES_TO]->(n:Entity {uuid: node_uuid})
        RETURN 1 AS score, node_uuid AS uuid
        """

    # Find the shortest path to center node
    results, header, _ = await driver.execute_query(
        query,
        node_uuids=filtered_uuids,
        center_uuid=center_node_uuid,
        routing_='r',
    )
    if driver.provider == GraphProvider.FALKORDB:
        results = [dict(zip(header, row, strict=True)) for row in results]

    for result in results:
        uuid = result['uuid']
        score = result['score']
        scores[uuid] = score

    for uuid in filtered_uuids:
        if uuid not in scores:
            scores[uuid] = float('inf')

    # rerank on shortest distance
    filtered_uuids.sort(key=lambda cur_uuid: scores[cur_uuid])

    # add back in filtered center uuid if it was filtered out
    if center_node_uuid in node_uuids:
        scores[center_node_uuid] = 0.1
        filtered_uuids = [center_node_uuid] + filtered_uuids

    return [uuid for uuid in filtered_uuids if (1 / scores[uuid]) >= min_score], [
        1 / scores[uuid] for uuid in filtered_uuids if (1 / scores[uuid]) >= min_score
    ]


async def episode_mentions_reranker(
    driver: GraphDriver, node_uuids: list[list[str]], min_score: float = 0
) -> tuple[list[str], list[float]]:
    if driver.search_interface:
        try:
            return await driver.search_interface.episode_mentions_reranker(
                driver, node_uuids, min_score
            )
        except NotImplementedError:
            pass

    # use rrf as a preliminary ranker
    sorted_uuids, _ = rrf(node_uuids)
    scores: dict[str, float] = {}

    # Find the shortest path to center node
    results, _, _ = await driver.execute_query(
        """
        UNWIND $node_uuids AS node_uuid
        MATCH (episode:Episodic)-[r:MENTIONS]->(n:Entity {uuid: node_uuid})
        RETURN count(*) AS score, n.uuid AS uuid
        """,
        node_uuids=sorted_uuids,
        routing_='r',
    )

    for result in results:
        scores[result['uuid']] = result['score']

    for uuid in sorted_uuids:
        if uuid not in scores:
            scores[uuid] = float('inf')

    # rerank on shortest distance
    sorted_uuids.sort(key=lambda cur_uuid: scores[cur_uuid])

    return [uuid for uuid in sorted_uuids if scores[uuid] >= min_score], [
        scores[uuid] for uuid in sorted_uuids if scores[uuid] >= min_score
    ]


def maximal_marginal_relevance(
    query_vector: list[float],
    candidates: dict[str, list[float]],
    mmr_lambda: float = DEFAULT_MMR_LAMBDA,
    min_score: float = -2.0,
) -> tuple[list[str], list[float]]:
    start = time()
    query_array = np.array(query_vector)
    candidate_arrays: dict[str, NDArray] = {}
    for uuid, embedding in candidates.items():
        candidate_arrays[uuid] = normalize_l2(embedding)

    uuids: list[str] = list(candidate_arrays.keys())

    similarity_matrix = np.zeros((len(uuids), len(uuids)))

    for i, uuid_1 in enumerate(uuids):
        for j, uuid_2 in enumerate(uuids[:i]):
            u = candidate_arrays[uuid_1]
            v = candidate_arrays[uuid_2]
            similarity = np.dot(u, v)

            similarity_matrix[i, j] = similarity
            similarity_matrix[j, i] = similarity

    mmr_scores: dict[str, float] = {}
    for i, uuid in enumerate(uuids):
        max_sim = np.max(similarity_matrix[i, :])
        mmr = mmr_lambda * np.dot(query_array, candidate_arrays[uuid]) + (mmr_lambda - 1) * max_sim
        mmr_scores[uuid] = mmr

    uuids.sort(reverse=True, key=lambda c: mmr_scores[c])

    end = time()
    logger.debug(f'Completed MMR reranking in {(end - start) * 1000} ms')

    return [uuid for uuid in uuids if mmr_scores[uuid] >= min_score], [
        mmr_scores[uuid] for uuid in uuids if mmr_scores[uuid] >= min_score
    ]
