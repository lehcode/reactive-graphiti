"""
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

from graphiti_core.driver.driver import GraphDriver, GraphProvider
from graphiti_core.edges import EntityEdge, get_entity_edge_from_record
from graphiti_core.graph_queries import get_vector_cosine_func_query
from graphiti_core.nodes import (
    CommunityNode,
    EntityNode,
    get_community_node_from_record,
    get_entity_node_from_record,
)
from graphiti_core.search.search_filters import (
    SearchFilters,
    edge_search_filter_query_constructor,
    node_search_filter_query_constructor,
)

from .utils import DEFAULT_MIN_SCORE, RELEVANT_SCHEMA_LIMIT

logger = logging.getLogger(__name__)


async def edge_similarity_search(
    driver: GraphDriver,
    search_vector: list[float],
    source_node_uuid: str | None,
    target_node_uuid: str | None,
    search_filter: SearchFilters,
    group_ids: list[str] | None = None,
    limit: int = RELEVANT_SCHEMA_LIMIT,
    min_score: float = DEFAULT_MIN_SCORE,
) -> list[EntityEdge]:
    if driver.search_interface:
        return await driver.search_interface.edge_similarity_search(
            driver,
            search_vector,
            source_node_uuid,
            target_node_uuid,
            search_filter,
            group_ids,
            limit,
            min_score,
        )

    filter_queries, filter_params = edge_search_filter_query_constructor(
        search_filter, driver.provider
    )

    if group_ids is not None:
        filter_queries.append('e.group_id IN $group_ids')
        filter_params['group_ids'] = group_ids

    if source_node_uuid is not None:
        filter_queries.append('n.uuid = $source_node_uuid')
        filter_params['source_node_uuid'] = source_node_uuid

    if target_node_uuid is not None:
        filter_queries.append('m.uuid = $target_node_uuid')
        filter_params['target_node_uuid'] = target_node_uuid

    filter_query = ''
    if filter_queries:
        filter_query = ' WHERE ' + (' AND '.join(filter_queries))

    if driver.provider == GraphProvider.KUZU:
        embedding_size = len(search_vector)
        query = (
            """
            MATCH (n:Entity)-[:RELATES_TO]->(e:RelatesToNode_)-[:RELATES_TO]->(m:Entity)
            """
            + filter_query
            + """
            WITH e, n, m, """
            + get_vector_cosine_func_query(
                'e.fact_embedding', f'CAST($embedding AS FLOAT[{embedding_size}])', driver.provider
            )
            + """ AS score
            WHERE score > $min_score
            RETURN
                e.uuid AS uuid,
                n.uuid AS source_node_uuid,
                m.uuid AS target_node_uuid,
                e.created_at AS created_at,
                e.name AS name,
                e.group_id AS group_id,
                e.fact AS fact,
                e.fact_embedding AS fact_embedding,
                e.episodes AS episodes,
                e.expired_at AS expired_at,
                e.valid_at AS valid_at,
                e.invalid_at AS invalid_at,
                e.attributes AS attributes,
                score
            ORDER BY score DESC
            LIMIT $limit
            """
        )
    else:
        query = (
            """
            MATCH (n:Entity)-[e:RELATES_TO]->(m:Entity)
            """
            + filter_query
            + """
            WITH e, n, m, """
            + get_vector_cosine_func_query('e.fact_embedding', '$embedding', driver.provider)
            + """ AS score
            WHERE score > $min_score
            RETURN
                e.uuid AS uuid,
                startNode(e).uuid AS source_node_uuid,
                endNode(e).uuid AS target_node_uuid,
                e.created_at AS created_at,
                e.name AS name,
                e.group_id AS group_id,
                e.fact AS fact,
                e.fact_embedding AS fact_embedding,
                e.episodes AS episodes,
                e.expired_at AS expired_at,
                e.valid_at AS valid_at,
                e.invalid_at AS invalid_at,
                properties(e) AS attributes,
                score
            ORDER BY score DESC
            LIMIT $limit
            """
        )

    results, _, _ = await driver.execute_query(
        query,
        embedding=search_vector,
        limit=limit,
        min_score=min_score,
        routing_='r',
        **filter_params,
    )

    return [get_entity_edge_from_record(record, driver.provider) for record in results]


async def node_similarity_search(
    driver: GraphDriver,
    search_vector: list[float],
    search_filter: SearchFilters,
    group_ids: list[str] | None = None,
    limit: int = RELEVANT_SCHEMA_LIMIT,
    min_score: float = DEFAULT_MIN_SCORE,
) -> list[EntityNode]:
    if driver.search_interface:
        return await driver.search_interface.node_similarity_search(
            driver, search_vector, search_filter, group_ids, limit, min_score
        )

    filter_queries, filter_params = node_search_filter_query_constructor(
        search_filter, driver.provider
    )

    if group_ids is not None:
        filter_queries.append('n.group_id IN $group_ids')
        filter_params['group_ids'] = group_ids

    filter_query = ''
    if filter_queries:
        filter_query = ' WHERE ' + (' AND '.join(filter_queries))

    if driver.provider == GraphProvider.KUZU:
        embedding_size = len(search_vector)
        query = (
            """
            MATCH (n:Entity)
            """
            + filter_query
            + """
            WITH n, """
            + get_vector_cosine_func_query(
                'n.name_embedding', f'CAST($embedding AS FLOAT[{embedding_size}])', driver.provider
            )
            + """ AS score
            WHERE score > $min_score
            RETURN
                n.uuid AS uuid,
                n.name AS name,
                n.name_embedding AS name_embedding,
                n.group_id AS group_id,
                n.created_at AS created_at,
                n.summary AS summary,
                n.labels AS labels,
                n.attributes AS attributes,
                score
            ORDER BY score DESC
            LIMIT $limit
            """
        )
    else:
        query = (
            """
            MATCH (n:Entity)
            """
            + filter_query
            + """
            WITH n, """
            + get_vector_cosine_func_query('n.name_embedding', '$embedding', driver.provider)
            + """ AS score
            WHERE score > $min_score
            RETURN
                n.uuid AS uuid,
                n.name AS name,
                n.name_embedding AS name_embedding,
                n.group_id AS group_id,
                n.created_at AS created_at,
                n.summary AS summary,
                labels(n) AS labels,
                properties(n) AS attributes,
                score
            ORDER BY score DESC
            LIMIT $limit
            """
        )

    results, _, _ = await driver.execute_query(
        query,
        embedding=search_vector,
        limit=limit,
        min_score=min_score,
        routing_='r',
        **filter_params,
    )

    return [get_entity_node_from_record(record, driver.provider) for record in results]


async def community_similarity_search(
    driver: GraphDriver,
    search_vector: list[float],
    group_ids: list[str] | None = None,
    limit: int = RELEVANT_SCHEMA_LIMIT,
    min_score: float = DEFAULT_MIN_SCORE,
) -> list[CommunityNode]:
    if driver.search_interface:
        return await driver.search_interface.community_similarity_search(
            driver, search_vector, group_ids, limit, min_score
        )

    filter_query = ''
    filter_params = {}
    if group_ids is not None:
        filter_query = ' WHERE c.group_id IN $group_ids'
        filter_params['group_ids'] = group_ids

    query = (
        """
        MATCH (c:Community)
        """
        + filter_query
        + """
        WITH c, """
        + get_vector_cosine_func_query('c.name_embedding', '$embedding', driver.provider)
        + """ AS score
        WHERE score > $min_score
        RETURN
            c.uuid AS uuid,
            c.group_id AS group_id,
            c.name AS name,
            c.summary AS summary,
            c.created_at AS created_at,
            c.explanation AS explanation,
            c.full_summary AS full_summary,
            c.level AS level,
            score
        ORDER BY score DESC
        LIMIT $limit
        """
    )

    results, _, _ = await driver.execute_query(
        query,
        embedding=search_vector,
        limit=limit,
        min_score=min_score,
        routing_='r',
        **filter_params,
    )

    return [get_community_node_from_record(record) for record in results]


async def get_embeddings_for_nodes(
    driver: GraphDriver, nodes: list[EntityNode]
) -> dict[str, list[float]]:
    if driver.graph_operations_interface:
        return await driver.graph_operations_interface.node_load_embeddings_bulk(driver, nodes)
    elif driver.provider == GraphProvider.NEPTUNE:
        query = """
        MATCH (n:Entity)
        WHERE n.uuid IN $node_uuids
        RETURN DISTINCT
            n.uuid AS uuid,
            split(n.name_embedding, ",") AS name_embedding
        """
    else:
        query = """
        MATCH (n:Entity)
        WHERE n.uuid IN $node_uuids
        RETURN DISTINCT
            n.uuid AS uuid,
            n.name_embedding AS name_embedding
        """
    results, _, _ = await driver.execute_query(
        query,
        node_uuids=[node.uuid for node in nodes],
        routing_='r',
    )

    embeddings_dict: dict[str, list[float]] = {}
    for result in results:
        uuid: str = result.get('uuid')
        embedding: list[float] = result.get('name_embedding')
        if uuid is not None and embedding is not None:
            embeddings_dict[uuid] = embedding

    return embeddings_dict


async def get_embeddings_for_communities(
    driver: GraphDriver, communities: list[CommunityNode]
) -> dict[str, list[float]]:
    if driver.search_interface:
        try:
            return await driver.search_interface.get_embeddings_for_communities(driver, communities)
        except NotImplementedError:
            pass

    if driver.provider == GraphProvider.NEPTUNE:
        query = """
        MATCH (c:Community)
        WHERE c.uuid IN $community_uuids
        RETURN DISTINCT
            c.uuid AS uuid,
            split(c.name_embedding, ",") AS name_embedding
        """
    else:
        query = """
        MATCH (c:Community)
        WHERE c.uuid IN $community_uuids
        RETURN DISTINCT
            c.uuid AS uuid,
            c.name_embedding AS name_embedding
        """
    results, _, _ = await driver.execute_query(
        query,
        community_uuids=[community.uuid for community in communities],
        routing_='r',
    )

    embeddings_dict: dict[str, list[float]] = {}
    for result in results:
        uuid: str = result.get('uuid')
        embedding: list[float] = result.get('name_embedding')
        if uuid is not None and embedding is not None:
            embeddings_dict[uuid] = embedding

    return embeddings_dict


async def get_embeddings_for_edges(
    driver: GraphDriver, edges: list[EntityEdge]
) -> dict[str, list[float]]:
    if driver.graph_operations_interface:
        return await driver.graph_operations_interface.edge_load_embeddings_bulk(driver, edges)
    elif driver.provider == GraphProvider.NEPTUNE:
        query = """
        MATCH (n:Entity)-[e:RELATES_TO]-(m:Entity)
        WHERE e.uuid IN $edge_uuids
        RETURN DISTINCT
            e.uuid AS uuid,
            split(e.fact_embedding, ",") AS fact_embedding
        """
    else:
        match_query = """
            MATCH (n:Entity)-[e:RELATES_TO]-(m:Entity)
        """
        if driver.provider == GraphProvider.KUZU:
            match_query = """
                MATCH (n:Entity)-[:RELATES_TO]-(e:RelatesToNode_)-[:RELATES_TO]-(m:Entity)
            """

        query = (
            match_query
            + """
        WHERE e.uuid IN $edge_uuids
        RETURN DISTINCT
            e.uuid AS uuid,
            e.fact_embedding AS fact_embedding
        """
        )
    results, _, _ = await driver.execute_query(
        query,
        edge_uuids=[edge.uuid for edge in edges],
        routing_='r',
    )

    embeddings_dict: dict[str, list[float]] = {}
    for result in results:
        uuid: str = result.get('uuid')
        embedding: list[float] = result.get('fact_embedding')
        if uuid is not None and embedding is not None:
            embeddings_dict[uuid] = embedding

    return embeddings_dict
