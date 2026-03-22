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
from time import time

from graphiti_core.driver.driver import GraphDriver, GraphProvider
from graphiti_core.edges import EntityEdge, get_entity_edge_from_record
from graphiti_core.graph_queries import get_nodes_query, get_vector_cosine_func_query
from graphiti_core.helpers import semaphore_gather
from graphiti_core.models.nodes.node_db_queries import COMMUNITY_NODE_RETURN
from graphiti_core.nodes import (
    CommunityNode,
    EntityNode,
    EpisodicNode,
    get_community_node_from_record,
    get_entity_node_from_record,
)
from graphiti_core.search.search_filters import (
    SearchFilters,
    edge_search_filter_query_constructor,
    node_search_filter_query_constructor,
)

from .fulltext import fulltext_query, node_fulltext_search
from .rerankers import rrf
from .utils import DEFAULT_MIN_SCORE, RELEVANT_SCHEMA_LIMIT

logger = logging.getLogger(__name__)


async def get_episodes_by_mentions(
    driver: GraphDriver,
    nodes: list[EntityNode],
    edges: list[EntityEdge],
    limit: int = RELEVANT_SCHEMA_LIMIT,
) -> list[EpisodicNode]:
    episode_uuids: list[str] = []
    for edge in edges:
        episode_uuids.extend(edge.episodes)

    episodes = await EpisodicNode.get_by_uuids(driver, episode_uuids[:limit])

    return episodes


async def get_mentioned_nodes(
    driver: GraphDriver, episodes: list[EpisodicNode]
) -> list[EntityNode]:
    if driver.graph_operations_interface:
        try:
            return await driver.graph_operations_interface.get_mentioned_nodes(driver, episodes)
        except NotImplementedError:
            pass

    episode_uuids = [episode.uuid for episode in episodes]

    from graphiti_core.models.nodes.node_db_queries import get_entity_node_return_query

    records, _, _ = await driver.execute_query(
        """
        MATCH (episode:Episodic)-[:MENTIONS]->(n:Entity)
        WHERE episode.uuid IN $uuids
        RETURN DISTINCT
        """
        + get_entity_node_return_query(driver.provider),
        uuids=episode_uuids,
        routing_='r',
    )

    nodes = [get_entity_node_from_record(record, driver.provider) for record in records]

    return nodes


async def get_communities_by_nodes(
    driver: GraphDriver, nodes: list[EntityNode]
) -> list[CommunityNode]:
    if driver.graph_operations_interface:
        try:
            return await driver.graph_operations_interface.get_communities_by_nodes(driver, nodes)
        except NotImplementedError:
            pass

    node_uuids = [node.uuid for node in nodes]

    records, _, _ = await driver.execute_query(
        """
        MATCH (c:Community)-[:HAS_MEMBER]->(m:Entity)
        WHERE m.uuid IN $uuids
        RETURN DISTINCT
        """
        + COMMUNITY_NODE_RETURN,
        uuids=node_uuids,
        routing_='r',
    )

    communities = [get_community_node_from_record(record) for record in records]

    return communities


async def hybrid_node_search(
    driver: GraphDriver,
    embeddings: list[list[float]],
    queries: list[str],
    search_filter: SearchFilters,
    group_ids: list[str] | None = None,
    limit: int = RELEVANT_SCHEMA_LIMIT,
) -> list[EntityNode]:
    start = time()
    results: list[list[EntityNode]] = list(
        await semaphore_gather(
            *[
                node_fulltext_search(driver, q, search_filter, group_ids, 2 * limit)
                for q in queries
            ],
            *[
                driver.search_interface.node_similarity_search(
                    driver, e, search_filter, group_ids, 2 * limit, 0.0
                )
                if driver.search_interface
                else node_similarity_search(driver, e, search_filter, group_ids, 2 * limit, 0.0)
                for e in embeddings
            ],
        )
    )

    node_uuid_map: dict[str, EntityNode] = {
        node.uuid: node for result in results for node in result
    }
    result_uuids = [[node.uuid for node in result] for result in results]

    ranked_uuids, _ = rrf(result_uuids)

    relevant_nodes: list[EntityNode] = [node_uuid_map[uuid] for uuid in ranked_uuids]

    end = time()
    logger.debug(f'Found relevant nodes: {ranked_uuids} in {(end - start) * 1000} ms')
    return relevant_nodes


# Helper for hybrid search
async def node_similarity_search(driver, search_vector, search_filter, group_ids, limit, min_score):
    from .similarity import node_similarity_search as ns

    return await ns(driver, search_vector, search_filter, group_ids, limit, min_score)


async def get_relevant_nodes(
    driver: GraphDriver,
    nodes: list[EntityNode],
    search_filter: SearchFilters,
    min_score: float = DEFAULT_MIN_SCORE,
    limit: int = RELEVANT_SCHEMA_LIMIT,
) -> list[list[EntityNode]]:
    if len(nodes) == 0:
        return []

    group_id = nodes[0].group_id
    query_nodes = [
        {
            'uuid': node.uuid,
            'name': node.name,
            'name_embedding': node.name_embedding,
            'fulltext_query': fulltext_query(node.name, [node.group_id], driver),
        }
        for node in nodes
    ]

    filter_queries, filter_params = node_search_filter_query_constructor(
        search_filter, driver.provider
    )

    filter_query = ''
    if filter_queries:
        filter_query = 'WHERE ' + (' AND '.join(filter_queries))

    if driver.provider == GraphProvider.KUZU:
        embedding_size = len(nodes[0].name_embedding) if nodes[0].name_embedding is not None else 0
        if embedding_size == 0:
            return []

        # FIXME: Kuzu currently does not support using variables such as `node.fulltext_query` as an input to FTS
        query = (
            """
            UNWIND $nodes AS node
            MATCH (n:Entity {group_id: $group_id})
            """
            + filter_query
            + """
            WITH node, n, """
            + get_vector_cosine_func_query(
                'n.name_embedding',
                f'CAST(node.name_embedding AS FLOAT[{embedding_size}])',
                driver.provider,
            )
            + """ AS score
            WHERE score > $min_score
            WITH node, collect(n)[:$limit] AS top_vector_nodes, collect(n.uuid) AS vector_node_uuids
            """
            + get_nodes_query(
                'node_name_and_summary',
                'node.fulltext_query',
                limit=limit,
                provider=driver.provider,
            )
            + """
            WITH node AS m
            WHERE m.group_id = $group_id AND NOT m.uuid IN vector_node_uuids
            WITH node, top_vector_nodes, collect(m) AS fulltext_nodes

            WITH node, list_concat(top_vector_nodes, fulltext_nodes) AS combined_nodes

            UNWIND combined_nodes AS x
            WITH node, collect(DISTINCT {
                uuid: x.uuid,
                name: x.name,
                name_embedding: x.name_embedding,
                group_id: x.group_id,
                created_at: x.created_at,
                summary: x.summary,
                labels: x.labels,
                attributes: x.attributes
            }) AS matches

            RETURN
            node.uuid AS search_node_uuid, matches
            """
        )
    else:
        query = (
            """
            UNWIND $nodes AS node
            MATCH (n:Entity {group_id: $group_id})
            """
            + filter_query
            + """
            WITH node, n, """
            + get_vector_cosine_func_query(
                'n.name_embedding', 'node.name_embedding', driver.provider
            )
            + """ AS score
            WHERE score > $min_score
            WITH node, collect(n)[..$limit] AS top_vector_nodes, collect(n.uuid) AS vector_node_uuids
            """
            + get_nodes_query(
                'node_name_and_summary',
                'node.fulltext_query',
                limit=limit,
                provider=driver.provider,
            )
            + """
            YIELD node AS m
            WHERE m.group_id = $group_id
            WITH node, top_vector_nodes, vector_node_uuids, collect(m) AS fulltext_nodes

            WITH node,
                top_vector_nodes,
                [m IN fulltext_nodes WHERE NOT m.uuid IN vector_node_uuids] AS filtered_fulltext_nodes

            WITH node, top_vector_nodes + filtered_fulltext_nodes AS combined_nodes

            UNWIND combined_nodes AS combined_node
            WITH node, collect(DISTINCT combined_node) AS deduped_nodes

            RETURN
            node.uuid AS search_node_uuid,
            [x IN deduped_nodes | {
                uuid: x.uuid,
                name: x.name,
                name_embedding: x.name_embedding,
                group_id: x.group_id,
                created_at: x.created_at,
                summary: x.summary,
                labels: labels(x),
                attributes: properties(x)
            }] AS matches
            """
        )

    results, _, _ = await driver.execute_query(
        query,
        nodes=query_nodes,
        group_id=group_id,
        limit=limit,
        min_score=min_score,
        routing_='r',
        **filter_params,
    )

    relevant_nodes_dict: dict[str, list[EntityNode]] = {
        result['search_node_uuid']: [
            get_entity_node_from_record(record, driver.provider) for record in result['matches']
        ]
        for result in results
    }

    relevant_nodes = [relevant_nodes_dict.get(node.uuid, []) for node in nodes]

    return relevant_nodes


async def get_relevant_edges(
    driver: GraphDriver,
    edges: list[EntityEdge],
    search_filter: SearchFilters,
    min_score: float = DEFAULT_MIN_SCORE,
    limit: int = RELEVANT_SCHEMA_LIMIT,
) -> list[list[EntityEdge]]:
    if len(edges) == 0:
        return []

    filter_queries, filter_params = edge_search_filter_query_constructor(
        search_filter, driver.provider
    )

    filter_query = ''
    if filter_queries:
        filter_query = ' WHERE ' + (' AND '.join(filter_queries))

    if driver.provider == GraphProvider.NEPTUNE:
        query = (
            """
            UNWIND $edges AS edge
            MATCH (n:Entity {uuid: edge.source_node_uuid})-[e:RELATES_TO {group_id: edge.group_id}]-(m:Entity {uuid: edge.target_node_uuid})
            """
            + filter_query
            + """
            WITH e, edge
            RETURN DISTINCT id(e) as id, e.fact_embedding as source_embedding, edge.uuid as search_edge_uuid,
            edge.fact_embedding as target_embedding
            """
        )
        resp, _, _ = await driver.execute_query(
            query,
            edges=[edge.model_dump() for edge in edges],
            limit=limit,
            min_score=min_score,
            routing_='r',
            **filter_params,
        )

        from .utils import calculate_cosine_similarity

        # Calculate Cosine similarity then return the edge ids
        input_ids = []
        for r in resp:
            score = calculate_cosine_similarity(
                list(map(float, r['source_embedding'].split(','))), r['target_embedding']
            )
            if score > min_score:
                input_ids.append({'id': r['id'], 'score': score, 'uuid': r['search_edge_uuid']})

        # Match the edge ides and return the values
        query = """
        UNWIND $ids AS edge
        MATCH ()-[e]->()
        WHERE id(e) = edge.id
        WITH edge, e
        ORDER BY edge.score DESC
        RETURN edge.uuid AS search_edge_uuid,
            collect({
                uuid: e.uuid,
                source_node_uuid: startNode(e).uuid,
                target_node_uuid: endNode(e).uuid,
                created_at: e.created_at,
                name: e.name,
                group_id: e.group_id,
                fact: e.fact,
                fact_embedding: [x IN split(e.fact_embedding, ",") | toFloat(x)],
                episodes: split(e.episodes, ","),
                expired_at: e.expired_at,
                valid_at: e.valid_at,
                invalid_at: e.invalid_at,
                attributes: properties(e)
            })[..$limit] AS matches
                """

        results, _, _ = await driver.execute_query(
            query,
            ids=input_ids,
            edges=[edge.model_dump() for edge in edges],
            limit=limit,
            min_score=min_score,
            routing_='r',
            **filter_params,
        )
    else:
        if driver.provider == GraphProvider.KUZU:
            embedding_size = (
                len(edges[0].fact_embedding) if edges[0].fact_embedding is not None else 0
            )
            if embedding_size == 0:
                return []

            query = (
                """
                UNWIND $edges AS edge
                MATCH (n:Entity {uuid: edge.source_node_uuid})-[:RELATES_TO]-(e:RelatesToNode_ {group_id: edge.group_id})-[:RELATES_TO]-(m:Entity {uuid: edge.target_node_uuid})
                """
                + filter_query
                + """
                WITH e, edge, n, m, """
                + get_vector_cosine_func_query(
                    'e.fact_embedding',
                    f'CAST(edge.fact_embedding AS FLOAT[{embedding_size}])',
                    driver.provider,
                )
                + """ AS score
                WHERE score > $min_score
                WITH e, edge, n, m, score
                ORDER BY score DESC
                LIMIT $limit
                RETURN
                    edge.uuid AS search_edge_uuid,
                    collect({
                        uuid: e.uuid,
                        source_node_uuid: n.uuid,
                        target_node_uuid: m.uuid,
                        created_at: e.created_at,
                        name: e.name,
                        group_id: e.group_id,
                        fact: e.fact,
                        fact_embedding: e.fact_embedding,
                        episodes: e.episodes,
                        expired_at: e.expired_at,
                        valid_at: e.valid_at,
                        invalid_at: e.invalid_at,
                        attributes: e.attributes
                    }) AS matches
                """
            )
        else:
            query = (
                """
                UNWIND $edges AS edge
                MATCH (n:Entity {uuid: edge.source_node_uuid})-[e:RELATES_TO {group_id: edge.group_id}]-(m:Entity {uuid: edge.target_node_uuid})
                """
                + filter_query
                + """
                WITH e, edge, """
                + get_vector_cosine_func_query(
                    'e.fact_embedding', 'edge.fact_embedding', driver.provider
                )
                + """ AS score
                WHERE score > $min_score
                WITH edge, e, score
                ORDER BY score DESC
                RETURN
                    edge.uuid AS search_edge_uuid,
                    collect({
                        uuid: e.uuid,
                        source_node_uuid: startNode(e).uuid,
                        target_node_uuid: endNode(e).uuid,
                        created_at: e.created_at,
                        name: e.name,
                        group_id: e.group_id,
                        fact: e.fact,
                        fact_embedding: e.fact_embedding,
                        episodes: e.episodes,
                        expired_at: e.expired_at,
                        valid_at: e.valid_at,
                        invalid_at: e.invalid_at,
                        attributes: properties(e)
                    })[..$limit] AS matches
                """
            )

        results, _, _ = await driver.execute_query(
            query,
            edges=[edge.model_dump() for edge in edges],
            limit=limit,
            min_score=min_score,
            routing_='r',
            **filter_params,
        )

    relevant_edges_dict: dict[str, list[EntityEdge]] = {
        result['search_edge_uuid']: [
            get_entity_edge_from_record(record, driver.provider) for record in result['matches']
        ]
        for result in results
    }

    relevant_edges = [relevant_edges_dict.get(edge.uuid, []) for edge in edges]

    return relevant_edges


async def get_edge_invalidation_candidates(
    driver: GraphDriver,
    edges: list[EntityEdge],
    search_filter: SearchFilters,
    min_score: float = DEFAULT_MIN_SCORE,
    limit: int = RELEVANT_SCHEMA_LIMIT,
) -> list[list[EntityEdge]]:
    if len(edges) == 0:
        return []

    filter_queries, filter_params = edge_search_filter_query_constructor(
        search_filter, driver.provider
    )

    filter_query = ''
    if filter_queries:
        filter_query = ' AND ' + (' AND '.join(filter_queries))

    if driver.provider == GraphProvider.NEPTUNE:
        query = (
            """
            UNWIND $edges AS edge
            MATCH (n:Entity)-[e:RELATES_TO {group_id: edge.group_id}]->(m:Entity)
            WHERE n.uuid IN [edge.source_node_uuid, edge.target_node_uuid] OR m.uuid IN [edge.target_node_uuid, edge.source_node_uuid]
            """
            + filter_query
            + """
            WITH e, edge
            RETURN DISTINCT id(e) as id, e.fact_embedding as source_embedding,
            edge.fact_embedding as target_embedding,
            edge.uuid as search_edge_uuid
            """
        )
        resp, _, _ = await driver.execute_query(
            query,
            edges=[edge.model_dump() for edge in edges],
            limit=limit,
            min_score=min_score,
            routing_='r',
            **filter_params,
        )

        from .utils import calculate_cosine_similarity

        # Calculate Cosine similarity then return the edge ids
        input_ids = []
        for r in resp:
            score = calculate_cosine_similarity(
                list(map(float, r['source_embedding'].split(','))), r['target_embedding']
            )
            if score > min_score:
                input_ids.append({'id': r['id'], 'score': score, 'uuid': r['search_edge_uuid']})

        # Match the edge ides and return the values
        query = """
        UNWIND $ids AS edge
        MATCH ()-[e]->()
        WHERE id(e) = edge.id
        WITH edge, e
        ORDER BY edge.score DESC
        RETURN edge.uuid AS search_edge_uuid,
            collect({
                uuid: e.uuid,
                source_node_uuid: startNode(e).uuid,
                target_node_uuid: endNode(e).uuid,
                created_at: e.created_at,
                name: e.name,
                group_id: e.group_id,
                fact: e.fact,
                fact_embedding: [x IN split(e.fact_embedding, ",") | toFloat(x)],
                episodes: split(e.episodes, ","),
                expired_at: e.expired_at,
                valid_at: e.valid_at,
                invalid_at: e.invalid_at,
                attributes: properties(e)
            })[..$limit] AS matches
                """
        results, _, _ = await driver.execute_query(
            query,
            ids=input_ids,
            edges=[edge.model_dump() for edge in edges],
            limit=limit,
            min_score=min_score,
            routing_='r',
            **filter_params,
        )
    else:
        if driver.provider == GraphProvider.KUZU:
            embedding_size = (
                len(edges[0].fact_embedding) if edges[0].fact_embedding is not None else 0
            )
            if embedding_size == 0:
                return []

            query = (
                """
                UNWIND $edges AS edge
                MATCH (n:Entity)-[:RELATES_TO]->(e:RelatesToNode_ {group_id: edge.group_id})-[:RELATES_TO]->(m:Entity)
                WHERE (n.uuid IN [edge.source_node_uuid, edge.target_node_uuid] OR m.uuid IN [edge.target_node_uuid, edge.source_node_uuid])
                """
                + filter_query
                + """
                WITH edge, e, n, m, """
                + get_vector_cosine_func_query(
                    'e.fact_embedding',
                    f'CAST(edge.fact_embedding AS FLOAT[{embedding_size}])',
                    driver.provider,
                )
                + """ AS score
                WHERE score > $min_score
                WITH edge, e, n, m, score
                ORDER BY score DESC
                LIMIT $limit
                RETURN
                    edge.uuid AS search_edge_uuid,
                    collect({
                        uuid: e.uuid,
                        source_node_uuid: n.uuid,
                        target_node_uuid: m.uuid,
                        created_at: e.created_at,
                        name: e.name,
                        group_id: e.group_id,
                        fact: e.fact,
                        fact_embedding: e.fact_embedding,
                        episodes: e.episodes,
                        expired_at: e.expired_at,
                        valid_at: e.valid_at,
                        invalid_at: e.invalid_at,
                        attributes: e.attributes
                    }) AS matches
                """
            )
        else:
            query = (
                """
                UNWIND $edges AS edge
                MATCH (n:Entity)-[e:RELATES_TO {group_id: edge.group_id}]->(m:Entity)
                WHERE n.uuid IN [edge.source_node_uuid, edge.target_node_uuid] OR m.uuid IN [edge.target_node_uuid, edge.source_node_uuid]
                """
                + filter_query
                + """
                WITH edge, e, """
                + get_vector_cosine_func_query(
                    'e.fact_embedding', 'edge.fact_embedding', driver.provider
                )
                + """ AS score
                WHERE score > $min_score
                WITH edge, e, score
                ORDER BY score DESC
                RETURN
                    edge.uuid AS search_edge_uuid,
                    collect({
                        uuid: e.uuid,
                        source_node_uuid: startNode(e).uuid,
                        target_node_uuid: endNode(e).uuid,
                        created_at: e.created_at,
                        name: e.name,
                        group_id: e.group_id,
                        fact: e.fact,
                        fact_embedding: e.fact_embedding,
                        episodes: e.episodes,
                        expired_at: e.expired_at,
                        valid_at: e.valid_at,
                        invalid_at: e.invalid_at,
                        attributes: properties(e)
                    })[..$limit] AS matches
                """
            )

        results, _, _ = await driver.execute_query(
            query,
            edges=[edge.model_dump() for edge in edges],
            limit=limit,
            min_score=min_score,
            routing_='r',
            **filter_params,
        )
    invalidation_edges_dict: dict[str, list[EntityEdge]] = {
        result['search_edge_uuid']: [
            get_entity_edge_from_record(record, driver.provider) for record in result['matches']
        ]
        for result in results
    }

    invalidation_edges = [invalidation_edges_dict.get(edge.uuid, []) for edge in edges]

    return invalidation_edges
