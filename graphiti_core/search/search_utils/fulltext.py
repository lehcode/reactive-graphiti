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

from graphiti_core.driver.driver import GraphDriver, GraphProvider
from graphiti_core.edges import EntityEdge, get_entity_edge_from_record
from graphiti_core.graph_queries import get_nodes_query, get_relationships_query
from graphiti_core.helpers import lucene_sanitize, validate_group_ids
from graphiti_core.nodes import (
    CommunityNode,
    EntityNode,
    EpisodicNode,
    get_community_node_from_record,
    get_entity_node_from_record,
    get_episodic_node_from_record,
)
from graphiti_core.search.search_filters import (
    SearchFilters,
    edge_search_filter_query_constructor,
    node_search_filter_query_constructor,
)

from .utils import MAX_QUERY_LENGTH, RELEVANT_SCHEMA_LIMIT

logger = logging.getLogger(__name__)


def fulltext_query(query: str, group_ids: list[str] | int | None, driver: GraphDriver):
    if isinstance(group_ids, int):
        group_ids = [str(group_ids)]
    validate_group_ids(group_ids)

    if driver.provider == GraphProvider.KUZU:
        # Kuzu only supports simple queries.
        if len(query.split(' ')) > MAX_QUERY_LENGTH:
            return ''
        return query
    elif driver.provider == GraphProvider.FALKORDB:
        return driver.build_fulltext_query(query, group_ids, MAX_QUERY_LENGTH)
    group_ids_filter_list = (
        [driver.fulltext_syntax + f'group_id:"{g}"' for g in group_ids]
        if group_ids is not None
        else []
    )
    group_ids_filter = ''
    for f in group_ids_filter_list:
        group_ids_filter += f if not group_ids_filter else f' OR {f}'

    group_ids_filter += ' AND ' if group_ids_filter else ''

    lucene_query = lucene_sanitize(query)
    # If the lucene query is too long return no query
    if len(lucene_query.split(' ')) + len(group_ids or '') >= MAX_QUERY_LENGTH:
        return ''

    full_query = group_ids_filter + '(' + lucene_query + ')'

    return full_query


async def edge_fulltext_search(
    driver: GraphDriver,
    query: str,
    search_filter: SearchFilters,
    group_ids: list[str] | None = None,
    limit=RELEVANT_SCHEMA_LIMIT,
) -> list[EntityEdge]:
    if driver.search_interface:
        return await driver.search_interface.edge_fulltext_search(
            driver, query, search_filter, group_ids, limit
        )

    # fulltext search over facts
    fuzzy_query = fulltext_query(query, group_ids, driver)

    if fuzzy_query == '':
        return []

    match_query = """
    YIELD relationship AS rel, score
    MATCH (n:Entity)-[e:RELATES_TO {uuid: rel.uuid}]->(m:Entity)
    """
    if driver.provider == GraphProvider.KUZU:
        match_query = """
        YIELD node, score
        MATCH (n:Entity)-[:RELATES_TO]->(e:RelatesToNode_ {uuid: node.uuid})-[:RELATES_TO]->(m:Entity)
        """

    filter_queries, filter_params = edge_search_filter_query_constructor(
        search_filter, driver.provider
    )

    if group_ids is not None:
        filter_queries.append('e.group_id IN $group_ids')
        filter_params['group_ids'] = group_ids

    filter_query = ''
    if filter_queries:
        filter_query = ' WHERE ' + (' AND '.join(filter_queries))

    if driver.provider == GraphProvider.NEPTUNE:
        res = driver.run_aoss_query('edge_name_and_fact', query)  # pyright: ignore reportAttributeAccessIssue
        if res['hits']['total']['value'] > 0:
            input_ids = []
            for r in res['hits']['hits']:
                input_ids.append({'id': r['_source']['uuid'], 'score': r['_score']})

            # Match the edge ids and return the values
            query = (
                """
                                UNWIND $ids as id
                                MATCH (n:Entity)-[e:RELATES_TO]->(m:Entity)
                                WHERE e.group_id IN $group_ids
                                AND id(e)=id
                                """
                + filter_query
                + """
                AND id(e)=id
                WITH e, id.score as score, startNode(e) AS n, endNode(e) AS m
                RETURN
                    e.uuid AS uuid,
                    e.group_id AS group_id,
                    n.uuid AS source_node_uuid,
                    m.uuid AS target_node_uuid,
                    e.created_at AS created_at,
                    e.name AS name,
                    e.fact AS fact,
                    split(e.episodes, ",") AS episodes,
                    e.expired_at AS expired_at,
                    e.valid_at AS valid_at,
                    e.invalid_at AS invalid_at,
                    properties(e) AS attributes,
                    score
                ORDER BY score DESC
                LIMIT $limit
                """
            )
            records, _, _ = await driver.execute_query(
                query,
                ids=input_ids,
                group_ids=group_ids,
                limit=limit,
                routing_='r',
                **filter_params,
            )
            return [get_entity_edge_from_record(record, driver.provider) for record in records]
        return []

    records, _, _ = await driver.execute_query(
        get_relationships_query('edge_name_and_fact', limit, driver.provider)
        + match_query
        + filter_query
        + """
        RETURN
            e.uuid AS uuid,
            e.group_id AS group_id,
            n.uuid AS source_node_uuid,
            m.uuid AS target_node_uuid,
            e.created_at AS created_at,
            e.name AS name,
            e.fact AS fact,
            e.episodes AS episodes,
            e.expired_at AS expired_at,
            e.valid_at AS valid_at,
            e.invalid_at AS invalid_at,
            e.attributes AS attributes,
            score
        ORDER BY score DESC
        LIMIT $limit
        """,
        query=fuzzy_query,
        limit=limit,
        routing_='r',
        **filter_params,
    )

    return [get_entity_edge_from_record(record, driver.provider) for record in records]


async def node_fulltext_search(
    driver: GraphDriver,
    query: str,
    search_filter: SearchFilters,
    group_ids: list[str] | None = None,
    limit=RELEVANT_SCHEMA_LIMIT,
) -> list[EntityNode]:
    if driver.search_interface:
        return await driver.search_interface.node_fulltext_search(
            driver, query, search_filter, group_ids, limit
        )

    # fulltext search over facts
    fuzzy_query = fulltext_query(query, group_ids, driver)

    if fuzzy_query == '':
        return []

    filter_queries, filter_params = node_search_filter_query_constructor(
        search_filter, driver.provider
    )

    if group_ids is not None:
        filter_queries.append('n.group_id IN $group_ids')
        filter_params['group_ids'] = group_ids

    filter_query = ''
    if filter_queries:
        filter_query = ' WHERE ' + (' AND '.join(filter_queries))

    if driver.provider == GraphProvider.NEPTUNE:
        res = driver.run_aoss_query('node_name_and_summary', query)  # pyright: ignore reportAttributeAccessIssue
        if res['hits']['total']['value'] > 0:
            input_ids = []
            for r in res['hits']['hits']:
                input_ids.append({'id': r['_source']['uuid'], 'score': r['_score']})

            # Match the node ids and return the values
            query = (
                """
                                UNWIND $ids as id
                                MATCH (n:Entity)
                                WHERE n.group_id IN $group_ids
                                """
                + filter_query
                + """
                AND id(n)=id
                RETURN
                    n.uuid AS uuid,
                    n.group_id AS group_id,
                    n.name AS name,
                    n.summary AS summary,
                    n.created_at AS created_at,
                    labels(n) as labels,
                    properties(n) as attributes,
                    id.score as score
                ORDER BY score DESC
                LIMIT $limit
                """
            )
            records, _, _ = await driver.execute_query(
                query,
                ids=input_ids,
                group_ids=group_ids,
                limit=limit,
                routing_='r',
                **filter_params,
            )
            return [get_entity_node_from_record(record, driver.provider) for record in records]
        return []

    records, _, _ = await driver.execute_query(
        get_nodes_query('node_name_and_summary', '$query', limit, driver.provider)
        + """
        WITH node AS n, score
        """
        + filter_query
        + """
        RETURN
            n.uuid AS uuid,
            n.group_id AS group_id,
            n.name AS name,
            n.summary AS summary,
            n.created_at AS created_at,
            n.labels AS labels,
            n.attributes AS attributes,
            score
        ORDER BY score DESC
        LIMIT $limit
        """,
        query=fuzzy_query,
        limit=limit,
        routing_='r',
        **filter_params,
    )

    return [get_entity_node_from_record(record, driver.provider) for record in records]


async def community_fulltext_search(
    driver: GraphDriver,
    query: str,
    group_ids: list[str] | None = None,
    limit=RELEVANT_SCHEMA_LIMIT,
) -> list[CommunityNode]:
    if driver.search_interface:
        return await driver.search_interface.community_fulltext_search(
            driver, query, group_ids, limit
        )

    # fulltext search over facts
    fuzzy_query = fulltext_query(query, group_ids, driver)

    if fuzzy_query == '':
        return []

    filter_query = ''
    filter_params = {}
    if group_ids is not None:
        filter_query = ' WHERE c.group_id IN $group_ids'
        filter_params['group_ids'] = group_ids

    records, _, _ = await driver.execute_query(
        get_nodes_query('community_name_and_summary', '$query', limit, driver.provider)
        + """
        WITH node AS c, score
        """
        + filter_query
        + """
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
        """,
        query=fuzzy_query,
        limit=limit,
        routing_='r',
        **filter_params,
    )

    return [get_community_node_from_record(record) for record in records]


async def episode_fulltext_search(
    driver: GraphDriver,
    query: str,
    _search_filter: SearchFilters,
    group_ids: list[str] | None = None,
    limit=RELEVANT_SCHEMA_LIMIT,
) -> list[EpisodicNode]:
    if driver.search_interface:
        return await driver.search_interface.episode_fulltext_search(
            driver, query, group_ids, limit
        )

    # fulltext search over facts
    fuzzy_query = fulltext_query(query, group_ids, driver)

    if fuzzy_query == '':
        return []

    filter_query = ''
    filter_params = {}
    if group_ids is not None:
        filter_query = ' WHERE e.group_id IN $group_ids'
        filter_params['group_ids'] = group_ids

    records, _, _ = await driver.execute_query(
        get_nodes_query('episode_content', '$query', limit, driver.provider)
        + """
        WITH node AS e, score
        """
        + filter_query
        + """
        RETURN
            e.uuid AS uuid,
            e.group_id AS group_id,
            e.name AS name,
            e.content AS content,
            e.created_at AS created_at,
            e.valid_at AS valid_at,
            score
        ORDER BY score DESC
        LIMIT $limit
        """,
        query=fuzzy_query,
        limit=limit,
        routing_='r',
        **filter_params,
    )

    return [get_episodic_node_from_record(record) for record in records]
