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
from graphiti_core.nodes import EntityNode, get_entity_node_from_record
from graphiti_core.search.search_filters import (
    SearchFilters,
    edge_search_filter_query_constructor,
    node_search_filter_query_constructor,
)

from .utils import RELEVANT_SCHEMA_LIMIT

logger = logging.getLogger(__name__)


async def edge_bfs_search(
    driver: GraphDriver,
    bfs_origin_node_uuids: list[str],
    bfs_max_depth: int,
    search_filter: SearchFilters,
    group_ids: list[str] | None = None,
    limit: int = RELEVANT_SCHEMA_LIMIT,
) -> list[EntityEdge]:
    if driver.search_interface:
        try:
            return await driver.search_interface.edge_bfs_search(
                driver, bfs_origin_node_uuids, bfs_max_depth, search_filter, group_ids, limit
            )
        except NotImplementedError:
            pass

    filter_queries, filter_params = edge_search_filter_query_constructor(
        search_filter, driver.provider
    )

    if group_ids is not None:
        filter_queries.append('e.group_id IN $group_ids')
        filter_params['group_ids'] = group_ids

    filter_query = ''
    if filter_queries:
        filter_query = ' WHERE ' + (' AND '.join(filter_queries))

    if driver.provider == GraphProvider.KUZU:
        query = (
            """
            MATCH (n:Entity)-[e:RELATES_TO*1.."""
            + str(bfs_max_depth)
            + """]-(m:Entity)
            WHERE n.uuid IN $node_uuids
            """
            + filter_query.replace(' WHERE ', ' AND ')
            + """
            RETURN DISTINCT
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
                e.attributes AS attributes
            LIMIT $limit
            """
        )
    else:
        query = (
            """
            MATCH (n:Entity)-[e:RELATES_TO*1.."""
            + str(bfs_max_depth)
            + """]-(m:Entity)
            WHERE n.uuid IN $node_uuids
            """
            + filter_query.replace(' WHERE ', ' AND ')
            + """
            WITH DISTINCT last(e) AS e
            RETURN
                e.uuid AS uuid,
                e.group_id AS group_id,
                startNode(e).uuid AS source_node_uuid,
                endNode(e).uuid AS target_node_uuid,
                e.created_at AS created_at,
                e.name AS name,
                e.fact AS fact,
                e.fact_embedding AS fact_embedding,
                e.episodes AS episodes,
                e.expired_at AS expired_at,
                e.valid_at AS valid_at,
                e.invalid_at AS invalid_at,
                properties(e) AS attributes
            LIMIT $limit
            """
        )

    records, _, _ = await driver.execute_query(
        query,
        node_uuids=bfs_origin_node_uuids,
        limit=limit,
        routing_='r',
        **filter_params,
    )

    return [get_entity_edge_from_record(record, driver.provider) for record in records]


async def node_bfs_search(
    driver: GraphDriver,
    bfs_origin_node_uuids: list[str],
    search_filter: SearchFilters,
    bfs_max_depth: int,
    group_ids: list[str] | None = None,
    limit: int = RELEVANT_SCHEMA_LIMIT,
) -> list[EntityNode]:
    if driver.search_interface:
        try:
            return await driver.search_interface.node_bfs_search(
                driver, bfs_origin_node_uuids, search_filter, bfs_max_depth, group_ids, limit
            )
        except NotImplementedError:
            pass

    filter_queries, filter_params = node_search_filter_query_constructor(
        search_filter, driver.provider
    )

    if group_ids is not None:
        filter_queries.append('m.group_id IN $group_ids')
        filter_params['group_ids'] = group_ids

    filter_query = ''
    if filter_queries:
        filter_query = ' WHERE ' + (' AND '.join(filter_queries))

    query = (
        """
        MATCH (n:Entity)-[:RELATES_TO*1.."""
        + str(bfs_max_depth)
        + """]-(m:Entity)
        WHERE n.uuid IN $node_uuids
        """
        + filter_query.replace(' WHERE ', ' AND ')
        + """
        RETURN DISTINCT
            m.uuid AS uuid,
            m.name AS name,
            m.name_embedding AS name_embedding,
            m.group_id AS group_id,
            m.created_at AS created_at,
            m.summary AS summary,
            labels(m) AS labels,
            properties(m) AS attributes
        LIMIT $limit
        """
    )
    if driver.provider == GraphProvider.KUZU:
        query = (
            """
            MATCH (n:Entity)-[:RELATES_TO*1.."""
            + str(bfs_max_depth)
            + """]-(m:Entity)
            WHERE n.uuid IN $node_uuids
            """
            + filter_query.replace(' WHERE ', ' AND ')
            + """
            RETURN DISTINCT
                m.uuid AS uuid,
                m.name AS name,
                m.name_embedding AS name_embedding,
                m.group_id AS group_id,
                m.created_at AS created_at,
                m.summary AS summary,
                m.labels AS labels,
                m.attributes AS attributes
            LIMIT $limit
            """
        )

    records, _, _ = await driver.execute_query(
        query,
        node_uuids=bfs_origin_node_uuids,
        limit=limit,
        routing_='r',
        **filter_params,
    )

    return [get_entity_node_from_record(record, driver.provider) for record in records]
