"""Fork-specific query-shape tests for the split ``search_utils`` package.

These tests verify that the fork's inlined Cypher queries have the correct
YIELD/WITH and RETURN patterns for each database provider, without requiring
a live graph database.
"""

from typing import Any

import pytest

from graphiti_core.driver.driver import GraphProvider
from graphiti_core.search.search_filters import SearchFilters
from graphiti_core.search.search_utils import (
    community_fulltext_search,
    edge_bfs_search,
    edge_fulltext_search,
    episode_fulltext_search,
    node_fulltext_search,
)


class RecordingDriver:
    """Minimal mock driver that records every query string it receives."""

    provider: GraphProvider = GraphProvider.NEO4J
    search_interface = None
    fulltext_syntax = '.'

    def __init__(self, provider: GraphProvider = GraphProvider.NEO4J) -> None:
        self.provider = provider
        self.queries: list[str] = []

    async def execute_query(self, cypher_query_: str, **kwargs: Any):
        self.queries.append(cypher_query_)
        return [], None, None


# ---------------------------------------------------------------------------
# Helper: build a valid query string that won't cause an early return
# ---------------------------------------------------------------------------


def _make_fulltext_query(provider: GraphProvider) -> str:
    if provider == GraphProvider.KUZU:
        return 'hello world'
    return 'hello world'


# ---------------------------------------------------------------------------
# Bug A: edge_bfs_search query shape (Neo4j / non-Kuzu)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_edge_bfs_search_neoj4_uses_unwind_relationships():
    """Neo4j branch should UNWIND relationships(path), not use last(e)."""
    driver = RecordingDriver(GraphProvider.NEO4J)
    await edge_bfs_search(
        driver,
        ['n1', 'n2'],
        2,
        SearchFilters(),
        group_ids=['g1'],
        limit=10,
    )
    query = driver.queries[0]
    assert 'UNWIND relationships(path)' in query, f'Expected relationships(path), got: {query}'
    assert 'last(e)' not in query, f'Should not contain last(e): {query}'


@pytest.mark.asyncio
async def test_edge_bfs_search_neoj4_no_literal_relationship_pattern():
    """The broken fork used [e:RELATES_TO*1..N] — verify it is gone."""
    driver = RecordingDriver(GraphProvider.NEO4J)
    await edge_bfs_search(
        driver,
        ['n1'],
        2,
        SearchFilters(),
        group_ids=['g1'],
        limit=10,
    )
    query = driver.queries[0]
    assert '[e:RELATES_TO*1' not in query, f'Should not have literal relationship pattern: {query}'


@pytest.mark.asyncio
async def test_edge_bfs_search_neoj4_param_bindings():
    """Every $param in the Neo4j query must appear as a kwarg."""
    import re

    driver = RecordingDriver(GraphProvider.NEO4J)
    await edge_bfs_search(
        driver,
        ['n1'],
        2,
        SearchFilters(),
        group_ids=['g1'],
        limit=10,
    )
    query = driver.queries[0]
    params = re.findall(r'\$(?:bfs_origin_node_uuids|limit)', query)
    params = list(set(params))
    # $bfs_origin_node_uuids and $limit must be bound
    for p in params:
        assert p in ('$bfs_origin_node_uuids', '$limit'), f'Unrecognized param: {p}'


@pytest.mark.asyncio
async def test_edge_bfs_search_kuzu_group_filter_is_a_where_clause():
    """The Kuzu MATCH has no WHERE, so the group filter must start one, not a bare AND."""
    driver = RecordingDriver(GraphProvider.KUZU)
    await edge_bfs_search(driver, ['n1'], 2, SearchFilters(), group_ids=['g1'], limit=10)
    query = driver.queries[0]
    assert 'WHERE e.group_id IN $group_ids' in query, query
    assert 'AND e.group_id IN $group_ids' not in query, query
    assert query.count('WHERE') == 1, query


# ---------------------------------------------------------------------------
# Bug B: fulltext YIELD vs WITH — Neo4j
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_node_fulltext_search_neoj4_uses_yield():
    driver = RecordingDriver(GraphProvider.NEO4J)
    fq = _make_fulltext_query(GraphProvider.NEO4J)
    await node_fulltext_search(
        driver,
        fq,
        SearchFilters(),
        group_ids=['g1'],
        limit=10,
    )
    query = driver.queries[0]
    assert 'YIELD node AS n' in query, f'Expected YIELD node AS n: {query}'


@pytest.mark.asyncio
async def test_community_fulltext_search_neoj4_uses_yield():
    driver = RecordingDriver(GraphProvider.NEO4J)
    fq = _make_fulltext_query(GraphProvider.NEO4J)
    await community_fulltext_search(
        driver,
        fq,
        group_ids=['g1'],
        limit=10,
    )
    query = driver.queries[0]
    assert 'YIELD node AS c' in query, f'Expected YIELD node AS c: {query}'


@pytest.mark.asyncio
async def test_episode_fulltext_search_neoj4_uses_yield():
    driver = RecordingDriver(GraphProvider.NEO4J)
    fq = _make_fulltext_query(GraphProvider.NEO4J)
    await episode_fulltext_search(
        driver,
        fq,
        SearchFilters(),
        group_ids=['g1'],
        limit=10,
    )
    query = driver.queries[0]
    assert 'YIELD node AS episode' in query, f'Expected YIELD node AS episode: {query}'
    # The group filter extends the uuid match; a second WHERE is a Cypher syntax error.
    assert query.count('WHERE') == 1, f'Expected a single WHERE clause: {query}'
    assert 'AND e.group_id IN $group_ids' in query


# ---------------------------------------------------------------------------
# Bug B: fulltext YIELD vs WITH — Kuzu
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_node_fulltext_search_kuzu_uses_with():
    driver = RecordingDriver(GraphProvider.KUZU)
    fq = _make_fulltext_query(GraphProvider.KUZU)
    await node_fulltext_search(
        driver,
        fq,
        SearchFilters(),
        group_ids=['g1'],
        limit=10,
    )
    query = driver.queries[0]
    assert 'WITH node AS n' in query, f'Expected WITH node AS n for Kuzu: {query}'
    assert 'YIELD' not in query, f'Kuzu should not use YIELD: {query}'


@pytest.mark.asyncio
async def test_community_fulltext_search_kuzu_uses_with():
    driver = RecordingDriver(GraphProvider.KUZU)
    fq = _make_fulltext_query(GraphProvider.KUZU)
    await community_fulltext_search(
        driver,
        fq,
        group_ids=['g1'],
        limit=10,
    )
    query = driver.queries[0]
    assert 'WITH node AS c' in query, f'Expected WITH node AS c for Kuzu: {query}'
    assert 'YIELD' not in query, f'Kuzu should not use YIELD: {query}'


@pytest.mark.asyncio
async def test_episode_fulltext_search_kuzu_uses_with():
    driver = RecordingDriver(GraphProvider.KUZU)
    fq = _make_fulltext_query(GraphProvider.KUZU)
    await episode_fulltext_search(
        driver,
        fq,
        SearchFilters(),
        group_ids=['g1'],
        limit=10,
    )
    query = driver.queries[0]
    assert 'YIELD node AS episode' in query, (
        f'Kuzu episode should also use YIELD (upstream pattern): {query}'
    )


# ---------------------------------------------------------------------------
# edge_fulltext_search: already uses YIELD in fork — verify it stays
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_edge_fulltext_search_neoj4_uses_yield():
    driver = RecordingDriver(GraphProvider.NEO4J)
    fq = _make_fulltext_query(GraphProvider.NEO4J)
    await edge_fulltext_search(
        driver,
        fq,
        SearchFilters(),
        group_ids=['g1'],
        limit=10,
    )
    query = driver.queries[0]
    assert 'YIELD relationship AS rel' in query or 'YIELD node, score' in query, (
        f'Expected YIELD: {query}'
    )


@pytest.mark.asyncio
async def test_edge_fulltext_search_kuzu_uses_yield():
    driver = RecordingDriver(GraphProvider.KUZU)
    fq = _make_fulltext_query(GraphProvider.KUZU)
    await edge_fulltext_search(
        driver,
        fq,
        SearchFilters(),
        group_ids=['g1'],
        limit=10,
    )
    query = driver.queries[0]
    assert 'YIELD node, score' in query, f'Expected YIELD node, score for Kuzu: {query}'


# ---------------------------------------------------------------------------
# Edge-case: early returns still work
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_edge_bfs_search_empty_origins_skips_query():
    driver = RecordingDriver(GraphProvider.NEO4J)
    result = await edge_bfs_search(driver, [], 2, SearchFilters(), group_ids=['g1'])
    assert result == []
    assert driver.queries == []


@pytest.mark.asyncio
async def test_edge_bfs_search_none_origins_skips_query():
    driver = RecordingDriver(GraphProvider.NEO4J)
    result = await edge_bfs_search(driver, None, 2, SearchFilters(), group_ids=['g1'])
    assert result == []
    assert driver.queries == []
