"""Unit tests verifying Cypher attribute-return patterns across providers.

Ensures that generated Cypher queries use the correct attributes syntax for each
provider:
- Kuzu:   ``e.attributes AS attributes`` / ``n.attributes AS attributes`` (column)
- Neo4j:  ``properties(e) AS attributes`` / ``properties(n) AS attributes``
- FalkorDB: ``properties(e) AS attributes`` / ``properties(n) AS attributes``

Regression test for the ``'NoneType' object has no attribute 'pop'`` ingestion
breakage introduced by commit a16c94c7.
"""

from unittest.mock import AsyncMock, Mock

import pytest

from graphiti_core.driver.driver import GraphProvider
from graphiti_core.search.search_filters import SearchFilters
from graphiti_core.search.search_utils import (
    edge_bfs_search,
    edge_fulltext_search,
    edge_similarity_search,
    node_bfs_search,
    node_fulltext_search,
    node_similarity_search,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_driver(provider: GraphProvider) -> Mock:
    """Build a minimal mock driver with the given provider."""
    driver = Mock()
    driver.provider = provider
    driver.search_interface = None
    driver.execute_query = AsyncMock(return_value=([], {}, {}))
    return driver


def _get_query(call_args) -> str:
    """Extract the Cypher query string from execute_query call args."""
    # First positional argument is the built query string.
    return call_args[0][0]


# ---------------------------------------------------------------------------
# edge_fulltext_search — fulltext.py
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'provider, expected_attr, negation',
    [
        (GraphProvider.KUZU, 'e.attributes AS attributes', None),
        (GraphProvider.NEO4J, 'properties(e) AS attributes', 'e.attributes AS attributes'),
        (GraphProvider.FALKORDB, 'properties(e) AS attributes', 'e.attributes AS attributes'),
    ],
)
async def test_edge_fulltext_search_attribute_syntax(provider, expected_attr, negation):
    driver = _make_driver(provider)
    await edge_fulltext_search(driver, 'test query', SearchFilters())
    call_args = driver.execute_query.call_args
    assert call_args is not None, 'execute_query was not called'
    query = _get_query(call_args)
    assert expected_attr in query, f'Expected {expected_attr!r} in query, got:\n{query}'
    if negation:
        assert negation not in query


# ---------------------------------------------------------------------------
# node_fulltext_search — fulltext.py
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'provider, expected_attr, negation',
    [
        (GraphProvider.KUZU, 'n.attributes AS attributes', None),
        (GraphProvider.NEO4J, 'properties(n) AS attributes', 'n.attributes AS attributes'),
        (GraphProvider.FALKORDB, 'properties(n) AS attributes', 'n.attributes AS attributes'),
    ],
)
async def test_node_fulltext_search_attribute_syntax(provider, expected_attr, negation):
    driver = _make_driver(provider)
    await node_fulltext_search(driver, 'test query', SearchFilters())
    call_args = driver.execute_query.call_args
    assert call_args is not None, 'execute_query was not called'
    query = _get_query(call_args)
    assert expected_attr in query, f'Expected {expected_attr!r} in query, got:\n{query}'
    if negation:
        assert negation not in query


# ---------------------------------------------------------------------------
# edge_similarity_search — similarity.py (already correct, smoke test)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'provider, expected_attr, negation',
    [
        (GraphProvider.KUZU, 'e.attributes AS attributes', None),
        (GraphProvider.NEO4J, 'properties(e) AS attributes', 'e.attributes AS attributes'),
        (GraphProvider.FALKORDB, 'properties(e) AS attributes', 'e.attributes AS attributes'),
    ],
)
async def test_edge_similarity_search_attribute_syntax(provider, expected_attr, negation):
    driver = _make_driver(provider)
    await edge_similarity_search(driver, [0.1] * 4, None, None, SearchFilters())
    call_args = driver.execute_query.call_args
    assert call_args is not None, 'execute_query was not called'
    query = _get_query(call_args)
    assert expected_attr in query, f'Expected {expected_attr!r} in query, got:\n{query}'
    if negation:
        assert negation not in query


# ---------------------------------------------------------------------------
# node_similarity_search — similarity.py (already correct, smoke test)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'provider, expected_attr, negation',
    [
        (GraphProvider.KUZU, 'n.attributes AS attributes', None),
        (GraphProvider.NEO4J, 'properties(n) AS attributes', 'n.attributes AS attributes'),
        (GraphProvider.FALKORDB, 'properties(n) AS attributes', 'n.attributes AS attributes'),
    ],
)
async def test_node_similarity_search_attribute_syntax(provider, expected_attr, negation):
    driver = _make_driver(provider)
    await node_similarity_search(driver, [0.1] * 4, SearchFilters())
    call_args = driver.execute_query.call_args
    assert call_args is not None, 'execute_query was not called'
    query = _get_query(call_args)
    assert expected_attr in query, f'Expected {expected_attr!r} in query, got:\n{query}'
    if negation:
        assert negation not in query


# ---------------------------------------------------------------------------
# edge_bfs_search — traversal.py (already correct, smoke test)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'provider, expected_attr, negation',
    [
        (GraphProvider.KUZU, 'e.attributes AS attributes', None),
        (GraphProvider.NEO4J, 'properties(e) AS attributes', 'e.attributes AS attributes'),
        (GraphProvider.FALKORDB, 'properties(e) AS attributes', 'e.attributes AS attributes'),
    ],
)
async def test_edge_bfs_search_attribute_syntax(provider, expected_attr, negation):
    driver = _make_driver(provider)
    await edge_bfs_search(driver, ['node1'], 2, SearchFilters())
    call_args = driver.execute_query.call_args
    assert call_args is not None, 'execute_query was not called'
    query = _get_query(call_args)
    assert expected_attr in query, f'Expected {expected_attr!r} in query, got:\n{query}'
    if negation:
        assert negation not in query


# ---------------------------------------------------------------------------
# node_bfs_search — traversal.py (already correct, smoke test)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'provider, expected_attr, negation',
    [
        (GraphProvider.KUZU, 'm.attributes AS attributes', None),
        (GraphProvider.NEO4J, 'properties(m) AS attributes', 'm.attributes AS attributes'),
        (GraphProvider.FALKORDB, 'properties(m) AS attributes', 'm.attributes AS attributes'),
    ],
)
async def test_node_bfs_search_attribute_syntax(provider, expected_attr, negation):
    driver = _make_driver(provider)
    await node_bfs_search(driver, ['node1'], SearchFilters(), 2)
    call_args = driver.execute_query.call_args
    assert call_args is not None, 'execute_query was not called'
    query = _get_query(call_args)
    assert expected_attr in query, f'Expected {expected_attr!r} in query, got:\n{query}'
    if negation:
        assert negation not in query
