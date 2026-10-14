"""Guard and delegation tests for the fork's split ``search_utils`` package (no live database).

The fork moved upstream's ``search_utils.py`` into ``search_utils/``; these pin the
upstream behavior the split must keep:

- BFS with no origin nodes returns ``[]`` without querying the database.
- ``episode_fulltext_search`` forwards the search filter to ``search_interface``
  in the interface's positional order.
"""

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from graphiti_core.driver.driver import GraphProvider
from graphiti_core.search.search_filters import SearchFilters
from graphiti_core.search.search_utils import (
    edge_bfs_search,
    episode_fulltext_search,
    node_bfs_search,
)


class RecordingDriver:
    provider = GraphProvider.NEO4J
    search_interface = None
    fulltext_syntax = ''

    def __init__(self):
        self.queries: list[str] = []

    async def execute_query(self, cypher_query_: str, **kwargs: Any):
        self.queries.append(cypher_query_)
        return [], None, None


@pytest.mark.asyncio
@pytest.mark.parametrize('origins', [None, []])
async def test_edge_bfs_without_origins_skips_query(origins):
    driver = RecordingDriver()
    assert await edge_bfs_search(driver, origins, 2, SearchFilters()) == []
    assert driver.queries == []


@pytest.mark.asyncio
@pytest.mark.parametrize('origins, depth', [(None, 2), ([], 2), (['n1'], 0)])
async def test_node_bfs_without_origins_or_depth_skips_query(origins, depth):
    driver = RecordingDriver()
    assert await node_bfs_search(driver, origins, SearchFilters(), depth) == []
    assert driver.queries == []


@pytest.mark.asyncio
async def test_episode_fulltext_search_forwards_filter_to_search_interface():
    search_filter = SearchFilters()
    driver = MagicMock()
    driver.search_interface.episode_fulltext_search = AsyncMock(return_value=[])

    await episode_fulltext_search(driver, 'query', search_filter, ['g1'], 7)

    driver.search_interface.episode_fulltext_search.assert_awaited_once_with(
        driver, 'query', search_filter, ['g1'], 7
    )
