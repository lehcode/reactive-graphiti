"""Recipe selection in the MCP search tools when the bge cross-encoder is active.

These call the real search_nodes / search_memory_facts tools with a stub client
and assert which core search path and recipe each one uses.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from graphiti_core.search.search_config_recipes import (
    EDGE_HYBRID_SEARCH_CROSS_ENCODER,
    NODE_HYBRID_SEARCH_CROSS_ENCODER,
    NODE_HYBRID_SEARCH_NODE_DISTANCE,
    NODE_HYBRID_SEARCH_RRF,
)

import graphiti_mcp_server as server


async def _async_return(value):
    return value


def install_service(monkeypatch, bge_active: bool):
    client = SimpleNamespace(
        search_=AsyncMock(return_value=SimpleNamespace(nodes=[], edges=[])),
        search=AsyncMock(return_value=[]),
    )
    service = SimpleNamespace(
        get_client=lambda: _async_return(client),
        _bge_reranker_active=bge_active,
    )
    monkeypatch.setattr(server, 'graphiti_service', service)
    monkeypatch.setattr(
        server,
        'config',
        SimpleNamespace(graphiti=SimpleNamespace(group_id='main')),
        raising=False,
    )
    return client


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('bge_active', 'center_node_uuid', 'expected'),
    [
        (True, None, NODE_HYBRID_SEARCH_CROSS_ENCODER),
        (True, 'center', NODE_HYBRID_SEARCH_NODE_DISTANCE),
        (False, None, NODE_HYBRID_SEARCH_RRF),
        (False, 'center', NODE_HYBRID_SEARCH_NODE_DISTANCE),
    ],
)
async def test_search_nodes_recipe(monkeypatch, bge_active, center_node_uuid, expected):
    client = install_service(monkeypatch, bge_active)

    await server.search_nodes(query='q', center_node_uuid=center_node_uuid)

    client.search_.assert_awaited_once()
    assert client.search_.await_args.kwargs['config'] is expected


@pytest.mark.asyncio
async def test_search_facts_uses_cross_encoder_recipe_when_bge_active(monkeypatch):
    client = install_service(monkeypatch, bge_active=True)

    await server.search_memory_facts(query='q', max_facts=7)

    client.search.assert_not_awaited()
    client.search_.assert_awaited_once()
    kwargs = client.search_.await_args.kwargs
    assert kwargs['group_ids'] == ['main']
    used = kwargs['config']
    assert used is not EDGE_HYBRID_SEARCH_CROSS_ENCODER
    assert used.limit == 7
    assert used.edge_config == EDGE_HYBRID_SEARCH_CROSS_ENCODER.edge_config
    # The shared module-level recipe must not be mutated.
    assert EDGE_HYBRID_SEARCH_CROSS_ENCODER.limit != 7


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ('bge_active', 'center_node_uuid'),
    [(True, 'center'), (False, None)],
)
async def test_search_facts_uses_legacy_search(monkeypatch, bge_active, center_node_uuid):
    client = install_service(monkeypatch, bge_active)

    await server.search_memory_facts(query='q', max_facts=7, center_node_uuid=center_node_uuid)

    client.search_.assert_not_awaited()
    client.search.assert_awaited_once()
    kwargs = client.search.await_args.kwargs
    assert kwargs['num_results'] == 7
    assert kwargs['center_node_uuid'] == center_node_uuid
