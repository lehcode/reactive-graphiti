"""add_memory appends an advisory size warning without changing queueing."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import graphiti_mcp_server as server

PLAIN = "Episode 'n' queued for processing in group 'g'"


class FakeQueue:
    def __init__(self):
        self.calls = []

    async def add_episode(self, **kwargs):
        self.calls.append(kwargs)


@pytest.fixture
def queue(monkeypatch):
    q = FakeQueue()
    llm = object()
    monkeypatch.setattr(server, 'queue_service', q)
    monkeypatch.setattr(
        server,
        'graphiti_service',
        SimpleNamespace(
            client=SimpleNamespace(llm_client=llm),
            entity_types=None,
            edge_types=None,
            edge_type_map=None,
        ),
    )
    q.llm = llm
    return q


async def _add():
    return await server.add_memory(name='n', episode_body='body', group_id='g')


async def test_warning_is_appended_and_episode_still_queued(queue, monkeypatch):
    seen = []

    def fake(llm, body, source, uuid):
        seen.append((llm, body, uuid))
        return 'too big'

    monkeypatch.setattr(server, 'episode_size_warning', fake)
    resp = await _add()
    assert resp['message'] == f'{PLAIN}. WARNING: too big'
    assert len(queue.calls) == 1
    assert seen == [(queue.llm, 'body', None)]


@pytest.mark.parametrize('helper', [lambda *a: None, None, 'raises'])
async def test_plain_message_otherwise(queue, monkeypatch, helper):
    if helper == 'raises':

        def helper(*a):
            raise RuntimeError('boom')

    monkeypatch.setattr(server, 'episode_size_warning', helper)
    resp = await _add()
    assert resp['message'] == PLAIN
    assert len(queue.calls) == 1
