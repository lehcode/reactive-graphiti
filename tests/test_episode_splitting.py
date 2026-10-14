"""Tests for splitting oversized episodes into a saga in ``Graphiti.add_episode``."""

import inspect
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from graphiti_core.edges import EntityEdge, EpisodicEdge
from graphiti_core.graphiti import AddEpisodeResults, Graphiti
from graphiti_core.nodes import EntityNode, EpisodeType, EpisodicNode, SagaNode
from graphiti_core.utils import episode_splitting

NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)
# Default ratio 0.4 -> threshold and chunk size of 400 tokens (~1600 chars).
BUDGET = 1000
LONG_TEXT = ' '.join(f'Sentence number {i} mentions entity E{i}.' for i in range(400))
SHORT_TEXT = 'Alice met Bob.'
INTERNAL_SIGNATURE = inspect.signature(Graphiti._add_episode_internal)


def _episode(name: str) -> EpisodicNode:
    return EpisodicNode(
        name=name,
        group_id='g',
        source=EpisodeType.text,
        source_description='test',
        content='',
        valid_at=NOW,
    )


def _make_graphiti(budget=BUDGET, results=None):
    """Graphiti without a driver; the single-episode path is replaced by a recorder.

    Returns the instance and the list of normalized ``_add_episode_internal`` calls.
    """
    graphiti = Graphiti.__new__(Graphiti)
    graphiti.llm_client = MagicMock()
    graphiti.llm_client.input_token_budget.return_value = budget
    calls: list[dict] = []

    async def record(*args, **kwargs):
        bound = INTERNAL_SIGNATURE.bind(graphiti, *args, **kwargs).arguments
        call = {k: v for k, v in bound.items() if k != 'self'}
        calls.append(call)
        if results is not None:
            return results[len(calls) - 1]
        return AddEpisodeResults(
            episode=_episode(call['name']),
            episodic_edges=[],
            nodes=[],
            edges=[],
            communities=[],
            community_edges=[],
        )

    graphiti._add_episode_internal = AsyncMock(side_effect=record)
    return graphiti, calls


async def _add(graphiti, **overrides):
    kwargs = {
        'name': 'doc',
        'episode_body': LONG_TEXT,
        'source_description': 'test',
        'reference_time': NOW,
        'source': EpisodeType.text,
    }
    kwargs.update(overrides)
    return await graphiti.add_episode(**kwargs)


@pytest.mark.asyncio
async def test_oversized_episode_is_split_into_a_new_saga_named_after_it():
    graphiti, calls = _make_graphiti()

    result = await _add(graphiti)

    n = len(calls)
    assert n >= 2
    assert [c['name'] for c in calls] == [f'doc (part {i}/{n})' for i in range(1, n + 1)]
    assert all(c['saga'] == 'doc' for c in calls)
    assert all(c['uuid'] is None for c in calls)
    # group_id=None must reach the single-episode path unchanged (provider default group).
    assert all(c['group_id'] is None for c in calls)
    assert all(c['previous_episode_uuids'] is None for c in calls)
    assert all(c['episode_body'] in LONG_TEXT for c in calls)
    assert calls[0]['episode_body'].startswith('Sentence number 0 ')
    assert 'entity E399.' in calls[-1]['episode_body']
    assert result.episode.name == f'doc (part {n}/{n})'


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'saga',
    ['caller-saga', SagaNode(name='caller-saga', group_id='g', created_at=NOW)],
    ids=['saga-name', 'saga-node'],
)
async def test_parts_join_the_callers_saga_and_chain_from_its_previous_episode(saga):
    graphiti, calls = _make_graphiti()

    await _add(graphiti, saga=saga, saga_previous_episode_uuid='prev-uuid', group_id='grp')

    assert len(calls) >= 2
    assert all(c['saga'] is saga for c in calls)
    assert all(c['group_id'] == 'grp' for c in calls)
    assert calls[0]['saga_previous_episode_uuid'] == 'prev-uuid'


@pytest.mark.asyncio
async def test_each_part_chains_to_the_previous_parts_episode():
    episodes = [_episode(f'p{i}') for i in range(20)]
    results = [
        AddEpisodeResults(
            episode=e, episodic_edges=[], nodes=[], edges=[], communities=[], community_edges=[]
        )
        for e in episodes
    ]
    graphiti, calls = _make_graphiti(results=results)

    await _add(graphiti)

    assert calls[0]['saga_previous_episode_uuid'] is None
    for i in range(1, len(calls)):
        assert calls[i]['saga_previous_episode_uuid'] == episodes[i - 1].uuid


@pytest.mark.asyncio
async def test_explicit_previous_episode_uuids_accumulate_earlier_parts():
    episodes = [_episode(f'p{i}') for i in range(20)]
    results = [
        AddEpisodeResults(
            episode=e, episodic_edges=[], nodes=[], edges=[], communities=[], community_edges=[]
        )
        for e in episodes
    ]
    graphiti, calls = _make_graphiti(results=results)

    # An explicit empty list means "no previous-episode context" and must not become None.
    await _add(graphiti, previous_episode_uuids=[])

    assert len(calls) >= 3
    for i, call in enumerate(calls):
        assert call['previous_episode_uuids'] == [e.uuid for e in episodes[:i]]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'budget, body, uuid',
    [
        (BUDGET, SHORT_TEXT, None),
        (None, LONG_TEXT, None),
        (MagicMock(), LONG_TEXT, None),
        (BUDGET, LONG_TEXT, 'existing-uuid'),
    ],
    ids=['under-threshold', 'no-budget', 'non-int-budget', 'existing-episode'],
)
async def test_episode_is_processed_once_unchanged_when_not_split(budget, body, uuid):
    graphiti, calls = _make_graphiti(budget=budget)

    await _add(graphiti, episode_body=body, uuid=uuid)

    assert len(calls) == 1
    call = calls[0]
    assert call['name'] == 'doc'
    assert call['episode_body'] == body
    assert call['uuid'] == uuid
    assert call['saga'] is None
    assert call['group_id'] is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'source, chunker_name',
    [
        (EpisodeType.text, 'chunk_text_content'),
        (EpisodeType.message, 'chunk_message_content'),
        (EpisodeType.json, 'chunk_json_content'),
    ],
)
async def test_chunker_matches_source_with_budget_sized_chunks_and_positive_overlap(
    source, chunker_name
):
    graphiti, calls = _make_graphiti()
    chunker = MagicMock(return_value=['part one', 'part two'])

    with patch.object(episode_splitting, chunker_name, chunker):
        await _add(graphiti, source=source)

    chunker.assert_called_once()
    content, chunk_size, overlap = chunker.call_args.args
    assert content == LONG_TEXT
    assert chunk_size == 400
    assert 0 < overlap < chunk_size
    assert [c['episode_body'] for c in calls] == ['part one', 'part two']
    assert all(c['source'] == source for c in calls)


@pytest.mark.asyncio
async def test_json_parts_are_valid_json():
    body = json.dumps([{'id': i, 'name': f'Entity {i}', 'note': 'x' * 40} for i in range(200)])
    graphiti, calls = _make_graphiti()

    await _add(graphiti, episode_body=body, source=EpisodeType.json)

    assert len(calls) >= 2
    for call in calls:
        json.loads(call['episode_body'])


@pytest.mark.asyncio
async def test_unsplittable_oversized_body_is_processed_once_as_is():
    graphiti, calls = _make_graphiti()

    with patch.object(episode_splitting, 'chunk_text_content', return_value=[LONG_TEXT]):
        await _add(graphiti, saga_previous_episode_uuid='prev-uuid')

    assert graphiti._add_episode_internal.await_count == 1
    assert calls[0]['name'] == 'doc'
    assert calls[0]['episode_body'] == LONG_TEXT
    assert calls[0]['saga'] is None
    assert calls[0]['saga_previous_episode_uuid'] == 'prev-uuid'


@pytest.mark.asyncio
async def test_results_are_aggregated_across_parts():
    def node(uuid, name):
        return EntityNode(uuid=uuid, name=name, group_id='g')

    def edge(uuid, fact):
        return EntityEdge(
            uuid=uuid,
            group_id='g',
            source_node_uuid='a',
            target_node_uuid='b',
            created_at=NOW,
            name='RELATES_TO',
            fact=fact,
        )

    def episodic_edge(uuid):
        return EpisodicEdge(
            uuid=uuid, group_id='g', source_node_uuid='e', target_node_uuid='n', created_at=NOW
        )

    first, last = _episode('p1'), _episode('p2')
    results = [
        AddEpisodeResults(
            episode=first,
            episodic_edges=[episodic_edge('ee1')],
            nodes=[node('n1', 'old'), node('n2', 'two')],
            edges=[edge('e1', 'old fact')],
            communities=[],
            community_edges=[],
        ),
        AddEpisodeResults(
            episode=last,
            episodic_edges=[episodic_edge('ee2')],
            nodes=[node('n3', 'three'), node('n1', 'new')],
            edges=[edge('e2', 'other'), edge('e1', 'new fact')],
            communities=[],
            community_edges=[],
        ),
    ]
    graphiti, _ = _make_graphiti(results=results)

    with patch.object(episode_splitting, 'chunk_text_content', return_value=['a', 'b']):
        result = await _add(graphiti)

    assert result.episode is last
    assert [e.uuid for e in result.episodic_edges] == ['ee1', 'ee2']
    assert [(n.uuid, n.name) for n in result.nodes] == [
        ('n1', 'new'),
        ('n2', 'two'),
        ('n3', 'three'),
    ]
    assert [(e.uuid, e.fact) for e in result.edges] == [('e1', 'new fact'), ('e2', 'other')]


@pytest.mark.parametrize(
    'value, expected',
    [('0.5', 0.5), ('1', 1.0), (None, 0.4), ('abc', 0.4), ('0', 0.4), ('1.5', 0.4)],
)
def test_episode_content_budget_ratio_env_parsing(value, expected):
    # A subprocess keeps the module-level parse isolated from this interpreter's imports.
    env = {k: v for k, v in os.environ.items() if k != 'EPISODE_CONTENT_BUDGET_RATIO'}
    if value is not None:
        env['EPISODE_CONTENT_BUDGET_RATIO'] = value
    out = subprocess.run(
        [
            sys.executable,
            '-c',
            'from graphiti_core.helpers import EPISODE_CONTENT_BUDGET_RATIO as r; print(r)',
        ],
        env=env,
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    )
    assert float(out.stdout.strip()) == expected
