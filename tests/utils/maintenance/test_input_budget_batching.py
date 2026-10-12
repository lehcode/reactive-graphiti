"""Input-token budget splitting of dedupe, edge extraction and edge resolution prompts.

The module-level ``estimate_tokens`` is patched to a flat 10 tokens per string, so a prompt
rendered as one message costs 10, the response schema 10 and every node/candidate entry 10.
"""

import logging
from datetime import datetime, timezone
from itertools import combinations
from unittest.mock import AsyncMock, MagicMock

import pytest

from graphiti_core.edges import EntityEdge
from graphiti_core.nodes import EntityNode, EpisodeType, EpisodicNode
from graphiti_core.utils.maintenance import edge_operations, node_operations
from graphiti_core.utils.maintenance.dedup_helpers import DedupResolutionState
from graphiti_core.utils.maintenance.edge_operations import (
    extract_edges,
    resolve_extracted_edge,
)
from graphiti_core.utils.maintenance.node_operations import (
    _build_candidate_indexes,
    _resolve_with_llm,
)

TOKENS_PER_STRING = 10
# One rendered message + the response schema
FIXED_COST = 2 * TOKENS_PER_STRING


def _episode() -> EpisodicNode:
    return EpisodicNode(
        name='episode',
        group_id='group',
        source=EpisodeType.text,
        source_description='test',
        content='content',
        valid_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )


def _llm_client(budget: int | None) -> MagicMock:
    client = MagicMock()
    client.input_token_budget.return_value = budget
    client.generate_response = AsyncMock()
    return client


def _capture_prompt(monkeypatch, target: str) -> list[dict]:
    contexts: list[dict] = []

    def fake_prompt(context):
        contexts.append(context)
        return ['prompt']

    monkeypatch.setattr(target, fake_prompt)
    return contexts


@pytest.fixture
def flat_estimate(monkeypatch):
    for module in (node_operations, edge_operations):
        monkeypatch.setattr(module, 'estimate_tokens', lambda _text: TOKENS_PER_STRING)


# --- dedupe sub-batches (_resolve_with_llm) -------------------------------------------------


async def _run_dedupe(monkeypatch, budget, responses, extracted_count=4):
    extracted = [
        EntityNode(name=f'Entity {i}', group_id='group', labels=['Entity'])
        for i in range(extracted_count)
    ]
    candidate = EntityNode(name='Existing', group_id='group', labels=['Entity'])
    state = DedupResolutionState(
        resolved_nodes=[None] * extracted_count,
        uuid_map={},
        unresolved_indices=list(range(extracted_count)),
    )
    contexts = _capture_prompt(
        monkeypatch,
        'graphiti_core.utils.maintenance.node_operations.prompt_library.dedupe_nodes.nodes',
    )
    llm_client = _llm_client(budget)
    llm_client.generate_response.side_effect = responses

    await _resolve_with_llm(
        llm_client,
        extracted,
        _build_candidate_indexes([candidate]),
        state,
        episode=_episode(),
        previous_episodes=None,
        entity_types=None,
    )
    return extracted, candidate, state, contexts, llm_client


def _resolution(local_id: int, duplicate: int = -1) -> dict:
    return {'id': local_id, 'name': f'n{local_id}', 'duplicate_candidate_id': duplicate}


@pytest.mark.asyncio
async def test_dedupe_batches_are_renumbered_and_mapped_back(monkeypatch, flat_estimate):
    # Room for 2 entities per call -> 2 calls; the model answers with per-call ids 0..1
    responses = [
        {'entity_resolutions': [_resolution(0), _resolution(1)]},
        {'entity_resolutions': [_resolution(0), _resolution(1, duplicate=0)]},
    ]
    extracted, candidate, state, contexts, _ = await _run_dedupe(
        monkeypatch, FIXED_COST + 2 * TOKENS_PER_STRING, responses
    )

    sent = [ctx for ctx in contexts if ctx['extracted_nodes']]
    assert [[e['id'] for e in ctx['extracted_nodes']] for ctx in sent] == [[0, 1], [0, 1]]
    assert [e['name'] for e in sent[1]['extracted_nodes']] == ['Entity 2', 'Entity 3']
    # Batch 2's local id 1 is global entity 3
    assert state.resolved_nodes[3].uuid == candidate.uuid
    assert [n.uuid for n in state.resolved_nodes[:3]] == [n.uuid for n in extracted[:3]]
    assert state.duplicate_pairs == [(extracted[3], candidate)]


@pytest.mark.asyncio
async def test_dedupe_out_of_range_batch_id_does_not_alias(monkeypatch, flat_estimate, caplog):
    # Batch 1 answers id 2 (outside its 0..1); it must not resolve batch 2's first entity
    responses = [
        {
            'entity_resolutions': [
                _resolution(0),
                _resolution(1),
                _resolution(2, duplicate=0),
            ]
        },
        {'entity_resolutions': [_resolution(0), _resolution(1)]},
    ]
    with caplog.at_level(logging.WARNING):
        extracted, _candidate, state, _, _ = await _run_dedupe(
            monkeypatch, FIXED_COST + 2 * TOKENS_PER_STRING, responses
        )

    assert [n.uuid for n in state.resolved_nodes] == [n.uuid for n in extracted]
    assert state.duplicate_pairs == []
    assert 'invalid LLM dedupe id' in caplog.text


@pytest.mark.asyncio
async def test_dedupe_splits_without_previous_episodes(monkeypatch, flat_estimate):
    responses = [{'entity_resolutions': [_resolution(0)]} for _ in range(4)]
    *_, llm_client = await _run_dedupe(monkeypatch, FIXED_COST + TOKENS_PER_STRING, responses)

    assert llm_client.generate_response.await_count == 4


@pytest.mark.asyncio
async def test_dedupe_single_call_without_budget(monkeypatch, flat_estimate):
    responses = [{'entity_resolutions': [_resolution(i) for i in range(4)]}]
    _, _, state, contexts, llm_client = await _run_dedupe(monkeypatch, None, responses)

    assert llm_client.generate_response.await_count == 1
    assert [e['id'] for e in contexts[-1]['extracted_nodes']] == [0, 1, 2, 3]
    assert all(node is not None for node in state.resolved_nodes)


# --- edge extraction covering chunks (extract_edges) ----------------------------------------


def _nodes(count: int) -> list[EntityNode]:
    return [EntityNode(name=f'N{i}', group_id='group', labels=['Entity']) for i in range(count)]


async def _run_extract_edges(monkeypatch, budget, nodes, respond):
    contexts = _capture_prompt(
        monkeypatch,
        'graphiti_core.utils.maintenance.edge_operations.prompt_library.extract_edges.edge',
    )
    sent: list[dict] = []

    def call_llm(*_args, **_kwargs):
        # The last rendered prompt is the one being sent (earlier renders only measure)
        sent.append(contexts[-1])
        return respond(contexts[-1])

    llm_client = _llm_client(budget)
    llm_client.generate_response.side_effect = call_llm
    clients = MagicMock()
    clients.llm_client = llm_client

    edges = await extract_edges(
        clients,
        _episode(),
        nodes,
        previous_episodes=[],
        edge_type_map={('Entity', 'Entity'): ['KNOWS']},
    )
    return edges, sent, llm_client


def _no_edges(_context):
    return {'edges': []}


@pytest.mark.asyncio
async def test_extract_edges_single_call_when_nodes_fit(monkeypatch, flat_estimate):
    nodes = _nodes(40)
    _, contexts, llm_client = await _run_extract_edges(
        monkeypatch, FIXED_COST + 40 * TOKENS_PER_STRING, nodes, _no_edges
    )

    assert llm_client.generate_response.await_count == 1
    assert len(contexts[-1]['nodes']) == 40


@pytest.mark.asyncio
async def test_extract_edges_single_call_without_budget(monkeypatch, flat_estimate):
    _, contexts, llm_client = await _run_extract_edges(monkeypatch, None, _nodes(40), _no_edges)

    assert llm_client.generate_response.await_count == 1
    assert len(contexts[-1]['nodes']) == 40


@pytest.mark.asyncio
async def test_extract_edges_covering_chunks_when_over_budget(monkeypatch, flat_estimate):
    nodes = _nodes(5)

    def respond(context):
        names = {n['name'] for n in context['nodes']}
        edges = [
            {
                'source_entity_name': src,
                'target_entity_name': tgt,
                'relation_type': 'KNOWS',
                'fact': f'{src} knows {tgt}',
            }
            for src, tgt in combinations(sorted(names), 2)
        ]
        return {'edges': edges}

    # Room for 3 nodes per call
    edges, contexts, llm_client = await _run_extract_edges(
        monkeypatch, FIXED_COST + 3 * TOKENS_PER_STRING, nodes, respond
    )

    chunks = [{n['name'] for n in ctx['nodes']} for ctx in contexts if ctx['nodes']]
    assert llm_client.generate_response.await_count == len(chunks) > 1
    assert all(len(chunk) <= 3 for chunk in chunks)
    for a, b in combinations([n.name for n in nodes], 2):
        assert any({a, b} <= chunk for chunk in chunks), (a, b)
    # Pairs covered by several chunks come back once
    facts = [edge.fact for edge in edges]
    assert sorted(facts) == sorted(
        f'{a} knows {b}' for a, b in combinations(sorted(n.name for n in nodes), 2)
    )


@pytest.mark.asyncio
async def test_extract_edges_chunk_failure_propagates(monkeypatch, flat_estimate):
    def respond(_context):
        raise RuntimeError('llm down')

    with pytest.raises(RuntimeError, match='llm down'):
        await _run_extract_edges(
            monkeypatch, FIXED_COST + 3 * TOKENS_PER_STRING, _nodes(5), respond
        )


# --- edge resolution candidate trim (resolve_extracted_edge) --------------------------------


def _edge(fact: str) -> EntityEdge:
    return EntityEdge(
        source_node_uuid='source',
        target_node_uuid='target',
        name='KNOWS',
        group_id='group',
        fact=fact,
        episodes=[],
        created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )


@pytest.mark.asyncio
async def test_resolve_edge_trims_lowest_ranked_candidates(monkeypatch, flat_estimate):
    related = [_edge('related 0'), _edge('related 1')]
    existing = [_edge('existing 0'), _edge('existing 1'), _edge('existing 2')]
    contexts = _capture_prompt(
        monkeypatch,
        'graphiti_core.utils.maintenance.edge_operations.prompt_library.dedupe_edges.resolve_edge',
    )
    # Room for 3 candidates: both related edges and the top invalidation candidate
    llm_client = _llm_client(FIXED_COST + 3 * TOKENS_PER_STRING)
    llm_client.generate_response.return_value = {
        'duplicate_facts': [1],
        'contradicted_facts': [2],
    }

    resolved, _invalidated, duplicates = await resolve_extracted_edge(
        llm_client, _edge('new fact'), related, existing, _episode()
    )

    sent = contexts[-1]
    assert sent['existing_edges'] == [
        {'idx': 0, 'fact': 'related 0'},
        {'idx': 1, 'fact': 'related 1'},
    ]
    assert sent['edge_invalidation_candidates'] == [{'idx': 2, 'fact': 'existing 0'}]
    assert resolved.uuid == related[1].uuid
    assert [edge.uuid for edge in duplicates] == [related[1].uuid]


@pytest.mark.asyncio
async def test_resolve_edge_trims_related_edges_and_logs(monkeypatch, flat_estimate, caplog):
    related = [_edge('related 0'), _edge('related 1'), _edge('related 2')]
    existing = [_edge('existing 0'), _edge('existing 1')]
    contexts = _capture_prompt(
        monkeypatch,
        'graphiti_core.utils.maintenance.edge_operations.prompt_library.dedupe_edges.resolve_edge',
    )
    # Room for 2 candidates: the third related edge and all invalidation candidates drop
    llm_client = _llm_client(FIXED_COST + 2 * TOKENS_PER_STRING)
    llm_client.generate_response.return_value = {'duplicate_facts': [1], 'contradicted_facts': []}

    with caplog.at_level(logging.DEBUG, logger=edge_operations.logger.name):
        resolved, invalidated, duplicates = await resolve_extracted_edge(
            llm_client, _edge('new fact'), related, existing, _episode()
        )

    sent = contexts[-1]
    assert [c['idx'] for c in sent['existing_edges']] == [0, 1]
    assert sent['edge_invalidation_candidates'] == []
    assert resolved.uuid == related[1].uuid
    assert [edge.uuid for edge in duplicates] == [related[1].uuid]
    assert invalidated == []
    assert 'dropped 1 of 3 EXISTING FACTS and 2 of 2 INVALIDATION CANDIDATES' in caplog.text


@pytest.mark.asyncio
async def test_extract_edges_falls_back_to_one_call_above_chunk_limit(
    monkeypatch, flat_estimate, caplog
):
    monkeypatch.setattr(edge_operations, 'MAX_EDGE_EXTRACTION_CHUNKS', 2)
    nodes = _nodes(6)

    # Room for 2 nodes per call -> 15 pair chunks, above the limit of 2
    with caplog.at_level(logging.WARNING):
        _, sent, llm_client = await _run_extract_edges(
            monkeypatch, FIXED_COST + 2 * TOKENS_PER_STRING, nodes, _no_edges
        )

    assert llm_client.generate_response.await_count == 1
    assert len(sent[0]['nodes']) == 6
    assert 'covering chunks' in caplog.text


# --- summary flights (_build_budget_flights) ------------------------------------------------


def _summary_nodes(count: int, **attributes) -> list[EntityNode]:
    return [
        EntityNode(
            name=f'Node {i}',
            group_id='group',
            labels=['Entity'],
            summary='A short summary.',
            attributes=dict(attributes),
        )
        for i in range(count)
    ]


def _budget_episode(content: str) -> EpisodicNode:
    episode = _episode()
    episode.content = content
    return episode


def test_flights_count_episode_content_as_fixed_cost():
    nodes = _summary_nodes(12)
    llm_client = _llm_client(None)
    # Room for all 12 nodes next to a short episode; a ~5000-token episode alone exceeds it
    llm_client.input_token_budget.return_value = 4000
    short = node_operations._build_budget_flights(
        nodes, 30, llm_client, episode=_budget_episode('Short.')
    )
    long = node_operations._build_budget_flights(
        nodes, 30, llm_client, episode=_budget_episode('Long sentence here. ' * 1000)
    )

    assert len(short) == 1
    assert len(long) > 1
    assert [n.uuid for flight in long for n in flight] == [n.uuid for n in nodes]


def test_flights_measure_the_episode_prompt_when_it_is_sent(monkeypatch):
    rendered: list[str] = []

    def spy(name):
        def builder(_context):
            rendered.append(name)
            return ['prompt']

        return builder

    monkeypatch.setattr(
        node_operations.prompt_library.extract_nodes,
        'extract_summaries_batch',
        spy('batch'),
    )
    monkeypatch.setattr(
        node_operations.prompt_library.extract_nodes,
        'extract_entity_summaries_from_episodes',
        spy('episodes'),
    )
    llm_client = _llm_client(10_000)

    node_operations._build_budget_flights(
        _summary_nodes(2), 30, llm_client, use_episode_prompt=True
    )
    node_operations._build_budget_flights(_summary_nodes(2), 30, llm_client)

    assert rendered == ['episodes', 'batch']


def test_flights_tolerate_non_json_attributes():
    nodes = _summary_nodes(3, seen_at=datetime(2026, 1, 1, tzinfo=timezone.utc))

    flights = node_operations._build_budget_flights(nodes, 30, _llm_client(10_000))

    assert [n.uuid for flight in flights for n in flight] == [n.uuid for n in nodes]


@pytest.mark.asyncio
async def test_summary_batch_passes_flight_inputs(monkeypatch):
    captured: dict = {}

    def spy_flights(nodes, max_items, llm_client, **kwargs):
        captured.update(kwargs, max_items=max_items)
        return []

    monkeypatch.setattr(node_operations, '_build_budget_flights', spy_flights)
    episode = _episode()
    entity_types = {'Person': MagicMock(__doc__='A person.')}

    await node_operations._extract_entity_summaries_batch(
        _llm_client(10_000),
        _summary_nodes(1),
        episode=episode,
        previous_episodes=None,
        should_summarize_node=None,
        edges_by_node={},
        skip_fact_appending=True,
        entity_types=entity_types,
    )

    assert captured == {
        'episode': episode,
        'entity_types': entity_types,
        'use_episode_prompt': True,
        'max_items': node_operations.MAX_NODES,
    }
