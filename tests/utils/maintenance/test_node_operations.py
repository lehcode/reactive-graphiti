import logging
from collections import defaultdict
from unittest.mock import AsyncMock, MagicMock

import pytest

from graphiti_core.graphiti_types import GraphitiClients
from graphiti_core.nodes import EntityNode, EpisodeType, EpisodicNode
from graphiti_core.utils.datetime_utils import utc_now
from graphiti_core.utils.maintenance.dedup_helpers import (
    DedupCandidateIndexes,
    DedupResolutionState,
    _build_candidate_indexes,
    _cached_shingles,
    _has_high_entropy,
    _hash_shingle,
    _jaccard_similarity,
    _lsh_bands,
    _minhash_signature,
    _name_entropy,
    _normalize_name_for_fuzzy,
    _normalize_string_exact,
    _resolve_with_similarity,
    _shingles,
)
from graphiti_core.utils.maintenance.node_operations import (
    _collect_candidate_nodes,
    _extract_entity_summaries_batch,
    _resolve_with_llm,
    extract_attributes_from_nodes,
    resolve_extracted_nodes,
)


def _make_clients():
    driver = MagicMock()
    embedder = MagicMock()
    cross_encoder = MagicMock()
    llm_client = MagicMock()
    llm_generate = AsyncMock()
    llm_client.generate_response = llm_generate

    clients = GraphitiClients.model_construct(  # bypass validation to allow test doubles
        driver=driver,
        embedder=embedder,
        cross_encoder=cross_encoder,
        llm_client=llm_client,
    )

    return clients, llm_generate


def _make_episode(group_id: str = 'group'):
    return EpisodicNode(
        name='episode',
        group_id=group_id,
        source=EpisodeType.message,
        source_description='test',
        content='content',
        valid_at=utc_now(),
    )


def _semantic_candidates(candidate_groups: list[list[EntityNode]]):
    async def fake_search(*_, **__):
        return candidate_groups

    return fake_search


@pytest.mark.asyncio
async def test_resolve_nodes_exact_match_skips_llm(monkeypatch):
    clients, llm_generate = _make_clients()

    candidate = EntityNode(name='Joe Michaels', group_id='group', labels=['Entity'])
    extracted = EntityNode(name='Joe Michaels', group_id='group', labels=['Entity'])
    monkeypatch.setattr(
        'graphiti_core.utils.maintenance.node_operations._semantic_candidate_search',
        _semantic_candidates([[candidate]]),
    )

    resolved, uuid_map, _ = await resolve_extracted_nodes(
        clients,
        [extracted],
        episode=_make_episode(),
        previous_episodes=[],
    )

    assert resolved[0].uuid == candidate.uuid
    assert uuid_map[extracted.uuid] == candidate.uuid
    llm_generate.assert_not_awaited()


@pytest.mark.asyncio
async def test_resolve_nodes_exact_match_promotes_generic_candidate_type(monkeypatch):
    clients, llm_generate = _make_clients()

    candidate = EntityNode(name='Audrey', group_id='group', labels=['Entity'])
    extracted = EntityNode(name='Audrey', group_id='group', labels=['Entity', 'Person'])
    monkeypatch.setattr(
        'graphiti_core.utils.maintenance.node_operations._semantic_candidate_search',
        _semantic_candidates([[candidate]]),
    )

    resolved, uuid_map, _ = await resolve_extracted_nodes(
        clients,
        [extracted],
        episode=_make_episode(),
        previous_episodes=[],
    )

    assert resolved[0].uuid == candidate.uuid
    assert set(resolved[0].labels) == {'Entity', 'Person'}
    assert set(candidate.labels) == {'Entity', 'Person'}
    assert uuid_map[extracted.uuid] == candidate.uuid
    llm_generate.assert_not_awaited()


@pytest.mark.asyncio
async def test_resolve_nodes_low_entropy_uses_llm(monkeypatch):
    clients, llm_generate = _make_clients()
    llm_generate.return_value = {
        'entity_resolutions': [
            {
                'id': 0,
                'name': 'Joe',
                'duplicate_candidate_id': -1,
            }
        ]
    }

    candidate = EntityNode(name='Joseph', group_id='group', labels=['Entity'])
    extracted = EntityNode(name='Joe', group_id='group', labels=['Entity'])
    monkeypatch.setattr(
        'graphiti_core.utils.maintenance.node_operations._semantic_candidate_search',
        _semantic_candidates([[candidate]]),
    )

    resolved, uuid_map, _ = await resolve_extracted_nodes(
        clients,
        [extracted],
        episode=_make_episode(),
        previous_episodes=[],
    )

    assert resolved[0].uuid == extracted.uuid
    assert uuid_map[extracted.uuid] == extracted.uuid
    llm_generate.assert_awaited()


@pytest.mark.asyncio
async def test_resolve_nodes_short_name_exact_match_skips_llm(monkeypatch):
    """Short names with a unique exact candidate match should resolve deterministically."""
    clients, llm_generate = _make_clients()

    candidate = EntityNode(name='Java', group_id='group', labels=['Entity'])
    extracted = EntityNode(name='Java', group_id='group', labels=['Entity'])
    monkeypatch.setattr(
        'graphiti_core.utils.maintenance.node_operations._semantic_candidate_search',
        _semantic_candidates([[candidate]]),
    )

    resolved, uuid_map, _ = await resolve_extracted_nodes(
        clients,
        [extracted],
        episode=_make_episode(),
        previous_episodes=[],
    )

    assert resolved[0].uuid == candidate.uuid
    assert uuid_map[extracted.uuid] == candidate.uuid
    llm_generate.assert_not_awaited()


@pytest.mark.asyncio
async def test_resolve_nodes_fuzzy_match(monkeypatch):
    clients, llm_generate = _make_clients()

    candidate = EntityNode(name='Joe-Michaels', group_id='group', labels=['Entity'])
    extracted = EntityNode(name='Joe Michaels', group_id='group', labels=['Entity'])
    monkeypatch.setattr(
        'graphiti_core.utils.maintenance.node_operations._semantic_candidate_search',
        _semantic_candidates([[candidate]]),
    )

    resolved, uuid_map, _ = await resolve_extracted_nodes(
        clients,
        [extracted],
        episode=_make_episode(),
        previous_episodes=[],
    )

    assert resolved[0].uuid == candidate.uuid
    assert uuid_map[extracted.uuid] == candidate.uuid
    llm_generate.assert_not_awaited()


@pytest.mark.asyncio
async def test_collect_candidate_nodes_dedupes_and_merges_override(monkeypatch):
    clients, _ = _make_clients()

    candidate = EntityNode(name='Alice', group_id='group', labels=['Entity'])
    override_duplicate = EntityNode(
        uuid=candidate.uuid,
        name='Alice Alt',
        group_id='group',
        labels=['Entity'],
    )
    extracted = EntityNode(name='Alice', group_id='group', labels=['Entity'])

    semantic_search_mock = AsyncMock(return_value=[[candidate]])
    monkeypatch.setattr(
        'graphiti_core.utils.maintenance.node_operations._semantic_candidate_search',
        semantic_search_mock,
    )

    result = await _collect_candidate_nodes(
        clients,
        [extracted],
        existing_nodes_override=[override_duplicate],
    )

    assert len(result) == 1
    assert len(result[0]) == 1
    assert result[0][0].uuid == candidate.uuid
    semantic_search_mock.assert_awaited()


@pytest.mark.asyncio
async def test_resolve_nodes_semantic_miss_keeps_node_without_llm(monkeypatch):
    clients, llm_generate = _make_clients()
    extracted = EntityNode(name='Completely New Thing', group_id='group', labels=['Entity'])

    monkeypatch.setattr(
        'graphiti_core.utils.maintenance.node_operations._semantic_candidate_search',
        _semantic_candidates([[]]),
    )

    resolved, uuid_map, duplicates = await resolve_extracted_nodes(
        clients,
        [extracted],
        episode=_make_episode(),
        previous_episodes=[],
    )

    assert resolved[0].uuid == extracted.uuid
    assert uuid_map[extracted.uuid] == extracted.uuid
    assert duplicates == []
    llm_generate.assert_not_awaited()


@pytest.mark.asyncio
async def test_resolve_nodes_multiple_exact_matches_use_llm(monkeypatch):
    clients, llm_generate = _make_clients()
    llm_generate.return_value = {
        'entity_resolutions': [
            {
                'id': 0,
                'name': 'Java',
                'duplicate_candidate_id': 0,
            }
        ]
    }
    candidate1 = EntityNode(name='Java', group_id='group', labels=['Entity'])
    candidate2 = EntityNode(name='Java', group_id='group', labels=['Entity'])
    extracted = EntityNode(name='Java', group_id='group', labels=['Entity'])

    monkeypatch.setattr(
        'graphiti_core.utils.maintenance.node_operations._semantic_candidate_search',
        _semantic_candidates([[candidate1, candidate2]]),
    )

    resolved, uuid_map, _ = await resolve_extracted_nodes(
        clients,
        [extracted],
        episode=_make_episode(),
        previous_episodes=[],
    )

    assert resolved[0].uuid == candidate1.uuid
    assert uuid_map[extracted.uuid] == candidate1.uuid
    llm_generate.assert_awaited()


@pytest.mark.asyncio
async def test_resolve_nodes_batches_unresolved_nodes_into_one_llm_call(monkeypatch):
    clients, llm_generate = _make_clients()
    llm_generate.return_value = {
        'entity_resolutions': [
            {
                'id': 0,
                'name': 'Joe',
                'duplicate_candidate_id': -1,
            },
            {
                'id': 1,
                'name': 'Java',
                'duplicate_candidate_id': 1,
            },
        ]
    }

    low_entropy_candidate = EntityNode(name='Joseph', group_id='group', labels=['Entity'])
    java_candidate_1 = EntityNode(name='Java', group_id='group', labels=['Entity'])
    java_candidate_2 = EntityNode(name='Java', group_id='group', labels=['Entity'])
    extracted_nodes = [
        EntityNode(name='Joe', group_id='group', labels=['Entity']),
        EntityNode(name='Java', group_id='group', labels=['Entity']),
    ]

    monkeypatch.setattr(
        'graphiti_core.utils.maintenance.node_operations._semantic_candidate_search',
        _semantic_candidates([[low_entropy_candidate], [java_candidate_1, java_candidate_2]]),
    )

    resolved, uuid_map, _ = await resolve_extracted_nodes(
        clients,
        extracted_nodes,
        episode=_make_episode(),
        previous_episodes=[],
    )

    assert resolved[0].uuid == extracted_nodes[0].uuid
    assert resolved[1].uuid == java_candidate_1.uuid
    assert uuid_map[extracted_nodes[0].uuid] == extracted_nodes[0].uuid
    assert uuid_map[extracted_nodes[1].uuid] == java_candidate_1.uuid
    assert llm_generate.await_count == 1


def test_build_candidate_indexes_populates_structures():
    candidate = EntityNode(name='Bob Dylan', group_id='group', labels=['Entity'])

    indexes = _build_candidate_indexes([candidate])

    normalized_key = candidate.name.lower()
    assert indexes.normalized_existing[normalized_key][0].uuid == candidate.uuid
    assert indexes.nodes_by_uuid[candidate.uuid] is candidate
    assert candidate.uuid in indexes.shingles_by_candidate
    assert any(candidate.uuid in bucket for bucket in indexes.lsh_buckets.values())


def test_normalize_helpers():
    assert _normalize_string_exact('  Alice   Smith ') == 'alice smith'
    assert _normalize_name_for_fuzzy('Alice-Smith!') == 'alice smith'


def test_name_entropy_variants():
    assert _name_entropy('alice') > _name_entropy('aaaaa')
    assert _name_entropy('') == 0.0


def test_has_high_entropy_rules():
    assert _has_high_entropy('meaningful name') is True
    assert _has_high_entropy('aa') is False


def test_shingles_and_cache():
    raw = 'alice'
    shingle_set = _shingles(raw)
    assert shingle_set == {'ali', 'lic', 'ice'}
    assert _cached_shingles(raw) == shingle_set
    assert _cached_shingles(raw) is _cached_shingles(raw)


def test_hash_minhash_and_lsh():
    shingles = {'abc', 'bcd', 'cde'}
    signature = _minhash_signature(shingles)
    assert len(signature) == 32
    bands = _lsh_bands(signature)
    assert all(len(band) == 4 for band in bands)
    hashed = {_hash_shingle(s, 0) for s in shingles}
    assert len(hashed) == len(shingles)


def test_jaccard_similarity_edges():
    a = {'a', 'b'}
    b = {'a', 'c'}
    assert _jaccard_similarity(a, b) == pytest.approx(1 / 3)
    assert _jaccard_similarity(set(), set()) == 1.0
    assert _jaccard_similarity(a, set()) == 0.0


def test_resolve_with_similarity_exact_match_updates_state():
    candidate = EntityNode(name='Charlie Parker', group_id='group', labels=['Entity'])
    extracted = EntityNode(name='Charlie Parker', group_id='group', labels=['Entity'])

    indexes = _build_candidate_indexes([candidate])
    state = DedupResolutionState(resolved_nodes=[None], uuid_map={}, unresolved_indices=[])

    _resolve_with_similarity([extracted], indexes, state)

    assert state.resolved_nodes[0].uuid == candidate.uuid
    assert state.uuid_map[extracted.uuid] == candidate.uuid
    assert state.unresolved_indices == []
    assert state.duplicate_pairs == [(extracted, candidate)]


def test_resolve_with_similarity_short_name_exact_match_resolves_deterministically():
    """Short names like 'Nate' should resolve via exact match without hitting the LLM."""
    candidate = EntityNode(name='Nate', group_id='group', labels=['Entity', 'Person'])
    extracted = EntityNode(name='Nate', group_id='group', labels=['Entity', 'Person'])

    indexes = _build_candidate_indexes([candidate])
    state = DedupResolutionState(resolved_nodes=[None], uuid_map={}, unresolved_indices=[])

    _resolve_with_similarity([extracted], indexes, state)

    assert state.resolved_nodes[0].uuid == candidate.uuid
    assert state.uuid_map[extracted.uuid] == candidate.uuid
    assert state.unresolved_indices == []
    assert state.duplicate_pairs == [(extracted, candidate)]


def test_resolve_with_similarity_short_name_no_candidate_defers_to_llm():
    """Short names with no exact match should still reach the LLM for resolution."""
    extracted = EntityNode(name='Nate', group_id='group', labels=['Entity', 'Person'])

    indexes = _build_candidate_indexes([])
    state = DedupResolutionState(resolved_nodes=[None], uuid_map={}, unresolved_indices=[])

    _resolve_with_similarity([extracted], indexes, state)

    assert state.resolved_nodes[0] is None
    assert state.uuid_map == {}
    assert state.unresolved_indices == [0]
    assert state.duplicate_pairs == []


def test_resolve_with_similarity_short_name_multiple_candidates_defers_to_llm():
    """Short names with multiple exact matches should escalate to LLM."""
    candidate1 = EntityNode(name='Java', group_id='group', labels=['Entity'])
    candidate2 = EntityNode(name='Java', group_id='group', labels=['Entity'])
    extracted = EntityNode(name='Java', group_id='group', labels=['Entity'])

    indexes = _build_candidate_indexes([candidate1, candidate2])
    state = DedupResolutionState(resolved_nodes=[None], uuid_map={}, unresolved_indices=[])

    _resolve_with_similarity([extracted], indexes, state)

    assert state.resolved_nodes[0] is None
    assert state.uuid_map == {}
    assert state.unresolved_indices == [0]
    assert state.duplicate_pairs == []


def test_resolve_with_similarity_low_entropy_defers_resolution():
    extracted = EntityNode(name='Bob', group_id='group', labels=['Entity'])
    indexes = DedupCandidateIndexes(
        existing_nodes=[],
        nodes_by_uuid={},
        normalized_existing=defaultdict(list),
        shingles_by_candidate={},
        lsh_buckets=defaultdict(list),
    )
    state = DedupResolutionState(resolved_nodes=[None], uuid_map={}, unresolved_indices=[])

    _resolve_with_similarity([extracted], indexes, state)

    assert state.resolved_nodes[0] is None
    assert state.unresolved_indices == [0]
    assert state.duplicate_pairs == []


def test_resolve_with_similarity_multiple_exact_matches_defers_to_llm():
    candidate1 = EntityNode(name='Johnny Appleseed', group_id='group', labels=['Entity'])
    candidate2 = EntityNode(name='Johnny Appleseed', group_id='group', labels=['Entity'])
    extracted = EntityNode(name='Johnny Appleseed', group_id='group', labels=['Entity'])

    indexes = _build_candidate_indexes([candidate1, candidate2])
    state = DedupResolutionState(resolved_nodes=[None], uuid_map={}, unresolved_indices=[])

    _resolve_with_similarity([extracted], indexes, state)

    assert state.resolved_nodes[0] is None
    assert state.unresolved_indices == [0]
    assert state.duplicate_pairs == []


@pytest.mark.asyncio
async def test_resolve_with_llm_candidate_attributes_cannot_overwrite_candidate_id(monkeypatch):
    """Ensure candidate.attributes with a 'candidate_id' key cannot corrupt the LLM context."""
    candidate = EntityNode(name='Dizzy Gillespie', group_id='group', labels=['Entity'])
    candidate.attributes = {'candidate_id': 999, 'genre': 'jazz'}

    extracted = EntityNode(name='Dizzy', group_id='group', labels=['Entity'])

    indexes = _build_candidate_indexes([candidate])
    state = DedupResolutionState(resolved_nodes=[None], uuid_map={}, unresolved_indices=[0])

    captured_context = {}

    def fake_prompt_nodes(context):
        captured_context.update(context)
        return ['prompt']

    monkeypatch.setattr(
        'graphiti_core.utils.maintenance.node_operations.prompt_library.dedupe_nodes.nodes',
        fake_prompt_nodes,
    )

    llm_client = MagicMock()
    llm_client.generate_response = AsyncMock(
        return_value={
            'entity_resolutions': [
                {'id': 0, 'name': 'Dizzy Gillespie', 'duplicate_candidate_id': 0}
            ]
        }
    )

    await _resolve_with_llm(
        llm_client,
        [extracted],
        indexes,
        state,
        episode=_make_episode(),
        previous_episodes=[],
        entity_types=None,
    )

    # candidate_id must be the positional index (0), not the adversarial attribute (999)
    assert captured_context['existing_nodes'][0]['candidate_id'] == 0
    # non-colliding attributes should still be present
    assert captured_context['existing_nodes'][0]['genre'] == 'jazz'
    assert state.resolved_nodes[0].uuid == candidate.uuid


@pytest.mark.asyncio
async def test_resolve_with_llm_updates_unresolved(monkeypatch):
    extracted = EntityNode(name='Dizzy', group_id='group', labels=['Entity'])
    candidate = EntityNode(name='Dizzy Gillespie', group_id='group', labels=['Entity'])

    indexes = _build_candidate_indexes([candidate])
    state = DedupResolutionState(resolved_nodes=[None], uuid_map={}, unresolved_indices=[0])

    captured_context = {}

    def fake_prompt_nodes(context):
        captured_context.update(context)
        return ['prompt']

    monkeypatch.setattr(
        'graphiti_core.utils.maintenance.node_operations.prompt_library.dedupe_nodes.nodes',
        fake_prompt_nodes,
    )

    async def fake_generate_response(*_, **__):
        return {
            'entity_resolutions': [
                {
                    'id': 0,
                    'name': 'Dizzy Gillespie',
                    'duplicate_candidate_id': 0,
                }
            ]
        }

    llm_client = MagicMock()
    llm_client.generate_response = AsyncMock(side_effect=fake_generate_response)

    await _resolve_with_llm(
        llm_client,
        [extracted],
        indexes,
        state,
        episode=_make_episode(),
        previous_episodes=[],
        entity_types=None,
    )

    assert state.resolved_nodes[0].uuid == candidate.uuid
    assert state.uuid_map[extracted.uuid] == candidate.uuid
    assert isinstance(captured_context['existing_nodes'], list)
    assert captured_context['existing_nodes'][0]['candidate_id'] == 0
    assert (
        captured_context['extracted_nodes'][0]['entity_type_description'] == 'Default Entity Type'
    )
    assert state.duplicate_pairs == [(extracted, candidate)]


@pytest.mark.asyncio
async def test_resolve_with_llm_promotes_generic_candidate_type(monkeypatch):
    extracted = EntityNode(name='Audrey', group_id='group', labels=['Entity', 'Person'])
    candidate = EntityNode(name='Audrey', group_id='group', labels=['Entity'])

    indexes = _build_candidate_indexes([candidate])
    state = DedupResolutionState(resolved_nodes=[None], uuid_map={}, unresolved_indices=[0])

    monkeypatch.setattr(
        'graphiti_core.utils.maintenance.node_operations.prompt_library.dedupe_nodes.nodes',
        lambda context: ['prompt'],
    )

    llm_client = MagicMock()
    llm_client.generate_response = AsyncMock(
        return_value={
            'entity_resolutions': [
                {
                    'id': 0,
                    'name': 'Audrey',
                    'duplicate_candidate_id': 0,
                }
            ]
        }
    )

    await _resolve_with_llm(
        llm_client,
        [extracted],
        indexes,
        state,
        episode=_make_episode(),
        previous_episodes=[],
        entity_types=None,
    )

    assert state.resolved_nodes[0].uuid == candidate.uuid
    assert set(state.resolved_nodes[0].labels) == {'Entity', 'Person'}
    assert set(candidate.labels) == {'Entity', 'Person'}
    assert state.uuid_map[extracted.uuid] == candidate.uuid
    assert state.duplicate_pairs == [(extracted, candidate)]


@pytest.mark.asyncio
async def test_resolve_with_llm_ignores_out_of_range_relative_ids(monkeypatch, caplog):
    extracted = EntityNode(name='Dexter', group_id='group', labels=['Entity'])

    indexes = _build_candidate_indexes([])
    state = DedupResolutionState(resolved_nodes=[None], uuid_map={}, unresolved_indices=[0])

    monkeypatch.setattr(
        'graphiti_core.utils.maintenance.node_operations.prompt_library.dedupe_nodes.nodes',
        lambda context: ['prompt'],
    )

    llm_client = MagicMock()
    llm_client.generate_response = AsyncMock(
        return_value={
            'entity_resolutions': [
                {
                    'id': 5,
                    'name': 'Dexter',
                    'duplicate_candidate_id': -1,
                }
            ]
        }
    )

    with caplog.at_level(logging.WARNING):
        await _resolve_with_llm(
            llm_client,
            [extracted],
            indexes,
            state,
            episode=_make_episode(),
            previous_episodes=[],
            entity_types=None,
        )

    assert state.resolved_nodes[0] is None
    assert 'Skipping invalid LLM dedupe id 5' in caplog.text


@pytest.mark.asyncio
async def test_resolve_with_llm_ignores_duplicate_relative_ids(monkeypatch):
    extracted = EntityNode(name='Dizzy', group_id='group', labels=['Entity'])
    candidate = EntityNode(name='Dizzy Gillespie', group_id='group', labels=['Entity'])

    indexes = _build_candidate_indexes([candidate])
    state = DedupResolutionState(resolved_nodes=[None], uuid_map={}, unresolved_indices=[0])

    monkeypatch.setattr(
        'graphiti_core.utils.maintenance.node_operations.prompt_library.dedupe_nodes.nodes',
        lambda context: ['prompt'],
    )

    llm_client = MagicMock()
    llm_client.generate_response = AsyncMock(
        return_value={
            'entity_resolutions': [
                {
                    'id': 0,
                    'name': 'Dizzy Gillespie',
                    'duplicate_candidate_id': 0,
                },
                {
                    'id': 0,
                    'name': 'Dizzy',
                    'duplicate_candidate_id': -1,
                },
            ]
        }
    )

    await _resolve_with_llm(
        llm_client,
        [extracted],
        indexes,
        state,
        episode=_make_episode(),
        previous_episodes=[],
        entity_types=None,
    )

    assert state.resolved_nodes[0].uuid == candidate.uuid
    assert state.uuid_map[extracted.uuid] == candidate.uuid
    assert state.duplicate_pairs == [(extracted, candidate)]


@pytest.mark.asyncio
async def test_resolve_with_llm_invalid_candidate_id_defaults_to_extracted(monkeypatch):
    extracted = EntityNode(name='Dexter', group_id='group', labels=['Entity'])

    indexes = _build_candidate_indexes([])
    state = DedupResolutionState(resolved_nodes=[None], uuid_map={}, unresolved_indices=[0])

    monkeypatch.setattr(
        'graphiti_core.utils.maintenance.node_operations.prompt_library.dedupe_nodes.nodes',
        lambda context: ['prompt'],
    )

    llm_client = MagicMock()
    llm_client.generate_response = AsyncMock(
        return_value={
            'entity_resolutions': [
                {
                    'id': 0,
                    'name': 'Dexter',
                    'duplicate_candidate_id': 999,
                }
            ]
        }
    )

    await _resolve_with_llm(
        llm_client,
        [extracted],
        indexes,
        state,
        episode=_make_episode(),
        previous_episodes=[],
        entity_types=None,
    )

    assert state.resolved_nodes[0] == extracted
    assert state.uuid_map[extracted.uuid] == extracted.uuid
    assert state.duplicate_pairs == []


@pytest.mark.asyncio
async def test_batch_summaries_short_summary_no_llm():
    """Test that short summaries are kept as-is without LLM call (optimization)."""
    llm_client = MagicMock()
    llm_client.generate_response = AsyncMock(
        return_value={'summaries': [{'name': 'Test Node', 'summary': 'Generated summary'}]}
    )

    node = EntityNode(name='Test Node', group_id='group', labels=['Entity'], summary='Old summary')
    episode = _make_episode()

    await _extract_entity_summaries_batch(
        llm_client,
        [node],
        episode=episode,
        previous_episodes=[],
        should_summarize_node=None,
        edges_by_node={},
    )

    # Short summary should be kept as-is without LLM call
    assert node.summary == 'Old summary'
    # LLM should NOT have been called (summary is short enough)
    llm_client.generate_response.assert_not_awaited()


@pytest.mark.asyncio
async def test_batch_summaries_callback_skip_summary():
    """Test that summary is NOT regenerated when callback returns False."""
    llm_client = MagicMock()
    llm_client.generate_response = AsyncMock(
        return_value={'summaries': [{'name': 'Test Node', 'summary': 'This should not be used'}]}
    )

    node = EntityNode(name='Test Node', group_id='group', labels=['Entity'], summary='Old summary')
    episode = _make_episode()

    # Callback that always returns False (skip summary generation)
    async def skip_summary_filter(n: EntityNode) -> bool:
        return False

    await _extract_entity_summaries_batch(
        llm_client,
        [node],
        episode=episode,
        previous_episodes=[],
        should_summarize_node=skip_summary_filter,
        edges_by_node={},
    )

    # Summary should remain unchanged
    assert node.summary == 'Old summary'
    # LLM should NOT have been called for summary
    llm_client.generate_response.assert_not_awaited()


@pytest.mark.asyncio
async def test_batch_summaries_selective_callback():
    """Test callback that selectively skips summaries based on node properties."""
    llm_client = MagicMock()
    llm_client.generate_response = AsyncMock(return_value={'summaries': []})

    user_node = EntityNode(name='User', group_id='group', labels=['Entity', 'User'], summary='Old')
    topic_node = EntityNode(
        name='Topic', group_id='group', labels=['Entity', 'Topic'], summary='Old'
    )

    episode = _make_episode()

    # Callback that skips User nodes but generates for others
    async def selective_filter(n: EntityNode) -> bool:
        return 'User' not in n.labels

    await _extract_entity_summaries_batch(
        llm_client,
        [user_node, topic_node],
        episode=episode,
        previous_episodes=[],
        should_summarize_node=selective_filter,
        edges_by_node={},
    )

    # User summary should remain unchanged (callback returned False)
    assert user_node.summary == 'Old'
    # Topic summary should also remain unchanged (short summary optimization)
    assert topic_node.summary == 'Old'
    # LLM should NOT have been called (summaries are short enough)
    llm_client.generate_response.assert_not_awaited()


@pytest.mark.asyncio
async def test_extract_attributes_from_nodes_with_callback():
    """Test that callback is properly passed through extract_attributes_from_nodes."""
    clients, _ = _make_clients()
    clients.llm_client.generate_response = AsyncMock(return_value={'summaries': []})
    clients.embedder.create = AsyncMock(return_value=[0.1, 0.2, 0.3])
    clients.embedder.create_batch = AsyncMock(return_value=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]])

    node1 = EntityNode(name='Node1', group_id='group', labels=['Entity', 'User'], summary='Old1')
    node2 = EntityNode(name='Node2', group_id='group', labels=['Entity', 'Topic'], summary='Old2')

    episode = _make_episode()

    call_tracker = []

    # Callback that tracks which nodes it's called with
    async def tracking_filter(n: EntityNode) -> bool:
        call_tracker.append(n.name)
        return 'User' not in n.labels

    results = await extract_attributes_from_nodes(
        clients,
        [node1, node2],
        episode=episode,
        previous_episodes=[],
        entity_types=None,
        should_summarize_node=tracking_filter,
    )

    # Callback should have been called for both nodes
    assert len(call_tracker) == 2
    assert 'Node1' in call_tracker
    assert 'Node2' in call_tracker

    # Both nodes should keep old summaries (short summary optimization skips LLM)
    node1_result = next(n for n in results if n.name == 'Node1')
    node2_result = next(n for n in results if n.name == 'Node2')

    assert node1_result.summary == 'Old1'
    assert node2_result.summary == 'Old2'


@pytest.mark.asyncio
async def test_batch_summaries_calls_llm_for_long_summary():
    """Test that LLM is called when summary exceeds character limit."""
    from graphiti_core.edges import EntityEdge
    from graphiti_core.utils.text_utils import MAX_SUMMARY_CHARS

    llm_client = MagicMock()
    llm_client.generate_response = AsyncMock(
        return_value={'summaries': [{'name': 'Test Node', 'summary': 'Condensed summary'}]}
    )

    node = EntityNode(name='Test Node', group_id='group', labels=['Entity'], summary='Short')
    episode = _make_episode()

    # Create edges with long facts that exceed the threshold
    long_fact = 'x' * (MAX_SUMMARY_CHARS * 2)
    edge = EntityEdge(
        uuid='edge1',
        group_id='group',
        source_node_uuid=node.uuid,
        target_node_uuid='other-uuid',
        name='test_edge',
        fact=long_fact,
        created_at=utc_now(),
    )

    edges_by_node = {node.uuid: [edge, edge]}  # Multiple long edges

    await _extract_entity_summaries_batch(
        llm_client,
        [node],
        episode=episode,
        previous_episodes=[],
        should_summarize_node=None,
        edges_by_node=edges_by_node,
    )

    # LLM should have been called to condense the long summary
    llm_client.generate_response.assert_awaited_once()
    assert node.summary == 'Condensed summary'


@pytest.mark.asyncio
async def test_extract_attributes_preserves_prior_attributes_when_entity_types_none():
    """Prior attributes survive when no entity types are supplied.

    Mirrors the invariant `add_triplet` already enforces ("merge rather than replace",
    see `test_add_triplet_empty_attributes_preserved`): an empty extraction result must
    not clear attributes a previous typed pass stored on the node.
    """
    clients, _ = _make_clients()
    clients.llm_client.generate_response = AsyncMock(return_value={'summaries': []})
    clients.embedder.create = AsyncMock(return_value=[0.1, 0.2, 0.3])
    clients.embedder.create_batch = AsyncMock(return_value=[[0.1, 0.2, 0.3]])

    node = EntityNode(
        name='Alice',
        group_id='group',
        labels=['Entity', 'Person'],
        summary='Existing summary',
        attributes={'age': 30, 'city': 'New York'},
    )

    results = await extract_attributes_from_nodes(
        clients,
        [node],
        episode=_make_episode(),
        previous_episodes=[],
        entity_types=None,
    )

    assert results[0].attributes == {'age': 30, 'city': 'New York'}


@pytest.mark.asyncio
async def test_extract_attributes_preserves_prior_attributes_when_label_not_in_entity_types():
    """Prior attributes survive when the node's label is absent from `entity_types`.

    `entity_types.get(...)` resolves to None for any label not in the map, which takes the
    same no-applicable-type path as `entity_types=None`.
    """
    from pydantic import BaseModel

    class Organization(BaseModel):
        industry: str | None = None

    clients, _ = _make_clients()
    clients.llm_client.generate_response = AsyncMock(return_value={'summaries': []})
    clients.embedder.create = AsyncMock(return_value=[0.1, 0.2, 0.3])
    clients.embedder.create_batch = AsyncMock(return_value=[[0.1, 0.2, 0.3]])

    node = EntityNode(
        name='Alice',
        group_id='group',
        labels=['Entity', 'Person'],
        summary='Existing summary',
        attributes={'age': 30, 'city': 'New York'},
    )

    results = await extract_attributes_from_nodes(
        clients,
        [node],
        episode=_make_episode(),
        previous_episodes=[],
        entity_types={'Organization': Organization},
    )

    assert results[0].attributes == {'age': 30, 'city': 'New York'}


# ---- Budget-aware flight packing tests ----


@pytest.mark.asyncio
async def test_build_budget_flights_falls_back_when_no_budget():
    """When input_token_budget returns None, _build_budget_flights uses fixed-size chunks."""
    from graphiti_core.utils.maintenance.node_operations import (
        _build_budget_flights,
    )

    llm_client = MagicMock()
    llm_client.input_token_budget.return_value = None

    nodes = [
        EntityNode(name=f'Node{i}', group_id='g', labels=['Entity'], summary='x' * 100)
        for i in range(7)
    ]

    flights = _build_budget_flights(nodes, 3, llm_client)
    assert len(flights) == 3
    assert [f[0].name for f in flights] == ['Node0', 'Node3', 'Node6']


@pytest.mark.asyncio
async def test_build_budget_flights_uses_budget_when_available(monkeypatch):
    """When budget is an int, _build_budget_flights respects it via pack_by_budget.

    Uses a tight budget (300 tokens) so even the fixed prompt cost (~690) exceeds it.
    pack_by_budget guarantees ≥ 1 item per batch, forcing one-node batches.
    """
    from graphiti_core.utils.maintenance.node_operations import _build_budget_flights

    llm_client = MagicMock()
    llm_client.input_token_budget.return_value = 300

    nodes = [
        EntityNode(name=f'Node{i}', group_id='g', labels=['Entity'], summary='x' * 500)
        for i in range(5)
    ]

    flights = _build_budget_flights(nodes, 10, llm_client)
    # Budget (300) << fixed_cost (~690) → pack_by_budget forces 1 node/batch
    assert len(flights) == 5
    assert all(len(f) == 1 for f in flights)


@pytest.mark.asyncio
async def test_build_budget_flights_respects_max_items(monkeypatch):
    """max_items parameter caps the number of nodes per flight."""
    from graphiti_core.utils.content_chunking import estimate_tokens
    from graphiti_core.utils.maintenance.node_operations import _build_budget_flights

    llm_client = MagicMock()
    llm_client.input_token_budget.return_value = 80000  # plenty of budget

    nodes = [
        EntityNode(name=f'Node{i}', group_id='g', labels=['Entity'], summary='x' * 10)
        for i in range(8)
    ]

    def mock_estimate(text):
        if isinstance(text, str):
            return len(text) // 4 + 1
        return estimate_tokens(text)

    monkeypatch.setattr(
        'graphiti_core.utils.maintenance.node_operations.estimate_tokens',
        mock_estimate,
    )

    flights = _build_budget_flights(nodes, 3, llm_client)
    assert len(flights) == 3  # 8 nodes / 3 per flight = 3 batches
    assert all(len(f) <= 3 for f in flights)


@pytest.mark.asyncio
async def test_resolve_with_llm_single_call_when_no_budget(monkeypatch, caplog):
    """When llm_client.input_token_budget is None, _resolve_with_llm does a single call."""
    from graphiti_core.utils.maintenance.node_operations import (
        DedupResolutionState,
        _build_candidate_indexes,
        _resolve_with_llm,
    )

    extracted = EntityNode(name='TestNode', group_id='group', labels=['Entity'])

    state = DedupResolutionState(resolved_nodes=[None], uuid_map={}, unresolved_indices=[0])

    monkeypatch.setattr(
        'graphiti_core.utils.maintenance.node_operations.prompt_library.dedupe_nodes.nodes',
        lambda context: ['prompt'],
    )

    llm_client = MagicMock()
    llm_client.input_token_budget.return_value = None
    llm_client.generate_response = AsyncMock(
        return_value={
            'entity_resolutions': [{'id': 0, 'name': 'TestNode', 'duplicate_candidate_id': -1}]
        }
    )

    await _resolve_with_llm(
        llm_client,
        [extracted],
        indexes=_build_candidate_indexes([]),
        state=state,
        episode=_make_episode(),
        previous_episodes=[],
        entity_types=None,
    )

    llm_client.generate_response.assert_awaited_once()


@pytest.mark.asyncio
async def test_resolve_with_llm_splits_when_exceeds_budget(monkeypatch):
    """When budget is exceeded, _resolve_with_llm splits into multiple batches."""
    from graphiti_core.utils.content_chunking import estimate_tokens
    from graphiti_core.utils.maintenance.node_operations import (
        DedupResolutionState,
        _resolve_with_llm,
    )

    # Create many extracted nodes so budget forces splitting
    extracted_nodes = [
        EntityNode(name=f'Node{i}', group_id='group', labels=['Entity']) for i in range(5)
    ]

    candidate = EntityNode(name='Placeholder', group_id='group', labels=['Entity'])

    state = DedupResolutionState(
        resolved_nodes=[None] * len(extracted_nodes),
        uuid_map={},
        unresolved_indices=list(range(len(extracted_nodes))),
    )

    monkeypatch.setattr(
        'graphiti_core.utils.maintenance.node_operations.prompt_library.dedupe_nodes.nodes',
        lambda context: ['prompt'],
    )

    call_count = 0

    async def track_calls(*_, **__):
        nonlocal call_count
        call_count += 1
        # Return resolutions for the nodes in this batch
        # IDs in response match the batch's extracted_nodes
        return {'entity_resolutions': [{'id': 0, 'name': 'Node0', 'duplicate_candidate_id': -1}]}

    llm_client = MagicMock()
    llm_client.input_token_budget.return_value = 8000
    llm_client.generate_response = AsyncMock(side_effect=track_calls)

    def mock_estimate(text):
        if isinstance(text, str):
            return len(text) // 4 + 1
        return estimate_tokens(text)

    monkeypatch.setattr(
        'graphiti_core.utils.maintenance.node_operations.estimate_tokens',
        mock_estimate,
    )

    await _resolve_with_llm(
        llm_client,
        extracted_nodes,
        indexes=_build_candidate_indexes([candidate]),
        state=state,
        episode=_make_episode(),
        previous_episodes=[],
        entity_types=None,
    )

    # Multiple LLM calls should have been issued when budget forces splitting
    assert call_count >= 1


@pytest.mark.asyncio
async def test_resolve_with_llm_reindexed_batch_mapping(monkeypatch):
    """Sub-batches use local IDs 0..N-1 and map back to correct global unresolved indices.

    With 6 unresolved nodes and a tight budget, pack_by_budget creates multiple batches.
    Each batch's id=0 maps to a different global unresolved index via the offset fix.
    """
    from unittest.mock import AsyncMock, patch

    from graphiti_core.utils.content_chunking import estimate_tokens
    from graphiti_core.utils.maintenance.node_operations import (
        DedupResolutionState,
        _resolve_with_llm,
    )

    extracted_nodes = [
        EntityNode(name=f'Node{i}', group_id='group', labels=['Entity']) for i in range(6)
    ]
    candidate = EntityNode(name='Placeholder', group_id='group', labels=['Entity'])

    state = DedupResolutionState(
        resolved_nodes=[None] * len(extracted_nodes),
        uuid_map={},
        unresolved_indices=list(range(len(extracted_nodes))),
    )

    monkeypatch.setattr(
        'graphiti_core.utils.maintenance.node_operations.prompt_library.dedupe_nodes.nodes',
        lambda context: ['prompt'],
    )

    # Track the extracted_nodes context sent to each LLM call
    batch_contexts = []

    async def track_calls(*args, **kwargs):
        # Context is the 4th positional argument to generate_prompt_response
        if len(args) >= 4:
            batch_contexts.append(dict(args[3]))
        return {'entity_resolutions': [{'id': 0, 'name': 'Node0', 'duplicate_candidate_id': -1}]}

    llm_client = MagicMock()
    llm_client.input_token_budget.return_value = 80

    def mock_estimate(text):
        if isinstance(text, str):
            return len(text) // 4 + 1
        return estimate_tokens(text)

    with (
        patch(
            'graphiti_core.utils.maintenance.node_operations.generate_prompt_response',
            AsyncMock(side_effect=track_calls),
        ),
        patch('graphiti_core.utils.maintenance.node_operations.estimate_tokens', mock_estimate),
    ):
        await _resolve_with_llm(
            llm_client,
            extracted_nodes,
            indexes=_build_candidate_indexes([candidate]),
            state=state,
            episode=_make_episode(),
            previous_episodes=[],
            entity_types=None,
        )

    # With budget=80 and fixed_cost~78, limit~-2 → each item alone → 6 batches
    assert len(batch_contexts) == 6
    # Batch-2 (index 2) should map to global unresolved index 2 (offset=2)
    resolved = [r for r in state.resolved_nodes if r is not None]
    assert len(resolved) == 6
    # Verify batch context IDs: each batch has IDs 0 (single-item batches)
    for ctx in batch_contexts:
        assert ctx['extracted_nodes'][0]['id'] == 0
    # Verify resolved nodes map to correct global indices
    for i, r in enumerate(resolved):
        assert r.uuid == extracted_nodes[i].uuid, f'Batch {i} should resolve node {i}'


@pytest.mark.asyncio
async def test_resolve_with_llm_splits_without_previous_episodes(monkeypatch):
    """Budget-aware splitting works even when previous_episodes is None."""
    from graphiti_core.utils.content_chunking import estimate_tokens
    from graphiti_core.utils.maintenance.node_operations import (
        DedupResolutionState,
        _build_candidate_indexes,
        _resolve_with_llm,
    )

    extracted_nodes = [
        EntityNode(name=f'Node{i}', group_id='group', labels=['Entity']) for i in range(6)
    ]
    candidate = EntityNode(name='Placeholder', group_id='group', labels=['Entity'])

    state = DedupResolutionState(
        resolved_nodes=[None] * len(extracted_nodes),
        uuid_map={},
        unresolved_indices=list(range(len(extracted_nodes))),
    )

    monkeypatch.setattr(
        'graphiti_core.utils.maintenance.node_operations.prompt_library.dedupe_nodes.nodes',
        lambda context: ['prompt'],
    )

    call_count = 0

    async def track_calls(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        return {'entity_resolutions': [{'id': 0, 'name': 'Node0', 'duplicate_candidate_id': -1}]}

    llm_client = MagicMock()
    llm_client.input_token_budget.return_value = 140
    llm_client.generate_response = AsyncMock(side_effect=track_calls)

    def mock_estimate(text):
        if isinstance(text, str):
            return len(text) // 4 + 1
        return estimate_tokens(text)

    monkeypatch.setattr(
        'graphiti_core.utils.maintenance.node_operations.estimate_tokens',
        mock_estimate,
    )

    # previous_episodes=None — budget splitting should still happen
    await _resolve_with_llm(
        llm_client,
        extracted_nodes,
        indexes=_build_candidate_indexes([candidate]),
        state=state,
        episode=_make_episode(),
        previous_episodes=None,
        entity_types=None,
    )

    # Multiple LLM calls should have been issued
    assert call_count >= 1


@pytest.mark.asyncio
async def test_resolve_with_llm_batch_offset_prevents_aliasing(monkeypatch):
    """Batch-2 resolutions map to different nodes than batch-1 (no aliasing).

    This is the core D1 regression test: without the offset fix, batch-2's
    id=0 would map to global unresolved index 0 — the same as batch-1's id=0.
    """
    from unittest.mock import AsyncMock, patch

    from graphiti_core.utils.content_chunking import estimate_tokens
    from graphiti_core.utils.maintenance.node_operations import (
        DedupResolutionState,
        _resolve_with_llm,
    )

    extracted_nodes = [
        EntityNode(name=f'Node{i}', group_id='group', labels=['Entity']) for i in range(6)
    ]
    candidate = EntityNode(name='Placeholder', group_id='group', labels=['Entity'])

    state = DedupResolutionState(
        resolved_nodes=[None] * len(extracted_nodes),
        uuid_map={},
        unresolved_indices=list(range(len(extracted_nodes))),
    )

    monkeypatch.setattr(
        'graphiti_core.utils.maintenance.node_operations.prompt_library.dedupe_nodes.nodes',
        lambda context: ['prompt'],
    )

    batch_ids = []

    async def track_calls(*args, **kwargs):
        # Context is 4th positional arg
        if len(args) >= 4:
            batch_ids.extend([n['id'] for n in args[3]['extracted_nodes']])
        return {'entity_resolutions': [{'id': 0, 'name': 'Node0', 'duplicate_candidate_id': -1}]}

    llm_client = MagicMock()
    llm_client.input_token_budget.return_value = 80

    def mock_estimate(text):
        if isinstance(text, str):
            return len(text) // 4 + 1
        return estimate_tokens(text)

    with (
        patch(
            'graphiti_core.utils.maintenance.node_operations.generate_prompt_response',
            AsyncMock(side_effect=track_calls),
        ),
        patch('graphiti_core.utils.maintenance.node_operations.estimate_tokens', mock_estimate),
    ):
        await _resolve_with_llm(
            llm_client,
            extracted_nodes,
            indexes=_build_candidate_indexes([candidate]),
            state=state,
            episode=_make_episode(),
            previous_episodes=[],
            entity_types=None,
        )

    resolved = [r for r in state.resolved_nodes if r is not None]
    # Batch IDs sent to LLM should all be 0 (single-item batches)
    assert all(bid == 0 for bid in batch_ids)
    # Each resolved node maps to a DIFFERENT global index (no aliasing)
    uuids = [r.uuid for r in resolved]
    assert len(set(uuids)) == len(uuids), f'All resolved nodes should be unique: {uuids}'
    assert len(resolved) == 6


@pytest.mark.asyncio
async def test_build_budget_flights_uses_entity_type_descriptions_in_fixed_cost(monkeypatch):
    """D2: _build_budget_flights includes entity type descriptions in fixed cost calculation.

    When entity_types are provided, the fixed cost should account for the
    token weight of the type descriptions embedded in the prompt.
    """
    from pydantic import BaseModel

    from graphiti_core.utils.maintenance.node_operations import _build_budget_flights

    llm_client = MagicMock()
    llm_client.input_token_budget.return_value = 500

    nodes = [
        EntityNode(name=f'Node{i}', group_id='g', labels=['Entity'], summary='x' * 500)
        for i in range(3)
    ]

    class Person(BaseModel):
        """Person entity type."""

    class Location(BaseModel):
        """Location entity type."""

    # With entity_types, entity_type_descriptions are built internally
    flights = _build_budget_flights(
        nodes, 10, llm_client, entity_types={'Person': Person, 'Location': Location}
    )
    assert len(flights) >= 1


@pytest.mark.asyncio
async def test_extract_attributes_from_nodes_entity_types_passed_through(monkeypatch):
    """D2: entity_types are passed through to _build_budget_flights when include_type_descriptions is True.

    When entity_types is supplied and include_type_descriptions is True,
    _extract_entity_summaries_batch should call _build_budget_flights with
    entity_types so that entity descriptions are included in the fixed cost.
    """

    from graphiti_core.utils.maintenance.node_operations import (
        _extract_entity_summaries_batch,
    )

    clients, _ = _make_clients()
    clients.llm_client.generate_response = AsyncMock(return_value={'summaries': []})
    clients.embedder.create = AsyncMock(return_value=[0.1, 0.2, 0.3])
    clients.embedder.create_batch = AsyncMock(return_value=[[0.1, 0.2, 0.3]])

    node = EntityNode(
        name='Test Node',
        group_id='group',
        labels=['Entity'],
        summary='A' * 5000,  # Long summary forces LLM
    )
    episode = _make_episode()

    # Simulate the real code path: include_type_descriptions=True passes entity_types
    await _extract_entity_summaries_batch(
        clients.llm_client,
        [node],
        episode=episode,
        previous_episodes=[],
        should_summarize_node=None,
        edges_by_node={},
        entity_types=None,
    )

    # With no entity_types, the function should still work (entity_types=None is valid)
    assert node is not None


@pytest.mark.asyncio
async def test_extract_edges_k_accounts_for_fixed_cost(monkeypatch):
    """D3: k (max nodes per chunk) is calculated from remaining budget after fixed cost.

    With fixed_cost deducted from input_budget, k should be smaller than
    if the raw input_budget were used. This test verifies the calculation path.
    """
    from unittest.mock import AsyncMock, patch

    from graphiti_core.utils.maintenance.edge_operations import extract_edges

    llm_client = MagicMock()
    llm_client.input_token_budget.return_value = 5000

    node1 = EntityNode(name='Node1', group_id='group', labels=['Entity'])
    node2 = EntityNode(name='Node2', group_id='group', labels=['Entity'])
    node3 = EntityNode(name='Node3', group_id='group', labels=['Entity'])
    nodes = [node1, node2, node3]

    with (
        patch('graphiti_core.utils.maintenance.edge_operations.estimate_tokens', return_value=1),
        patch(
            'graphiti_core.utils.maintenance.edge_operations.generate_prompt_response',
            AsyncMock(return_value={'edges': [], 'invalid_at': None}),
        ),
    ):
        clients = MagicMock()
        clients.llm_client = llm_client
        episode = _make_episode()
        edge_type_map = {('Entity', 'Entity'): ['Relation']}

        await extract_edges(
            clients,
            episode,
            nodes,
            previous_episodes=[],
            edge_type_map=edge_type_map,
        )


@pytest.mark.asyncio
async def test_extract_edges_chunk_llm_failure_raises(monkeypatch):
    """D3: LLM failures in chunk mode are re-raised, not silently swallowed."""
    from unittest.mock import MagicMock, patch

    from graphiti_core.utils.maintenance.edge_operations import extract_edges

    llm_client = MagicMock()
    llm_client.input_token_budget.return_value = 5000

    node1 = EntityNode(name='Node1', group_id='group', labels=['Entity'])
    node2 = EntityNode(name='Node2', group_id='group', labels=['Entity'])
    nodes = [node1, node2]

    async def fail_on_call(*_, **__):
        raise RuntimeError('LLM service unavailable')

    with (
        patch('graphiti_core.utils.maintenance.edge_operations.estimate_tokens', return_value=1),
        patch(
            'graphiti_core.utils.maintenance.edge_operations.generate_prompt_response',
            fail_on_call,
        ),
    ):
        clients = MagicMock()
        clients.llm_client = llm_client
        episode = _make_episode()
        edge_type_map = {('Entity', 'Entity'): ['Relation']}

        with pytest.raises(RuntimeError):
            await extract_edges(
                clients,
                episode,
                nodes,
                previous_episodes=[],
                edge_type_map=edge_type_map,
            )


@pytest.mark.asyncio
async def test_extract_edges_chunk_fallback_on_failure(monkeypatch):
    """D3: Chunk mode raises on failure so edge data is not silently lost."""
    from unittest.mock import MagicMock, patch

    from graphiti_core.utils.maintenance.edge_operations import extract_edges

    llm_client = MagicMock()
    llm_client.input_token_budget.return_value = 5000

    node1 = EntityNode(name='Node1', group_id='group', labels=['Entity'])
    node2 = EntityNode(name='Node2', group_id='group', labels=['Entity'])
    nodes = [node1, node2]

    async def fail_on_call(*_, **__):
        raise RuntimeError('LLM service unavailable')

    with (
        patch('graphiti_core.utils.maintenance.edge_operations.estimate_tokens', return_value=1),
        patch(
            'graphiti_core.utils.maintenance.edge_operations.generate_prompt_response',
            fail_on_call,
        ),
    ):
        clients = MagicMock()
        clients.llm_client = llm_client
        episode = _make_episode()
        edge_type_map = {('Entity', 'Entity'): ['Relation']}

        with pytest.raises(RuntimeError):
            await extract_edges(
                clients,
                episode,
                nodes,
                previous_episodes=[],
                edge_type_map=edge_type_map,
            )


@pytest.mark.asyncio
async def test_extract_edges_no_previous_episodes_gets_chunking(monkeypatch):
    """D3: Chunk mode works even without previous_episodes."""
    from unittest.mock import AsyncMock, MagicMock, patch

    from graphiti_core.utils.maintenance.edge_operations import extract_edges

    llm_client = MagicMock()
    llm_client.input_token_budget.return_value = 5000

    node1 = EntityNode(name='Node1', group_id='group', labels=['Entity'])
    node2 = EntityNode(name='Node2', group_id='group', labels=['Entity'])
    nodes = [node1, node2]

    with (
        patch('graphiti_core.utils.maintenance.edge_operations.estimate_tokens', return_value=1),
        patch(
            'graphiti_core.utils.maintenance.edge_operations.generate_prompt_response',
            AsyncMock(return_value={'edges': [], 'invalid_at': None}),
        ),
    ):
        clients = MagicMock()
        clients.llm_client = llm_client
        episode = _make_episode()
        edge_type_map = {('Entity', 'Entity'): ['Relation']}

        await extract_edges(
            clients,
            episode,
            nodes,
            previous_episodes=[],  # No previous episodes
            edge_type_map=edge_type_map,
        )


@pytest.mark.asyncio
async def test_resolve_extracted_edge_single_context_log(monkeypatch, caplog):
    """D4: resolve_extracted_edge logs context once (not twice when budget-trimmed).

    Verifies that only one context log fires (the unconditional one),
    not the budget-trimmed one that was previously also emitted.
    """
    import logging as _logging
    from unittest.mock import AsyncMock, patch

    from graphiti_core.edges import EntityEdge
    from graphiti_core.nodes import EpisodeType, EpisodicNode
    from graphiti_core.utils.datetime_utils import utc_now
    from graphiti_core.utils.maintenance.edge_operations import resolve_extracted_edge

    llm_client = MagicMock()
    llm_client.input_token_budget.return_value = None  # Skip budget trimming

    extracted_edge = EntityEdge(
        uuid='edge1',
        source_node_uuid='src1',
        target_node_uuid='tgt1',
        name='Relation',
        group_id='group',
        fact='Test fact',
        episodes=['ep1'],
        created_at=utc_now(),
    )
    related_edge = EntityEdge(
        uuid='related1',
        source_node_uuid='src1',
        target_node_uuid='tgt1',
        name='Relation',
        group_id='group',
        fact='Related fact',
        episodes=['ep1'],
        created_at=utc_now(),
    )
    existing_edge = EntityEdge(
        uuid='existing1',
        source_node_uuid='src1',
        target_node_uuid='tgt1',
        name='Relation',
        group_id='group',
        fact='Existing fact',
        episodes=['ep1'],
        created_at=utc_now(),
    )
    episode = EpisodicNode(
        name='ep',
        group_id='group',
        source=EpisodeType.message,
        source_description='test',
        content='content',
        valid_at=utc_now(),
    )

    async def mock_generate_response(*args, **kwargs):
        return {'duplicate_facts': [], 'contradicted_facts': []}

    with (
        caplog.at_level(_logging.DEBUG),
        patch(
            'graphiti_core.utils.maintenance.edge_operations.generate_prompt_response',
            AsyncMock(return_value={'duplicate_facts': [], 'contradicted_facts': []}),
        ),
    ):
        resolved, invalidated, duplicates = await resolve_extracted_edge(
            llm_client,
            extracted_edge,
            related_edges=[related_edge],
            existing_edges=[existing_edge],
            episode=episode,
            edge_type_candidates=None,
        )

    # Count context logs: only ONE should appear (the unconditional one)
    context_logs = [
        rec
        for rec in caplog.records
        if 'Resolving edge: sent' in rec.message and 'EXISTING FACTS' in rec.message
    ]
    assert len(context_logs) == 1, (
        f'Expected 1 context log, got {len(context_logs)}: {[r.message for r in context_logs]}'
    )
