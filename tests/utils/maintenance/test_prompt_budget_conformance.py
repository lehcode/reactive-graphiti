"""Cross-call-site conformance of the input-token budget.

Every budgeted call site runs its real prompt builders against a fake LLM client and a small
``max_input_tokens`` with oversized inputs. The asserted quantity is the one the client guard
checks: the estimate of the rendered messages plus the response schema, relative to the budget
(never absolute token counts or prompt wording).

Prompts are classified three ways and ``test_every_prompt_name_is_classified`` fails for a new
prompt name until it is placed:

* ``BUDGETED_CASES``: rendered estimate must stay ``<= BUDGET``.
* ``KNOWN_UNBUDGETED``: the call site has no budgeting. The oversized input must still exceed
  ``BUDGET``; the test fails once such a prompt becomes budgeted, so the entry is moved out.
* ``NO_CALL_SITE``: no production call site (checked by scanning the sources).
"""

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import get_args
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import BaseModel

import graphiti_core
from graphiti_core.edges import EntityEdge
from graphiti_core.nodes import EntityNode, EpisodeType, EpisodicNode
from graphiti_core.prompts.names import PromptName
from graphiti_core.utils.content_chunking import estimate_tokens
from graphiti_core.utils.maintenance.community_operations import (
    generate_summary_description,
    summarize_pair,
)
from graphiti_core.utils.maintenance.dedup_helpers import DedupResolutionState
from graphiti_core.utils.maintenance.edge_operations import (
    _extract_edge_timestamps,
    extract_edges,
    resolve_extracted_edge,
)
from graphiti_core.utils.maintenance.node_operations import (
    _build_candidate_indexes,
    _extract_entity_attributes,
    _extract_entity_summaries_batch,
    _resolve_with_llm,
    extract_nodes,
)
from graphiti_core.utils.maintenance.saga_budget import summarize_saga_by_budget

BUDGET = 6000
# Chars of text that cost roughly 20x the budget in tokens
HUGE = BUDGET * 20 * 4
NOW = datetime(2026, 1, 1, tzinfo=timezone.utc)

# Canned, schema-valid replies keyed by prompt name; anything else gets an empty object.
REPLIES: dict[str, dict] = {
    'extract_nodes.extract_message': {'extracted_entities': []},
    'extract_nodes.extract_json': {'extracted_entities': []},
    'extract_nodes.extract_text': {'extracted_entities': []},
    'extract_nodes.extract_summaries_batch': {'summaries': []},
    'extract_nodes.extract_entity_summaries_from_episodes': {'summaries': []},
    'dedupe_nodes.nodes': {'entity_resolutions': []},
    'extract_edges.edge': {'edges': []},
    'dedupe_edges.resolve_edge': {'duplicate_facts': [], 'contradicted_facts': []},
    'summarize_sagas.summarize_saga': {'summary': 'Summary.'},
    'summarize_nodes.summarize_pair': {'summary': 'Summary.'},
    'summarize_nodes.summary_description': {'description': 'Description.'},
}


class Calls(list):
    """``(prompt_name, rendered_estimate)`` per captured chat call."""

    def of(self, name: str) -> list[int]:
        return [rendered for n, rendered in self if n == name]


def _fake_client() -> MagicMock:
    client = MagicMock()
    client.input_token_budget.return_value = BUDGET
    client.calls = Calls()

    async def capture(messages, response_model=None, prompt_name=None, **_kwargs):
        rendered = sum(estimate_tokens(m.content) for m in messages)
        if response_model is not None:
            rendered += estimate_tokens(json.dumps(response_model.model_json_schema()))
        client.calls.append((prompt_name, rendered))
        return REPLIES.get(prompt_name, {})

    client.generate_response = AsyncMock(side_effect=capture)
    return client


def _episode(content: str = 'Alice met Bob. ' * 100, source=EpisodeType.text) -> EpisodicNode:
    # The episode body stays within the content share, as the ingestion pipeline guarantees
    return EpisodicNode(
        name='episode',
        group_id='group',
        source=source,
        source_description='test',
        content=content,
        valid_at=NOW,
    )


def _previous_episodes() -> list[EpisodicNode]:
    return [_episode(f'Previous {i}. ' + 'Old news. ' * 800) for i in range(HUGE // 8000)]


def _nodes(count: int, **kwargs) -> list[EntityNode]:
    return [
        EntityNode(name=f'Entity {i}', group_id='group', labels=['Entity'], **kwargs)
        for i in range(count)
    ]


def _edge(fact: str) -> EntityEdge:
    return EntityEdge(
        source_node_uuid='source',
        target_node_uuid='target',
        name='KNOWS',
        group_id='group',
        fact=fact,
        episodes=[],
        created_at=NOW,
    )


class _Attributes(BaseModel):
    """Custom attribute schema."""

    role: str | None = None


# --- budgeted call sites ---------------------------------------------------------------------


async def _extract_nodes(client, source):
    clients = MagicMock()
    clients.llm_client = client
    await extract_nodes(clients, _episode(source=source), _previous_episodes())


async def _dedupe_nodes(client):
    extracted = _nodes(300)
    state = DedupResolutionState(
        resolved_nodes=[None] * len(extracted),
        uuid_map={},
        unresolved_indices=list(range(len(extracted))),
    )
    indexes = _build_candidate_indexes(
        [EntityNode(name=f'Existing {i}', group_id='group', labels=['Entity']) for i in range(20)]
    )
    await _resolve_with_llm(
        client, extracted, indexes, state, _episode(), _previous_episodes(), None
    )


async def _extract_edges(client):
    clients = MagicMock()
    clients.llm_client = client
    await extract_edges(
        clients,
        _episode(),
        _nodes(60),
        previous_episodes=_previous_episodes(),
        edge_type_map={('Entity', 'Entity'): ['KNOWS']},
    )


async def _resolve_edge(client):
    related = [_edge(f'Related fact {i}. ' + 'detail ' * 40) for i in range(150)]
    existing = [_edge(f'Existing fact {i}. ' + 'detail ' * 40) for i in range(150)]
    await resolve_extracted_edge(client, _edge('A new fact.'), related, existing, _episode())


async def _summaries(client, use_episode_prompt):
    await _extract_entity_summaries_batch(
        client,
        _nodes(150),
        _episode(),
        _previous_episodes(),
        None,
        {},
        skip_fact_appending=use_episode_prompt,
    )


async def _node_attributes(client):
    node = EntityNode(name='Alice', group_id='group', labels=['Entity'])
    await _extract_entity_attributes(client, node, _episode(), _previous_episodes(), _Attributes)


async def _saga(client):
    episodes = [f'Episode {i}. ' + 'Fact sentence. ' * 80 for i in range(200)]
    await summarize_saga_by_budget(client, 'saga', '', episodes)


BUDGETED_CASES = {
    'extract_nodes.extract_message': lambda c: _extract_nodes(c, EpisodeType.message),
    'extract_nodes.extract_json': lambda c: _extract_nodes(c, EpisodeType.json),
    'extract_nodes.extract_text': lambda c: _extract_nodes(c, EpisodeType.text),
    'dedupe_nodes.nodes': _dedupe_nodes,
    'extract_edges.edge': _extract_edges,
    'dedupe_edges.resolve_edge': _resolve_edge,
    'extract_nodes.extract_summaries_batch': lambda c: _summaries(c, False),
    'extract_nodes.extract_entity_summaries_from_episodes': lambda c: _summaries(c, True),
    'extract_nodes.extract_attributes': _node_attributes,
    'summarize_sagas.summarize_saga': _saga,
}


# Budgeted sites that still overrun: strict xfail, so a fix turns the test red until removed here
KNOWN_OVERRUN: dict[str, str] = {}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    'name',
    [
        pytest.param(n, marks=pytest.mark.xfail(strict=True, reason=KNOWN_OVERRUN[n]))
        if n in KNOWN_OVERRUN
        else n
        for n in BUDGETED_CASES
    ],
)
async def test_budgeted_call_site_stays_within_budget(name):
    client = _fake_client()

    await BUDGETED_CASES[name](client)

    rendered = client.calls.of(name)
    assert rendered, f'{name} was never called'
    assert max(rendered) <= BUDGET


# --- known gaps: call sites that send their input unbudgeted -----------------------------------


async def _edge_timestamps(client):
    await _extract_edge_timestamps(client, _edge('fact ' * HUGE), _episode())


async def _edge_attributes(client):
    await resolve_extracted_edge(
        client,
        _edge('fact ' * HUGE),
        [],
        [],
        _episode(),
        edge_type_candidates={'KNOWS': _Attributes},
    )


async def _community_pair(client):
    await summarize_pair(client, ('summary ' * HUGE, 'summary ' * HUGE))


async def _community_description(client):
    await generate_summary_description(client, 'summary ' * HUGE)


# prompt name -> (driver with an oversized input, reason)
KNOWN_UNBUDGETED = {
    'extract_edges.extract_timestamps': (
        _edge_timestamps,
        'edge_operations._extract_edge_timestamps sends the raw edge fact',
    ),
    'extract_edges.extract_attributes': (
        _edge_attributes,
        'edge_operations.resolve_extracted_edge sends the raw fact and existing attributes',
    ),
    'summarize_nodes.summarize_pair': (
        _community_pair,
        'community_operations.summarize_pair; inputs are node summaries capped at MAX_SUMMARY_CHARS',
    ),
    'summarize_nodes.summary_description': (
        _community_description,
        'community_operations.generate_summary_description; input is a capped community summary',
    ),
}


@pytest.mark.asyncio
@pytest.mark.parametrize('name', KNOWN_UNBUDGETED)
async def test_known_unbudgeted_prompt_still_exceeds_budget(name):
    client = _fake_client()

    await KNOWN_UNBUDGETED[name][0](client)

    rendered = client.calls.of(name)
    assert rendered, f'{name} was never called'
    assert max(rendered) > BUDGET, f'{name} is now budgeted: move it to BUDGETED_CASES'


# --- prompts without a production call site ---------------------------------------------------

NO_CALL_SITE = {
    'extract_nodes.classify_nodes': 'defined in the prompt library, never sent',
    'extract_nodes.extract_summary': 'defined in the prompt library, never sent',
    'dedupe_nodes.node': 'defined in the prompt library, never sent',
    'dedupe_nodes.node_list': 'defined in the prompt library, never sent',
    'summarize_nodes.summarize_context': 'defined in the prompt library, never sent',
    # combined_extraction.py is reachable only through the bulk_utils flag, default off
    'extract_nodes_and_edges.extract_message': 'unwired use_combined_extraction path',
    'extract_edges.extract_timestamps_batch': 'unwired use_combined_extraction path',
    'eval.query_expansion': 'evaluation prompt, not part of ingestion',
    'eval.qa_prompt': 'evaluation prompt, not part of ingestion',
    'eval.eval_prompt': 'evaluation prompt, not part of ingestion',
    'eval.eval_add_episode_results': 'evaluation prompt, not part of ingestion',
}


def test_every_prompt_name_is_classified():
    classified = [*BUDGETED_CASES, *KNOWN_UNBUDGETED, *NO_CALL_SITE]

    assert len(classified) == len(set(classified)), 'a prompt name is classified twice'
    assert set(get_args(PromptName)) == set(classified)


@pytest.mark.parametrize('name', NO_CALL_SITE)
def test_prompt_without_call_site_is_not_sent(name):
    root = Path(graphiti_core.__file__).parent
    skipped = {'names.py', 'lib.py', 'combined_extraction.py'}
    pattern = re.compile(rf"""['"]{re.escape(name)}['"]""")
    users = [
        str(p.relative_to(root))
        for p in root.rglob('*.py')
        if p.name not in skipped and pattern.search(p.read_text())
    ]

    assert not users, f'{name} now has a call site in {users}: classify it'
