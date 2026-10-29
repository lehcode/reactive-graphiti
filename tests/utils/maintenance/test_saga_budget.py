import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from graphiti_core.prompts.summarize_sagas import SagaSummary
from graphiti_core.utils.content_chunking import estimate_tokens
from graphiti_core.utils.maintenance.saga_budget import saga_batches, summarize_saga_by_budget
from graphiti_core.utils.text_utils import MAX_SUMMARY_CHARS

BUDGET = 1500


def _client(budget, replies=None):
    """Fake LLM client recording (rendered tokens, existing_summary prompt text) per call."""
    client = MagicMock()
    client.input_token_budget.return_value = budget
    client.calls = []
    schema = estimate_tokens(json.dumps(SagaSummary.model_json_schema()))

    async def generate(messages, **kwargs):
        rendered = sum(estimate_tokens(m.content) for m in messages) + schema
        client.calls.append((rendered, messages[1].content))
        n = len(client.calls)
        return {'summary': replies[n - 1] if replies else f'summary-{n}'}

    client.generate_response = AsyncMock(side_effect=generate)
    return client


def _episodes(n, size):
    return [f'ep{i:03d} ' + ('Fact sentence. ' * (size // 15)) for i in range(n)]


@pytest.mark.asyncio
async def test_many_large_episodes_fold_in_budgeted_chronological_batches():
    client = _client(BUDGET)
    episodes = _episodes(200, 1200)

    result = await summarize_saga_by_budget(client, 'saga', '', episodes)

    assert len(client.calls) > 1
    assert all(rendered <= BUDGET for rendered, _ in client.calls)
    assert result == f'summary-{len(client.calls)}'
    # chronological: first marker of each call increases, and every episode is seen once
    seen = [int(tag) for _, text in client.calls for tag in _markers(text)]
    assert seen == list(range(200))
    # each call after the first receives the previous output as existing_summary
    for i in range(1, len(client.calls)):
        assert f'summary-{i}\n</EXISTING_KNOWLEDGE>' in client.calls[i][1]


def _markers(text):
    import re

    return re.findall(r'^ep(\d{3}) ', text, flags=re.M)


@pytest.mark.asyncio
async def test_small_episodes_use_a_single_call():
    client = _client(BUDGET)

    result = await summarize_saga_by_budget(client, 'saga', 'old', ['a.', 'b.', 'c.'])

    assert len(client.calls) == 1
    assert result == 'summary-1'


@pytest.mark.asyncio
async def test_oversized_episode_is_truncated_at_a_sentence_and_fits():
    client = _client(BUDGET)
    huge = 'Sentence here. ' * 2000

    await summarize_saga_by_budget(client, 'saga', '', [huge])

    assert len(client.calls) == 1
    assert client.calls[0][0] <= BUDGET
    assert 'Sentence here.\n' in client.calls[0][1]


@pytest.mark.asyncio
async def test_no_budget_makes_one_call_with_all_episodes():
    client = _client(None)
    episodes = _episodes(50, 1200)

    await summarize_saga_by_budget(client, 'saga', '', episodes)

    assert len(client.calls) == 1
    assert len(_markers(client.calls[0][1])) == 50


@pytest.mark.asyncio
async def test_room_exhausted_warns_and_sends_one_call(caplog):
    client = _client(10)

    with caplog.at_level('WARNING'):
        await summarize_saga_by_budget(client, 'saga', '', _episodes(5, 1200))

    assert len(client.calls) == 1
    assert any('exceeds' in r.message for r in caplog.records)


@pytest.mark.asyncio
async def test_intermediate_summaries_are_capped():
    long = 'x' * (MAX_SUMMARY_CHARS * 3)
    client = _client(BUDGET, replies=[long] * 50)

    result = await summarize_saga_by_budget(client, 'saga', '', _episodes(30, 1200))

    assert len(result) == MAX_SUMMARY_CHARS
    assert all(rendered <= BUDGET for rendered, _ in client.calls)


@pytest.mark.asyncio
async def test_empty_reply_keeps_running_summary():
    client = _client(BUDGET, replies=['', ''])

    result = await summarize_saga_by_budget(client, 'saga', 'kept', ['a.'])

    assert result == 'kept'


def test_saga_batches_without_budget_is_one_batch():
    assert saga_batches(_client(None), 'saga', '', ['a', 'b']) == [['a', 'b']]
