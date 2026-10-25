"""Tests for input-token budget integration (LLMConfig, LiteLLMClient)."""

import importlib
import json
import logging
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from graphiti_core import helpers
from graphiti_core.errors import GraphitiError, PromptBudgetExceededError
from graphiti_core.llm_client.config import LLMConfig
from graphiti_core.llm_client.litellm_client import LiteLLMClient
from graphiti_core.prompts.models import Message


@pytest.fixture(autouse=True)
def _pin_safety_ratio(monkeypatch):
    """Budget assertions assume the default ratio; ignore any INPUT_TOKEN_SAFETY_RATIO env."""
    monkeypatch.setattr('graphiti_core.helpers.INPUT_TOKEN_SAFETY_RATIO', 0.8)


class TestLLMConfigMaxInputTokens:
    """Tests for LLMConfig.max_input_tokens parameter."""

    def test_default_none(self):
        """max_input_tokens defaults to None (no budgeting)."""
        config = LLMConfig()
        assert config.max_input_tokens is None

    def test_explicit_value(self):
        """Explicit max_input_tokens is stored."""
        config = LLMConfig(api_key='key', model='test', max_input_tokens=8000)
        assert config.max_input_tokens == 8000


class TestInputTokenBudget:
    """Tests for LiteLLMClient.input_token_budget()."""

    @pytest.fixture
    def client(self):
        """Create a LiteLLMClient with mocked OpenAI client."""
        mock_client = MagicMock()
        mock_client.chat = MagicMock()
        mock_client.chat.completions = MagicMock()
        mock_client.chat.completions.create = AsyncMock()
        return LiteLLMClient(config=LLMConfig(), client=mock_client)

    def test_none_budget_when_max_input_none(self, client):
        """LLMConfig with max_input_tokens=None → None budget."""
        assert client.input_token_budget() is None

    def test_budget_applied_with_ratio(self):
        """max_input_tokens → int(max_input_tokens * safety_ratio)."""
        mock_client = MagicMock()
        mock_client.chat = MagicMock()
        mock_client.chat.completions = MagicMock()
        mock_client.chat.completions.create = AsyncMock()
        config = LLMConfig(api_key='test-key', max_input_tokens=10000)
        client = LiteLLMClient(config=config, client=mock_client)
        budget = client.input_token_budget()
        # Default ratio is 0.8, so 10000 * 0.8 = 8000
        assert budget == 8000

    def test_budget_int_cast(self):
        """Budget is int (truncated, not rounded)."""
        mock_client = MagicMock()
        mock_client.chat = MagicMock()
        mock_client.chat.completions = MagicMock()
        mock_client.chat.completions.create = AsyncMock()
        config = LLMConfig(api_key='test-key', max_input_tokens=10001)
        client = LiteLLMClient(config=config, client=mock_client)
        assert client.input_token_budget() == 8000  # int(10001 * 0.8)


class TestLiteLLMOverBudget:
    """Tests for LiteLLMClient over-budget WARNING."""

    @pytest.fixture
    def client(self):
        """Create a client with config that has max_input_tokens set."""
        mock_client = MagicMock()
        mock_client.chat = MagicMock()
        mock_client.chat.completions = MagicMock()
        mock_client.chat.completions.create = AsyncMock()
        config = LLMConfig(api_key='test-key', max_input_tokens=1000)
        return LiteLLMClient(config=config, client=mock_client)

    @pytest.mark.asyncio
    async def test_over_budget_warning(self, client, caplog):
        """Prompt over budget → WARNING logged, request still sent."""
        mock_choice = MagicMock()
        mock_choice.message.content = json.dumps({'ok': True})
        mock_response = MagicMock()
        mock_response.usage = MagicMock(prompt_tokens=50, completion_tokens=20)
        mock_response.choices = [mock_choice]
        client.client.chat.completions.create.return_value = mock_response

        with (
            caplog.at_level(logging.DEBUG),
            patch.object(client, '_apply_attribute_extraction_preamble'),
        ):
            # Content long enough to exceed budget (800 tokens = 3200 chars)
            result = await client.generate_response(
                [Message(role='system', content='x' * 3500), Message(role='user', content='Hello')],
                prompt_name='over_budget_test',
            )

        assert result == {'ok': True}
        assert 'LLM prompt over input budget' in caplog.text
        assert 'over_budget_test' in caplog.text

    @pytest.mark.asyncio
    async def test_under_budget_no_warning(self, client, caplog):
        """Prompt under budget → no WARNING, request still sent."""
        mock_choice = MagicMock()
        mock_choice.message.content = json.dumps({'ok': True})
        mock_response = MagicMock()
        mock_response.usage = MagicMock(prompt_tokens=10, completion_tokens=5)
        mock_response.choices = [mock_choice]
        client.client.chat.completions.create.return_value = mock_response

        with (
            caplog.at_level(logging.DEBUG),
            patch.object(client, '_apply_attribute_extraction_preamble'),
        ):
            result = await client.generate_response(
                [Message(role='system', content='Hi'), Message(role='user', content='Hello')],
                prompt_name='under_budget_test',
            )

        assert result == {'ok': True}
        assert 'LLM prompt over input budget' not in caplog.text


class TestParseFailureStillRecordsUsage:
    """Tests for token usage recording even when JSON parse fails."""

    @pytest.fixture
    def client(self):
        """Create a client with async-mock OpenAI client."""
        mock_client = MagicMock()
        mock_client.chat = MagicMock()
        mock_client.chat.completions = MagicMock()
        mock_client.chat.completions.create = AsyncMock()
        config = LLMConfig(api_key='test-key')
        return LiteLLMClient(config=config, client=mock_client)

    @pytest.mark.asyncio
    async def test_usage_recorded_on_parse_failure(self, client, caplog, monkeypatch):
        """JSON parse failure still records token usage on first attempt."""
        # Skip tenacity's real backoff between the JSONDecodeError retries
        from tenacity import wait_none

        from graphiti_core.llm_client.client import LLMClient

        monkeypatch.setattr(LLMClient._generate_response_with_retry.retry, 'wait', wait_none())
        # Return malformed JSON that will cause _extract_json to fail
        mock_choice = MagicMock()
        mock_choice.message.content = 'Not valid JSON at all'

        mock_response = MagicMock()
        mock_response.usage = MagicMock(prompt_tokens=123, completion_tokens=45)
        mock_response.choices = [mock_choice]
        client.client.chat.completions.create.return_value = mock_response

        # Clear any prior usage
        client.token_tracker._usage = {}

        with (
            caplog.at_level(logging.INFO),
            pytest.raises(json.JSONDecodeError),
            patch.object(client, '_apply_attribute_extraction_preamble'),
        ):
            await client.generate_response(
                [Message(role='user', content='Hello')],
                prompt_name='parse_fail_prompt',
            )

        # Token usage should be recorded (may be on multiple retries due to retry
        # decorator; check that it was recorded at least once)
        usage = client.token_tracker.get_usage()
        assert 'parse_fail_prompt' in usage


class TestInputBudgetModeSetting:
    """LLM_INPUT_BUDGET_MODE parsing in helpers."""

    @pytest.fixture(autouse=True)
    def _restore_helpers(self, monkeypatch):
        yield
        monkeypatch.delenv('LLM_INPUT_BUDGET_MODE', raising=False)
        importlib.reload(helpers)

    @pytest.mark.parametrize('raw,expected', [(None, 'warn'), (' RAISE ', 'raise')])
    def test_valid_values(self, monkeypatch, raw, expected):
        if raw is None:
            monkeypatch.delenv('LLM_INPUT_BUDGET_MODE', raising=False)
        else:
            monkeypatch.setenv('LLM_INPUT_BUDGET_MODE', raw)
        importlib.reload(helpers)
        assert expected == helpers.LLM_INPUT_BUDGET_MODE

    def test_invalid_value_warns_and_defaults(self, monkeypatch, caplog):
        monkeypatch.setenv('LLM_INPUT_BUDGET_MODE', 'bogus')
        with caplog.at_level(logging.WARNING, logger='graphiti_core'):
            importlib.reload(helpers)
        assert helpers.LLM_INPUT_BUDGET_MODE == 'warn'
        assert 'Invalid LLM_INPUT_BUDGET_MODE=bogus' in caplog.text


def test_prompt_budget_exceeded_error_carries_fields():
    estimate = {'system': 1, 'user': 2, 'schema': 3, 'total': 6, 'rendered': 6}
    err = PromptBudgetExceededError('p', estimate, 5, 10)
    assert isinstance(err, GraphitiError)
    assert (err.prompt_name, err.estimate, err.budget, err.max_input_tokens) == (
        'p',
        estimate,
        5,
        10,
    )
    assert 'p' in err.message and '5' in err.message
