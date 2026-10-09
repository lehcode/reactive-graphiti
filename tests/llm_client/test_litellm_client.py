"""Tests for LiteLLM Client.

Tests the OpenAI-compatible client used for LiteLLM, Ollama, vLLM, etc.
"""

import asyncio
import json
import logging
import re
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from graphiti_core.llm_client.config import LLMConfig
from graphiti_core.llm_client.litellm_client import LiteLLMClient
from graphiti_core.prompts.models import Message


class TestIsSchemaReturnedAsData:
    """Tests for _is_schema_returned_as_data method."""

    @pytest.fixture
    def client(self):
        """Create a client instance for testing with mocked OpenAI client."""
        with patch('graphiti_core.llm_client.litellm_client.AsyncOpenAI'):
            config = LLMConfig(api_key='test-key')
            return LiteLLMClient(config=config)

    def test_detects_schema_with_properties(self, client):
        """Schema with 'properties' key should be detected."""
        response = {
            'type': 'object',
            'properties': {'name': {'type': 'string'}},
            'required': ['name'],
        }
        assert client._is_schema_returned_as_data(response) is True

    def test_detects_schema_with_defs(self, client):
        """Schema with '$defs' key should be detected."""
        response = {'$defs': {'Person': {'type': 'object'}}}
        assert client._is_schema_returned_as_data(response) is True

    def test_detects_schema_with_schema_key(self, client):
        """Schema with '$schema' key should be detected."""
        response = {'$schema': 'http://json-schema.org/draft-07/schema#'}
        assert client._is_schema_returned_as_data(response) is True

    def test_detects_schema_with_definitions(self, client):
        """Schema with 'definitions' key should be detected."""
        response = {'definitions': {'Item': {'type': 'string'}}}
        assert client._is_schema_returned_as_data(response) is True

    def test_detects_type_object_at_top_level(self, client):
        """Top-level 'type': 'object' indicates schema."""
        response = {'type': 'object'}
        assert client._is_schema_returned_as_data(response) is True

    def test_real_data_not_detected_as_schema(self, client):
        """Normal data responses should not be detected as schema."""
        response = {'name': 'John', 'age': 30, 'city': 'NYC'}
        assert client._is_schema_returned_as_data(response) is False

    def test_nested_type_object_not_detected(self, client):
        """Nested 'type' fields in data should not trigger detection."""
        response = {'entity': {'type': 'person', 'name': 'John'}}
        assert client._is_schema_returned_as_data(response) is False

    def test_empty_response_not_detected(self, client):
        """Empty response should not be detected as schema."""
        response = {}
        assert client._is_schema_returned_as_data(response) is False


class TestExtractJson:
    """Tests for _extract_json method."""

    @pytest.fixture
    def client(self):
        """Create a client instance for testing with mocked OpenAI client."""
        with patch('graphiti_core.llm_client.litellm_client.AsyncOpenAI'):
            config = LLMConfig(api_key='test-key')
            return LiteLLMClient(config=config)

    def test_extracts_clean_json(self, client):
        """Clean JSON without trailing text should parse normally."""
        text = '{"name": "John", "age": 30}'
        result = client._extract_json(text)
        assert result == {'name': 'John', 'age': 30}

    def test_extracts_json_with_trailing_text(self, client):
        """JSON followed by text should extract only the JSON."""
        text = '{"name": "John"}\n\nThis is some explanation text.'
        result = client._extract_json(text)
        assert result == {'name': 'John'}

    def test_extracts_json_with_trailing_newlines(self, client):
        """JSON with trailing whitespace should parse correctly."""
        text = '{"status": "ok"}\n\n\n'
        result = client._extract_json(text)
        assert result == {'status': 'ok'}

    def test_handles_nested_braces(self, client):
        """Nested JSON objects should be extracted correctly."""
        text = '{"outer": {"inner": {"deep": true}}}\nExtra text here'
        result = client._extract_json(text)
        assert result == {'outer': {'inner': {'deep': True}}}

    def test_handles_strings_with_braces(self, client):
        """Braces inside strings should not affect extraction."""
        text = '{"message": "Hello {world}"}\nTrailing text'
        result = client._extract_json(text)
        assert result == {'message': 'Hello {world}'}

    def test_handles_escaped_quotes(self, client):
        """Escaped quotes in strings should be handled correctly."""
        text = '{"quote": "He said \\"hello\\""}\nMore text'
        result = client._extract_json(text)
        assert result == {'quote': 'He said "hello"'}

    def test_raises_on_no_json(self, client):
        """Text without JSON should raise JSONDecodeError."""
        text = 'This is just plain text'
        with pytest.raises(json.JSONDecodeError):
            client._extract_json(text)

    def test_raises_on_incomplete_json(self, client):
        """Incomplete JSON should raise JSONDecodeError."""
        text = '{"name": "John"'
        with pytest.raises(json.JSONDecodeError):
            client._extract_json(text)

    def test_handles_whitespace_before_json(self, client):
        """Leading whitespace should be stripped."""
        text = '  \n  {"data": true}'
        result = client._extract_json(text)
        assert result == {'data': True}

    def test_real_world_gemini_response(self, client):
        """Test with real-world Gemini response pattern."""
        text = """{"entities": [{"name": "User", "type": "Person"}]}

I've extracted the entities from your text. The main entity identified is "User" which appears to be a Person type."""
        result = client._extract_json(text)
        assert result == {'entities': [{'name': 'User', 'type': 'Person'}]}


class TestLiteLLMClientTokenLogging:
    """Tests for prompt size logging and token usage tracking in LiteLLMClient."""

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
    async def test_success_with_usage(self, client, caplog):
        """Mock completion with usage → verify token_tracker records 123/45, INFO log contains values."""
        mock_usage = MagicMock()
        mock_usage.prompt_tokens = 123
        mock_usage.completion_tokens = 45

        mock_choice = MagicMock()
        mock_choice.message.content = json.dumps({'test': 'data'})

        mock_response = MagicMock()
        mock_response.usage = mock_usage
        mock_response.choices = [mock_choice]

        client.client.chat.completions.create.return_value = mock_response

        with caplog.at_level(logging.INFO):
            result = await client.generate_response(
                [Message(role='user', content='Hello')],
                prompt_name='test_prompt',
            )

        assert result == {'test': 'data'}
        usage = client.token_tracker.get_usage()
        assert usage['test_prompt'].total_input_tokens == 123
        assert usage['test_prompt'].total_output_tokens == 45
        assert 'LLM prompt size: prompt=test_prompt' in caplog.text
        assert 'est_total=' in caplog.text
        assert 'prompt_tokens=123' in caplog.text
        assert 'completion_tokens=45' in caplog.text

    @pytest.mark.asyncio
    async def test_usage_none(self, client, caplog):
        """response.usage is None → no crash, 0/0 recorded."""
        mock_choice = MagicMock()
        mock_choice.message.content = json.dumps({'test': 'data'})

        mock_response = MagicMock()
        mock_response.usage = None
        mock_response.choices = [mock_choice]

        client.client.chat.completions.create.return_value = mock_response

        with caplog.at_level(logging.INFO):
            result = await client.generate_response(
                [Message(role='user', content='Hello')],
                prompt_name='none_usage_prompt',
            )

        assert result == {'test': 'data'}
        usage = client.token_tracker.get_usage()
        assert usage['none_usage_prompt'].total_input_tokens == 0
        assert usage['none_usage_prompt'].total_output_tokens == 0

    @pytest.mark.asyncio
    async def test_json_schema_vs_json_object_estimates(self, client, caplog):
        """json_schema → est_schema > 0; json_object → est_schema = 0."""
        from pydantic import BaseModel

        class MyResponse(BaseModel):
            name: str

        mock_choice = MagicMock()
        mock_choice.message.content = json.dumps({'name': 'test'})

        mock_response = MagicMock()
        mock_response.usage = MagicMock(prompt_tokens=50, completion_tokens=20)
        mock_response.choices = [mock_choice]

        # Test json_schema mode (default) → est_schema > 0
        client.client.chat.completions.create.return_value = mock_response
        with caplog.at_level(logging.INFO):
            await client.generate_response(
                [Message(role='system', content='System'), Message(role='user', content='User')],
                response_model=MyResponse,
                prompt_name='json_schema_test',
            )

        assert 'LLM prompt size: prompt=json_schema_test' in caplog.text
        match = re.search(r'est_schema=(\d+)', caplog.text)
        assert match is not None
        assert int(match.group(1)) > 0

        # Test json_object mode → est_schema = 0
        config = LLMConfig(api_key='test-key')
        mock_client2 = MagicMock()
        mock_client2.chat = MagicMock()
        mock_client2.chat.completions = MagicMock()
        mock_client2.chat.completions.create = AsyncMock()
        client2 = LiteLLMClient(
            config=config, client=mock_client2, structured_output_mode='json_object'
        )
        client2.client.chat.completions.create.return_value = mock_response
        with caplog.at_level(logging.INFO):
            await client2.generate_response(
                [Message(role='system', content='System'), Message(role='user', content='User')],
                response_model=MyResponse,
                prompt_name='json_object_test',
            )

        assert 'LLM prompt size: prompt=json_object_test' in caplog.text
        # Parse the json_object_test log line
        lines = [line for line in caplog.text.split('\n') if 'json_object_test' in line]
        assert len(lines) == 1
        match = re.search(r'est_schema=(\d+)', lines[0])
        assert match is not None
        assert int(match.group(1)) == 0

    @pytest.mark.asyncio
    async def test_failure_logs_warning(self, client, caplog):
        """Completion raises → WARNING log contains est_total and prompt name."""
        client.client.chat.completions.create.side_effect = RuntimeError('API error')
        with caplog.at_level(logging.WARNING), pytest.raises(RuntimeError):
            await client.generate_response(
                [Message(role='user', content='Hello')],
                prompt_name='fail_prompt',
            )

        assert 'LLM call failed: prompt=fail_prompt' in caplog.text
        assert 'est_total=' in caplog.text

    @pytest.mark.asyncio
    async def test_concurrency_isolation(self, client, caplog):
        """Two asyncio.gather calls → usage recorded under correct names."""
        mock_choice = MagicMock()
        mock_choice.message.content = json.dumps({'ok': True})

        mock_response = MagicMock()
        mock_response.usage = MagicMock(prompt_tokens=100, completion_tokens=50)
        mock_response.choices = [mock_choice]

        async def call(prompt_name):
            # Each call resets the mock to return the same mock_response
            client.client.chat.completions.create.return_value = mock_response
            return await client.generate_response(
                [Message(role='user', content='Hello')],
                prompt_name=prompt_name,
            )

        await asyncio.gather(call('prompt_a'), call('prompt_b'))

        usage = client.token_tracker.get_usage()
        assert 'prompt_a' in usage
        assert 'prompt_b' in usage
        assert usage['prompt_a'].total_input_tokens == 100
        assert usage['prompt_b'].total_input_tokens == 100
