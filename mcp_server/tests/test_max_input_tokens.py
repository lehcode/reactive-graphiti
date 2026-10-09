"""Tests for resolve_max_input_tokens helper in factories.py."""

import json
import logging
from unittest.mock import MagicMock, patch

import httpx
import pytest

# Import directly from the module
try:
    from src.services.factories import (
        _core_llm_config_accepts_max_input_tokens,
        resolve_max_input_tokens,
    )
except ImportError:
    # When run as package: from mcp_server.src.services.factories
    import sys

    sys.path.insert(0, 'src')
    from services.factories import (
        _core_llm_config_accepts_max_input_tokens,
        resolve_max_input_tokens,
    )


def make_response(status_code: int, json_data: list | None = None) -> httpx.Response:
    """Create an httpx.Response shaped like LiteLLM's /model/info ({"data": [...]})."""
    body = json.dumps({'data': json_data}).encode() if json_data is not None else b'{}'
    return httpx.Response(status_code, content=body)


@pytest.mark.parametrize(
    'api_url,model,configured,expected',
    [
        ('http://litellm:4000/v1', 'gpt-4', None, 8000),
        ('http://litellm:4000/v1/', 'gpt-4', None, 8000),
        ('http://litellm:4000/', 'gpt-4', None, 8000),
        ('http://litellm:4000', 'gpt-4', None, 8000),
        ('http://litellm:4000/v1', 'gpt-4', 4096, 8000),
        ('http://litellm:4000/v1', 'gpt-4.1-mini', None, 8192),
        ('http://litellm:4000/v1', 'gpt-3.5-turbo', None, 4096),
        ('http://litellm:4000/v1', 'unknown-model', None, None),
        ('http://litellm:4000/v1', 'unknown-model', 8192, 8192),
    ],
)
def test_resolve_from_model_info_success(api_url, model, configured, expected):
    """Successful /model/info lookup → correct value."""
    mock_data = [
        {'model_name': 'gpt-4', 'model_info': {'max_input_tokens': 8000}},
        {'model_name': 'gpt-4.1-mini', 'model_info': {'max_input_tokens': 8192}},
        {'model_name': 'gpt-3.5-turbo', 'model_info': {'max_input_tokens': 4096}},
    ]
    expected_response = make_response(200, mock_data)

    with patch('httpx.Client') as mock_client_cls:
        mock_client = MagicMock()
        mock_client.get.return_value = expected_response
        mock_client_cls.return_value = mock_client

        result = resolve_max_input_tokens(
            api_url,
            'sk-fake-key',
            model,
            configured,
        )
        assert result == expected
        mock_client.get.assert_called()


def test_v1_stripping():
    """Trailing /v1 is stripped before appending /model/info."""
    mock_data = [{'model_name': 'gpt-4', 'model_info': {'max_input_tokens': 8000}}]
    expected_response = make_response(200, mock_data)

    with patch('httpx.Client') as mock_client_cls:
        mock_client = MagicMock()
        mock_client.get.return_value = expected_response
        mock_client_cls.return_value = mock_client

        resolve_max_input_tokens(
            'http://litellm:4000/v1/',
            'sk-fake-key',
            'gpt-4',
            None,
        )
        # Should be called with base (no /v1) + /model/info
        called_url = mock_client.get.call_args[0][0]
        assert called_url.endswith('/model/info')
        assert '/v1' not in called_url.replace('/model/info', '')


def test_model_missing_fallback_configured():
    """Model not found → configured fallback."""
    mock_data = [{'model_name': 'other-model', 'model_info': {'max_input_tokens': 8000}}]
    expected_response = make_response(200, mock_data)

    with patch('httpx.Client') as mock_client_cls:
        mock_client = MagicMock()
        mock_client.get.return_value = expected_response
        mock_client_cls.return_value = mock_client

        result = resolve_max_input_tokens(
            'http://litellm:4000/v1',
            'sk-fake-key',
            'gpt-4',
            4096,
        )
        assert result == 4096


def test_http_error_fallback():
    """Non-200 response → configured fallback."""
    expected_response = make_response(500)

    with patch('httpx.Client') as mock_client_cls:
        mock_client = MagicMock()
        mock_client.get.return_value = expected_response
        mock_client_cls.return_value = mock_client

        result = resolve_max_input_tokens(
            'http://litellm:4000/v1',
            'sk-fake-key',
            'gpt-4',
            8192,
        )
        assert result == 8192


def test_none_when_nothing_configured():
    """No LiteLLM value and no configured → None."""
    mock_data = [{'model_name': 'other-model', 'model_info': {'max_input_tokens': 8000}}]
    expected_response = make_response(200, mock_data)

    with patch('httpx.Client') as mock_client_cls:
        mock_client = MagicMock()
        mock_client.get.return_value = expected_response
        mock_client_cls.return_value = mock_client

        result = resolve_max_input_tokens(
            'http://litellm:4000/v1',
            'sk-fake-key',
            'gpt-4',
            None,
        )
        assert result is None


def test_mismatch_warning(caplog):
    """LiteLLM value differs from configured → WARNING logged."""
    mock_data = [{'model_name': 'gpt-4', 'model_info': {'max_input_tokens': 8000}}]
    expected_response = make_response(200, mock_data)

    with patch('httpx.Client') as mock_client_cls, caplog.at_level(logging.WARNING):
        mock_client = MagicMock()
        mock_client.get.return_value = expected_response
        mock_client_cls.return_value = mock_client

        resolve_max_input_tokens(
            'http://litellm:4000/v1',
            'sk-fake-key',
            'gpt-4',
            4096,
        )
        assert 'mismatch' in caplog.text.lower()
        assert 'lite_llm=8000' in caplog.text or 'LiteLLM=8000' in caplog.text


def test_api_key_not_in_logs(caplog):
    """API key should never appear in log output."""
    mock_data = [{'model_name': 'other-model', 'model_info': {'max_input_tokens': 8000}}]
    expected_response = make_response(200, mock_data)

    with patch('httpx.Client') as mock_client_cls, caplog.at_level(logging.WARNING):
        mock_client = MagicMock()
        mock_client.get.return_value = expected_response
        mock_client_cls.return_value = mock_client

        resolve_max_input_tokens(
            'http://litellm:4000/v1',
            'sk-secret-12345',
            'gpt-4',
            None,
        )
        assert 'sk-secret-12345' not in caplog.text
        assert 'sk-secret' not in caplog.text


def test_top_level_limit_is_not_read():
    """max_input_tokens outside model_info (not LiteLLM's shape) is ignored."""
    body = json.dumps({'data': [{'model_name': 'gpt-4', 'max_input_tokens': 8000}]}).encode()
    client = MagicMock()
    client.get.return_value = httpx.Response(200, content=body)
    assert resolve_max_input_tokens('http://proxy:4000/v1', 'k', 'gpt-4', 2048, client=client) == (
        2048
    )


def test_core_llm_config_support_check():
    """Fork LLMConfig accepts max_input_tokens; upstream-style LLMConfig does not."""

    class ForkConfig:
        def __init__(self, api_key=None, max_input_tokens=None):
            pass

    class UpstreamConfig:
        def __init__(self, api_key=None, max_tokens=None):
            pass

    assert _core_llm_config_accepts_max_input_tokens(ForkConfig) is True
    assert _core_llm_config_accepts_max_input_tokens(UpstreamConfig) is False


def test_empty_api_url_returns_configured_without_request():
    client = MagicMock()
    assert resolve_max_input_tokens(None, 'k', 'gpt-4', 4096, client=client) == 4096
    assert resolve_max_input_tokens('', 'k', 'gpt-4', None, client=client) is None
    client.get.assert_not_called()


def test_official_openai_host_skips_lookup():
    """api.openai.com has no /model/info: no credentialed request, configured value used."""
    client = MagicMock()
    result = resolve_max_input_tokens(
        'https://api.openai.com/v1', 'secret', 'gpt-4', 8192, client=client
    )
    assert result == 8192
    client.get.assert_not_called()


def _factory_module():
    import sys

    return sys.modules[resolve_max_input_tokens.__module__]


@pytest.mark.parametrize('supports_kwarg', [True, False])
def test_factory_forwards_resolved_limit_only_when_supported(monkeypatch, supports_kwarg):
    """LLMClientFactory passes the resolved limit into the core LLMConfig only if it accepts it."""
    import graphiti_core.llm_client.config as core_config

    from config.schema import LLMConfig as McpLLMConfig
    from config.schema import LLMProvidersConfig, OpenAIProviderConfig

    captured: dict = {}

    if supports_kwarg:

        class FakeCoreConfig:
            def __init__(self, max_input_tokens=None, **kwargs):
                captured.update(kwargs, max_input_tokens=max_input_tokens)

    else:

        class FakeCoreConfig:
            def __init__(self, **kwargs):
                captured.update(kwargs)

    factories = _factory_module()
    monkeypatch.setattr(core_config, 'LLMConfig', FakeCoreConfig)
    monkeypatch.setattr(factories, 'resolve_max_input_tokens', lambda *a, **k: 14336)
    monkeypatch.setattr(factories, 'LiteLLMClient', MagicMock())

    config = McpLLMConfig(
        provider='openai_generic',
        model='tier1-graphiti-kg-local',
        providers=LLMProvidersConfig(
            openai=OpenAIProviderConfig(api_key='k', api_url='http://litellm:4000/v1')
        ),
    )
    factories.LLMClientFactory.create(config)

    if supports_kwarg:
        assert captured['max_input_tokens'] == 14336
    else:
        assert 'max_input_tokens' not in captured
