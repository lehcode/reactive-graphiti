#!/usr/bin/env python3
"""Unit tests for CrossEncoderFactory reranker selection."""

import builtins
import logging
import sys
from pathlib import Path
from unittest.mock import AsyncMock, Mock

import pytest

# Add the src directory to the path (mirrors the other factory tests)
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from graphiti_core.cross_encoder.gemini_reranker_client import GeminiRerankerClient
from graphiti_core.cross_encoder.openai_reranker_client import OpenAIRerankerClient

import graphiti_mcp_server
from config.schema import (
    AnthropicProviderConfig,
    CrossEncoderConfig,
    DatabaseConfig,
    EmbedderConfig,
    EmbedderProvidersConfig,
    GeminiProviderConfig,
    GraphitiConfig,
    LLMConfig,
    LLMProvidersConfig,
    OpenAIProviderConfig,
    VoyageProviderConfig,
)
from services.factories import CrossEncoderFactory


class TestCrossEncoderFactory:
    """The reranker is inferred from the providers, so a non-OpenAI setup does not need OPENAI_API_KEY."""

    def test_openai_llm_uses_openai_reranker(self):
        llm = LLMConfig(
            provider='openai',
            providers=LLMProvidersConfig(openai=OpenAIProviderConfig(api_key='test-key')),
        )
        embedder = EmbedderConfig(
            provider='openai',
            providers=EmbedderProvidersConfig(openai=OpenAIProviderConfig(api_key='test-key')),
        )
        assert isinstance(CrossEncoderFactory.create(llm, embedder), OpenAIRerankerClient)

    def test_anthropic_llm_falls_back_to_gemini_embedder(self):
        # Anthropic has no native reranker, so the factory should pick up the Gemini embedder's
        # key instead of defaulting to OpenAIRerankerClient (which would need OPENAI_API_KEY).
        llm = LLMConfig(
            provider='anthropic',
            providers=LLMProvidersConfig(anthropic=AnthropicProviderConfig(api_key='test-key')),
        )
        embedder = EmbedderConfig(
            provider='gemini',
            providers=EmbedderProvidersConfig(gemini=GeminiProviderConfig(api_key='test-key')),
        )
        assert isinstance(CrossEncoderFactory.create(llm, embedder), GeminiRerankerClient)

    def test_missing_local_reranker_dependency_is_actionable(self, monkeypatch, caplog):
        llm = LLMConfig(
            provider='anthropic',
            providers=LLMProvidersConfig(anthropic=AnthropicProviderConfig(api_key='test-key')),
        )
        embedder = EmbedderConfig(
            provider='voyage',
            providers=EmbedderProvidersConfig(voyage=VoyageProviderConfig(api_key='test-key')),
        )
        real_import = builtins.__import__

        def import_without_bge(name, *args, **kwargs):
            if name == 'graphiti_core.cross_encoder.bge_reranker_client':
                raise ImportError('sentence-transformers is not installed')
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, '__import__', import_without_bge)
        caplog.set_level(logging.INFO)

        with pytest.raises(ValueError, match="MCP server's 'providers' extra"):
            CrossEncoderFactory.create(llm, embedder)

        assert '~2.3 GB' in caplog.text


class TestCrossEncoderFactoryBGE:
    """Tests for the cross_encoder=bge provider path (mcp-gpu-reranker)."""

    def test_bge_selected_with_default_model(self, monkeypatch):
        """bge with no config.model uses the default model name."""
        mock_cls = Mock()
        mock_cls.return_value.model_name = 'BAAI/bge-reranker-v2-m3'
        monkeypatch.setattr(
            'graphiti_core.cross_encoder.bge_reranker_client.BGERerankerClient',
            mock_cls,
        )
        llm = LLMConfig(provider='openai')
        embedder = EmbedderConfig(provider='openai')
        cross_enc = CrossEncoderConfig(
            enabled=True,
            provider='bge',
        )
        client = CrossEncoderFactory.create(llm, embedder, cross_enc)

        mock_cls.assert_called_once_with(
            model_name='BAAI/bge-reranker-v2-m3',
            device=None,
        )
        assert client.model_name == 'BAAI/bge-reranker-v2-m3'

    def test_bge_custom_model_and_device(self, monkeypatch):
        """bge with explicit model + device passes them through."""
        mock_cls = Mock()
        mock_cls.return_value.model_name = 'custom/bge-model'
        monkeypatch.setattr(
            'graphiti_core.cross_encoder.bge_reranker_client.BGERerankerClient',
            mock_cls,
        )
        llm = LLMConfig(provider='openai')
        embedder = EmbedderConfig(provider='openai')
        cross_enc = CrossEncoderConfig(
            enabled=True,
            provider='bge',
            model='custom/bge-model',
            device='cuda:1',
        )
        client = CrossEncoderFactory.create(llm, embedder, cross_enc)

        mock_cls.assert_called_once_with(
            model_name='custom/bge-model',
            device='cuda:1',
        )
        assert client.model_name == 'custom/bge-model'

    def test_disabled_bge_uses_old_path(self, monkeypatch):
        """cross_encoder.enabled=False falls through to provider logic."""
        mock_ce = Mock()
        monkeypatch.setattr(
            'graphiti_core.cross_encoder.bge_reranker_client.CrossEncoder',
            mock_ce,
        )
        llm = LLMConfig(
            provider='openai',
            providers=LLMProvidersConfig(openai=OpenAIProviderConfig(api_key='k')),
        )
        embedder = EmbedderConfig(
            provider='openai',
            providers=EmbedderProvidersConfig(openai=OpenAIProviderConfig(api_key='k')),
        )
        cross_enc = CrossEncoderConfig(
            enabled=False,
            provider='bge',
        )
        result = CrossEncoderFactory.create(llm, embedder, cross_enc)

        # Should use OpenAIRerankerClient, not BGERerankerClient
        assert isinstance(result, OpenAIRerankerClient)
        mock_ce.assert_not_called()

    def test_non_bge_provider_uses_old_path(self, monkeypatch):
        """cross_encoder.provider='openai' ignores cross_encoder config."""
        mock_ce = Mock()
        monkeypatch.setattr(
            'graphiti_core.cross_encoder.bge_reranker_client.CrossEncoder',
            mock_ce,
        )
        llm = LLMConfig(
            provider='openai',
            providers=LLMProvidersConfig(openai=OpenAIProviderConfig(api_key='k')),
        )
        embedder = EmbedderConfig(
            provider='openai',
            providers=EmbedderProvidersConfig(openai=OpenAIProviderConfig(api_key='k')),
        )
        cross_enc = CrossEncoderConfig(
            enabled=True,
            provider='openai',  # not bge
        )
        result = CrossEncoderFactory.create(llm, embedder, cross_enc)

        assert isinstance(result, OpenAIRerankerClient)
        mock_ce.assert_not_called()


@pytest.mark.asyncio
async def test_graphiti_service_does_not_swallow_reranker_configuration_error(monkeypatch):
    error = ValueError('reranker setup failed')

    def fail_reranker_setup(*_args):
        raise error

    fake_client = Mock()
    fake_client.build_indices_and_constraints = AsyncMock()
    monkeypatch.setattr(CrossEncoderFactory, 'create', fail_reranker_setup)
    monkeypatch.setattr(graphiti_mcp_server, 'Graphiti', Mock(return_value=fake_client))
    service = graphiti_mcp_server.GraphitiService(
        GraphitiConfig(database=DatabaseConfig(provider='neo4j'))
    )

    with pytest.raises(ValueError, match='reranker setup failed'):
        await service.initialize()
