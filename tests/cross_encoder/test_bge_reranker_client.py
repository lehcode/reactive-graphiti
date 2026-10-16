"""
Copyright 2024, Zep Software, Inc.
Copyright 2025-2026, Anton Repin <robot@pimeleon.org>

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from unittest.mock import MagicMock

import pytest

from graphiti_core.cross_encoder.bge_reranker_client import BGERerankerClient


@pytest.fixture
def mock_cross_encoder(monkeypatch):
    """Provide a mocked CrossEncoder so no model is downloaded."""
    mock = MagicMock()
    monkeypatch.setattr(
        'graphiti_core.cross_encoder.bge_reranker_client.CrossEncoder',
        mock,
    )
    return mock


class TestBGERerankerClientInit:
    """Unit tests for BGERerankerClient.__init__ with mocked CrossEncoder."""

    def test_default_args(self, mock_cross_encoder):
        """BGERerankerClient() uses default model and device=None."""
        BGERerankerClient()

        mock_cross_encoder.assert_called_once_with(
            'BAAI/bge-reranker-v2-m3',
            device=None,
        )

    def test_custom_model_and_device(self, mock_cross_encoder):
        """Custom model_name and device are forwarded to CrossEncoder."""
        BGERerankerClient(model_name='custom-model', device='cuda:1')

        mock_cross_encoder.assert_called_once_with(
            'custom-model',
            device='cuda:1',
        )

    def test_custom_model_only(self, mock_cross_encoder):
        """Custom model with device=None (default)."""
        BGERerankerClient(model_name='another-model')

        mock_cross_encoder.assert_called_once_with(
            'another-model',
            device=None,
        )


class TestBGERerankerClientRank:
    """Unit tests for BGERerankerClient.rank with mocked predict."""

    @pytest.mark.asyncio
    async def test_rank_sorts_descending(self, mock_cross_encoder):
        """rank() returns passages sorted by score descending."""
        mock_model = MagicMock()
        mock_model.predict.return_value = [0.2, 0.9, 0.5]
        mock_cross_encoder.return_value = mock_model

        client = BGERerankerClient(model_name='test-model', device='cuda:0')
        result = await client.rank('query', ['p1', 'p2', 'p3'])

        # Check descending order
        scores = [score for _, score in result]
        assert scores == sorted(scores, reverse=True)
        assert result[0][0] == 'p2'  # highest score
        assert result[0][1] == 0.9

    @pytest.mark.asyncio
    async def test_rank_empty_passages(self, mock_cross_encoder):
        """rank() returns empty list for no passages."""
        client = BGERerankerClient()
        result = await client.rank('query', [])

        assert result == []

    @pytest.mark.asyncio
    async def test_rank_single_passage(self, mock_cross_encoder):
        """rank() returns single passage correctly."""
        mock_model = MagicMock()
        mock_model.predict.return_value = [0.85]
        mock_cross_encoder.return_value = mock_model

        client = BGERerankerClient()
        result = await client.rank('query', ['single passage'])

        assert len(result) == 1
        assert result[0][0] == 'single passage'
        assert isinstance(result[0][1], float)
