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

import json
from datetime import datetime

from graphiti_core.nodes import EpisodeType, EpisodicNode
from graphiti_core.utils.content_chunking import estimate_tokens
from graphiti_core.utils.text_utils import (
    MAX_SUMMARY_CHARS,
    build_previous_episodes_context,
    truncate_at_sentence,
)


def test_truncate_at_sentence_short_text():
    """Test that short text is returned unchanged."""
    text = 'This is a short sentence.'
    result = truncate_at_sentence(text, 100)
    assert result == text


def test_truncate_at_sentence_empty():
    """Test that empty text is handled correctly."""
    assert truncate_at_sentence('', 100) == ''
    assert truncate_at_sentence(None, 100) is None


def test_truncate_at_sentence_exact_length():
    """Test text at exactly max_chars."""
    text = 'A' * 100
    result = truncate_at_sentence(text, 100)
    assert result == text


def test_truncate_at_sentence_with_period():
    """Test truncation at sentence boundary with period."""
    text = 'First sentence. Second sentence. Third sentence. Fourth sentence.'
    result = truncate_at_sentence(text, 40)
    assert result == 'First sentence. Second sentence.'
    assert len(result) <= 40


def test_truncate_at_sentence_with_question():
    """Test truncation at sentence boundary with question mark."""
    text = 'What is this? This is a test. More text here.'
    result = truncate_at_sentence(text, 30)
    assert result == 'What is this? This is a test.'
    assert len(result) <= 32


def test_truncate_at_sentence_with_exclamation():
    """Test truncation at sentence boundary with exclamation mark."""
    text = 'Hello world! This is exciting. And more text.'
    result = truncate_at_sentence(text, 30)
    assert result == 'Hello world! This is exciting.'
    assert len(result) <= 32


def test_truncate_at_sentence_no_boundary():
    """Test truncation when no sentence boundary exists before max_chars."""
    text = 'This is a very long sentence without any punctuation marks near the beginning'
    result = truncate_at_sentence(text, 30)
    assert len(result) <= 30
    assert result.startswith('This is a very long sentence')


def test_truncate_at_sentence_multiple_periods():
    """Test with multiple sentence endings."""
    text = 'A. B. C. D. E. F. G. H.'
    result = truncate_at_sentence(text, 10)
    assert result == 'A. B. C.'
    assert len(result) <= 10


def test_truncate_at_sentence_strips_trailing_whitespace():
    """Test that trailing whitespace is stripped."""
    text = 'First sentence.   Second sentence.'
    result = truncate_at_sentence(text, 20)
    assert result == 'First sentence.'
    assert not result.endswith(' ')


def test_max_summary_chars_constant():
    """Test that MAX_SUMMARY_CHARS is set to expected value."""
    assert MAX_SUMMARY_CHARS == 1000


def test_truncate_at_sentence_realistic_summary():
    """Test with a realistic entity summary."""
    text = (
        'John is a software engineer who works at a tech company in San Francisco. '
        'He has been programming for over 10 years and specializes in Python and distributed systems. '
        'John enjoys hiking on weekends and is learning to play guitar. '
        'He graduated from MIT with a degree in computer science.'
    )
    result = truncate_at_sentence(text, MAX_SUMMARY_CHARS)
    assert len(result) <= MAX_SUMMARY_CHARS
    # Should keep complete sentences
    assert result.endswith('.')
    # Should include at least the first sentence
    assert 'John is a software engineer' in result


def test_build_previous_episodes_context_empty():
    """Test with empty input."""
    result = build_previous_episodes_context([])
    assert result == []


def test_build_previous_episodes_context_single():
    """Test with a single episode — same dict shape as inline builders."""
    ep = EpisodicNode(
        name='test episode',
        group_id='',
        labels=[],
        source=EpisodeType.message,
        source_description='test',
        content='This is episode content.',
        valid_at=datetime(2024, 1, 1, 12, 0, 0),
        entity_edges=[],
    )
    result = build_previous_episodes_context([ep])
    assert len(result) == 1
    assert result[0]['content'] == 'This is episode content.'
    assert result[0]['timestamp'] == '2024-01-01T12:00:00'


def test_build_previous_episodes_context_no_truncation():
    """Test that no truncation happens when PREVIOUS_EPISODE_MAX_CHARS is 0 (default)."""
    content = 'A' * 5000
    ep = EpisodicNode(
        name='test',
        group_id='',
        labels=[],
        source=EpisodeType.message,
        source_description='test',
        content=content,
        valid_at=datetime(2024, 1, 1),
        entity_edges=[],
    )
    result = build_previous_episodes_context([ep])
    assert result[0]['content'] == content


def test_build_previous_episodes_context_truncation(monkeypatch):
    """Test truncation is applied when PREVIOUS_EPISODE_MAX_CHARS > 0."""
    from graphiti_core import helpers

    monkeypatch.setattr(helpers, 'PREVIOUS_EPISODE_MAX_CHARS', 50)
    content = 'First sentence. Second sentence. Third sentence. Fourth sentence.'
    ep = EpisodicNode(
        name='test',
        group_id='',
        labels=[],
        source=EpisodeType.message,
        source_description='test',
        content=content,
        valid_at=datetime(2024, 1, 1),
        entity_edges=[],
    )
    result = build_previous_episodes_context([ep])
    assert len(result[0]['content']) <= 50
    assert result[0]['content'].endswith('.')
    assert result[0]['timestamp'] == '2024-01-01T00:00:00'


class TestBuildPreviousEpisodesContextBudget:
    """Tests for budget-limited episode selection."""

    def test_budget_none_unchanged(self):
        """budget=None → same as before (all episodes)."""
        eps = [
            EpisodicNode(
                name='ep1',
                group_id='',
                labels=[],
                source=EpisodeType.message,
                source_description='t',
                content='First episode.',
                valid_at=datetime(2024, 1, 1),
                entity_edges=[],
            ),
            EpisodicNode(
                name='ep2',
                group_id='',
                labels=[],
                source=EpisodeType.message,
                source_description='t',
                content='Second episode.',
                valid_at=datetime(2024, 1, 2),
                entity_edges=[],
            ),
        ]
        result = build_previous_episodes_context(eps, budget_tokens=None)
        assert len(result) == 2

    def test_budget_zero(self):
        """budget=0 → empty list."""
        ep = EpisodicNode(
            name='ep1',
            group_id='',
            labels=[],
            source=EpisodeType.message,
            source_description='t',
            content='content.',
            valid_at=datetime(2024, 1, 1),
            entity_edges=[],
        )
        result = build_previous_episodes_context([ep], budget_tokens=0)
        assert result == []

    def test_budget_negative(self):
        """budget<0 → empty list."""
        ep = EpisodicNode(
            name='ep1',
            group_id='',
            labels=[],
            source=EpisodeType.message,
            source_description='t',
            content='content.',
            valid_at=datetime(2024, 1, 1),
            entity_edges=[],
        )
        result = build_previous_episodes_context([ep], budget_tokens=-5)
        assert result == []

    def test_budget_tight_keeps_newest_first(self):
        """Tight budget keeps newest episodes first."""
        from graphiti_core.utils.content_chunking import estimate_tokens

        # Create 3 episodes with distinct timestamps
        eps = [
            EpisodicNode(
                name='old',
                group_id='',
                labels=[],
                source=EpisodeType.message,
                source_description='t',
                content='A' * 100,
                valid_at=datetime(2024, 1, 1),
                entity_edges=[],
            ),
            EpisodicNode(
                name='mid',
                group_id='',
                labels=[],
                source=EpisodeType.message,
                source_description='t',
                content='B' * 100,
                valid_at=datetime(2024, 6, 1),
                entity_edges=[],
            ),
            EpisodicNode(
                name='new',
                group_id='',
                labels=[],
                source=EpisodeType.message,
                source_description='t',
                content='C' * 100,
                valid_at=datetime(2024, 12, 1),
                entity_edges=[],
            ),
        ]

        # Budget tight enough for only the newest 2 (each ~110 tokens with overhead)
        budget = (
            estimate_tokens(json.dumps({'content': 'C' * 100, 'timestamp': '2024-12-01T00:00:00'}))
            + 10
        )
        budget += (
            estimate_tokens(json.dumps({'content': 'B' * 100, 'timestamp': '2024-06-01T00:00:00'}))
            + 10
        )
        # Add a little more but not enough for the oldest
        budget = int(budget) + 1

        result = build_previous_episodes_context(eps, budget_tokens=budget)
        # Should have 2 episodes (newest and mid), not oldest
        assert len(result) == 2
        timestamps = {ep['timestamp'] for ep in result}
        assert '2024-12-01T00:00:00' in timestamps
        assert '2024-06-01T00:00:00' in timestamps
        assert '2024-01-01T00:00:00' not in timestamps

    def test_chronological_output_order(self):
        """Output is chronological despite newest-first selection."""
        eps = [
            EpisodicNode(
                name='ep1',
                group_id='',
                labels=[],
                source=EpisodeType.message,
                source_description='t',
                content='First.',
                valid_at=datetime(2024, 1, 1),
                entity_edges=[],
            ),
            EpisodicNode(
                name='ep2',
                group_id='',
                labels=[],
                source=EpisodeType.message,
                source_description='t',
                content='Second.',
                valid_at=datetime(2024, 6, 1),
                entity_edges=[],
            ),
        ]
        # Tight budget for only 1 episode
        budget = (
            estimate_tokens(json.dumps({'content': 'Second.', 'timestamp': '2024-06-01T00:00:00'}))
            + 10
        )

        result = build_previous_episodes_context(eps, budget_tokens=budget)
        assert len(result) == 1
        # The newest (June) should be selected
        assert result[0]['timestamp'] == '2024-06-01T00:00:00'

    def test_partial_truncation_boundary(self):
        """When budget allows partial truncation at boundary."""

        # Create a longer content (~500 tokens / 2000 chars)
        long_content = '. '.join(
            f'Sentence number {i} with additional descriptive words.' for i in range(500)
        )
        ep = EpisodicNode(
            name='ep',
            group_id='',
            labels=[],
            source=EpisodeType.message,
            source_description='t',
            content=long_content,
            valid_at=datetime(2024, 1, 1),
            entity_edges=[],
        )

        # Budget: 100 tokens (400 chars) — fits about 10 sentences after truncation
        # (>= _PARTIAL_TRUNCATION_THRESHOLD=50)
        budget = 100

        result = build_previous_episodes_context([ep], budget_tokens=budget)
        assert len(result) == 1
        # Content should be truncated to fit budget
        assert len(result[0]['content']) < len(long_content)
        # Should respect sentence boundaries
        assert result[0]['content'].endswith('.')

    def test_max_chars_still_applied(self):
        """PREVIOUS_EPISODE_MAX_CHARS is still the upper bound."""
        from graphiti_core import helpers

        # Create long content with sentence boundaries
        large_content = '. '.join(f'Sentence {i} with more words.' for i in range(500))
        ep = EpisodicNode(
            name='ep',
            group_id='',
            labels=[],
            source=EpisodeType.message,
            source_description='t',
            content=large_content,
            valid_at=datetime(2024, 1, 1),
            entity_edges=[],
        )

        # Set MAX_CHARS to something small
        from unittest.mock import patch

        with patch.object(helpers, 'PREVIOUS_EPISODE_MAX_CHARS', 100):
            result = build_previous_episodes_context(
                [ep],
                budget_tokens=99999,  # huge budget
            )
            assert len(result) == 1
            assert len(result[0]['content']) <= 100
            # May end with '.' if truncated at sentence boundary, or with content
            assert len(result[0]['content']) > 0


class TestRemainingBudgetForPreviousEpisodes:
    @staticmethod
    def _client(budget):
        from unittest.mock import MagicMock

        client = MagicMock()
        client.input_token_budget.return_value = budget
        return client

    def test_no_budget_returns_none(self):
        from graphiti_core.utils.text_utils import remaining_budget_for_previous_episodes

        builder_calls = []
        result = remaining_budget_for_previous_episodes(
            self._client(None), lambda ctx: builder_calls.append(ctx) or [], {}
        )
        assert result is None
        assert builder_calls == []

    def test_subtracts_rendered_prompt_and_schema(self):
        from pydantic import BaseModel

        from graphiti_core.prompts.models import Message
        from graphiti_core.utils.text_utils import remaining_budget_for_previous_episodes

        class Out(BaseModel):
            value: str

        seen = []

        def builder(ctx):
            seen.append(ctx)
            return [
                Message(role='system', content='s' * 400),
                Message(role='user', content='u' * 800),
            ]

        context = {'previous_episodes': [{'content': 'x'}], 'other': 1}
        result = remaining_budget_for_previous_episodes(
            self._client(1000), builder, context, response_model=Out
        )

        schema_tokens = estimate_tokens(json.dumps(Out.model_json_schema()))
        assert result == 1000 - 100 - 200 - schema_tokens
        # Fixed cost is measured without previous episodes; caller's context is not mutated
        assert seen[0]['previous_episodes'] == []
        assert context['previous_episodes'] == [{'content': 'x'}]

    def test_clamps_to_zero(self):
        from graphiti_core.utils.text_utils import remaining_budget_for_previous_episodes

        result = remaining_budget_for_previous_episodes(
            self._client(10), lambda ctx: ['y' * 4000], {}
        )
        assert result == 0
