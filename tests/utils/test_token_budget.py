"""Tests for graphiti_core.utils.token_budget.pack_by_budget."""

from graphiti_core.utils.token_budget import pack_by_budget


class TestPackByBudget:
    """Tests for the greedy budget batcher."""

    def test_empty_input(self):
        """Empty list → empty result regardless of params."""
        assert pack_by_budget([], lambda x: 1, budget=100) == []
        assert pack_by_budget([], lambda x: 1, budget=None, max_items=2) == []

    def test_none_budget_no_limit(self):
        """budget=None → all items in one batch."""
        items = ['a', 'b', 'c']
        result = pack_by_budget(items, lambda x: len(x), budget=None)
        assert result == [['a', 'b', 'c']]

    def test_none_budget_with_max_items(self):
        """budget=None + max_items → batch by max_items only."""
        items = ['a', 'b', 'c', 'd', 'e']
        result = pack_by_budget(items, lambda x: len(x), budget=None, max_items=2)
        assert result == [['a', 'b'], ['c', 'd'], ['e']]

    def test_budget_constraint(self):
        """Items packed until budget is exceeded."""
        items = ['a', 'bb', 'ccc', 'dddd']  # costs: 1, 2, 3, 4
        result = pack_by_budget(items, lambda x: len(x), budget=5)
        assert result == [['a', 'bb'], ['ccc'], ['dddd']]

    def test_oversize_item_goes_alone(self):
        """Item larger than remaining budget goes alone."""
        items = ['a', 'x' * 100, 'b']  # costs: 1, 100, 1
        result = pack_by_budget(items, lambda x: len(x), budget=50)
        assert result == [['a'], ['x' * 100], ['b']]

    def test_fixed_cost_reduces_budget(self):
        """fixed_cost is deducted before considering items."""
        # budget=10, fixed_cost=7 → only 3 tokens available for items
        items = ['a', 'bb', 'ccc', 'dddd']  # costs: 1, 2, 3, 4
        result = pack_by_budget(items, lambda x: len(x), budget=10, fixed_cost=7)
        assert result == [['a', 'bb'], ['ccc'], ['dddd']]

    def test_order_preserved(self):
        """Items appear in original order across batches."""
        items = list(range(10))
        result = pack_by_budget(items, lambda x: 1, budget=3)
        flat = [item for batch in result for item in batch]
        assert flat == items

    def test_over_budget_item_alone(self):
        """Single oversized item goes alone, surrounding items batch normally."""
        items = ['a', 'b', 'x' * 1000, 'c', 'd']
        result = pack_by_budget(items, lambda x: len(x), budget=100)
        # 'a' (1) + 'b' (1) = 2 ≤ 100 → batch together
        # 'xxx...' (1000) > 100 → alone
        # 'c' (1) + 'd' (1) = 2 ≤ 100 → batch together
        assert len(result) == 3
        assert result[0] == ['a', 'b']
        assert result[1] == ['x' * 1000]
        assert result[2] == ['c', 'd']


def test_oversized_first_item_produces_no_empty_batch():
    assert pack_by_budget([50, 1, 1], lambda x: x, 10) == [[50], [1, 1]]


def test_fixed_cost_exceeding_budget_yields_single_item_batches():
    assert pack_by_budget([5, 5, 5], lambda x: x, 10, fixed_cost=20) == [[5], [5], [5]]


def test_no_batch_is_ever_empty():
    for items in ([50], [1, 50, 1], [50, 50], [3, 3, 3, 3]):
        batches = pack_by_budget(items, lambda x: x, 5, max_items=2)
        assert all(batches)
        assert [i for b in batches for i in b] == items
