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

from collections.abc import Sequence
from typing import Callable, TypeVar  # noqa: UP035 (Callable has no collections.abc equivalent)

T = TypeVar('T')


def pack_by_budget(
    items: Sequence[T],
    cost: Callable[[T], int],
    budget: int | None,
    fixed_cost: int = 0,
    max_items: int | None = None,
) -> list[list[T]]:
    """Greedy, order-preserving batcher bounded by an input-token budget.

    * ``budget=None`` → batch only by ``max_items`` (or a single batch).
    * Every batch has >= 1 item; an item larger than the remaining budget
      goes alone.
    * ``fixed_cost`` is deducted from the budget before any item cost is
      considered (e.g. system prompt size).

    Args:
        items: Items to batch.
        cost: Callable returning the token cost of a single item.
        budget: Maximum total tokens per batch. ``None`` means unbounded
                (only ``max_items`` applies).
        fixed_cost: Cost already paid per batch (e.g. system prompt).
        max_items: Maximum number of items per batch. ``None`` means no limit.

    Returns:
        A list of batches (lists of items in original order). Empty list when
        ``items`` is empty.
    """
    if not items:
        return []

    limit = budget - fixed_cost if budget is not None else None
    batches: list[list[T]] = []
    current_batch: list[T] = []
    current_cost = 0  # cost of items in current batch only

    for item in items:
        item_cost = cost(item)
        over_budget = limit is not None and current_cost + item_cost > limit
        over_count = max_items is not None and len(current_batch) >= max_items
        # Close the current batch only when it holds something, so every batch has >= 1
        # item and an oversized item still goes out alone.
        if current_batch and (over_budget or over_count):
            batches.append(current_batch)
            current_batch = []
            current_cost = 0
        current_batch.append(item)
        current_cost += item_cost

    if current_batch:
        batches.append(current_batch)

    return batches
