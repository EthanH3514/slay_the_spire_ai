"""Public bounded subset choices; native commands never enter policy actions."""
from itertools import combinations
from typing import Mapping


def exhaust_subsets(combat: Mapping) -> tuple[tuple[int, ...], ...]:
    limit = combat.get("selection_max_count")
    hand = combat.get("hand")
    if type(limit) is not int or not 1 <= limit <= 10:
        raise ValueError("multi-exhaust requires a public maximum in 1..10")
    if not isinstance(hand, (list, tuple)) or not 0 <= len(hand) <= 10 or any(not isinstance(c, Mapping) for c in hand):
        raise ValueError("multi-exhaust requires a valid public hand of at most ten cards")
    return tuple(group for count in range(min(limit, len(hand)) + 1)
                 for group in combinations(range(len(hand)), count))


def pending_exhaust_options(combat: Mapping) -> tuple[tuple[int, ...], ...]:
    expected = exhaust_subsets(combat)
    values = combat.get("selection_options")
    if not isinstance(values, (list, tuple)) or any(
        not isinstance(group, (list, tuple)) or any(type(i) is not int for i in group)
        for group in values
    ) or tuple(tuple(group) for group in values) != expected:
        raise ValueError("multi-exhaust requires the complete public subset options including zero")
    return expected
