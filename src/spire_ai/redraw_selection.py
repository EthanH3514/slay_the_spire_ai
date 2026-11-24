"""Bounded public discard/redraw choices; no hidden draw order is consulted."""
from itertools import combinations
from typing import Mapping


def redraw_subsets(combat: Mapping) -> tuple[tuple[int, ...], ...]:
    hand = combat.get("hand")
    if (not isinstance(hand, (list, tuple)) or len(hand) > 10
            or any(not isinstance(card, Mapping) or not card.get("id") for card in hand)):
        raise ValueError("Gamble requires a bounded structured public hand")
    return tuple(group for count in range(len(hand) + 1)
                 for group in combinations(range(len(hand)), count))


def pending_redraw_options(combat: Mapping) -> tuple[tuple[int, ...], ...]:
    expected = redraw_subsets(combat)
    hand = combat["hand"]
    ids = [card.get("unique_id") for card in hand]
    if any(type(uid) is not int or uid < 0 for uid in ids) or len(set(ids)) != len(ids):
        raise ValueError("Gamble requires distinct public hand identities")
    options = combat.get("selection_options")
    if (not isinstance(options, (list, tuple))
            or any(not isinstance(group, (list, tuple)) or any(type(i) is not int for i in group)
                   for group in options)
            or tuple(tuple(group) for group in options) != expected):
        raise ValueError("Gamble requires complete public subsets including zero")
    return expected
