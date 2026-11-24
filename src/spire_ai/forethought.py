"""Public ordered bottom choices and conservative free-play state contracts."""
from itertools import combinations
from typing import Mapping

from .identifiers import canonical_identifier
from .public_piles import public_draw_cards


def bottom_choices(combat: Mapping) -> tuple[tuple[int, ...], ...]:
    hand = combat.get("hand")
    optional = combat.get("selection_any_number")
    if type(optional) is not bool or not isinstance(hand, (list, tuple)) or not 1 <= len(hand) <= 10:
        raise ValueError("Forethought requires public hand and boolean selection mode")
    if any(not isinstance(c, Mapping) for c in hand):
        raise ValueError("Forethought requires structured hand cards")
    if not optional:
        return tuple((i,) for i in range(len(hand)))
    # All subsets, two deterministic orders each. Bounded at 2037 choices for ten
    # cards, not 9,864,101 ordered permutations. Policy may prefer either order.
    result = [()]
    for count in range(1, len(hand) + 1):
        for group in combinations(range(len(hand)), count):
            result.append(group)
            if count > 1:
                result.append(tuple(reversed(group)))
    return tuple(result)


def pending_bottom_choices(combat: Mapping) -> tuple[tuple[int, ...], ...]:
    expected = bottom_choices(combat)
    actual = combat.get("selection_options")
    if not isinstance(actual, (list, tuple)) or any(
        not isinstance(g, (list, tuple)) or any(type(i) is not int for i in g) for g in actual
    ) or tuple(tuple(g) for g in actual) != expected:
        raise ValueError("Forethought requires the reviewed bounded ordered choices")
    return expected


def free_play_state_reason(combat: Mapping, *, require_order_certainty: bool = True) -> str | None:
    if not isinstance(combat, Mapping):
        return None
    if require_order_certainty and combat.get("bottom_order_uncertain", False) is not False:
        return "public bottom order after random insertion is not supported"
    cards = list(public_draw_cards(combat))
    for pile in ("hand", "discard_pile", "exhaust_pile"):
        cards.extend(c for c in combat.get(pile, ()) if isinstance(c, Mapping))
    required = str(combat.get("card_select_task", "")).upper() == "FORETHOUGHT" or any(
        canonical_identifier(c.get("id"), "card") == "forethought" for c in cards
    )
    if required and combat.get("bottom_knowledge_tracked") is not True:
        return "Forethought requires tracked public bottom knowledge"
    for card in cards:
        if required or "free_to_play_once" in card:
            if type(card.get("free_to_play_once")) is not bool or any(
                type(card.get(k)) is not int for k in ("base_cost", "cost_for_turn")
            ):
                return "free-play state requires public combat/turn costs and boolean one-use flag"
    return None
