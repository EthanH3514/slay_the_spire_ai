"""Shared public selection contracts, independent of cards versus potions."""
from itertools import permutations

SELECTION_TASKS = frozenset({
    "EXHAUST_ONE", "EXHAUST_MANY", "FORETHOUGHT", "ARMAMENTS", "HEADBUTT", "WARCRY",
    "EXHUME", "SECRET_TECHNIQUE", "SECRET_WEAPON", "DUAL_WIELD", "DISCOVERY", "GAMBLE",
    "CODEX", "LIQUID_MEMORIES_POTION",
})


def retrieval_choices(combat):
    count = combat.get("selection_pick_count", 1)
    cards = combat.get("selection_candidates", ())
    if type(count) is not int or count not in (1, 2) or count > len(cards):
        raise ValueError("Liquid Memories requires a public retrieval count of one or two")
    # Selection order determines which card enters a nearly full hand.
    return tuple(permutations(range(len(cards)), count))
