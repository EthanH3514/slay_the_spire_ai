"""Validate public Discovery offers, never infer an unseen random result."""
from typing import Mapping
from functools import lru_cache

from .card_recipes import DiscoveryGeneration
from .identifiers import canonical_identifier
from .random_generation import generation_pool
from .rule_catalog import RuleCatalog


@lru_cache(maxsize=1)
def _pool_ids() -> frozenset[str]:
    # Cache static bundled metadata only, never an offer or sampled game state.
    return frozenset(canonical_identifier(c["id"], "card") for c in
                     generation_pool(DiscoveryGeneration.IRONCLAD_THREE, RuleCatalog.bundled()))


def discovery_candidates(combat: Mapping) -> tuple[Mapping, ...]:
    count = combat.get("selection_copy_count")
    if type(count) is not int or count not in (1, 2):
        raise ValueError("Discovery requires a public copy count of one or two")
    kind = combat.get("selection_card_type", "INVALID")
    if kind not in {"INVALID", "ATTACK", "SKILL", "POWER", "STATUS"}:
        raise ValueError("Discovery requires a public offer pool type")
    cards = combat.get("selection_candidates", ())
    if not isinstance(cards, (list, tuple)) or len(cards) != 3 or not all(isinstance(c, Mapping) for c in cards):
        raise ValueError("Discovery requires three structured public candidates")
    if kind == "STATUS":  # Native DiscoveryAction's public colorless-pool marker.
        from .card_recipes import RandomHandGeneration
        pool = {canonical_identifier(c["id"], "card") for c in generation_pool(
            RandomHandGeneration.COLORLESS_TURN_START, RuleCatalog.bundled())}
    else:
        pool = _pool_ids()
    keys = [canonical_identifier(c.get("id"), "card") for c in cards]
    if len(set(keys)) != 3 or not set(keys) <= pool or any(c.get("upgrades", 0) != 0 for c in cards):
        raise ValueError("Discovery candidates must be distinct base cards from their public pool")
    if kind in {"ATTACK", "SKILL", "POWER"} and any(
            RuleCatalog.bundled().lookup("card", c["id"])["type"] != kind for c in cards):
        raise ValueError("Discovery candidates disagree with public offer type")
    return tuple(cards)


def ironclad_offer_candidates(combat: Mapping, *, source: str) -> tuple[Mapping, ...]:
    """Discovery and Codex share offers, not costs or card insertion semantics."""
    cards = combat.get("selection_candidates", ())
    if not isinstance(cards, (list, tuple)) or len(cards) != 3 or not all(isinstance(c, Mapping) for c in cards):
        raise ValueError(f"{source} requires three structured public candidates")
    keys = [canonical_identifier(c.get("id"), "card") for c in cards]
    if len(set(keys)) != 3 or not set(keys) <= _pool_ids() or any(c.get("upgrades", 0) != 0 for c in cards):
        raise ValueError(f"{source} candidates must be distinct base cards from the reviewed Ironclad pool")
    return tuple(cards)
