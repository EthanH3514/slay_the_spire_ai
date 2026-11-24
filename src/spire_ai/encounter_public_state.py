"""Visible encounter state and recoverable cards; never latent native caches."""
from collections.abc import Mapping

from .identifiers import canonical_identifier


def public_stasis_cards(combat: Mapping) -> tuple[Mapping, ...]:
    return tuple(m["stasis_card"] for m in combat.get("monsters", ())
                 if isinstance(m, Mapping) and isinstance(m.get("stasis_card"), Mapping))


def encounter_public_state_reason(combat: Mapping) -> str | None:
    from .mechanism_evidence import DEFAULT_MECHANISM_EVIDENCE, EvidenceCapability
    monsters = combat.get("monsters", ())
    for monster in monsters:
        if not isinstance(monster, Mapping):
            continue
        if "public_generation" in monster and (
            type(monster["public_generation"]) is not int or monster["public_generation"] < 0
        ):
            return "public monster generation must be a nonnegative observed spawn count"
        active_stasis = (not monster.get("is_gone", False) and any(
            isinstance(p, Mapping) and canonical_identifier(p.get("id"), "power") == "stasis"
            for p in monster.get("powers", ())))
        card = monster.get("stasis_card")
        if active_stasis != (card is not None):
            return "Stasis requires the visibly held card, and no card after its return"
        if card is not None and (not isinstance(card, Mapping) or
            not DEFAULT_MECHANISM_EVIDENCE.supports(
                "card", card.get("id"), EvidenceCapability.SEARCH, imported=True)):
            return "Stasis held card lacks scoped search evidence"
    player = combat.get("player", {})
    if isinstance(player, Mapping) and any(
        isinstance(p, Mapping) and canonical_identifier(p.get("id"), "power") == "surrounded"
        for p in player.get("powers", ())
    ):
        facing = player.get("facing_monster_slot")
        if type(facing) is not int or not 0 <= facing < len(monsters):
            return "Surrounded requires a public facing monster slot"
    return None


def with_recoverable_cards(combat: Mapping) -> Mapping:
    """Guard-only view: Stasis cards can return, but are NOT draw candidates.

    Put them in the exhaust candidate set for existing whole-combat validators.
    Do not change the observation, actual piles, draw features or card counts.
    """
    held = public_stasis_cards(combat)
    if not held:
        return combat
    return dict(combat, exhaust_pile=tuple(combat.get("exhaust_pile", ())) + held)
