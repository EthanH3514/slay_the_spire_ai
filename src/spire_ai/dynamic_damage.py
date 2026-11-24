"""Public dynamic attack amounts, before ordinary combat modifiers.

No draw ordering, RNG or native special_data fallback is consulted. Missing
Rampage damage is unknown, not a fresh eight-damage card.
"""
from typing import Mapping

from .card_recipes import DamageScaling, DynamicDamage
from .identifiers import canonical_identifier
from .public_piles import public_draw_cards

# Original STRIKE tag, not localized name substring matching. This set does not
# grant executable support to cards belonging to other characters.
STRIKE_CARD_IDS = frozenset({
    "strike_red", "strike_green", "strike_blue", "strike_purple", "twin_strike",
    "wild_strike", "pommel_strike", "perfected_strike", "swift_strike",
    "thunder_strike", "meteor_strike", "sneaky_strike", "windmill_strike",
})


def strike_count(combat: Mapping) -> int:
    return sum(
        canonical_identifier(card.get("id"), "card") in STRIKE_CARD_IDS
        for cards in (combat.get("hand", ()), public_draw_cards(combat), combat.get("discard_pile", ()))
        for card in cards if isinstance(card, Mapping)
    )


def dynamic_damage(profile: DynamicDamage, card: Mapping, combat: Mapping) -> int:
    upgrades = card.get("upgrades", card.get("upgraded", 0))
    if profile.source is DamageScaling.UPGRADE_COUNT and type(upgrades) is not int:
        raise ValueError("Searing Blow requires an integer public upgrade count, not a boolean")
    if isinstance(upgrades, bool):
        upgrades = int(upgrades)
    if not isinstance(upgrades, int) or upgrades < 0:
        raise ValueError("dynamic damage needs a nonnegative public upgrade count")
    if profile.source is DamageScaling.DRAW_PILE_COUNT:
        # The known top is split out of the unordered pile, not an extra card.
        # Do not treat a missing pile as empty or consult master deck / audit RNG.
        key = "draw_pile" if "draw_pile" in combat else "draw_pile_unordered"
        values = combat.get(key)
        if not isinstance(values, (tuple, list)) or any(not isinstance(c, Mapping) for c in values):
            raise ValueError("Mind Blast requires a complete public draw pile")
        if key == "draw_pile_unordered" and "draw_pile_known_top" in combat:
            if not isinstance(combat["draw_pile_known_top"], Mapping):
                raise ValueError("Mind Blast requires a valid public known top")
        return len(public_draw_cards(combat))
    if profile.source is DamageScaling.STRIKE_COUNT:
        return 6 + (3 if upgrades else 2) * strike_count(combat)
    if profile.source is DamageScaling.UPGRADE_COUNT:
        return 12 + upgrades * (upgrades + 7) // 2
    value = card.get("base_damage")
    if profile.source is DamageScaling.PERSISTENT_CARD_BASE_DAMAGE:
        if type(value) is not int or not 15 <= value <= 32767:
            raise ValueError("persistent attack requires public base_damage in reviewed native range 15..32767")
        return value
    if type(value) is not int or value < 8:
        raise ValueError("Rampage requires public base_damage; missing growth is unknown")
    return value


def dynamic_state_reason(combat: Mapping) -> str | None:
    for pile in ("hand", "draw_pile", "discard_pile", "exhaust_pile"):
        cards = public_draw_cards(combat) if pile == "draw_pile" else combat.get(pile, ())
        for card in cards:
            if not isinstance(card, Mapping):
                continue
            identifier = canonical_identifier(card.get("id"), "card")
            source = {"rampage": DamageScaling.CARD_BASE_DAMAGE,
                      "ritual_dagger": DamageScaling.PERSISTENT_CARD_BASE_DAMAGE,
                      "mind_blast": DamageScaling.DRAW_PILE_COUNT,
                      "searing_blow": DamageScaling.UPGRADE_COUNT}.get(identifier)
            if source is not None:
                try:
                    dynamic_damage(DynamicDamage(source), card, combat)
                except ValueError as error:
                    return f"{pile}: {error}"
    return None
