"""Ordinary Ironclad potion mechanisms and public transition contracts.

Card movement uses the same native public-order ledger as card effects; potion
history must not pretend that any change of pile makes the future unknowable.
"""
from __future__ import annotations

from dataclasses import dataclass

from .actions import ActionKind

POTION_CONTRACT_VERSION = "public-potion-contracts-v3-complete-ironclad"
EMPTY_SLOTS = frozenset({"emptypotionslot", "emptypotionid", "potionslot"})


def potion_key(identifier) -> str:
    return "".join(c for c in str(identifier).lower() if c.isalnum())


@dataclass(frozen=True)
class PotionContract:
    id: str
    effect: str
    amount: int
    target: bool = False

    def as_dict(self):
        return {"effect": self.effect, "amount": self.amount, "target": self.target}


CONTRACTS = (
    PotionContract("BLOCK_POTION", "block", 12),
    PotionContract("FIRE_POTION", "damage", 20, True),
    PotionContract("EXPLOSIVE_POTION", "damage_all", 10),
    PotionContract("ENERGY_POTION", "energy", 2),
    PotionContract("STRENGTH_POTION", "strength", 2),
    PotionContract("DEXTERITY_POTION", "dexterity", 2),
    PotionContract("WEAK_POTION", "weak", 3, True),
    PotionContract("FEAR_POTION", "vulnerable", 3, True),
    PotionContract("ANCIENT_POTION", "artifact", 1),
    PotionContract("BLOOD_POTION", "heal_percent", 20),
    PotionContract("CULTIST_POTION", "ritual", 1),
    PotionContract("ESSENCE_OF_STEEL", "plated_armor", 4),
    PotionContract("FLEX_POTION", "temporary_strength", 5),
    PotionContract("FRUIT_JUICE", "max_hp", 5),
    PotionContract("HEART_OF_IRON", "metallicize", 6),
    PotionContract("LIQUID_BRONZE", "thorns", 3),
    PotionContract("REGEN_POTION", "regen", 5),
    PotionContract("SPEED_POTION", "temporary_dexterity", 5),
    PotionContract("BLESSING_OF_THE_FORGE", "upgrade_hand", 1),
    PotionContract("SWIFT_POTION", "draw", 3),
    PotionContract("GAMBLERS_BREW", "redraw", 1),
    PotionContract("LIQUID_MEMORIES", "retrieve", 1),
    PotionContract("ELIXIR_POTION", "exhaust", 10),
    PotionContract("SNECKO_OIL", "draw_random_cost", 5),
    PotionContract("ATTACK_POTION", "discovery_attack", 1),
    PotionContract("SKILL_POTION", "discovery_skill", 1),
    PotionContract("POWER_POTION", "discovery_power", 1),
    PotionContract("COLORLESS_POTION", "discovery_colorless", 1),
    PotionContract("DISTILLED_CHAOS", "autoplay", 3),
    PotionContract("DUPLICATION_POTION", "duplication", 1),
    PotionContract("ENTROPIC_BREW", "refill", 0),
    PotionContract("SMOKE_BOMB", "escape", 0),
    PotionContract("FAIRY_POTION", "passive_revive", 30),
)
CONTRACT_BY_KEY = {potion_key(c.id): c for c in CONTRACTS}


def potion_history_reason(before, action, after) -> str | None:
    """Check legal consumption and public movement evidence, not latent effects.

    Native tracked top/bottom/opening constraints remain authoritative. Uncertainty
    is a current-root condition and may be cleared by a later observed shuffle;
    it must not poison the entire combat's history permanently.
    """
    prefix = "potion history: "
    if action.kind is not ActionKind.USE_POTION or action not in before.legal_actions:
        return prefix + "action is not a legal potion use"
    slots = before.observation.get("inventory", {}).get("potions", ())
    index = action.potion_index
    if type(index) is not int or not 0 <= index < len(slots):
        return prefix + "missing public potion slot"
    contract = CONTRACT_BY_KEY.get(potion_key(slots[index].get("id")))
    if contract is None:
        return prefix + "unreviewed potion " + str(slots[index].get("id"))
    if contract.effect == "passive_revive":
        return prefix + "Fairy is passive and cannot be used manually"
    old = before.observation.get("combat", {})
    new = after.observation.get("combat", {})
    if old.get("input_state") != "PLAYER_NORMAL" or old.get("effects_resolved") is not True:
        return prefix + "requires a resolved normal source"
    if new.get("input_state") not in {"PLAYER_NORMAL", "CARD_SELECT"}:
        return prefix + "requires a public decision boundary"
    if new.get("input_state") == "PLAYER_NORMAL" and new.get("effects_resolved") is not True:
        return prefix + "normal destination has pending effects"
    if new.get("input_state") == "CARD_SELECT" and contract.effect not in {
        "redraw", "retrieve", "exhaust", "autoplay", "discovery_attack",
        "discovery_skill", "discovery_power", "discovery_colorless",
    }:
        return prefix + "effect cannot open a selection"
    if contract.target:
        target = action.target_index
        monsters = old.get("monsters", ())
        if (type(target) is not int or not 0 <= target < len(monsters)
                or monsters[target].get("current_hp", 0) <= 0 or monsters[target].get("is_gone")):
            return prefix + "missing living public target"
    elif action.target_index is not None:
        return prefix + "unexpected target"
    remaining = after.observation.get("inventory", {}).get("potions", ())
    if len(slots) != len(remaining):
        return prefix + "slot consumption mismatch"
    for i, (a, b) in enumerate(zip(slots, remaining)):
        old_id, new_id = potion_key(a.get("id")), potion_key(b.get("id"))
        if contract.effect == "refill" and (i == index or old_id in EMPTY_SLOTS):
            if new_id not in CONTRACT_BY_KEY and new_id not in EMPTY_SLOTS:
                return prefix + "refill produced an out-of-scope potion"
        elif i == index:
            if new_id not in EMPTY_SLOTS:
                return prefix + "slot consumption mismatch"
        elif old_id != new_id and not (old_id == "fairypotion" and new_id in EMPTY_SLOTS):
            return prefix + "unrelated potion slot changed"
    if old.get("turn") != new.get("turn"):
        return prefix + "potion unexpectedly advanced the turn"
    for combat in (old, new):
        if (combat.get("bottom_knowledge_tracked") is not True
                or combat.get("opening_order_schema") != 1
                or combat.get("top_knowledge_schema") != 1):
            return prefix + "missing native public-order tracking schema"
    # Retain the stronger no-movement invariant for mechanisms that cannot move
    # cards. Damage/debuff death callbacks (e.g. Stasis) and draw/selection potions
    # instead use the native tracked movement ledger and subsequent root checks.
    if contract.effect in {"block", "energy", "strength", "dexterity", "artifact", "heal_percent",
            "ritual", "plated_armor", "temporary_strength", "max_hp", "metallicize", "thorns",
            "regen", "temporary_dexterity", "duplication", "refill"}:
        def identity(card):
            if type(card.get("unique_id")) is not int:
                raise ValueError("missing public card identity")
            return tuple(card.get(k) for k in ("unique_id", "id", "upgrades", "cost_for_turn",
                "base_cost", "free_to_play_once", "special_data"))
        try:
            for pile in ("hand", "draw_pile", "draw_pile_unordered", "discard_pile", "exhaust_pile",
                         "draw_pile_known_bottom"):
                if (pile in old) != (pile in new):
                    return prefix + "pile knowledge changed: " + pile
                a, b = [identity(c) for c in old.get(pile, ())], [identity(c) for c in new.get(pile, ())]
                if pile == "draw_pile_unordered":
                    a, b = sorted(a, key=repr), sorted(b, key=repr)
                if a != b:
                    return prefix + "card movement or mutation: " + pile
            a, b = old.get("draw_pile_known_top"), new.get("draw_pile_known_top")
            if (identity(a) if a else None) != (identity(b) if b else None):
                return prefix + "public draw knowledge changed"
            for field in ("public_known_top_ids", "public_known_bottom_ids", "top_order_uncertain",
                          "bottom_order_uncertain", "public_opening_priority_ids", "opening_order_uncertain"):
                if old.get(field) != new.get(field):
                    return prefix + "public turn/order contract changed"
        except (ValueError, TypeError, AttributeError) as error:
            return prefix + str(error)
    return None
