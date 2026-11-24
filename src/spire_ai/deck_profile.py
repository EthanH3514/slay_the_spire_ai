"""Public master-deck capabilities, not a turn simulator or an archetype label.

Units are deliberately separate (damage, block, cards, energy, trigger sources).
Conditional and delayed effects are never added to immediate resources. Unknown
valuation is retained explicitly even when the combat recipe itself is reviewed.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Sequence

from .card_recipes import (
    DEFAULT_CARD_RECIPES, AmountSource, AttackCountSource, CardTrait, CombatEvent,
    DamageScaling, EffectPrimitive as E, EffectTarget, ExhaustTriggerEffect, TurnEffectKind,
)
from .rule_catalog import RuleCatalog


DECK_PROFILE_VERSION = "public-master-deck-v1"


@dataclass(frozen=True)
class CardCapabilities:
    identifier: str
    upgrades: int
    features: Mapping[str, float]
    unknown: tuple[str, ...]

    def as_dict(self) -> dict:
        return {"id": self.identifier, "upgrades": self.upgrades,
                "features": dict(self.features), "unknown": list(self.unknown)}


@dataclass(frozen=True)
class DeckProfile:
    cards: tuple[CardCapabilities, ...]
    features: Mapping[str, float]
    unknown: tuple[str, ...]

    def as_dict(self) -> dict:
        return {"version": DECK_PROFILE_VERSION, "size": len(self.cards),
                "features": dict(self.features), "unknown": list(self.unknown),
                "cards": [card.as_dict() for card in self.cards]}


def card_capabilities(card: Mapping, catalog: RuleCatalog) -> CardCapabilities:
    identifier = catalog.canonical_id("card", card.get("id"))
    upgrades = card.get("upgrades", card.get("upgraded", 0))
    if isinstance(upgrades, bool):
        upgrades = int(upgrades)
    if type(upgrades) is not int or upgrades < 0:
        raise ValueError("invalid public card upgrade count")
    upgraded = upgrades > 0
    entry = catalog.lookup("card", identifier)
    recipe = DEFAULT_CARD_RECIPES.lookup(identifier)
    f: Counter[str] = Counter()
    unknown: list[str] = []
    if entry is None or recipe is None:
        return CardCapabilities(identifier, upgrades, MappingProxyType({}),
                                ("unreviewed_card",))
    variant = entry["variants"][int(upgraded)]
    # Combat cost/cost_for_turn must not leak into permanent acquisition value.
    cost = variant["energy_cost"]
    if cost >= 0:
        f["printed_cost"] = cost
        f["high_cost_cards"] = int(cost >= 2)
    elif cost == -1:
        f["x_cost_cards"] = 1
    f["attacks"] = int(entry["type"] == "ATTACK")
    f["skills"] = int(entry["type"] == "SKILL")
    f["setup_cards"] = int(entry["type"] == "POWER")
    f["dead_draws"] = int(entry["type"] in {"CURSE", "STATUS"})
    f["self_exhaust"] = int(recipe.exhausts(upgraded))
    f["ethereal"] = int(recipe.ethereal(upgraded))
    f["innate"] = int(recipe.innate(upgraded))
    if recipe.intrinsic_effects or recipe.resident_rule or recipe.self_copy:
        unknown.append("passive_or_resident_valuation")
    temporary_strength = any(s.primitive is E.APPLY_LOSE_STRENGTH for s in recipe.effects)

    fixed = {
        E.GAIN_BLOCK: "block", E.DRAW_CARDS: "draw", E.GAIN_ENERGY: "energy",
        E.HEAL: "heal", E.LOSE_HP: "hp_loss", E.APPLY_WEAK: "weak",
        E.APPLY_VULNERABLE: "vulnerable", E.ENEMY_STRENGTH_LOSS: "strength_reduction",
        E.GAIN_INTANGIBLE: "intangible", E.GAIN_ARTIFACT: "artifact",
        E.SELF_VULNERABLE: "self_vulnerable", E.APPLY_NO_DRAW: "no_draw",
        E.APPLY_NO_BLOCK: "no_block", E.RETAIN_BLOCK: "retain_block",
        E.END_TURN_BLOCK: "passive_block", E.CURRENT_BLOCK_GAIN: "block_multiplier",
        E.CURRENT_BLOCK_DAMAGE: "block_payoff", E.GAIN_CURRENT_STRENGTH: "strength_multiplier",
        E.ON_HP_LOSS_COST_REDUCTION: "hp_loss_cost_payoff",
    }
    attacks = {E.DIRECT_DAMAGE, E.AREA_DAMAGE, E.ATTACK_SEQUENCE, E.FATAL_ATTACK,
               E.AREA_LIFESTEAL, E.DYNAMIC_DAMAGE}
    exhaust = {E.EXHAUST_RANDOM_HAND, E.EXHAUST_CHOSEN_HAND, E.EXHAUST_UP_TO_HAND,
               E.EXHAUST_HAND_GROUP}
    for step in recipe.effects:
        primitive = step.primitive
        amount = step.amount(upgraded) if step.amount_source is AmountSource.FIXED else 1
        prefix = "conditional_" if step.condition else ""
        if primitive in fixed:
            f[prefix + fixed[primitive]] += amount
            if primitive is E.LOSE_HP and amount:
                f["hp_loss_sources"] += 1
        elif primitive in attacks:
            hits = 1
            conditional = bool(step.condition) or CardTrait.PUBLIC_HAND_PLAYABILITY in recipe.traits
            if step.attack_sequence:
                if step.attack_sequence.count_source is AttackCountSource.ENERGY_ON_USE:
                    conditional = True  # One-energy potential, NOT a predicted X value.
                else:
                    hits = step.attack_sequence.hits(upgraded)
                f["strength_hits"] += hits * step.attack_sequence.strength_multiplier(upgraded)
            else:
                f["strength_hits"] += 1
            if step.dynamic_damage:
                if step.dynamic_damage.source is DamageScaling.PERSISTENT_CARD_BASE_DAMAGE:
                    # Both backends export master-deck misc, which is public damage.
                    damage = card.get("base_damage", card.get("misc"))
                    if type(damage) is int and 15 <= damage <= 32767:
                        amount = damage
                    else:
                        unknown.append("missing_persistent_damage")
                        amount = 0
                else:
                    conditional = True
                    unknown.append("dynamic_damage_valuation")
            key = "conditional_damage" if conditional else "damage"
            f[key] += amount * hits
            if step.target_for(upgraded) is EffectTarget.ALL_ENEMIES:
                f["conditional_area" if conditional else "area"] += amount * hits
            if primitive is E.AREA_LIFESTEAL:
                f["lifesteal"] += amount
            if step.fatal_reward:
                f["fatal_payoff"] += 1
                unknown.append("fatal_reward_valuation")
        elif primitive is E.GAIN_STRENGTH:
            f[prefix + ("temporary_strength" if temporary_strength else "strength")] += amount
        elif primitive is E.APPLY_LOSE_STRENGTH:
            pass  # Already represented as temporary rather than permanent Strength.
        elif primitive in exhaust:
            f["exhaust_sources"] += int(amount > 0 or step.hand_exhaust is not None)
            if step.hand_exhaust:
                unknown.append("hand_group_payoff_valuation")
        elif primitive is E.REGISTER_EXHAUST_TRIGGER:
            key = ("exhaust_draw_payoff" if step.exhaust_trigger.effect is ExhaustTriggerEffect.DRAW
                   else "exhaust_block_payoff")
            f[key] += amount
        elif primitive is E.ON_EXHAUST_ENERGY:
            f["exhaust_energy_payoff"] += amount
        elif primitive is E.REGISTER_EVENT_TRIGGER:
            key = {
                CombatEvent.SELF_HP_LOST: "hp_loss_payoff",
                CombatEvent.ATTACK_PLAYED: "attack_play_payoff",
                CombatEvent.BLOCK_GAINED: "block_event_payoff",
            }.get(step.event_trigger.event)
            if key:
                f[key] += amount
            else:
                unknown.append(primitive.value)
        elif primitive is E.REGISTER_TURN_TRIGGER:
            for effect in step.turn_trigger.effects:
                key = {
                    TurnEffectKind.DRAW: "delayed_draw", TurnEffectKind.ENERGY: "delayed_energy",
                    TurnEffectKind.STRENGTH: "delayed_strength",
                    TurnEffectKind.SELF_HP_LOSS: "delayed_hp_loss",
                    TurnEffectKind.NON_ATTACK_AREA_DAMAGE: "delayed_area",
                }.get(effect.kind)
                if key:
                    f[key] += effect.amount(upgraded)
                    if effect.kind is TurnEffectKind.SELF_HP_LOSS:
                        f["delayed_hp_loss_sources"] += 1
                else:
                    unknown.append(effect.kind.value)
        elif primitive is E.GENERATE_CARD:
            generated = catalog.lookup("card", step.generated_card_id)
            if generated and generated["type"] in {"STATUS", "CURSE"}:
                f["generated_dead_draws"] += amount
            else:
                unknown.append(primitive.value)
        elif primitive is E.REGISTER_CORRUPTION:
            f["skill_exhaust_engine"] += 1
        else:
            unknown.append(primitive.value)
    # These are described but intentionally NOT monetized in the first scorer.
    # Do not imply that a reviewed execution recipe grants complete strategic value.
    unvalued = {"artifact", "conditional_damage", "conditional_area", "conditional_block",
                "conditional_energy", "fatal_payoff", "ethereal", "innate", "x_cost_cards"}
    unknown.extend(f"unvalued_feature:{key}" for key in sorted(unvalued) if f[key])
    return CardCapabilities(identifier, upgrades,
                            MappingProxyType(dict(sorted((k, v) for k, v in f.items() if v))),
                            tuple(sorted(set(unknown))))


def profile_cards(cards: Sequence[Mapping], catalog: RuleCatalog | None = None) -> DeckProfile:
    catalog = catalog or RuleCatalog.bundled()
    if not isinstance(cards, (list, tuple)) or any(not isinstance(c, Mapping) for c in cards):
        raise ValueError("master deck must be a public card sequence")
    described = sorted((card_capabilities(card, catalog) for card in cards),
                       key=lambda c: (c.identifier, c.upgrades, tuple(c.features.items()), c.unknown))
    totals: Counter[str] = Counter()
    for card in described:
        totals.update(card.features)
    unknown = tuple(sorted({f"{c.identifier}:{reason}" for c in described for reason in c.unknown}))
    return DeckProfile(tuple(described), MappingProxyType(dict(sorted(totals.items()))), unknown)


def profile_observation(observation: Mapping, catalog: RuleCatalog | None = None) -> DeckProfile:
    inventory = observation.get("inventory")
    if not isinstance(inventory, Mapping) or "deck" not in inventory:
        raise ValueError("public master deck is missing")
    return profile_cards(inventory["deck"], catalog)


def profile_value(profile: DeckProfile, *, hp_fraction: float = 1.0, act: int = 1) -> dict[str, float]:
    """Versioned hand-set utility components, NOT expected HP, DPS or win rate.

    Density penalizes dilution; saturated capacities give diminishing returns.
    Source/payoff products express potential only, never guaranteed co-draws.
    Printed costs remain conservative: no relic energy or conditional discount
    is invented. This small initial model explicitly leaves richer effects open.
    """
    f = Counter(profile.features)
    n = max(1, len(profile.cards))
    cap = lambda value, scale: max(0.0, value) / (scale + max(0.0, value))
    danger = 1.0 + (1.0 - hp_fraction)
    strength = f["strength"] + .4 * f["conditional_strength"] + .7 * f["delayed_strength"]
    exhaust_sources = (f["exhaust_sources"] + .4 * f["self_exhaust"]
                       + min(f["skills"], 5) * f["skill_exhaust_engine"])
    hp_sources = f["hp_loss_sources"] + .5 * f["delayed_hp_loss_sources"]
    draw = f["draw"] + .4 * f["conditional_draw"] + .5 * f["delayed_draw"]
    energy = f["energy"] + .5 * f["delayed_energy"]
    return {
        "immediate_damage_density": (14.4 if act == 1 else 12.0) * cap(f["damage"] / n, 6),
        "immediate_block_density": 16.8 * danger * cap(f["block"] / n, 5),
        "area_coverage": 3.0 * cap(f["area"], 10),
        "draw_consistency": 3.0 * cap(draw, 4) / (1 + f["high_cost_cards"] / n),
        "energy_curve": 3.0 * cap(energy, 3) * cap(f["printed_cost"] + draw, n),
        "strength_synergy": 4.0 * cap(strength, 3) * cap(f["strength_hits"], 5),
        "strength_growth": 2.0 * cap(strength, 3) * cap(f["strength_multiplier"], 1),
        "temporary_strength": 1.0 * cap(f["temporary_strength"], 3) * cap(f["strength_hits"], 5),
        "exhaust_synergy": 4.0 * cap(exhaust_sources, 3) * cap(f["exhaust_block_payoff"] / 3 + f["exhaust_draw_payoff"] + f["exhaust_energy_payoff"], 1),
        "hp_loss_synergy": 2.5 * cap(hp_sources, 2) * cap(f["hp_loss_payoff"] + f["hp_loss_cost_payoff"], 2),
        "block_synergy": 3.0 * cap(f["block"] + f["passive_block"], 20) * cap(f["block_payoff"] + f["block_event_payoff"] + f["retain_block"] + f["block_multiplier"], 2),
        "attack_play_synergy": 2.0 * cap(f["attacks"], 5) * cap(f["attack_play_payoff"], 3),
        "sustain": danger * 3.0 * cap(f["heal"] + .5 * f["lifesteal"], 5),
        "mitigation": danger * 3.0 * cap(f["weak"] + f["strength_reduction"] + 3 * f["intangible"], 4),
        "vulnerable": 2.0 * cap(f["vulnerable"], 3) * cap(f["damage"], 25),
        "passive_defense": danger * 2.0 * cap(f["passive_block"], 4),
        "delayed_damage": 2.0 * cap(f["delayed_area"], 8),
        "cost_pressure": -2.5 * f["printed_cost"] / n / (1 + .15 * energy),
        "setup_pressure": -1.5 * f["setup_cards"] / n,
        "hp_cost": -danger * (f["hp_loss"] + f["delayed_hp_loss"]) / n,
        "draw_pollution": -3.0 * (f["dead_draws"] + .5 * f["generated_dead_draws"]) / n,
        "restrictions": -(f["self_vulnerable"] + f["no_draw"] + f["no_block"]) / n,
    }
