"""Opt-in run-learning inputs; the sealed v2 encoder remains unchanged.

Static card descriptors are potential mechanisms, NOT predicted damage, a rule
score, or a joint-draw assumption. Live combat costs remain live public fields.
The supplemental native export is deliberately separate from battle snapshots.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import json
import math
from types import MappingProxyType

from .card_recipes import CARD_RECIPE_VERSION, DEFAULT_CARD_RECIPES, EffectPrimitive
from .deck_profile import card_capabilities
from .policy_api import PolicyInput
from .public_encoding import EncodedDecision, FeatureTable, PublicEncoder, SCALES, _key
from .rule_coverage import _validate_input


VERSION = "public-run-learning-inputs-v1"
RUN_PUBLIC_ABI = "public-run-supplement-v1"
CAPABILITIES = tuple("""area artifact attack_play_payoff attacks block block_event_payoff
block_multiplier block_payoff conditional_area conditional_damage conditional_draw
conditional_energy conditional_strength damage dead_draws delayed_area delayed_draw
delayed_energy delayed_hp_loss delayed_hp_loss_sources delayed_strength draw energy
ethereal exhaust_block_payoff exhaust_draw_payoff exhaust_energy_payoff exhaust_sources
fatal_payoff generated_dead_draws heal high_cost_cards hp_loss hp_loss_cost_payoff
hp_loss_payoff hp_loss_sources innate intangible lifesteal no_block no_draw passive_block
printed_cost retain_block self_exhaust self_vulnerable setup_cards skill_exhaust_engine
skills strength strength_hits strength_multiplier strength_reduction temporary_strength
vulnerable weak x_cost_cards""".split())
PRIMITIVES = tuple(sorted(p.value for p in EffectPrimitive))
BOTTLES = ("bottled_attack", "bottled_skill", "bottled_power")
EXTRA_COLUMNS = {
    "cards": (*CAPABILITIES, *("mechanism:" + p for p in PRIMITIVES),
              "recipe_reviewed", "descriptor_incomplete", "self_retain", *BOTTLES),
    "global": ("selected_count", "selection_count", "selection_remaining"),
}


@dataclass(frozen=True, slots=True)
class RunEncodedDecision(EncodedDecision):
    extra_tables: Mapping[str, FeatureTable]


def _number(value):
    if value is None:
        return 0.0, False
    if type(value) not in (int, float, bool) or not math.isfinite(value):
        raise ValueError("invalid supplemental public number")
    return float(value), True


def _copy_public(value):
    if isinstance(value, Mapping):
        return {k: _copy_public(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_copy_public(v) for v in value]
    return value


def _table(columns, rows):
    cells = [tuple(_number(row.get(c)) for c in columns) for row in rows]
    return FeatureTable(columns, tuple(tuple(v for v, _ in r) for r in cells),
                        tuple(tuple(k for _, k in r) for r in cells))


def validate_run_public(value, deck, catalog):
    """Strict public-only side channel. No generic backend/audit passthrough."""
    if value is None:
        return None
    if not isinstance(value, Mapping) or set(value) != {"version", "deck"} or value["version"] != RUN_PUBLIC_ABI:
        raise ValueError("unknown public run supplement")
    cards = value["deck"]
    if not isinstance(cards, (list, tuple)) or len(cards) != len(deck):
        raise ValueError("stale public run supplement deck")
    for actual, reference in zip(cards, deck, strict=True):
        if not isinstance(actual, Mapping) or set(actual) != {"id", "upgrades", "misc", *BOTTLES}:
            raise ValueError("unexpected public run card fields")
        if (catalog.canonical_id("card", actual["id"]) != catalog.canonical_id("card", reference.get("id"))
                or actual["upgrades"] != reference.get("upgrades", 0)
                or actual["misc"] != reference.get("misc", 0)):
            raise ValueError("stale public run supplement card")
        if any(type(actual[k]) is not bool for k in BOTTLES):
            raise ValueError("bottled relation must be boolean")
    if any(sum(c[k] for c in cards) > 1 for k in BOTTLES):
        raise ValueError("duplicate bottled assignment")
    return cards


class RunLearningEncoder(PublicEncoder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.legacy_schema_id = self.schema_id
        # Catalogue values, not just its vocabulary, affect these model inputs.
        catalogue = list(self.catalog.entries("card"))
        self._card_ids = {self._tokens["card"][_key(e["id"])]: e["id"] for e in catalogue}
        schema = {"version": VERSION, "base": self.legacy_schema_id,
                  "extra_columns": EXTRA_COLUMNS, "recipes": CARD_RECIPE_VERSION,
                  "catalogue": catalogue, "run_public": RUN_PUBLIC_ABI}
        self.schema_id = hashlib.sha256(json.dumps(schema, sort_keys=True, default=dict).encode()).hexdigest()

    def describe(self):
        return dict(super().describe(), version=VERSION, schema_id=self.schema_id,
                    legacy_schema_id=self.legacy_schema_id, extra_columns=EXTRA_COLUMNS,
                    recipe_version=CARD_RECIPE_VERSION, run_public_abi=RUN_PUBLIC_ABI,
                    static_semantics="potential_not_predicted_outcome", training_ready=False)

    def _enrich_card(self, card):
        entry = self.catalog.lookup("card", card.get("id"))
        upgrades = card.get("upgrades", 0)
        if type(upgrades) is not int or upgrades < 0:
            raise ValueError("invalid run card upgrade count")
        if entry is None:
            return
        variant = entry["variants"][int(upgrades > 0)]
        card.setdefault("type", entry["type"])
        # Only permanent deck/offer rows: NEVER substitute printed costs for
        # missing live combat costs, or manufacture cost_for_turn.
        for target, source in (("base_cost", "energy_cost"), ("exhausts", "exhaust"),
                               ("ethereal", "ethereal"), ("has_target", "targets_enemy")):
            card.setdefault(target, variant[source])
        identifier = entry["id"]
        if identifier == "RITUAL_DAGGER":
            if type(card.get("misc")) is int and card["misc"] >= 15:
                card.setdefault("base_damage", card["misc"])
        elif identifier != "SEARING_BLOW" and variant["base_damage"] >= 0:
            card.setdefault("base_damage", variant["base_damage"])

    def encode(self, public: PolicyInput, *, run_public=None):
        _validate_input(public)  # Reject forbidden data before copying/enriching.
        observation = _copy_public(public.observation)
        deck = observation.get("inventory", {}).get("deck", [])
        bottled = validate_run_public(run_public, deck, self.catalog)
        for card in deck:
            self._enrich_card(card)
        screen = observation.get("screen", {})
        for group in ("decision_options", "card_reward_options"):
            for option in screen.get(group, ()):
                if isinstance(option.get("card"), dict):
                    self._enrich_card(option["card"])
                event = option.get("event_choice", {})
                # This exact display tells us the operation, not its amount.
                # Restrict to the complete one-clause offer so order is unambiguous.
                if (event.get("event_name") == "World of Goop"
                        and event.get("unknown_clauses") == ["Lose some gold"]
                        and not event.get("display_order")):
                    cost = {"kind": "lose_gold", "amount_source": "unspecified_in_display"}
                    event.update(version=VERSION, status="parsed", unknown_clauses=[],
                                 display_order=[cost], costs=[cost], effects=[])
        base = super().encode(PolicyInput(observation, public.legal_actions))
        cards = base.tables["cards"]
        columns = cards.columns
        extra_cards = []
        for values, known in zip(cards.values, cards.known, strict=True):
            def raw(name):
                i = columns.index(name)
                return values[i] * SCALES.get(name, 1) if known[i] else None
            token, pile = raw("card_id"), raw("pile_id")
            identifier = self._card_ids.get(token, "unknown")
            entry = self.catalog.lookup("card", identifier)
            upgrades = raw("upgrades")
            row = {}
            recipe = DEFAULT_CARD_RECIPES.lookup(identifier)
            reviewed = entry is not None and recipe is not None and upgrades is not None
            row["recipe_reviewed"] = reviewed
            if reviewed:
                card = {"id": identifier, "upgrades": int(round(upgrades))}
                if raw("misc") is not None:
                    card["misc"] = int(round(raw("misc")))
                capabilities = card_capabilities(card, self.catalog)
                row.update({k: capabilities.features.get(k, 0) for k in CAPABILITIES})
                # Dynamic coefficients/placeholders must not look like known
                # damage. Identity + mechanism flags remain available.
                if any(s in capabilities.unknown for s in ("dynamic_damage_valuation", "missing_persistent_damage")):
                    for k in ("damage", "area", "conditional_damage", "conditional_area"):
                        row[k] = None
                effects = {step.primitive.value for step in recipe.effects}
                row.update({"mechanism:" + p: p in effects for p in PRIMITIVES})
                row["descriptor_incomplete"] = bool(capabilities.unknown)
                row["self_retain"] = entry["variants"][int(upgrades > 0)]["self_retain"]
            deck_token = self._tokens["pile"]["deck"]
            if pile == deck_token and bottled is not None:
                row.update({k: bottled[len(extra_cards)][k] for k in BOTTLES})
            extra_cards.append(row)
        context = screen.get("decision_context", {})
        selection = {k: context.get(k) for k in ("selected_count", "selection_count")}
        selected, count = selection.values()
        if selected is not None or count is not None:
            if type(selected) is not int or type(count) is not int or not 0 <= selected <= count:
                raise ValueError("invalid public selection progress")
            selection["selection_remaining"] = count - selected
        extras = MappingProxyType({"cards": _table(EXTRA_COLUMNS["cards"], extra_cards),
                                   "global": _table(EXTRA_COLUMNS["global"], [selection])})
        gaps = set(base.gaps)
        if bottled is None:
            gaps.add("bottled_assignment_not_exported")
        return RunEncodedDecision(base.schema_id, base.tables, base.legal_actions, base.legal_mask,
                                  base.scope_mask, base.unknown_categories, tuple(sorted(gaps)), extras)


def legacy_view(decision):
    """Discard supplements explicitly; used only by the shared tensor adapter."""
    return EncodedDecision(decision.schema_id, decision.tables, decision.legal_actions,
                           decision.legal_mask, decision.scope_mask, decision.unknown_categories, decision.gaps)
