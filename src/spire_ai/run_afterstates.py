"""Opt-in, public-only local afterstates; not a live policy or a full room replay.

An afterstate resolves ONE declared local choice. Its old menu is consumed and
the next menu is UNKNOWN, not empty/terminal. In particular, removing a card is
not an entire shop visit, and upgrading is not all campfire/relic settlement.
No hidden simulator state, seed, outcome label or dataset identity is accepted.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from functools import lru_cache
import hashlib
import json
from types import MappingProxyType

from .actions import Action, ActionKind
from .expert_cards import EGG_TYPES
from .policy_api import (PolicyInput, _freeze_public_value, _thaw_public_value,
                         _reject_forbidden_policy_keys)
from .public_encoding import FeatureTable
from .rule_catalog import RuleCatalog
from .rule_coverage import _validate_input
from .run_learning_encoding import RunLearningEncoder, validate_run_public, BOTTLES


VERSION = "public-local-run-afterstates-v1"
PHASES = ("decision", "local_effect_resolved")
UNREMOVABLE = frozenset(("ASCENDERS_BANE", "NECRONOMICURSE", "CURSE_OF_THE_BELL"))
RAW_HISTORY_FIELDS = frozenset(("seed_played", "play_id", "master_deck", "damage_taken",
    "current_hp_per_floor", "gold_per_floor", "floor_reached", "killed_by", "victory",
    "score", "source_sha256", "group", "audit_only_future_outcome"))


def _reject_raw_history(value):
    """Raw metrics wrappers/outcomes are not public decision observations.

The general no-SL boundary already rejects native RNG keys. Also refuse legacy
history metadata here, even though the numeric encoder would otherwise ignore it.
"""
    if isinstance(value, Mapping):
        for key, item in value.items():
            if str(key).lower() in RAW_HISTORY_FIELDS:
                raise ValueError("raw history metadata in public local state: " + str(key))
            _reject_raw_history(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _reject_raw_history(item)


@lru_cache(maxsize=1)
def _catalog():
    return RuleCatalog.bundled()


class LocalEffectGap(ValueError):
    """A declared operation cannot be resolved from this public observation."""


def _int(value, name, *, minimum=0, maximum=1000000):
    if type(value) is not int or not minimum <= value <= maximum:
        raise ValueError("invalid public integer: " + name)
    return value


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


@dataclass(frozen=True, slots=True)
class PublicRunState:
    public: PolicyInput
    phase: str
    relic_order_known: bool
    run_public: Mapping | None
    opportunities_known: bool

    def fingerprint(self):
        return _digest(dict(version=VERSION, public=self.public.as_dict(), phase=self.phase,
            relic_order_known=self.relic_order_known, opportunities_known=self.opportunities_known,
            run_public=_thaw_public_value(self.run_public)))


def capture(public, *, relic_order_known, run_public=None):
    """Own an immutable snapshot. Caller must declare historical order masking.

This entry deliberately rejects a seed instead of silently sanitizing a raw
environment. Call the existing policy-input boundary first for live states.
"""
    _validate_input(public)
    _reject_raw_history(public.observation)
    for action in public.legal_actions:
        if (not isinstance(action, Action) or action.kind in {
                ActionKind.START, ActionKind.WAIT, ActionKind.REQUEST_STATE} or
                any(getattr(action, k) is not None for k in ("seed", "character", "ascension", "wait_frames"))):
            raise ValueError("game decisions only, no episode control metadata")
        for key in ("choice_index", "card_index", "target_index", "potion_index"):
            if getattr(action, key) is not None:
                _int(getattr(action, key), key)
    if type(relic_order_known) is not bool:
        raise TypeError("relic order knowledge must be explicit")
    if "combat" in public.observation:
        raise LocalEffectGap("outside_combat_states_only")
    obs = public.as_dict()["observation"]
    inventory = obs.get("inventory", {})
    if not isinstance(inventory, dict) or "deck" not in inventory or "relics" not in inventory:
        raise LocalEffectGap("complete_public_inventory_required")
    if not isinstance(inventory["deck"], list) or not isinstance(inventory["relics"], list):
        raise ValueError("invalid inventory collections")
    if len(inventory["deck"]) > 1000 or len(inventory["relics"]) > 1000:
        raise ValueError("bounded local inventory required")
    for c in inventory["deck"]:
        if not isinstance(c, dict) or not isinstance(c.get("id"), str):
            raise ValueError("invalid public card")
        _int(c.get("upgrades"), "upgrades", maximum=10000)
    for r in inventory["relics"]:
        if not isinstance(r, dict) or not isinstance(r.get("id"), str):
            raise ValueError("invalid public relic")
    player = obs.get("player", {})
    for field in ("hp", "max_hp", "gold"):
        if player.get(field) is not None:
            _int(player[field], field, minimum=int(field == "max_hp"))
    if player.get("hp") is not None and player.get("max_hp") is not None:
        if player["hp"] > player["max_hp"]:
            raise ValueError("HP exceeds maximum")
    if run_public is not None:
        # No arbitrary side channels. Reuse the reviewed public bottle ABI.
        validate_run_public(run_public, inventory["deck"], _catalog())
    # Check JSON finiteness and detach every descendant, including direct callers.
    _digest(obs)
    owned = PolicyInput(_freeze_public_value(obs), tuple(public.legal_actions))
    return PublicRunState(owned, "decision", relic_order_known,
        _freeze_public_value(_thaw_public_value(run_public)), bool(public.legal_actions))


class RunStateEncoder:
    """Same reviewed entity features, explicit state phase, no fake legal action."""
    def __init__(self, base=None):
        self.base = base or RunLearningEncoder()
        self.catalog = self.base.catalog
        self.schema_id = _digest(dict(version=VERSION, base=self.base.schema_id,
                                     phases=PHASES))

    def encode(self, state):
        if not isinstance(state, PublicRunState) or state.phase not in PHASES:
            raise TypeError("versioned public run state required")
        _validate_input(state.public)
        _reject_raw_history(state.public.observation)
        encoded = self.base.encode(state.public, run_public=state.run_public)
        if not state.relic_order_known:
            table = encoded.tables["relics"]
            i = table.columns.index("order")
            values, known = [list(r) for r in table.values], [list(r) for r in table.known]
            for row, mask in zip(values, known, strict=True):
                row[i], mask[i] = 0, False
            tables = dict(encoded.tables)
            tables["relics"] = FeatureTable(table.columns, tuple(map(tuple, values)),
                                            tuple(map(tuple, known)))
            encoded = replace(encoded, tables=MappingProxyType(tables))
        return EncodedRunState(self.schema_id, encoded, PHASES.index(state.phase),
                               state.opportunities_known)


@dataclass(frozen=True, slots=True)
class EncodedRunState:
    schema_id: str
    entities: object
    phase: int
    opportunities_known: bool


@dataclass(frozen=True, slots=True)
class LocalCandidate:
    # Mapping back to choices is audit/decoder metadata, NEVER a model feature.
    operation: str
    target_index: int | None
    action: Action | None
    state: PublicRunState
    transition: LocalTransition


@dataclass(frozen=True, slots=True)
class LocalTransition:
    """A declared local operation, not an entire room or an outcome target.

Known deltas are separate from unknown absolute resources. Requests are NOT
guaranteed realized changes (healing can be capped/blocked; max-HP loss clamped).
Target indexes only identify public card instances; never scalar model inputs.
"""
    before: PublicRunState
    after: PublicRunState
    operation: str
    target_index: int | None
    effects: Mapping


def _relics(state, catalog):
    result = set()
    for row in state.public.observation["inventory"]["relics"]:
        item = catalog.lookup("relic", row["id"])
        if item is None:
            raise LocalEffectGap("unknown_relic_effect:" + row["id"])
        result.add(item["id"])
    return result


def _card(c, catalog):
    _reject_forbidden_policy_keys(c, reject_seeds=True)
    _reject_raw_history(c)
    item = catalog.lookup("card", c.get("id"))
    if item is None or item["color"] not in {"RED", "COLORLESS", "CURSE"}:
        raise LocalEffectGap("card_outside_reviewed_ironclad_scope")
    result = {"id": item["id"], "upgrades": _int(c.get("upgrades"), "upgrades", maximum=10000)}
    # Permanent fields only; discard stale printed/live costs on an upgrade.
    if c.get("misc") is not None:
        result["misc"] = _int(c["misc"], "misc")
    return result, item


def _upgrade_alias(updated, original, *, native_acquisition=False):
    # Native Card::getUpgraded() and misc expose the SAME Searing Blow
    # counter. This is a representation alias, not a second upgrade effect.
    # Preserve unknown misc in historical rows; a new native acquisition has
    # a known counter. Other cards (notably Ritual Dagger) own different misc.
    if updated["id"] != "SEARING_BLOW":
        return
    if "misc" in original and original["misc"] != original["upgrades"]:
        raise LocalEffectGap("inconsistent_native_searing_blow_counter")
    if "misc" in original or native_acquisition:
        updated["misc"] = updated["upgrades"]


def _heal(player, amount, relics):
    if "MARK_OF_THE_BLOOM" in relics:
        return
    hp, maximum = player.get("hp"), player.get("max_hp")
    if hp is None or maximum is None:
        player.pop("hp", None)  # An unknown cap cannot become a fabricated HP.
    else:
        player["hp"] = min(maximum, hp + amount)


def local_transition(state, operation, *, target_index=None, card=None, gold_cost=0, catalog=None):
    """Resolve a local primitive, not a claim that a live menu offers this action.

Removal's explicit cost, if any, is charged here. Callers MUST NOT charge a
previously paid fee again. Whole-shop/whole-event/campfire callbacks are outside
this API. Use actual observed successors for full decision-boundary evaluation.
"""
    if not isinstance(state, PublicRunState) or state.phase != "decision":
        raise ValueError("local effect requires an unresolved decision snapshot")
    _validate_input(state.public)
    _reject_raw_history(state.public.observation)
    catalog = catalog or _catalog()
    relics = _relics(state, catalog)
    obs = state.public.as_dict()["observation"]
    deck, player = obs["inventory"]["deck"], obs.setdefault("player", {})
    # Produced by the SAME resolver as the afterstate, not a second rules engine.
    effects = dict(hp_delta=0, max_hp_delta=0, gold_delta=0,
                   healing_requested=0, max_hp_loss_requested=0)
    info = _thaw_public_value(state.run_public)
    cost = _int(gold_cost, "gold_cost")
    if cost and operation != "remove":
        raise ValueError("only declared removal fees supported")
    if cost:
        if player.get("gold") is None:
            raise LocalEffectGap("unknown_removal_affordability")
        if player["gold"] < cost:
            raise ValueError("unaffordable removal")
        if "MAW_BANK" in relics:
            raise LocalEffectGap("paid_removal_requires_shop_context_for_maw_bank")
        player["gold"] -= cost
        effects["gold_delta"] = -cost
    if operation in {"upgrade", "remove"}:
        _int(target_index, "target_index", maximum=len(deck)-1)
        if card is not None:
            raise ValueError("deck operation must not supply an extra card")
        updated, entry = _card(deck[target_index], catalog)
        if operation == "upgrade":
            if entry["type"] not in {"ATTACK", "SKILL", "POWER"} or (
                    updated["upgrades"] and updated["id"] != "SEARING_BLOW"):
                raise ValueError("card cannot upgrade")
            updated["upgrades"] += 1
            _upgrade_alias(updated, deck[target_index])
            deck[target_index] = updated
            if info is not None:
                info["deck"][target_index]["upgrades"] = updated["upgrades"]
                if "misc" in updated:
                    info["deck"][target_index]["misc"] = updated["misc"]
        else:
            if updated["id"] in UNREMOVABLE:
                raise ValueError("card cannot be purged")
            bottled_relics = relics & {"BOTTLED_FLAME", "BOTTLED_LIGHTNING", "BOTTLED_TORNADO"}
            if bottled_relics and info is None:
                raise LocalEffectGap("unknown_bottled_removal_eligibility")
            if info is not None and any(info["deck"][target_index][k] for k in BOTTLES):
                raise ValueError("bottled card cannot be purged")
            if updated["id"] == "PARASITE":
                # Keep boundary arithmetic scoped: native clamps differ at <=3.
                maximum = player.get("max_hp")
                effects.update(max_hp_loss_requested=3, hp_delta=None,
                               max_hp_delta=-3 if maximum is not None else None)
                if maximum is not None and maximum <= 3:
                    raise LocalEffectGap("parasite_minimum_max_hp_boundary")
                if maximum is not None:
                    player["max_hp"] = maximum - 3
                    if player.get("hp") is not None:
                        player["hp"] = min(player["hp"], maximum - 3)
                else:
                    player.pop("hp", None)
            del deck[target_index]
            if info is not None:
                del info["deck"][target_index]
    elif operation == "take_card":
        if target_index is not None or not isinstance(card, Mapping):
            raise ValueError("take requires a card, not a deck index")
        updated, entry = _card(card, catalog)
        if entry["type"] not in {"ATTACK", "SKILL", "POWER"}:
            raise LocalEffectGap("curse_and_status_acquisition_not_local_reward")
        if not updated["upgrades"] and any(EGG_TYPES.get(r) == entry["type"] for r in relics):
            updated["upgrades"] = 1
        _upgrade_alias(updated, card, native_acquisition=info is not None)
        if "CERAMIC_FISH" in relics and "ECTOPLASM" not in relics:
            effects["gold_delta"] = 9
            if player.get("gold") is not None:
                player["gold"] += 9
            if "BLOODY_IDOL" in relics:
                effects.update(healing_requested=5, hp_delta=0 if "MARK_OF_THE_BLOOM" in relics else None)
                _heal(player, 5, relics)
        deck.append(updated)
        if info is not None:
            info["deck"].append(dict(updated, misc=updated.get("misc", 0), **dict.fromkeys(BOTTLES, False)))
    elif operation == "singing_bowl":
        if "SINGING_BOWL" not in relics:
            raise ValueError("Singing Bowl absent")
        if target_index is not None or card is not None:
            raise ValueError("Bowl must not modify a card")
        if player.get("max_hp") is not None:
            player["max_hp"] += 2
        effects.update(max_hp_delta=2, healing_requested=2,
                       hp_delta=0 if "MARK_OF_THE_BLOOM" in relics else None)
        _heal(player, 2, relics)
    elif operation == "skip":
        if target_index is not None or card is not None:
            raise ValueError("skip must not modify a card")
    else:
        raise LocalEffectGap("unknown_local_operation:" + str(operation))
    # Consume the old menu, including card offers, event text and selection
    # counters. Do NOT invent a MAP menu, a fresh opportunity or a terminal.
    obs["screen"] = {}
    result = capture(PolicyInput(obs, ()), relic_order_known=state.relic_order_known, run_public=info)
    after = replace(result, phase="local_effect_resolved", opportunities_known=False)
    previous = state.public.observation.get("player", {})
    for key in ("hp", "max_hp", "gold"):
        if previous.get(key) is not None and player.get(key) is not None:
            actual = player[key] - previous[key]
            if effects[key+"_delta"] is not None and effects[key+"_delta"] != actual:
                raise ValueError("declared local effect/afterstate disagreement: " + key)
            effects[key+"_delta"] = actual
    return LocalTransition(state, after, operation, target_index, MappingProxyType(effects))


def local_effect(state, operation, **kwargs):
    """Compatibility state-only view; numeric/state semantics remain unchanged."""
    return local_transition(state, operation, **kwargs).after


def card_reward_candidates(state, catalog=None):
    """Expand every actual structured reward option, preserving decoder order."""
    catalog = catalog or _catalog()
    if state.phase != "decision" or state.public.observation.get("screen", {}).get("type") != "CARD_REWARD":
        raise ValueError("card reward decision required")
    screen = state.public.observation["screen"]
    options = screen.get("card_reward_options", screen.get("decision_options"))
    if options is None:
        raise LocalEffectGap("structured_reward_options_required")
    if len(state.public.legal_actions) != len(set(state.public.legal_actions)):
        raise ValueError("duplicate legal reward action")
    mapped = {}
    for option in options:
        action = Action.from_dict(option["action"])
        if action in mapped:
            raise ValueError("duplicate reward metadata")
        mapped[action] = option
    if set(mapped) != set(state.public.legal_actions) or not mapped:
        raise ValueError("reward options/legality mismatch")
    result = []
    for action in state.public.legal_actions:
        option = mapped[action]
        kind = option.get("kind")
        kwargs = {}
        if kind == "card_reward" and action.kind is ActionKind.CHOOSE:
            operation, kwargs = "take_card", {"card": option["card"]}
        elif kind in {"leave", "skip"} and action.kind in {ActionKind.CANCEL, ActionKind.PROCEED}:
            operation = "skip"
        elif (kind in {"max_hp", "singing_bowl"} and action.kind is ActionKind.CHOOSE
              and type(option.get("amount")) is int and option["amount"] == 2):
            operation = "singing_bowl"
        else:
            raise LocalEffectGap("unreviewed_reward_option:" + str(kind))
        transition = local_transition(state, operation, catalog=catalog, **kwargs)
        result.append(LocalCandidate(operation, None, action, transition.after, transition))
    return tuple(result)
