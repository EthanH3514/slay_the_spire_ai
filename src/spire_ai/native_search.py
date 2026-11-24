"""Public-only transport for the sole production C++ combat implementation.

No EnvironmentState, audit fields, RNG words or raw draw order enter this module.
The native simulation service owns sampled worlds; rule evaluation receives only
public projections. Missing/stale native implementations fail explicitly. The
frozen Python oracle lives under tests/reference and is never a fallback.
"""
from __future__ import annotations

from dataclasses import fields, is_dataclass
from enum import Enum
from functools import lru_cache
import json
from pathlib import Path
import sys
from typing import Mapping

from .card_recipes import DEFAULT_CARD_RECIPES
from .identifiers import canonical_identifier
from .mechanism_evidence import DEFAULT_MECHANISM_EVIDENCE, EvidenceCapability
from .policy_api import PolicyInput
from .public_piles import public_draw_cards
from .rule_catalog import RuleCatalog

NATIVE_SEARCH_ABI = "public-flat-search-v6"
SESSION_SEARCH_ABI = "public-session-search-v2"


def compatible_session(module) -> bool:
    return compatible_versions(module) and getattr(module, "public_session_abi", None) == SESSION_SEARCH_ABI


def compatible_versions(module) -> bool:
    from .combat_search import RULE_VERSION, SEARCH_VERSION, VALUE_VERSION
    from .card_recipes import CARD_RECIPE_VERSION
    return all(getattr(module, key, None) == expected for key, expected in {
        "public_search_abi": NATIVE_SEARCH_ABI, "public_search_rule_version": RULE_VERSION,
        "public_search_algorithm_version": SEARCH_VERSION, "public_search_value_version": VALUE_VERSION,
        "public_search_recipe_version": CARD_RECIPE_VERSION,
    }.items())


def token(value: object, kind: str = "card") -> str:
    return "".join(c for c in (canonical_identifier(value, kind) or "") if c.isalnum())


def _plain(value):
    if isinstance(value, Enum):
        return value.name
    if is_dataclass(value):
        return {f.name: _plain(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, Mapping):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(v) for v in value]
    return value


@lru_cache(maxsize=1)
def _catalog() -> RuleCatalog:
    return RuleCatalog.bundled()


def compatible_rule(rule) -> bool:
    from .combat_search import StarterRulePolicy
    return (type(rule) is StarterRulePolicy and "choose" not in vars(rule)
            and rule.recipes is DEFAULT_CARD_RECIPES
            and type(rule.catalog) is RuleCatalog
            and rule.catalog._entries == _catalog()._entries
            and rule.catalog._aliases == _catalog()._aliases)


@lru_cache(maxsize=1)
def metadata_json() -> str:
    from .card_recipes import DiscoveryGeneration, RandomDrawGeneration, RandomHandGeneration
    from .random_generation import generation_pool
    catalog = _catalog()
    result = {}
    for entry in catalog.entries("card"):
        recipe = DEFAULT_CARD_RECIPES.lookup(entry["id"])
        if recipe is None:
            continue
        row = _plain(recipe)
        row.update(type=entry["type"], variants=_plain(entry["variants"]))
        row["exhaust"] = [recipe.exhausts(False), recipe.exhausts(True)]
        row["ethereal"] = [recipe.ethereal(False), recipe.ethereal(True)]
        row["audited"] = all(DEFAULT_MECHANISM_EVIDENCE.supports(
            "card", entry["id"], capability, imported=True,
        ) for capability in (EvidenceCapability.SEARCH, EvidenceCapability.PUBLIC_HISTORY))
        result[token(entry["id"])] = row
    result["$pools"] = {contract.name: [token(c["id"]) for c in generation_pool(contract, catalog)]
                        for enum in (DiscoveryGeneration, RandomDrawGeneration, RandomHandGeneration)
                        for contract in enum}
    from .potion_contracts import CONTRACT_BY_KEY
    result["$potions"] = {k: c.as_dict() for k, c in CONTRACT_BY_KEY.items()}
    return json.dumps(result, sort_keys=True, separators=(",", ":"))


def public_projection(public: PolicyInput) -> dict:
    """A reduced owned public view; unknown draw order is always discarded.

    Deliberately not a policy observation replacement. Autoplay receives only
    public top candidates, never the unknown pile's realized order.
    The full input and all existing root guards remain on the Python boundary.
    """
    combat = public.observation["combat"]
    from .public_piles import opening_priority_ids
    opening = opening_priority_ids(combat)

    def card(c, hand=False):
        result = {"id": token(c["id"]), "up": int(c.get("upgrades", 0)),
                  "cost": c["cost_for_turn"], "base_cost": c["base_cost"],
                  "free": c["free_to_play_once"], "type": c["type"],
                  "damage": c.get("base_damage", 0), "uid": c.get("unique_id", -1),
                  "special": c.get("special_data", 0)}
        if hand:
            result["playable"] = c["is_playable"]
        return result

    def powers(entity):
        return {token(p["id"], "power"): p["amount"] for p in entity["powers"]}

    p = combat["player"]
    player = {k: p[k] for k in ("current_hp", "max_hp", "block", "energy")}
    player["powers"] = powers(p)
    player["panache_damage"] = next((x.get("damage", 0) for x in p["powers"]
                                       if token(x["id"], "power") == "panache"), 0)
    monsters = []
    for m in combat["monsters"]:
        row = {k: m[k] for k in ("current_hp", "max_hp", "block", "move_base_damage",
                                 "move_hits", "is_gone", "half_dead")}
        row["powers"] = powers(m)
        monsters.append(row)
    draw = [card(c) for c in public_draw_cards(combat)]
    draw.sort(key=lambda c: json.dumps(c, sort_keys=True, separators=(",", ":")))
    from .top_card_autoplay import top_card_candidates
    top = [card(c) for c in top_card_candidates(combat)]
    top.sort(key=lambda c: json.dumps(c, sort_keys=True, separators=(",", ":")))
    return {"player": player, "monsters": monsters,
            "hand": [card(c, True) for c in combat["hand"]], "draw": draw,
            "discard": [card(c) for c in combat["discard_pile"]],
            "exhaust": [card(c) for c in combat["exhaust_pile"]],
            "held": [card(m["stasis_card"]) for m in combat["monsters"] if "stasis_card" in m],
            "relics": sorted(token(r["id"], "relic") for r in public.observation["inventory"]["relics"]),
            "potions": [token(p.get("id", ""), "potion") for p in public.observation["inventory"].get("potions", ())],
            "input": combat["input_state"], "selection": combat.get("card_select_task", ""),
            "top_candidates": top,
            "character": public.observation.get("run", {}).get("character", ""),
            "times_damaged": combat.get("times_damaged", -1),
            "bottom_tracked": combat.get("bottom_knowledge_tracked", False),
            "bomb_schema": combat.get("bomb_state_schema", 0),
            "powers_complete": combat.get("powers_complete", False),
            "resolved": combat.get("effects_resolved", False),
            "limbo_empty": not combat.get("limbo"),
            "power_records": [{"id": token(x["id"], "power"), "amount": x["amount"],
                               **{k: x[k] for k in ("damage", "hp_loss") if k in x}}
                              for x in p["powers"]],
            "selection_candidates": [card(c, "is_playable" in c) for c in combat.get("selection_candidates", ())],
            "selection_options": [list(x) for x in combat.get("selection_options", ())],
            "selection_any": combat.get("selection_any_number", False),
            "selection_max": combat.get("selection_max_count", 0),
            "selection_copies": combat.get("selection_copy_count", 0),
            "selection_pick": combat.get("selection_pick_count", 1),
            "selection_type": combat.get("selection_card_type", "INVALID"),
            "opening_schema": combat.get("opening_order_schema", 0),
            "opening_priority_ids": list(opening),
            "opening_uncertain": combat.get("opening_order_uncertain", False),
            "uncertain_order": bool(combat.get("top_order_uncertain") or combat.get("bottom_order_uncertain")
                                    or combat.get("opening_order_uncertain"))}


def projection_json(public: PolicyInput) -> str:
    return json.dumps(public_projection(public), sort_keys=True, separators=(",", ":"))


def session_projection_json(public: PolicyInput) -> str:
    """Versioned session view: never identify unknown duplicate copies by UID."""
    result = public_projection(public)
    for name in ("hand", "draw", "discard", "exhaust", "held", "top_candidates", "selection_candidates"):
        for card in result[name]:
            card["uid"] = -1
        if name in ("draw", "top_candidates"):
            result[name].sort(key=lambda c: json.dumps(c, sort_keys=True, separators=(",", ":")))
    result["opening_priority_ids"] = []
    return json.dumps(result, sort_keys=True, separators=(",", ":"))


def session_seed_input(public: PolicyInput) -> PolicyInput:
    """ID-invariant proposal namespace, NOT a cache/freshness certificate.

    Root validity still compares the complete native public state/history. A
    deterministic seed namespace need not encode every public field, but it
    must not depend on private copy IDs or a hidden draw-pile permutation.
    """
    return PolicyInput({"session_projection": json.loads(session_projection_json(public))}, public.legal_actions)


def require_native():
    """Reuse the environment's loaded extension or the project-local build.

    This loads code only, never a game session or a hidden observation. Installed
    callers can load their explicitly located extension with load_sts_lightspeed
    first. An incompatible already-loaded module is never silently replaced.
    """
    from .environment import EnvironmentError
    from .simulator import load_sts_lightspeed, SimulatorLoadError
    try:
        module = sys.modules.get("slaythespire")
        if module is None:
            module = load_sts_lightspeed(Path(__file__).resolve().parents[2] /
                                        "runtime/external/sts_lightspeed/build")
    except SimulatorLoadError as error:
        raise EnvironmentError("C++ combat module unavailable; build the pinned native runtime. "
                               "Python fallback is disabled: " + str(error)) from error
    required = ("public_rule_choose", "public_rule_decision", "public_potion_choose", "public_combat_value", "public_ranked_choose")
    if not compatible_versions(module) or not all(callable(getattr(module, name, None)) for name in required):
        raise EnvironmentError("C++ combat contract/version mismatch; rebuild the pinned native runtime. "
                               "Python fallback is disabled")
    return module


def rule_projection(public: PolicyInput) -> dict:
    """Normalize optional display fields, without weakening search admission.

    Rule-only input historically permits static costs/types and empty absent
    piles. Those same public/static defaults are transport, not a second scorer.
    Dynamic fields (damage history, selection mapping, Bomb/order evidence) are
    NOT manufactured. Search still uses the strict projection above.
    """
    catalog = _catalog()
    obs = public.observation
    combat = dict(obs.get("combat", {}))
    from .play_triggers import play_trigger_state_reason
    reason = play_trigger_state_reason(combat)
    if reason:
        raise ValueError(reason)
    if str(combat.get("card_select_task", "")).upper() == "ARMAMENTS":
        candidates = combat.get("selection_candidates")
        if (not isinstance(candidates, (tuple, list)) or not candidates or any(
                not isinstance(c, Mapping) or not c.get("id") or c not in combat.get("hand", ())
                for c in candidates)):
            raise ValueError("upgrade selection has no valid structured public candidates")

    def card(c):
        if not isinstance(c, Mapping) or not c.get("id"):
            raise ValueError("card/candidate requires structured public fields")
        entry = catalog.lookup("card", c.get("id")) or {}
        up = int(c.get("upgrades", c.get("upgraded", 0)))
        variants = entry.get("variants", ())
        printed = int(variants[int(up > 0)]["energy_cost"]) if len(variants) == 2 else 0
        return {**c, "upgrades": up, "type": str(c.get("type", entry.get("type", ""))).upper(),
                "cost_for_turn": c.get("cost_for_turn", c.get("cost", printed)),
                "base_cost": c.get("base_cost", printed),
                "free_to_play_once": c.get("free_to_play_once", False),
                "is_playable": c.get("is_playable", True)}

    for pile in ("hand", "draw_pile", "draw_pile_unordered", "discard_pile", "exhaust_pile", "selection_candidates"):
        if pile in combat or pile in {"hand", "discard_pile", "exhaust_pile"}:
            combat[pile] = [card(c) for c in combat.get(pile, ())]
    # Known-top candidates must carry the same static defaults as pile entries.
    if isinstance(combat.get("draw_pile_known_top"), Mapping):
        combat["draw_pile_known_top"] = card(combat["draw_pile_known_top"])
    if "draw_pile_known_bottom" in combat:
        combat["draw_pile_known_bottom"] = [card(c) for c in combat["draw_pile_known_bottom"]]
    p = combat.get("player", obs.get("player", {}))
    combat["player"] = {"current_hp": p.get("current_hp", p.get("hp", 0)), "max_hp": p.get("max_hp", 80),
                        "energy": p.get("energy", 0), "block": p.get("block", 0), "powers": p.get("powers", ())}
    combat["monsters"] = [{**m, "current_hp": m.get("current_hp", 0),
        "max_hp": m.get("max_hp", m.get("current_hp", 0)), "block": m.get("block", 0),
        "move_base_damage": m.get("move_base_damage", m.get("intent_base_damage", -1)),
        "move_hits": m.get("move_hits", m.get("intent_hits", 0)),
        "is_gone": m.get("is_gone", False), "half_dead": m.get("half_dead", False),
        "powers": m.get("powers", ()), **({"stasis_card": card(m["stasis_card"])} if "stasis_card" in m else {})}
        for m in combat.get("monsters", ())]
    combat["input_state"] = str(combat.get("input_state", "PLAYER_NORMAL")).upper()
    if "card_select_task" in combat:
        combat["card_select_task"] = str(combat["card_select_task"]).upper()
    inv = {"relics": (), **obs.get("inventory", {})}
    view = public_projection(PolicyInput({**obs, "combat": combat, "inventory": inv}, public.legal_actions))
    view["potions"] = [token(p.get("id", ""), "potion") for p in inv.get("potions", ())]
    return view


def _decision(public: PolicyInput, method: str, *, base=None):
    from .actions import Action
    from .environment import InvalidActionError
    from .rule_coverage import _validate_input
    _validate_input(public)
    module = require_native()
    try:
        view = json.dumps(rule_projection(public), sort_keys=True, separators=(",", ":"))
        actions = json.dumps([a.as_dict() for a in public.legal_actions])
        if base is None:
            extra = metadata_json()
        else:
            extra = ('{"base":' + json.dumps(base.as_dict()) + ',"metadata":'
                     + metadata_json() + '}')
        document = json.loads(getattr(module, method)(view, actions, extra))
        if method == "public_rule_decision":
            if (set(document) != {"action", "scope_gaps"} or not isinstance(document["scope_gaps"], list)
                    or any(not isinstance(gap, str) or not gap for gap in document["scope_gaps"])):
                raise ValueError("malformed native rule decision diagnostics")
            result = Action.from_dict(document["action"])
        else:
            result = Action.from_dict(document)
    except (KeyError, TypeError, ValueError, RuntimeError, IndexError) as error:
        raise InvalidActionError("C++ public combat decision refused: " + str(error)) from error
    if result not in public.legal_actions:
        raise InvalidActionError("C++ decision returned an action outside public legal actions")
    return (result, tuple(document["scope_gaps"])) if method == "public_rule_decision" else result


def rule_choice(public: PolicyInput):
    return rule_decision(public)[0]


def rule_decision(public: PolicyInput):
    return _decision(public, "public_rule_decision")


def potion_choice(public: PolicyInput, base):
    return _decision(public, "public_potion_choose", base=base)


def leaf_value(public: PolicyInput, outcome) -> float:
    from .combat_sampling import CombatOutcome
    from .rule_coverage import _validate_input
    _validate_input(public)
    kinds = (CombatOutcome.ONGOING, CombatOutcome.VICTORY, CombatOutcome.LOSS, CombatOutcome.ESCAPED)
    return require_native().public_combat_value(json.dumps(rule_projection(public)), kinds.index(outcome))


def ranked_choice(public: PolicyInput, estimates):
    from dataclasses import asdict
    from .actions import Action
    from .environment import InvalidActionError
    rows = [{**asdict(e), "action": e.action.as_dict()} for e in estimates]
    result = json.loads(require_native().public_ranked_choose(
        json.dumps(rule_projection(public)), metadata_json(), json.dumps(rows, allow_nan=False)))
    result["action"] = Action.from_dict(result["action"])
    if result["action"] not in public.legal_actions:
        raise InvalidActionError("C++ risk ranker returned an illegal action")
    return result
