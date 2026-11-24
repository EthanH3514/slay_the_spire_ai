"""Policy eligibility is separate from backend legality and hidden game state."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Mapping, Sequence

from .actions import Action, ActionKind
from .environment import InvalidActionError


POLICY_SCOPE_VERSION = "ironclad-public-combat-scope-v2"
_ACQUISITION_SCREENS = frozenset({
    "SHOP", "SHOP_SCREEN", "SHOP_ROOM", "COMBAT_REWARD", "REWARDS", "BOSS_REWARD",
    "BOSS_RELIC_REWARDS",
})
# These are milestone restrictions, not changes to the game's legal actions.
_DEFERRED_RELICS = {
    "prismaticshard": "prismatic_shard",
    "runicdome": "runic_dome",
}


class UnsupportedPolicyScopeError(InvalidActionError):
    """The current milestone cannot safely make this decision."""


@dataclass(frozen=True, slots=True)
class ExcludedAction:
    action: Action
    reason: str


@dataclass(frozen=True, slots=True)
class ScopeDecision:
    eligible_actions: tuple[Action, ...]
    excluded_actions: tuple[ExcludedAction, ...]
    version: str = POLICY_SCOPE_VERSION


def _mapping(value: object) -> Mapping:
    return value if isinstance(value, Mapping) else {}


def _items(value: object) -> Sequence:
    return value if isinstance(value, (tuple, list)) else ()


def _key(value: object) -> str:
    return "".join(c for c in str(value).casefold() if c.isalnum())


def _relic_reason(identifier: object) -> str | None:
    slug = _DEFERRED_RELICS.get(_key(identifier))
    return f"{slug}_acquisition_deferred" if slug else None


def _structured_reason(row: Mapping) -> str | None:
    if row.get("kind") in {"relic", "buy_relic", "boss_relic"}:
        if not isinstance(row.get("item_id"), str) or not row["item_id"].strip():
            return "unresolved_acquisition_choice"
        return _relic_reason(row["item_id"])
    event = _mapping(row.get("event_choice"))
    if event.get("status") != "parsed":
        return None  # Never infer rewards from unknown/localized event prose.
    for effect in _items(event.get("effects")):
        effect = _mapping(effect)
        if effect.get("kind") == "obtain_relic":
            if reason := _relic_reason(effect.get("id")):
                return reason
        if effect.get("kind") == "random_relic" and effect.get("rarity") == "boss":
            # Neow's boss swap can yield Dome. Exclude the public pool choice,
            # regardless of the hidden outcome; never inspect/re-roll that RNG.
            return "random_boss_relic_acquisition_deferred"
    return None


def assess_policy_scope(observation: Mapping, actions: Sequence[Action]) -> ScopeDecision:
    """Exclude deferred relic acquisition without mutating/renumbering actions.

    CommunicationMod choices use localized names, resolved via public relic IDs
    and names in screen state. Native choices use versioned semantic options.
    Random boss offers are excluded by public pool, not by secretly rolled item.
    Already-owned Shard or Dome is refused at every policy decision boundary.
    """
    inventory = _mapping(observation.get("inventory"))
    for relic in _items(inventory.get("relics")):
        slug = _DEFERRED_RELICS.get(_key(_mapping(relic).get("id")))
        if slug:
            raise UnsupportedPolicyScopeError(f"owned_{slug}: outside the supported combat scope")

    screen = _mapping(observation.get("screen"))
    screen_type = str(screen.get("type", "")).upper()
    if screen_type not in _ACQUISITION_SCREENS | {"EVENT"}:
        return ScopeDecision(tuple(actions), ())
    if not any(action.kind is ActionKind.CHOOSE for action in actions):
        return ScopeDecision(tuple(actions), ())  # Safe exits need no choice metadata.

    structured = None
    if "run_options_version" in screen or "decision_options" in screen:
        # Import at call time: the adapter's decoder also uses reward policies.
        from .policy_api import PolicyInput
        from .run_options import public_run_options
        try:
            rows = public_run_options(PolicyInput(observation, tuple(actions)))
        except (ValueError, TypeError, KeyError) as error:
            raise UnsupportedPolicyScopeError(f"invalid public scope options: {error}") from error
        structured = dict(zip(actions, rows, strict=True))
    elif screen_type == "EVENT":
        return ScopeDecision(tuple(actions), ())  # No guessed reward from prose.

    names = dict(_DEFERRED_RELICS)
    state = _mapping(screen.get("state"))
    relics = list(_items(state.get("relics")))
    relics.extend(_mapping(reward).get("relic") for reward in _items(state.get("rewards")))
    for relic in relics:
        relic = _mapping(relic)
        slug = _DEFERRED_RELICS.get(_key(relic.get("id")))
        if slug and relic.get("name"):
            names[_key(relic["name"])] = slug

    choices = _items(screen.get("choices"))
    eligible, excluded = [], []
    for action in actions:
        if action.kind is not ActionKind.CHOOSE:
            eligible.append(action)
            continue
        if structured is not None:
            reason = _structured_reason(structured[action])
            if reason:
                excluded.append(ExcludedAction(action, reason))
            else:
                eligible.append(action)
            continue
        index = action.choice_index
        if type(index) is not int or not 0 <= index < len(choices):
            excluded.append(ExcludedAction(action, "unresolved_acquisition_choice"))
            continue
        choice = choices[index]
        # Native ConsoleSimulator::printShopRoomActions adds a trailing ' N g'
        # (without the space before g). Do not use substring matching.
        label = re.sub(r"\s+\d+g$", "", choice.strip()) if isinstance(choice, str) else ""
        if _key(label) in names:
            excluded.append(ExcludedAction(action, names[_key(label)] + "_acquisition_deferred"))
        elif not label:
            excluded.append(ExcludedAction(action, "unresolved_acquisition_choice"))
        else:
            eligible.append(action)
    decision = ScopeDecision(tuple(eligible), tuple(excluded))
    if not decision.eligible_actions:
        reasons = ", ".join(item.reason for item in excluded)
        raise UnsupportedPolicyScopeError(f"no scope-eligible actions: {reasons}")
    return decision
