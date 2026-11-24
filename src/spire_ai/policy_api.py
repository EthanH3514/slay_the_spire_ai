from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Protocol, runtime_checkable

from .actions import Action
from .environment import EnvironmentState, InvalidActionError
from .policy_scope import ScopeDecision, assess_policy_scope
from .public_map import _ValidatedPublicMap, freeze_public_map


_SEED_KEYS = {"seed", "seed_numeric", "seed_text"}
_FORBIDDEN_POLICY_KEYS = {
    "audit_state",
    "backend_features",
    "rng",
    "rng_state",
    "rng_streams",
    "streams",
    "monster_list",
    "elite_monster_list",
    "common_relic_pool",
    "uncommon_relic_pool",
    "rare_relic_pool",
    "shop_relic_pool",
    "boss_relic_pool",
}


PublicValue = (
    None
    | bool
    | int
    | float
    | str
    | tuple["PublicValue", ...]
    | Mapping[str, "PublicValue"]
)


class PolicySpecialist(str, Enum):
    COMBAT = "combat"
    RUN = "run"
    ROUTE = "route"


@dataclass(frozen=True, slots=True)
class PolicyInput:
    """Immutable information boundary for a normal no-SL policy.

    A policy deliberately receives neither ``EnvironmentState`` nor an escape hatch
    back to it. Exact RNG audit state, backend-specific features, and reproducibility
    metadata such as the numeric seed stay outside this object.
    """

    observation: Mapping[str, PublicValue]
    legal_actions: tuple[Action, ...]
    # Only the sanitizing factory owns all descendants. A caller can construct a
    # PolicyInput around a mutable dict (or a proxy onto one), so that path must
    # NEVER memoize by object identity. Neither field enters policy serialization.
    _owned: bool = field(default=False, init=False, repr=False, compare=False)
    _fingerprint: str | None = field(default=None, init=False, repr=False, compare=False)

    def as_dict(self) -> dict[str, Any]:
        return {
            "observation": _thaw_public_value(self.observation),
            "legal_actions": [action.as_dict() for action in self.legal_actions],
        }

    def fingerprint(self) -> str:
        """The existing byte-for-byte public hash, memoized only for owned input."""
        if self._owned and self._fingerprint is not None:
            return self._fingerprint
        payload = json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":"))
        result = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        if self._owned:
            object.__setattr__(self, "_fingerprint", result)
        return result

    def with_legal_actions(self, actions: tuple[Action, ...]) -> PolicyInput:
        """Reuse owned observation, but never a hash of a different action mask."""
        actions = tuple(actions)
        # Dataclass equality conflates e.g. index 0 and False; JSON (and the
        # sampling seed) does not. Reuse only the very same immutable actions.
        if len(actions) == len(self.legal_actions) and all(
            a is b for a, b in zip(actions, self.legal_actions, strict=True)
        ):
            return self
        result = PolicyInput(self.observation, actions)
        if self._owned and _immutable_actions(actions):
            object.__setattr__(result, "_owned", True)
        return result


@runtime_checkable
class DecisionPolicy(Protocol):
    def choose(self, policy_input: PolicyInput) -> Action: ...


@dataclass(slots=True)
class HierarchicalPolicy:
    """Route public decisions to combat, run-management, or map specialists."""

    combat: DecisionPolicy
    run: DecisionPolicy
    route: DecisionPolicy
    last_scope_decision: ScopeDecision | None = field(default=None, init=False)

    def choose(self, policy_input: PolicyInput) -> Action:
        self.last_scope_decision = None
        self.last_scope_decision = assess_policy_scope(
            policy_input.observation, policy_input.legal_actions,
        )
        policy_input = policy_input.with_legal_actions(self.last_scope_decision.eligible_actions)
        specialist = self.specialist_for(policy_input)
        policy = {
            PolicySpecialist.COMBAT: self.combat,
            PolicySpecialist.RUN: self.run,
            PolicySpecialist.ROUTE: self.route,
        }[specialist]
        action = policy.choose(policy_input)
        if action not in policy_input.legal_actions:
            raise InvalidActionError(
                f"{specialist.value} policy returned an illegal action: {action.as_dict()}"
            )
        return action

    @staticmethod
    def specialist_for(policy_input: PolicyInput) -> PolicySpecialist:
        observation = policy_input.observation
        if isinstance(observation.get("combat"), Mapping):
            return PolicySpecialist.COMBAT
        screen = observation.get("screen")
        screen_type = str(screen.get("type", "")) if isinstance(screen, Mapping) else ""
        if screen_type.upper() in {"MAP", "MAP_SCREEN"}:
            return PolicySpecialist.ROUTE
        return PolicySpecialist.RUN


def policy_input_from_state(state: EnvironmentState) -> PolicyInput:
    """Create a fail-closed no-SL view from a backend environment state."""

    source = dict(state.observation)
    if "map" in source:
        source["map"] = freeze_public_map(source)
    observation = copy.deepcopy(source)
    _strip_seed_fields(observation)
    _hide_unrevealed_draw_order(
        observation,
        known_top_unique_id=state.info.get("public_draw_pile_known_top_unique_id"),
    )
    _hide_unrevealed_intents(observation)
    _reject_forbidden_policy_keys(observation)
    result = PolicyInput(
        observation=_freeze_public_value(observation),
        legal_actions=tuple(state.legal_actions),
    )
    if _immutable_actions(result.legal_actions):
        object.__setattr__(result, "_owned", True)
    return result


def _immutable_actions(actions: tuple[Action, ...]) -> bool:
    # Action is frozen, but type hints alone do not stop a caller from putting a
    # list in one of its scalar fields. Such nonstandard inputs remain uncached.
    return all(type(action) is Action and all(
        type(value) in (type(None), bool, int, float, str)
        for value in action.as_dict().values()
    ) for action in actions)


def _hide_unrevealed_draw_order(
    observation: dict[str, Any], *, known_top_unique_id: object = None,
) -> None:
    """Expose only publicly known draw order absent Frozen Eye."""

    combat = observation.get("combat")
    if not isinstance(combat, dict) or "draw_pile" not in combat:
        return
    # A public event ledger, never inferred from the hidden pile's contents.
    top_ids = combat.get("public_known_top_ids", [])
    if not isinstance(top_ids, (list, tuple)) or any(type(i) is not int for i in top_ids):
        raise TypeError("known draw-pile top IDs must be integers")
    native_pile = combat["draw_pile"]
    if not isinstance(native_pile, list):
        raise TypeError("combat.draw_pile must be a list")
    if len(set(top_ids)) != len(top_ids) or len(top_ids) > len(native_pile) or any(
        not isinstance(native_pile[len(native_pile)-len(top_ids)+i], Mapping)
        or native_pile[len(native_pile)-len(top_ids)+i].get("unique_id") != uid
        for i, uid in enumerate(top_ids)
    ):
        raise ValueError("known draw-pile top suffix does not match native pile")
    if top_ids:
        if known_top_unique_id is not None and known_top_unique_id != top_ids[-1]:
            raise ValueError("conflicting public top knowledge")
        known_top_unique_id = top_ids[-1]
    from .public_piles import opening_priority_ids
    opening_ids = set(opening_priority_ids(combat))
    if opening_ids and not combat.get("opening_order_uncertain"):
        # Validate the declared partial order, never reveal within-band order.
        left = len(combat.get("public_known_bottom_ids", ()))
        right = max(left, len(native_pile) - (len(top_ids) or int(known_top_unique_id is not None)))
        segment = native_pile[left:right]
        count = sum(c.get("unique_id") in opening_ids for c in segment)
        if any((c.get("unique_id") in opening_ids) != (i >= len(segment) - count)
               for i, c in enumerate(segment)):
            raise ValueError("public opening-priority group is not a draw suffix")
    inventory = observation.get("inventory")
    relics = inventory.get("relics", []) if isinstance(inventory, Mapping) else []
    has_frozen_eye = any(
        _identifier_key(relic.get("id")) == "frozeneye"
        for relic in relics
        if isinstance(relic, Mapping)
    )
    if has_frozen_eye:
        combat.pop("public_known_bottom_ids", None)
        return

    draw_pile = combat.pop("draw_pile")
    if not isinstance(draw_pile, list):
        raise TypeError("combat.draw_pile must be a list")
    if known_top_unique_id is not None and type(known_top_unique_id) is not int:
        raise TypeError("known draw-pile top unique ID must be an integer")
    bottom_ids = combat.pop("public_known_bottom_ids", [])
    if not isinstance(bottom_ids, (list, tuple)) or any(type(i) is not int for i in bottom_ids):
        raise TypeError("known draw-pile bottom IDs must be integers")
    if len(set(bottom_ids)) != len(bottom_ids) or len(bottom_ids) > len(draw_pile) or any(
        not isinstance(draw_pile[i], Mapping) or draw_pile[i].get("unique_id") != uid
        for i, uid in enumerate(bottom_ids)
    ):
        raise ValueError("known draw-pile bottom does not match native pile")
    if bottom_ids:
        combat["draw_pile_known_bottom"] = draw_pile[:len(bottom_ids)]
        del draw_pile[:len(bottom_ids)]
    if known_top_unique_id in bottom_ids and (draw_pile or bottom_ids[-1] != known_top_unique_id):
        raise ValueError("known top overlaps bottom inconsistently")
    if known_top_unique_id is not None and known_top_unique_id not in bottom_ids:
        if (
            isinstance(known_top_unique_id, bool)
            or not isinstance(known_top_unique_id, int)
        ):
            raise TypeError("known draw-pile top unique ID must be an integer")
        if (
            not draw_pile
            or not isinstance(draw_pile[-1], Mapping)
            or draw_pile[-1].get("unique_id") != known_top_unique_id
        ):
            raise ValueError("known draw-pile top does not match the native pile top")
        combat["draw_pile_known_top"] = draw_pile.pop()
    combat["draw_pile_unordered"] = sorted(
        draw_pile,
        key=lambda card: json.dumps(card, sort_keys=True, separators=(",", ":")),
    )


def _identifier_key(value: object) -> str:
    return "".join(character for character in str(value).lower() if character.isalnum())


def hidden_enemy_intents(observation: Mapping[str, Any]) -> bool:
    """Runic Dome makes the current enemy intent unavailable to a no-SL policy."""

    inventory = observation.get("inventory")
    relics = inventory.get("relics", ()) if isinstance(inventory, Mapping) else ()
    return any(
        _identifier_key(relic.get("id")) == "runicdome"
        for relic in relics
        if isinstance(relic, Mapping)
    )


def _hide_unrevealed_intents(observation: dict[str, Any]) -> None:
    if not hidden_enemy_intents(observation):
        return
    combat = observation.get("combat")
    if not isinstance(combat, dict):
        return
    for monster in combat.get("monsters", ()):
        if isinstance(monster, dict):
            for key in list(monster):
                if key.startswith(("move_", "intent")) or key == "is_attacking":
                    monster.pop(key)


def _reject_forbidden_policy_keys(value: object, path: str = "$", *, reject_seeds=False) -> None:
    if type(value) is _ValidatedPublicMap:
        return  # Exact schema admits no hidden keys; owned immutable descendants.
    if path == "$" and isinstance(value, Mapping) and "map" in value:
        from .public_map import validate_public_map

        validate_public_map(value)
    if isinstance(value, Mapping):
        for raw_key, item in value.items():
            key = str(raw_key)
            if key.lower() in _FORBIDDEN_POLICY_KEYS:
                raise ValueError(f"forbidden no-SL policy field at {path}.{key}")
            if reject_seeds and key.lower() in _SEED_KEYS:
                raise ValueError("coverage public input contains seed metadata")
            _reject_forbidden_policy_keys(item, f"{path}.{key}", reject_seeds=reject_seeds)
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _reject_forbidden_policy_keys(item, f"{path}[{index}]", reject_seeds=reject_seeds)


def _strip_seed_fields(value: object) -> None:
    if isinstance(value, dict):
        for key in list(value):
            if str(key).lower() in _SEED_KEYS:
                value.pop(key)
            else:
                _strip_seed_fields(value[key])
    elif isinstance(value, list):
        for item in value:
            _strip_seed_fields(item)


def _freeze_public_value(value: object) -> PublicValue:
    if type(value) is _ValidatedPublicMap:
        return value
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze_public_value(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_public_value(item) for item in value)
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError(f"policy observation contains unsupported value: {type(value).__name__}")


def _thaw_public_value(value: PublicValue | Sequence[PublicValue]) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw_public_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_public_value(item) for item in value]
    return value
