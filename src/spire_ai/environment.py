from __future__ import annotations

import copy
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping, Protocol, runtime_checkable

from .actions import Action


class EnvironmentError(RuntimeError):
    """Base error for the backend-neutral environment API."""


class InvalidActionError(EnvironmentError):
    """Raised when a policy submits an action that is not currently legal."""


class CounterfactualActionError(InvalidActionError):
    """An offline trace cannot answer an action different from the recorded one."""


class EpisodeExhaustedError(EnvironmentError):
    """Raised when step is called before reset or after an episode has ended."""


RewardComponents = Mapping[str, int | float | bool]


@dataclass(frozen=True, slots=True)
class EnvironmentState:
    """One backend decision snapshot; normal policies receive ``PolicyInput``.

    ``observation`` uses the compact trace's structured schema. Snapshot references
    are materialized by replay environments. Instrumentation details such as exact
    draw-pile order are sanitized before normal policy use. ``backend_features`` carries an
    optional fixed-width vector without polluting the semantic observation.
    ``audit_state`` carries hidden backend diagnostics such as RNG counters. It is
    deliberately separate from ``observation`` and must never be fed to a normal
    (non-oracle) policy.
    """

    observation: Mapping[str, Any]
    legal_actions: tuple[Action, ...]
    backend_features: tuple[int | float, ...] = ()
    audit_state: Mapping[str, Any] = field(default_factory=dict)
    info: Mapping[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "observation": copy.deepcopy(dict(self.observation)),
            "legal_actions": [action.as_dict() for action in self.legal_actions],
            "backend_features": list(self.backend_features),
            "audit_state": copy.deepcopy(dict(self.audit_state)),
            "info": copy.deepcopy(dict(self.info)),
        }


@dataclass(frozen=True, slots=True)
class StepResult:
    """Gym-style result shared by live simulation and offline trace replay."""

    state: EnvironmentState
    reward: RewardComponents
    terminated: bool
    truncated: bool
    info: Mapping[str, Any] = field(default_factory=dict)

    @property
    def done(self) -> bool:
        return self.terminated or self.truncated

    def as_dict(self) -> dict[str, Any]:
        return {
            "state": self.state.as_dict(),
            "reward": dict(self.reward),
            "terminated": self.terminated,
            "truncated": self.truncated,
            "done": self.done,
            "info": copy.deepcopy(dict(self.info)),
        }


@runtime_checkable
class SpireEnvironment(Protocol):
    """Minimal interface for evaluators/collectors, separate from policy inputs."""

    backend_name: str

    def reset(
        self,
        *,
        seed: int | None = None,
        character: str | None = None,
        ascension: int | None = None,
    ) -> EnvironmentState: ...

    def step(self, action: Action) -> StepResult: ...


class CompactReplayEnvironment:
    """Replay a compact real-game episode through the common environment API.

    This is intentionally an offline environment: it can advance only along the
    action actually observed in the source trajectory. It is suitable for behavior
    cloning evaluation, loaders, feature work and API parity tests, but it does not
    pretend to answer counterfactual actions.
    """

    backend_name = "compact_replay"

    def __init__(self, path: Path) -> None:
        self.path = path.expanduser().resolve()
        self.episode, self.decisions = _read_compact_episode(self.path)
        if not self.decisions:
            raise EnvironmentError(f"compact episode contains no decisions: {self.path}")
        self._position: int | None = None

    def reset(
        self,
        *,
        seed: int | None = None,
        character: str | None = None,
        ascension: int | None = None,
    ) -> EnvironmentState:
        self._validate_requested_episode(seed, character, ascension)
        self._position = 0
        return self._decision_state(0)

    def step(self, action: Action) -> StepResult:
        if self._position is None:
            raise EpisodeExhaustedError("reset must be called before step")
        if self._position >= len(self.decisions):
            raise EpisodeExhaustedError("the compact replay episode has ended")

        decision = self.decisions[self._position]
        legal_actions = self._legal_actions(decision)
        if action not in legal_actions:
            raise InvalidActionError(
                f"action is not legal at replay step {self._position}: {action.as_dict()}"
            )
        recorded_action = Action.from_dict(_mapping(decision.get("action")))
        if action != recorded_action:
            raise CounterfactualActionError(
                "compact replay has no counterfactual next state; submit the recorded "
                f"action {recorded_action.as_dict()}"
            )

        current_position = self._position
        self._position += 1
        source_done = bool(decision.get("done"))
        exhausted = self._position >= len(self.decisions)
        outcome_status = str(_mapping(self.episode.get("outcome")).get("status", "unknown"))
        truncated = exhausted and not source_done and outcome_status == "truncated"
        terminated = source_done

        if exhausted:
            next_state = self._terminal_state(decision)
        else:
            next_state = self._decision_state(self._position)

        return StepResult(
            state=next_state,
            reward=_reward(decision.get("reward")),
            terminated=terminated,
            truncated=truncated,
            info={
                "backend": self.backend_name,
                "episode_id": self.episode.get("episode_id"),
                "step": current_position,
                "source_sequence": decision.get("source_sequence"),
                "next_source_sequence": decision.get("next_source_sequence"),
                "recorded": True,
            },
        )

    def recorded_action(self) -> Action:
        if self._position is None or self._position >= len(self.decisions):
            raise EpisodeExhaustedError("there is no current replay decision")
        return Action.from_dict(_mapping(self.decisions[self._position].get("action")))

    def _decision_state(self, position: int) -> EnvironmentState:
        decision = self.decisions[position]
        return EnvironmentState(
            observation=_materialize_observation(
                _mapping(decision.get("observation")), self.episode
            ),
            legal_actions=self._legal_actions(decision),
            audit_state=copy.deepcopy(_mapping(decision.get("audit_state"))),
            info={
                "backend": self.backend_name,
                "episode_id": self.episode.get("episode_id"),
                "step": position,
                "source_sequence": decision.get("source_sequence"),
                "recorded_action": copy.deepcopy(decision.get("action")),
                "observation_fidelity": "structured",
            },
        )

    def _terminal_state(self, decision: Mapping[str, Any]) -> EnvironmentState:
        return EnvironmentState(
            observation=_materialize_observation(
                _mapping(decision.get("next_observation")), self.episode
            ),
            legal_actions=(),
            audit_state=copy.deepcopy(_mapping(decision.get("next_audit_state"))),
            info={
                "backend": self.backend_name,
                "episode_id": self.episode.get("episode_id"),
                "step": len(self.decisions),
                "observation_fidelity": "structured",
            },
        )

    def _legal_actions(self, decision: Mapping[str, Any]) -> tuple[Action, ...]:
        vocabulary = _sequence(self.episode.get("action_vocabulary"))
        mask = _sequence(decision.get("legal_action_mask"))
        if len(mask) != len(vocabulary):
            raise EnvironmentError("legal action mask length does not match action vocabulary")
        actions: list[Action] = []
        for enabled, raw_action in zip(mask, vocabulary, strict=True):
            if bool(enabled):
                actions.append(Action.from_dict(_mapping(raw_action)))
        return tuple(actions)

    def _validate_requested_episode(
        self,
        seed: int | None,
        character: str | None,
        ascension: int | None,
    ) -> None:
        game = _mapping(self.episode.get("game"))
        checks = (
            ("seed", seed, game.get("seed_numeric")),
            ("character", character.upper() if character else None, game.get("character")),
            ("ascension", ascension, game.get("ascension")),
        )
        mismatches = [
            f"{name}={requested!r} (episode has {actual!r})"
            for name, requested, actual in checks
            if requested is not None and requested != actual
        ]
        if mismatches:
            raise EnvironmentError("replay reset does not match episode: " + "; ".join(mismatches))


def _read_compact_episode(
    path: Path,
) -> tuple[Mapping[str, Any], list[Mapping[str, Any]]]:
    if not path.is_file():
        raise EnvironmentError(f"compact episode does not exist: {path}")
    records: list[Mapping[str, Any]] = []
    try:
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, Mapping):
                    raise EnvironmentError(f"record {line_number} is not a JSON object")
                records.append(value)
    except (OSError, json.JSONDecodeError) as exc:
        raise EnvironmentError(f"failed to read compact episode {path}: {exc}") from exc

    if not records or records[0].get("record_type") != "episode":
        raise EnvironmentError("first compact record must be an episode record")
    if records[0].get("schema_version") != 1:
        raise EnvironmentError(
            f"unsupported compact schema version: {records[0].get('schema_version')!r}"
        )
    decisions = [record for record in records[1:] if record.get("record_type") == "decision"]
    return records[0], decisions


def _materialize_observation(
    observation: Mapping[str, Any], episode: Mapping[str, Any]
) -> dict[str, Any]:
    result = copy.deepcopy(dict(observation))
    inventory = result.get("inventory")
    if not isinstance(inventory, dict):
        return result
    tables = _mapping(episode.get("snapshot_tables"))
    for singular, plural in (
        ("deck", "decks"),
        ("relics", "relics"),
        ("potions", "potions"),
        ("map", "maps"),
    ):
        reference_key = f"{singular}_ref"
        reference = inventory.pop(reference_key, None)
        if reference is None:
            continue
        table = _sequence(tables.get(plural))
        if not isinstance(reference, int) or not 0 <= reference < len(table):
            raise EnvironmentError(f"invalid {reference_key}: {reference!r}")
        inventory[singular] = copy.deepcopy(table[reference])
    return result


def _reward(value: object) -> dict[str, int | float | bool]:
    reward = _mapping(value)
    result: dict[str, int | float | bool] = {}
    for key, item in reward.items():
        if isinstance(item, bool) or isinstance(item, (int, float)):
            result[str(key)] = item
    return result


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _sequence(value: object) -> list[Any]:
    return list(value) if isinstance(value, (list, tuple)) else []
