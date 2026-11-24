from __future__ import annotations

import hashlib
import json
import math
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from .environment import EnvironmentState, SpireEnvironment
from .policy_api import DecisionPolicy, policy_input_from_state
from .rule_coverage import RuleCoverage


@dataclass(frozen=True, slots=True)
class SeedRange:
    start: int
    count: int


@dataclass(frozen=True, slots=True)
class SeedManifest:
    schema_version: int
    name: str
    ruleset: str
    character: str
    ascension: int
    victory_condition: str
    algorithm: str
    namespace: str
    splits: Mapping[str, SeedRange]

    @classmethod
    def load(cls, path: Path) -> "SeedManifest":
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, Mapping):
            raise ValueError("seed manifest must be a JSON object")
        generator = _mapping(value.get("generator"))
        raw_splits = _mapping(value.get("splits"))
        manifest = cls(
            schema_version=_required_int(value, "schema_version"),
            name=_required_string(value, "name"),
            ruleset=_required_string(value, "ruleset"),
            character=_required_string(value, "character").upper(),
            ascension=_required_int(value, "ascension"),
            victory_condition=_required_string(value, "victory_condition"),
            algorithm=_required_string(generator, "algorithm"),
            namespace=_required_string(generator, "namespace"),
            splits={
                str(name): SeedRange(
                    start=_required_int(_mapping(raw_range), "start"),
                    count=_required_int(_mapping(raw_range), "count"),
                )
                for name, raw_range in raw_splits.items()
            },
        )
        manifest.validate()
        return manifest

    def validate(self) -> None:
        if self.schema_version != 1:
            raise ValueError(f"unsupported seed manifest schema: {self.schema_version}")
        if self.ruleset != "no_sl":
            raise ValueError("only the no_sl ruleset is currently supported")
        if not 0 <= self.ascension <= 20:
            raise ValueError("ascension must be between 0 and 20")
        if self.victory_condition != "act3_boss":
            raise ValueError("only act3_boss victory is currently supported")
        if self.algorithm != "sha256-counter-v1":
            raise ValueError(f"unsupported seed algorithm: {self.algorithm}")
        if not self.namespace:
            raise ValueError("seed namespace cannot be empty")
        if not self.splits:
            raise ValueError("seed manifest must define at least one split")

        occupied: set[int] = set()
        for name, seed_range in self.splits.items():
            if not name:
                raise ValueError("seed split name cannot be empty")
            if seed_range.start < 0 or seed_range.count < 1:
                raise ValueError(f"invalid seed range for split {name}")
            indexes = set(range(seed_range.start, seed_range.start + seed_range.count))
            if occupied.intersection(indexes):
                raise ValueError("seed split counter ranges must not overlap")
            occupied.update(indexes)

    def seeds(self, split: str) -> tuple[int, ...]:
        try:
            seed_range = self.splits[split]
        except KeyError as exc:
            available = ", ".join(sorted(self.splits))
            raise ValueError(f"unknown seed split {split!r}; choose one of: {available}") from exc
        return tuple(
            _seed_from_counter(self.namespace, counter)
            for counter in range(seed_range.start, seed_range.start + seed_range.count)
        )


@dataclass(frozen=True, slots=True)
class NoSlEpisodeResult:
    seed: int
    steps: int
    won: bool
    terminated: bool
    truncated: bool
    final_floor: int | None
    final_hp: int | None
    final_screen: str | None
    reload_count: int
    elapsed_seconds: float
    action_counts: Mapping[str, int]


@dataclass(frozen=True, slots=True)
class NoSlEvaluationResult:
    manifest: str
    split: str
    ruleset: str
    character: str
    ascension: int
    victory_condition: str
    max_steps_per_seed: int
    attempts: int
    wins: int
    losses: int
    truncated: int
    clear_rate: float
    clear_rate_wilson_95_low: float
    clear_rate_wilson_95_high: float
    mean_final_floor: float
    mean_steps: float
    total_elapsed_seconds: float
    episodes: tuple[NoSlEpisodeResult, ...]

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, indent=2)


PolicyFactory = Callable[[int], DecisionPolicy]


def evaluate_no_sl(
    environment: SpireEnvironment,
    policy_factory: PolicyFactory,
    manifest: SeedManifest,
    *,
    split: str,
    max_steps_per_seed: int,
    coverage: RuleCoverage | None = None,
) -> NoSlEvaluationResult:
    """Evaluate exactly one zero-reload attempt for every seed in a fixed split."""

    if max_steps_per_seed < 1:
        raise ValueError("max_steps_per_seed must be positive")
    seeds = manifest.seeds(split)
    if len(seeds) != len(set(seeds)):
        raise ValueError("seed derivation produced a duplicate within the evaluation split")

    episodes: list[NoSlEpisodeResult] = []
    for episode_index, seed in enumerate(seeds):
        policy = policy_factory(episode_index)
        state = environment.reset(
            seed=seed,
            character=manifest.character,
            ascension=manifest.ascension,
        )
        episodes.append(
            _evaluate_episode(
                environment,
                policy,
                state,
                seed=seed,
                victory_condition=manifest.victory_condition,
                max_steps=max_steps_per_seed,
                coverage=coverage,
                context=f"{manifest.name}/{split}/episode-{episode_index}",
            )
        )

    attempts = len(episodes)
    wins = sum(episode.won for episode in episodes)
    truncated = sum(episode.truncated for episode in episodes)
    losses = attempts - wins - truncated
    low, high = _wilson_interval(wins, attempts)
    return NoSlEvaluationResult(
        manifest=manifest.name,
        split=split,
        ruleset=manifest.ruleset,
        character=manifest.character,
        ascension=manifest.ascension,
        victory_condition=manifest.victory_condition,
        max_steps_per_seed=max_steps_per_seed,
        attempts=attempts,
        wins=wins,
        losses=losses,
        truncated=truncated,
        clear_rate=wins / attempts,
        clear_rate_wilson_95_low=low,
        clear_rate_wilson_95_high=high,
        mean_final_floor=sum(episode.final_floor or 0 for episode in episodes) / attempts,
        mean_steps=sum(episode.steps for episode in episodes) / attempts,
        total_elapsed_seconds=sum(episode.elapsed_seconds for episode in episodes),
        episodes=tuple(episodes),
    )


def _evaluate_episode(
    environment: SpireEnvironment,
    policy: DecisionPolicy,
    state: EnvironmentState,
    *,
    seed: int,
    victory_condition: str,
    max_steps: int,
    coverage: RuleCoverage | None = None,
    context: str = "evaluation",
) -> NoSlEpisodeResult:
    started = time.perf_counter()
    action_counts: Counter[str] = Counter()
    terminated = False
    truncated = False
    steps = 0
    if coverage is not None:
        coverage.observe(policy_input_from_state(state), context=f"{context}/step-0")

    while steps < max_steps:
        if not state.legal_actions:
            truncated = True
            break
        action = policy.choose(policy_input_from_state(state))
        action_counts[action.kind.value] += 1
        result = environment.step(action)
        state = result.state
        steps += 1
        if coverage is not None:
            coverage.observe(
                policy_input_from_state(state), context=f"{context}/step-{steps}",
            )
        if result.done:
            terminated = result.terminated
            truncated = result.truncated
            break
    else:
        truncated = True

    won = _is_win(state, terminated, victory_condition)
    run = _mapping(state.observation.get("run"))
    player = _mapping(state.observation.get("player"))
    screen = _mapping(state.observation.get("screen"))
    return NoSlEpisodeResult(
        seed=seed,
        steps=steps,
        won=won,
        terminated=terminated,
        truncated=truncated,
        final_floor=_optional_int(run.get("floor")),
        final_hp=_optional_int(player.get("hp")),
        final_screen=str(screen["type"]) if screen.get("type") is not None else None,
        reload_count=0,
        elapsed_seconds=time.perf_counter() - started,
        action_counts=dict(sorted(action_counts.items())),
    )


def _is_win(state: EnvironmentState, terminated: bool, victory_condition: str) -> bool:
    if victory_condition != "act3_boss" or not terminated:
        return False
    run = _mapping(state.observation.get("run"))
    if str(run.get("outcome", "")).upper() == "PLAYER_VICTORY":
        return True
    screen = _mapping(state.observation.get("screen"))
    screen_state = _mapping(screen.get("state"))
    return bool(screen_state.get("victory", False))


def _seed_from_counter(namespace: str, counter: int) -> int:
    digest = hashlib.sha256(f"{namespace}:{counter}".encode("utf-8")).digest()
    seed = int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)
    return seed or 1


def _wilson_interval(successes: int, attempts: int) -> tuple[float, float]:
    if attempts < 1:
        raise ValueError("Wilson interval requires at least one attempt")
    z = 1.959963984540054
    rate = successes / attempts
    denominator = 1 + z * z / attempts
    center = (rate + z * z / (2 * attempts)) / denominator
    margin = z * math.sqrt(
        rate * (1 - rate) / attempts + z * z / (4 * attempts * attempts)
    ) / denominator
    return max(0.0, center - margin), min(1.0, center + margin)


def _required_string(value: Mapping[str, Any], key: str) -> str:
    item = value.get(key)
    if not isinstance(item, str) or not item:
        raise ValueError(f"seed manifest field {key!r} must be a non-empty string")
    return item


def _required_int(value: Mapping[str, Any], key: str) -> int:
    item = value.get(key)
    if not isinstance(item, int) or isinstance(item, bool):
        raise ValueError(f"seed manifest field {key!r} must be an integer")
    return item


def _optional_int(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _mapping(value: object) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}
