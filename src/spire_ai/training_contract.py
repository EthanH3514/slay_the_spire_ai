"""Small, versioned training contracts. No tensor/runtime dependency here."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import math

from .environment import StepResult
from .evaluation import SeedManifest


TRAINING_VERSION = "public-fixed-combat-ppo-v2"
REWARD_VERSION = "terminal-first-progress-v1"
SEED_VERSION = "training-sha256-counter-v1"


@dataclass(frozen=True)
class TrainingConfig:
    version: str = TRAINING_VERSION
    model_seed: int = 731
    seed_namespace: str = "spire-ai/train/ironclad-a0/v2"
    fixed_combat: str = "native_rule"
    search_sampling_seed: int = 19000
    updates: int = 2
    rollout_steps: int = 256
    max_env_steps: int = 1024
    max_episode_steps: int = 1000
    worker_timeout_seconds: int = 180
    width: int = 32
    epochs: int = 2
    minibatch_size: int = 32
    learning_rate: float = 0.0003
    gamma: float = 0.999
    gae_lambda: float = 0.95
    clip: float = 0.2
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: float = 0.03
    progress_cap: float = 0.2

    def validate(self):
        if self.version != TRAINING_VERSION:
            raise ValueError("unsupported training protocol")
        from .fixed_combat import fixed_combat_contract
        fixed_combat_contract(self.fixed_combat, self.search_sampling_seed)
        limits = {"model_seed": (0, 2**31 - 1), "updates": (1, 256),
                  "rollout_steps": (2, 2048), "max_env_steps": (2, 100_000),
                  "max_episode_steps": (2, 10_000), "worker_timeout_seconds": (1, 3600),
                  "width": (8, 128), "epochs": (1, 10), "minibatch_size": (1, 256)}
        for name, (low, high) in limits.items():
            value = getattr(self, name)
            if type(value) is not int or not low <= value <= high:
                raise ValueError("invalid training integer: " + name)
        if not isinstance(self.seed_namespace, str) or not self.seed_namespace.startswith("spire-ai/train/"):
            raise ValueError("a separate training seed namespace is required")
        for name, low, high in (("learning_rate", 0, .1), ("gamma", 0, 1),
            ("gae_lambda", 0, 1), ("clip", 0, 1), ("entropy_coef", 0, 1),
            ("value_coef", 0, 10), ("max_grad_norm", 0, 100),
            ("target_kl", 0, 1), ("progress_cap", 0, .25)):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value) or not low <= value <= high:
                raise ValueError("invalid training number: " + name)
        if min(self.learning_rate, self.gamma, self.clip, self.max_grad_norm, self.target_kl) <= 0:
            raise ValueError("positive optimizer/discount settings required")
        return self

    def as_dict(self):
        return asdict(self)


class TrainingSeedStream:
    """Lazy training-only stream, with explicit collision exclusions and cursor.

    Exclusion derives seed identities, NEVER runs held-out episodes. This metadata
    belongs to the collector/checkpoint, not either network's input.
    """

    def __init__(self, manifest: SeedManifest, namespace: str, *, state=None):
        if not namespace.startswith("spire-ai/train/") or namespace == manifest.namespace:
            raise ValueError("separate training namespace required")
        if (manifest.character, manifest.ascension, manifest.victory_condition) != ("IRONCLAD", 0, "act3_boss"):
            raise ValueError("pilot requires the Ironclad A0 act3 manifest")
        self.namespace = namespace
        self.excluded = {seed for name in manifest.splits for seed in manifest.seeds(name)}
        self.exclusion_hash = hashlib.sha256(
            ",".join(map(str, sorted(self.excluded))).encode()).hexdigest()
        self.counter = 0
        self.issued: set[int] = set()
        self.skipped_collisions = 0
        if state is not None:
            if (state.get("version") != SEED_VERSION or state.get("namespace") != namespace
                    or state.get("exclusion_hash") != self.exclusion_hash):
                raise ValueError("seed checkpoint does not match training exclusions")
            counter, issued = state.get("counter"), state.get("issued")
            if (type(counter) is not int or not 0 <= counter <= 1_000_000
                    or not isinstance(issued, list) or any(type(v) is not int or not 0 < v < 2**63 for v in issued)
                    or len(issued) != len(set(issued)) or self.excluded.intersection(issued)):
                raise ValueError("malformed seed checkpoint")
            # Verify history rather than trusting a cursor which could repeat a seed.
            for _ in range(len(issued)):
                self.next_seed()
            if self.counter != counter or self.issued != set(issued) or self.skipped_collisions != state.get("skipped_collisions"):
                raise ValueError("seed checkpoint history mismatch")

    def next_seed(self) -> int:
        while self.counter < 1_000_000:
            digest = hashlib.sha256(f"{self.namespace}:{self.counter}".encode()).digest()
            self.counter += 1
            seed = (int.from_bytes(digest[:8], "big") & ((1 << 63) - 1)) or 1
            if seed in self.excluded or seed in self.issued:
                self.skipped_collisions += 1
                continue
            self.issued.add(seed)
            return seed
        raise ValueError("pilot training seed stream exhausted")

    def state_dict(self):
        return {"version": SEED_VERSION, "namespace": self.namespace, "counter": self.counter,
                "issued": sorted(self.issued), "exclusion_hash": self.exclusion_hash,
                "skipped_collisions": self.skipped_collisions}


class TrainingReward:
    """Terminal +/-1 and capped first-progress auxiliary reward; no HP penalty.

    This auxiliary objective is NOT policy-invariant shaping. Report it separately
    and keep true clears independent. Repeated rooms/actions cannot farm progress.
    """

    def __init__(self, progress_cap: float = .2):
        if type(progress_cap) not in (int, float) or not math.isfinite(progress_cap) or not 0 <= progress_cap <= .25:
            raise ValueError("invalid progress cap")
        self.cap = progress_cap
        self.highest_floor = 0

    def reset(self, observation):
        floor = observation.get("run", {}).get("floor", 0)
        if type(floor) is not int or floor < 0:
            raise ValueError("invalid public floor")
        self.highest_floor = min(floor, 51)

    def step(self, result: StepResult) -> dict:
        run = result.state.observation.get("run", {})
        floor = run.get("floor")
        if type(floor) is not int or floor < 0:
            raise ValueError("invalid public floor")
        next_floor = max(self.highest_floor, min(floor, 51))
        progress = self.cap * (next_floor - self.highest_floor) / 51
        terminal = 0.0
        if result.terminated:
            outcome = run.get("outcome")
            if outcome not in {"PLAYER_VICTORY", "PLAYER_LOSS"}:
                raise ValueError("unrecognized terminal outcome; not a training death")
            terminal = 1.0 if outcome == "PLAYER_VICTORY" else -1.0
        elif run.get("outcome") in {"PLAYER_VICTORY", "PLAYER_LOSS"}:
            raise ValueError("inconsistent nonterminal outcome")
        self.highest_floor = next_floor
        return {"terminal": terminal, "progress": progress, "total": terminal + progress}


def generalized_advantages(rewards, values, next_values, terminated, truncated, *, gamma, lam):
    """GAE through fixed-combat transitions; truncate bootstraps but cuts the trace."""
    lengths = {len(x) for x in (rewards, values, next_values, terminated, truncated)}
    if len(lengths) != 1 or not rewards:
        raise ValueError("nonempty aligned trajectory arrays required")
    if not 0 < gamma <= 1 or not 0 <= lam <= 1:
        raise ValueError("invalid GAE discount")
    if any(type(x) is not bool for x in (*terminated, *truncated)):
        raise ValueError("boolean termination masks required")
    if not all(math.isfinite(v) for series in (rewards, values, next_values) for v in series):
        raise ValueError("non-finite rollout")
    advantages = [0.0] * len(rewards)
    following = 0.0
    for i in range(len(rewards) - 1, -1, -1):
        delta = rewards[i] + (0 if terminated[i] else gamma * next_values[i]) - values[i]
        following = delta + (0 if terminated[i] or truncated[i] else gamma * lam * following)
        advantages[i] = following
    return advantages, [a + v for a, v in zip(advantages, values)]
