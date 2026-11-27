"""Fresh ordinary runs: +1 Act1, +2 Act2, +3 native Act3 victory.

Act2-stop trajectories are censored for this objective, never ordinary losses.
The first A20 Act3 boss grants no final reward. No heart/Act4 objective.
"""
from dataclasses import dataclass
import copy, hashlib, json, math
from .stage_reward_targets import Boundary, from_row, completed_act

VERSION = "observed-three-act-milestone-return-v1"


@dataclass(frozen=True)
class RewardSchedule:
    act1: float = 1.0
    act2: float = 2.0
    act3: float = 3.0

    def __post_init__(self):
        if (
            any(
                type(v) not in (int, float) or not math.isfinite(v)
                for v in (self.act1, self.act2, self.act3)
            )
            or not 0 < self.act1 < self.act2 < self.act3
        ):
            raise ValueError("positive increasing stage rewards required")

    def wire(self):
        return dict(
            version=VERSION,
            act1=float(self.act1),
            act2=float(self.act2),
            act3=float(self.act3),
            failure=0.0,
            gamma=1.0,
            gae_lambda=1.0,
        )

    @property
    def objective(self):
        return (
            VERSION
            + "/"
            + hashlib.sha256(
                json.dumps(self.wire(), sort_keys=True).encode()
            ).hexdigest()[:16]
        )


DEFAULT = RewardSchedule()


def completed_stage(b):
    first = completed_act(b)
    if first:
        return first
    if b.after.get("outcome") == "PLAYER_VICTORY":
        if (
            b.before.get("act") != 3
            or b.after.get("act") != 3
            or b.before.get("room_type") != "BOSS"
            or not b.terminated
            or b.truncated
            or not isinstance(b.battle, dict)
            or b.battle.get("version") != "public-battle-result-v1"
            or b.battle.get("outcome") != 1
            or b.battle.get("hp", 0) <= 0
        ):
            raise ValueError("ordinary victory needs terminal Act3 boss evidence")
        return 3
    return None


@dataclass(frozen=True)
class OrdinaryEpisode:
    samples: tuple
    sample_indices: tuple
    boundaries: tuple
    schedule: RewardSchedule
    outcome: str
    split: str


def validate(e):
    if (
        not isinstance(e, OrdinaryEpisode)
        or not isinstance(e.schedule, RewardSchedule)
        or e.split != "train"
    ):
        raise ValueError("ordinary TRAIN episode required")
    bs = e.boundaries
    if not bs or bs[0].before.get("act") != 1 or bs[0].before.get("floor") != 0:
        raise ValueError("fresh run required")
    if (
        len(e.samples) != len(e.sample_indices)
        or any(type(i) is not int or not 0 <= i < len(bs) for i in e.sample_indices)
        or tuple(sorted(set(e.sample_indices))) != e.sample_indices
    ):
        raise ValueError("unaligned decisions")
    cleared = 0
    milestones = []
    for i, b in enumerate(bs):
        if (
            not isinstance(b, Boundary)
            or b.index != i
            or type(b.index) is not int
            or type(b.terminated) is not bool
            or b.truncated is not False
            or (b.terminated and i != len(bs) - 1)
        ):
            raise ValueError("incomplete or reordered trajectory")
        if i and b.before != bs[i - 1].after:
            raise ValueError("run discontinuity")
        act = b.before.get("act")
        if act not in (1, 2, 3) or act not in (cleared, cleared + 1):
            raise ValueError("stage progression lacks boss evidence")
        stage = completed_stage(b)
        if stage:
            if stage != cleared + 1:
                raise ValueError("duplicate or out-of-order milestone")
            cleared = stage
            milestones.append((i, getattr(e.schedule, f"act{stage}")))
    last = bs[-1]
    if not last.terminated or last.after.get("outcome") not in (
        "PLAYER_LOSS",
        "PLAYER_VICTORY",
    ):
        raise ValueError("ordinary episode must finish; Act2 cutoff is censored")
    if last.after["outcome"] == "PLAYER_VICTORY" and cleared != 3:
        raise ValueError("ordinary victory lacks all stages")
    if e.outcome != last.after["outcome"]:
        raise ValueError("outcome mismatch")
    return milestones


def complete_episode(samples, indices, boundaries, *, schedule=DEFAULT, split="train"):
    bs = tuple(copy.deepcopy(boundaries))
    outcome = bs[-1].after.get("outcome") if bs else None
    e = OrdinaryEpisode(tuple(samples), tuple(indices), bs, schedule, outcome, split)
    validate(e)
    return e


def returns(e):
    milestones = validate(e)
    return tuple(
        float(sum(v for i, v in milestones if i >= j)) for j in e.sample_indices
    )


def targets(episodes, *, schedule=DEFAULT):
    import torch

    adv = []
    ret = []
    for e in episodes:
        if not isinstance(e, OrdinaryEpisode) or e.schedule != schedule:
            raise ValueError("cannot mix reward objectives")
        values = returns(e)
        ret.extend(values)
        adv.extend(v - s.old_value for v, s in zip(values, e.samples, strict=True))
    return torch.tensor(adv, dtype=torch.float32), torch.tensor(
        ret, dtype=torch.float32
    )
