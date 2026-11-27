"""Collect fresh simulator games and continue training the released models."""

import argparse
import json
import os
from pathlib import Path
import tempfile

import torch

from play import FrozenPolicy, MODELS, ROOT, configure, install, load
from spire_ai.ordinary_reward_targets import completed_stage
from training.combat import CombatTrainer, CombatTrainingPolicy
from training.outside import OutsideTrainingPolicy

INITIAL = MODELS / "training.pt"
VERSION = "published-training-checkpoint-v1"


def save_checkpoint(path, state):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        dir=path.parent, suffix=".tmp", delete=False
    ) as handle:
        temporary = Path(handle.name)
    try:
        torch.save(state, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def emit(**report):
    print(json.dumps(report, allow_nan=False), flush=True)


def runs(policy, seeds, max_actions):
    for seed in seeds:
        count, outcome = policy.play(
            seed, 2 * seed + 1, 2 * seed + 2, max_actions, lambda *_: None
        )
        stages = [
            completed_stage(boundary) for boundary in policy.environment.boundaries
        ]
        public = policy.environment._current_public_input().observation
        yield dict(
            seed=seed,
            actions=count,
            outcome=outcome,
            floor=public["run"]["floor"],
            hp=public["player"]["hp"],
            act_clears=[stage for stage in stages if stage],
        )


def evaluate(checkpoint, args):
    models = CombatTrainer(checkpoint["combat"], training=False)
    policy = FrozenPolicy(
        args.module_dir,
        construction_state=checkpoint["outside"],
        combat_weights=models.weights(),
    )
    first = args.start_seed if args.start_seed is not None else 91_000_000
    results = []
    for result in runs(policy, range(first, first + args.games), args.max_actions):
        emit(**result)
        results.append(result)
    emit(
        games=len(results),
        full_runs=sum(r["outcome"] != "action_limit" for r in results),
        act_clears={
            str(act): sum(act in r["act_clears"] for r in results) for act in (1, 2, 3)
        },
    )


def outside_round(checkpoint, args, seeds):
    teacher = CombatTrainer(checkpoint["combat"], training=False)
    policy = OutsideTrainingPolicy(
        args.module_dir, checkpoint["outside"], teacher.weights()
    )
    construction, route = [], []
    for seed in seeds:
        count, outcome, construction_episode, route_episode = policy.collect(
            seed, args.max_actions
        )
        emit(seed=seed, actions=count, outcome=outcome)
        if construction_episode is not None:
            construction.append(construction_episode)
            route.append(route_episode)
    if not construction:
        raise ValueError(
            "outside training needs complete runs. Increase --max-actions."
        )
    return policy.update(construction, route, checkpoint["outside"])


def combat_round(checkpoint, args, seeds):
    trainer = CombatTrainer(checkpoint["combat"], training=True)
    policy = CombatTrainingPolicy(args.module_dir, checkpoint["outside"], trainer)
    for seed in seeds:
        count, outcome = policy.collect(seed, args.max_actions)
        emit(seed=seed, actions=count, outcome=outcome)
    state, report = trainer.fit(policy.rows, args.batch_size)
    return state, dict(report, completed_battles=policy.battles)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("outside", "combat", "evaluate"))
    parser.add_argument("--checkpoint", type=Path, default=INITIAL)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--module-dir", type=Path, default=ROOT / "runtime/native")
    parser.add_argument(
        "--games",
        type=int,
        default=2,
        help="Fresh games per round, or evaluation games",
    )
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--max-actions", type=int, default=5000)
    parser.add_argument(
        "--batch-size", type=int, default=64, help="Combat optimizer minibatch"
    )
    parser.add_argument(
        "--start-seed",
        type=int,
        help="First simulator seed, overriding the checkpoint cursor",
    )
    args = parser.parse_args()
    if min(args.games, args.rounds, args.max_actions, args.batch_size) < 1:
        parser.error("game, round, action and batch counts must be positive")
    checkpoint_path = args.checkpoint.resolve()
    output = (
        args.output
        or (
            ROOT / "runtime/training/checkpoint.pt"
            if checkpoint_path == INITIAL
            else checkpoint_path
        )
    ).resolve()
    if args.mode != "evaluate" and (
        output == INITIAL or output.exists() and output != checkpoint_path
    ):
        parser.error(
            "choose a new output path or resume from the existing output checkpoint"
        )
    checkpoint = load(checkpoint_path)
    if checkpoint["version"] != VERSION:
        parser.error("unsupported training checkpoint format")
    start = args.start_seed if args.start_seed is not None else checkpoint["next_seed"]
    if not 0 <= start <= 2**63 - 1 - args.games * args.rounds:
        parser.error("seed range is out of bounds")
    configure()
    install()
    torch.set_num_threads(1)
    if args.mode == "evaluate":
        evaluate(checkpoint, args)
        return
    train = outside_round if args.mode == "outside" else combat_round
    for step in range(args.rounds):
        seeds = range(start, start + args.games)
        state, report = train(checkpoint, args, seeds)
        start = seeds.stop
        checkpoint = dict(checkpoint, **{args.mode: state}, next_seed=start)
        save_checkpoint(output, checkpoint)
        emit(round=step + 1, checkpoint=str(output), **report)


if __name__ == "__main__":
    main()
