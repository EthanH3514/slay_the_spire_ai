# Continue training Spire AI

Follow the [quick start](../README.md#quick-start) to build the simulator and
install CPU PyTorch. Run the commands below from the repository root.

`models/training.pt` contains the released deck-building, route, combat prior
and combat value weights and their Adam optimizer state. The training command
collects new simulator games using the current checkpoint. It saves a single
resumable checkpoint after each round.

## Deck building and routes

```powershell
& $python scripts/train.py outside --games 2 --output runtime/training/checkpoint.pt
```

This runs A10 Ironclad games with the combat policy fixed. It records the
probability and value of each sampled construction and route choice, and
updates both heads with PPO after the batch. Clearing Acts 1, 2 and 3 adds 1,
2 and 3 reward points respectively. PPO uses complete runs. A game stopped by
the action cap is excluded from the update.

## Combat

```powershell
& $python scripts/train.py combat --checkpoint runtime/training/checkpoint.pt --games 2
```

This runs games with the current deck-building and route policies. The prior
learns from the distribution of root visits for eligible search decisions.
The value model learns from the actual outcomes and remaining resources of
completed fights. The collector reads public combat state before the
decision. It attaches the observed outcome after the fight.

For a short check, collect one complete opening fight and update both combat
heads in a separate checkpoint.

```powershell
& $python scripts/train.py combat --games 1 --start-seed 20261125 --max-actions 12 --output runtime/training/combat-smoke.pt
```

## Resume and compare

Pass `--checkpoint` to resume from your last save. The default output for a
resumed run is the same checkpoint. Pass a new `--output` to keep the input
checkpoint as a baseline. Without `--checkpoint`, training starts from
`models/training.pt`. Use `--rounds` for multiple collection and update
cycles. Each round advances the simulator seed cursor; `--start-seed` allows
an explicit starting seed. The local `runtime/` directory holds checkpoints
and any logs you choose to save.

```powershell
& $python scripts/train.py evaluate --games 8 --start-seed 91000000
& $python scripts/train.py evaluate --checkpoint runtime/training/checkpoint.pt --games 8 --start-seed 91000000
```

Use the same evaluation seeds for both checkpoints and keep them separate
from training seeds. The evaluator reports each game's outcome, floor,
remaining HP and act clears. It uses the same action budget and policy
dispatcher as the released runner.
