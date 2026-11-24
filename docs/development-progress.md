# Release maintenance

## Current state

This branch packages the round-66 combat weights and generation-15
outside-combat weights for standalone simulator play. The simulator is pinned
to the maintained fork at `a738d336f24ec75c754168b8cefa7ee19ca591d3`.
The public READMEs describe the training method and results. The full pipeline
and datasets remain in the development workspace.

## Verified baseline

Use `scripts/build_native.py` and `scripts/play.py --max-actions 12` to verify a
fresh build and a short run. See the README for the verified platform.
The historical 32-seed paired A10 evaluation recorded 11 Act 1 clears with
round 66 and 7 with the earlier frozen weights. Both arms recorded zero Act 2
and ordinary clears. Both used the same generation-15 run policy and search
budget. Release parity matched 524 actions and both terminal results across
two archived runs on floors 6 and 42.

## Known gaps

The packaged runner targets A10 Ironclad. The strategy still loses difficult
runs. A third-act clear rate is not established by the release smoke check.
Round 66 won 38 of 52 fixed battles against 41 with the earlier weights.
The late Act 2 route trial identified a clear-count difference at only one
of 19 forks, leaving the full-map candidate without enough fit examples.

## Next task

Use the standalone runner to reproduce any reported error before modifying
model artifacts or the simulator pin.

## Log

- 2026-11-25. Prepared the initial source and inference release.
- 2026-11-25. Added the model's observed performance and policy entry points
  to the README.
- Added a [Simplified Chinese README](../README.zh-CN.md) with the same setup
  commands, evaluation results and policy entry points as the English version.
- Documented the research objective, public-conditioned combat search,
  training approach and experimental limitations in both READMEs.
- Presented public evaluation results by release behavior. Internal model
  identifiers and their comparison results are recorded in this handoff.
- Cited AlphaZero in both READMEs as the inspiration for combining learned
  action priors and leaf values with Monte Carlo tree search.
- Removed release-time weight checksums and the Python combat receipt's
  model ID comparison. Model format and tensor shape checks still apply.
  Native binding IDs and public-state fingerprints serve their runtime
  roles. The packaged files and reserialized copies both completed the
  12-action smoke with the same actions and public-state fingerprints.
- Edited both public READMEs for clarity and aligned their descriptions of
  the combat search, training objectives, evaluation results and setup.
- Added matching Mermaid diagrams of the simulator decision loop to the
  English and Chinese READMEs.
