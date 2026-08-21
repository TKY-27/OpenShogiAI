# Phase 10 Arena start pool

Frozen repair: 2026-08-21

The Phase 10 Arena start pool uses only the Phase 10A-approved
`aobazero-no-noise-exact100` population. It does not admit a pending/denied artifact, the legacy
Phase 3 final test, the Taya public-test family, or the dlshogi public evaluation test.

## Statistical design

`general_opening` is a broad control population (`positionIndex < 24`), not the residual left after
style classification. Ibisya and opponent-Furibisha are descriptive tags from the existing
sequence/position-structure classifier. The hard middle/endgame group preserves the legacy
exclusive residual: `positionIndex >= 24` with an unclassified or Furibisha continuation.
General-opening/style tags can overlap; each canonical position is nevertheless stored once.

The deterministic allocator uses seed `20260821`, handles the scarce general-opening population
first, caps a source game at five reserved positions per assigned group, and assigns 200 globally
canonical-distinct positions to each of the four Arena reporting groups. The 800-position reserve
supports 1,600 paired-color games without start reuse. The frozen pilot, entry, and objective gates
still consume only 5, 20, and 50 starts per group; no game count or statistical threshold changed.

Protected history priority is resolved before canonical deduplication. Any history whose effective
split is the final holdout is removed. Every canonical identity observed anywhere in that holdout
is then removed. Validation has priority over train for remaining cross-split duplicates.

## Frozen artifacts

- `artifacts/phase10/start-pool-manifest.json`: 800 central position/provenance rows, group
  eligibility, assigned group, source split/game/raw identity, SFEN, side to move, initial SFEN,
  and the complete move history to the start.
- `artifacts/phase10/start-pool-overlap-report.json`: exact group intersections, duplicate and
  transposition diagnosis, split overlap, and external-holdout unknowns.
- `artifacts/phase10/start-pool-legality-report.json`: all 800 reserved SFENs checked by the Rust
  engine with `perft --depth 0`, bound to the start-manifest and engine hashes.

Current protected candidate counts are 375 general opening, 3,852 Ibisya, 2,097
opponent-Furibisha, and 3,886 hard middle/endgame. Every group reserves 200 positions. Exact
legacy-final-test canonical and protected-history overlap is zero. Exact Taya/dlshogi public-test
overlap remains unmeasured because those artifacts are not authorized or acquired.

## Reproduction and validation

Generate only from the already-present approved local Phase 3 artifact:

```sh
PYTHONPATH=training .venv/bin/python -m open_shogi_training.phase10_execution build-start-pool \
  --positions local/campaign-inputs/data-processed/phase3/aobazero-no-noise-pd-sample100/positions-00000.jsonl.gz \
  --dataset-manifest local/campaign-inputs/data-processed/phase3/aobazero-no-noise-pd-sample100/manifest.json \
  --audit-manifest artifacts/phase10a/audit-manifest.json \
  --output artifacts/phase10/start-pool-manifest.json \
  --overlap-output artifacts/phase10/start-pool-overlap-report.json
```

The publisher is create-only. Regeneration should target a new empty directory, compare hashes,
and replace the freeze only under explicit revision authority.

Run structural and frozen-control validation without training or Arena execution:

```sh
PYTHONPATH=training .venv/bin/python -m open_shogi_training.phase10_execution \
  verify-start-pool --manifest artifacts/phase10/start-pool-manifest.json
make phase10-verify-frozen
```
