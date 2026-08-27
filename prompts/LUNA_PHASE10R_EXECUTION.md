# Luna Max execution prompt — Phase 10R

You are Luna Max executing a frozen, long-running OpenShogiAI experiment.

Target repository: the `OpenShogiAI` repository root containing this prompt

Required branch: `codex/phase10r-curriculum`

Subagents: forbidden. Do not spawn Luna, Terra, Sol, or any other agent.

## Authority boundary

Read `PHASE_10R_FROZEN_PLAN.md`, all four Phase 10R model documents, every
`configs/phase10r/*.yaml`, and the frozen hash manifest before acting. Treat them as immutable.
You may implement missing training/runtime commands and run the frozen campaign. You may not alter
architecture, feature definitions/dimensions/hashes, source approval, source proportions, target
semantics, splits, holdouts, scale rungs, gates, thresholds, Arena contract, self-play entry,
resource policy, promotion policy, or frozen hashes. If implementation cannot satisfy a frozen
control, stop and report the exact conflict. Do not weaken a test.

Do not publish, push, deploy, promote, inspect a reserved/public holdout, or activate a pending or
denied source. Generated weights are local experimental artifacts and may not be published.

## Initial exact commands

```sh
repository_root=$(git rev-parse --show-toplevel)
test "$(basename "$repository_root")" = OpenShogiAI
cd "$repository_root"
git switch codex/phase10r-curriculum
git status --short --branch
make phase10r-validate
make phase10r-sanity
make phase10r-micro-overfit
make phase10r-memory
make check
PYTHONPATH=training uv run --frozen python -m open_shogi_training.data phase10r-validate --registry configs/phase10r/source-registry.yaml
PYTHONPATH=training uv run --frozen python -m open_shogi_training.data phase10r-dry-run --registry configs/phase10r/source-registry.yaml
```

Stop immediately if the branch is wrong, the worktree contains unexplained changes, a hash differs,
or any command fails.

## Required execution interface

Implement the module `open_shogi_training.phase10r_run` if it is absent, without changing frozen
controls. It must expose these exact commands and write immutable JSON receipts below
`local/phase10r-runs/`:

```sh
PYTHONPATH=training uv run --frozen python -m open_shogi_training.phase10r_run preflight --root .
PYTHONPATH=training uv run --frozen python -m open_shogi_training.phase10r_run prepare --root . --scale 1m
PYTHONPATH=training uv run --frozen python -m open_shogi_training.phase10r_run train --root . --scale 1m --variant sparse-pair-policy-wdl --resume
PYTHONPATH=training uv run --frozen python -m open_shogi_training.phase10r_run train --root . --scale 1m --variant factorized-pair-triple-policy-score --resume
PYTHONPATH=training uv run --frozen python -m open_shogi_training.phase10r_run evaluate --root . --scale 1m --all-source-held-out --cross-runtime --incremental-parity
PYTHONPATH=training uv run --frozen python -m open_shogi_training.phase10r_run select-hard --root . --scale 1m
PYTHONPATH=training uv run --frozen python -m open_shogi_training.phase10r_run label-hard --root . --scale 1m --resume
PYTHONPATH=training uv run --frozen python -m open_shogi_training.phase10r_run report --root . --scale 1m
```

`preflight` must re-run hashes, sanity, micro-overfit, memory, disk, RSS/thermal checks, and verify
that the final runtime mode is pure. `prepare` streams only approved registry artifacts, assigns
protected splits before extraction, deduplicates, records canonical target and realized shares, and never reads reserved
content. A missing approved archive may be acquired only with the existing resumable rights-gated
data command and the 150 GiB free-space floor. WCSC LZH archives require an audited bounded
extractor; skip and report any archive whose format path is not implemented rather than improvising.

The runtime module must add and pass feature-key/move-index parity across Python/Rust/Wasm,
incremental/full recompute and unmake parity, legal-root policy uniqueness, qsearch leaf semantics,
and history-context tests before `train` accepts more than the micro dataset.

## Progressive scale commands

After 1m passes every expansion gate, run the same exact command sequence with only `--scale`
changed, in this order:

```sh
# 10m
PYTHONPATH=training uv run --frozen python -m open_shogi_training.phase10r_run prepare --root . --scale 10m
PYTHONPATH=training uv run --frozen python -m open_shogi_training.phase10r_run train --root . --scale 10m --variant sparse-pair-policy-wdl --resume
PYTHONPATH=training uv run --frozen python -m open_shogi_training.phase10r_run train --root . --scale 10m --variant factorized-pair-triple-policy-score --resume
PYTHONPATH=training uv run --frozen python -m open_shogi_training.phase10r_run evaluate --root . --scale 10m --all-source-held-out --cross-runtime --incremental-parity
PYTHONPATH=training uv run --frozen python -m open_shogi_training.phase10r_run select-hard --root . --scale 10m
PYTHONPATH=training uv run --frozen python -m open_shogi_training.phase10r_run label-hard --root . --scale 10m --resume
PYTHONPATH=training uv run --frozen python -m open_shogi_training.phase10r_run report --root . --scale 10m

# Repeat the identical seven commands for 50m, 100m, 500m, then 1b, only after the prior report says EXPAND.
```

The scale values are exactly `1m`, `10m`, `50m`, `100m`, `500m`, `1b`; streamed examples are
exactly 1M, 10M, 50M, 100M, 500M, and 1B. Teacher-label caps and nodes/MultiPV come only from
`active-learning.yaml`. Do not expand because time passed or offline loss alone improved. On a
plateau, restore the best checkpoint and stop scale expansion.

## Arena commands

Only a frozen rung winner that passes offline, throughput, cross-runtime, incremental, calibration,
source-held-out, and tactical gates may run Arena. Use:

```sh
PYTHONPATH=training uv run --frozen python -m open_shogi_training.phase10r_run arena --root . --gate short_screen --resume
PYTHONPATH=training uv run --frozen python -m open_shogi_training.phase10r_run arena --root . --gate arena_400 --resume
PYTHONPATH=training uv run --frozen python -m open_shogi_training.phase10r_run arena --root . --gate arena_800 --resume
PYTHONPATH=training uv run --frozen python -m open_shogi_training.phase10r_run arena --root . --gate practical_20s --resume
PYTHONPATH=training uv run --frozen python -m open_shogi_training.phase10r_run arena --root . --gate final_1600 --resume
```

Every command validates the fixed start-manifest hash, reversed colors, one-pair maximum reuse,
equal wall clock, opening off, candidate `PureValue`, incumbent `handcrafted-experimental`, and
zero handcrafted leaf contribution. Policy ordering, when enabled, is included in candidate time.
Capped/incomplete games are not draws. Do not retry or conceal an unexplained crash. Do not inspect
the final holdout before `final_1600` completes.

## Bootstrapped cross-play and self-play commands

After the short screen, bootstrapped generation is allowed before 48%:

```sh
PYTHONPATH=training uv run --frozen python -m open_shogi_training.phase10r_run crossplay --root . --opponents handcrafted-experimental,older-pure --resume
PYTHONPATH=training uv run --frozen python -m open_shogi_training.phase10r_run select-decisive-errors --root .
PYTHONPATH=training uv run --frozen python -m open_shogi_training.phase10r_run label-decisive-errors --root . --resume
```

The handcrafted score is selection evidence only and never a target. Pure self-play is forbidden
until `arena_400` records candidate score at least 48% and all zero-failure requirements:

```sh
PYTHONPATH=training uv run --frozen python -m open_shogi_training.phase10r_run selfplay --root . --games 1000 --resume
PYTHONPATH=training uv run --frozen python -m open_shogi_training.phase10r_run selfplay --root . --games 5000 --resume
PYTHONPATH=training uv run --frozen python -m open_shogi_training.phase10r_run selfplay --root . --games 20000 --resume
PYTHONPATH=training uv run --frozen python -m open_shogi_training.phase10r_run selfplay --root . --games 100000 --resume
```

Run only one generation size at a time and require the frozen expansion gate between sizes. No
generation promotes automatically.

## Resource behavior

Never run more than two heavy workers: one trainer plus one teacher or Arena worker, and never
trainer plus teacher concurrently. Aggregate RSS target is 16 GiB. At warning/critical memory or
serious thermal pressure, checkpoint and stop as specified. Keep at least 150 GiB free. Do not
materialize all examples. There is no wall-clock deadline.

Launch long work as one resumable process and wait on process completion or checkpoint events.
Do not create a polling-heavy agent loop. Periodic job receipts are every 30 minutes; do not send
chat updates merely because a receipt is unchanged.

## Stop conditions

Stop and preserve artifacts on any hash/sanity/parity/legality failure, unauthorized source,
holdout contact, resource breach, checkpoint corruption, NaN/Inf, unexplained crash, illegal move,
tactical regression, source-held-out regression above 0.01, Arena regression above 0.02, two-rung
plateau, or failed gate. Do not change a threshold or run a replacement architecture.

## Final report

Write `local/phase10r-runs/PHASE10R_EXECUTION_REPORT.md` and a machine-readable JSON report with:

- commit and dirty state; every config/model/data/teacher/start/checkpoint hash;
- acquired/streamed/unique/deduplicated/quarantined examples by source/split/rung;
- canonical source targets, realized shares, repetition factors, and all attribution obligations;
- sanity, micro-overfit, feature/move/incremental/cross-runtime results;
- every head metric, source-held-out metric, calibration, tactical result, latency, RSS, disk, and thermal receipt;
- active-learning selections/budgets and checkpoint lineage;
- every Arena W-L-D, finished/capped/failure count, score, both 95% intervals, colors/groups, and wall time;
- cross-play/self-play generations, rollback/plateau decisions, and zero promotion/push/publication;
- whether the immutable 55%/1,600-game objective passed; and
- the next exact command, or the exact stop reason.

End by running `make phase10r-validate` and `make check`. Do not push.
