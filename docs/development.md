# Development

Run commands from the repository root. Setup is in [README](../README.md).

| Command | Purpose |
| --- | --- |
| `make check` | Locked environment, boundaries, licenses, provenance, links, format/lint, tests and builds |
| `make test` | Rust and Python regression suites, using fixtures |
| `make build` | Native workspace and byte-for-byte development Wasm regeneration |
| `make pure-build` | Native and Wasm with `pure-only`, under `target/pure/` |
| `make frozen-smoke` | Explicit local baseline hash/load/native/Wasm smoke; no Arena or training |
| `make format` / `make lint` | Rustfmt, Clippy and Ruff |
| `make phase3-validate-registry` | Validate source-rights catalog without acquisition |
| `make model-validate` | Validate legacy model/feature/training schemas |

Python modules under `training/open_shogi_training/` retain model, data, teacher and evidence
primitives with their regression tests. Historical `phase*` control files are compatibility
fixtures, not current execution plans. The completed campaign freeze entry points reject
execution; old documents and all-receipt preservation checks are not development gates.
Old campaign snapshots can be studied through Git history when specifically needed.
Optional model comparison commands require explicit model/data inputs; champion-gate commands
require an explicit config. Historical candidate paths are unavailable, not fallback defaults.

## Local storage

Use one generated-asset root, `local/`: `frozen/` holds the comparison manifest, minimal
reproduction, provenance and sealed evaluation originals; `runs/` holds future experimental
outputs; `teacher/` holds the optional external teacher; `tooling/` holds the pinned Wasm tool.
`maintenance/` contains only compact local cleanup/verification records. Cargo builds stay
in `target/`; `.venv/` and `.uv-cache/` are reusable Python infrastructure.

Never add local assets or trained weights to Git. Before removing a generated subtree,
confirm references, idle processes and file handles, containment, symlinks and mount boundaries.
Preserve unique evaluation inputs and their source-game/descendant partition identities.
Do not create archival copies of all obsolete build or experiment outputs. Record deletion
paths, reasons and approximate sizes locally. Git-history size is separate from worktree savings.

## Compatibility checks

`make check` checks generated development bindings against the locked toolchain. Pure-only
bindings are separate build outputs. The shared runtime tests cover model rejection, legal
moves, history, accumulators and native/Wasm contracts. The local baseline smoke tests run
only a small fixed search and explicit failure-on-bad-model checks; they do not remeasure strength.

No CI workflow is configured in this tree. Local passes do not claim remote CI, device/browser UI,
publication, training recovery or large-evaluation availability.

## Local computation-control prototype

The trained controller and frozen W256 stay local. Their expected identities and aggregate
training counts are in `configs/core-prototype.json`; detailed machine-readable evidence is in
`local/core-prototype/`. No download of a large archive or teacher is needed for this prototype.

Build the pure native/Wasm runtime once after a source change:

```sh
make pure-build
```

With the existing local artifacts, open the adjacent UI checkout:

```sh
cd ../OpenShogiUI
npm ci --ignore-scripts
npm run dev -- --host 127.0.0.1 --port 5176 --strictPort
```

Visit `http://127.0.0.1:5176/#/core-prototype`. Choose the explicit frozen/candidate evaluator, your side, standard/high quality,
3/10 minutes sudden death and computation control ON/OFF, then start. No opening book is used.
The old controller is off by default; a new evaluator without matching controller cannot enable it. Stop preserves the
confirmed position and clocks; Resume reloads the same assets. Rematch permits a new side/control
choice. The controller, W256 and pure Wasm are served only by loopback development middleware.
The existing public default evaluator, pinned standard UI bindings and production build are
unchanged. Accordingly, the legacy `npm run integration:ai` checks a different standard artifact
snapshot and is not the prototype integration check. See the UI README for that contract.

The native collection driver uses qdepth 4, TT 2 MiB and one thread, matching the prototype's
Wasm eco profile. To inspect or reproduce the bounded experiment, use new output directories;
the scripts refuse replacement. These commands are documentation, not startup requirements:

```sh
cargo build --release --locked -p open-shogi-core --example core_probe --no-default-features --features pure-only
python3.12 scripts/collect_core_prototype.py --probe target/release/examples/core_probe \
  --leaf local/frozen/baseline/model.osaval03 \
  --leaf-sha256 859e922b3f503ddeecf0afeb9a05fccac080a9faca3b19fce9d8253c9039c480 \
  --output local/core-prototype/reproduction-cohort
PYTHONPATH=training uv run --frozen python -m open_shogi_training.core_prototype \
  local/core-prototype/reproduction-cohort/records.jsonl local/core-prototype/reproduction-fit \
  --leaf-sha256 859e922b3f503ddeecf0afeb9a05fccac080a9faca3b19fce9d8253c9039c480 \
  --split-guard local/frozen/metadata/split-guard.json
node scripts/check_core_prototype.mjs target/pure/bindings/open_shogi_wasm.js \
  local/frozen/baseline/model.osaval03 859e922b3f503ddeecf0afeb9a05fccac080a9faca3b19fce9d8253c9039c480 \
  local/core-prototype/controller.json 66110bae4ef5fbedd6a3c5537813f0fae5b9276b576a745b2364b4a093141b63 \
  local/core-prototype/wasm-contract.json
```

Collection is capped at 24 trajectories × 8 samples, depth 4 / 20,000 nodes / 1 second;
training is one full-batch logistic fit, 300 updates, one best checkpoint and one resume state.
The saved original label probe is `local/core-prototype/label-probe`. Time-limited collection
and compiler changes can change later traces, so reproduction creates a new identity rather
than claiming byte equality with the original data. Do not automatically rerun or enlarge it.


## Approved evaluator execution

`local/handoff/` holds the compact run handoff (not tracked);
[evaluator-main.json](../configs/evaluator-main.json)
is the one current machine-readable plan. The sealed executable copy is
`local/runs/defense-20260912/recovery-r3/run.json`, with `seal.json` and `state.json` alongside.
Do not infer a run name from old r3 directories. From this Git root:

```sh
PYTHONPATH=training uv run --frozen python -m open_shogi_training.evaluator_run start local/runs/defense-20260912/recovery-r3
PYTHONPATH=training uv run --frozen python -m open_shogi_training.evaluator_run status local/runs/defense-20260912/recovery-r3
PYTHONPATH=training uv run --frozen python -m open_shogi_training.evaluator_run stop local/runs/defense-20260912/recovery-r3
```

`start` resumes the same stopped run. It cannot restart `needs_astra` or an awaiting-review
terminal state. `seal configs/evaluator-main.json` is an Astra preparation operation after the
source/config commit and short preflight, performed once. It does not launch learning.
The stage sequence is generation, verified preparation, bounded training, export/move-quality
audit, then fixed r3 matches. The contract contains exact resource/data/sampling limits,
commands, output references, retention and recovery rules. It finishes at `awaiting_astra_review`.
`stage RUN audit` and `stage RUN arena` require successful preceding stages and never bypass
identity, stop or review states. Best export publication is part of training, not a fresh fit.

Astra owns diagnosis, settings, code, review and browser integration. The user starts the separate
Luna Max/max execution session. No Luna child agent, constant LLM polling, automatic resealing,
branch changes while running, paid compute, promotion, main merge or public weight release.
A requested stop preserves source receipts, best export and two coherent resume checkpoints;
optimizer, RNG, sampler order/offset and exposures restore together. Only the contract's bounded
recovery is automatic.
The phase10r campaign and bounded-training modules publish a `<name>.pt.sha256` digest receipt
beside every checkpoint they write. Their resume and evaluation loads verify the file bytes against
that receipt before deserialization and load with `weights_only=True` under a pinned numpy
allowlist; a checkpoint without a matching receipt is refused rather than trusted. Sibling
training tools outside those two modules keep their existing loaders. Code file hashes permit status/results-only documentation commits.

All play is book-free. Training may use offline scenarios and the existing hash-pinned
[Apery 2.0.0](https://github.com/HiraokaTakuya/apery_rust/releases/tag/v2.0.0) teacher.
The teacher binary (GPL-3.0) and its separately licensed evaluation assets (MIT) remain local,
with the install manifest, original notices and exact hashes. No new public archive is required.
The new run accepts completed-depth labels only, clears teacher TT before every independent
analysis. A root depth miss becomes a durable local task with shared finite retry accounting.
The successor retains the parent shards and incomplete history; coverage gates, not universal
root success, control training. Consult the current contract for queue and replenishment limits.

Real 3/10-minute paired games and an offline fixed-node move screen are separate measurements.
Only one validation-selected candidate receives the fixed comparisons; development test never
selects checkpoints. Original r3 evaluation data remain excluded from training. Known diagnosis
and preflight positions are development-only and excluded from new partitions; final holdout
contents stay unopened. Actual browser identity/clock/stop/end checks belong to Astra after
learning and do not establish human shodan strength.
