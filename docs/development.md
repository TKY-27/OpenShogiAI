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


## Approved positional-evaluator run

The current approved run is `configs/evaluator-main.json`. Its one operational entry point is:

```sh
PYTHONPATH=training uv run --frozen python -m open_shogi_training.evaluator_run seal configs/evaluator-main.json
PYTHONPATH=training uv run --frozen python -m open_shogi_training.evaluator_run start local/runs/evaluator-20260910/main-r2
PYTHONPATH=training uv run --frozen python -m open_shogi_training.evaluator_run status local/runs/evaluator-20260910/main-r2
PYTHONPATH=training uv run --frozen python -m open_shogi_training.evaluator_run stop local/runs/evaluator-20260910/main-r2
```

`seal` runs once after the code commit and local preflight; `start` also resumes. Do not reseal
or edit a running contract. `docs/status.md` records whether it has actually been sealed/launched.
The run binds code, native/Wasm files, teacher configuration, frozen model, split metadata and seed.
The supervisor advances generation, preparation, training, export audit and the predefined arena.
The current r2 recovery preserves all 327 verified completed trajectories from the stopped first
run through a hash-bound continuation manifest. The source raw labels/receipts are immutable
hard links; they are not regenerated or counted as newly acquired twice. The original campaign
start and swap baseline carry into the prepared state, so the 24-hour limit is not reset.
It stops at `awaiting_astra_browser`; this does not promote or publish a model.

Luna (`gpt-5.6-luna`, `max`) executes this contract, reads state on meaningful changes, and owns
no design choices, thresholds, Git operations or model promotion. The script samples process and
output progress every 15 seconds; finite limits are 24 hours, one defined interruption retry,
80 GiB free space, 16 GiB owned-process RSS and 4 GiB additional swap. Persistent numerical,
identity, leakage or logic failures return to Astra. STOP retains completed trajectories,
the best inference export and two latest coherent resume states. `state.json`, `monitor.jsonl`,
per-stage logs and `fit/{progress,resume,training}.json` are the run's operational evidence.
Each stage records its process identity before launching children. Its workers and teacher stay
in one dedicated process group, reclaimed before the stage leader is reaped. If both supervisor
and stage disappear with a residual group, restart fails closed with `needs_astra`; an uncertain
process identity is never permission to kill another process or start a second writer. The
teacher adapter's usual isolated-process default is unchanged outside this run.

After the r2 host-swap stop, a single r3 resource recovery may reduce teacher workers to one.
Admission requires at least 60 seconds of normal OS memory pressure and no new swapout/pageout
pages, plus a fresh pre-start check. Its 4 GiB swap-growth bound is measured from the new
resource interval, not from the original campaign baseline. Both baselines and their difference
are retained; the original 24-hour deadline and learning/adoption criteria remain unchanged.
A second swap-limit stop must not automatically rebaseline or restart. Low owned RSS alone
does not prove the run did not contribute to earlier memory pressure.

The main candidate retains the independent W256 OSAVAL03 architecture (8.68 MB) and retrains
all layers on the scalar evaluation path, starting from a copy of the frozen baseline.
No piece values, king-safety rules or external runtime evaluator enter nonterminal scoring.
CPU/four-thread Q20 accumulation was measured faster than one thread; MPS is not used because
the verified exact accumulation path requires f64. The old controller predicts decision change,
not expected improvement in move quality, and is not certified for the changed evaluator.
It remains OFF in the main comparison; OFF is not presented as a successful new controller.

The data are new owned trajectories generated with the existing offline Apery 2.0.0 teacher,
not repeated exposure to the old 180-position controller cohort. The official distribution is
[Apery Rust v2.0.0](https://github.com/HiraokaTakuya/apery_rust/releases/tag/v2.0.0).
Its engine and separately licensed evaluation assets remain local; the pinned local install
manifest and teacher configuration preserve their source, version and file hashes.
The engine is GPL-3.0; the bundled evaluation README/LICENSE identifies MIT terms. No teacher
assets or weights are redistributed. Candidate nodchip data were checked but not acquired:
the range-downloadable PSV release lacks original game identities and would require a newly
validated decoder/label scale, whereas the existing teacher permits known source trajectories.

Before feature generation each trajectory gets a deterministic train/validation/development-test
assignment. All repeated positions and color/board symmetries across partitions are removed,
including known diagnostic/preflight positions and frozen split-guard hashes. No final holdout
contents are opened. New trajectories share a teacher and starting state; they are not independent
human games. Every move is checked by our native rules engine. Roots and teacher candidate children
use typed scores with explicit side-to-move conversion; mate and terminal children are masked.
Weak baseline-preferred deviations receive a separate teacher analysis. A teacher declaration
response is independently checked under its CSA entering-king rules; it is retained as a typed
ending, never converted into an invented scalar label. The ordinary strict MultiPV adapter
contract and the engine's separate JSA rules remain unchanged. Raw labels and lineage remain
available. Scores are not mixed with another teacher or converted from unknown WDL scales.

The fixed main plan uses 2,048 trajectories, at most 192 plies, two teacher workers at 25,000 nodes,
and samples every second root plus candidate children. It targets roughly 250,000–700,000 distinct
positions; actual unique roots/children, source trajectories and extended training exposures are
reported separately. CPU training uses batch 256, at most 8,192 updates/eight exposures per row,
learning rate 0.0002 with 100-step warmup and cosine decay to 0.00001. Validation every 512 updates
uses patience five and relative improvement 0.002. Small-preflight weakness does not block this
approved expansion; invalid code/data contracts do. Sampler order/offset, RNG, optimizer, exposure
counts and checkpoint-bound best model are restored together.

Before results are opened, comparison is fixed to 32 scored games from 16 development-start pairs:
24 at three minutes and eight at ten minutes, with colors reversed and the same updated engine,
clock and controller OFF. Eight normal-start demonstrations are separate. All games and failures
are retained. Adoption requires valid completion, score above 0.5 in both clock groups, and an
overall paired-start bootstrap one-sided 95% lower bound above 0.5. Two wins, training loss alone,
or a max-plies adjudication cannot certify improvement. The development test is separate from
training/validation selection, and the historical final holdout stays sealed. Human shodan strength
remains untested. Astra reviews identity, export/native/Wasm parity and arena outcomes, then adds
an explicit local candidate descriptor and rechecks real browser play. Public default selection,
main merge, deployment and weight publication require separate authorization.

Clock acceptance uses all 3/10-minute × standard/high-quality × both-side combinations. Initial
and quiet opening replies should ordinarily take a few seconds (initial black under 10 seconds
on this M5), never the old roughly 60-second allocation. The measured adaptive hard limit must
hold in every case including scarce time and stop. Preparation and real search/user wait are
separate; actual delivery and move application remain charged to the side's clock. Stable replies
may stop before their soft target. Cancellation uses a shared atomic flag observed inside Wasm,
plus an independent watchdog and request generations; a stop message alone is not the proof.
